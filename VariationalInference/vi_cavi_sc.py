"""Hierarchical (single-cell) DRGP model — Chapter 7 of the paper.

Sibling to flat `CAVI` (vi_cavi.py). Shares:
  - all gene-side updates (ϕ, β̃ slab, ρ, π, η, s) via inheritance
  - the PG-CAVI supervised-regression kernel via vi_pg_kernel.py

Differs from flat in:
  - Two new variational factor groups:
      Θ (G × K)  patient-level program activity (= supervision design)
      ζ (T × K)  cell-type × program affinity
  - Cell-level θ has structured Gamma prior:
      θ_{iℓ} ~ Gamma(α_θ, ξ_i · Θ_{g_i,ℓ} · ζ_{t_i,ℓ})
    so its update (C.3, C.4) drops the supervised correction R_{iℓ} and
    uses a structured rate.
  - Supervision moves from y_i (cell) to y_g (patient). The PG-CAVI kernel
    is called with E_design = E[Θ] (G × K), y_patient (G × κ), and
    X_aux_patient (G × p_aux). The R^Θ correction returned by the kernel
    feeds Θ's b-rate (eqs C.5-C.7), not θ's.
  - PG augmentation moves to (G × κ). Sample weights are per patient.

Setting Θ ≡ 1, ζ ≡ 1, and re-attaching supervision to θ_i recovers flat CAVI
exactly on the expression side — the hierarchical model is a strict
generalization (see paper §7.1).

Identifiability: only the products ξ_i Θ_{g_i,ℓ} ζ_{t_i,ℓ} and θ β^T are
identified. `_rescale_factors` is currently disabled in CAVI (breaks ELBO
monotonicity); when re-enabled, any per-factor rescaling of Θ must mirror
into υ to preserve A_{gk} (paper §7.6).
"""
from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import time
import numpy as np
import scipy.sparse as sp

try:
    from .vi_cavi import (
        CAVI, _is_oom_error,
        _elbo_poisson_recon, _elbo_theta_entropy_chunked,
        _elbo_beta_block, _elbo_xi_block, _elbo_eta_block,
        _elbo_v_block, _elbo_gamma_aux_block,
    )
    from .jax_backend import (
        xp, USE_JAX, HAS_GPU, to_device, to_numpy,
        digamma, gammaln, scatter_add_to,
    )
    from .vi_pg_kernel import (
        pg_tilt, pg_R_correction, pg_update_v, pg_update_gamma, pg_Lsup,
    )
except ImportError:
    from vi_cavi import (  # type: ignore[no-redef]
        CAVI, _is_oom_error,
        _elbo_poisson_recon, _elbo_theta_entropy_chunked,
        _elbo_beta_block, _elbo_xi_block, _elbo_eta_block,
        _elbo_v_block, _elbo_gamma_aux_block,
    )
    from jax_backend import (  # type: ignore[no-redef]
        xp, USE_JAX, HAS_GPU, to_device, to_numpy,
        digamma, gammaln, scatter_add_to,
    )
    from vi_pg_kernel import (  # type: ignore[no-redef]
        pg_tilt, pg_R_correction, pg_update_v, pg_update_gamma, pg_Lsup,
    )


class CAVIHierarchical(CAVI):
    """Chapter 7 hierarchical DRGP model. See module docstring.

    Extra hyperparameters (Gamma priors on the new factor groups):
        alpha_Theta, lambda_Theta — shape/rate for Θ_{gℓ} prior.
        alpha_zeta,  lambda_zeta  — shape/rate for ζ_{tℓ} prior.

    All other hyperparameters (a, ap, c, cp, b_v, sigma_gamma,
    regression_weight, mode, pathway_mask, etc.) inherited from CAVI.

    Call convention (set in `fit`):
        patient_ids   : (n,) int array in [0, G)
        cell_type_ids : (n,) int array in [0, T)
        y_patient     : (G,) or (G, κ) patient-level labels
        X_aux_patient : (G, p_aux) patient-level aux (intercept prepended internally)
    """

    def __init__(
        self,
        n_factors: int,
        alpha_Theta: float = 1.0,
        lambda_Theta: float = 1.0,
        alpha_zeta: float = 1.0,
        lambda_zeta: float = 1.0,
        **kwargs,
    ):
        super().__init__(n_factors=n_factors, **kwargs)
        self.alpha_Theta = alpha_Theta
        self.lambda_Theta = lambda_Theta
        self.alpha_zeta = alpha_zeta
        self.lambda_zeta = lambda_zeta
        # Set in fit():
        self.G: Optional[int] = None
        self.T: Optional[int] = None
        self.patient_id = None       # (n,) int, on device
        self.cell_type_id = None     # (n,) int, on device
        self.y_patient = None        # (G, κ) float, on device
        self.X_aux_patient = None    # (G, p_aux_with_intercept) float, on device

    # ------------------------------------------------------------------
    # Properties for Θ, ζ
    # ------------------------------------------------------------------
    @property
    def E_Theta(self):
        return self.a_Theta / self.b_Theta

    @property
    def E_zeta(self):
        return self.a_zeta / self.b_zeta

    @property
    def E_log_Theta(self):
        return digamma(self.a_Theta) - xp.log(self.b_Theta)

    @property
    def E_log_zeta(self):
        return digamma(self.a_zeta) - xp.log(self.b_zeta)

    # ------------------------------------------------------------------
    # Hier-specific initialization (called after parent _initialize)
    # ------------------------------------------------------------------
    def _initialize_hier(
        self,
        patient_id_per_cell: np.ndarray,
        cell_type_id_per_cell: np.ndarray,
        y_patient: np.ndarray,
        X_aux_patient: np.ndarray,
    ):
        """Wire patient/cell-type metadata, init Θ and ζ, retarget PG buffers to (G, κ)."""
        self.G = int(patient_id_per_cell.max()) + 1
        self.T = int(cell_type_id_per_cell.max()) + 1
        self.patient_id = to_device(patient_id_per_cell.astype(np.int32))
        self.cell_type_id = to_device(cell_type_id_per_cell.astype(np.int32))

        # Patient-level y (G, κ).
        if y_patient.ndim == 1:
            y_patient = y_patient[:, None]
        self.y_patient = to_device(y_patient.astype(np.float32))
        self.kappa = self.y_patient.shape[1]

        # Patient-level X_aux with intercept already prepended (caller's job).
        self.X_aux_patient = to_device(np.asarray(X_aux_patient, dtype=np.float32))
        self.p_aux = self.X_aux_patient.shape[1]

        # Re-init regression head sized to (κ, K) and (κ, p_aux) — parent set these
        # using cell-level kappa/p_aux; overwrite now that the patient sizes are known.
        self.mu_v = xp.zeros((self.kappa, self.K), dtype=np.float32)
        self.sigma_v_diag = xp.full((self.kappa, self.K), self.b_v ** 2, dtype=np.float32)
        self.mu_gamma = xp.zeros((self.kappa, self.p_aux), dtype=np.float32)
        self.Sigma_gamma = xp.broadcast_to(
            (self.sigma_gamma ** 2) * xp.eye(self.p_aux),
            (self.kappa, self.p_aux, self.p_aux),
        )
        # JAX broadcast_to returns read-only views; we need writable for in-place update,
        # so concretize to a real (kappa, p_aux, p_aux) tensor.
        self.Sigma_gamma = xp.array(self.Sigma_gamma)

        # PG aug at patient level
        self.c_pg = xp.zeros((self.G, self.kappa), dtype=np.float32)
        self.wbar = xp.full((self.G, self.kappa), 0.25, dtype=np.float32)

        # Sample weights — class-rebalancing per outcome at patient level.
        if self.use_class_weights and self.G > 0:
            sw = np.ones((self.G, self.kappa), dtype=np.float32)
            for k in range(self.kappa):
                yk = to_numpy(self.y_patient[:, k]).astype(int)
                pos = max(int(yk.sum()), 1)
                neg = max(int(self.G - yk.sum()), 1)
                sw[yk == 1, k] = 0.5 * self.G / pos
                sw[yk == 0, k] = 0.5 * self.G / neg
            self._sample_weights = to_device(sw)
        else:
            self._sample_weights = xp.ones((self.G, self.kappa), dtype=np.float32)

        # Init Θ (G, K) and ζ (T, K) at prior mean ± 50% (HPF pattern).
        rs = np.random.RandomState(self.seed_used_)
        self.a_Theta = to_device(
            rs.uniform(0.5 * self.alpha_Theta, 1.5 * self.alpha_Theta,
                       size=(self.G, self.K)).astype(np.float32)
        )
        self.b_Theta = to_device(
            np.full((self.G, self.K), self.lambda_Theta, dtype=np.float32)
        )
        self.a_zeta = to_device(
            rs.uniform(0.5 * self.alpha_zeta, 1.5 * self.alpha_zeta,
                       size=(self.T, self.K)).astype(np.float32)
        )
        self.b_zeta = to_device(
            np.full((self.T, self.K), self.lambda_zeta, dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # Update overrides
    # ------------------------------------------------------------------
    def _update_xi(self):
        """C.2 — b_ξ rate now includes Σ_ℓ E[Θ_{g_i,ℓ}] E[ζ_{t_i,ℓ}] E[θ_{iℓ}]."""
        E_theta = self.a_theta / self.b_theta
        E_Theta_pc = self.E_Theta[self.patient_id]    # (n, K)
        E_zeta_pc = self.E_zeta[self.cell_type_id]    # (n, K)
        rate_incr = (E_Theta_pc * E_zeta_pc * E_theta).sum(axis=1)  # (n,)
        self.b_xi = self.bp + rate_incr
        self.b_xi = xp.maximum(self.b_xi, 1e-6)

    def _update_theta(self, z_sum_theta, y=None, X_aux=None, ramp=1.0):
        """C.3, C.4 — drop R_{iℓ} (supervision moved to Θ); structured rate."""
        self.a_theta = self.a + z_sum_theta
        if self._active_beta is not None:
            beta_sum = xp.where(self._active_beta, self.E_beta, 0.0).sum(axis=0)
        else:
            beta_sum = self.E_beta.sum(axis=0)

        E_Theta_pc = self.E_Theta[self.patient_id]
        E_zeta_pc = self.E_zeta[self.cell_type_id]
        rate_struct = self.E_xi[:, None] * E_Theta_pc * E_zeta_pc  # (n, K)
        b_theta_new = rate_struct + beta_sum[None, :]

        # Floors mirroring flat code (load-bearing — CLAUDE.md stability fixes).
        b_theta_new = xp.maximum(b_theta_new, 0.1 * rate_struct)
        b_theta_new = xp.maximum(b_theta_new, self.bp)
        b_theta_new = xp.maximum(b_theta_new, 1e-2)
        b_theta_new = xp.maximum(b_theta_new, self.a_theta / 1e4)

        self.b_theta = b_theta_new
        self._theta_inner_iters = 1
        self._invalidate_theta_cache()

    def _update_zeta(self):
        """C.8 — conjugate (no supervision). Pools all cells of type t across patients."""
        E_theta = self.a_theta / self.b_theta
        E_Theta_pc = self.E_Theta[self.patient_id]
        per_cell_contrib = self.E_xi[:, None] * E_Theta_pc * E_theta  # (n, K)

        b_zeta_sums = xp.zeros((self.T, self.K), dtype=xp.float32)
        b_zeta_sums = scatter_add_to(b_zeta_sums, self.cell_type_id, per_cell_contrib,
                                      sorted_indices=False)

        ones_n = xp.ones(self.n, dtype=xp.float32)
        ct_counts = xp.zeros(self.T, dtype=xp.float32)
        ct_counts = scatter_add_to(ct_counts, self.cell_type_id, ones_n,
                                   sorted_indices=False)

        self.a_zeta = self.alpha_zeta + ct_counts[:, None] * self.a
        self.b_zeta = xp.maximum(self.lambda_zeta + b_zeta_sums, 1e-6)

    def _update_Theta(self, ramp: float = 1.0):
        """C.5, C.6, C.7 — supervised. Pools the patient's cells + cohort prior + PG correction.

        Solves the quadratic
            b² - (b_base + R_lin) b - R_quad · a = 0
        with R_lin, R_quad = effective_rw · supervised contributions (kernel).
        """
        # 1. Per-patient pooled term: Σ_{i∈C_g} E[ξ_i] E[ζ_{t_i,ℓ}] E[θ_{iℓ}]
        E_theta = self.a_theta / self.b_theta
        E_zeta_pc = self.E_zeta[self.cell_type_id]
        per_cell = self.E_xi[:, None] * E_zeta_pc * E_theta   # (n, K)
        pooled = xp.zeros((self.G, self.K), dtype=xp.float32)
        pooled = scatter_add_to(pooled, self.patient_id, per_cell, sorted_indices=False)

        # 2. |C_g| per patient
        ones_n = xp.ones(self.n, dtype=xp.float32)
        cg_counts = xp.zeros(self.G, dtype=xp.float32)
        cg_counts = scatter_add_to(cg_counts, self.patient_id, ones_n,
                                   sorted_indices=False)

        # 3. R^Θ via PG kernel (patient-level design).
        E_Theta = self.E_Theta
        Var_Theta = E_Theta / self.b_Theta
        effective_rw = ramp * self.regression_weight
        R_lin, R_quad, self._row_chunk = pg_R_correction(
            E_design=E_Theta,
            mu_v=self.mu_v,
            sigma_v_diag=self.sigma_v_diag,
            mu_gamma=self.mu_gamma,
            X_aux=self.X_aux_patient,
            wbar=self.wbar,
            y=self.y_patient,
            sample_weights=self._sample_weights,
            effective_rw=effective_rw,
            row_chunk=self._row_chunk,
        )

        # 4. Solve quadratic for b_Θ.
        a_Theta_new = self.alpha_Theta + cg_counts[:, None] * self.a   # (G, K)
        b_base = self.lambda_Theta + pooled
        b_full = b_base + R_lin
        disc = xp.sqrt(xp.square(b_full) + 4.0 * R_quad * a_Theta_new)
        b_Theta_new = (b_full + disc) / 2.0
        # Floors
        b_Theta_new = xp.maximum(b_Theta_new, 0.1 * b_base)
        b_Theta_new = xp.maximum(b_Theta_new, self.lambda_Theta)
        b_Theta_new = xp.maximum(b_Theta_new, 1e-6)
        b_Theta_new = xp.maximum(b_Theta_new, a_Theta_new / 1e4)

        self.a_Theta = a_Theta_new
        self.b_Theta = b_Theta_new

    # PG-related overrides — design is now E[Θ], units are G patients.

    def _update_omega(self, X_aux=None):
        E_Theta = self.E_Theta
        Var_Theta = E_Theta / self.b_Theta
        self.c_pg, self.wbar, self._row_chunk = pg_tilt(
            E_design=E_Theta,
            Var_design=Var_Theta,
            mu_v=self.mu_v,
            sigma_v_diag=self.sigma_v_diag,
            mu_gamma=self.mu_gamma,
            X_aux=self.X_aux_patient,
            row_chunk=self._row_chunk,
        )

    def _update_v(self, y=None, X_aux=None, iteration=0, ramp=1.0):
        E_Theta = self.E_Theta
        Var_Theta = E_Theta / self.b_Theta
        effective_rw = ramp * self.regression_weight
        self.mu_v, self.sigma_v_diag, self._row_chunk = pg_update_v(
            E_design=E_Theta,
            Var_design=Var_Theta,
            mu_v=self.mu_v,
            sigma_v_diag=self.sigma_v_diag,
            mu_gamma=self.mu_gamma,
            X_aux=self.X_aux_patient,
            y=self.y_patient,
            wbar=self.wbar,
            sample_weights=self._sample_weights,
            b_v=self.b_v,
            K=self.K,
            effective_rw=effective_rw,
            ramp=ramp,
            row_chunk=self._row_chunk,
        )

    def _update_gamma(self, y=None, X_aux=None, iteration=0, ramp=1.0):
        if self.p_aux == 0:
            return
        E_Theta = self.E_Theta
        effective_rw = ramp * self.regression_weight
        self.mu_gamma, self.Sigma_gamma, self._row_chunk = pg_update_gamma(
            E_design=E_Theta,
            mu_v=self.mu_v,
            mu_gamma=self.mu_gamma,
            Sigma_gamma=self.Sigma_gamma,
            sigma_gamma_param=self.sigma_gamma,
            X_aux=self.X_aux_patient,
            y=self.y_patient,
            wbar=self.wbar,
            sample_weights=self._sample_weights,
            K=self.K,
            effective_rw=effective_rw,
            ramp=ramp,
            p_aux=self.p_aux,
            row_chunk=self._row_chunk,
        )

    # ------------------------------------------------------------------
    # fit() — Algorithm 2 ordering. Minimal harness vs parent's; missing the
    # parent's heavy diagnostics / restart / multi-mode plumbing — those can
    # be ported as needed once the math is verified.
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train,
        y_train,
        X_aux_train=None,
        X_val=None,
        y_val=None,
        X_aux_val=None,
        *,
        patient_ids: np.ndarray,
        cell_type_ids: np.ndarray,
        patient_ids_val: Optional[np.ndarray] = None,
        cell_type_ids_val: Optional[np.ndarray] = None,
        max_iter: int = 600,
        check_freq: int = 5,
        tol: float = 0.001,
        v_warmup: int = 50,
        verbose: bool = True,
        early_stopping: str = "heldout_ll",
        holl_patience: int = 100,
        holl_n_iter: int = 20,
        **_kwargs,
    ):
        """Algorithm 2 hierarchical PG-CAVI.

        Args:
            X_train : (n, p) count matrix (sparse or dense).
            y_train : (n,) or (n, κ) cell-level label broadcast — deduped per patient.
            X_aux_train : (n, p_aux_raw) cell-level aux — deduped per patient.
            patient_ids : (n,) int — cell → patient index in [0, G).
            cell_type_ids : (n,) int — cell → cell-type index in [0, T).
            early_stopping : 'heldout_ll' (held-out Poisson LL, label-blind) or 'elbo'.
        """
        if patient_ids is None or cell_type_ids is None:
            raise ValueError("CAVIHierarchical.fit requires patient_ids and cell_type_ids.")

        # Run parent's _initialize for gene block + ξ + θ + Bayesian-Lasso s.
        # (Pass cell-level y and X_aux so the parent sizes things; the hier
        # init then re-points kappa/p_aux/sample_weights to the patient level.)
        if y_train.ndim == 1:
            y_cell = y_train.astype(np.float32)[:, None]
        else:
            y_cell = y_train.astype(np.float32)
        if X_aux_train is None:
            X_aux_cell = np.zeros((X_train.shape[0], 0), dtype=np.float32)
        else:
            X_aux_cell = np.asarray(X_aux_train, dtype=np.float32)
        X_aux_cell = self._prepend_intercept(X_aux_cell, n=X_train.shape[0])
        super()._initialize(X_train, y_cell, X_aux_cell)

        # Dedupe to per-patient y, X_aux. Both must be constant within a patient.
        n = X_train.shape[0]
        G = int(np.max(patient_ids)) + 1
        y_pat = np.zeros((G, y_cell.shape[1]), dtype=np.float32)
        Xaux_pat = np.zeros((G, X_aux_cell.shape[1]), dtype=np.float32)
        seen = np.zeros(G, dtype=bool)
        y_cell_np = to_numpy(y_cell) if hasattr(y_cell, "device") else y_cell
        Xaux_np = to_numpy(X_aux_cell) if hasattr(X_aux_cell, "device") else X_aux_cell
        for i in range(n):
            g = int(patient_ids[i])
            if not seen[g]:
                y_pat[g] = y_cell_np[i]
                Xaux_pat[g] = Xaux_np[i]
                seen[g] = True
            else:
                if not np.allclose(y_pat[g], y_cell_np[i]):
                    raise ValueError(
                        f"y is not constant within patient {g}: dedup failed."
                    )
                # Aux can include a constant intercept; just confirm match.
                if not np.allclose(Xaux_pat[g], Xaux_np[i]):
                    raise ValueError(
                        f"X_aux is not constant within patient {g}: dedup failed."
                    )

        # Initialize hier-specific state (Θ, ζ, patient-level PG buffers, head sized to κ).
        self._initialize_hier(
            patient_id_per_cell=np.asarray(patient_ids, dtype=np.int32),
            cell_type_id_per_cell=np.asarray(cell_type_ids, dtype=np.int32),
            y_patient=y_pat,
            X_aux_patient=Xaux_pat,
        )

        # Auto-scale regression_weight (consistent with flat) — see vi_cavi.py:1824.
        if self._nnz is not None and self.n is not None:
            self.regression_weight = self.regression_weight * float(self._nnz) / float(self.n)
            if verbose:
                print(f"  [hier] regression_weight scaled to {self.regression_weight:.1f} "
                      f"(nnz/n = {self._nnz}/{self.n})")

        # Held-out fold-in for Poisson LL early stopping (label-blind path).
        ramp_iters = 200
        self.elbo_history_ = []
        self.holl_history_ = []
        t_start = time.time()

        # HO-LL early stopping needs val cells + their patient/cell-type ids so
        # transform_hier can fold them in. Falls back to ELBO-tolerance stopping
        # when any of these is missing.
        holl_ready = (
            early_stopping == "heldout_ll"
            and X_val is not None
            and patient_ids_val is not None
            and cell_type_ids_val is not None
        )
        use_elbo_stop = not holl_ready
        best_holl = -np.inf
        iters_since_best = 0

        for t in range(max_iter):
            # 0. Gene-block: ϕ, β̃, ρ, π, η.
            z_sum_beta, z_sum_theta = self._compute_phi_sparse(random_init=(t == 0))
            theta_col_sum = (self.a_theta / self.b_theta).sum(axis=0)
            self._update_beta(z_sum_beta)
            self._update_eta()
            if self.use_spike_slab:
                self._update_r_beta(z_sum_beta, theta_col_sum)
            self._refresh_log_caches()

            # 1. θ — structured rate, no R.
            ramp = min(1.0, max(0.0, (t - v_warmup + 1) / ramp_iters)) if t >= v_warmup else 0.0
            self._update_theta(z_sum_theta, ramp=ramp)
            self._update_xi()

            # 2. ζ, Θ — new hierarchical levels.
            self._update_zeta()
            if t >= v_warmup:
                self._update_Theta(ramp=ramp)

            # 3. Supervised head — only after warmup. PG tilt before υ/γ so they
            # see the freshest ω̄.
            if t >= v_warmup:
                self._update_omega()
                self._update_v(ramp=ramp)
                self._update_gamma(ramp=ramp)
                # Refresh tilt after parameter changes so ELBO is at the optimum.
                self._update_omega()

            # 4. Progress / stop.
            if t % check_freq == 0 or t == max_iter - 1:
                try:
                    elbo, poisson_ll, reg_ll = self._compute_elbo_hier()
                except Exception as exc:
                    if verbose:
                        print(f"  iter {t}: ELBO compute skipped ({exc})")
                    elbo, poisson_ll, reg_ll = float("nan"), float("nan"), float("nan")
                self.elbo_history_.append((t, float(elbo), float(poisson_ll), float(reg_ll)))
                if verbose:
                    print(f"  iter {t:4d}  ELBO={elbo:.3e}  Pois={poisson_ll:.3e}  "
                          f"Reg={reg_ll:.3e}  ramp={ramp:.3f}")

                if use_elbo_stop and len(self.elbo_history_) >= 3:
                    prev = self.elbo_history_[-2][1]
                    if prev != 0 and abs((elbo - prev) / abs(prev)) < tol:
                        if verbose:
                            print(f"  ELBO tol reached at iter {t}; stopping.")
                        break

                if holl_ready and t >= v_warmup:
                    try:
                        holl = self._compute_heldout_ll_hier(
                            X_val, patient_ids_val, cell_type_ids_val,
                            n_iter=holl_n_iter,
                        )
                    except Exception as exc:
                        if verbose:
                            print(f"  iter {t}: HO-LL skipped ({exc})")
                        holl = float("nan")
                    self.holl_history_.append((t, float(holl)))
                    if verbose:
                        print(f"  iter {t:4d}  HO-LL={holl:.4e}  best={best_holl:.4e}  "
                              f"iters_since_best={iters_since_best}")
                    if np.isfinite(holl) and holl > best_holl:
                        best_holl = holl
                        iters_since_best = 0
                    else:
                        iters_since_best += check_freq
                    if iters_since_best >= holl_patience and t >= v_warmup + holl_patience:
                        if verbose:
                            print(f"  HO-LL patience exhausted at iter {t}; stopping.")
                        break

        if verbose:
            print(f"\nHierarchical CAVI done in {time.time() - t_start:.1f}s. "
                  f"Final ELBO = {self.elbo_history_[-1][1] if self.elbo_history_ else 'n/a'}")
            if self.holl_history_:
                print(f"  Best HO-LL = {best_holl:.4e}")
        return self

    # ------------------------------------------------------------------
    # ELBO override (patient-level L_sup; adds Γ-prior/entropy for Θ, ζ)
    # ------------------------------------------------------------------
    def _compute_elbo_hier(self) -> Tuple[float, float, float]:
        """Tempered hier ELBO (full).

        Structural differences vs flat (see parent _compute_elbo):
          (1) θ prior uses structured rate E[ξ_i Θ_{g_i,ℓ} ζ_{t_i,ℓ}].
          (2) L_sup is patient-level (design = E[Θ], y = y_patient).
          (3) Two new factor groups: Θ (G × K), ζ (T × K) — each adds a
              Gamma prior + entropy.

        Common terms (Poisson recon, θ entropy, β gene block, ξ, η, v, γ)
        are delegated to module-level _elbo_*_block helpers shared with
        the flat ELBO — see vi_cavi.py.
        """
        E_beta = self.E_beta
        E_log_beta = self._E_log_beta_cache
        E_xi = self.E_xi
        E_eta = self.E_eta
        E_log_xi = self._digamma_a_xi - xp.log(self.b_xi)
        E_log_eta = self._digamma_a_eta - xp.log(self.b_eta)
        E_Theta = self.E_Theta
        E_zeta = self.E_zeta
        E_log_Theta = self.E_log_Theta
        E_log_zeta = self.E_log_zeta

        min_chunk = max(1024, self.n // 256)

        # ===== Poisson LL via Raikov decomposition (shared) =========
        poisson_ll, E_log_theta, self._row_chunk = _elbo_poisson_recon(
            a_theta=self.a_theta, b_theta=self.b_theta,
            E_log_beta=E_log_beta, E_beta=E_beta,
            X_row=self._X_row, X_col=self._X_col, X_data=self._X_data,
            nnz=self._nnz, n=self.n, K=self.K,
            row_chunk=self._row_chunk, effective_chunk=self._effective_chunk,
            min_chunk=min_chunk, active_beta=self._active_beta,
            gammaln_data_sum=self._gammaln_data_sum,
        )
        elbo = poisson_ll

        # ===== Patient-level L_sup via kernel (TEMPERED) =============
        Var_Theta = E_Theta / self.b_Theta
        Lsup_raw, self._row_chunk = pg_Lsup(
            E_design=E_Theta,
            Var_design=Var_Theta,
            mu_v=self.mu_v,
            sigma_v_diag=self.sigma_v_diag,
            mu_gamma=self.mu_gamma,
            X_aux=self.X_aux_patient,
            y=self.y_patient,
            c=self.c_pg,
            wbar=self.wbar,
            sample_weights=self._sample_weights,
            row_chunk=self._row_chunk,
        )
        regression_ll = float(Lsup_raw)
        elbo += self.regression_weight * regression_ll

        # ===== θ prior (STRUCTURED RATE) — chunked, hier-specific ====
        # (a-1) E[log θ] + a (E[log ξ] + E[log Θ_{g_i}] + E[log ζ_{t_i}])
        #                  - E[ξ Θ_{g_i} ζ_{t_i}] E[θ] - log Γ(a)
        theta_prior = 0.0
        i0 = 0
        while i0 < self.n:
            i1 = min(i0 + self._row_chunk, self.n)
            try:
                a_theta_c = self.a_theta[i0:i1]
                b_theta_c = self.b_theta[i0:i1]
                E_theta_c = a_theta_c / b_theta_c
                E_log_theta_c = E_log_theta[i0:i1]

                E_log_Theta_pc = E_log_Theta[self.patient_id[i0:i1]]
                E_log_zeta_pc = E_log_zeta[self.cell_type_id[i0:i1]]
                E_Theta_pc = E_Theta[self.patient_id[i0:i1]]
                E_zeta_pc = E_zeta[self.cell_type_id[i0:i1]]
                struct_rate = E_xi[i0:i1, None] * E_Theta_pc * E_zeta_pc

                theta_prior = theta_prior + xp.sum(
                    (self.a - 1) * E_log_theta_c
                    + self.a * (E_log_xi[i0:i1, None] + E_log_Theta_pc + E_log_zeta_pc)
                    - struct_rate * E_theta_c
                )
                i0 = i1
            except Exception as exc:
                if _is_oom_error(exc) and self._row_chunk > min_chunk:
                    self._row_chunk = max(min_chunk, self._row_chunk // 2)
                    continue
                raise
        del E_log_theta

        theta_entropy, self._row_chunk = _elbo_theta_entropy_chunked(
            a_theta=self.a_theta, b_theta=self.b_theta, n=self.n,
            row_chunk=self._row_chunk, min_chunk=min_chunk,
        )
        elbo += theta_prior
        elbo -= self.n * self.K * gammaln(self.a)
        elbo += theta_entropy

        # ===== Shared gene-side / hyperprior blocks ==================
        elbo += _elbo_beta_block(
            a_beta=self.a_beta, b_beta=self.b_beta,
            E_log_eta=E_log_eta, E_eta=E_eta, c_prior=self.c,
            p=self.p, K=self.K,
            active_beta=self._active_beta, n_active_beta=self._n_active_beta,
            use_spike_slab=self.use_spike_slab, pw_active=self._pw_active,
            r_beta=self.r_beta,
            a_pi=getattr(self, "a_pi", None), b_pi=getattr(self, "b_pi", None),
            alpha_pi=self.alpha_pi, beta_pi=getattr(self, "beta_pi", None),
        )
        elbo += _elbo_xi_block(
            E_log_xi=E_log_xi, E_xi=E_xi,
            a_xi=self.a_xi, b_xi=self.b_xi,
            gammaln_a_xi_cached=self._gammaln_a_xi,
            digamma_a_xi_cached=self._digamma_a_xi,
            ap=self.ap, bp=self.bp, n=self.n,
        )
        elbo += _elbo_eta_block(
            E_log_eta=E_log_eta, E_eta=E_eta,
            a_eta=self.a_eta, b_eta=self.b_eta,
            gammaln_a_eta_cached=self._gammaln_a_eta,
            digamma_a_eta_cached=self._digamma_a_eta,
            cp=self.cp, dp=self.dp, p=self.p,
        )
        elbo += _elbo_v_block(
            mu_v=self.mu_v, sigma_v_diag=self.sigma_v_diag, b_v=self.b_v,
        )
        elbo += _elbo_gamma_aux_block(
            mu_gamma=self.mu_gamma, Sigma_gamma=self.Sigma_gamma,
            sigma_gamma=self.sigma_gamma, kappa=self.kappa, p_aux=self.p_aux,
        )

        # ===== Θ prior + entropy (hier-specific) =====================
        log_lambda_Theta = float(np.log(self.lambda_Theta))
        gammaln_alpha_Theta = float(gammaln(self.alpha_Theta))
        elbo += xp.sum(
            (self.alpha_Theta - 1) * E_log_Theta
            + self.alpha_Theta * log_lambda_Theta
            - self.lambda_Theta * E_Theta
        )
        elbo -= self.G * self.K * gammaln_alpha_Theta
        elbo += xp.sum(self.a_Theta - xp.log(self.b_Theta) + gammaln(self.a_Theta)
                       + (1 - self.a_Theta) * digamma(self.a_Theta))

        # ===== ζ prior + entropy (hier-specific) =====================
        log_lambda_zeta = float(np.log(self.lambda_zeta))
        gammaln_alpha_zeta = float(gammaln(self.alpha_zeta))
        elbo += xp.sum(
            (self.alpha_zeta - 1) * E_log_zeta
            + self.alpha_zeta * log_lambda_zeta
            - self.lambda_zeta * E_zeta
        )
        elbo -= self.T * self.K * gammaln_alpha_zeta
        elbo += xp.sum(self.a_zeta - xp.log(self.b_zeta) + gammaln(self.a_zeta)
                       + (1 - self.a_zeta) * digamma(self.a_zeta))

        return float(elbo), float(poisson_ll), float(regression_ll)

    # ------------------------------------------------------------------
    # Held-out Poisson LL (label-blind early-stopping signal).
    # ------------------------------------------------------------------
    def _compute_heldout_ll_hier(
        self,
        X_val,
        patient_id_val: np.ndarray,
        cell_type_val: np.ndarray,
        n_iter: int = 20,
        unseen_celltype: str = "prior",
    ) -> float:
        """Mean held-out Poisson LL per cell after transform_hier fold-in.

        Returns Σ_ij [x_ij log(Σ_k θ_ik β_jk) - Σ_k θ_ik β_jk] / n_val,
        where θ is the Poisson-only fold-in (R = 0, R^Θ = 0). Label-blind.
        """
        if sp.issparse(X_val):
            X_coo = X_val.tocoo()
        else:
            X_coo = sp.coo_matrix(X_val)
        n_val = X_val.shape[0]

        result = self.transform_hier(
            X_new=X_val,
            patient_id_new=patient_id_val,
            cell_type_new=cell_type_val,
            n_iter=n_iter,
            unseen_celltype=unseen_celltype,
        )
        E_theta_v = to_device(result["E_theta_new"].astype(np.float32))

        # Reconstruction rate sum: Σ_ik θ_ik · (Σ_j β_jk) on active beta only.
        if self._active_beta is not None:
            beta_col_sum = xp.where(self._active_beta, self.E_beta, 0.0).sum(axis=0)
        else:
            beta_col_sum = self.E_beta.sum(axis=0)
        rate_penalty = float(xp.sum(E_theta_v @ beta_col_sum))

        # Poisson log-sum-exp on nnz entries.
        from VariationalInference.jax_backend import logsumexp_rows as _lse_rows
        row = to_device(X_coo.row.astype(np.int32))
        col = to_device(X_coo.col.astype(np.int32))
        data = to_device(X_coo.data.astype(np.float32))
        E_log_theta_v = xp.log(xp.maximum(E_theta_v, 1e-30))
        E_log_beta = self._E_log_beta_cache
        log_rates = E_log_theta_v[row] + E_log_beta[col]
        log_sum = _lse_rows(log_rates).ravel()
        log_sum = xp.maximum(log_sum, -100.0)
        recon_ll = float(xp.dot(data, log_sum))
        return (recon_ll - rate_penalty) / float(n_val)

    # ------------------------------------------------------------------
    # Poisson-only fold-in for held-out patients (Stage 5).
    # ------------------------------------------------------------------
    def transform_hier(
        self,
        X_new,
        patient_id_new: np.ndarray,
        cell_type_new: np.ndarray,
        n_iter: int = 50,
        unseen_celltype: str = "error",
    ) -> Dict[str, Any]:
        """Inductive fold-in for new patients (and their cells).

        Inference is Poisson-only:
          - θ_i for new cells updates with R_{iℓ} = 0 (no supervision).
          - Θ_g for new patients updates with R^Θ_{gℓ} = 0 (Poisson-only fold-in).
          - ζ_t for known cell types is held at the trained posterior mean.
            Unseen types either raise (default) or use the cohort prior mean
            (`unseen_celltype='prior'`) — eq C.8's prior mean is α_ζ / λ_ζ.

        Returns dict with E_theta_new (n_new, K), E_Theta_new (G_new, K), and
        the patient/cell-type mappings used.
        """
        if X_new.ndim != 2:
            raise ValueError("X_new must be 2D (n_new, p).")
        if sp.issparse(X_new):
            X_coo = X_new.tocoo()
        else:
            X_coo = sp.coo_matrix(X_new)
        n_new = X_new.shape[0]
        patient_id_new = np.asarray(patient_id_new, dtype=np.int32)
        cell_type_new = np.asarray(cell_type_new, dtype=np.int32)
        if patient_id_new.shape[0] != n_new or cell_type_new.shape[0] != n_new:
            raise ValueError("patient_id_new / cell_type_new must align with X_new.")

        # Validate cell types against trained ζ.
        seen = set(range(self.T))
        unseen = sorted(set(int(t) for t in cell_type_new) - seen)
        if unseen:
            known = sorted(seen)
            if unseen_celltype == "error":
                raise ValueError(
                    f"Unseen cell-type id(s) in test: {unseen}. "
                    f"Known: {known}. Pass unseen_celltype='prior' to back off to "
                    f"the cohort prior mean α_ζ/λ_ζ."
                )
            elif unseen_celltype == "prior":
                prior_zeta = (self.alpha_zeta / self.lambda_zeta) * xp.ones(self.K, dtype=np.float32)
                # Extend E_zeta with prior rows for unseen types.
                T_ext = max(int(cell_type_new.max()) + 1, self.T)
                E_zeta_ext = xp.zeros((T_ext, self.K), dtype=np.float32)
                E_zeta_ext = E_zeta_ext.at[:self.T].set(self.E_zeta) if USE_JAX else (
                    np.concatenate([to_numpy(self.E_zeta),
                                     np.tile(to_numpy(prior_zeta), (T_ext - self.T, 1))],
                                    axis=0))
                if USE_JAX:
                    for t in unseen:
                        E_zeta_ext = E_zeta_ext.at[t].set(prior_zeta)
            else:
                raise ValueError(f"unseen_celltype must be 'error' or 'prior', got {unseen_celltype!r}")
        else:
            E_zeta_ext = self.E_zeta

        # Re-map patients to a dense [0, G_new) range.
        unique_pat, pat_inv = np.unique(patient_id_new, return_inverse=True)
        G_new = len(unique_pat)
        pat_inv = pat_inv.astype(np.int32)
        cell_type_dev = to_device(cell_type_new)
        patient_dev = to_device(pat_inv)

        # Initialize new Θ, θ, ξ at prior moments.
        a_theta_new = to_device(np.full((n_new, self.K), self.a, dtype=np.float32))
        b_theta_new = to_device(np.full((n_new, self.K), self.lambda_Theta, dtype=np.float32))
        a_Theta_new = to_device(np.full((G_new, self.K), self.alpha_Theta, dtype=np.float32))
        b_Theta_new = to_device(np.full((G_new, self.K), self.lambda_Theta, dtype=np.float32))
        a_xi_new = to_device(np.full(n_new, self.ap + self.K * self.a, dtype=np.float32))
        b_xi_new = to_device(np.full(n_new, self.bp, dtype=np.float32))

        # COO indices on device.
        row = to_device(X_coo.row.astype(np.int32))
        col = to_device(X_coo.col.astype(np.int32))
        data = to_device(X_coo.data.astype(np.float32))
        from VariationalInference.jax_backend import phi_chunk_core
        E_log_beta = self._E_log_beta_cache
        if self._active_beta is not None:
            beta_col_sums = xp.where(self._active_beta, self.E_beta, 0.0).sum(axis=0)
        else:
            beta_col_sums = self.E_beta.sum(axis=0)

        for _ in range(n_iter):
            E_log_theta = digamma(a_theta_new) - xp.log(b_theta_new)
            E_theta = a_theta_new / b_theta_new
            E_xi = a_xi_new / b_xi_new
            E_Theta = a_Theta_new / b_Theta_new

            # 1. Raikov allocation → z_sum_theta
            z_sum = xp.zeros((n_new, self.K), dtype=np.float32)
            chunk = 1_000_000
            for s in range(0, len(data), chunk):
                e = min(s + chunk, len(data))
                row_c = row[s:e]
                col_c = col[s:e]
                data_c = data[s:e]
                Xphi = phi_chunk_core(E_log_theta[row_c], E_log_beta[col_c], data_c)
                z_sum = scatter_add_to(z_sum, row_c, Xphi, sorted_indices=False)
                del Xphi

            # 2. θ update — Poisson-only (no R correction).
            E_zeta_cell = E_zeta_ext[cell_type_dev]              # (n_new, K)
            E_Theta_cell = E_Theta[patient_dev]                  # (n_new, K)
            rate_struct = E_xi[:, None] * E_Theta_cell * E_zeta_cell
            a_theta_new = self.a + z_sum
            b_theta_new = rate_struct + beta_col_sums[None, :]
            b_theta_new = xp.maximum(b_theta_new, 1e-2)

            # 3. ξ update.
            E_theta = a_theta_new / b_theta_new
            rate_incr = (E_Theta_cell * E_zeta_cell * E_theta).sum(axis=1)
            b_xi_new = self.bp + rate_incr
            b_xi_new = xp.maximum(b_xi_new, 1e-6)

            # 4. Θ update — Poisson-only (drop R^Θ): pool cells per new patient.
            per_cell = (a_xi_new / b_xi_new)[:, None] * E_zeta_cell * (a_theta_new / b_theta_new)
            pooled = xp.zeros((G_new, self.K), dtype=np.float32)
            pooled = scatter_add_to(pooled, patient_dev, per_cell, sorted_indices=False)
            ones_n = xp.ones(n_new, dtype=np.float32)
            cg_counts = xp.zeros(G_new, dtype=np.float32)
            cg_counts = scatter_add_to(cg_counts, patient_dev, ones_n, sorted_indices=False)
            a_Theta_new = self.alpha_Theta + cg_counts[:, None] * self.a
            b_Theta_new = xp.maximum(self.lambda_Theta + pooled, 1e-6)

        return {
            "E_theta_new": to_numpy(a_theta_new / b_theta_new),
            "E_Theta_new": to_numpy(a_Theta_new / b_Theta_new),
            "patient_id_dense_to_orig": unique_pat,
            "patient_id_per_cell_dense": pat_inv,
        }
