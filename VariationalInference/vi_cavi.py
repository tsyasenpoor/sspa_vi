"""
Coordinate Ascent VI for Supervised Poisson Factorization (DRGP)
================================================================

Poisson factorization core follows scHPF (Levitin et al., MSB 2019)
exactly. Supervision via Jaakkola-Jordan logistic regression bound
from the DRGP manuscript appendix.

scHPF update equations (Gopalan et al. 2014, Eqs 7-9):
  phi_{ij} ~ exp{ psi(a^theta_{ik}) - log(b^theta_{ik}) + psi(a^beta_{jk}) - log(b^beta_{jk}) }
  a^beta_{jk} = c + Sum_i x_{ij} phi_{ijk}
  b^beta_{jk} = E[eta_j] + Sum_i E[theta_{ik}]
  a^theta_{ik} = a + Sum_j x_{ij} phi_{ijk}
  b^theta_{ik} = E[xi_i] + Sum_j E[beta_{jk}]  (+ JJ regression correction)

Hierarchical priors:
  xi_i ~ Gamma(a', b')     ->  a^xi_i = a' + K*a  (constant)
                                b^xi_i = b' + Sum_k E[theta_{ik}]
  eta_j ~ Gamma(c', d')    ->  a^eta_j = c' + K*c  (constant)
                                b^eta_j = d' + Sum_k E[beta_{jk}]

Key differences from previous vi_cavi.py:
1. bp, dp are SCALARS (scHPF), not per-cell/gene vectors
2. Random init: U(0.5*prior, 1.5*prior) for both shape AND rate
3. First iteration uses random Dirichlet phi (scHPF symmetry breaking)
4. No damping -- pure coordinate ascent
5. No diversity noise, no annealing, no warmup phases
6. All parameters (theta, v, gamma, zeta) learned jointly from iteration 0
7. No spike-and-slab

GPU/JAX acceleration:
- When JAX is installed the hot numerical paths (phi, ELBO, updates)
  run on GPU if available, otherwise on CPU via XLA.
- Falls back transparently to NumPy/SciPy when JAX is absent.

References:
- Gopalan, Hofman, Blei (2014) "Scalable Recommendation with HPF"
- Levitin et al. (2019) "scHPF", Molecular Systems Biology
- Jaakkola & Jordan (2000) variational logistic bound
- Hoffman et al. (2013) "Stochastic Variational Inference"
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Any, List
import time

try:
    from .jax_backend import (
        xp, USE_JAX, HAS_GPU, to_device, to_numpy,
        digamma, gammaln, logsumexp_rows, softmax_rows,
        log_expit, lambda_jj, scatter_add_to, phi_chunk_core,
        expit as _expit, backend_info,
    )
except ImportError:
    from jax_backend import (
        xp, USE_JAX, HAS_GPU, to_device, to_numpy,
        digamma, gammaln, logsumexp_rows, softmax_rows,
        log_expit, lambda_jj, scatter_add_to, phi_chunk_core,
        expit as _expit, backend_info,
    )


def _auto_chunk_size(nnz, K, target_gb=None):
    """Auto-tune chunk size to target a given work-array memory budget.

    With K factors, each chunk of C entries uses C * K * 8 bytes.
    On GPU, uses a larger budget (up to available VRAM) to reduce loop overhead.
    On CPU, targets ~4GB by default.
    """
    if target_gb is None:
        if HAS_GPU:
            # Use up to ~50% of GPU memory for work arrays (leave room for
            # parameters, caches, and OS).  Query actual VRAM if possible.
            try:
                import jax
                dev = [d for d in jax.devices() if d.platform == "gpu"][0]
                mem_bytes = dev.memory_stats()["bytes_limit"]
                target_gb = mem_bytes / (1024 ** 3) * 0.50
            except Exception:
                target_gb = 12.0  # conservative GPU default
        else:
            target_gb = 4.0
    max_by_mem = int(target_gb * (1024 ** 3) / (K * 8))
    # At least 1M, at most nnz (single pass)
    return max(1_000_000, min(max_by_mem, nnz))


class CAVI:
    """
    CAVI for Supervised Poisson Factorization.

    Parameters
    ----------
    n_factors : int
        Number of latent factors K.
    a : float
        Gamma shape prior for theta (cell loadings). Default 0.3 (scHPF).
    ap : float
        Gamma shape prior for xi (cell capacity). Default 1.0.
    c : float
        Gamma shape prior for beta (gene loadings). Default 0.3 (scHPF).
    cp : float
        Gamma shape prior for eta (gene capacity). Default 1.0.
    sigma_v : float
        Gaussian prior std for v (regression weights). Used when v_prior='normal'.
    b_v : float
        Laplace prior scale for v (regression weights). Used when v_prior='laplace'.
        Smaller b_v = stronger sparsity. Var[v] = 2*b_v^2.
    v_prior : str
        Prior distribution for v: 'normal' (Gaussian) or 'laplace' (Bayesian Lasso).
    sigma_gamma : float
        Gaussian prior std for gamma (auxiliary covariate weights).
    regression_weight : float
        Scalar weight for the classification term.
    use_class_weights : bool
        If True, apply balanced class weights per label to the Bernoulli
        regression loss.  Positive samples for label k are weighted by
        n / (2 * n_pos_k) and negatives by n / (2 * n_neg_k), so the
        total weighted count per label stays the same but rare classes
        contribute proportionally more.
    mode : str
        'unmasked', 'masked', 'pathway_init', 'combined'.
    """

    def __init__(
        self,
        n_factors: int,
        a: float = 0.3,
        ap: float = 1.0,
        c: float = 0.3,
        cp: float = 1.0,
        sigma_v: float = 1.0,
        b_v: float = 1.0,
        v_prior: str = 'normal',
        sigma_gamma: float = 1.0,
        regression_weight: float = 1.0,
        use_class_weights: bool = True,
        random_state: Optional[int] = None,
        mode: str = 'unmasked',
        pathway_mask: Optional[np.ndarray] = None,
        pathway_names: Optional[List[str]] = None,
        n_pathway_factors: Optional[int] = None,
        nnz_chunk_size: int = 1_000_000,
        **_ignored,
    ):
        self.K = n_factors
        self.a = a
        self.ap = ap
        self.c = c
        self.cp = cp
        self.sigma_v = sigma_v
        self.b_v = b_v
        self.v_prior = v_prior.lower()
        if self.v_prior not in ('normal', 'laplace'):
            raise ValueError(f"v_prior must be 'normal' or 'laplace', got '{v_prior}'")
        self.sigma_gamma = sigma_gamma
        self.regression_weight = regression_weight
        self.use_class_weights = use_class_weights
        # Cap on JJ auxiliary zeta.  Keeps lambda_JJ(zeta) >= lambda_JJ(zeta_max) > 0,
        # preventing the quadratic braking on theta from vanishing.
        # At zeta_max=4: lambda_min ~ 0.060.  The JJ bound remains valid for any zeta.
        self.zeta_max = 4.0

        self.nnz_chunk_size = nnz_chunk_size
        self.mode = mode
        self.pathway_mask = pathway_mask
        self.pathway_names = pathway_names
        self.n_pathway_factors = n_pathway_factors

        if mode in ['masked', 'pathway_init'] and pathway_mask is None:
            raise ValueError(f"pathway_mask required for mode='{mode}'")
        if mode == 'combined':
            if pathway_mask is None or n_pathway_factors is None:
                raise ValueError("pathway_mask and n_pathway_factors required for combined mode")

        if random_state is not None:
            np.random.seed(random_state)
            self.seed_used_ = random_state
        else:
            self.seed_used_ = None

        # Will be set in fit()
        self.n = self.p = self.kappa = self.p_aux = None
        self.bp = self.dp = None

    # =================================================================
    # scHPF empirical hyperparameters
    # =================================================================

    @staticmethod
    def _mean_var_ratio(X, axis):
        """ap * mean / var along axis, as in scHPF."""
        if sp.issparse(X):
            sums = np.asarray(X.sum(axis=axis)).ravel()
        else:
            sums = np.asarray(X.sum(axis=axis)).ravel()
        return float(np.mean(sums) / max(np.var(sums), 1e-10))

    # =================================================================
    # Initialization (scHPF pattern)
    # =================================================================

    def _initialize(self, X, y, X_aux):
        """
        scHPF initialization:
        1. Empirical bp, dp (scalars)
        2. Random Gamma params: U(0.5*prior, 1.5*prior)
        3. xi.shape, eta.shape set to constants
        4. Transfer all arrays to device (GPU if available)
        """
        if sp.issparse(X):
            self.n, self.p = X.shape
        else:
            self.n, self.p = X.shape

        self.kappa = 1 if y.ndim == 1 else y.shape[1]
        self.p_aux = X_aux.shape[1] if X_aux is not None and X_aux.size > 0 else 0

        K = self.K

        # --- Empirical hyperparameters (scHPF: scalar bp, dp) ---
        self.bp = self.ap * self._mean_var_ratio(X, axis=1)
        self.dp = self.cp * self._mean_var_ratio(X, axis=0)
        # Clip if bp >> dp (scHPF)
        if self.bp > 1000 * self.dp:
            old_dp = self.dp
            self.dp = self.bp / 1000
            print(f"Clipping dp: was {old_dp:.4f} now {self.dp:.4f}")

        bp, dp = self.bp, self.dp

        # --- xi: Gamma(ap + K*a, b_xi) ---
        # shape is CONSTANT = ap + K*a
        self.a_xi = np.full(self.n, self.ap + K * self.a)
        self.b_xi = np.random.uniform(0.5 * bp, 1.5 * bp, self.n)

        # --- theta: Gamma(a_theta, b_theta) ---
        self.a_theta = np.random.uniform(0.5 * self.a, 1.5 * self.a,
                                         (self.n, K))
        self.b_theta = np.random.uniform(0.5 * bp, 1.5 * bp, (self.n, K))

        # --- eta: Gamma(cp + K*c, b_eta) ---
        self.a_eta = np.full(self.p, self.cp + K * self.c)
        self.b_eta = np.random.uniform(0.5 * dp, 1.5 * dp, self.p)

        # --- beta: Gamma(a_beta, b_beta) ---
        self.a_beta = np.random.uniform(0.5 * self.c, 1.5 * self.c,
                                        (self.p, K))
        self.b_beta = np.random.uniform(0.5 * dp, 1.5 * dp, (self.p, K))

        # --- Apply pathway mask if needed ---
        self._init_beta_mask()

        # --- v: init small random ---
        self.mu_v = np.random.randn(self.kappa, K) * 0.01
        if self.v_prior == 'normal':
            # N(0, sigma_v^2/K) prior -- K-scaled so total logit theta*v has variance ~ sigma_v^2
            self.sigma_v_diag = np.full((self.kappa, K), (self.sigma_v ** 2) / K)
        else:
            # Laplace (Bayesian Lasso): init sigma_v_diag from Laplace variance = 2*b_v^2
            self.sigma_v_diag = np.full((self.kappa, K), 2.0 * self.b_v ** 2)

        # --- gamma: N(0, sigma_gamma^2) ---
        if self.p_aux > 0:
            self.mu_gamma = np.zeros((self.kappa, self.p_aux))
            self.Sigma_gamma = np.stack([
                np.eye(self.p_aux) * self.sigma_gamma ** 2
                for _ in range(self.kappa)
            ])
        else:
            self.mu_gamma = np.zeros((self.kappa, 0))
            self.Sigma_gamma = np.zeros((self.kappa, 0, 0))

        # --- JJ auxiliary ---
        self.zeta = np.ones((self.n, self.kappa))

        # --- Oscillation tracking for v update ---
        self._v_prev_mu = None
        self._v_raw_prev = None

        # --- Class weights for imbalanced labels ---
        y_2d = y if y.ndim > 1 else y[:, None]
        if self.use_class_weights:
            self._sample_weights = np.ones((self.n, self.kappa), dtype=np.float64)
            for k in range(self.kappa):
                n_pos = np.sum(y_2d[:, k] > 0.5)
                n_neg = self.n - n_pos
                if n_pos > 0 and n_neg > 0:
                    w_pos = self.n / (2.0 * n_pos)
                    w_neg = self.n / (2.0 * n_neg)
                    self._sample_weights[:, k] = np.where(
                        y_2d[:, k] > 0.5, w_pos, w_neg
                    )
        else:
            self._sample_weights = np.ones((self.n, self.kappa), dtype=np.float64)

        if self.use_class_weights:
            for k in range(self.kappa):
                n_pos = np.sum(y_2d[:, k] > 0.5)
                n_neg = self.n - n_pos
                w_pos = self.n / (2.0 * n_pos) if n_pos > 0 else 1.0
                w_neg = self.n / (2.0 * n_neg) if n_neg > 0 else 1.0
                print(f"  class_weights[{k}]: pos={n_pos} neg={n_neg} "
                      f"w_pos={w_pos:.3f} w_neg={w_neg:.3f}")

        # --- Store sparse structure for O(nnz*K) phi ---
        if sp.issparse(X):
            X_coo = X.tocoo()
        else:
            X_coo = sp.coo_matrix(X)
        row = X_coo.row.astype(np.int32)
        col = X_coo.col.astype(np.int32)
        data = X_coo.data.astype(np.float64)
        self._nnz = len(data)

        # Pre-sort by row for cache locality and segment_sum on GPU
        row_order = np.argsort(row, kind='mergesort')
        self._X_row = row[row_order]
        self._X_col = col[row_order]
        self._X_data = data[row_order]

        # Auto-tune chunk size based on K (target ~4GB work arrays)
        self._effective_chunk = _auto_chunk_size(self._nnz, K)
        n_chunks = (self._nnz + self._effective_chunk - 1) // self._effective_chunk
        print(f"  Chunk size: {self._effective_chunk:,} "
              f"({n_chunks} chunks for {self._nnz:,} nnz)")

        # Cache gammaln(data+1) -- constant, only needed for ELBO
        self._gammaln_data_cache = gammaln(self._X_data + 1)
        self._gammaln_data_sum = float(np.sum(self._gammaln_data_cache))

        # Initialize E_theta/E_beta caches
        self._E_theta_cache = None
        self._E_beta_cache = None

        # Cache digamma/gammaln for constant-shape parameters (a_xi, a_eta)
        self._digamma_a_xi = digamma(self.a_xi)
        self._gammaln_a_xi = gammaln(self.a_xi)
        self._digamma_a_eta = digamma(self.a_eta)
        self._gammaln_a_eta = gammaln(self.a_eta)

        # ── Transfer all numpy arrays to device (GPU if available) ──
        self._to_device()

        # Pre-compute digamma caches (now on device)
        self._refresh_log_caches()

        print(f"Initialized: n={self.n}, p={self.p}, K={K}, nnz={self._nnz}")
        print(f"  bp={bp:.4f}, dp={dp:.4f}")
        print(f"  E[theta] range: [{float(self.E_theta.min()):.4f}, {float(self.E_theta.max()):.4f}]")
        print(f"  E[beta] range: [{float(self.E_beta.min()):.4f}, {float(self.E_beta.max()):.4f}]")

    def _to_device(self):
        """Transfer all numpy parameter arrays to JAX device (no-op without JAX)."""
        if not USE_JAX:
            return
        for name in list(vars(self)):
            val = getattr(self, name)
            if isinstance(val, np.ndarray):
                setattr(self, name, to_device(val))

    def _init_beta_mask(self):
        """Build beta_mask array for pathway modes."""
        if self.mode == 'masked' and self.pathway_mask is not None:
            # pathway_mask: (n_pathways, n_genes) -> transpose to (n_genes, K)
            # pad or truncate to K columns
            pm = self.pathway_mask.T  # (p, n_pathways)
            if pm.shape[1] < self.K:
                pad = np.zeros((self.p, self.K - pm.shape[1]))
                self.beta_mask = np.hstack([pm, pad])
            else:
                self.beta_mask = pm[:, :self.K]
            # Zero out beta where mask is 0
            small_a = self.c * 0.01
            large_b = 100.0
            self.a_beta = np.where(self.beta_mask > 0.5, self.a_beta, small_a)
            self.b_beta = np.where(self.beta_mask > 0.5, self.b_beta, large_b)
        elif self.mode == 'combined' and self.pathway_mask is not None:
            pm = self.pathway_mask.T
            npath = self.n_pathway_factors
            self.beta_mask = np.ones((self.p, self.K))
            # First npath factors are pathway-constrained
            if pm.shape[1] >= npath:
                mask_part = pm[:, :npath]
            else:
                mask_part = np.hstack([pm, np.zeros((self.p, npath - pm.shape[1]))])
            self.beta_mask[:, :npath] = mask_part
            small_a = self.c * 0.01
            large_b = 100.0
            for k in range(npath):
                self.a_beta[:, k] = np.where(
                    self.beta_mask[:, k] > 0.5, self.a_beta[:, k], small_a)
                self.b_beta[:, k] = np.where(
                    self.beta_mask[:, k] > 0.5, self.b_beta[:, k], large_b)
        else:
            self.beta_mask = None

    def _enforce_beta_mask(self):
        """Re-enforce mask after beta update."""
        if self.beta_mask is None:
            return
        small_a = self.c * 0.01
        large_b = 100.0
        # beta_mask is 1.0 where factor is active, 0.0 where suppressed.
        # Works for both 'masked' (all columns) and 'combined' (pathway cols
        # have 0/1, free cols have 1.0).
        mask = self.beta_mask > 0.5
        self.a_beta = xp.where(mask, self.a_beta, small_a)
        self.b_beta = xp.where(mask, self.b_beta, large_b)

    # =================================================================
    # Expected values (properties for convenience)
    # =================================================================

    @property
    def E_theta(self):
        if self._E_theta_cache is None:
            self._E_theta_cache = self.a_theta / self.b_theta
        return self._E_theta_cache

    @property
    def E_log_theta(self):
        return digamma(self.a_theta) - xp.log(self.b_theta)

    @property
    def E_beta(self):
        if self._E_beta_cache is None:
            self._E_beta_cache = self.a_beta / self.b_beta
        return self._E_beta_cache

    @property
    def E_log_beta(self):
        return digamma(self.a_beta) - xp.log(self.b_beta)

    def _invalidate_theta_cache(self):
        """Invalidate E_theta cache after a_theta or b_theta changes."""
        self._E_theta_cache = None

    def _invalidate_beta_cache(self):
        """Invalidate E_beta cache after a_beta or b_beta changes."""
        self._E_beta_cache = None

    def _refresh_log_caches(self):
        """Recompute cached digamma arrays after theta/beta updates.

        Also caches raw digamma(a_theta) and digamma(a_beta) for reuse
        in ELBO entropy computation (avoids recomputing 30M+ digamma evals).
        """
        self._digamma_a_theta = digamma(self.a_theta)
        self._digamma_a_beta = digamma(self.a_beta)
        self._E_log_theta_cache = self._digamma_a_theta - xp.log(self.b_theta)
        self._E_log_beta_cache = self._digamma_a_beta - xp.log(self.b_beta)

    @property
    def E_xi(self):
        return self.a_xi / self.b_xi

    @property
    def E_eta(self):
        return self.a_eta / self.b_eta

    # =================================================================
    # phi computation (sparse, O(nnz*K))
    # =================================================================

    def _compute_phi_sparse(self, random_init=False):
        """
        Compute Xphi using only nonzero entries, processed in chunks to
        bound peak memory at O(chunk_size * K) instead of O(nnz * K).

        Returns:
            z_sum_beta: (p, K) = Sum_i x_{ij} phi_{ijk}
            z_sum_theta: (n, K) = Sum_j x_{ij} phi_{ijk}
        """
        K = self.K
        nnz = self._nnz
        chunk = self._effective_chunk
        z_sum_beta = xp.zeros((self.p, K))
        z_sum_theta = xp.zeros((self.n, K))

        for start in range(0, nnz, chunk):
            end = min(start + chunk, nnz)
            row_c = self._X_row[start:end]
            col_c = self._X_col[start:end]
            data_c = self._X_data[start:end]

            if random_init:
                # numpy random for Dirichlet, then transfer to device
                Xphi_np = np.random.dirichlet(np.ones(K), end - start)
                Xphi_np *= to_numpy(data_c)[:, None]
                Xphi = to_device(Xphi_np)
            else:
                Xphi = phi_chunk_core(
                    self._E_log_theta_cache[row_c],
                    self._E_log_beta_cache[col_c],
                    data_c,
                )

            # Accumulate via vectorized scatter-add
            # row indices are pre-sorted, enabling segment_sum on GPU
            z_sum_beta = scatter_add_to(z_sum_beta, col_c, Xphi)
            z_sum_theta = scatter_add_to(z_sum_theta, row_c, Xphi,
                                         sorted_indices=True)
            del Xphi

        return z_sum_beta, z_sum_theta

    # =================================================================
    # CAVI Updates
    # =================================================================

    def _update_beta(self, z_sum_beta):
        """beta shape and rate (scHPF Eq 8, gene side)."""
        # a^beta_{jk} = c + Sum_i x_{ij} phi_{ijk}
        self.a_beta = self.c + z_sum_beta
        # b^beta_{jk} = E[eta_j] + Sum_i E[theta_{ik}]
        theta_sum = self.E_theta.sum(axis=0)  # (K,)
        self.b_beta = self.E_eta[:, None] + theta_sum[None, :]
        self._enforce_beta_mask()
        self._invalidate_beta_cache()

    def _update_eta(self):
        """eta rate (scHPF): b^eta_j = d' + Sum_k E[beta_{jk}]."""
        # a^eta is constant = cp + K*c (set in init, never changes)
        self.b_eta = self.dp + self.E_beta.sum(axis=1)

    def _update_theta(self, z_sum_theta, y, X_aux):
        """
        theta shape and rate (scHPF Eq 7, cell side + JJ regression).

        Solves the quadratic for b_theta in one step:
            b^2 - b_base*b - c_quad*a_theta = 0
            b = (b_base + sqrt(b_base^2 + 4*c_quad*a_theta)) / 2
        which is always positive (given c_quad > 0, guaranteed by the zeta cap).
        """
        # a^theta_{ik} = a + Sum_j x_{ij} phi_{ijk}
        self.a_theta = self.a + z_sum_theta

        # b_base = E[xi_i] + Sum_j E[beta_{jk}] + regression_weight * R_linear
        beta_sum = self.E_beta.sum(axis=0)  # (K,)
        b_poisson = self.E_xi[:, None] + beta_sum[None, :]

        if self.regression_weight > 0:
            R_linear, R_quad_coeff = self._regression_rate_parts(y, X_aux)
            # Fuse: b_base, discriminant, b_theta
            b_base = b_poisson + self.regression_weight * R_linear
            c_quad = self.regression_weight * R_quad_coeff
            disc = xp.sqrt(xp.square(b_base) + 4.0 * c_quad * self.a_theta)
            self.b_theta = (b_base + disc) / 2.0
            self._theta_inner_iters = 1

            # Re-tighten zeta for updated theta
            self._invalidate_theta_cache()
            self._update_zeta(X_aux)
        else:
            self.b_theta = b_poisson
            self._theta_inner_iters = 0
        self._invalidate_theta_cache()

    def _update_xi(self):
        """xi rate (scHPF): b^xi_i = b' + Sum_k E[theta_{ik}]."""
        # a^xi is constant = ap + K*a (set in init)
        self.b_xi = self.bp + self.E_theta.sum(axis=1)

    def _update_zeta(self, X_aux):
        """JJ auxiliary: zeta_{ik} = min(sqrt(E[A^2_{ik}]), zeta_max).

        Capping zeta keeps lambda_JJ(zeta) bounded from below, which prevents the
        quadratic braking on theta from vanishing.  The JJ lower bound is valid
        for ANY zeta, so capping gives a slightly looser but still valid bound.
        Each CAVI update still provably increases this capped-zeta ELBO.
        """
        E_theta = self.E_theta
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K) -- tiny

        Var_theta = self.a_theta / xp.square(self.b_theta)   # (n, K)

        E_A = E_theta @ self.mu_v.T                          # (n, kappa) -- BLAS
        if self.p_aux > 0:
            E_A = E_A + X_aux @ self.mu_gamma.T

        E_A_sq = xp.square(E_A) + Var_theta @ E_v_sq.T
        self.zeta = xp.minimum(xp.sqrt(xp.maximum(E_A_sq, 1e-8)), self.zeta_max)

    def _regression_rate_parts(self, y, X_aux):
        """
        Compute the JJ regression correction to theta rate, split into linear
        and quadratic parts.

        The full correction is R = R_linear + R_quad_coeff * E[theta], where:
          R_linear_{il} = Sum_k [-(y_{ik} - 0.5) v_{kl}
                                + 2*lambda(zeta_{ik}) v_{kl} C^{(-l)}_{ik}]
          R_quad_coeff_{il} = Sum_k [2*lambda(zeta_{ik}) E[v^2_{kl}]]

        Memory-efficient: avoids (n, kappa, K) C_minus intermediate by
        algebraic decomposition into (n, kappa) @ (kappa, K) matmuls.

        Returns
        -------
        R_linear : (n, K) -- terms not depending on E[theta_{il}]
        R_quad_coeff : (n, K) -- coefficient of E[theta_{il}], always >= 0
        """
        y_exp = y if y.ndim > 1 else y[:, None]  # (n, kappa)
        lam = lambda_jj(self.zeta)                 # (n, kappa)
        W = self._sample_weights                    # (n, kappa)
        E_theta = self.E_theta                      # (n, K)
        E_v = self.mu_v                             # (kappa, K)
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag # (kappa, K)

        # Full linear predictor: (n, kappa) -- BLAS
        theta_v = E_theta @ E_v.T
        if self.p_aux > 0:
            theta_v = theta_v + X_aux @ self.mu_gamma.T

        W_lam = W * lam  # (n, kappa)

        # BLAS matmuls to avoid (n, kappa, K) C_minus intermediate:
        E_v_sq_col = xp.square(E_v)                             # (kappa, K) -- tiny
        R_linear = -(W * (y_exp - 0.5)) @ E_v                   # (n, K)
        R_linear = R_linear + 2.0 * ((W_lam * theta_v) @ E_v)   # fuse add
        R_linear = R_linear - 2.0 * E_theta * (W_lam @ E_v_sq_col)  # fuse subtract

        # R_quad via single BLAS matmul
        R_quad_coeff = (2.0 * W_lam) @ E_v_sq                   # (n, K)

        return R_linear, R_quad_coeff

    def _update_v(self, y, X_aux, iteration=0):
        """
        v posterior (diagonal Gaussian, JJ bound).

        For v_prior='normal':
            precision_{kl} = 1/sigma^2_eff + 2 Sum_i lambda(zeta_{ik}) E[theta^2_{il}]
            sigma^2_eff = sigma^2_v / K

        For v_prior='laplace' (Bayesian Lasso):
            precision_{kl} = E_q[1/s_{kl}] + 2 Sum_i lambda(zeta_{ik}) E[theta^2_{il}]
            where E_q[1/s_{kl}] is computed from the inverse Gaussian
            variational posterior on the scale mixture variable s_{kl}.

        Memory-efficient: avoids (n, kappa, K) C_minus intermediate by
        decomposing into (kappa, n) @ (n, K) matmuls -> (kappa, K).
        """
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = lambda_jj(self.zeta)
        W = self._sample_weights                    # (n, kappa)
        E_theta = self.E_theta
        E_v = self.mu_v

        # E_theta^2
        E_theta_plain_sq = xp.square(E_theta)        # (n, K)

        Var_theta = self.a_theta / xp.square(self.b_theta)
        E_theta_sq = E_theta_plain_sq + Var_theta    # (n, K)

        # Full predictor: (n, kappa) -- BLAS matmul
        theta_v = E_theta @ E_v.T
        if self.p_aux > 0:
            theta_v = theta_v + X_aux @ self.mu_gamma.T

        if self.v_prior == 'normal':
            sigma_v_eff_sq = (self.sigma_v ** 2) / self.K
            prior_precision = 1.0 / sigma_v_eff_sq
        else:
            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K)
            omega = xp.sqrt(xp.maximum(E_v_sq, 1e-12))
            prior_precision = 1.0 / (self.b_v * omega) + 1.0 / (omega ** 2)

        # Precision and mean via BLAS matmuls
        W_lam = W * lam                              # (n, kappa)
        precision = prior_precision + 2 * (W_lam.T @ E_theta_sq)  # (kappa, K)

        W_y = W * (y_exp - 0.5)                      # (n, kappa)
        term1 = W_y.T @ E_theta                      # (kappa, K)

        # Decompose term2 using BLAS matmuls:
        part_a = (W_lam * theta_v).T @ E_theta        # (kappa, K)
        part_b = E_v * (W_lam.T @ E_theta_plain_sq)   # (kappa, K)
        term2 = 2.0 * (part_a - part_b)

        mean_prec = term1 - term2

        sigma_v_diag_new = 1.0 / precision

        if self.v_prior == 'normal':
            # Clip v per-element
            v_clip = min(5.0, 10.0 / np.sqrt(self.K))
            mu_v_new = xp.clip(mean_prec / precision, -v_clip, v_clip)

            # Adaptive damping
            alpha_max = 0.15
            alpha = min(alpha_max, 0.05 + (alpha_max - 0.05) * (iteration / max(200, iteration)))
            mu_v_candidate = (1.0 - alpha) * self.mu_v + alpha * mu_v_new

            # --- Period-2 oscillation detection and correction ---
            if self._v_prev_mu is None:
                self._v_prev_mu = xp.zeros_like(self.mu_v)
                self._v_raw_prev = xp.zeros_like(self.mu_v)

            sign_flip_now = (self.mu_v * mu_v_candidate) < 0
            sign_flip_prev = (self._v_prev_mu * self.mu_v) < 0
            oscillating = sign_flip_now & sign_flip_prev

            avg_raw = 0.5 * (mu_v_new + self._v_raw_prev)
            corrected = (1.0 - alpha) * self.mu_v + alpha * avg_raw
            mu_v_candidate = xp.where(oscillating, corrected, mu_v_candidate)

            self._v_prev_mu = xp.array(self.mu_v)
            self._v_raw_prev = xp.array(mu_v_new)
            self.mu_v = mu_v_candidate
            # Damp sigma_v_diag consistently with mu_v to keep q(v) coherent
            self.sigma_v_diag = (1.0 - alpha) * self.sigma_v_diag + alpha * sigma_v_diag_new
        else:
            # Laplace: no clip needed -- adaptive E[1/s] acts as automatic shrinkage
            mu_v_new = mean_prec / precision
            self.mu_v = mu_v_new
            self.sigma_v_diag = sigma_v_diag_new

    def _update_gamma(self, y, X_aux, iteration=0):
        """gamma posterior (Gaussian, JJ bound)."""
        if self.p_aux == 0:
            return
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = lambda_jj(self.zeta)
        W = self._sample_weights                    # (n, kappa)
        E_theta = self.E_theta

        # Same adaptive damping schedule as v
        alpha_max = 0.15
        alpha = min(alpha_max, 0.05 + (alpha_max - 0.05) * (iteration / max(200, iteration)))

        for k in range(self.kappa):
            prec_prior = xp.eye(self.p_aux) / (self.sigma_gamma ** 2)
            # Precision: 1/sigma^2 I + 2 Sum_i W_{ik} lambda(zeta_{ik}) x^aux_i x^aux_i^T
            W_lam_k = W[:, k] * lam[:, k]  # (n,) -- class-weighted lambda
            weighted_X = X_aux * (2 * W_lam_k)[:, None]
            prec_lik = weighted_X.T @ X_aux
            prec = prec_prior + prec_lik

            # Residual: W*(y - 0.5) - 2*W*lambda(zeta) theta*v
            theta_v_k = E_theta @ self.mu_v[k]
            residual = W[:, k] * (y_exp[:, k] - 0.5) - 2 * W_lam_k * theta_v_k
            mean_prec = X_aux.T @ residual

            if USE_JAX:
                self.Sigma_gamma = self.Sigma_gamma.at[k].set(xp.linalg.inv(prec))
            else:
                self.Sigma_gamma[k] = np.linalg.inv(prec)
            mu_gamma_new = self.Sigma_gamma[k] @ mean_prec
            new_mu_k = (1.0 - alpha) * self.mu_gamma[k] + alpha * mu_gamma_new
            if USE_JAX:
                self.mu_gamma = self.mu_gamma.at[k].set(new_mu_k)
            else:
                self.mu_gamma[k] = new_mu_k

    # =================================================================
    # ELBO
    # =================================================================

    def _compute_elbo(self, X_dense, y, X_aux):
        """Compute ELBO = E[log p] - E[log q]."""
        E_theta = self.E_theta
        E_log_theta = self._E_log_theta_cache
        E_beta = self.E_beta
        E_log_beta = self._E_log_beta_cache
        E_xi = self.E_xi
        E_eta = self.E_eta
        # Reuse cached digamma for constant-shape a_xi, a_eta
        E_log_xi = self._digamma_a_xi - xp.log(self.b_xi)
        E_log_eta = self._digamma_a_eta - xp.log(self.b_eta)

        elbo = 0.0

        # === Poisson likelihood (collapsed z) -- chunked to avoid (nnz, K) ===
        poisson_ll = 0.0
        nnz = self._nnz
        chunk = self._effective_chunk
        for start in range(0, nnz, chunk):
            end = min(start + chunk, nnz)
            row_c = self._X_row[start:end]
            col_c = self._X_col[start:end]
            data_c = self._X_data[start:end]
            log_rates_c = E_log_theta[row_c] + E_log_beta[col_c]  # (chunk, K)
            log_sum_c = logsumexp_rows(log_rates_c).ravel()
            poisson_ll += xp.dot(data_c, log_sum_c)
            del log_rates_c
        poisson_ll -= xp.sum(E_theta.sum(axis=0) * E_beta.sum(axis=0))
        poisson_ll -= self._gammaln_data_sum  # cached at init
        elbo += poisson_ll

        # === JJ Bernoulli likelihood ===
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = lambda_jj(self.zeta)
        E_A = E_theta @ self.mu_v.T                   # BLAS
        if self.p_aux > 0:
            E_A = E_A + X_aux @ self.mu_gamma.T
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag   # tiny (kappa, K)
        Var_theta = self.a_theta / xp.square(self.b_theta)
        E_A_sq = E_A ** 2 + Var_theta @ E_v_sq.T

        W = self._sample_weights  # (n, kappa)
        regression_ll = xp.sum(W * ((y_exp - 0.5) * E_A - lam * E_A_sq))
        regression_ll += xp.sum(W * (lam * self.zeta ** 2 - 0.5 * self.zeta
                                + log_expit(self.zeta)))
        elbo += self.regression_weight * regression_ll

        # === Prior: p(theta|xi) ===
        elbo += xp.sum((self.a - 1) * E_log_theta
                       + self.a * E_log_xi[:, None]
                       - E_xi[:, None] * E_theta)
        elbo -= self.n * self.K * gammaln(self.a)

        # === Prior: p(beta|eta) ===
        elbo += xp.sum((self.c - 1) * E_log_beta
                       + self.c * E_log_eta[:, None]
                       - E_eta[:, None] * E_beta)
        elbo -= self.p * self.K * gammaln(self.c)

        # === Prior: p(xi) ===
        elbo += xp.sum((self.ap - 1) * E_log_xi
                       + self.ap * xp.log(self.bp) - self.bp * E_xi)
        elbo -= self.n * gammaln(self.ap)

        # === Prior: p(eta) ===
        elbo += xp.sum((self.cp - 1) * E_log_eta
                       + self.cp * xp.log(self.dp) - self.dp * E_eta)
        elbo -= self.p * gammaln(self.cp)

        # === Prior: p(v) ===
        if self.v_prior == 'normal':
            sigma_v_eff_sq = (self.sigma_v ** 2) / self.K
            elbo -= 0.5 * xp.sum(self.mu_v ** 2 + self.sigma_v_diag) / sigma_v_eff_sq
            elbo -= 0.5 * self.kappa * self.K * xp.log(2 * xp.pi * sigma_v_eff_sq)
        else:
            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K)
            omega = xp.sqrt(xp.maximum(E_v_sq, 1e-12))
            E_inv_s = 1.0 / (self.b_v * omega) + 1.0 / (omega ** 2)

            mu_s = self.b_v / omega
            E_log_s = xp.log(xp.maximum(mu_s, 1e-12))
            elbo += xp.sum(-0.5 * xp.log(2 * xp.pi) - 0.5 * E_log_s
                           - 0.5 * E_v_sq * E_inv_s)

            rate_s = 1.0 / (2.0 * self.b_v ** 2)
            elbo += xp.sum(xp.log(rate_s) - mu_s * rate_s)

            lambda_s = 1.0 / (self.b_v ** 2)
            elbo += 0.5 * xp.sum(xp.log(2 * xp.pi * xp.e * mu_s ** 3
                                        / xp.maximum(lambda_s, 1e-12)))

        # === Entropy: -E[log q] ===
        # q(theta) Gamma entropy -- reuse cached digamma(a_theta) from _refresh_log_caches
        psi_a_theta = self._digamma_a_theta
        elbo += xp.sum(self.a_theta - xp.log(self.b_theta)
                       + gammaln(self.a_theta)
                       + (1 - self.a_theta) * psi_a_theta)
        # q(beta)
        psi_a_beta = self._digamma_a_beta
        elbo += xp.sum(self.a_beta - xp.log(self.b_beta)
                       + gammaln(self.a_beta)
                       + (1 - self.a_beta) * psi_a_beta)
        # q(xi)
        elbo += xp.sum(self.a_xi - xp.log(self.b_xi)
                       + self._gammaln_a_xi
                       + (1 - self.a_xi) * self._digamma_a_xi)
        # q(eta)
        elbo += xp.sum(self.a_eta - xp.log(self.b_eta)
                       + self._gammaln_a_eta
                       + (1 - self.a_eta) * self._digamma_a_eta)
        # q(v) Gaussian entropy
        elbo += 0.5 * xp.sum(xp.log(2 * xp.pi * xp.e * self.sigma_v_diag))

        return float(elbo), float(poisson_ll), float(regression_ll)

    # =================================================================
    # Held-out log-likelihood (scHPF pattern: mean negative Poisson LL)
    # =================================================================

    def compute_heldout_ll(self, X_val, y_val=None, X_aux_val=None, n_iter=20):
        """
        Held-out log-likelihood on validation data.

        Returns
        -------
        total_ll : float
            Mean total LL per sample (Poisson + weighted regression if labels provided).
        poisson_ll : float
            Mean Poisson LL per sample.
        regression_ll : float or None
            Mean regression LL per sample (before weighting), or None if no labels.
        """
        if sp.issparse(X_val):
            X_val_coo = X_val.tocoo()
        else:
            X_val_coo = sp.coo_matrix(X_val)

        n_val = X_val.shape[0]

        # Infer theta for validation cells (freeze globals)
        a_theta_v = to_device(np.random.uniform(0.5 * self.a, 1.5 * self.a,
                                                 (n_val, self.K)))
        b_theta_v = to_device(np.full((n_val, self.K), self.bp))
        a_xi_v = to_device(np.full(n_val, self.ap + self.K * self.a))
        b_xi_v = to_device(np.full(n_val, self.bp))

        E_log_beta = self._E_log_beta_cache
        E_beta = self.E_beta
        beta_col_sums = E_beta.sum(axis=0)

        row = to_device(X_val_coo.row.astype(np.int32))
        col = to_device(X_val_coo.col.astype(np.int32))
        data_np = X_val_coo.data if X_val_coo.data.dtype == np.float64 else X_val_coo.data.astype(np.float64)
        data = to_device(data_np)
        nnz_val = len(data_np)
        chunk = _auto_chunk_size(nnz_val, self.K)

        for _ in range(n_iter):
            E_log_theta_v = digamma(a_theta_v) - xp.log(b_theta_v)
            E_theta_v = a_theta_v / b_theta_v
            E_xi_v = a_xi_v / b_xi_v

            # phi sparse -- chunked
            z_sum = xp.zeros((n_val, self.K))
            for start in range(0, nnz_val, chunk):
                end = min(start + chunk, nnz_val)
                row_c = row[start:end]
                col_c = col[start:end]
                data_c = data[start:end]
                Xphi = phi_chunk_core(E_log_theta_v[row_c], E_log_beta[col_c], data_c)
                z_sum = scatter_add_to(z_sum, row_c, Xphi)
                del Xphi

            a_theta_v = self.a + z_sum
            b_theta_v = E_xi_v[:, None] + beta_col_sums[None, :]
            b_xi_v = self.bp + E_theta_v.sum(axis=1)

        # Poisson LL per sample -- chunked
        E_log_theta_v = digamma(a_theta_v) - xp.log(b_theta_v)
        E_theta_v = a_theta_v / b_theta_v
        poisson_ll = 0.0
        gammaln_term = float(xp.sum(gammaln(data + 1)))
        for start in range(0, nnz_val, chunk):
            end = min(start + chunk, nnz_val)
            row_c = row[start:end]
            col_c = col[start:end]
            data_c = data[start:end]
            log_rates_c = E_log_theta_v[row_c] + E_log_beta[col_c]
            log_sum_c = logsumexp_rows(log_rates_c).ravel()
            poisson_ll += xp.dot(data_c, log_sum_c)
            del log_rates_c

        poisson_ll -= xp.sum(E_theta_v.sum(axis=0) * E_beta.sum(axis=0))
        poisson_ll -= gammaln_term
        poisson_ll_per_sample = float(poisson_ll) / n_val

        # Regression LL on validation data (if labels provided)
        regression_ll_per_sample = None
        if y_val is not None:
            y_v = to_device(np.asarray(y_val, dtype=np.float64))
            if y_v.ndim == 1:
                y_v = y_v[:, None]
            if X_aux_val is None:
                X_aux_v = to_device(np.zeros((n_val, self.p_aux if self.p_aux > 0 else 0)))
            else:
                X_aux_v = to_device(np.asarray(X_aux_val, dtype=np.float64))

            E_A = E_theta_v @ self.mu_v.T
            if self.p_aux > 0:
                E_A = E_A + X_aux_v @ self.mu_gamma.T

            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag
            Var_theta_v = a_theta_v / (b_theta_v ** 2)
            E_A_sq = E_A ** 2 + Var_theta_v @ E_v_sq.T

            zeta_val = xp.sqrt(xp.maximum(E_A_sq, 1e-8))
            lam = lambda_jj(zeta_val)

            reg_ll = xp.sum((y_v - 0.5) * E_A - lam * E_A_sq)
            reg_ll += xp.sum(lam * zeta_val ** 2 - 0.5 * zeta_val
                             + log_expit(zeta_val))
            regression_ll_per_sample = float(reg_ll / n_val)

        total_ll = poisson_ll_per_sample
        if regression_ll_per_sample is not None:
            total_ll += self.regression_weight * regression_ll_per_sample

        return float(total_ll), float(poisson_ll_per_sample), regression_ll_per_sample

    # =================================================================
    # fit()
    # =================================================================

    def fit(self, X_train, y_train, X_aux_train=None,
            X_val=None, y_val=None, X_aux_val=None,
            max_iter=600, check_freq=5, tol=0.001,
            v_warmup=50, verbose=True,
            early_stopping='heldout_ll'):
        """
        Fit the supervised Poisson factorization model.

        Parameters
        ----------
        X_train : sparse or dense (n, p)
        y_train : (n,) or (n, kappa)
        X_aux_train : (n, p_aux) or None
        X_val, y_val, X_aux_val : validation data (optional)
        max_iter : int
        check_freq : int
        tol : float -- convergence if |pct_change| < tol twice in a row
        v_warmup : int -- deprecated, kept for CLI compatibility (ignored)
        verbose : bool
        early_stopping : str
            'heldout_ll' -- stop on held-out LL / regression LL plateau (default).
            'elbo' -- stop only on ELBO convergence.
            'none' -- disable all early stopping, run all iterations.
        """
        t0 = time.time()

        # Print backend info
        info = backend_info()
        print(f"Backend: {info['device']}"
              + (f"  (JAX {info.get('jax_version', '')})" if info['backend'] == 'jax' else ""))

        if X_aux_train is None:
            X_aux_train = np.zeros((X_train.shape[0], 0))
        y = np.asarray(y_train, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        X_aux = np.asarray(X_aux_train, dtype=np.float64)

        # Initialize (creates numpy arrays, then transfers to device)
        self._initialize(X_train, y, X_aux)

        # Transfer training labels / aux to device for hot-path computations
        y = to_device(y)
        X_aux = to_device(X_aux)

        # Dense X for ELBO (if small enough) -- otherwise use sparse
        if sp.issparse(X_train):
            X_dense = None  # Will use sparse ELBO
        else:
            X_dense = X_train

        # Validation setup
        if X_val is not None and X_aux_val is None:
            X_aux_val = np.zeros((X_val.shape[0], 0))

        self.elbo_history_ = []
        self.holl_history_ = []
        loss_list = []
        pct_changes = []

        best_holl = -np.inf
        best_params = None

        # Regression early stopping: stop if Reg degrades for too long
        best_reg_ll = -np.inf
        best_reg_params = None
        best_reg_iter = 0
        reg_patience = 10  # stop after this many consecutive checks of Reg degradation

        # HO-LL regression early stopping
        best_holl_reg = -np.inf
        best_holl_reg_iter = 0
        best_holl_reg_params = None
        holl_reg_patience = 10

        for t in range(max_iter):
            diag = verbose and (t % check_freq == 0)
            t_iter_start = time.time()

            # 1. Compute phi and z_sums (sparse)
            random_phi = (t == 0)  # scHPF: random Dirichlet on first iter
            z_sum_beta, z_sum_theta = self._compute_phi_sparse(random_init=random_phi)
            if diag:
                print(f"  [timing t={t}] phi: {time.time() - t_iter_start:.1f}s")

            # 2. Update beta, eta (gene side first -- scHPF order)
            self._update_beta(z_sum_beta)
            self._update_eta()
            if diag:
                print(f"  [diag t={t}] after beta,eta: "
                      f"E[beta]=[{float(self.E_beta.min()):.4e},{float(self.E_beta.max()):.4e}] "
                      f"E[eta]=[{float(self.E_eta.min()):.4e},{float(self.E_eta.max()):.4e}]")

            # 3. Update zeta (JJ bound tightening)
            self._update_zeta(X_aux)
            if diag:
                lam_diag = lambda_jj(self.zeta)
                print(f"  [diag t={t}] after zeta:   "
                      f"zeta=[{float(self.zeta.min()):.4e},{float(self.zeta.max()):.4e}] "
                      f"lambda(zeta)=[{float(lam_diag.min()):.4e},"
                      f"{float(lam_diag.max()):.4e}]")

            # 4. Update theta, xi (cell side -- with full regression from start)
            self._update_theta(z_sum_theta, y, X_aux)
            self._update_xi()
            if diag:
                E_th = self.E_theta
                inner_info = f" inner={self._theta_inner_iters}" if self._theta_inner_iters > 0 else ""
                print(f"  [diag t={t}] after theta,xi: "
                      f"E[theta]=[{float(E_th.min()):.4e},{float(E_th.max()):.4e}] mean={float(E_th.mean()):.4e} "
                      f"b_theta=[{float(self.b_theta.min()):.4e},{float(self.b_theta.max()):.4e}] "
                      f"a_theta=[{float(self.a_theta.min()):.4e},{float(self.a_theta.max()):.4e}]{inner_info}")
                if self._theta_inner_iters > 0:
                    lam_diag2 = lambda_jj(self.zeta)
                    print(f"  [diag t={t}] post-theta zeta: "
                          f"zeta=[{float(self.zeta.min()):.4e},{float(self.zeta.max()):.4e}] "
                          f"lambda(zeta)=[{float(lam_diag2.min()):.4e},"
                          f"{float(lam_diag2.max()):.4e}]")

            # Refresh digamma caches after theta/beta changed
            self._refresh_log_caches()

            # 5. Update v, gamma
            self._update_v(y, X_aux, iteration=t)
            self._update_gamma(y, X_aux, iteration=t)

            # 5b. Re-tighten zeta after v/gamma changed
            self._update_zeta(X_aux)

            if diag:
                t_updates = time.time() - t_iter_start
                if self.p_aux > 0:
                    print(f"  [diag t={t}] after v,gamma: "
                          f"mu_v=[{float(self.mu_v.min()):.4e},{float(self.mu_v.max()):.4e}] "
                          f"sigma2_v=[{float(self.sigma_v_diag.min()):.4e},{float(self.sigma_v_diag.max()):.4e}] "
                          f"mu_gamma=[{float(self.mu_gamma.min()):.4e},{float(self.mu_gamma.max()):.4e}]")
                else:
                    print(f"  [diag t={t}] after v:   "
                          f"mu_v=[{float(self.mu_v.min()):.4e},{float(self.mu_v.max()):.4e}] "
                          f"sigma2_v=[{float(self.sigma_v_diag.min()):.4e},{float(self.sigma_v_diag.max()):.4e}]")
                print(f"  [timing t={t}] updates: {t_updates:.1f}s")

            # 6. Check convergence
            if t % check_freq == 0:
                elbo, pois_ll, reg_ll = self._compute_elbo(X_dense, y, X_aux)
                self.elbo_history_.append((t, elbo))

                # ELBO monotonicity check
                if diag and len(self.elbo_history_) >= 2:
                    prev_elbo = self.elbo_history_[-2][1]
                    delta = elbo - prev_elbo
                    if delta < -1.0:  # allow tiny numerical noise
                        print(f"  [WARN t={t}] ELBO DECREASED by {delta:.4e} "
                              f"({prev_elbo:.4e} -> {elbo:.4e})")

                # Held-out LL (with breakdown)
                holl = None
                holl_pois = None
                holl_reg = None
                if X_val is not None:
                    holl, holl_pois, holl_reg = self.compute_heldout_ll(
                        X_val, y_val=y_val, X_aux_val=X_aux_val)
                    self.holl_history_.append((t, holl, holl_pois,
                                              holl_reg if holl_reg is not None else 0.0))
                    if holl > best_holl:
                        best_holl = holl
                        best_params = self._checkpoint()

                    # Track best HO-LL regression for early stopping
                    if holl_reg is not None and holl_reg > best_holl_reg:
                        best_holl_reg = holl_reg
                        best_holl_reg_iter = t
                        best_holl_reg_params = self._checkpoint()

                # Track best training regression LL for early stopping
                if reg_ll > best_reg_ll:
                    best_reg_ll = reg_ll
                    best_reg_params = self._checkpoint()
                    best_reg_iter = t

                # Loss = mean negative ELBO (tracks full objective including regression)
                curr_loss = -elbo / self.n
                loss_list.append(curr_loss)

                if len(loss_list) >= 2:
                    prev = loss_list[-2]
                    pct = 100 * (curr_loss - prev) / max(abs(prev), 1e-10)
                    pct_changes.append(pct)
                else:
                    pct_changes.append(100.0)

                if verbose:
                    if holl is not None:
                        holl_parts = [f"  HO-LL={holl:.2f}"]
                        holl_parts.append(f"  HO-Pois={holl_pois:.2f}")
                        if holl_reg is not None:
                            holl_parts.append(f"  HO-Reg={holl_reg:.4e}")
                        holl_str = "".join(holl_parts)
                    else:
                        holl_str = ""
                    t_total = time.time() - t_iter_start
                    mu_v_preview = to_numpy(self.mu_v).ravel()[:3]
                    print(f"Iter {t:4d}: ELBO={elbo:.4e}  "
                          f"Pois={pois_ll:.4e}  Reg={reg_ll:.4e}  "
                          f"v={mu_v_preview}{holl_str}  "
                          f"[{t_total:.1f}s]")

                # Early stopping checks (gated by early_stopping mode)
                if early_stopping == 'heldout_ll':
                    # HO-LL regression early stopping (preferred when validation available)
                    if X_val is not None and holl_reg is not None:
                        iters_since_best = t - best_holl_reg_iter
                        if iters_since_best >= holl_reg_patience and t >= 30:
                            if verbose:
                                print(f"HO-LL Reg early stop at iter {t}: "
                                      f"HO-Reg hasn't improved in {iters_since_best} iters "
                                      f"(best HO-Reg={best_holl_reg:.4e} at iter {best_holl_reg_iter})")
                            break

                    # Regression early stopping (training)
                    iters_since_best = t - best_reg_iter
                    if iters_since_best >= reg_patience and t >= 30:
                        if verbose:
                            print(f"Regression early stop at iter {t}: "
                                  f"Reg hasn't improved in {iters_since_best} iters "
                                  f"(best Reg={best_reg_ll:.4e} at iter {best_reg_iter})")
                        break

                if early_stopping in ('heldout_ll', 'elbo'):
                    # Convergence check on full ELBO (not just Poisson term)
                    if len(loss_list) >= 3 and t >= 30:
                        c1 = abs(pct_changes[-1]) < tol
                        c2 = abs(pct_changes[-2]) < tol
                        if c1 and c2:
                            if verbose:
                                print(f"Converged at iter {t}")
                            break

        elapsed = time.time() - t0
        if verbose:
            print(f"\nTraining complete in {elapsed:.1f}s")
            if best_params is not None:
                print(f"Best HO-LL: {best_holl:.4f}")

        # Restore best checkpoint (skip when early stopping is disabled)
        if early_stopping != 'none':
            if best_holl_reg_params is not None:
                if verbose:
                    print(f"Restoring best HO-LL regression checkpoint (iter {best_holl_reg_iter}, "
                          f"HO-Reg={best_holl_reg:.4e})")
                self._restore(best_holl_reg_params)
            elif best_params is not None:
                if verbose:
                    print(f"Restoring best HO-LL checkpoint (HO-LL={best_holl:.4f})")
                self._restore(best_params)
            elif best_reg_params is not None:
                if verbose:
                    print(f"Restoring best regression checkpoint (iter {best_reg_iter})")
                self._restore(best_reg_params)
        elif verbose:
            print("Early stopping disabled -- keeping final parameters.")

        return self

    # =================================================================
    # Checkpoint/restore
    # =================================================================

    def _checkpoint(self):
        return {
            'a_beta': to_numpy(self.a_beta).copy(),
            'b_beta': to_numpy(self.b_beta).copy(),
            'a_eta': to_numpy(self.a_eta).copy(),
            'b_eta': to_numpy(self.b_eta).copy(),
            'a_theta': to_numpy(self.a_theta).copy(),
            'b_theta': to_numpy(self.b_theta).copy(),
            'a_xi': to_numpy(self.a_xi).copy(),
            'b_xi': to_numpy(self.b_xi).copy(),
            'mu_v': to_numpy(self.mu_v).copy(),
            'sigma_v_diag': to_numpy(self.sigma_v_diag).copy(),
            'mu_gamma': to_numpy(self.mu_gamma).copy(),
            'Sigma_gamma': to_numpy(self.Sigma_gamma).copy(),
            'zeta': to_numpy(self.zeta).copy(),
        }

    def _restore(self, cp):
        for k, v in cp.items():
            setattr(self, k, to_device(v))
        self._invalidate_theta_cache()
        self._invalidate_beta_cache()
        self._refresh_log_caches()

    # =================================================================
    # predict / transform (API compat)
    # =================================================================

    def _infer_theta_sparse(self, X_coo, n_new, n_iter=20):
        """Infer theta for new data using chunked sparse phi. Returns a_theta, b_theta."""
        a_theta = to_device(np.random.uniform(0.5 * self.a, 1.5 * self.a, (n_new, self.K)))
        b_theta = to_device(np.full((n_new, self.K), self.bp))
        a_xi = to_device(np.full(n_new, self.ap + self.K * self.a))
        b_xi = to_device(np.full(n_new, self.bp))

        E_log_beta = self.E_log_beta
        beta_col_sums = self.E_beta.sum(axis=0)

        row = to_device(X_coo.row.astype(np.int32))
        col = to_device(X_coo.col.astype(np.int32))
        data_np = X_coo.data if X_coo.data.dtype == np.float64 else X_coo.data.astype(np.float64)
        data = to_device(data_np)
        nnz = len(data_np)
        chunk = _auto_chunk_size(nnz, self.K)
        K = self.K

        for _ in range(n_iter):
            E_log_theta = digamma(a_theta) - xp.log(b_theta)
            E_theta = a_theta / b_theta
            E_xi = a_xi / b_xi

            # Chunked phi + scatter
            z_sum = xp.zeros((n_new, K))
            for start in range(0, nnz, chunk):
                end = min(start + chunk, nnz)
                row_c = row[start:end]
                col_c = col[start:end]
                data_c = data[start:end]
                Xphi = phi_chunk_core(E_log_theta[row_c], E_log_beta[col_c], data_c)
                z_sum = scatter_add_to(z_sum, row_c, Xphi)
                del Xphi

            a_theta = self.a + z_sum
            b_theta = E_xi[:, None] + beta_col_sums[None, :]
            b_xi = self.bp + E_theta.sum(axis=1)

        return a_theta, b_theta

    def predict_proba(self, X_new, X_aux_new=None, n_iter=20):
        """Predict P(y=1 | X_new)."""
        if sp.issparse(X_new):
            X_coo = X_new.tocoo()
        else:
            X_coo = sp.coo_matrix(X_new)

        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = to_device(np.zeros((n_new, self.p_aux if self.p_aux > 0 else 0)))
        else:
            X_aux_new = to_device(np.asarray(X_aux_new, dtype=np.float64))

        a_theta, b_theta = self._infer_theta_sparse(X_coo, n_new, n_iter)

        E_theta = a_theta / b_theta
        logits = E_theta @ self.mu_v.T
        if self.p_aux > 0:
            logits = logits + X_aux_new @ self.mu_gamma.T

        return to_numpy(_expit(logits)).squeeze()

    def transform(self, X_new, y_new=None, X_aux_new=None, n_iter=20, **kwargs):
        """Infer theta for new data. Returns dict with E_theta, a_theta, b_theta."""
        if sp.issparse(X_new):
            X_coo = X_new.tocoo()
        else:
            X_coo = sp.coo_matrix(X_new)

        n_new = X_new.shape[0]
        a_theta, b_theta = self._infer_theta_sparse(X_coo, n_new, n_iter)

        return {
            'E_theta': to_numpy(a_theta / b_theta),
            'a_theta': to_numpy(a_theta),
            'b_theta': to_numpy(b_theta),
        }
