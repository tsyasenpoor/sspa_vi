"""
Coordinate Ascent VI for Supervised Poisson Factorization (DRGP)
================================================================

Poisson factorization core follows scHPF (Levitin et al., MSB 2019)
exactly. Supervision via Jaakkola-Jordan logistic regression bound
from the DRGP manuscript appendix.

scHPF update equations (Gopalan et al. 2014, Eqs 7-9):
  φ_{ij} ∝ exp{ ψ(a^θ_{ik}) - log(b^θ_{ik}) + ψ(a^β_{jk}) - log(b^β_{jk}) }
  a^β_{jk} = c + Σ_i x_{ij} φ_{ijk}
  b^β_{jk} = E[η_j] + Σ_i E[θ_{ik}]
  a^θ_{ik} = a + Σ_j x_{ij} φ_{ijk}
  b^θ_{ik} = E[ξ_i] + Σ_j E[β_{jk}]  (+ JJ regression correction)

Hierarchical priors:
  ξ_i ~ Gamma(a', b')     →  a^ξ_i = a' + K*a  (constant)
                              b^ξ_i = b' + Σ_k E[θ_{ik}]
  η_j ~ Gamma(c', d')     →  a^η_j = c' + K*c  (constant)
                              b^η_j = d' + Σ_k E[β_{jk}]

Key differences from previous vi_cavi.py:
1. bp, dp are SCALARS (scHPF), not per-cell/gene vectors
2. Random init: U(0.5*prior, 1.5*prior) for both shape AND rate
3. First iteration uses random Dirichlet φ (scHPF symmetry breaking)
4. No damping — pure coordinate ascent
5. No diversity noise, no annealing, no warmup phases
6. All parameters (θ, v, γ, ζ) learned jointly from iteration 0
7. No spike-and-slab

References:
- Gopalan, Hofman, Blei (2014) "Scalable Recommendation with HPF"
- Levitin et al. (2019) "scHPF", Molecular Systems Biology
- Jaakkola & Jordan (2000) variational logistic bound
- Hoffman et al. (2013) "Stochastic Variational Inference"
"""

import numpy as np
import scipy.sparse as sp
from scipy.special import digamma, gammaln
from typing import Tuple, Optional, Dict, Any, List
import time


def _logsumexp_rows(a):
    """Fast row-wise logsumexp, avoids scipy dispatch overhead."""
    a_max = a.max(axis=1, keepdims=True)
    return a_max + np.log(np.exp(a - a_max).sum(axis=1, keepdims=True))


def _scatter_add(indices, values, n_out):
    """
    Scatter-add: out[i] += values[idx == i] for each column.

    ~10-20× faster than np.add.at by using np.bincount which is
    implemented as a single C loop with no Python overhead.

    Parameters
    ----------
    indices : (nnz,) int array — row or col indices
    values : (nnz, K) float array
    n_out : int — number of output rows

    Returns
    -------
    out : (n_out, K) float array
    """
    K = values.shape[1]
    out = np.empty((n_out, K))
    for k in range(K):
        out[:, k] = np.bincount(indices, weights=values[:, k], minlength=n_out)
    return out


class CAVI:
    """
    CAVI for Supervised Poisson Factorization.

    Parameters
    ----------
    n_factors : int
        Number of latent factors K.
    a : float
        Gamma shape prior for θ (cell loadings). Default 0.3 (scHPF).
    ap : float
        Gamma shape prior for ξ (cell capacity). Default 1.0.
    c : float
        Gamma shape prior for β (gene loadings). Default 0.3 (scHPF).
    cp : float
        Gamma shape prior for η (gene capacity). Default 1.0.
    sigma_v : float
        Gaussian prior std for v (regression weights). Used when v_prior='normal'.
    b_v : float
        Laplace prior scale for v (regression weights). Used when v_prior='laplace'.
        Smaller b_v = stronger sparsity. Var[v] = 2*b_v^2.
    v_prior : str
        Prior distribution for v: 'normal' (Gaussian) or 'laplace' (Bayesian Lasso).
    sigma_gamma : float
        Gaussian prior std for γ (auxiliary covariate weights).
    regression_weight : float
        Scalar weight for the classification term.
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
        random_state: Optional[int] = None,
        mode: str = 'unmasked',
        pathway_mask: Optional[np.ndarray] = None,
        pathway_names: Optional[List[str]] = None,
        n_pathway_factors: Optional[int] = None,
        # Ignored SVI-compat kwargs
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
        # Cap on JJ auxiliary ζ.  Keeps λ_JJ(ζ) ≥ λ_JJ(ζ_max) > 0,
        # preventing the quadratic braking on θ from vanishing.
        # At ζ_max=4: λ_min ≈ 0.060.  The JJ bound remains valid for any ζ.
        self.zeta_max = 4.0

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
    # JJ bound helper
    # =================================================================

    @staticmethod
    def _lambda_jj(zeta):
        """λ(ζ) = tanh(ζ/2) / (4ζ), with λ(0)=1/8."""
        safe = np.maximum(np.abs(zeta), 1e-8)
        return np.tanh(safe / 2.0) / (4.0 * safe)

    # =================================================================
    # Initialization (scHPF pattern)
    # =================================================================

    def _initialize(self, X, y, X_aux):
        """
        scHPF initialization:
        1. Empirical bp, dp (scalars)
        2. Random Gamma params: U(0.5*prior, 1.5*prior)
        3. xi.shape, eta.shape set to constants
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
        # Floor bp and dp to prevent extreme rate parameters
        self.bp = max(self.bp, 0.01)
        self.dp = max(self.dp, 0.01)
        # Clip if bp >> dp (scHPF)
        if self.bp > 1000 * self.dp:
            old_dp = self.dp
            self.dp = self.bp / 1000
            print(f"Clipping dp: was {old_dp:.4f} now {self.dp:.4f}")

        bp, dp = self.bp, self.dp

        # --- ξ: Gamma(ap + K*a, b_xi) ---
        # shape is CONSTANT = ap + K*a
        self.a_xi = np.full(self.n, self.ap + K * self.a)
        self.b_xi = np.random.uniform(0.5 * bp, 1.5 * bp, self.n)

        # --- θ: Gamma(a_theta, b_theta) ---
        self.a_theta = np.random.uniform(0.5 * self.a, 1.5 * self.a,
                                         (self.n, K))
        self.b_theta = np.random.uniform(0.5 * bp, 1.5 * bp, (self.n, K))

        # --- η: Gamma(cp + K*c, b_eta) ---
        self.a_eta = np.full(self.p, self.cp + K * self.c)
        self.b_eta = np.random.uniform(0.5 * dp, 1.5 * dp, self.p)

        # --- β: Gamma(a_beta, b_beta) ---
        self.a_beta = np.random.uniform(0.5 * self.c, 1.5 * self.c,
                                        (self.p, K))
        self.b_beta = np.random.uniform(0.5 * dp, 1.5 * dp, (self.p, K))

        # --- Apply pathway mask if needed ---
        self._init_beta_mask()

        # --- v: init small random ---
        self.mu_v = np.random.randn(self.kappa, K) * 0.01
        if self.v_prior == 'normal':
            # N(0, sigma_v^2/K) prior — K-scaled so total logit θ·v has variance ~ sigma_v^2
            self.sigma_v_diag = np.full((self.kappa, K), (self.sigma_v ** 2) / K)
        else:
            # Laplace (Bayesian Lasso): init sigma_v_diag from Laplace variance = 2*b_v^2
            self.sigma_v_diag = np.full((self.kappa, K), 2.0 * self.b_v ** 2)

        # --- γ: N(0, sigma_gamma^2) ---
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

        # --- Store sparse structure for O(nnz*K) phi ---
        if sp.issparse(X):
            X_coo = X.tocoo()
        else:
            X_coo = sp.coo_matrix(X)
        self._X_row = X_coo.row.astype(np.int32)
        self._X_col = X_coo.col.astype(np.int32)
        self._X_data = X_coo.data.astype(np.float64)
        self._nnz = len(self._X_data)

        # Pre-compute digamma caches (avoid repeated recomputation)
        self._refresh_log_caches()

        print(f"Initialized: n={self.n}, p={self.p}, K={K}, nnz={self._nnz}")
        print(f"  bp={bp:.4f}, dp={dp:.4f}")
        print(f"  E[theta] range: [{self.E_theta.min():.4f}, {self.E_theta.max():.4f}]")
        print(f"  E[beta] range: [{self.E_beta.min():.4f}, {self.E_beta.max():.4f}]")

    def _init_beta_mask(self):
        """Build beta_mask array for pathway modes."""
        if self.mode == 'masked' and self.pathway_mask is not None:
            # pathway_mask: (n_pathways, n_genes) → transpose to (n_genes, K)
            # pad or truncate to K columns
            pm = self.pathway_mask.T  # (p, n_pathways)
            if pm.shape[1] < self.K:
                pad = np.zeros((self.p, self.K - pm.shape[1]))
                self.beta_mask = np.hstack([pm, pad])
            else:
                self.beta_mask = pm[:, :self.K]
            # Zero out β where mask is 0
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
        if self.mode == 'masked':
            self.a_beta = np.where(self.beta_mask > 0.5, self.a_beta, small_a)
            self.b_beta = np.where(self.beta_mask > 0.5, self.b_beta, large_b)
        elif self.mode == 'combined':
            npath = self.n_pathway_factors
            for k in range(npath):
                self.a_beta[:, k] = np.where(
                    self.beta_mask[:, k] > 0.5, self.a_beta[:, k], small_a)
                self.b_beta[:, k] = np.where(
                    self.beta_mask[:, k] > 0.5, self.b_beta[:, k], large_b)

    # =================================================================
    # Expected values (properties for convenience)
    # =================================================================

    @property
    def E_theta(self):
        return self.a_theta / self.b_theta

    @property
    def E_log_theta(self):
        return digamma(self.a_theta) - np.log(self.b_theta)

    @property
    def E_beta(self):
        return self.a_beta / self.b_beta

    @property
    def E_log_beta(self):
        return digamma(self.a_beta) - np.log(self.b_beta)

    def _refresh_log_caches(self):
        """Recompute cached digamma arrays after theta/beta updates."""
        self._E_log_theta_cache = digamma(self.a_theta) - np.log(self.b_theta)
        self._E_log_beta_cache = digamma(self.a_beta) - np.log(self.b_beta)

    @property
    def E_xi(self):
        return self.a_xi / self.b_xi

    @property
    def E_eta(self):
        return self.a_eta / self.b_eta

    # =================================================================
    # φ computation (sparse, O(nnz*K))
    # =================================================================

    def _compute_phi_sparse(self, random_init=False):
        """
        Compute Xφ using only nonzero entries.

        Returns:
            z_sum_beta: (p, K) = Σ_i x_{ij} φ_{ijk}
            z_sum_theta: (n, K) = Σ_j x_{ij} φ_{ijk}
        """
        K = self.K

        if random_init:
            # First iteration: random Dirichlet φ (scHPF)
            phi = np.random.dirichlet(np.ones(K), self._nnz)
        else:
            # φ_{ijk} ∝ exp(ψ(a^θ_{ik}) - log(b^θ_{ik}) + ψ(a^β_{jk}) - log(b^β_{jk}))
            log_phi = (self._E_log_theta_cache[self._X_row]
                       + self._E_log_beta_cache[self._X_col])
            log_phi = np.clip(log_phi, -700, 700)  # prevent exp overflow
            log_phi -= _logsumexp_rows(log_phi)
            phi = np.exp(log_phi)

        # x_{ij} * φ_{ijk}
        Xphi = self._X_data[:, None] * phi  # (nnz, K)

        # Accumulate via bincount (10-20× faster than np.add.at)
        z_sum_beta = _scatter_add(self._X_col, Xphi, self.p)
        z_sum_theta = _scatter_add(self._X_row, Xphi, self.n)

        return z_sum_beta, z_sum_theta

    # =================================================================
    # CAVI Updates
    # =================================================================

    def _update_beta(self, z_sum_beta):
        """β shape and rate (scHPF Eq 8, gene side)."""
        # a^β_{jk} = c + Σ_i x_{ij} φ_{ijk}
        self.a_beta = self.c + z_sum_beta
        # b^β_{jk} = E[η_j] + Σ_i E[θ_{ik}]
        theta_sum = self.E_theta.sum(axis=0)  # (K,)
        self.b_beta = self.E_eta[:, None] + theta_sum[None, :]
        self._enforce_beta_mask()

    def _update_eta(self):
        """η rate (scHPF): b^η_j = d' + Σ_k E[β_{jk}]."""
        # a^η is constant = cp + K*c (set in init, never changes)
        self.b_eta = self.dp + self.E_beta.sum(axis=1)

    def _update_theta(self, z_sum_theta, y, X_aux):
        """
        θ shape and rate (scHPF Eq 7, cell side + JJ regression).

        Solves the quadratic for b_theta in one step:
            b² - b_base·b - c_quad·a_theta = 0
            b = (b_base + sqrt(b_base² + 4·c_quad·a_theta)) / 2
        which is always positive (given c_quad > 0, guaranteed by the ζ cap).

        Uses adaptive damping and rate clipping for numerical stability.
        """
        # Store old b_theta for damping
        b_theta_old = self.b_theta.copy()

        # a^θ_{ik} = a + Σ_j x_{ij} φ_{ijk}
        self.a_theta = self.a + z_sum_theta
        self.a_theta = np.maximum(self.a_theta, self.a * 0.5)  # never too small

        # b_base = E[ξ_i] + Σ_j E[β_{jk}] + regression_weight * R_linear
        beta_sum = self.E_beta.sum(axis=0)  # (K,)
        b_poisson = self.E_xi[:, None] + beta_sum[None, :]

        if self.regression_weight > 0:
            # Single-step quadratic solve (no inner θ-ζ loop).
            # The ζ cap keeps λ bounded from below, ensuring the
            # quadratic braking on θ never vanishes.
            R_linear, R_quad_coeff = self._regression_rate_parts(y, X_aux)
            b_base = b_poisson + self.regression_weight * R_linear
            c_quad = self.regression_weight * R_quad_coeff

            # Safe quadratic: clamp discriminant to avoid NaN from negative values
            discriminant = b_base ** 2 + 4.0 * c_quad * self.a_theta
            discriminant = np.maximum(discriminant, 0.0)
            new_b = (b_base + np.sqrt(discriminant)) / 2.0

            # Adaptive damping (same schedule as v/γ)
            alpha = min(0.25, 0.05 + 0.20 * (self.current_iter / 200.0))
            self.b_theta = (1.0 - alpha) * b_theta_old + alpha * new_b

            # Hard floor — prevents b_theta → 0 which causes E[θ] explosion
            self.b_theta = np.maximum(self.b_theta, 1e-4)
            self._theta_inner_iters = 1

            # Re-tighten ζ for updated θ
            self._update_zeta(X_aux)
        else:
            self.b_theta = b_poisson
            self._theta_inner_iters = 0

    def _update_xi(self):
        """ξ rate (scHPF): b^ξ_i = b' + Σ_k E[θ_{ik}]."""
        # a^ξ is constant = ap + K*a (set in init)
        self.b_xi = self.bp + self.E_theta.sum(axis=1)

    def _clip_rates(self):
        """Global safety net: floor all rate parameters to prevent division-by-zero / overflow."""
        self.b_theta = np.maximum(self.b_theta, 1e-4)
        self.b_beta  = np.maximum(self.b_beta,  1e-6)
        self.b_xi    = np.maximum(self.b_xi,    1e-4)
        self.b_eta   = np.maximum(self.b_eta,   1e-6)

    def _update_zeta(self, X_aux):
        """JJ auxiliary: ζ_{ik} = min(sqrt(E[A²_{ik}]), ζ_max).

        Capping ζ keeps λ_JJ(ζ) bounded from below, which prevents the
        quadratic braking on θ from vanishing.  The JJ lower bound is valid
        for ANY ζ, so capping gives a slightly looser but still valid bound.
        Each CAVI update still provably increases this capped-ζ ELBO.
        """
        E_theta = self.E_theta
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K)
        Var_theta = self.a_theta / (self.b_theta ** 2)  # (n, K)

        E_A = E_theta @ self.mu_v.T  # (n, kappa)
        if self.p_aux > 0:
            E_A = E_A + X_aux @ self.mu_gamma.T

        E_A_sq = E_A ** 2 + Var_theta @ E_v_sq.T  # (n, kappa)
        zeta_raw = np.sqrt(np.maximum(E_A_sq, 1e-8))
        self.zeta = np.minimum(zeta_raw, self.zeta_max)

    def _regression_rate_parts(self, y, X_aux):
        """
        Compute the JJ regression correction to θ rate, split into linear
        and quadratic parts.

        The full correction is R = R_linear + R_quad_coeff * E[θ], where:
          R_linear_{iℓ} = Σ_k [-(y_{ik} - 0.5) v_{kℓ}
                                + 2λ(ζ_{ik}) v_{kℓ} C^{(-ℓ)}_{ik}]
          R_quad_coeff_{iℓ} = Σ_k [2λ(ζ_{ik}) E[v²_{kℓ}]]

        Returns
        -------
        R_linear : (n, K) — terms not depending on E[θ_{iℓ}]
        R_quad_coeff : (n, K) — coefficient of E[θ_{iℓ}], always >= 0
        """
        y_exp = y if y.ndim > 1 else y[:, None]  # (n, kappa)
        lam = self._lambda_jj(self.zeta)           # (n, kappa)
        E_theta = self.E_theta                      # (n, K)
        E_v = self.mu_v                             # (kappa, K)
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag # (kappa, K)

        # Full linear predictor: (n, kappa)
        theta_v = E_theta @ E_v.T
        if self.p_aux > 0:
            theta_v = theta_v + X_aux @ self.mu_gamma.T

        # C^{(-ℓ)}_{ik} = A_{ik} - θ_{iℓ} v_{kℓ}
        full_lp = theta_v[:, :, None]  # (n, kappa, 1)
        C_minus = full_lp - E_theta[:, None, :] * E_v[None, :, :]  # (n, kappa, K)

        # Linear part: (n, kappa, K) → sum over kappa → (n, K)
        R_linear = (
            -(y_exp[:, :, None] - 0.5) * E_v[None, :, :]
            + 2 * lam[:, :, None] * E_v[None, :, :] * C_minus
        ).sum(axis=1)

        # Quadratic coefficient: (n, kappa, K) → sum over kappa → (n, K)
        R_quad_coeff = (2 * lam[:, :, None] * E_v_sq[None, :, :]).sum(axis=1)

        return R_linear, R_quad_coeff

    def _update_v(self, y, X_aux, iteration=0):
        """
        v posterior (diagonal Gaussian, JJ bound).

        For v_prior='normal':
            precision_{kℓ} = 1/σ²_eff + 2 Σ_i λ(ζ_{ik}) E[θ²_{iℓ}]
            σ²_eff = σ²_v / K

        For v_prior='laplace' (Bayesian Lasso):
            precision_{kℓ} = E_q[1/s_{kℓ}] + 2 Σ_i λ(ζ_{ik}) E[θ²_{iℓ}]
            where E_q[1/s_{kℓ}] is computed from the inverse Gaussian
            variational posterior on the scale mixture variable s_{kℓ}.
        """
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = self._lambda_jj(self.zeta)
        E_theta = self.E_theta
        Var_theta = self.a_theta / (self.b_theta ** 2)
        E_theta_sq = E_theta ** 2 + Var_theta
        E_v = self.mu_v

        # Full predictor: (n, kappa)
        theta_v = E_theta @ E_v.T
        if self.p_aux > 0:
            theta_v = theta_v + X_aux @ self.mu_gamma.T

        # C^{(-ℓ)}: (n, kappa, K)
        C_minus = theta_v[:, :, None] - E_theta[:, None, :] * E_v[None, :, :]

        if self.v_prior == 'normal':
            # K-scaled prior: each v_kl has prior variance sigma_v^2 / K
            sigma_v_eff_sq = (self.sigma_v ** 2) / self.K
            prior_precision = 1.0 / sigma_v_eff_sq
        else:
            # Laplace (Bayesian Lasso): adaptive precision from E_q[1/s_{kl}]
            # ω_{kl} = sqrt(E_q[v_{kl}^2]) = sqrt(mu_v^2 + sigma_v_diag)
            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K)
            omega = np.sqrt(np.maximum(E_v_sq, 1e-12))
            # E_q[s^{-1}] = 1/(b_v * ω) + 1/ω^2
            prior_precision = 1.0 / (self.b_v * omega) + 1.0 / (omega ** 2)

        # Precision and mean: both use raw λ_JJ (ζ-capped)
        precision = prior_precision + 2 * np.einsum('ik,id->kd', lam, E_theta_sq)

        term1 = np.einsum('ik,id->kd', y_exp - 0.5, E_theta)
        term2 = 2 * np.einsum('ik,ikd,id->kd', lam, C_minus, E_theta)
        mean_prec = term1 - term2

        sigma_v_diag_new = 1.0 / precision

        if self.v_prior == 'normal':
            # Clip v per-element: bound so max logit contribution per factor is
            # bounded.  For K=50 this is ~1.41, for K=348 this is ~0.54.
            v_clip = min(5.0, 10.0 / np.sqrt(self.K))
            mu_v_new = np.clip(mean_prec / precision, -v_clip, v_clip)

            # Adaptive damping: starts at alpha=0.05 (conservative), ramps toward
            # alpha_max over ~200 iterations.
            alpha_max = 0.15
            alpha = min(alpha_max, 0.05 + (alpha_max - 0.05) * (iteration / max(200, iteration)))
            self.mu_v = (1.0 - alpha) * self.mu_v + alpha * mu_v_new
            # Damp sigma_v_diag consistently with mu_v to keep q(v) coherent
            self.sigma_v_diag = (1.0 - alpha) * self.sigma_v_diag + alpha * sigma_v_diag_new
        else:
            # Laplace: no clip needed — adaptive E[1/s] acts as automatic shrinkage
            mu_v_new = mean_prec / precision
            self.mu_v = mu_v_new
            self.sigma_v_diag = sigma_v_diag_new

    def _update_gamma(self, y, X_aux, iteration=0):
        """γ posterior (Gaussian, JJ bound)."""
        if self.p_aux == 0:
            return
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = self._lambda_jj(self.zeta)
        E_theta = self.E_theta

        # Same adaptive damping schedule as v
        alpha_max = 0.15
        alpha = min(alpha_max, 0.05 + (alpha_max - 0.05) * (iteration / max(200, iteration)))

        for k in range(self.kappa):
            prec_prior = np.eye(self.p_aux) / (self.sigma_gamma ** 2)
            # Precision: 1/σ² I + 2 Σ_i λ(ζ_{ik}) x^aux_i x^aux_i^T
            weighted_X = X_aux * (2 * lam[:, k])[:, None]
            prec_lik = weighted_X.T @ X_aux
            prec = prec_prior + prec_lik

            # Residual: y - 0.5 - 2λ(ζ) θ·v
            theta_v_k = E_theta @ self.mu_v[k]
            residual = (y_exp[:, k] - 0.5) - 2 * lam[:, k] * theta_v_k
            mean_prec = X_aux.T @ residual

            self.Sigma_gamma[k] = np.linalg.inv(prec)
            mu_gamma_new = self.Sigma_gamma[k] @ mean_prec
            self.mu_gamma[k] = (1.0 - alpha) * self.mu_gamma[k] + alpha * mu_gamma_new

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
        E_log_xi = digamma(self.a_xi) - np.log(self.b_xi)
        E_log_eta = digamma(self.a_eta) - np.log(self.b_eta)

        elbo = 0.0

        # === Poisson likelihood (collapsed z) ===
        # Use sparse computation
        E_log_theta_nnz = E_log_theta[self._X_row]
        E_log_beta_nnz = E_log_beta[self._X_col]
        log_rates = E_log_theta_nnz + E_log_beta_nnz  # (nnz, K)
        log_rates_max = log_rates.max(axis=1)
        log_sum_rates = log_rates_max + np.log(
            np.exp(log_rates - log_rates_max[:, None]).sum(axis=1))  # (nnz,)
        poisson_ll = np.sum(self._X_data * log_sum_rates)
        poisson_ll -= np.sum(E_theta.sum(axis=0) * E_beta.sum(axis=0))
        poisson_ll -= np.sum(gammaln(self._X_data + 1))
        elbo += poisson_ll

        # === JJ Bernoulli likelihood ===
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = self._lambda_jj(self.zeta)
        E_A = E_theta @ self.mu_v.T
        if self.p_aux > 0:
            E_A = E_A + X_aux @ self.mu_gamma.T
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag
        Var_theta = self.a_theta / (self.b_theta ** 2)
        E_A_sq = E_A ** 2 + Var_theta @ E_v_sq.T

        regression_ll = np.sum((y_exp - 0.5) * E_A - lam * E_A_sq)
        regression_ll += np.sum(lam * self.zeta ** 2 - 0.5 * self.zeta
                                + np.log(1 / (1 + np.exp(-self.zeta))))
        elbo += self.regression_weight * regression_ll

        # === Prior: p(θ|ξ) ===
        elbo += np.sum((self.a - 1) * E_log_theta
                       + self.a * E_log_xi[:, None]
                       - E_xi[:, None] * E_theta)
        elbo -= self.n * self.K * gammaln(self.a)

        # === Prior: p(β|η) ===
        elbo += np.sum((self.c - 1) * E_log_beta
                       + self.c * E_log_eta[:, None]
                       - E_eta[:, None] * E_beta)
        elbo -= self.p * self.K * gammaln(self.c)

        # === Prior: p(ξ) ===
        elbo += np.sum((self.ap - 1) * E_log_xi
                       + self.ap * np.log(self.bp) - self.bp * E_xi)
        elbo -= self.n * gammaln(self.ap)

        # === Prior: p(η) ===
        elbo += np.sum((self.cp - 1) * E_log_eta
                       + self.cp * np.log(self.dp) - self.dp * E_eta)
        elbo -= self.p * gammaln(self.cp)

        # === Prior: p(v) ===
        if self.v_prior == 'normal':
            # K-scaled Gaussian: each v_kl ~ N(0, sigma_v^2 / K)
            sigma_v_eff_sq = (self.sigma_v ** 2) / self.K
            elbo -= 0.5 * np.sum(self.mu_v ** 2 + self.sigma_v_diag) / sigma_v_eff_sq
            elbo -= 0.5 * self.kappa * self.K * np.log(2 * np.pi * sigma_v_eff_sq)
        else:
            # Laplace (Bayesian Lasso): augmented model with s_{kl}
            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K)
            omega = np.sqrt(np.maximum(E_v_sq, 1e-12))
            E_inv_s = 1.0 / (self.b_v * omega) + 1.0 / (omega ** 2)

            # E_q[log p(v|s)] = -0.5*log(2π) - 0.5*E_q[log s] - 0.5*E_q[v^2]*E_q[1/s]
            # Approximate E_q[log s] ≈ log(mu_s) for the inverse Gaussian
            mu_s = self.b_v / omega  # mu_s = b_v * sqrt(1/E[v^2])... = b_v / omega
            E_log_s = np.log(np.maximum(mu_s, 1e-12))  # approximation
            elbo += np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * E_log_s
                           - 0.5 * E_v_sq * E_inv_s)

            # E_q[log p(s)] = log(1/(2*b_v^2)) - E_q[s]/(2*b_v^2)
            # E_q[s] = mu_s for inverse Gaussian
            rate_s = 1.0 / (2.0 * self.b_v ** 2)
            elbo += np.sum(np.log(rate_s) - mu_s * rate_s)

            # H[q(s)] = 0.5*log(2πe * mu_s^3 / lambda_s)
            lambda_s = 1.0 / (self.b_v ** 2)
            elbo += 0.5 * np.sum(np.log(2 * np.pi * np.e * mu_s ** 3
                                        / np.maximum(lambda_s, 1e-12)))

        # === Entropy: -E[log q] ===
        # q(θ) Gamma entropy
        elbo += np.sum(self.a_theta - np.log(self.b_theta)
                       + gammaln(self.a_theta)
                       + (1 - self.a_theta) * digamma(self.a_theta))
        # q(β)
        elbo += np.sum(self.a_beta - np.log(self.b_beta)
                       + gammaln(self.a_beta)
                       + (1 - self.a_beta) * digamma(self.a_beta))
        # q(ξ)
        elbo += np.sum(self.a_xi - np.log(self.b_xi)
                       + gammaln(self.a_xi)
                       + (1 - self.a_xi) * digamma(self.a_xi))
        # q(η)
        elbo += np.sum(self.a_eta - np.log(self.b_eta)
                       + gammaln(self.a_eta)
                       + (1 - self.a_eta) * digamma(self.a_eta))
        # q(v) Gaussian entropy
        elbo += 0.5 * np.sum(np.log(2 * np.pi * np.e * self.sigma_v_diag))

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

        # Infer θ for validation cells (freeze globals)
        a_theta_v = np.random.uniform(0.5 * self.a, 1.5 * self.a,
                                       (n_val, self.K))
        b_theta_v = np.full((n_val, self.K), self.bp)
        a_xi_v = np.full(n_val, self.ap + self.K * self.a)
        b_xi_v = np.full(n_val, self.bp)

        E_log_beta = self._E_log_beta_cache
        E_beta = self.E_beta
        beta_col_sums = E_beta.sum(axis=0)

        row = X_val_coo.row
        col = X_val_coo.col
        data = X_val_coo.data if X_val_coo.data.dtype == np.float64 else X_val_coo.data.astype(np.float64)

        for _ in range(n_iter):
            E_log_theta_v = digamma(a_theta_v) - np.log(b_theta_v)
            E_theta_v = a_theta_v / b_theta_v
            E_xi_v = a_xi_v / b_xi_v

            # φ sparse
            log_phi = E_log_theta_v[row] + E_log_beta[col]
            log_phi -= _logsumexp_rows(log_phi)
            Xphi = data[:, None] * np.exp(log_phi)

            z_sum = _scatter_add(row, Xphi, n_val)

            a_theta_v = self.a + z_sum
            b_theta_v = E_xi_v[:, None] + beta_col_sums[None, :]
            b_xi_v = self.bp + E_theta_v.sum(axis=1)

        # Poisson LL per sample
        E_log_theta_v = digamma(a_theta_v) - np.log(b_theta_v)
        E_theta_v = a_theta_v / b_theta_v
        log_rates = E_log_theta_v[row] + E_log_beta[col]
        log_rates_max = log_rates.max(axis=1, keepdims=True)
        log_sum_rates = (log_rates_max.ravel()
                         + np.log(np.exp(log_rates - log_rates_max).sum(axis=1)))

        poisson_ll = np.sum(data * log_sum_rates)
        poisson_ll -= np.sum(E_theta_v.sum(axis=0) * E_beta.sum(axis=0))
        poisson_ll -= np.sum(gammaln(data + 1))
        poisson_ll_per_sample = poisson_ll / n_val

        # Regression LL on validation data (if labels provided)
        regression_ll_per_sample = None
        if y_val is not None:
            y_v = np.asarray(y_val, dtype=np.float64)
            if y_v.ndim == 1:
                y_v = y_v[:, None]
            if X_aux_val is None:
                X_aux_v = np.zeros((n_val, self.p_aux if self.p_aux > 0 else 0))
            else:
                X_aux_v = np.asarray(X_aux_val, dtype=np.float64)

            E_A = E_theta_v @ self.mu_v.T
            if self.p_aux > 0:
                E_A = E_A + X_aux_v @ self.mu_gamma.T

            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag
            Var_theta_v = a_theta_v / (b_theta_v ** 2)
            E_A_sq = E_A ** 2 + Var_theta_v @ E_v_sq.T

            zeta_val = np.sqrt(np.maximum(E_A_sq, 1e-8))
            lam = self._lambda_jj(zeta_val)

            reg_ll = np.sum((y_v - 0.5) * E_A - lam * E_A_sq)
            reg_ll += np.sum(lam * zeta_val ** 2 - 0.5 * zeta_val
                             + np.log(1 / (1 + np.exp(-zeta_val))))
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
            v_warmup=50, verbose=True):
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
        tol : float — convergence if |pct_change| < tol twice in a row
        v_warmup : int — deprecated, kept for CLI compatibility (ignored)
        verbose : bool
        """
        t0 = time.time()

        if X_aux_train is None:
            X_aux_train = np.zeros((X_train.shape[0], 0))
        y = np.asarray(y_train, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        X_aux = np.asarray(X_aux_train, dtype=np.float64)

        # Initialize
        self._initialize(X_train, y, X_aux)

        # Dense X for ELBO (if small enough) — otherwise use sparse
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

        self.current_iter = 0  # needed for damping schedule in _update_theta

        for t in range(max_iter):
            self.current_iter = t
            diag = verbose and (t % check_freq == 0)

            # 1. Compute φ and z_sums (sparse)
            random_phi = (t == 0)  # scHPF: random Dirichlet on first iter
            z_sum_beta, z_sum_theta = self._compute_phi_sparse(random_init=random_phi)

            # 2. Update β, η (gene side first — scHPF order)
            self._update_beta(z_sum_beta)
            self._update_eta()
            self._clip_rates()  # safety net after gene-side rate updates
            if diag:
                print(f"  [diag t={t}] after β,η: "
                      f"E[β]=[{self.E_beta.min():.4e},{self.E_beta.max():.4e}] "
                      f"E[η]=[{self.E_eta.min():.4e},{self.E_eta.max():.4e}]")

            # 3. Update ζ (JJ bound tightening)
            self._update_zeta(X_aux)
            if diag:
                print(f"  [diag t={t}] after ζ:   "
                      f"ζ=[{self.zeta.min():.4e},{self.zeta.max():.4e}] "
                      f"λ(ζ)=[{self._lambda_jj(self.zeta).min():.4e},"
                      f"{self._lambda_jj(self.zeta).max():.4e}]")

            # 4. Update θ, ξ (cell side — with full regression from start)
            self._update_theta(z_sum_theta, y, X_aux)
            self._update_xi()
            self._clip_rates()  # safety net after cell-side rate updates
            if diag:
                E_th = self.E_theta
                inner_info = f" inner={self._theta_inner_iters}" if self._theta_inner_iters > 0 else ""
                print(f"  [diag t={t}] after θ,ξ: "
                      f"E[θ]=[{E_th.min():.4e},{E_th.max():.4e}] mean={E_th.mean():.4e} "
                      f"b_θ=[{self.b_theta.min():.4e},{self.b_theta.max():.4e}] "
                      f"a_θ=[{self.a_theta.min():.4e},{self.a_theta.max():.4e}]{inner_info}")
                if self._theta_inner_iters > 0:
                    print(f"  [diag t={t}] post-θ ζ: "
                          f"ζ=[{self.zeta.min():.4e},{self.zeta.max():.4e}] "
                          f"λ(ζ)=[{self._lambda_jj(self.zeta).min():.4e},"
                          f"{self._lambda_jj(self.zeta).max():.4e}]")

            # Refresh digamma caches after theta/beta changed
            self._refresh_log_caches()

            # 5. Update v, γ
            self._update_v(y, X_aux, iteration=t)
            self._update_gamma(y, X_aux, iteration=t)

            # 5b. Re-tighten ζ after v/γ changed, so ELBO is evaluated at
            # the optimal ζ for the current (θ, v, γ).
            self._update_zeta(X_aux)

            if diag:
                print(f"  [diag t={t}] after v,γ: "
                      f"μ_v=[{self.mu_v.min():.4e},{self.mu_v.max():.4e}] "
                      f"σ²_v=[{self.sigma_v_diag.min():.4e},{self.sigma_v_diag.max():.4e}] "
                      f"μ_γ=[{self.mu_gamma.min():.4e},{self.mu_gamma.max():.4e}]"
                      if self.p_aux > 0 else
                      f"  [diag t={t}] after v:   "
                      f"μ_v=[{self.mu_v.min():.4e},{self.mu_v.max():.4e}] "
                      f"σ²_v=[{self.sigma_v_diag.min():.4e},{self.sigma_v_diag.max():.4e}]")

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
                              f"({prev_elbo:.4e} → {elbo:.4e})")

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
                    print(f"Iter {t:4d}: ELBO={elbo:.4e}  "
                          f"Pois={pois_ll:.4e}  Reg={reg_ll:.4e}  "
                          f"v={self.mu_v.ravel()[:3]}{holl_str}")

                # HO-LL regression early stopping (preferred when validation available)
                if X_val is not None and holl_reg is not None:
                    iters_since_best = t - best_holl_reg_iter
                    if iters_since_best >= holl_reg_patience and t >= 30:
                        if verbose:
                            print(f"HO-LL Reg early stop at iter {t}: "
                                  f"HO-Reg hasn't improved in {iters_since_best} iters "
                                  f"(best HO-Reg={best_holl_reg:.4e} at iter {best_holl_reg_iter})")
                        break

                # Regression early stopping (training): if Reg has been degrading for
                # reg_patience consecutive iters, stop and restore best.
                iters_since_best = t - best_reg_iter
                if iters_since_best >= reg_patience and t >= 30:
                    if verbose:
                        print(f"Regression early stop at iter {t}: "
                              f"Reg hasn't improved in {iters_since_best} iters "
                              f"(best Reg={best_reg_ll:.4e} at iter {best_reg_iter})")
                    break

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

        # Restore best: prefer HO-LL reg checkpoint > HO-LL total > training Reg
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

        return self

    # =================================================================
    # Checkpoint/restore
    # =================================================================

    def _checkpoint(self):
        return {
            'a_beta': self.a_beta.copy(), 'b_beta': self.b_beta.copy(),
            'a_eta': self.a_eta.copy(), 'b_eta': self.b_eta.copy(),
            'a_theta': self.a_theta.copy(), 'b_theta': self.b_theta.copy(),
            'a_xi': self.a_xi.copy(), 'b_xi': self.b_xi.copy(),
            'mu_v': self.mu_v.copy(), 'sigma_v_diag': self.sigma_v_diag.copy(),
            'mu_gamma': self.mu_gamma.copy(), 'Sigma_gamma': self.Sigma_gamma.copy(),
            'zeta': self.zeta.copy(),
        }

    def _restore(self, cp):
        for k, v in cp.items():
            setattr(self, k, v)

    # =================================================================
    # predict / transform (API compat)
    # =================================================================

    def predict_proba(self, X_new, X_aux_new=None, n_iter=20):
        """Predict P(y=1 | X_new)."""
        if sp.issparse(X_new):
            X_coo = X_new.tocoo()
        else:
            X_coo = sp.coo_matrix(X_new)

        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, self.p_aux if self.p_aux > 0 else 0))

        # Infer θ
        a_theta = np.random.uniform(0.5 * self.a, 1.5 * self.a, (n_new, self.K))
        b_theta = np.full((n_new, self.K), self.bp)
        a_xi = np.full(n_new, self.ap + self.K * self.a)
        b_xi = np.full(n_new, self.bp)

        E_log_beta = self.E_log_beta
        E_beta = self.E_beta
        beta_col_sums = E_beta.sum(axis=0)

        row, col = X_coo.row, X_coo.col
        data = X_coo.data if X_coo.data.dtype == np.float64 else X_coo.data.astype(np.float64)

        for _ in range(n_iter):
            E_log_theta = digamma(a_theta) - np.log(b_theta)
            E_theta = a_theta / b_theta
            E_xi = a_xi / b_xi

            log_phi = E_log_theta[row] + E_log_beta[col]
            log_phi -= _logsumexp_rows(log_phi)
            Xphi = data[:, None] * np.exp(log_phi)
            z_sum = _scatter_add(row, Xphi, n_new)

            a_theta = self.a + z_sum
            b_theta = E_xi[:, None] + beta_col_sums[None, :]
            b_xi = self.bp + E_theta.sum(axis=1)

        E_theta = a_theta / b_theta
        logits = E_theta @ self.mu_v.T
        if self.p_aux > 0:
            logits = logits + X_aux_new @ self.mu_gamma.T

        from scipy.special import expit
        return expit(logits).squeeze()

    def transform(self, X_new, y_new=None, X_aux_new=None, n_iter=20, **kwargs):
        """Infer θ for new data. Returns dict with E_theta, a_theta, b_theta."""
        if sp.issparse(X_new):
            X_coo = X_new.tocoo()
        else:
            X_coo = sp.coo_matrix(X_new)

        n_new = X_new.shape[0]
        a_theta = np.random.uniform(0.5 * self.a, 1.5 * self.a, (n_new, self.K))
        b_theta = np.full((n_new, self.K), self.bp)
        a_xi = np.full(n_new, self.ap + self.K * self.a)
        b_xi = np.full(n_new, self.bp)

        E_log_beta = self.E_log_beta
        beta_col_sums = self.E_beta.sum(axis=0)
        row, col = X_coo.row, X_coo.col
        data = X_coo.data if X_coo.data.dtype == np.float64 else X_coo.data.astype(np.float64)

        for _ in range(n_iter):
            E_log_theta = digamma(a_theta) - np.log(b_theta)
            E_theta = a_theta / b_theta
            E_xi = a_xi / b_xi

            log_phi = E_log_theta[row] + E_log_beta[col]
            log_phi -= _logsumexp_rows(log_phi)
            Xphi = data[:, None] * np.exp(log_phi)
            z_sum = _scatter_add(row, Xphi, n_new)

            a_theta = self.a + z_sum
            b_theta = E_xi[:, None] + beta_col_sums[None, :]
            b_xi = self.bp + E_theta.sum(axis=1)

        return {
            'E_theta': a_theta / b_theta,
            'a_theta': a_theta,
            'b_theta': b_theta,
        }