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
6. Simple v_warmup: skip v/γ/ζ updates for first N iterations
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
from scipy.special import logsumexp as sp_logsumexp
from typing import Tuple, Optional, Dict, Any, List
import time


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
        Gaussian prior std for v (regression weights).
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
        self.sigma_gamma = sigma_gamma
        self.regression_weight = regression_weight

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

        # --- v: N(0, sigma_v^2) prior → init small random ---
        self.mu_v = np.random.randn(self.kappa, K) * 0.01
        self.sigma_v_diag = np.full((self.kappa, K), self.sigma_v ** 2)

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
            E_log_theta_nnz = self.E_log_theta[self._X_row]  # (nnz, K)
            E_log_beta_nnz = self.E_log_beta[self._X_col]    # (nnz, K)
            log_phi = E_log_theta_nnz + E_log_beta_nnz        # (nnz, K)
            log_phi -= sp_logsumexp(log_phi, axis=1, keepdims=True)
            phi = np.exp(log_phi)

        # x_{ij} * φ_{ijk}
        Xphi = self._X_data[:, None] * phi  # (nnz, K)

        # Accumulate
        z_sum_beta = np.zeros((self.p, K))
        z_sum_theta = np.zeros((self.n, K))
        np.add.at(z_sum_beta, self._X_col, Xphi)
        np.add.at(z_sum_theta, self._X_row, Xphi)

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
        """θ shape and rate (scHPF Eq 7, cell side + JJ regression)."""
        # a^θ_{ik} = a + Σ_j x_{ij} φ_{ijk}
        self.a_theta = self.a + z_sum_theta

        # b^θ_{ik} = E[ξ_i] + Σ_j E[β_{jk}]
        beta_sum = self.E_beta.sum(axis=0)  # (K,)
        b_theta_new = self.E_xi[:, None] + beta_sum[None, :]

        # JJ regression correction (if active)
        if self.regression_weight > 0:
            R = self._regression_rate_correction(y, X_aux)
            b_theta_new = b_theta_new + self.regression_weight * R

        # Floor to prevent negative rates
        self.b_theta = np.maximum(b_theta_new, 1e-6)

    def _update_xi(self):
        """ξ rate (scHPF): b^ξ_i = b' + Σ_k E[θ_{ik}]."""
        # a^ξ is constant = ap + K*a (set in init)
        self.b_xi = self.bp + self.E_theta.sum(axis=1)

    def _update_zeta(self, X_aux):
        """JJ auxiliary: ζ_{ik} = sqrt(E[A²_{ik}])."""
        E_theta = self.E_theta
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K)
        Var_theta = self.a_theta / (self.b_theta ** 2)  # (n, K)

        E_A = E_theta @ self.mu_v.T  # (n, kappa)
        if self.p_aux > 0:
            E_A = E_A + X_aux @ self.mu_gamma.T

        E_A_sq = E_A ** 2 + Var_theta @ E_v_sq.T  # (n, kappa)
        self.zeta = np.sqrt(np.maximum(E_A_sq, 1e-8))

    def _regression_rate_correction(self, y, X_aux):
        """
        Compute the JJ regression correction to θ rate.

        R_{iℓ} = Σ_k [ -(y_{ik} - 0.5) v_{kℓ}
                       + 2λ(ζ_{ik}) v_{kℓ} C^{(-ℓ)}_{ik}
                       + 2λ(ζ_{ik}) E[v²_{kℓ}] E[θ_{iℓ}] ]

        Returns: (n, K) array.
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
        # We need (n, kappa, K) tensor
        full_lp = theta_v[:, :, None]  # (n, kappa, 1)
        C_minus = full_lp - E_theta[:, None, :] * E_v[None, :, :]  # (n, kappa, K)

        # R: (n, kappa, K)
        R = (
            -(y_exp[:, :, None] - 0.5) * E_v[None, :, :]
            + 2 * lam[:, :, None] * E_v[None, :, :] * C_minus
            + 2 * lam[:, :, None] * E_v_sq[None, :, :] * E_theta[:, None, :]
        )
        return R.sum(axis=1)  # (n, K)

    def _update_v(self, y, X_aux):
        """
        v posterior (diagonal Gaussian, JJ bound).

        precision_{kℓ} = 1/σ²_v + 2 Σ_i λ(ζ_{ik}) E[θ²_{iℓ}]
        mean*prec_{kℓ} = Σ_i [(y_{ik}-0.5) - 2λ(ζ_{ik})C^{(-ℓ)}_{ik}] E[θ_{iℓ}]
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

        # Precision: (kappa, K)
        precision = 1.0 / (self.sigma_v ** 2) + 2 * np.einsum('ik,id->kd', lam, E_theta_sq)

        # Mean*precision: (kappa, K)
        term1 = np.einsum('ik,id->kd', y_exp - 0.5, E_theta)
        term2 = 2 * np.einsum('ik,ikd,id->kd', lam, C_minus, E_theta)
        mean_prec = term1 - term2

        self.mu_v = np.clip(mean_prec / precision, -10, 10)
        self.sigma_v_diag = 1.0 / precision

    def _update_gamma(self, y, X_aux):
        """γ posterior (Gaussian, JJ bound)."""
        if self.p_aux == 0:
            return
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = self._lambda_jj(self.zeta)
        E_theta = self.E_theta

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
            self.mu_gamma[k] = self.Sigma_gamma[k] @ mean_prec

    # =================================================================
    # ELBO
    # =================================================================

    def _compute_elbo(self, X_dense, y, X_aux):
        """Compute ELBO = E[log p] - E[log q]."""
        E_theta = self.E_theta
        E_log_theta = self.E_log_theta
        E_beta = self.E_beta
        E_log_beta = self.E_log_beta
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
        log_sum_rates = sp_logsumexp(log_rates, axis=1)  # (nnz,)
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
        elbo -= 0.5 * np.sum(self.mu_v ** 2 + self.sigma_v_diag) / (self.sigma_v ** 2)
        elbo -= 0.5 * self.kappa * self.K * np.log(2 * np.pi * self.sigma_v ** 2)

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
        """Mean negative Poisson log-likelihood on held-out data."""
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

        E_log_beta = self.E_log_beta
        E_beta = self.E_beta
        beta_col_sums = E_beta.sum(axis=0)

        row = X_val_coo.row
        col = X_val_coo.col
        data = X_val_coo.data.astype(np.float64)

        for _ in range(n_iter):
            E_log_theta_v = digamma(a_theta_v) - np.log(b_theta_v)
            E_theta_v = a_theta_v / b_theta_v
            E_xi_v = a_xi_v / b_xi_v

            # φ sparse
            log_phi = E_log_theta_v[row] + E_log_beta[col]
            log_phi -= sp_logsumexp(log_phi, axis=1, keepdims=True)
            phi = np.exp(log_phi)
            Xphi = data[:, None] * phi

            z_sum = np.zeros((n_val, self.K))
            np.add.at(z_sum, row, Xphi)

            a_theta_v = self.a + z_sum
            b_theta_v = E_xi_v[:, None] + beta_col_sums[None, :]
            b_xi_v = self.bp + E_theta_v.sum(axis=1)

        # Poisson LL per sample
        E_log_theta_v = digamma(a_theta_v) - np.log(b_theta_v)
        E_theta_v = a_theta_v / b_theta_v
        log_rates = E_log_theta_v[row] + E_log_beta[col]
        log_sum_rates = sp_logsumexp(log_rates, axis=1)

        ll = np.sum(data * log_sum_rates)
        ll -= np.sum(E_theta_v.sum(axis=0) * E_beta.sum(axis=0))
        ll -= np.sum(gammaln(data + 1))

        return ll / n_val

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
        v_warmup : int — iterations before turning on regression
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

        for t in range(max_iter):
            in_warmup = (t < v_warmup)

            # 1. Compute φ and z_sums (sparse)
            random_phi = (t == 0)  # scHPF: random Dirichlet on first iter
            z_sum_beta, z_sum_theta = self._compute_phi_sparse(random_init=random_phi)

            # 2. Update β, η (gene side first — scHPF order)
            self._update_beta(z_sum_beta)
            self._update_eta()

            # 3. Update ζ (JJ bound tightening — only after warmup)
            if not in_warmup:
                self._update_zeta(X_aux)

            # 4. Update θ, ξ (cell side)
            if in_warmup:
                effective_rw = 0.0
            else:
                ramp_iters = 50  # ramp over 50 iterations after warmup
                frac = min(1.0, (t - v_warmup) / ramp_iters)
                effective_rw = self.regression_weight * frac

            old_rw = self.regression_weight
            self.regression_weight = effective_rw
            self._update_theta(z_sum_theta, y, X_aux)
            self.regression_weight = old_rw
            self._update_xi()

            # 5. Update v, γ (only after warmup)
            if not in_warmup:
                self._update_v(y, X_aux)
                self._update_gamma(y, X_aux)

            # 6. Check convergence
            if t % check_freq == 0:
                elbo, pois_ll, reg_ll = self._compute_elbo(X_dense, y, X_aux)
                self.elbo_history_.append((t, elbo))

                # Held-out LL
                holl = None
                if X_val is not None:
                    holl = self.compute_heldout_ll(X_val)
                    self.holl_history_.append((t, holl))
                    if holl > best_holl:
                        best_holl = holl
                        best_params = self._checkpoint()

                # Loss = mean negative Poisson LL on training (scHPF)
                curr_loss = -pois_ll / self.n
                loss_list.append(curr_loss)

                if len(loss_list) >= 2:
                    prev = loss_list[-2]
                    pct = 100 * (curr_loss - prev) / max(abs(prev), 1e-10)
                    pct_changes.append(pct)
                else:
                    pct_changes.append(100.0)

                if verbose:
                    phase = "WU" if in_warmup else "FT"
                    holl_str = f"  HO-LL={holl:.2f}" if holl is not None else ""
                    print(f"Iter {t:4d} [{phase}]: ELBO={elbo:.4e}  "
                          f"Pois={pois_ll:.4e}  Reg={reg_ll:.4e}  "
                          f"v={self.mu_v.ravel()[:3]}{holl_str}")

                # Convergence check (scHPF style)
                if len(loss_list) >= 3 and t >= max(30, v_warmup + 10):
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

        # Restore best if we have validation
        if best_params is not None:
            self._restore(best_params)

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

        row, col, data = X_coo.row, X_coo.col, X_coo.data.astype(np.float64)

        for _ in range(n_iter):
            E_log_theta = digamma(a_theta) - np.log(b_theta)
            E_theta = a_theta / b_theta
            E_xi = a_xi / b_xi

            log_phi = E_log_theta[row] + E_log_beta[col]
            log_phi -= sp_logsumexp(log_phi, axis=1, keepdims=True)
            phi = np.exp(log_phi)
            Xphi = data[:, None] * phi
            z_sum = np.zeros((n_new, self.K))
            np.add.at(z_sum, row, Xphi)

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
        row, col, data = X_coo.row, X_coo.col, X_coo.data.astype(np.float64)

        for _ in range(n_iter):
            E_log_theta = digamma(a_theta) - np.log(b_theta)
            E_theta = a_theta / b_theta
            E_xi = a_xi / b_xi

            log_phi = E_log_theta[row] + E_log_beta[col]
            log_phi -= sp_logsumexp(log_phi, axis=1, keepdims=True)
            Xphi = data[:, None] * np.exp(log_phi)
            z_sum = np.zeros((n_new, self.K))
            np.add.at(z_sum, row, Xphi)

            a_theta = self.a + z_sum
            b_theta = E_xi[:, None] + beta_col_sums[None, :]
            b_xi = self.bp + E_theta.sum(axis=1)

        return {
            'E_theta': a_theta / b_theta,
            'a_theta': a_theta,
            'b_theta': b_theta,
        }