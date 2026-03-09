"""
Coordinate Ascent Variational Inference for Supervised Poisson Factorization
=============================================================================

Full-batch CAVI following the scHPF update pattern (Levitin et al., MSB 2019)
extended with Jaakkola-Jordan logistic regression terms from the BRay model.

Key differences from SVI (svi_corrected.py):
1. No learning rate schedule — closed-form coordinate updates
2. No mini-batching — full dataset per iteration (scale = 1)
3. Damping replaces Robbins-Monro averaging for stability
4. Convergence via ELBO relative change (deterministic, no EMA needed)

Gamma updates mirror scHPF exactly:
  shape_θ_iℓ = α_θ + Σ_j x_{ij} φ_{ijℓ}
  rate_θ_iℓ  = E[ξ_i] + Σ_j E[β_{jℓ}] + regression_correction

Regression (JJ-bound) updates follow Appendix A of the manuscript;
ζ is the Jaakkola-Jordan auxiliary variable.

References:
- Levitin et al. (2019) "De novo gene signature identification from
  single-cell RNA-seq with hierarchical Poisson factorization", MSB
- Jaakkola & Jordan (2000) "Bayesian parameter estimation via variational
  methods", Statistics and Computing 10(1):25–37
- Blei, Kucukelbir & McAuliffe (2017) "Variational inference: a review for
  statisticians", JASA 112(518):859–877

JAX is used for vectorized operations; no natural-parameter bookkeeping
is needed since CAVI updates are direct coordinate updates.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.scipy.special import logsumexp
from jax import random
from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np
import scipy.sparse as sp


class CAVI:
    """
    Coordinate Ascent Variational Inference for Supervised Poisson Factorization.

    API-compatible with SVICorrected: same __init__ signature subset,
    same fit / transform / predict_proba interface so that downstream
    code (quick_reference.py, utils.py) works unchanged.

    Parameters
    ----------
    n_factors : int
        Number of latent factors d.
    alpha_theta, alpha_beta, alpha_xi, alpha_eta : float
        Gamma prior shape parameters.
    lambda_xi, lambda_eta : float
        Gamma prior rate parameters.
    sigma_v, sigma_gamma : float
        Gaussian prior std for regression weights.
    use_spike_slab : bool
        Enable spike-and-slab priors on β and v.
    pi_beta, pi_v : float
        Prior inclusion probabilities for spike-and-slab.
    regression_weight : float
        Weight for the classification term in ELBO / updates.
    random_state : int or None
        Seed for reproducibility.
    mode : str
        'unmasked', 'masked', 'pathway_init', 'combined'.
    pathway_mask, pathway_names, n_pathway_factors
        Pathway configuration (see SVICorrected).
    """

    def __init__(
        self,
        n_factors: int,
        # Priors
        alpha_theta: float = 1.1,
        alpha_beta: float = 1.1,
        alpha_xi: float = 2.0,
        alpha_eta: float = 2.0,
        lambda_xi: float = 1.0,
        lambda_eta: float = 1.0,
        sigma_v: float = 1.0,
        sigma_gamma: float = 1.0,
        # Spike-and-slab
        use_spike_slab: bool = False,
        pi_beta: float = 0.1,
        pi_v: float = 0.5,
        # Supervision
        regression_weight: float = 1.0,
        # Random state
        random_state: Optional[int] = None,
        # Pathway mode
        mode: str = 'unmasked',
        pathway_mask: Optional[np.ndarray] = None,
        pathway_names: Optional[List[str]] = None,
        n_pathway_factors: Optional[int] = None,
        # Ignored SVI args (accepted for config compat, unused)
        **_ignored,
    ):
        self.d = n_factors
        self.alpha_theta = alpha_theta
        self.alpha_beta = alpha_beta
        self.alpha_xi = alpha_xi
        self.alpha_eta = alpha_eta
        self.lambda_xi = lambda_xi
        self.lambda_eta = lambda_eta
        self.sigma_v = sigma_v
        self.sigma_gamma = sigma_gamma

        self.use_spike_slab = use_spike_slab
        self.pi_beta = pi_beta
        self.pi_v = pi_v

        self.regression_weight = regression_weight

        self.mode = mode
        self.pathway_mask = pathway_mask
        self.pathway_names = pathway_names
        self.n_pathway_factors = n_pathway_factors

        if mode in ['masked', 'pathway_init'] and pathway_mask is None:
            raise ValueError(f"pathway_mask required for mode='{mode}'")
        if mode == 'combined':
            if pathway_mask is None:
                raise ValueError("pathway_mask required for mode='combined'")
            if n_pathway_factors is None:
                raise ValueError("n_pathway_factors required for mode='combined'")
        if mode == 'masked':
            use_spike_slab = False
            self.use_spike_slab = False

        # RNG
        if random_state is not None:
            self.rng_key = random.PRNGKey(random_state)
            self.seed_used_ = random_state
        else:
            random_seed = int(time.time() * 1000) % (2**32)
            self.rng_key = random.PRNGKey(random_seed)
            self.seed_used_ = None

        # Dimensionality (set in fit)
        self.n = None
        self.p = None
        self.kappa = None
        self.p_aux = None

    # =====================================================================
    # Numerical helpers (same as SVICorrected, duplicated to avoid coupling)
    # =====================================================================

    @staticmethod
    @jax.jit
    def _lambda_jj(zeta: jnp.ndarray) -> jnp.ndarray:
        """λ(ζ) = tanh(ζ/2) / (4ζ),  λ(0) = 1/8."""
        return jnp.where(
            jnp.abs(zeta) > 1e-8,
            jnp.tanh(zeta / 2) / (4 * zeta + 1e-20),
            0.125,
        )

    @staticmethod
    def _mean_var_ratio(X: jnp.ndarray, axis: int, clip_high: float = 1e6) -> jnp.ndarray:
        """
        scHPF empirical hyperparameter: E[X]/Var[X] per row (axis=1) or col (axis=0).
        Clipped to avoid extreme ratios for near-zero variance entries.
        """
        mean = jnp.mean(X, axis=axis)
        var = jnp.var(X, axis=axis, ddof=0)
        var = jnp.maximum(var, 1e-10)  # avoid div by zero
        ratio = mean / var
        return jnp.clip(ratio, 1e-6, clip_high)



    # =====================================================================
    # Initialization — scHPF-style random initialization (Levitin et al. 2019)
    # =====================================================================

    def _initialize_parameters(self, X: jnp.ndarray, y: jnp.ndarray, X_aux: jnp.ndarray):
        """
        Initialize variational parameters using scHPF patterns:
        1. Empirical hyperparameters bp/dp from mean/var ratio
        2. Random init: uniform(0.5*prior, 1.5*prior) for shape AND rate
        3. First iteration flag for random Dirichlet phi
        """
        self.n, self.p = X.shape
        self.kappa = 1 if y.ndim == 1 else y.shape[1]
        self.p_aux = X_aux.shape[1] if X_aux is not None and X_aux.size > 0 else 0

        # ---- Empirical hyperparameters (scHPF pattern) ----
        # bp[i] = ap * mean_i / var_i  (cell-specific rate prior)
        # dp[j] = cp * mean_j / var_j  (gene-specific rate prior)
        bp = self.alpha_xi * self._mean_var_ratio(X, axis=1)  # (n,)
        dp = self.alpha_eta * self._mean_var_ratio(X, axis=0)  # (p,)
        
        # CRITICAL: Floor dp/bp to prevent extreme E_beta/E_theta initialization
        # Without this, genes with tiny mean/var ratio get dp~1e-6 → E_beta~millions
        dp_min = 0.1  # Reasonable minimum rate
        bp_min = 0.1
        dp = jnp.maximum(dp, dp_min)
        bp = jnp.maximum(bp, bp_min)
        
        # Clip if bp >> dp (scHPF does this to prevent instability)
        dp_median = jnp.median(dp)
        bp_median = jnp.median(bp)
        if bp_median > 1000 * dp_median:
            dp = dp * (bp_median / (1000 * dp_median))
            print(f"Warning: scaled dp by factor {bp_median / (1000 * dp_median):.2f}")
        
        # Store for consistent use in updates AND ELBO
        self.bp_empirical = bp
        self.dp_empirical = dp
        # Override lambda_xi/lambda_eta so ELBO is consistent with updates
        self.lambda_xi = float(jnp.median(bp))
        self.lambda_eta = float(jnp.median(dp))

        key_b1, key_b2, key_t1, key_t2, key_xi, key_eta, self.rng_key = random.split(self.rng_key, 7)

        # ---- Beta initialization ----
        # scHPF pattern: shape ~ U(0.5*alpha, 1.5*alpha), rate ~ U(0.5*dp, 1.5*dp)
        if self.mode == 'combined':
            pathway_mask_T = self.pathway_mask.T
            mask_jnp = jnp.array(pathway_mask_T, dtype=jnp.float32)
            
            # Random per-factor initialization (scHPF style)
            a_beta_init = random.uniform(key_b1, (self.p, self.d), 
                                         minval=0.5 * self.alpha_beta, 
                                         maxval=1.5 * self.alpha_beta)
            b_beta_init = random.uniform(key_b2, (self.p, self.d), 
                                         minval=0.5 * dp[:, jnp.newaxis], 
                                         maxval=1.5 * dp[:, jnp.newaxis])
            
            # Pathway factors: enforce mask
            for k in range(self.n_pathway_factors):
                col_mask = mask_jnp[:, k]
                a_beta_init = a_beta_init.at[:, k].set(
                    jnp.where(col_mask > 0.5, a_beta_init[:, k], self.alpha_beta * 0.01))
                b_beta_init = b_beta_init.at[:, k].set(
                    jnp.where(col_mask > 0.5, b_beta_init[:, k], 10.0))
            
            self.a_beta = a_beta_init
            self.b_beta = b_beta_init
            self.beta_mask = mask_jnp[:, :self.n_pathway_factors]

        elif self.mode in ['masked', 'pathway_init']:
            pathway_mask_T = self.pathway_mask.T
            mask_jnp = jnp.array(pathway_mask_T, dtype=jnp.float32)
            
            if self.mode == 'masked':
                # Random per-factor init, masked off outside pathway
                a_beta_init = random.uniform(key_b1, (self.p, self.d),
                                             minval=0.5 * self.alpha_beta,
                                             maxval=1.5 * self.alpha_beta)
                b_beta_init = random.uniform(key_b2, (self.p, self.d),
                                             minval=0.5 * dp[:, jnp.newaxis],
                                             maxval=1.5 * dp[:, jnp.newaxis])
                self.a_beta = jnp.where(mask_jnp > 0.5, a_beta_init, self.alpha_beta * 0.01)
                self.b_beta = jnp.where(mask_jnp > 0.5, b_beta_init, 10.0)
                self.beta_mask = mask_jnp
            else:
                # pathway_init: random everywhere, boosted on pathway
                a_beta_init = random.uniform(key_b1, (self.p, self.d),
                                             minval=0.5 * self.alpha_beta,
                                             maxval=1.5 * self.alpha_beta)
                b_beta_init = random.uniform(key_b2, (self.p, self.d),
                                             minval=0.5 * dp[:, jnp.newaxis],
                                             maxval=1.5 * dp[:, jnp.newaxis])
                boost = random.uniform(key_t1, (self.p, self.d), minval=1.0, maxval=2.0)
                self.a_beta = jnp.where(mask_jnp > 0.5, a_beta_init * boost, a_beta_init)
                self.b_beta = b_beta_init
                self.beta_mask = None
        else:
            # Unmasked: scHPF-inspired init with controlled variance
            # Shape: random around prior
            self.a_beta = random.uniform(key_b1, (self.p, self.d),
                                         minval=0.5 * self.alpha_beta,
                                         maxval=1.5 * self.alpha_beta)
            # Rate: start at 1.0 with small noise (NOT using tiny dp directly)
            # This ensures E_beta = a/b ~ alpha_beta, reasonable magnitude
            rate_noise = random.uniform(key_b2, (self.p, self.d), minval=0.8, maxval=1.2)
            self.b_beta = rate_noise  # Will converge to proper values via updates
            self.beta_mask = None

        self.a_beta = jnp.maximum(self.a_beta, self.alpha_beta * 0.01)

        # ---- Eta (scHPF: shape random, rate = dp) ----
        self.a_eta = random.uniform(key_eta, (self.p,),
                                    minval=0.5 * self.alpha_eta,
                                    maxval=1.5 * self.alpha_eta)
        # Rate initialized to empirical dp (NOT randomized - scHPF uses dp directly)
        self.b_eta = dp

        # ---- Theta: random per cell per factor ----
        # Use controlled initialization - rate at 1.0 with noise
        self.a_theta = random.uniform(key_t1, (self.n, self.d),
                                      minval=0.5 * self.alpha_theta,
                                      maxval=1.5 * self.alpha_theta)
        # Rate initialized to ~1.0 with noise (will converge to proper values)
        theta_rate_noise = random.uniform(key_t2, (self.n, self.d), minval=0.8, maxval=1.2)
        self.b_theta = theta_rate_noise

        # ---- Xi (shape random, rate = bp empirical) ----
        self.a_xi = random.uniform(key_xi, (self.n,),
                                   minval=0.5 * self.alpha_xi,
                                   maxval=1.5 * self.alpha_xi)
        self.b_xi = bp  # Use empirical bp as rate

        # ---- v — regression weights (small random init) ----
        key_v1, self.rng_key = random.split(self.rng_key)
        self.mu_v = 0.1 * random.normal(key_v1, (self.kappa, self.d))  # smaller init
        self.sigma_v_diag = jnp.full((self.kappa, self.d), self.sigma_v ** 2)

        # ---- gamma — auxiliary covariate weights ----
        if self.p_aux > 0:
            self.mu_gamma = jnp.zeros((self.kappa, self.p_aux))
            self.Sigma_gamma = jnp.array([jnp.eye(self.p_aux) * self.sigma_gamma ** 2 for _ in range(self.kappa)])
        else:
            self.mu_gamma = jnp.zeros((self.kappa, 0))
            self.Sigma_gamma = jnp.zeros((self.kappa, 0, 0))

        # ---- Spike-and-slab ----
        if self.use_spike_slab:
            self.rho_beta = jnp.full((self.p, self.d), self.pi_beta)
            self.rho_v = jnp.full((self.kappa, self.d), self.pi_v)
        else:
            self.rho_beta = jnp.ones((self.p, self.d))
            self.rho_v = jnp.ones((self.kappa, self.d))

        # ---- JJ auxiliary ----
        self.zeta = jnp.ones((self.n, self.kappa))

        self._compute_expectations()
        self.initial_mu_v_ = self.mu_v.copy()
        print(f"Initial beta diversity: {np.std(np.array(self.E_beta), axis=1).mean():.4f}")
        print(f"Initial theta diversity: {np.std(np.array(self.a_theta / self.b_theta), axis=1).mean():.4f}")
        print(f"Initial E[beta] range: [{np.array(self.E_beta).min():.2f}, {np.array(self.E_beta).max():.2f}]")
        print(f"Initial v: {self.mu_v}")
        print(f"Empirical bp median: {np.median(np.array(bp)):.4f}, dp median: {np.median(np.array(dp)):.4f}")

    # =====================================================================
    # Expected sufficient statistics
    # =====================================================================

    def _compute_expectations(self):
        self.E_beta = self.a_beta / self.b_beta
        self.E_log_beta = jsp.digamma(self.a_beta) - jnp.log(self.b_beta)
        self.E_eta = self.a_eta / self.b_eta
        self.E_log_eta = jsp.digamma(self.a_eta) - jnp.log(self.b_eta)
        self.E_v = self.mu_v.copy()
        self.E_gamma = self.mu_gamma.copy()
        if self.use_spike_slab:
            self.E_beta_effective = self.rho_beta * self.E_beta
            self.E_log_beta_effective = self.rho_beta * self.E_log_beta + (1 - self.rho_beta) * (-20)
            self.E_v_effective = self.rho_v * self.E_v
        else:
            self.E_beta_effective = self.E_beta
            self.E_log_beta_effective = self.E_log_beta
            self.E_v_effective = self.E_v

    # =====================================================================
    # Cell batch size calculation
    # =====================================================================

    def _cell_batch_size(self, target_gb: float = 2.0) -> int:
        """Max cells per chunk so the (B, p, d) phi tensor stays under target_gb."""
        bytes_per_cell = self.p * self.d * 4 * 3  # phi + E_z + log_phi
        target_bytes = target_gb * 1024**3
        bs = max(64, int(target_bytes / bytes_per_cell))
        return min(bs, self.n)  # never exceed n

    # =====================================================================
    # CAVI Update: φ → θ → z_sum  (fused, per cell-batch)
    # =====================================================================

    def _update_theta_and_accumulate(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        X_aux: jnp.ndarray,
        s: int,
        e: int,
        damping: float,
        z_sum_theta_batch: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        For cells [s, e):
          If z_sum_theta_batch is provided (scHPF order): 
              use it for theta shape, skip phi computation
          Otherwise (legacy order): 
              compute phi, z_sum, and update theta

        Returns z_sum contribution (p, d) for beta.
        """
        B = e - s
        X_b = X[s:e]  # (B, p)
        y_b = y[s:e]  # (B, kappa)
        X_aux_b = X_aux[s:e] if X_aux.shape[1] > 0 else X_aux[s:e]

        if z_sum_theta_batch is not None:
            # scHPF order: phi/z_sum already computed, just update theta
            z_sum = jnp.zeros((self.p, self.d))  # dummy, not used
            a_theta_new = self.alpha_theta + z_sum_theta_batch  # (B, d)
        else:
            # Standard phi computation: softmax(E[log theta] + E[log beta])
            # The random initialization of theta/beta breaks symmetry naturally
            E_log_theta_b = jsp.digamma(self.a_theta[s:e]) - jnp.log(self.b_theta[s:e])  # (B, d)
            log_phi = E_log_theta_b[:, jnp.newaxis, :] + self.E_log_beta_effective[jnp.newaxis, :, :]
            log_phi = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
            phi = jnp.exp(log_phi)

            # --- z_sum for beta accumulation ---
            E_z = X_b[:, :, jnp.newaxis] * phi  # (B, p, d)
            z_sum = E_z.sum(axis=0)  # (p, d) for beta
            a_theta_new = self.alpha_theta + E_z.sum(axis=1)  # (B, d) for theta

        # --- theta rate (Poisson + JJ) ---
        E_xi_b = self.a_xi[s:e] / self.b_xi[s:e]
        b_theta_new = E_xi_b[:, jnp.newaxis] + self.E_beta_effective.sum(axis=0)[jnp.newaxis, :]

        E_theta_b = self.a_theta[s:e] / self.b_theta[s:e]
        y_expanded = y_b if y_b.ndim > 1 else y_b[:, jnp.newaxis]
        lam = self._lambda_jj(self.zeta[s:e])

        theta_v = E_theta_b @ self.E_v_effective.T
        if self.p_aux > 0:
            aux_term = X_aux_b @ self.E_gamma.T
        else:
            aux_term = 0.0

        full_lp = theta_v[:, :, jnp.newaxis]
        if self.p_aux > 0:
            full_lp = full_lp + aux_term[:, :, jnp.newaxis]

        C_minus_ell = full_lp - E_theta_b[:, jnp.newaxis, :] * self.E_v_effective[jnp.newaxis, :, :]
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag

        R = (
            -(y_expanded[:, :, jnp.newaxis] - 0.5) * self.E_v_effective[jnp.newaxis, :, :]
            + 2 * lam[:, :, jnp.newaxis] * self.E_v_effective[jnp.newaxis, :, :] * C_minus_ell
            + 2 * lam[:, :, jnp.newaxis] * E_v_sq[jnp.newaxis, :, :] * E_theta_b[:, jnp.newaxis, :]
        )
        R_sum = R.sum(axis=1)

        b_theta_new = b_theta_new + self.regression_weight * R_sum
        b_theta_new = jnp.maximum(b_theta_new, 1e-6)
        a_theta_new = jnp.maximum(a_theta_new, 1.001)

        self.a_theta = self.a_theta.at[s:e].set(
            (1 - damping) * self.a_theta[s:e] + damping * a_theta_new)
        self.b_theta = self.b_theta.at[s:e].set(
            (1 - damping) * self.b_theta[s:e] + damping * b_theta_new)

        return z_sum

    # =====================================================================
    # CAVI: Compute z_sum without updating theta (for scHPF order)
    # =====================================================================

    def _compute_z_sum_batch(
        self,
        X: jnp.ndarray,
        s: int,
        e: int,
        diversity_noise: float = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute phi and z_sums for cells [s, e) WITHOUT updating theta.
        Used for scHPF-style update order (beta before theta).

        Args:
            diversity_noise: Scale of random noise added to E_log_beta per factor

        Returns:
            z_sum_beta: (p, d) contribution for beta shape update
            z_sum_theta: (B, d) contribution for theta shape update
        """
        B = e - s
        X_b = X[s:e]  # (B, p)

        # Standard phi computation: softmax(E[log theta] + E[log beta])
        E_log_theta_b = jsp.digamma(self.a_theta[s:e]) - jnp.log(self.b_theta[s:e])  # (B, d)
        E_log_beta_effective = self.E_log_beta_effective
        
        # Add diversity noise if requested
        if diversity_noise > 0:
            key_noise, self.rng_key = random.split(self.rng_key)
            factor_noise = diversity_noise * random.normal(key_noise, (self.d,))
            E_log_beta_effective = E_log_beta_effective + factor_noise[jnp.newaxis, :]
        
        log_phi = E_log_theta_b[:, jnp.newaxis, :] + E_log_beta_effective[jnp.newaxis, :, :]
        log_phi = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
        phi = jnp.exp(log_phi)

        # --- z_sums ---
        E_z = X_b[:, :, jnp.newaxis] * phi  # (B, p, d)
        z_sum_beta = E_z.sum(axis=0)  # (p, d) for beta
        z_sum_theta = E_z.sum(axis=1)  # (B, d) for theta

        return z_sum_beta, z_sum_theta

    # =====================================================================
    # CAVI: Sparse-efficient z_sum (scHPF-style, O(nnz*d) memory)
    # =====================================================================

    def _compute_z_sum_sparse(self, diversity_noise: float = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute z_sums using only nonzero entries (scHPF pattern).
        
        Memory: O(nnz * d) instead of O(n * p * d)
        
        Args:
            diversity_noise: Scale of random noise added to E_log_beta per factor
                             to prevent CLT-induced factor collapse. Decays over iterations.
        
        Returns:
            z_sum_beta: (p, d) for beta shape update
            z_sum_theta: (n, d) for theta shape update
        """
        # Precompute E[log θ] and E[log β] once (scHPF optimization)
        E_log_theta = jsp.digamma(self.a_theta) - jnp.log(self.b_theta)  # (n, d)
        E_log_beta = self.E_log_beta_effective  # (p, d)
        
        # Add diversity noise to break CLT symmetry during early iterations
        # This prevents all factors from collapsing to identical values
        if diversity_noise > 0:
            key_noise, self.rng_key = random.split(self.rng_key)
            # Per-factor noise (same noise for all genes/cells within a factor)
            # This creates "specialization" pressure for each factor
            factor_noise = diversity_noise * random.normal(key_noise, (self.d,))
            E_log_beta = E_log_beta + factor_noise[jnp.newaxis, :]
        
        # Gather E_log for each nonzero entry
        E_log_theta_nnz = E_log_theta[self._X_row]  # (nnz, d)
        E_log_beta_nnz = E_log_beta[self._X_col]    # (nnz, d)
        
        # log_phi for each nonzero: (nnz, d)
        log_phi = E_log_theta_nnz + E_log_beta_nnz
        log_phi = log_phi - logsumexp(log_phi, axis=1, keepdims=True)
        phi = jnp.exp(log_phi)
        
        # Xphi: x_ij * phi_ij for each nonzero (nnz, d)
        Xphi = self._X_data[:, jnp.newaxis] * phi
        
        # Accumulate z_sum_beta[j, k] = sum over nonzeros with col=j
        # Using segment_sum (equivalent to scatter_add)
        z_sum_beta = jnp.zeros((self.p, self.d))
        z_sum_beta = z_sum_beta.at[self._X_col].add(Xphi)
        
        # Accumulate z_sum_theta[i, k] = sum over nonzeros with row=i
        z_sum_theta = jnp.zeros((self.n, self.d))
        z_sum_theta = z_sum_theta.at[self._X_row].add(Xphi)
        
        return z_sum_beta, z_sum_theta

    # =====================================================================
    # CAVI Update: ξ  (hierarchical depth prior)
    # =====================================================================
    # CAVI Update: θ sparse (all cells at once, using precomputed z_sum_theta)
    # =====================================================================

    def _update_theta_sparse(
        self,
        y: jnp.ndarray,
        X_aux: jnp.ndarray,
        z_sum_theta: jnp.ndarray,
        damping: float = 1.0,
    ):
        """
        Update all θ at once using sparse-computed z_sum_theta.
        
        Args:
            z_sum_theta: (n, d) from _compute_z_sum_sparse
        """
        # --- theta shape ---
        a_theta_new = self.alpha_theta + z_sum_theta  # (n, d)

        # --- theta rate (Poisson + JJ regression correction) ---
        E_xi = self.a_xi / self.b_xi  # (n,)
        b_theta_new = E_xi[:, jnp.newaxis] + self.E_beta_effective.sum(axis=0)[jnp.newaxis, :]

        # JJ regression correction (if regression_weight > 0)
        if self.regression_weight > 0:
            E_theta = self.a_theta / self.b_theta
            y_expanded = y if y.ndim > 1 else y[:, jnp.newaxis]
            lam = self._lambda_jj(self.zeta)

            theta_v = E_theta @ self.E_v_effective.T  # (n, kappa)
            if self.p_aux > 0:
                aux_term = X_aux @ self.E_gamma.T
            else:
                aux_term = 0.0

            # full_lp: (n, kappa, d)
            full_lp = theta_v[:, :, jnp.newaxis]
            if self.p_aux > 0:
                full_lp = full_lp + aux_term[:, :, jnp.newaxis]

            # C^{(-ℓ)} = full_lp - θ_iℓ * v_kℓ
            C_minus_ell = full_lp - E_theta[:, jnp.newaxis, :] * self.E_v_effective[jnp.newaxis, :, :]
            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag

            # Regression rate correction R
            R = (
                -(y_expanded[:, :, jnp.newaxis] - 0.5) * self.E_v_effective[jnp.newaxis, :, :]
                + 2 * lam[:, :, jnp.newaxis] * self.E_v_effective[jnp.newaxis, :, :] * C_minus_ell
                + 2 * lam[:, :, jnp.newaxis] * E_v_sq[jnp.newaxis, :, :] * E_theta[:, jnp.newaxis, :]
            )
            R_sum = R.sum(axis=1)  # (n, d)
            b_theta_new = b_theta_new + self.regression_weight * R_sum

        b_theta_new = jnp.maximum(b_theta_new, 1e-6)
        a_theta_new = jnp.maximum(a_theta_new, 1.001)

        # Damped update
        self.a_theta = (1 - damping) * self.a_theta + damping * a_theta_new
        self.b_theta = (1 - damping) * self.b_theta + damping * b_theta_new

    # =====================================================================
    # CAVI Update: ξ  (hierarchical depth prior)
    # =====================================================================

    def _update_xi(self, damping: float = 1.0):
        """ξ rate = bp_empirical + sum_k E[θ_ik] (scHPF pattern)"""
        E_theta = self.a_theta / self.b_theta
        a_xi_new = jnp.full(self.n, self.alpha_xi + self.d * self.alpha_theta)
        b_xi_new = self.bp_empirical + E_theta.sum(axis=1)  # Use empirical bp
        self.a_xi = (1 - damping) * self.a_xi + damping * a_xi_new
        self.b_xi = (1 - damping) * self.b_xi + damping * b_xi_new

    # =====================================================================
    # CAVI Update: β  (gene loadings)
    # =====================================================================

    def _update_beta(self, z_sum: jnp.ndarray, damping: float = 1.0):
        """
        β_{jℓ} ~ Gamma(a_β, η_j)

        Shape: a_β_jℓ = α_β + Σ_i x_{ij} φ_{ijℓ}   (z_sum already accumulated)
        Rate:  b_β_jℓ = E[η_j] + Σ_i E[θ_{iℓ}]
        """
        E_theta = self.a_theta / self.b_theta
        theta_sum = E_theta.sum(axis=0)  # (d,)

        a_beta_new = self.alpha_beta + z_sum
        b_beta_new = self.E_eta[:, jnp.newaxis] + theta_sum[jnp.newaxis, :]

        a_beta_new = jnp.maximum(a_beta_new, self.alpha_beta * 0.01)

        # Damped update
        self.a_beta = (1 - damping) * self.a_beta + damping * a_beta_new
        self.b_beta = (1 - damping) * self.b_beta + damping * b_beta_new

        # Enforce pathway mask
        self._enforce_beta_mask()

    def _enforce_beta_mask(self):
        small_a = self.alpha_beta * 0.01
        large_b = 10.0
        if self.mode == 'masked' and self.beta_mask is not None:
            self.a_beta = jnp.where(self.beta_mask > 0.5, self.a_beta, small_a)
            self.b_beta = jnp.where(self.beta_mask > 0.5, self.b_beta, large_b)
        elif self.mode == 'combined' and self.beta_mask is not None:
            pathway_a = jnp.where(self.beta_mask > 0.5,
                                  self.a_beta[:, :self.n_pathway_factors], small_a)
            pathway_b = jnp.where(self.beta_mask > 0.5,
                                  self.b_beta[:, :self.n_pathway_factors], large_b)
            self.a_beta = self.a_beta.at[:, :self.n_pathway_factors].set(pathway_a)
            self.b_beta = self.b_beta.at[:, :self.n_pathway_factors].set(pathway_b)

    # =====================================================================
    # CAVI Update: η  (gene-level hierarchical rate)
    # =====================================================================

    def _update_eta(self, damping: float = 1.0):
        """η rate = dp_empirical + sum_k E[β_jk] (scHPF pattern)"""
        beta_sum = self.E_beta_effective.sum(axis=1)
        a_eta_new = jnp.full(self.p, self.alpha_eta + self.d * self.alpha_beta)
        b_eta_new = self.dp_empirical + beta_sum  # Use empirical dp
        self.a_eta = (1 - damping) * self.a_eta + damping * a_eta_new
        self.b_eta = (1 - damping) * self.b_eta + damping * b_eta_new

    # =====================================================================
    # CAVI Update: ζ  (JJ auxiliary)
    # =====================================================================

    def _update_zeta(self, X_aux: jnp.ndarray):
        E_theta = self.a_theta / self.b_theta
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag
        Var_theta = self.a_theta / (self.b_theta ** 2)

        if self.p_aux > 0:
            aux_contrib = X_aux @ self.E_gamma.T
        else:
            aux_contrib = 0.0

        E_A = E_theta @ self.E_v_effective.T + aux_contrib
        E_A_sq = E_A ** 2 + Var_theta @ E_v_sq.T
        self.zeta = jnp.sqrt(jnp.maximum(E_A_sq, 1e-8))

    # =====================================================================
    # CAVI Update: v  (regression weights — Gaussian posterior)
    # =====================================================================

    def _update_v(self, y: jnp.ndarray, X_aux: jnp.ndarray, damping: float = 1.0):
        """
        Diagonal posterior: v_{kℓ} ~ N(μ, σ²)
        Precision: 1/σ_v² + 2 Σ_i λ(ζ_{ik}) E[θ²_{iℓ}]
        Mean·prec: Σ_i [(y_{ik}-0.5) - 2λ(ζ_{ik}) C^{(-ℓ)}_{ik}] E[θ_{iℓ}]
        """
        y_expanded = y if y.ndim > 1 else y[:, jnp.newaxis]
        lam = self._lambda_jj(self.zeta)
        E_theta = self.a_theta / self.b_theta
        Var_theta = self.a_theta / (self.b_theta ** 2)
        E_theta_sq = E_theta ** 2 + Var_theta

        if self.p_aux > 0:
            aux_contrib = X_aux @ self.E_gamma.T
        else:
            aux_contrib = 0.0
        full_theta_v = E_theta @ self.E_v_effective.T

        # Precision  (kappa, d)
        precision = jnp.full((self.kappa, self.d), 1.0 / self.sigma_v ** 2)
        precision = precision + 2 * jnp.einsum('ik,id->kd', lam, E_theta_sq)

        # C^{(-ℓ)}
        full_lp = full_theta_v[:, :, jnp.newaxis]
        if self.p_aux > 0:
            full_lp = full_lp + aux_contrib[:, :, jnp.newaxis]
        C_minus_ell = full_lp - E_theta[:, jnp.newaxis, :] * self.E_v_effective[jnp.newaxis, :, :]

        term1 = jnp.einsum('ik,id->kd', y_expanded - 0.5, E_theta)
        term2 = 2 * jnp.einsum('ik,ikd,id->kd', lam, C_minus_ell, E_theta)
        mean_contrib = term1 - term2

        mu_v_new = jnp.clip(mean_contrib / precision, -5, 5)
        sigma_v_diag_new = 1.0 / precision          # (kappa, d)

        self.mu_v = (1 - damping) * self.mu_v + damping * mu_v_new
        self.sigma_v_diag = (1 - damping) * self.sigma_v_diag + damping * sigma_v_diag_new

    # =====================================================================
    # CAVI Update: γ  (auxiliary covariate weights)
    # =====================================================================

    def _update_gamma(self, y: jnp.ndarray, X_aux: jnp.ndarray, damping: float = 1.0):
        if self.p_aux == 0:
            return
        y_expanded = y if y.ndim > 1 else y[:, jnp.newaxis]
        lam = self._lambda_jj(self.zeta)
        E_theta = self.a_theta / self.b_theta

        mu_gamma_new = jnp.zeros((self.kappa, self.p_aux))
        Sigma_gamma_new = jnp.zeros((self.kappa, self.p_aux, self.p_aux))

        for k in range(self.kappa):
            prec_prior = jnp.eye(self.p_aux) / self.sigma_gamma ** 2
            prec_lik = 2 * (X_aux.T * lam[:, k]) @ X_aux
            prec_hat = prec_prior + prec_lik

            theta_v = E_theta @ self.E_v_effective[k]
            mean_contrib = X_aux.T @ (y_expanded[:, k] - 0.5 - 2 * lam[:, k] * theta_v)

            sigma_k = jnp.linalg.inv(prec_hat + 1e-6 * jnp.eye(self.p_aux))
            mu_k = sigma_k @ mean_contrib

            mu_gamma_new = mu_gamma_new.at[k].set(jnp.clip(mu_k, -10, 10))
            sigma_k = 0.5 * (sigma_k + sigma_k.T)
            Sigma_gamma_new = Sigma_gamma_new.at[k].set(sigma_k)

        self.mu_gamma = (1 - damping) * self.mu_gamma + damping * mu_gamma_new
        self.Sigma_gamma = (1 - damping) * self.Sigma_gamma + damping * Sigma_gamma_new

    # =====================================================================
    # Spike-and-slab update (mirrors SVICorrected)
    # =====================================================================

    def _update_spike_slab(self):
        if not self.use_spike_slab:
            return
        log_prior_odds_beta = jnp.log(self.pi_beta / (1 - self.pi_beta + 1e-10))
        log_prior_odds_v = jnp.log(self.pi_v / (1 - self.pi_v + 1e-10))

        log_lik_ratio_beta = jnp.log(self.E_beta + 1e-10) - jnp.log(0.01)
        self.rho_beta = jsp.expit(jnp.clip(log_prior_odds_beta + log_lik_ratio_beta, -20, 20))

        v_snr = jnp.abs(self.mu_v) / (jnp.sqrt(self.sigma_v_diag) + 1e-10)
        log_lik_ratio_v = jnp.log1p(v_snr)
        self.rho_v = jsp.expit(jnp.clip(log_prior_odds_v + log_lik_ratio_v, -10, 10))
        self.rho_v = jnp.maximum(self.rho_v, 0.1)

    # =====================================================================
    # ELBO
    # =====================================================================

    def _compute_elbo(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        X_aux: jnp.ndarray,
        cell_bs: int = 1000,
    ) -> Tuple[float, float, float]:
        """
        Full-dataset ELBO, chunked over cells to avoid (n, p, d) OOM.
        """
        E_theta = self.a_theta / self.b_theta
        E_log_theta = jsp.digamma(self.a_theta) - jnp.log(self.b_theta)
        E_xi = self.a_xi / self.b_xi
        E_log_xi = jsp.digamma(self.a_xi) - jnp.log(self.b_xi)

        # --- Poisson (chunked over cells) ---
        elbo_x = 0.0
        for s in range(0, self.n, cell_bs):
            e = min(s + cell_bs, self.n)
            log_rates = E_log_theta[s:e, jnp.newaxis, :] + self.E_log_beta_effective[jnp.newaxis, :, :]
            log_sum_rates = logsumexp(log_rates, axis=2)
            elbo_x += float(jnp.sum(X[s:e] * log_sum_rates))
            elbo_x -= float(jnp.sum(jsp.gammaln(X[s:e] + 1)))
        elbo_x -= float(jnp.sum(E_theta.sum(axis=0) * self.E_beta_effective.sum(axis=0)))
        elbo = elbo_x

        # --- Bernoulli (JJ) ---
        y_expanded = y if y.ndim > 1 else y[:, jnp.newaxis]
        lam = self._lambda_jj(self.zeta)
        if self.p_aux > 0:
            aux_contrib = X_aux @ self.E_gamma.T
        else:
            aux_contrib = 0.0
        E_A = E_theta @ self.E_v_effective.T + aux_contrib
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag
        Var_theta = self.a_theta / (self.b_theta ** 2)
        E_A_sq = E_A ** 2 + Var_theta @ E_v_sq.T

        elbo_y = jnp.sum((y_expanded - 0.5) * E_A - lam * E_A_sq)
        elbo_y += jnp.sum(lam * self.zeta ** 2 - 0.5 * self.zeta
                          - jnp.log1p(jnp.exp(jnp.clip(self.zeta, -500, 500))))
        elbo += self.regression_weight * elbo_y

        # --- Priors (local) ---
        elbo += jnp.sum((self.alpha_theta - 1) * E_log_theta
                        + self.alpha_theta * E_log_xi[:, jnp.newaxis]
                        - E_xi[:, jnp.newaxis] * E_theta)
        elbo -= self.n * self.d * jsp.gammaln(self.alpha_theta)
        elbo += jnp.sum((self.alpha_xi - 1) * E_log_xi - self.lambda_xi * E_xi)
        elbo += self.n * (self.alpha_xi * jnp.log(self.lambda_xi) - jsp.gammaln(self.alpha_xi))

        # --- Priors (global) ---
        elbo += jnp.sum((self.alpha_beta - 1) * self.E_log_beta
                        + self.alpha_beta * self.E_log_eta[:, jnp.newaxis]
                        - self.E_eta[:, jnp.newaxis] * self.E_beta)
        elbo -= self.p * self.d * jsp.gammaln(self.alpha_beta)
        elbo += jnp.sum((self.alpha_eta - 1) * self.E_log_eta - self.lambda_eta * self.E_eta)
        elbo += self.p * (self.alpha_eta * jnp.log(self.lambda_eta) - jsp.gammaln(self.alpha_eta))

        for k in range(self.kappa):
            elbo -= 0.5 * self.d * jnp.log(2 * jnp.pi * self.sigma_v ** 2)
            elbo -= 0.5 / self.sigma_v ** 2 * (jnp.sum(self.mu_v[k] ** 2) + jnp.sum(self.sigma_v_diag[k]))
        if self.p_aux > 0:
            for k in range(self.kappa):
                elbo -= 0.5 * self.p_aux * jnp.log(2 * jnp.pi * self.sigma_gamma ** 2)
                elbo -= 0.5 / self.sigma_gamma ** 2 * (jnp.sum(self.mu_gamma[k] ** 2) + jnp.trace(self.Sigma_gamma[k]))
        # --- Entropy ---
        elbo += jnp.sum(self.a_theta - jnp.log(self.b_theta) + jsp.gammaln(self.a_theta)
                        + (1 - self.a_theta) * jsp.digamma(self.a_theta))
        elbo += jnp.sum(self.a_xi - jnp.log(self.b_xi) + jsp.gammaln(self.a_xi)
                        + (1 - self.a_xi) * jsp.digamma(self.a_xi))
        elbo += jnp.sum(self.a_beta - jnp.log(self.b_beta) + jsp.gammaln(self.a_beta)
                        + (1 - self.a_beta) * jsp.digamma(self.a_beta))
        elbo += jnp.sum(self.a_eta - jnp.log(self.b_eta) + jsp.gammaln(self.a_eta)
                        + (1 - self.a_eta) * jsp.digamma(self.a_eta))
        for k in range(self.kappa):
            logdet = jnp.sum(jnp.log(jnp.maximum(self.sigma_v_diag[k], 1e-30)))
            elbo += 0.5 * self.d * (1 + jnp.log(2 * jnp.pi)) + 0.5 * logdet
        if self.p_aux > 0:
            for k in range(self.kappa):
                sign, logdet = jnp.linalg.slogdet(self.Sigma_gamma[k])
                elbo += jnp.where(sign > 0, 0.5 * self.p_aux * (1 + jnp.log(2 * jnp.pi)) + 0.5 * logdet, 0.0)

        poisson_ll = float(elbo_x)
        regression_ll = float(elbo_y)
        return float(elbo), poisson_ll, regression_ll

    # =====================================================================
    # Checkpoint / restore (same interface as SVICorrected)
    # =====================================================================

    def _checkpoint_params(self) -> Dict[str, Any]:
        cp = {
            'a_beta': np.array(self.a_beta), 'b_beta': np.array(self.b_beta),
            'a_eta': np.array(self.a_eta), 'b_eta': np.array(self.b_eta),
            'mu_v': np.array(self.mu_v), 'sigma_v_diag': np.array(self.sigma_v_diag),
            'mu_gamma': np.array(self.mu_gamma), 'Sigma_gamma': np.array(self.Sigma_gamma),
            'a_theta': np.array(self.a_theta), 'b_theta': np.array(self.b_theta),
            'a_xi': np.array(self.a_xi), 'b_xi': np.array(self.b_xi),
        }
        if self.use_spike_slab:
            cp['rho_beta'] = np.array(self.rho_beta)
            cp['rho_v'] = np.array(self.rho_v)
        return cp

    def _restore_params(self, cp: Dict[str, Any]):
        for k in ['a_beta', 'b_beta', 'a_eta', 'b_eta', 'mu_v', 'sigma_v_diag',
                   'mu_gamma', 'Sigma_gamma', 'a_theta', 'b_theta', 'a_xi', 'b_xi']:
            setattr(self, k, jnp.array(cp[k]))
        if self.use_spike_slab and 'rho_beta' in cp:
            self.rho_beta = jnp.array(cp['rho_beta'])
            self.rho_v = jnp.array(cp['rho_v'])
        self._compute_expectations()

    # =====================================================================
    # FIT  — main CAVI loop
    # =====================================================================

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_aux: np.ndarray,
        max_iter: int = 200,
        tol: float = 10.0,
        rel_tol: float = 2e-4,
        elbo_freq: int = 10,
        min_iter: int = 50,
        patience: int = 5,
        theta_damping: float = 0.8,
        beta_damping: float = 0.8,
        v_damping: float = 0.1,
        gamma_damping: float = 0.1,
        xi_damping: float = 0.9,
        eta_damping: float = 0.9,
        adaptive_damping: bool = True,
        v_warmup: int = 50,
        v_anneal: int = 50,
        v_update_freq: int = 5,
        verbose: bool = True,
        debug: bool = False,
        # Held-out data (optional, for heldout_ll tracking)
        X_heldout: Optional[np.ndarray] = None,
        y_heldout: Optional[np.ndarray] = None,
        X_aux_heldout: Optional[np.ndarray] = None,
        heldout_freq: int = 5,
        # Early stopping / held-out-LL (for API compat with SVI)
        early_stopping: bool = False,
        use_sparse: bool = True,  # Use sparse-efficient computation (scHPF-style)
        **_ignored,
    ):
        """
        Fit the model via Coordinate Ascent VI (full-batch).

        Parameters mirror VIConfig.training_params(); see config.py for docs.
        """
        # ---- Store COO sparse format for efficient iteration (scHPF pattern) ----
        if sp.issparse(X):
            X_coo = X.tocoo()
        else:
            X_coo = sp.coo_matrix(X)
        
        self._X_data = jnp.array(X_coo.data, dtype=jnp.float32)
        self._X_row = jnp.array(X_coo.row, dtype=jnp.int32)
        self._X_col = jnp.array(X_coo.col, dtype=jnp.int32)
        self._nnz = len(X_coo.data)
        
        # Also keep dense for regression updates (needs full rows)
        if sp.issparse(X):
            X = jnp.array(X.toarray())
        else:
            X = jnp.array(X)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y = jnp.array(y)
        X_aux = jnp.array(X_aux) if X_aux is not None else jnp.zeros((X.shape[0], 0))

        self._initialize_parameters(X, y, X_aux)

        self.elbo_history_ = []
        self.poisson_ll_history_ = []
        self.regression_ll_history_ = []
        self.heldout_loglik_history_ = []
        self.convergence_history_ = []

        best_elbo = -np.inf
        best_params = None
        no_improve = 0
        start_time = time.time()

        cell_bs = self._cell_batch_size()
        n_cell_batches = (self.n + cell_bs - 1) // cell_bs
        
        # Sparse mode stats
        sparsity = 1.0 - self._nnz / (self.n * self.p)
        dense_mem_gb = (self.n * self.p * self.d * 4) / 1e9
        sparse_mem_gb = (self._nnz * self.d * 4) / 1e9
        if verbose:
            print(f"Sparse mode: {use_sparse}  |  Sparsity: {sparsity:.1%}  |  nnz: {self._nnz:,}")
            print(f"Memory: dense φ would be {dense_mem_gb:.1f}GB, sparse Xφ is {sparse_mem_gb:.2f}GB")
            if not use_sparse:
                print(f"Cell batch size: {cell_bs}  ({n_cell_batches} batches)")

        target_rw = self.regression_weight
        anneal_end = v_warmup + v_anneal
        if verbose:
            print(f"v_warmup={v_warmup} iters (pure Poisson), "
                  f"v_anneal={v_anneal} iters (ramp regression_weight 0→{target_rw})")

        # Diversity noise schedule: prevents CLT-induced factor collapse
        # Start high (2.0), decay exponentially over diversity_warmup iterations
        diversity_noise_init = 2.0  # Initial noise scale
        diversity_warmup = 100     # Iterations to decay noise to near-zero
        diversity_decay = 0.95     # Exponential decay rate
        if verbose:
            print(f"Diversity noise: init={diversity_noise_init}, decay={diversity_decay}, warmup={diversity_warmup} iters")

        for t in range(max_iter):
            # ----------------------------------------------------------
            # Phase logic:
            #   t < v_warmup         → regression_weight = 0 (pure Poisson)
            #   v_warmup <= t < end  → linear ramp 0 → target_rw
            #   t >= end             → full target_rw
            # ----------------------------------------------------------
            in_warmup = t < v_warmup
            in_anneal = v_warmup <= t < anneal_end

            if in_warmup:
                self.regression_weight = 0.0
            elif in_anneal:
                frac = (t - v_warmup + 1) / v_anneal
                self.regression_weight = target_rw * frac
            else:
                self.regression_weight = target_rw

            # 1. ζ — make JJ bound tight BEFORE θ uses it
            if not in_warmup:
                self._update_zeta(X_aux)

            # Compute diversity noise for this iteration (exponential decay)
            diversity_noise = diversity_noise_init * (diversity_decay ** t)
            if t >= diversity_warmup:
                diversity_noise = 0.0  # Turn off after warmup

            # === scHPF-style update order: β before θ ===
            if use_sparse:
                # Sparse-efficient: O(nnz * d) memory instead of O(n * p * d)
                z_sum_beta, z_sum_theta = self._compute_z_sum_sparse(diversity_noise=diversity_noise)
                z_sum_global = z_sum_beta
            else:
                # Dense batched: fallback for debugging
                z_sum_global = jnp.zeros((self.p, self.d))
                z_sum_theta_list = []
                for cb in range(n_cell_batches):
                    s = cb * cell_bs
                    e = min(s + cell_bs, self.n)
                    z_sum_beta_batch, z_sum_theta_batch = self._compute_z_sum_batch(
                        X, s, e, diversity_noise=diversity_noise)
                    z_sum_global = z_sum_global + z_sum_beta_batch
                    z_sum_theta_list.append((s, e, z_sum_theta_batch))

            # 2b. β update (uses z_sums computed with CURRENT theta/beta)
            self._update_beta(z_sum_global, damping=beta_damping)

            # 2c. η update
            self._update_eta(damping=eta_damping)

            # 2d. Recompute expectations (now E_beta is updated with new values)
            self._compute_expectations()

            # 2e. Update θ (using z_sum_theta and NEW E_beta)
            if use_sparse:
                # Sparse: update all theta at once
                self._update_theta_sparse(y, X_aux, z_sum_theta, damping=theta_damping)
            else:
                # Dense batched: use stored z_sum_theta per batch
                for s, e, z_sum_theta_batch in z_sum_theta_list:
                    self._update_theta_and_accumulate(
                        X, y, X_aux, s, e, damping=theta_damping,
                        z_sum_theta_batch=z_sum_theta_batch)

            # 3. ξ
            self._update_xi(damping=xi_damping)

            # 4. Regression params (only after warmup, every v_update_freq iters)
            if not in_warmup and (t - v_warmup) % v_update_freq == 0:
                self._update_v(y, X_aux, damping=v_damping)
                self._update_gamma(y, X_aux, damping=gamma_damping)

            # 5. spike-and-slab
            self._update_spike_slab()

            # 6. Final expectation recompute (after all updates)
            self._compute_expectations()

            # ---- ELBO / convergence ----
            if t % elbo_freq == 0:
                elbo, poisson_ll, regression_ll = self._compute_elbo(X, y, X_aux, cell_bs=cell_bs)
                if np.isfinite(elbo):
                    self.elbo_history_.append((t, elbo))
                    self.poisson_ll_history_.append((t, poisson_ll))
                    self.regression_ll_history_.append((t, regression_ll))

                    if elbo > best_elbo + tol:
                        best_elbo = elbo
                        best_params = self._checkpoint_params()
                        no_improve = 0
                    else:
                        no_improve += 1

                    # Adaptive damping: increase damping if ELBO improving
                    if adaptive_damping and len(self.elbo_history_) >= 3:
                        prev_elbo = self.elbo_history_[-2][1]
                        if elbo > prev_elbo:
                            theta_damping = min(1.0, theta_damping * 1.02)
                            beta_damping = min(1.0, beta_damping * 1.02)
                            if not in_warmup:
                                v_damping = min(0.8, v_damping * 1.01)
                                gamma_damping = min(0.8, gamma_damping * 1.01)
                        else:
                            theta_damping = max(0.3, theta_damping * 0.95)
                            beta_damping = max(0.3, beta_damping * 0.95)
                            if not in_warmup:
                                v_damping = max(0.1, v_damping * 0.9)
                                gamma_damping = max(0.1, gamma_damping * 0.9)

                    if verbose:
                        beta_div = np.std(np.array(self.E_beta), axis=1).mean()
                        if in_warmup:
                            phase = "WU"
                        elif in_anneal:
                            phase = f"AN{self.regression_weight:.2f}"
                        else:
                            phase = "FT"
                        # Show diversity noise while it's active
                        noise_str = f"  noise={diversity_noise:.2f}" if diversity_noise > 0.01 else ""
                        print(f"Iter {t:4d} [{phase}]: ELBO={elbo:.2e}  "
                              f"Poisson={poisson_ll:.2e}  Regression={regression_ll:.2e}  "
                              f"v={np.array(self.mu_v).ravel()[:3]}  β_div={beta_div:.3f}{noise_str}")

                    # Convergence: relative change (only after annealing complete)
                    if (len(self.elbo_history_) >= 2 and t >= min_iter
                            and t >= anneal_end):
                        prev = self.elbo_history_[-2][1]
                        rel_change = abs(elbo - prev) / (abs(prev) + 1e-10)
                        if rel_change < rel_tol and no_improve >= patience:
                            if verbose:
                                print(f"Converged at iter {t}: rel_change={rel_change:.2e}")
                            break

            # Held-out LL tracking
            if (X_heldout is not None and y_heldout is not None
                    and t % (elbo_freq * heldout_freq) == 0):
                ho_ll = self.compute_heldout_loglik(X_heldout, y_heldout, X_aux_heldout)
                self.heldout_loglik_history_.append((t, ho_ll))
                if verbose:
                    print(f"  HO-LL = {ho_ll:.4f}")

        self.training_time_ = time.time() - start_time

        # Restore best
        self.restored_to_best_ = False
        if best_params is not None and no_improve > 0:
            self._restore_params(best_params)
            self.restored_to_best_ = True
            if verbose:
                print("Restored parameters from best iteration.")

        # Store final training params in the SVI-compatible attribute names
        self.train_a_theta_ = self.a_theta
        self.train_b_theta_ = self.b_theta
        self.train_a_xi_ = self.a_xi
        self.train_b_xi_ = self.b_xi

        self.final_elbo_ema_ = best_elbo
        self.final_elbo_mean_ = best_elbo
        self.final_elbo_std_ = 0.0

        if verbose:
            print(f"\nTraining complete in {self.training_time_:.1f}s")
            print(f"Best ELBO: {best_elbo:.2e}")
            if hasattr(self, 'initial_mu_v_'):
                v_change = np.array(self.mu_v - self.initial_mu_v_)
                print(f"v change norm: {np.linalg.norm(v_change):.4f}")
                print(f"v range: [{np.array(self.mu_v).min():.4f}, {np.array(self.mu_v).max():.4f}]")

        return self

    # =====================================================================
    # TRANSFORM  (infer θ for new data with frozen globals)
    # =====================================================================

    def transform(self, X_new: np.ndarray, y_new: np.ndarray = None,
                  X_aux_new: np.ndarray = None, n_iter: int = 50,
                  average_last_n: int = 10, as_numpy: bool = True) -> dict:
        """Exactly mirrors SVICorrected.transform, batched over cells to avoid OOM."""
        if sp.issparse(X_new):
            X_new = jnp.array(X_new.toarray())
        else:
            X_new = jnp.array(X_new)

        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = jnp.zeros((n_new, self.p_aux if self.p_aux > 0 else 0))
        else:
            X_aux_new = jnp.array(X_aux_new)

        if y_new is None:
            y_new = jnp.full((n_new, self.kappa), 0.5)
        else:
            y_new = jnp.array(y_new)
            y_new = y_new.reshape(-1, 1) if y_new.ndim == 1 else y_new

        row_sums = X_new.sum(axis=1, keepdims=True) + 1
        factor_scales = jnp.linspace(0.5, 2.0, self.d)

        a_theta = jnp.full((n_new, self.d), self.alpha_theta + self.p * self.alpha_beta)
        b_theta = row_sums / self.d * factor_scales
        a_xi = jnp.full(n_new, self.alpha_xi + self.d * self.alpha_theta)
        b_xi = jnp.full(n_new, self.lambda_xi)

        E_beta = self.a_beta / self.b_beta
        E_log_beta = jsp.digamma(self.a_beta) - jnp.log(self.b_beta)
        E_v = self.mu_v
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag
        E_gamma = self.mu_gamma
        beta_col_sums = E_beta.sum(axis=0)

        E_theta_sum = jnp.zeros((n_new, self.d))

        # Batch size to keep (B, p, d) tensor under ~2GB
        bytes_per_cell = self.p * self.d * 4 * 3
        target_bytes = 2.0 * 1024**3
        cell_bs = max(32, min(n_new, int(target_bytes / bytes_per_cell)))

        for it in range(n_iter):
            E_theta = a_theta / b_theta
            E_log_theta = jsp.digamma(a_theta) - jnp.log(b_theta)
            E_xi = a_xi / b_xi
            E_theta_sq = (a_theta * (a_theta + 1)) / b_theta ** 2

            a_xi = self.alpha_xi + self.d * self.alpha_theta
            b_xi = self.lambda_xi + E_theta.sum(axis=1)
            E_xi = a_xi / b_xi

            # Batched phi computation + theta update
            shape_contrib = jnp.zeros((n_new, self.d))
            for s in range(0, n_new, cell_bs):
                e = min(s + cell_bs, n_new)
                log_phi_b = E_log_theta[s:e, jnp.newaxis, :] + E_log_beta[jnp.newaxis, :, :]
                log_phi_b = log_phi_b - logsumexp(log_phi_b, axis=2, keepdims=True)
                phi_b = jnp.exp(log_phi_b)
                shape_contrib = shape_contrib.at[s:e].set(jnp.einsum('ij,ijl->il', X_new[s:e], phi_b))

            if self.p_aux > 0:
                aux_contrib = X_aux_new @ E_gamma.T
            else:
                aux_contrib = 0.0

            theta_v = E_theta @ E_v.T
            E_A = theta_v + aux_contrib
            if self.p_aux > 0:
                E_A_sq = E_theta_sq @ E_v_sq.T + 2 * theta_v * aux_contrib + aux_contrib ** 2
            else:
                E_A_sq = E_theta_sq @ E_v_sq.T
            zeta = jnp.sqrt(jnp.maximum(E_A_sq, 1e-10))
            lam = jnp.tanh(zeta / 2) / (4 * zeta + 1e-10)

            a_theta = self.alpha_theta + shape_contrib
            b_theta = E_xi[:, jnp.newaxis] + beta_col_sums[jnp.newaxis, :]

            C_minus_ell = E_A[:, :, jnp.newaxis] - E_theta[:, jnp.newaxis, :] * E_v[jnp.newaxis, :, :]
            R = (-(y_new[:, :, jnp.newaxis] - 0.5) * E_v[jnp.newaxis, :, :]
                 + 2 * lam[:, :, jnp.newaxis] * E_v[jnp.newaxis, :, :] * C_minus_ell
                 + 2 * lam[:, :, jnp.newaxis] * E_v_sq[jnp.newaxis, :, :] * E_theta[:, jnp.newaxis, :])

            b_theta = b_theta + R.sum(axis=1)
            b_theta = jnp.maximum(b_theta, 1e-10)

            if it >= (n_iter - average_last_n):
                E_theta_sum = E_theta_sum + (a_theta / b_theta)

        E_theta_final = E_theta_sum / average_last_n if average_last_n > 0 else a_theta / b_theta

        if as_numpy:
            return {
                'E_theta': np.array(E_theta_final), 'a_theta': np.array(a_theta),
                'b_theta': np.array(b_theta), 'a_xi': np.array(a_xi), 'b_xi': np.array(b_xi),
            }
        return {
            'E_theta': E_theta_final, 'a_theta': a_theta,
            'b_theta': b_theta, 'a_xi': a_xi, 'b_xi': b_xi,
        }

    # =====================================================================
    # PREDICT
    # =====================================================================

    def predict_proba(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                      n_iter: int = 50, pip_threshold: float = None) -> np.ndarray:
        result = self.transform(X_new, y_new=None, X_aux_new=X_aux_new,
                                n_iter=n_iter, average_last_n=10, as_numpy=False)
        E_theta = result['E_theta']
        effective_v = self.mu_v
        if pip_threshold is not None and self.use_spike_slab:
            effective_v = self.mu_v * (self.rho_v > pip_threshold)

        if self.p_aux > 0:
            if X_aux_new is None:
                X_aux_jnp = jnp.zeros((E_theta.shape[0], self.p_aux))
            elif not isinstance(X_aux_new, jnp.ndarray):
                X_aux_jnp = jnp.array(X_aux_new)
            else:
                X_aux_jnp = X_aux_new
            logits = E_theta @ effective_v.T + X_aux_jnp @ self.mu_gamma.T
        else:
            logits = E_theta @ effective_v.T

        return np.array(jsp.expit(logits)).squeeze()

    def predict(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                n_iter: int = 50, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X_new, X_aux_new, n_iter)
        return (proba >= threshold).astype(int)

    # =====================================================================
    # Held-out log-likelihood (compatible with SVICorrected)
    # =====================================================================

    def compute_heldout_loglik(
        self,
        X_heldout: np.ndarray,
        y_heldout: np.ndarray,
        X_aux_heldout: Optional[np.ndarray] = None,
        n_iter: int = 30,
    ) -> float:
        if sp.issparse(X_heldout):
            X_heldout = jnp.array(X_heldout.toarray())
        elif not isinstance(X_heldout, jnp.ndarray):
            X_heldout = jnp.array(X_heldout)
        y_heldout = jnp.array(y_heldout) if not isinstance(y_heldout, jnp.ndarray) else y_heldout
        if y_heldout.ndim == 1:
            y_heldout = y_heldout[:, jnp.newaxis]

        n_heldout = X_heldout.shape[0]
        if X_aux_heldout is None:
            X_aux_heldout = jnp.zeros((n_heldout, self.p_aux if self.p_aux > 0 else 0))
        elif not isinstance(X_aux_heldout, jnp.ndarray):
            X_aux_heldout = jnp.array(X_aux_heldout)

        result = self.transform(X_heldout, y_new=y_heldout, X_aux_new=X_aux_heldout,
                                n_iter=n_iter, as_numpy=False)
        E_theta = result['E_theta']
        E_log_theta = jsp.digamma(result['a_theta']) - jnp.log(result['b_theta'])

        log_rates = E_log_theta[:, jnp.newaxis, :] + self.E_log_beta_effective[jnp.newaxis, :, :]
        log_sum_rates = logsumexp(log_rates, axis=2)
        loglik_counts = jnp.sum(X_heldout * log_sum_rates)
        loglik_counts -= jnp.sum(E_theta.sum(axis=1, keepdims=True) * self.E_beta_effective.sum(axis=0, keepdims=True))
        loglik_counts -= jnp.sum(jsp.gammaln(X_heldout + 1))

        if self.p_aux > 0:
            logits = E_theta @ self.E_v_effective.T + X_aux_heldout @ self.E_gamma.T
        else:
            logits = E_theta @ self.E_v_effective.T
        logits_clipped = jnp.clip(logits, -500, 500)
        loglik_labels = jnp.sum(y_heldout * logits_clipped - jnp.log1p(jnp.exp(logits_clipped)))

        total = float(loglik_counts) + self.regression_weight * float(loglik_labels)
        return total / n_heldout

    # =====================================================================
    # Batched helpers (same interface as SVICorrected)
    # =====================================================================

    def _compute_memory_efficient_batch_size(self, target_gb: float = 0.5) -> int:
        bytes_per_sample = self.p * self.d * 8
        target_bytes = target_gb * 1024 * 1024 * 1024
        return max(1, int(target_bytes / bytes_per_sample))

    def transform_batched(self, X_new, y_new=None, X_aux_new=None,
                          n_iter=50, batch_size=None, verbose=False) -> dict:
        if sp.issparse(X_new):
            X_new = X_new.toarray()
        X_new = np.array(X_new)
        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, self.p_aux if self.p_aux > 0 else 0))
        else:
            X_aux_new = np.array(X_aux_new)
        if y_new is None:
            y_new = np.full((n_new, self.kappa), 0.5)
        else:
            y_new = np.array(y_new)
            if y_new.ndim == 1:
                y_new = y_new[:, np.newaxis]
        if batch_size is None:
            batch_size = self._compute_memory_efficient_batch_size()

        E_theta_all = np.zeros((n_new, self.d))
        a_theta_all = np.zeros((n_new, self.d))
        b_theta_all = np.zeros((n_new, self.d))
        a_xi_all = np.zeros(n_new)
        b_xi_all = np.zeros(n_new)
        n_batches = (n_new + batch_size - 1) // batch_size
        for bi in range(n_batches):
            s, e = bi * batch_size, min((bi + 1) * batch_size, n_new)
            r = self.transform(X_new[s:e], y_new=y_new[s:e], X_aux_new=X_aux_new[s:e], n_iter=n_iter)
            E_theta_all[s:e] = r['E_theta']
            a_theta_all[s:e] = r['a_theta']
            b_theta_all[s:e] = r['b_theta']
            a_xi_all[s:e] = r['a_xi']
            b_xi_all[s:e] = r['b_xi']
        return {'E_theta': E_theta_all, 'a_theta': a_theta_all, 'b_theta': b_theta_all,
                'a_xi': a_xi_all, 'b_xi': b_xi_all}

    def predict_proba_batched(self, X_new, X_aux_new=None, n_iter=50,
                               batch_size=None, verbose=False) -> np.ndarray:
        if sp.issparse(X_new):
            X_new = X_new.toarray()
        X_new = np.array(X_new)
        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, self.p_aux if self.p_aux > 0 else 0))
        else:
            X_aux_new = np.array(X_aux_new)
        if batch_size is None:
            batch_size = self._compute_memory_efficient_batch_size()
        proba_all = np.zeros(n_new) if self.kappa == 1 else np.zeros((n_new, self.kappa))
        n_batches = (n_new + batch_size - 1) // batch_size
        for bi in range(n_batches):
            s, e = bi * batch_size, min((bi + 1) * batch_size, n_new)
            proba_all[s:e] = self.predict_proba(X_new[s:e], X_aux_new[s:e], n_iter=n_iter)
        return proba_all

    def predict_batched(self, X_new, X_aux_new=None, n_iter=50,
                        threshold=0.5, batch_size=None, verbose=False) -> np.ndarray:
        proba = self.predict_proba_batched(X_new, X_aux_new, n_iter, batch_size, verbose)
        return (proba >= threshold).astype(int)

    # =====================================================================
    # Sparse factor accessors (same interface as SVICorrected)
    # =====================================================================

    def get_sparse_factors(self, pip_threshold: float = 0.5) -> dict:
        if not self.use_spike_slab:
            return self._get_magnitude_factors()
        active_mask = np.array(self.rho_v > pip_threshold)
        v_values = np.array(self.mu_v)
        rho_values = np.array(self.rho_v)
        results = {'active_factors': [], 'v_values': [], 'rho_values': [], 'direction': []}
        for k in range(self.kappa):
            for ell in range(self.d):
                if active_mask[k, ell]:
                    results['active_factors'].append((k, ell))
                    results['v_values'].append(float(v_values[k, ell]))
                    results['rho_values'].append(float(rho_values[k, ell]))
                    results['direction'].append('risk' if v_values[k, ell] > 0 else 'protective')
        return results

    def _get_magnitude_factors(self, threshold: float = 0.1) -> dict:
        v_abs = np.abs(np.array(self.mu_v))
        active = v_abs > threshold
        results = {'active_factors': [], 'v_values': [], 'rho_values': [], 'direction': []}
        for k in range(self.kappa):
            for ell in range(self.d):
                if active[k, ell]:
                    results['active_factors'].append((k, ell))
                    results['v_values'].append(float(self.mu_v[k, ell]))
                    results['rho_values'].append(1.0)
                    results['direction'].append('risk' if self.mu_v[k, ell] > 0 else 'protective')
        return results

    # =====================================================================
    # Calibration (delegate to sklearn, same as SVICorrected)
    # =====================================================================

    def fit_calibration(self, X_cal, y_cal, X_aux_cal=None, method='platt',
                        n_iter=50, optimize_threshold=True, threshold_metric='f1',
                        verbose=False):
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import f1_score, roc_curve

        y_raw_proba = self.predict_proba(X_cal, X_aux_cal, n_iter=n_iter)
        if y_raw_proba.ndim > 1:
            y_raw_proba = y_raw_proba.ravel()
        y_cal = np.asarray(y_cal).ravel()

        if method == 'platt':
            logits = np.clip(y_raw_proba, 1e-6, 1 - 1e-6)
            logits = np.log(logits / (1 - logits))
            self.calibrator_ = LogisticRegression(solver='lbfgs', max_iter=1000, C=1e6)
            self.calibrator_.fit(logits.reshape(-1, 1), y_cal)
        elif method == 'isotonic':
            self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            self.calibrator_.fit(y_raw_proba, y_cal)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self.calibration_method_ = method
        y_cal_proba = self._apply_calibration(y_raw_proba)

        if optimize_threshold:
            self.optimal_threshold_ = self._find_optimal_threshold(y_cal, y_cal_proba, metric=threshold_metric)
        else:
            self.optimal_threshold_ = 0.5

    def _apply_calibration(self, y_proba):
        if not hasattr(self, 'calibrator_'):
            return np.asarray(y_proba).ravel()
        y_proba = np.asarray(y_proba).ravel()
        if self.calibration_method_ == 'platt':
            logits = np.clip(y_proba, 1e-6, 1 - 1e-6)
            logits = np.log(logits / (1 - logits))
            calibrated = self.calibrator_.predict_proba(logits.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator_.predict(y_proba)
        return np.clip(calibrated, 0, 1)

    def _find_optimal_threshold(self, y_true, y_proba, metric='f1'):
        from sklearn.metrics import f1_score, roc_curve
        if metric == 'youden':
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            return thresholds[np.argmax(tpr - fpr)]
        elif metric == 'f1':
            best_f1, best_t = 0, 0.5
            for t in np.linspace(0.1, 0.9, 81):
                f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            return best_t
        elif metric == 'accuracy':
            best_acc, best_t = 0, 0.5
            for t in np.linspace(0.1, 0.9, 81):
                acc = ((y_proba >= t).astype(int) == y_true).mean()
                if acc > best_acc:
                    best_acc, best_t = acc, t
            return best_t
        raise ValueError(f"Unknown metric: {metric}")

    def predict_calibrated(self, X_new, X_aux_new=None, n_iter=50, threshold=None):
        proba_raw = self.predict_proba(X_new, X_aux_new, n_iter=n_iter)
        proba_cal = self._apply_calibration(proba_raw)
        if threshold is None:
            threshold = getattr(self, 'optimal_threshold_', 0.5)
        return (proba_cal >= threshold).astype(int)


# Convenience aliases
VI = CAVI
