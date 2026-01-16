import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.scipy.special import logsumexp
from jax import random
from typing import Tuple, Optional, Dict, Any
import time
import numpy as np
import scipy.sparse as sp
from scipy.special import expit, xlogy


class SVILaplace:

    def __init__(
        self,
        n_factors: int,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.75,
        learning_rate_delay: float = 1.0,
        learning_rate_min: float = 1e-2,
        local_iterations: int = 10,
        
        # Priors for Poisson factorization
        alpha_theta: float = 1.1,
        alpha_beta: float = 1.1,
        alpha_xi: float = 2.0,
        alpha_eta: float = 2.0,
        lambda_xi: float = 1.0,
        lambda_eta: float = 1.0,
        
        # Laplace prior scale for v: v ~ Laplace(0, b_laplace)
        # Larger b = less regularization, smaller b = more shrinkage
        b_laplace: float = 1.0,
        
        # Gaussian prior for auxiliary coefficients γ
        sigma_gamma: float = 1.0,
        
        # Supervision weight
        regression_weight: float = 1.0,
        
        # Convergence tracking
        ema_decay: float = 0.95,
        convergence_tol: float = 1e-4,
        convergence_window: int = 10,
        
        random_state: Optional[int] = None,
        use_jit: bool = True,
        device: str = 'gpu'
    ):
        self.d = n_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_delay = learning_rate_delay
        self.learning_rate_min = learning_rate_min
        self.local_iterations = local_iterations
        
        # Batch size for prediction (separate from training batch size)
        self.predict_batch_size = min(batch_size, 512)  # Default to smaller batches for prediction

        # Poisson factorization priors
        self.alpha_theta = alpha_theta
        self.alpha_beta = alpha_beta
        self.alpha_xi = alpha_xi
        self.alpha_eta = alpha_eta
        self.lambda_xi = lambda_xi
        self.lambda_eta = lambda_eta
        
        # Laplace prior scale: v ~ Laplace(0, b)
        # Exponential rate for τ: τ ~ Exp(1/(2b²))
        self.b_laplace = b_laplace
        self.tau_rate = 1.0 / (2.0 * b_laplace**2)  # Rate parameter for Exp prior on τ
        
        self.sigma_gamma = sigma_gamma
        self.regression_weight = regression_weight
        
        # Convergence
        self.ema_decay = ema_decay
        self.convergence_tol = convergence_tol
        self.convergence_window = convergence_window
        
        # JAX settings
        self.use_jit = use_jit
        self.device = device
        self.rng_key = random.PRNGKey(random_state if random_state is not None else 0)
        
        # Dimensions (set during fit)
        self.n = None
        self.p = None
        self.kappa = None
        self.p_aux = None
    
    def _get_learning_rate(self, t: int) -> float:
        """Robbins-Monro schedule: ρ_t = lr · (τ + t)^(-κ)"""
        rho = self.learning_rate * (self.learning_rate_delay + t) ** (-self.learning_rate_decay)
        return max(rho, self.learning_rate_min)
    
    @staticmethod
    @jax.jit
    def _lambda_jj(zeta: jnp.ndarray) -> jnp.ndarray:
        """
        Jaakkola-Jordan auxiliary function: λ(ζ) = tanh(ζ/2) / (4ζ)
        Numerically stable with λ(0) = 1/8.
        """
        return jnp.where(
            jnp.abs(zeta) > 1e-8,
            jnp.tanh(zeta / 2) / (4 * zeta + 1e-20),
            0.125
        )
    
    # =========================================================================
    # NATURAL PARAMETER CONVERSIONS (for Gamma distributions)
    # =========================================================================
    
    @staticmethod
    @jax.jit
    def _gamma_to_natural(a: jnp.ndarray, b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Gamma(a, b) → natural parameters (η₁, η₂) = (a-1, -b)"""
        return a - 1, -b
    
    @staticmethod
    @jax.jit
    def _natural_to_gamma(eta1: jnp.ndarray, eta2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Natural parameters → Gamma(a, b) = (η₁+1, -η₂)"""
        a = jnp.maximum(eta1 + 1, 1.001)
        b = jnp.maximum(-eta2, 1e-6)
        return a, b
    
    @staticmethod
    @jax.jit
    def _gamma_entropy(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Gamma(shape=a, rate=b) differential entropy.
        
        H[Gamma(a,b)] = a - ln(b) + ln(Γ(a)) + (1-a)ψ(a)
        
        Numerically stable using gammaln and digamma.
        """
        a = jnp.maximum(a, 1e-6)
        b = jnp.maximum(b, 1e-6)
        return a - jnp.log(b) + jsp.gammaln(a) + (1.0 - a) * jsp.digamma(a)
    
    @staticmethod
    @jax.jit
    def _gaussian_entropy_2d(Sigma: jnp.ndarray) -> float:
        """
        Multivariate Gaussian entropy: H = 0.5 * (d * (1 + log(2π)) + log|Σ|)
        
        Uses slogdet for numerical stability.
        """
        d = Sigma.shape[0]
        sign, logdet = jnp.linalg.slogdet(Sigma)
        return 0.5 * (d * (1.0 + jnp.log(2.0 * jnp.pi)) + logdet)
    
    # =========================================================================
    # INVERSE GAUSSIAN EXPECTATIONS (for τ posterior)
    # =========================================================================
    
    @staticmethod
    @jax.jit
    def _compute_E_tau_inv(mu_v: jnp.ndarray, sigma_v_sq: jnp.ndarray, 
                           b_laplace: float) -> jnp.ndarray:
        """
        Compute E[τ⁻¹] under q(τ) = InvGaussian.
        
        For q(τ) = GIG(-1/2, 1/b², μ_v² + σ_v²):
            E[τ⁻¹] = 1/(b·√(μ_v² + σ_v²)) + 1/(μ_v² + σ_v²)
        
        The second term comes from the GIG expectation formula for p = -1/2.
        
        Parameters
        ----------
        mu_v : (κ, d) variational mean of v
        sigma_v_sq : (κ, d) variational variance of v
        b_laplace : Laplace scale parameter
        
        Returns
        -------
        E_tau_inv : (κ, d) expected inverse of τ
        """
        v_second_moment = mu_v**2 + sigma_v_sq  # E[v²] = μ² + σ²
        # Use a reasonable floor to prevent shrinkage spiral
        # With floor=0.01, max E[τ⁻¹] ≈ 1/(b*0.1) + 1/0.01 ≈ 111 + 100 = 211
        # This prevents the prior from completely overwhelming the likelihood
        v_second_moment = jnp.maximum(v_second_moment, 0.01)
        
        # E[τ⁻¹] for InvGaussian with our parameterization
        # Term 1: 1/(b · √(E[v²]))
        # Term 2: 1/E[v²] (from GIG formula)
        E_tau_inv = 1.0 / (b_laplace * jnp.sqrt(v_second_moment)) + 1.0 / v_second_moment
        
        return E_tau_inv
    
    @staticmethod
    @jax.jit
    def _compute_E_tau(mu_v: jnp.ndarray, sigma_v_sq: jnp.ndarray,
                       b_laplace: float) -> jnp.ndarray:
        """
        Compute E[τ] under q(τ) = InvGaussian.
        
        For q(τ) = GIG(-1/2, 1/b², μ_v² + σ_v²):
            E[τ] = b · √(μ_v² + σ_v²)
        """
        v_second_moment = mu_v**2 + sigma_v_sq
        v_second_moment = jnp.maximum(v_second_moment, 1e-10)
        return b_laplace * jnp.sqrt(v_second_moment)
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _initialize_parameters(self, X: jnp.ndarray, y: jnp.ndarray, X_aux: jnp.ndarray):
        """Initialize all variational parameters."""
        self.n, self.p = X.shape
        self.kappa = 1 if y.ndim == 1 else y.shape[1]
        self.p_aux = X_aux.shape[1] if X_aux is not None and X_aux.size > 0 else 0
        
        key1, key2, key3, key4, self.rng_key = random.split(self.rng_key, 5)

        # -----------------------------------------------------------------
        # β: Gene loadings ~ Gamma(a_β, b_β)
        # Initialize with diversity to break symmetry
        # -----------------------------------------------------------------
        col_means = X.mean(axis=0) + 1
        self.a_beta = jnp.full((self.p, self.d), self.alpha_beta)
        self.b_beta = jnp.full((self.p, self.d), 1.0)

        # Add random diversity
        random_boost = random.uniform(key1, (self.p, self.d), minval=0.1, maxval=3.0)
        gene_scale = col_means[:, jnp.newaxis]
        self.a_beta = self.a_beta + gene_scale * random_boost * 5.0

        # Factor-specific signatures
        factor_signatures = random.uniform(key2, (self.p, self.d), minval=0.0, maxval=2.0)
        sparsity_masks = random.bernoulli(key3, p=0.3, shape=(self.p, self.d))
        self.a_beta = self.a_beta + factor_signatures * sparsity_masks * gene_scale * 3.0

        # CRITICAL: Add factor-level scaling that survives sum over genes
        # This prevents all factors from collapsing to identical values
        # Each factor gets a different overall "strength" multiplier
        factor_scales = random.uniform(key4, (self.d,), minval=0.5, maxval=2.0)
        self.a_beta = self.a_beta * factor_scales[jnp.newaxis, :]

        self.a_beta = jnp.maximum(self.a_beta, self.alpha_beta * 0.1)
        
        # -----------------------------------------------------------------
        # η: Gene capacity ~ Gamma(a_η, b_η)
        # -----------------------------------------------------------------
        self.a_eta = jnp.full(self.p, self.alpha_eta + self.d * self.alpha_beta)
        self.b_eta = jnp.full(self.p, self.lambda_eta)
        
        # -----------------------------------------------------------------
        # v: Regression coefficients ~ Laplace(0, b) via scale mixture
        # q(v_kℓ) = N(μ_v, σ_v²)
        # -----------------------------------------------------------------
        key_v, self.rng_key = random.split(self.rng_key)
        # Initialize near zero with small noise
        self.mu_v = 0.1 * random.normal(key_v, (self.kappa, self.d))
        # Initial variance from Laplace: Var[Laplace(0,b)] = 2b²
        self.sigma_v_sq = jnp.full((self.kappa, self.d), 2.0 * self.b_laplace**2)
        
        # Store initial v for diagnostics
        self.initial_mu_v_ = self.mu_v.copy()
        
        # -----------------------------------------------------------------
        # τ: Auxiliary variance (implicit - we only need E[τ⁻¹])
        # q(τ_kℓ) = InvGaussian, computed from v parameters
        # -----------------------------------------------------------------
        # No explicit storage needed - computed on the fly from mu_v, sigma_v_sq
        
        # -----------------------------------------------------------------
        # γ: Auxiliary regression coefficients ~ N(μ_γ, Σ_γ)
        # -----------------------------------------------------------------
        if self.p_aux > 0:
            self.mu_gamma = jnp.zeros((self.kappa, self.p_aux))
            self.Sigma_gamma = jnp.array([
                jnp.eye(self.p_aux) * self.sigma_gamma**2 
                for _ in range(self.kappa)
            ])
        else:
            self.mu_gamma = jnp.zeros((self.kappa, 0))
            self.Sigma_gamma = jnp.zeros((self.kappa, 0, 0))
        
        # -----------------------------------------------------------------
        # Natural parameters for SVI (Gamma distributions only)
        # -----------------------------------------------------------------
        self.eta1_beta, self.eta2_beta = self._gamma_to_natural(self.a_beta, self.b_beta)
        self.eta1_eta, self.eta2_eta = self._gamma_to_natural(self.a_eta, self.b_eta)
        
        # Compute expectations
        self._compute_expectations()
        
        print(f"Initialized: n={self.n}, p={self.p}, d={self.d}, κ={self.kappa}, p_aux={self.p_aux}")
        print(f"Laplace scale b={self.b_laplace}, τ rate={self.tau_rate:.4f}")
        print(f"Initial β diversity: {np.std(np.array(self.E_beta), axis=1).mean():.4f}")
    
    def _compute_expectations(self):
        """Compute expected sufficient statistics."""
        # Gamma expectations
        self.E_beta = self.a_beta / self.b_beta
        self.E_log_beta = jsp.digamma(self.a_beta) - jnp.log(self.b_beta)
        self.E_eta = self.a_eta / self.b_eta
        self.E_log_eta = jsp.digamma(self.a_eta) - jnp.log(self.b_eta)
        
        # Gaussian expectations (v, γ)
        self.E_v = self.mu_v
        self.E_v_sq = self.mu_v**2 + self.sigma_v_sq  # E[v²] = μ² + σ²
        self.E_gamma = self.mu_gamma
        
        # τ expectations (from InvGaussian posterior)
        self.E_tau_inv = self._compute_E_tau_inv(self.mu_v, self.sigma_v_sq, self.b_laplace)
        self.E_tau = self._compute_E_tau(self.mu_v, self.sigma_v_sq, self.b_laplace)
    
    # =========================================================================
    # LOCAL PARAMETER UPDATES
    # =========================================================================
    
    def _update_local_parameters(
        self,
        X_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
        X_aux_batch: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Optimize local parameters (θ, ξ, ζ) for a mini-batch.
        
        Returns: (a_theta, b_theta, a_xi, b_xi, zeta)
        """
        batch_size = X_batch.shape[0]
        
        # Initialize
        row_sums = X_batch.sum(axis=1, keepdims=True) + 1
        a_theta = jnp.full((batch_size, self.d), self.alpha_theta) + row_sums / self.d
        b_theta = jnp.full((batch_size, self.d), 1.0)
        a_xi = jnp.full(batch_size, self.alpha_xi)
        b_xi = jnp.full(batch_size, self.lambda_xi)
        zeta = jnp.ones((batch_size, self.kappa))
        
        for _ in range(self.local_iterations):
            E_theta = a_theta / b_theta
            E_log_theta = jsp.digamma(a_theta) - jnp.log(b_theta)
            E_xi = a_xi / b_xi
            
            # φ update: multinomial allocations
            log_phi = E_log_theta[:, jnp.newaxis, :] + self.E_log_beta[jnp.newaxis, :, :]
            log_phi = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
            phi = jnp.exp(log_phi)
            
            # E[z_ijℓ] = x_ij · φ_ijℓ
            E_z = X_batch[:, :, jnp.newaxis] * phi
            
            # θ shape update
            a_theta_new = self.alpha_theta + E_z.sum(axis=1)
            
            # θ rate update (includes regression contribution)
            b_theta_new = E_xi[:, jnp.newaxis] + self.E_beta.sum(axis=0)[jnp.newaxis, :]
            
            # JJ bound parameter λ(ζ)
            lam = self._lambda_jj(zeta)
            
            # Linear predictor components
            theta_v = E_theta @ self.E_v.T  # (batch, kappa)
            if self.p_aux > 0:
                aux_term = X_aux_batch @ self.E_gamma.T
            else:
                aux_term = 0.0
            
            # C^(-ℓ): linear predictor minus ℓ-th factor contribution
            # FIXED: Correct order of operations
            C_minus_ell = (
                theta_v[:, :, jnp.newaxis] +  # Full θ·v: (batch, kappa, 1)
                (aux_term[:, :, jnp.newaxis] if self.p_aux > 0 else 0.0) -  # + aux
                E_theta[:, jnp.newaxis, :] * self.E_v[jnp.newaxis, :, :]  # - ℓ-th term
            )  # Result: (batch, kappa, d)
            
            # Expand y to (batch, kappa)
            y_expanded = y_batch if y_batch.ndim > 1 else y_batch[:, jnp.newaxis]
            
            # Regression contribution R_iℓ to θ rate
            R = (
                -(y_expanded[:, :, jnp.newaxis] - 0.5) * self.E_v[jnp.newaxis, :, :] +
                2 * lam[:, :, jnp.newaxis] * self.E_v[jnp.newaxis, :, :] * C_minus_ell +
                2 * lam[:, :, jnp.newaxis] * self.E_v_sq[jnp.newaxis, :, :] * E_theta[:, jnp.newaxis, :]
            )
            R_sum = R.sum(axis=1)  # Sum over outcomes
            
            b_theta_new = b_theta_new + self.regression_weight * R_sum
            b_theta_new = jnp.maximum(b_theta_new, 1e-6)
            a_theta_new = jnp.maximum(a_theta_new, 1.001)
            
            # ξ update
            E_theta_new = a_theta_new / b_theta_new
            a_xi_new = jnp.full(batch_size, self.alpha_xi + self.d * self.alpha_theta)
            b_xi_new = self.lambda_xi + E_theta_new.sum(axis=1)
            
            # ζ update (JJ auxiliary)
            if self.p_aux > 0:
                aux_contrib = X_aux_batch @ self.E_gamma.T
            else:
                aux_contrib = 0.0
            
            E_A = E_theta_new @ self.E_v.T + aux_contrib
            Var_theta = a_theta_new / (b_theta_new**2)
            E_A_sq = E_A**2 + (Var_theta @ self.E_v_sq.T)
            zeta = jnp.sqrt(jnp.maximum(E_A_sq, 1e-8))
            
            a_theta, b_theta = a_theta_new, b_theta_new
            a_xi, b_xi = a_xi_new, b_xi_new
        
        return a_theta, b_theta, a_xi, b_xi, zeta
    
    # =========================================================================
    # INTERMEDIATE GLOBAL PARAMETERS
    # =========================================================================
    
    def _compute_intermediate_beta(
        self,
        X_batch: jnp.ndarray,
        E_theta: jnp.ndarray,
        E_log_theta: jnp.ndarray,
        scale: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute intermediate β parameters."""
        log_phi = E_log_theta[:, jnp.newaxis, :] + self.E_log_beta[jnp.newaxis, :, :]
        log_phi = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
        phi = jnp.exp(log_phi)
        
        z_sum = (X_batch[:, :, jnp.newaxis] * phi).sum(axis=0)
        theta_sum = E_theta.sum(axis=0)
        
        eta1_hat = (self.alpha_beta - 1) + scale * z_sum
        eta2_hat = -self.E_eta[:, jnp.newaxis] - scale * theta_sum[jnp.newaxis, :]
        
        return self._natural_to_gamma(eta1_hat, eta2_hat)
    
    def _compute_intermediate_eta(self, scale: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute intermediate η parameters."""
        beta_sum = self.E_beta.sum(axis=1)
        
        eta1_hat = (self.alpha_eta - 1) + self.d * self.alpha_beta
        eta2_hat = -self.lambda_eta - beta_sum
        
        return self._natural_to_gamma(jnp.full(self.p, eta1_hat), eta2_hat)
    
    def _compute_intermediate_v(
        self,
        y_batch: jnp.ndarray,
        X_aux_batch: jnp.ndarray,
        E_theta: jnp.ndarray,
        a_theta: jnp.ndarray,
        b_theta: jnp.ndarray,
        zeta: jnp.ndarray,
        scale: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute intermediate v parameters with Laplace prior (Bayesian Lasso).
        
        Key difference from Gaussian prior:
        - Precision includes E[τ⁻¹] instead of 1/σ²
        - E[τ⁻¹] is computed from current v posterior, creating adaptive shrinkage
        
        Precision: Λ_v = E[τ⁻¹] + 2·Σ_i λ(ζ_ik)·E[θ_iℓ²]
        Mean·Prec: Λ_v·μ_v = Σ_i [(y_ik - ½) - 2λ(ζ_ik)·C_ik^(-ℓ)] · E[θ_iℓ]
        """
        y_expanded = y_batch if y_batch.ndim > 1 else y_batch[:, jnp.newaxis]
        lam = self._lambda_jj(zeta)
        
        # E[θ²]
        Var_theta = a_theta / (b_theta**2)
        E_theta_sq = E_theta**2 + Var_theta
        
        # Auxiliary contribution
        if self.p_aux > 0:
            aux_contrib = X_aux_batch @ self.E_gamma.T
        else:
            aux_contrib = 0.0
        
        full_theta_v = E_theta @ self.E_v.T
        
        # =====================================================================
        # PRECISION: E[τ⁻¹] + likelihood contribution
        # =====================================================================
        # Prior precision from Laplace (via scale mixture)
        # E[τ_kℓ⁻¹] computed from current q(v_kℓ)
        E_tau_inv = self._compute_E_tau_inv(self.mu_v, self.sigma_v_sq, self.b_laplace)
        
        # Precision: (κ, d)
        precision = E_tau_inv.copy()  # Prior contribution

        # Likelihood contribution: 2·Σ_i λ(ζ_ik)·E[θ_iℓ²]
        # Include regression_weight to match ELBO weighting
        precision_contrib = 2 * scale * self.regression_weight * jnp.einsum('ik,id->kd', lam, E_theta_sq)
        precision = precision + precision_contrib

        # =====================================================================
        # MEAN × PRECISION
        # =====================================================================
        # C^(-ℓ) for each factor
        C_minus_ell = (
            full_theta_v[:, :, jnp.newaxis] +
            (aux_contrib[:, :, jnp.newaxis] if self.p_aux > 0 else 0.0) -
            E_theta[:, jnp.newaxis, :] * self.E_v[jnp.newaxis, :, :]
        )

        # Mean × precision: Σ_i [(y_ik - ½) - 2λ(ζ_ik)·C_ik^(-ℓ)] · E[θ_iℓ]
        # Include regression_weight to match ELBO weighting
        term1 = scale * self.regression_weight * jnp.einsum('ik,id->kd', y_expanded - 0.5, E_theta)
        term2 = 2 * scale * self.regression_weight * jnp.einsum('ik,ikd,id->kd', lam, C_minus_ell, E_theta)
        mean_times_precision = term1 - term2
        
        # =====================================================================
        # POSTERIOR MEAN AND VARIANCE
        # =====================================================================
        sigma_v_sq_hat = 1.0 / precision
        mu_v_hat = mean_times_precision / precision
        
        # Numerical stability
        mu_v_hat = jnp.clip(mu_v_hat, -10, 10)
        sigma_v_sq_hat = jnp.clip(sigma_v_sq_hat, 1e-6, 100)
        
        return mu_v_hat, sigma_v_sq_hat
    
    def _compute_intermediate_gamma(
        self,
        y_batch: jnp.ndarray,
        X_aux_batch: jnp.ndarray,
        E_theta: jnp.ndarray,
        zeta: jnp.ndarray,
        scale: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute intermediate γ parameters (Gaussian prior)."""
        if self.p_aux == 0:
            return jnp.zeros((self.kappa, 0)), jnp.zeros((self.kappa, 0, 0))
        
        y_expanded = y_batch if y_batch.ndim > 1 else y_batch[:, jnp.newaxis]
        lam = self._lambda_jj(zeta)
        
        mu_gamma_hat = jnp.zeros((self.kappa, self.p_aux))
        Sigma_gamma_hat = jnp.zeros((self.kappa, self.p_aux, self.p_aux))
        
        for k in range(self.kappa):
            prec_prior = jnp.eye(self.p_aux) / self.sigma_gamma**2
            # Include regression_weight to match ELBO weighting
            prec_lik = 2 * scale * self.regression_weight * (X_aux_batch.T * lam[:, k]) @ X_aux_batch
            prec_hat = prec_prior + prec_lik

            theta_v = E_theta @ self.E_v[k]
            # Include regression_weight to match ELBO weighting
            mean_contrib = scale * self.regression_weight * X_aux_batch.T @ (y_expanded[:, k] - 0.5 - 2 * lam[:, k] * theta_v)

            Sigma_hat_k = jnp.linalg.inv(prec_hat + 1e-6 * jnp.eye(self.p_aux))
            mu_hat_k = Sigma_hat_k @ mean_contrib
            
            mu_gamma_hat = mu_gamma_hat.at[k].set(mu_hat_k)
            Sigma_gamma_hat = Sigma_gamma_hat.at[k].set(Sigma_hat_k)
        
        return mu_gamma_hat, Sigma_gamma_hat
    
    # =========================================================================
    # SVI GLOBAL UPDATES
    # =========================================================================
    
    def _svi_update_global(
        self,
        rho_t: float,
        a_beta_hat: jnp.ndarray, b_beta_hat: jnp.ndarray,
        a_eta_hat: jnp.ndarray, b_eta_hat: jnp.ndarray,
        mu_v_hat: jnp.ndarray, sigma_v_sq_hat: jnp.ndarray,
        mu_gamma_hat: jnp.ndarray, Sigma_gamma_hat: jnp.ndarray
    ):
        """
        SVI update for global parameters.
        
        Gamma: update natural parameters
        Gaussian (v, γ): update canonical parameters (more stable)
        """
        # β: Natural parameter update
        eta1_beta_hat, eta2_beta_hat = self._gamma_to_natural(a_beta_hat, b_beta_hat)
        self.eta1_beta = (1 - rho_t) * self.eta1_beta + rho_t * eta1_beta_hat
        self.eta2_beta = (1 - rho_t) * self.eta2_beta + rho_t * eta2_beta_hat
        self.a_beta, self.b_beta = self._natural_to_gamma(self.eta1_beta, self.eta2_beta)
        
        # η: Natural parameter update
        eta1_eta_hat, eta2_eta_hat = self._gamma_to_natural(a_eta_hat, b_eta_hat)
        self.eta1_eta = (1 - rho_t) * self.eta1_eta + rho_t * eta1_eta_hat
        self.eta2_eta = (1 - rho_t) * self.eta2_eta + rho_t * eta2_eta_hat
        self.a_eta, self.b_eta = self._natural_to_gamma(self.eta1_eta, self.eta2_eta)
        
        # v: Canonical parameter update (mean and variance)
        # Note: τ is implicit - E[τ⁻¹] recomputed from updated v
        self.mu_v = (1 - rho_t) * self.mu_v + rho_t * mu_v_hat
        self.sigma_v_sq = (1 - rho_t) * self.sigma_v_sq + rho_t * sigma_v_sq_hat
        self.sigma_v_sq = jnp.maximum(self.sigma_v_sq, 1e-6)
        
        # γ: Canonical parameter update
        if self.p_aux > 0:
            for k in range(self.kappa):
                self.mu_gamma = self.mu_gamma.at[k].set(
                    (1 - rho_t) * self.mu_gamma[k] + rho_t * mu_gamma_hat[k]
                )
                Sigma_new = (1 - rho_t) * self.Sigma_gamma[k] + rho_t * Sigma_gamma_hat[k]
                Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)  # Symmetrize
                self.Sigma_gamma = self.Sigma_gamma.at[k].set(Sigma_new)
        
        # Update all expectations
        self._compute_expectations()
    
    # =========================================================================
    # ELBO COMPUTATION
    # =========================================================================
    
    def _compute_elbo(
        self,
        X_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
        X_aux_batch: jnp.ndarray,
        a_theta: jnp.ndarray,
        b_theta: jnp.ndarray,
        a_xi: jnp.ndarray,
        b_xi: jnp.ndarray,
        zeta: jnp.ndarray,
        scale: float
    ) -> float:
        """
        Compute ELBO with Laplace prior on v.
        
        Key difference: KL[q(v,τ) || p(v,τ)] instead of KL[q(v) || p(v)]
        """
        batch_size = X_batch.shape[0]
        elbo = 0.0
        
        # Local expectations
        E_theta = a_theta / b_theta
        E_log_theta = jsp.digamma(a_theta) - jnp.log(b_theta)
        E_xi = a_xi / b_xi
        E_log_xi = jsp.digamma(a_xi) - jnp.log(b_xi)
        
        # === Poisson likelihood ===
        log_rates = E_log_theta[:, jnp.newaxis, :] + self.E_log_beta[jnp.newaxis, :, :]
        log_sum_rates = logsumexp(log_rates, axis=2)
        
        elbo_x = jnp.sum(X_batch * log_sum_rates)
        elbo_x -= jnp.sum(E_theta.sum(axis=0) * self.E_beta.sum(axis=0))
        elbo_x -= jnp.sum(jsp.gammaln(X_batch + 1))
        elbo += scale * elbo_x
        
        # === Bernoulli likelihood (JJ bound) ===
        y_expanded = y_batch if y_batch.ndim > 1 else y_batch[:, jnp.newaxis]
        lam = self._lambda_jj(zeta)
        
        if self.p_aux > 0:
            aux_contrib = X_aux_batch @ self.E_gamma.T
        else:
            aux_contrib = 0.0
        
        E_A = E_theta @ self.E_v.T + aux_contrib
        Var_theta = a_theta / (b_theta**2)
        E_A_sq = E_A**2 + (Var_theta @ self.E_v_sq.T)
        
        elbo_y = jnp.sum((y_expanded - 0.5) * E_A - lam * E_A_sq)
        # Polya-Gamma auxiliary: log(1 + exp(z)) = logaddexp(0, z) for numerical stability
        elbo_y += jnp.sum(lam * zeta**2 - 0.5 * zeta - jnp.logaddexp(0.0, jnp.clip(zeta, -500, 500)))
        elbo += scale * self.regression_weight * elbo_y
        
        # === Local priors ===
        # p(θ | ξ)
        elbo_theta = jnp.sum((self.alpha_theta - 1) * E_log_theta +
                           self.alpha_theta * E_log_xi[:, jnp.newaxis] -
                           E_xi[:, jnp.newaxis] * E_theta)
        elbo_theta -= batch_size * self.d * jsp.gammaln(self.alpha_theta)
        elbo += scale * elbo_theta
        
        # p(ξ)
        elbo_xi = jnp.sum((self.alpha_xi - 1) * E_log_xi - self.lambda_xi * E_xi)
        elbo_xi += batch_size * (self.alpha_xi * jnp.log(self.lambda_xi) - jsp.gammaln(self.alpha_xi))
        elbo += scale * elbo_xi
        
        # === Global priors (not scaled) ===
        # p(β | η)
        elbo_beta = jnp.sum((self.alpha_beta - 1) * self.E_log_beta +
                          self.alpha_beta * self.E_log_eta[:, jnp.newaxis] -
                          self.E_eta[:, jnp.newaxis] * self.E_beta)
        elbo_beta -= self.p * self.d * jsp.gammaln(self.alpha_beta)
        elbo += elbo_beta
        
        # p(η)
        elbo_eta = jnp.sum((self.alpha_eta - 1) * self.E_log_eta - self.lambda_eta * self.E_eta)
        elbo_eta += self.p * (self.alpha_eta * jnp.log(self.lambda_eta) - jsp.gammaln(self.alpha_eta))
        elbo += elbo_eta
        
        # === p(v, τ) - Laplace via scale mixture ===
        # p(v | τ) = N(0, τ) → E_q[log p(v|τ)] = -½ log(2π) - ½ E[log τ] - ½ E[v²]/E[τ]
        # p(τ) = Exp(1/(2b²)) → E_q[log p(τ)] = log(1/(2b²)) - E[τ]/(2b²)
        #
        # For tractability, we use:
        # E[log τ] ≈ log(E[τ]) (Jensen's approximation, tight for InvGaussian)
        # This is standard in Bayesian Lasso implementations
        
        elbo_v_tau = 0.0
        for k in range(self.kappa):
            for ell in range(self.d):
                E_tau_kl = self.E_tau[k, ell]
                E_tau_inv_kl = self.E_tau_inv[k, ell]
                E_v_sq_kl = self.E_v_sq[k, ell]
                
                # p(v | τ): -½ log(2π) - ½ log(τ) - v²/(2τ)
                # Using E[log τ] ≈ log(E[τ])
                elbo_v_tau -= 0.5 * jnp.log(2 * jnp.pi)
                elbo_v_tau -= 0.5 * jnp.log(E_tau_kl + 1e-10)
                elbo_v_tau -= 0.5 * E_v_sq_kl * E_tau_inv_kl
                
                # p(τ) = Exp(rate = 1/(2b²))
                # log p(τ) = log(rate) - rate * τ
                elbo_v_tau += jnp.log(self.tau_rate)
                elbo_v_tau -= self.tau_rate * E_tau_kl
        
        elbo += elbo_v_tau
        
        # p(γ) - Gaussian N(0, σ_γ² I)
        if self.p_aux > 0:
            elbo_gamma = 0.0
            for k in range(self.kappa):
                elbo_gamma -= 0.5 * self.p_aux * jnp.log(2 * jnp.pi * self.sigma_gamma**2)
                elbo_gamma -= 0.5 / self.sigma_gamma**2 * (
                    jnp.sum(self.mu_gamma[k]**2) + jnp.trace(self.Sigma_gamma[k])
                )
            elbo += elbo_gamma
        
        # === Entropy terms ===
        # H[q(θ)] - Use stable gamma entropy helper
        H_theta = jnp.sum(self._gamma_entropy(a_theta, b_theta))
        elbo += scale * H_theta
        
        # H[q(ξ)] - Use stable gamma entropy helper
        H_xi = jnp.sum(self._gamma_entropy(a_xi, b_xi))
        elbo += scale * H_xi
        
        # H[q(β)] - Use stable gamma entropy helper
        H_beta = jnp.sum(self._gamma_entropy(self.a_beta, self.b_beta))
        elbo += H_beta
        
        # H[q(η)] - Use stable gamma entropy helper
        H_eta = jnp.sum(self._gamma_entropy(self.a_eta, self.b_eta))
        elbo += H_eta
        
        # H[q(v)] - Gaussian entropy per factor
        # H[N(μ, σ²)] = ½ log(2πeσ²) = ½ (1 + log(2π) + log(σ²))
        H_v = 0.5 * jnp.sum(1.0 + jnp.log(2.0 * jnp.pi) + jnp.log(jnp.maximum(self.sigma_v_sq, 1e-10)))
        elbo += H_v
        
        # H[q(τ)] - InvGaussian entropy (approximation)
        # For InvGaussian, entropy depends on parameters in complex way
        # Standard approximation: H ≈ ½ log(2πe · Var[τ])
        # Var[τ] for InvGaussian(μ', λ') is μ'³/λ'
        # With our parameterization: μ' = b·√(E[v²]), λ' ≈ 1
        # Var[τ] ≈ b³ · (E[v²])^(3/2)
        H_tau = 0.5 * self.kappa * self.d * (1.0 + jnp.log(2.0 * jnp.pi))
        H_tau += 0.5 * jnp.sum(3.0 * jnp.log(self.b_laplace) + 1.5 * jnp.log(jnp.maximum(self.E_v_sq, 1e-10)))
        elbo += H_tau
        
        # H[q(γ)] - Multivariate Gaussian entropy using stable slogdet
        if self.p_aux > 0:
            for k in range(self.kappa):
                elbo += self._gaussian_entropy_2d(self.Sigma_gamma[k])
        
        return float(elbo)
    
    # =========================================================================
    # MAIN FITTING LOOP
    # =========================================================================
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_aux: np.ndarray = None,
        max_epochs: int = 100,
        elbo_freq: int = 10,
        verbose: bool = True,
        early_stopping: bool = False
    ):
        """
        Fit model using Stochastic Variational Inference.
        """
        # Convert to JAX arrays
        if sp.issparse(X):
            X = jnp.array(X.toarray())
        else:
            X = jnp.array(X)
        
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y = jnp.array(y)
        
        X_aux = jnp.array(X_aux) if X_aux is not None else jnp.zeros((X.shape[0], 0))
        
        # Initialize
        self._initialize_parameters(X, y, X_aux)
        
        n_batches = max(1, (self.n + self.batch_size - 1) // self.batch_size)
        iteration = 0
        self.elbo_history_ = []
        
        # Convergence tracking
        self.elbo_ema_ = None
        self.elbo_ema_prev_ = None
        self.convergence_history_ = []
        consecutive_converged = 0
        
        # Storage for training set local parameters
        self.train_a_theta_ = None
        self.train_b_theta_ = None
        self.train_a_xi_ = None
        self.train_b_xi_ = None
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            perm = np.random.permutation(self.n)
            
            epoch_a_theta = jnp.zeros((self.n, self.d))
            epoch_b_theta = jnp.zeros((self.n, self.d))
            epoch_a_xi = jnp.zeros(self.n)
            epoch_b_xi = jnp.zeros(self.n)
            
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.n)
                idx = perm[start:end]
                
                X_batch = X[idx]
                y_batch = y[idx]
                X_aux_batch = X_aux[idx]
                
                actual_batch_size = end - start
                scale = self.n / actual_batch_size
                
                # 1. Local parameter optimization
                a_theta, b_theta, a_xi, b_xi, zeta = self._update_local_parameters(
                    X_batch, y_batch, X_aux_batch
                )
                
                epoch_a_theta = epoch_a_theta.at[idx].set(a_theta)
                epoch_b_theta = epoch_b_theta.at[idx].set(b_theta)
                epoch_a_xi = epoch_a_xi.at[idx].set(a_xi)
                epoch_b_xi = epoch_b_xi.at[idx].set(b_xi)
                
                E_theta = a_theta / b_theta
                E_log_theta = jsp.digamma(a_theta) - jnp.log(b_theta)
                
                # 2. Intermediate global parameters
                a_beta_hat, b_beta_hat = self._compute_intermediate_beta(
                    X_batch, E_theta, E_log_theta, scale
                )
                a_eta_hat, b_eta_hat = self._compute_intermediate_eta(scale)
                mu_v_hat, sigma_v_sq_hat = self._compute_intermediate_v(
                    y_batch, X_aux_batch, E_theta, a_theta, b_theta, zeta, scale
                )
                mu_gamma_hat, Sigma_gamma_hat = self._compute_intermediate_gamma(
                    y_batch, X_aux_batch, E_theta, zeta, scale
                )
                
                # 3. SVI update
                rho_t = self._get_learning_rate(iteration)
                self._svi_update_global(
                    rho_t,
                    a_beta_hat, b_beta_hat,
                    a_eta_hat, b_eta_hat,
                    mu_v_hat, sigma_v_sq_hat,
                    mu_gamma_hat, Sigma_gamma_hat
                )
                
                # ELBO tracking
                if iteration % elbo_freq == 0:
                    elbo = self._compute_elbo(
                        X_batch, y_batch, X_aux_batch,
                        a_theta, b_theta, a_xi, b_xi, zeta, scale
                    )
                    if np.isfinite(elbo):
                        self.elbo_history_.append((iteration, elbo))
                        if self.elbo_ema_ is None:
                            self.elbo_ema_ = elbo
                        else:
                            self.elbo_ema_ = self.ema_decay * self.elbo_ema_ + (1 - self.ema_decay) * elbo
                
                iteration += 1
            
            # Store training parameters
            self.train_a_theta_ = epoch_a_theta
            self.train_b_theta_ = epoch_b_theta
            self.train_a_xi_ = epoch_a_xi
            self.train_b_xi_ = epoch_b_xi
            
            # Convergence check
            if self.elbo_ema_prev_ is not None and self.elbo_ema_ is not None:
                rel_change = abs(self.elbo_ema_ - self.elbo_ema_prev_) / (abs(self.elbo_ema_prev_) + 1e-10)
            else:
                rel_change = np.inf
            
            if epoch % 5 == 0:
                self.elbo_ema_prev_ = self.elbo_ema_
                
                # Count near-zero v coefficients (sparsity)
                v_near_zero = np.sum(np.abs(np.array(self.mu_v)) < 0.1)
                v_total = self.kappa * self.d
                sparsity = v_near_zero / v_total
                
                if verbose:
                    ema_str = f"{self.elbo_ema_:.2e}" if self.elbo_ema_ is not None else "N/A"
                    print(f"Epoch {epoch}: EMA={ema_str}, ρ_t={rho_t:.4f}, "
                          f"v_sparsity={sparsity:.1%}, "
                          f"v=[{np.array(self.mu_v).ravel()[:3]}], "
                          f"E[τ⁻¹]=[{np.array(self.E_tau_inv).ravel()[:3]}]")
            
            # Early stopping
            if early_stopping and rel_change < self.convergence_tol:
                consecutive_converged += 1
                if consecutive_converged >= self.convergence_window:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            else:
                consecutive_converged = 0
        
        self.training_time_ = time.time() - start_time
        
        if verbose:
            print(f"\nTraining complete in {self.training_time_:.1f}s")
            
            # Sparsity analysis
            v_abs = np.abs(np.array(self.mu_v))
            print(f"\nv Sparsity Analysis (Laplace b={self.b_laplace}):")
            print(f"  |v| < 0.01: {np.sum(v_abs < 0.01)} / {v_abs.size} ({np.mean(v_abs < 0.01):.1%})")
            print(f"  |v| < 0.1:  {np.sum(v_abs < 0.1)} / {v_abs.size} ({np.mean(v_abs < 0.1):.1%})")
            print(f"  |v| < 0.5:  {np.sum(v_abs < 0.5)} / {v_abs.size} ({np.mean(v_abs < 0.5):.1%})")
            print(f"  v range: [{v_abs.min():.4f}, {v_abs.max():.4f}]")
            print(f"  E[τ⁻¹] range: [{np.array(self.E_tau_inv).min():.4f}, {np.array(self.E_tau_inv).max():.4f}]")
        
        return self
    
    # =========================================================================
    # INFERENCE FOR NEW DATA
    # =========================================================================
    
    def _transform_batch(self, X_batch: jnp.ndarray, y_batch: jnp.ndarray,
                          X_aux_batch: jnp.ndarray, n_iter: int) -> dict:
        """
        Infer θ for a single batch of samples with frozen global parameters.

        This is the core inference routine, called by transform() for each batch.
        """
        batch_size = X_batch.shape[0]

        # Initialize
        row_sums = X_batch.sum(axis=1, keepdims=True) + 1
        factor_scales = jnp.linspace(0.5, 2.0, self.d)

        a_theta = jnp.full((batch_size, self.d), self.alpha_theta + self.p * self.alpha_beta)
        b_theta = row_sums / self.d * factor_scales
        a_xi = jnp.full(batch_size, self.alpha_xi + self.d * self.alpha_theta)
        b_xi = jnp.full(batch_size, self.lambda_xi)

        beta_col_sums = self.E_beta.sum(axis=0)

        for _ in range(n_iter):
            E_theta = a_theta / b_theta
            E_log_theta = jsp.digamma(a_theta) - jnp.log(b_theta)
            E_xi = a_xi / b_xi
            E_theta_sq = (a_theta * (a_theta + 1)) / b_theta**2

            # ξ update
            a_xi = self.alpha_xi + self.d * self.alpha_theta
            b_xi = self.lambda_xi + E_theta.sum(axis=1)
            E_xi = a_xi / b_xi

            # φ and z
            log_phi = E_log_theta[:, jnp.newaxis, :] + self.E_log_beta[jnp.newaxis, :, :]
            log_phi -= logsumexp(log_phi, axis=2, keepdims=True)
            phi = jnp.exp(log_phi)

            # ζ update
            if self.p_aux > 0:
                aux_contrib = X_aux_batch @ self.E_gamma.T
            else:
                aux_contrib = 0.0
            theta_v = E_theta @ self.E_v.T
            E_A = theta_v + aux_contrib
            E_A_sq = E_theta_sq @ self.E_v_sq.T + (2 * theta_v * aux_contrib if self.p_aux > 0 else 0)
            zeta = jnp.sqrt(jnp.maximum(E_A_sq, 1e-10))
            lam = self._lambda_jj(zeta)

            # θ update
            shape_contrib = jnp.einsum('ij,ijl->il', X_batch, phi)
            a_theta = self.alpha_theta + shape_contrib
            b_theta = E_xi[:, jnp.newaxis] + beta_col_sums[jnp.newaxis, :]

            # Regression contribution
            C_minus_ell = (
                E_A[:, :, jnp.newaxis] +
                (aux_contrib[:, :, jnp.newaxis] if self.p_aux > 0 else 0.0) -
                E_theta[:, jnp.newaxis, :] * self.E_v[jnp.newaxis, :, :]
            )

            R = (
                -(y_batch[:, :, jnp.newaxis] - 0.5) * self.E_v[jnp.newaxis, :, :] +
                2 * lam[:, :, jnp.newaxis] * self.E_v[jnp.newaxis, :, :] * C_minus_ell +
                2 * lam[:, :, jnp.newaxis] * self.E_v_sq[jnp.newaxis, :, :] * E_theta[:, jnp.newaxis, :]
            )
            R_sum = R.sum(axis=1)

            # Apply regression_weight to match training (fixes scale mismatch bug)
            b_theta = b_theta + self.regression_weight * R_sum
            b_theta = jnp.maximum(b_theta, 1e-10)

        return {
            'E_theta': a_theta / b_theta,
            'a_theta': a_theta,
            'b_theta': b_theta,
            'a_xi': a_xi,
            'b_xi': b_xi
        }

    def transform(self, X_new: np.ndarray, y_new: np.ndarray = None,
                  X_aux_new: np.ndarray = None, n_iter: int = 50) -> dict:
        """
        Infer θ for new samples with frozen global parameters.

        Processes data in batches to avoid OOM errors on large datasets.
        """
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

        # Process in batches to avoid OOM
        batch_size = self.predict_batch_size
        n_batches = (n_new + batch_size - 1) // batch_size

        # Preallocate output arrays
        all_E_theta = np.zeros((n_new, self.d))
        all_a_theta = np.zeros((n_new, self.d))
        all_b_theta = np.zeros((n_new, self.d))
        all_a_xi = np.zeros(n_new)
        all_b_xi = np.zeros(n_new)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_new)

            X_batch = X_new[start:end]
            y_batch = y_new[start:end]
            X_aux_batch = X_aux_new[start:end]

            result = self._transform_batch(X_batch, y_batch, X_aux_batch, n_iter)

            all_E_theta[start:end] = np.array(result['E_theta'])
            all_a_theta[start:end] = np.array(result['a_theta'])
            all_b_theta[start:end] = np.array(result['b_theta'])
            all_a_xi[start:end] = np.array(result['a_xi'])
            all_b_xi[start:end] = np.array(result['b_xi'])

        return {
            'E_theta': all_E_theta,
            'a_theta': all_a_theta,
            'b_theta': all_b_theta,
            'a_xi': all_a_xi,
            'b_xi': all_b_xi
        }
    
    def predict_proba(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                      n_iter: int = 50) -> np.ndarray:
        """
        Predict class probabilities.

        Processes data in batches to avoid OOM errors on large datasets.
        """
        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, self.p_aux if self.p_aux > 0 else 0))

        # transform already processes in batches
        result = self.transform(X_new, y_new=None, X_aux_new=X_aux_new, n_iter=n_iter)

        # Compute logits in batches to avoid any remaining memory issues
        batch_size = self.predict_batch_size
        n_batches = (n_new + batch_size - 1) // batch_size

        # Preallocate output
        all_proba = np.zeros((n_new, self.kappa))

        mu_v_np = np.array(self.mu_v)
        mu_gamma_np = np.array(self.mu_gamma) if self.p_aux > 0 else None

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_new)

            E_theta_batch = result['E_theta'][start:end]

            if self.p_aux > 0:
                logits = E_theta_batch @ mu_v_np.T + X_aux_new[start:end] @ mu_gamma_np.T
            else:
                logits = E_theta_batch @ mu_v_np.T

            # Use numerically stable sigmoid to avoid overflow/underflow
            all_proba[start:end] = expit(logits)

        return all_proba.squeeze()

    def predict(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                n_iter: int = 50, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Processes data in batches to avoid OOM errors on large datasets.
        """
        proba = self.predict_proba(X_new, X_aux_new, n_iter)
        return (proba >= threshold).astype(int)
    
    def get_sparse_factors(self, threshold: float = 0.1) -> dict:
        """
        Get factors with non-negligible disease association.
        
        Returns
        -------
        dict with:
            'active_factors': indices of factors with |v| > threshold
            'v_values': corresponding v values
            'E_tau_inv': adaptive precision (higher = more shrinkage)
        """
        v_abs = np.abs(np.array(self.mu_v))
        active = v_abs > threshold
        
        results = {
            'active_factors': [],
            'v_values': [],
            'E_tau_inv': [],
            'direction': []
        }
        
        for k in range(self.kappa):
            for ell in range(self.d):
                if active[k, ell]:
                    results['active_factors'].append((k, ell))
                    results['v_values'].append(float(self.mu_v[k, ell]))
                    results['E_tau_inv'].append(float(self.E_tau_inv[k, ell]))
                    results['direction'].append('risk' if self.mu_v[k, ell] > 0 else 'protective')
        
        return results


# Alias
SVI = SVILaplace


if __name__ == "__main__":
    print("Testing SVI with Laplace prior (Bayesian Lasso)...")
    np.random.seed(42)
    
    n_train, n_test, p, d = 200, 50, 50, 5
    
    # Generate data with sparse true v
    theta_true_train = np.random.gamma(2, 1, (n_train, d))
    theta_true_test = np.random.gamma(2, 1, (n_test, d))
    beta_true = np.random.gamma(2, 1, (p, d))
    
    X_train = np.random.poisson(theta_true_train @ beta_true.T)
    X_test = np.random.poisson(theta_true_test @ beta_true.T)
    
    # Sparse true v: only 2 of 5 factors are relevant
    v_true = np.array([[2.0, -1.5, 0.0, 0.0, 0.0]])
    
    logits_train = theta_true_train @ v_true.T
    y_train = (expit(logits_train) > 0.5).astype(float).ravel()
    
    logits_test = theta_true_test @ v_true.T
    y_test = (expit(logits_test) > 0.5).astype(float).ravel()
    
    X_aux_train = np.random.randn(n_train, 2)
    X_aux_test = np.random.randn(n_test, 2)
    
    # Test with different Laplace scales
    for b in [0.5, 1.0, 2.0]:
        print(f"\n{'='*60}")
        print(f"Testing with b_laplace = {b}")
        print(f"{'='*60}")
        
        model = SVILaplace(
            n_factors=d,
            batch_size=64,
            learning_rate=0.5,
            learning_rate_decay=0.6,
            b_laplace=b,
            local_iterations=20,
            random_state=42
        )
        model.fit(X_train, y_train, X_aux_train, max_epochs=50, verbose=True)
        
        print(f"\nTrue v:      {v_true.ravel()}")
        print(f"Learned v:   {np.array(model.mu_v).ravel()}")
        print(f"Correlation: {np.corrcoef(np.array(model.mu_v).ravel(), v_true.ravel())[0,1]:.3f}")
        
        # Sparsity recovery
        sparse_factors = model.get_sparse_factors(threshold=0.1)
        print(f"Active factors (|v|>0.1): {len(sparse_factors['active_factors'])}")
        
        # Prediction
        y_pred = model.predict(X_test, X_aux_test)
        acc = (y_pred == y_test).mean()
        print(f"Test accuracy: {acc:.3f}")