"""
Corrected Stochastic Variational Inference for Supervised Poisson Factorization
================================================================================

JAX-optimized version with GPU acceleration and sparse matrix support.

Key corrections from the original implementation:
1. Natural gradient updates for exponential family parameters
2. Proper intermediate parameter computation following Hoffman et al. (2013)
3. Consistent scaling for mini-batch -> full dataset extrapolation
4. Vectorized operations for GPU efficiency
5. Sparse matrix support for gene expression data

For Gamma(a, b): natural params η = (a-1, -b), sufficient stats t(x) = (log x, x)
For Gaussian N(μ, Σ): natural params η = (Σ⁻¹μ, -½Σ⁻¹), sufficient stats t(x) = (x, xxᵀ)

SVI Update (Eq. 34 from Hoffman et al.):
    λ^(t) = (1 - ρ_t) λ^(t-1) + ρ_t * λ̂
where λ̂ = α + N · E_φ[t(x_i, z_i)] is the intermediate parameter.

References:
- Hoffman et al. (2013) "Stochastic Variational Inference", JMLR
- Your PSB paper derivations (A.24-A.60)
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.scipy.special import logsumexp
from jax import random
from typing import Tuple, Optional, Dict, Any
import time
import numpy as np
import scipy.sparse as sp


class SVICorrected:
    """
    Corrected SVI implementation with JAX acceleration and sparse matrix support.
    
    Key architectural decisions:
    - Global params (β, η, v, γ): Updated via SVI with natural gradients
    - Local params (θ, ξ, ζ): Fully optimized per mini-batch
    - Vectorized operations for GPU efficiency
    - Sparse matrix support for gene expression data
    """
    
    def __init__(
        self,
        n_factors: int,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.75,  # κ in ρ_t = (τ + t)^{-κ}
        learning_rate_delay: float = 1.0,   # τ
        learning_rate_min: float = 1e-2,
        local_iterations: int = 10,
        
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
        count_scale: Optional[float] = None,  # For compatibility with bayes_opt
        
        # Convergence tracking
        ema_decay: float = 0.95,
        convergence_tol: float = 1e-4,
        convergence_window: int = 10,
        
        random_state: Optional[int] = None,
        
        # JAX-specific
        use_jit: bool = True,
        device: str = 'gpu'  # 'gpu' or 'cpu'
    ):
        self.d = n_factors  # Number of factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_delay = learning_rate_delay
        self.learning_rate_min = learning_rate_min
        self.local_iterations = local_iterations
        
        # Prior hyperparameters
        self.alpha_theta = alpha_theta
        self.alpha_beta = alpha_beta
        self.alpha_xi = alpha_xi
        self.alpha_eta = alpha_eta
        self.lambda_xi = lambda_xi
        self.lambda_eta = lambda_eta
        self.sigma_v = sigma_v
        self.sigma_gamma = sigma_gamma
        
        # Spike-and-slab
        self.use_spike_slab = use_spike_slab
        self.pi_beta = pi_beta
        self.pi_v = pi_v
        
        self.regression_weight = regression_weight
        
        # Convergence tracking (EMA + Welford)
        self.ema_decay = ema_decay
        self.convergence_tol = convergence_tol
        self.convergence_window = convergence_window
        self.count_scale = count_scale  # Optional count scaling (for compatibility)
        
        # JAX settings
        self.use_jit = use_jit
        self.device = device
        
        # RNG - JAX uses explicit random keys
        self.rng_key = random.PRNGKey(random_state if random_state is not None else 0)
        
        # Will be set during fit
        self.n = None  # num samples
        self.p = None  # num genes
        self.kappa = None  # num outcomes
        self.p_aux = None  # num auxiliary features
        
    def _get_learning_rate(self, t: int) -> float:
        """
        Robbins-Monro schedule: ρ_t = lr * (τ + t)^{-κ}
        Satisfies: Σ ρ_t = ∞, Σ ρ_t² < ∞
        """
        rho = self.learning_rate * (self.learning_rate_delay + t) ** (-self.learning_rate_decay)
        return max(rho, self.learning_rate_min)
    
    @staticmethod
    @jax.jit
    def _lambda_jj(zeta: jnp.ndarray) -> jnp.ndarray:
        """
        Jaakkola-Jordan auxiliary function: λ(ζ) = tanh(ζ/2) / (4ζ)
        For ζ→0: λ(0) = 1/8
        
        Vectorized and numerically stable.
        JIT-compiled for performance.
        """
        # Use jnp.where for conditional computation
        nonzero_mask = jnp.abs(zeta) > 1e-8
        result = jnp.where(
            nonzero_mask,
            jnp.tanh(zeta / 2) / (4 * zeta + 1e-20),
            0.125
        )
        return result
    
    # =========================================================================
    # NATURAL PARAMETER CONVERSIONS
    # =========================================================================
    
    @staticmethod
    @jax.jit
    def _gamma_to_natural(a: jnp.ndarray, b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Gamma(a, b) → natural parameters (η₁, η₂) = (a-1, -b)
        JIT-compiled for performance.
        """
        return a - 1, -b
    
    @staticmethod
    @jax.jit
    def _natural_to_gamma(eta1: jnp.ndarray, eta2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Natural parameters (η₁, η₂) → Gamma(a, b) = (η₁+1, -η₂)
        JIT-compiled for performance.
        """
        a = eta1 + 1
        b = -eta2
        # Ensure valid parameters
        a = jnp.maximum(a, 1.001)
        b = jnp.maximum(b, 1e-6)
        return a, b
    
    @staticmethod
    @jax.jit
    def _gaussian_to_natural(mu: jnp.ndarray, Sigma: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        N(μ, Σ) → natural parameters (η₁, η₂) = (Σ⁻¹μ, -½Σ⁻¹)
        JIT-compiled for performance.
        """
        Sigma_inv = jnp.linalg.inv(Sigma + 1e-6 * jnp.eye(Sigma.shape[0]))
        eta1 = Sigma_inv @ mu
        eta2 = -0.5 * Sigma_inv
        return eta1, eta2
    
    @staticmethod
    @jax.jit
    def _natural_to_gaussian(eta1: jnp.ndarray, eta2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Natural parameters (η₁, η₂) → N(μ, Σ)
        η₁ = Σ⁻¹μ, η₂ = -½Σ⁻¹
        ⟹ Σ = -½ η₂⁻¹, μ = Σ η₁
        JIT-compiled for performance.
        """
        # Add regularization for numerical stability
        eta2_reg = eta2 - 1e-6 * jnp.eye(eta2.shape[0])
        Sigma = -0.5 * jnp.linalg.inv(eta2_reg)
        # Ensure symmetry and positive definiteness
        Sigma = 0.5 * (Sigma + Sigma.T)
        eigvals = jnp.linalg.eigvalsh(Sigma)
        min_eigval = jnp.min(eigvals)
        Sigma = jnp.where(
            min_eigval < 1e-6,
            Sigma + (1e-6 - min_eigval + 1e-8) * jnp.eye(Sigma.shape[0]),
            Sigma
        )
        mu = Sigma @ eta1
        return mu, Sigma
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _initialize_global_parameters(self, X: jnp.ndarray, y: jnp.ndarray, X_aux: jnp.ndarray):
        """Initialize global variational parameters with factor diversity."""
        self.n, self.p = X.shape
        self.kappa = 1 if y.ndim == 1 else y.shape[1]
        self.p_aux = X_aux.shape[1] if X_aux is not None and X_aux.size > 0 else 0
        
        # Split RNG key for different initializations
        key1, key2, key3, self.rng_key = random.split(self.rng_key, 4)
        
        # β: Gene loadings, Gamma(a_β, b_β)
        # Initialize with HIGH DIVERSITY to break symmetry and explore gene programs
        col_means = X.mean(axis=0) + 1  # (p,)
        
        # Strategy: Each factor gets a random subset of genes with high loadings
        # This creates diverse gene programs from the start
        
        # Base initialization - start from prior
        self.a_beta = jnp.full((self.p, self.d), self.alpha_beta)
        self.b_beta = jnp.full((self.p, self.d), 1.0)
        
        # Add large random variations to create diversity
        # Each factor gets different random gene subsets boosted
        noise_scale = 5.0  # Increased from 0.2 to create MUCH more diversity
        random_boost = random.uniform(key1, (self.p, self.d), minval=0.1, maxval=3.0)
        
        # Add data-driven signal: scale by gene means
        gene_scale = col_means[:, jnp.newaxis]  # (p, 1)
        
        # Combine: base + gene_scale * random_boost
        self.a_beta = self.a_beta + gene_scale * random_boost * noise_scale
        
        # Create factor-specific "signatures" - each factor prefers different gene sets
        # Randomly assign genes to factors with varying strengths
        factor_signatures = random.uniform(key2, (self.p, self.d), minval=0.0, maxval=2.0)
        
        # Make some factors sparse, some dense (varying sparsity patterns)
        sparsity_masks = random.bernoulli(key3, p=0.3, shape=(self.p, self.d))  # 30% active
        
        # Apply signature with sparsity
        signature_boost = factor_signatures * sparsity_masks * gene_scale * 3.0
        self.a_beta = self.a_beta + signature_boost
        
        # Ensure all values are positive and reasonable
        self.a_beta = jnp.maximum(self.a_beta, self.alpha_beta * 0.1)
        
        # η: Gene activities, Gamma(a_η, b_η)
        self.a_eta = jnp.full(self.p, self.alpha_eta + self.d * self.alpha_beta)
        self.b_eta = jnp.full(self.p, self.lambda_eta)
        
        # v: Regression coefficients, N(μ_v, Σ_v)
        # Initialize with stronger signal to help learn discrimination
        # Use larger initial scale (0.5 instead of 0.1) for better exploration
        v_init_scale = 0.5
        key_v1, key_v2 = random.split(key2, 2)
        self.mu_v = v_init_scale * random.normal(key_v1, (self.kappa, self.d))
        self.Sigma_v = jnp.array([jnp.eye(self.d) * self.sigma_v**2 for _ in range(self.kappa)])
        
        # γ: Auxiliary coefficients, N(μ_γ, Σ_γ)
        if self.p_aux > 0:
            self.mu_gamma = jnp.zeros((self.kappa, self.p_aux))
            self.Sigma_gamma = jnp.array([jnp.eye(self.p_aux) * self.sigma_gamma**2 for _ in range(self.kappa)])
        else:
            self.mu_gamma = jnp.zeros((self.kappa, 0))
            self.Sigma_gamma = jnp.zeros((self.kappa, 0, 0))
        
        # Spike-and-slab indicators
        if self.use_spike_slab:
            self.rho_beta = jnp.full((self.p, self.d), self.pi_beta)
            self.rho_v = jnp.full((self.kappa, self.d), self.pi_v)
        else:
            self.rho_beta = jnp.ones((self.p, self.d))
            self.rho_v = jnp.ones((self.kappa, self.d))
        
        # Store natural parameters for SVI updates
        self._update_natural_params()
        self._compute_expectations()
        
        # Store initial v for tracking learning progress
        self.initial_mu_v_ = self.mu_v.copy()

        # Debug: check initial diversity
        print(f"Initial beta diversity: {np.std(np.array(self.E_beta), axis=1).mean():.4f}")
        print(f"Initial v: {self.mu_v}")
    
    def _update_natural_params(self):
        """Convert canonical to natural parameters for SVI."""
        # Beta: Gamma
        self.eta1_beta, self.eta2_beta = self._gamma_to_natural(self.a_beta, self.b_beta)
        # Eta: Gamma
        self.eta1_eta, self.eta2_eta = self._gamma_to_natural(self.a_eta, self.b_eta)
        # v: Gaussian (per outcome)
        self.eta1_v = []
        self.eta2_v = []
        for k in range(self.kappa):
            e1, e2 = self._gaussian_to_natural(self.mu_v[k], self.Sigma_v[k])
            self.eta1_v.append(e1)
            self.eta2_v.append(e2)
        # gamma: Gaussian
        self.eta1_gamma = []
        self.eta2_gamma = []
        for k in range(self.kappa):
            e1, e2 = self._gaussian_to_natural(self.mu_gamma[k], self.Sigma_gamma[k])
            self.eta1_gamma.append(e1)
            self.eta2_gamma.append(e2)
    
    def _convert_natural_to_canonical(self):
        """Convert natural back to canonical parameters."""
        self.a_beta, self.b_beta = self._natural_to_gamma(self.eta1_beta, self.eta2_beta)
        self.a_eta, self.b_eta = self._natural_to_gamma(self.eta1_eta, self.eta2_eta)
        for k in range(self.kappa):
            self.mu_v = self.mu_v.at[k].set(self._natural_to_gaussian(self.eta1_v[k], self.eta2_v[k])[0])
            self.Sigma_v = self.Sigma_v.at[k].set(self._natural_to_gaussian(self.eta1_v[k], self.eta2_v[k])[1])
            if self.p_aux > 0:
                self.mu_gamma = self.mu_gamma.at[k].set(self._natural_to_gaussian(self.eta1_gamma[k], self.eta2_gamma[k])[0])
                self.Sigma_gamma = self.Sigma_gamma.at[k].set(self._natural_to_gaussian(self.eta1_gamma[k], self.eta2_gamma[k])[1])
    
    def _compute_expectations(self):
        """Compute expected sufficient statistics from current parameters."""
        # E[β], E[log β]
        self.E_beta = self.a_beta / self.b_beta
        self.E_log_beta = jsp.digamma(self.a_beta) - jnp.log(self.b_beta)
        
        # E[η], E[log η]
        self.E_eta = self.a_eta / self.b_eta
        self.E_log_eta = jsp.digamma(self.a_eta) - jnp.log(self.b_eta)
        
        # E[v] = μ_v (for Gaussian)
        self.E_v = self.mu_v.copy()
        
        # E[γ] = μ_γ
        self.E_gamma = self.mu_gamma.copy()
        
        # Apply spike-and-slab masking
        if self.use_spike_slab:
            self.E_beta_effective = self.rho_beta * self.E_beta
            self.E_log_beta_effective = self.rho_beta * self.E_log_beta + \
                                        (1 - self.rho_beta) * (-20)  # log(ε)
            self.E_v_effective = self.rho_v * self.E_v
        else:
            self.E_beta_effective = self.E_beta
            self.E_log_beta_effective = self.E_log_beta
            self.E_v_effective = self.E_v

    def _update_spike_slab(self, E_theta: jnp.ndarray, E_beta: jnp.ndarray,
                           a_theta: jnp.ndarray, b_theta: jnp.ndarray):
        """
        Update spike-and-slab inclusion probabilities.

        For β: ρ_jℓ uses log-odds ratio approach comparing slab vs spike contribution.
        For v: ρ_kℓ uses SNR-based evidence with log(1+SNR) compression.

        This implements a variational approximation to the spike-and-slab prior:
        - Spike: δ_0 (point mass at zero, factor inactive)
        - Slab: Gamma(α_β, η) for β, N(0, σ_v²) for v

        The inclusion probability ρ represents P(factor is active).
        A floor of 0.1 is used for rho_v to keep all factors somewhat active.
        """
        if not self.use_spike_slab:
            return

        # Log prior odds for β and v
        log_prior_odds_beta = jnp.log(self.pi_beta / (1 - self.pi_beta + 1e-10))
        log_prior_odds_v = jnp.log(self.pi_v / (1 - self.pi_v + 1e-10))

        # === Update ρ_β (gene loading inclusion) ===
        # Compare expected contribution under slab vs spike
        # Under slab: E[β_jℓ] contributes to likelihood
        # Under spike: β_jℓ = 0, no contribution
        # Use log-likelihood ratio: how much does having this β help explain data?
        # Approximation: larger E[β] relative to threshold suggests inclusion

        # Evidence for inclusion: magnitude of E[β] scaled by prior expectations
        # Large E[β] → high inclusion probability
        log_lik_ratio_beta = jnp.log(self.E_beta + 1e-10) - jnp.log(0.01)  # vs spike at ε

        # Posterior inclusion probability using sigmoid
        log_odds_beta = log_prior_odds_beta + log_lik_ratio_beta
        self.rho_beta = jsp.expit(jnp.clip(log_odds_beta, -20, 20))

        # === Update ρ_v (regression coefficient inclusion) ===
        # Evidence for v factor inclusion: how much does factor ℓ contribute to prediction?
        # Use |μ_v| / σ_v as evidence (signal-to-noise ratio for factor importance)

        # Signal-to-noise ratio for each factor
        v_snr = jnp.abs(self.mu_v) / (jnp.sqrt(jnp.diagonal(self.Sigma_v, axis1=1, axis2=2)) + 1e-10)

        # Use a soft threshold: log(1 + SNR) to compress range
        log_lik_ratio_v = jnp.log1p(v_snr)

        # Posterior inclusion probability
        log_odds_v = log_prior_odds_v + log_lik_ratio_v
        self.rho_v = jsp.expit(jnp.clip(log_odds_v, -10, 10))

        # Floor: keep all factors at least somewhat active
        self.rho_v = jnp.maximum(self.rho_v, 0.1)

        # Update effective expectations with new inclusion probabilities
        self._compute_expectations()

    # =========================================================================
    # LOCAL PARAMETER UPDATES (Full optimization per mini-batch)
    # =========================================================================
    
    def _update_local_parameters(
        self,
        X_batch: jnp.ndarray,
        y_batch: jnp.ndarray,
        X_aux_batch: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Optimize local parameters (θ, ξ, ζ) for a mini-batch.
        
        VECTORIZED: Computes all outcomes simultaneously.
        
        Returns: (a_theta, b_theta, a_xi, b_xi, zeta)
        """
        batch_size = X_batch.shape[0]
        
        # Initialize local params
        row_sums = X_batch.sum(axis=1, keepdims=True) + 1
        theta_init = row_sums / self.d
        
        a_theta = jnp.full((batch_size, self.d), self.alpha_theta) + theta_init
        b_theta = jnp.full((batch_size, self.d), 1.0)
        a_xi = jnp.full(batch_size, self.alpha_xi)
        b_xi = jnp.full(batch_size, self.lambda_xi)
        zeta = jnp.ones((batch_size, self.kappa))
        
        for _ in range(self.local_iterations):
            # Current expectations
            E_theta = a_theta / b_theta
            E_log_theta = jsp.digamma(a_theta) - jnp.log(b_theta)
            E_xi = a_xi / b_xi
            
            # Update φ (multinomial allocations) - implicit in z expectations
            # φ_ijℓ ∝ exp(E[log θ_iℓ] + E[log β_jℓ])
            log_phi = E_log_theta[:, jnp.newaxis, :] + self.E_log_beta_effective[jnp.newaxis, :, :]
            log_phi_max = log_phi.max(axis=2, keepdims=True)
            phi = jnp.exp(log_phi - log_phi_max)
            phi = phi / (phi.sum(axis=2, keepdims=True) + 1e-10)
            
            # E[z_ijℓ] = x_ij * φ_ijℓ
            E_z = X_batch[:, :, jnp.newaxis] * phi  # (batch, p, d)
            
            # Update θ: Gamma(a_θ, b_θ)
            # VECTORIZED over all outcomes
            a_theta_new = self.alpha_theta + E_z.sum(axis=1)
            
            b_theta_new = E_xi[:, jnp.newaxis] + self.E_beta_effective.sum(axis=0)[jnp.newaxis, :]
            
            # === VECTORIZED regression contribution ===
            # Shape: (batch, kappa, d)
            lam = self._lambda_jj(zeta)  # (batch, kappa)
            
            # Compute theta_v for all outcomes at once: (batch, kappa)
            theta_v = E_theta @ self.E_v_effective.T  # (batch, d) @ (d, kappa) -> (batch, kappa)
            
            # Auxiliary contribution: (batch, kappa)
            if self.p_aux > 0:
                aux_term = X_aux_batch @ self.E_gamma.T  # (batch, p_aux) @ (p_aux, kappa) -> (batch, kappa)
            else:
                aux_term = 0.0
            
            # For each factor ℓ, compute C^{(-ℓ)} = sum_{m≠ℓ} E[θ_im]E[v_km] + aux_term
            # = (full θ·v + aux_term) - E[θ_iℓ]E[v_kℓ]
            # Broadcasting: (batch, kappa, 1) -> (batch, kappa, d)
            full_linear_predictor = theta_v[:, :, jnp.newaxis]  # (batch, kappa, 1)
            if self.p_aux > 0:
                full_linear_predictor = full_linear_predictor + aux_term[:, :, jnp.newaxis]

            C_minus_ell = (
                full_linear_predictor -  # Full contribution: θ·v + aux
                E_theta[:, jnp.newaxis, :] * self.E_v_effective[jnp.newaxis, :, :]  # Subtract ℓ-th term
            )  # Result: (batch, kappa, d)
            
            # E[v²_kℓ] for all outcomes: (kappa, d)
            E_v_sq = self.mu_v**2 + jnp.diagonal(self.Sigma_v, axis1=1, axis2=2)  # (kappa, d)
            
            # Expand y_batch to (batch, kappa)
            y_expanded = y_batch if y_batch.ndim > 1 else y_batch[:, jnp.newaxis]
            
            # Compute R for all outcomes and factors at once
            # R shape: (batch, kappa, d)
            R = (
                -(y_expanded[:, :, jnp.newaxis] - 0.5) * self.E_v_effective[jnp.newaxis, :, :] +
                2 * lam[:, :, jnp.newaxis] * self.E_v_effective[jnp.newaxis, :, :] * C_minus_ell +
                2 * lam[:, :, jnp.newaxis] * E_v_sq[jnp.newaxis, :, :] * E_theta[:, jnp.newaxis, :]
            )
            
            # Sum over outcomes: (batch, d)
            R_sum = R.sum(axis=1)
            
            b_theta_new = b_theta_new + self.regression_weight * R_sum
            
            b_theta_new = jnp.maximum(b_theta_new, 1e-6)
            a_theta_new = jnp.maximum(a_theta_new, 1.001)
            
            # Update ξ: Gamma(a_ξ, b_ξ)
            E_theta_new = a_theta_new / b_theta_new
            a_xi_new = jnp.full(batch_size, self.alpha_xi + self.d * self.alpha_theta)
            b_xi_new = self.lambda_xi + E_theta_new.sum(axis=1)
            
            # Update ζ (JJ auxiliary) - vectorized over all outcomes
            if self.p_aux > 0:
                aux_contrib = X_aux_batch @ self.E_gamma.T  # (batch, kappa)
            else:
                aux_contrib = 0.0
            
            E_A = E_theta_new @ self.E_v_effective.T + aux_contrib  # (batch, kappa)
            Var_theta = a_theta_new / (b_theta_new**2)  # (batch, d)
            # E[A²] = E[A]² + Var[A], where Var[A] = sum_ell Var[θ_iℓ] E[v²_kℓ]
            E_A_sq = E_A**2 + (Var_theta @ E_v_sq.T)  # (batch, kappa)
            zeta = jnp.sqrt(jnp.maximum(E_A_sq, 1e-8))
            
            a_theta, b_theta = a_theta_new, b_theta_new
            a_xi, b_xi = a_xi_new, b_xi_new
        
        return a_theta, b_theta, a_xi, b_xi, zeta
    
    # =========================================================================
    # INTERMEDIATE GLOBAL PARAMETERS (as if mini-batch were full dataset)
    # =========================================================================
    
    def _compute_intermediate_beta(
        self,
        X_batch: jnp.ndarray,
        E_theta: jnp.ndarray,
        E_log_theta: jnp.ndarray,
        scale: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute intermediate β parameters using natural gradient.

        From Hoffman et al., the intermediate parameter is:
        λ̂ = α + N · E[t(x, z)]

        For β ~ Gamma(α_β, η_j), the complete conditional natural params are:
        η₁ = α_β - 1 + Σ_i z_ijℓ
        η₂ = -η_j - Σ_i θ_iℓ
        """
        # Compute E[z_ijℓ] = x_ij · φ_ijℓ
        log_phi = E_log_theta[:, jnp.newaxis, :] + self.E_log_beta_effective[jnp.newaxis, :, :]
        log_phi = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
        phi = jnp.exp(log_phi)

        # Sufficient statistics
        # Σ_i z_ijℓ for each gene j and factor ℓ
        z_sum = (X_batch[:, :, jnp.newaxis] * phi).sum(axis=0)  # (p, d)
        theta_sum = E_theta.sum(axis=0)  # (d,)

        # Intermediate natural parameters (pretend batch is full dataset)
        eta1_hat = (self.alpha_beta - 1) + scale * z_sum
        eta2_hat = -self.E_eta[:, jnp.newaxis] - scale * theta_sum[jnp.newaxis, :]

        # Convert to canonical
        a_beta_hat, b_beta_hat = self._natural_to_gamma(eta1_hat, eta2_hat)

        return a_beta_hat, b_beta_hat
    
    def _compute_intermediate_eta(
        self,
        E_theta: jnp.ndarray,
        scale: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute intermediate η parameters.
        
        η_j ~ Gamma(α_η, λ_η)
        Complete conditional: η_j | β_j ~ Gamma(α_η + d·α_β, λ_η + Σ_ℓ β_jℓ)
        """
        # Sufficient statistic: Σ_ℓ β_jℓ
        beta_sum = self.E_beta_effective.sum(axis=1)  # (p,)
        
        eta1_hat = (self.alpha_eta - 1) + self.d * self.alpha_beta
        eta2_hat = -self.lambda_eta - beta_sum
        
        a_eta_hat, b_eta_hat = self._natural_to_gamma(
            jnp.full(self.p, eta1_hat), eta2_hat
        )
        
        return a_eta_hat, b_eta_hat
    
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
        Compute intermediate v parameters.

        VECTORIZED: Computes all outcomes and factors simultaneously.

        v_k ~ N(0, σ_v² I)
        From JJ bound, the approximate posterior is Gaussian with:
        - Precision: (1/σ_v²) I + 2 Σ_i λ(ζ_ik) E[θ_i⊗θ_i]
        - Mean precision: Σ_i (y_ik - 0.5) E[θ_i] - 2λ(ζ_ik) E[θ_i](x^aux · γ_k)

        For diagonal approximation (independent v_{kℓ}):
        - precision_ℓ = 1/σ_v² + 2 Σ_i λ(ζ_ik) E[θ²_iℓ]
        - mean·precision_ℓ = Σ_i [(y_ik - 0.5) - 2λ(ζ_ik) C_{ik}] E[θ_iℓ]

        where C_{ik} = Σ_{m≠ℓ} E[θ_im] E[v_km] + x^aux · γ_k

        Output is clipped to [-5, 5] for numerical stability.
        """
        # Expand y to (batch, kappa)
        y_expanded = y_batch if y_batch.ndim > 1 else y_batch[:, jnp.newaxis]

        lam = self._lambda_jj(zeta)  # (batch, kappa)

        # E[θ²] = E[θ]² + Var[θ]
        Var_theta = a_theta / (b_theta**2)  # (batch, d)
        E_theta_sq = E_theta**2 + Var_theta  # (batch, d)

        # Auxiliary contribution: (batch, kappa)
        if self.p_aux > 0:
            aux_contrib = X_aux_batch @ self.E_gamma.T  # (batch, kappa)
        else:
            aux_contrib = 0.0

        # Full model contribution: (batch, kappa)
        full_theta_v = E_theta @ self.E_v_effective.T  # (batch, kappa)

        # === VECTORIZED computation over all outcomes and factors ===
        # Precision for each (outcome, factor): (kappa, d)
        precision = jnp.full((self.kappa, self.d), 1.0 / self.sigma_v**2)

        # Precision contribution: 2 Σ_i λ(ζ_ik) E[θ²_iℓ]
        # Shape: (kappa, d) += sum over batch of (batch, kappa, d)
        precision_contrib = 2 * scale * jnp.einsum('ik,id->kd', lam, E_theta_sq)
        precision = precision + precision_contrib

        # For each factor ℓ, compute C^{(-ℓ)} = sum_{m≠ℓ} E[θ_im]E[v_km] + aux_term
        # = (full θ·v + aux_term) - E[θ_iℓ]E[v_kℓ]
        full_linear_predictor = full_theta_v[:, :, jnp.newaxis]  # (batch, kappa, 1)
        if self.p_aux > 0:
            full_linear_predictor = full_linear_predictor + aux_contrib[:, :, jnp.newaxis]

        C_minus_ell = (
            full_linear_predictor -  # Full contribution: θ·v + aux
            E_theta[:, jnp.newaxis, :] * self.E_v_effective[jnp.newaxis, :, :]  # Subtract ℓ-th term
        )

        # Mean*precision contribution: Σ_i [(y_ik - 0.5) - 2λ(ζ_ik) C_{ik}^{(-ℓ)}] E[θ_iℓ]
        # Shape: (kappa, d)
        # Vectorized computation using einsum
        term1 = scale * jnp.einsum('ik,id->kd', y_expanded - 0.5, E_theta)
        term2 = 2 * scale * jnp.einsum('ik,ikd,id->kd', lam, C_minus_ell, E_theta)
        mean_contrib = term1 - term2

        # Convert to mean and variance
        Sigma_v_hat = jnp.zeros((self.kappa, self.d, self.d))
        for k in range(self.kappa):
            Sigma_v_hat = Sigma_v_hat.at[k].set(jnp.diag(1.0 / precision[k]))

        mu_v_hat = mean_contrib / precision

        # Clip for stability - use moderate range to prevent extreme drift
        # The -15 boundary was causing v's to get stuck at extremes
        mu_v_hat = jnp.clip(mu_v_hat, -5, 5)

        return mu_v_hat, Sigma_v_hat
    
    def _compute_intermediate_gamma(
        self,
        y_batch: jnp.ndarray,
        X_aux_batch: jnp.ndarray,
        E_theta: jnp.ndarray,
        zeta: jnp.ndarray,
        scale: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute intermediate γ parameters.
        
        VECTORIZED: Computes all outcomes simultaneously where possible.
        """
        if self.p_aux == 0:
            return jnp.zeros((self.kappa, 0)), jnp.zeros((self.kappa, 0, 0))
        
        # Expand y to (batch, kappa)
        y_expanded = y_batch if y_batch.ndim > 1 else y_batch[:, jnp.newaxis]
        
        lam = self._lambda_jj(zeta)  # (batch, kappa)
        
        mu_gamma_hat = jnp.zeros((self.kappa, self.p_aux))
        Sigma_gamma_hat = jnp.zeros((self.kappa, self.p_aux, self.p_aux))
        
        # Loop over outcomes (could be vectorized further but might be memory-intensive)
        for k in range(self.kappa):
            # Precision
            prec_prior = jnp.eye(self.p_aux) / self.sigma_gamma**2
            # Vectorized: X_aux.T @ diag(lam[:, k]) @ X_aux
            prec_lik = 2 * scale * (X_aux_batch.T * lam[:, k]) @ X_aux_batch
            prec_hat = prec_prior + prec_lik
            
            # Mean
            theta_v = E_theta @ self.E_v_effective[k]  # (batch,)
            mean_contrib = scale * X_aux_batch.T @ (y_expanded[:, k] - 0.5 - 2 * lam[:, k] * theta_v)
            
            Sigma_gamma_hat_k = jnp.linalg.inv(prec_hat)
            mu_gamma_hat_k = Sigma_gamma_hat_k @ mean_contrib
            
            mu_gamma_hat = mu_gamma_hat.at[k].set(mu_gamma_hat_k)
            Sigma_gamma_hat = Sigma_gamma_hat.at[k].set(Sigma_gamma_hat_k)
        
        return mu_gamma_hat, Sigma_gamma_hat
    
    # =========================================================================
    # SVI UPDATES (Natural Gradient)
    # =========================================================================
    
    def _svi_update_global(
        self,
        rho_t: float,
        a_beta_hat: jnp.ndarray, b_beta_hat: jnp.ndarray,
        a_eta_hat: jnp.ndarray, b_eta_hat: jnp.ndarray,
        mu_v_hat: jnp.ndarray, Sigma_v_hat: jnp.ndarray,
        mu_gamma_hat: jnp.ndarray, Sigma_gamma_hat: jnp.ndarray
    ):
        """
        SVI update.
        
        For Gamma distributions, we use natural parameter updates:
        η^(t) = (1 - ρ_t) η^(t-1) + ρ_t η̂
        
        For Gaussian distributions, we update canonical parameters directly
        (equivalent when done properly, but more numerically stable):
        μ^(t) = (1 - ρ_t) μ^(t-1) + ρ_t μ̂
        Σ^(t) = (1 - ρ_t) Σ^(t-1) + ρ_t Σ̂
        """
        # Beta: update natural parameters for Gamma
        eta1_beta_hat, eta2_beta_hat = self._gamma_to_natural(a_beta_hat, b_beta_hat)
        self.eta1_beta = (1 - rho_t) * self.eta1_beta + rho_t * eta1_beta_hat
        self.eta2_beta = (1 - rho_t) * self.eta2_beta + rho_t * eta2_beta_hat
        self.a_beta, self.b_beta = self._natural_to_gamma(self.eta1_beta, self.eta2_beta)
        
        # Eta: update natural parameters for Gamma
        eta1_eta_hat, eta2_eta_hat = self._gamma_to_natural(a_eta_hat, b_eta_hat)
        self.eta1_eta = (1 - rho_t) * self.eta1_eta + rho_t * eta1_eta_hat
        self.eta2_eta = (1 - rho_t) * self.eta2_eta + rho_t * eta2_eta_hat
        self.a_eta, self.b_eta = self._natural_to_gamma(self.eta1_eta, self.eta2_eta)
        
        # v: update canonical parameters directly (more stable)
        for k in range(self.kappa):
            self.mu_v = self.mu_v.at[k].set((1 - rho_t) * self.mu_v[k] + rho_t * mu_v_hat[k])
            Sigma_new = (1 - rho_t) * self.Sigma_v[k] + rho_t * Sigma_v_hat[k]
            # Ensure positive definiteness
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
            eigvals = jnp.linalg.eigvalsh(Sigma_new)
            min_eigval = jnp.min(eigvals)
            Sigma_new = jnp.where(
                min_eigval < 1e-6,
                Sigma_new + (1e-6 - min_eigval + 1e-8) * jnp.eye(self.d),
                Sigma_new
            )
            self.Sigma_v = self.Sigma_v.at[k].set(Sigma_new)
        
        # gamma: update canonical parameters directly
        if self.p_aux > 0:
            for k in range(self.kappa):
                self.mu_gamma = self.mu_gamma.at[k].set((1 - rho_t) * self.mu_gamma[k] + rho_t * mu_gamma_hat[k])
                Sigma_new = (1 - rho_t) * self.Sigma_gamma[k] + rho_t * Sigma_gamma_hat[k]
                Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
                eigvals = jnp.linalg.eigvalsh(Sigma_new)
                min_eigval = jnp.min(eigvals)
                Sigma_new = jnp.where(
                    min_eigval < 1e-6,
                    Sigma_new + (1e-6 - min_eigval + 1e-8) * jnp.eye(self.p_aux),
                    Sigma_new
                )
                self.Sigma_gamma = self.Sigma_gamma.at[k].set(Sigma_new)
        
        # Update expectations
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
        Compute ELBO for mini-batch, scaled to full dataset.
        
        ELBO = E[log p(X, y, z, θ, ξ, β, η, v, γ)] - E[log q(all)]
        
        Statistical functions used from JAX/SciPy:
        - jsp.gammaln(x): log(Γ(x)) - log of gamma function (factorial for real numbers: log(x!) = log(Γ(x+1)))
        - jsp.digamma(x): ψ(x) = d/dx log(Γ(x)) - derivative of log gamma
        - jnp.linalg.slogdet: numerically stable log determinant
        - logsumexp: numerically stable log(sum(exp(x)))
        
        All PDFs computed using proper statistical formulas:
        - Poisson: P(x|λ) = λ^x exp(-λ) / Γ(x+1)
        - Gamma: p(x|a,b) = b^a x^(a-1) exp(-bx) / Γ(a)
        - Gaussian: p(x|μ,Σ) = (2π)^(-d/2) |Σ|^(-1/2) exp(-1/2 (x-μ)'Σ^(-1)(x-μ))
        """
        batch_size = X_batch.shape[0]
        elbo = 0.0
        
        # Local expectations
        E_theta = a_theta / b_theta
        E_log_theta = jsp.digamma(a_theta) - jnp.log(b_theta)
        E_xi = a_xi / b_xi
        E_log_xi = jsp.digamma(a_xi) - jnp.log(b_xi)
        
        # === Poisson likelihood (via collapsed z) ===
        # Poisson PMF: P(x|λ) = λ^x * exp(-λ) / x!
        # log P(x|λ) = x*log(λ) - λ - log(Γ(x+1))
        log_rates = E_log_theta[:, jnp.newaxis, :] + self.E_log_beta_effective[jnp.newaxis, :, :]
        log_sum_rates = logsumexp(log_rates, axis=2)  # (batch, p)
        
        # x * log(λ) term
        elbo_x = jnp.sum(X_batch * log_sum_rates)
        # -λ term (expectation of rate)
        elbo_x -= jnp.sum(E_theta.sum(axis=0) * self.E_beta_effective.sum(axis=0))
        # -log(x!) term using gammaln (works for real x: log(x!) = log(Γ(x+1)))
        elbo_x -= jnp.sum(jsp.gammaln(X_batch + 1))
        
        elbo += scale * elbo_x
        
        # === Bernoulli likelihood (JJ bound) - VECTORIZED ===
        y_expanded = y_batch if y_batch.ndim > 1 else y_batch[:, jnp.newaxis]
        lam = self._lambda_jj(zeta)  # (batch, kappa)
        
        if self.p_aux > 0:
            aux_contrib = X_aux_batch @ self.E_gamma.T  # (batch, kappa)
        else:
            aux_contrib = 0.0
        
        E_A = E_theta @ self.E_v_effective.T + aux_contrib  # (batch, kappa)
        E_v_sq = self.mu_v**2 + jnp.diagonal(self.Sigma_v, axis1=1, axis2=2)  # (kappa, d)
        Var_theta = a_theta / (b_theta**2)  # (batch, d)
        E_A_sq = E_A**2 + (Var_theta @ E_v_sq.T)  # (batch, kappa)
        
        # JJ bound for all outcomes
        elbo_y = jnp.sum((y_expanded - 0.5) * E_A - lam * E_A_sq)
        elbo_y += jnp.sum(lam * zeta**2 - 0.5 * zeta - jnp.log1p(jnp.exp(jnp.clip(zeta, -500, 500))))
        elbo += scale * self.regression_weight * elbo_y
        
        # === Priors on local parameters ===
        # p(θ | ξ) - Gamma distribution: Γ(α, ξ)
        # log p(θ|ξ) = (α-1)*log(θ) + α*log(ξ) - ξ*θ - log(Γ(α))
        elbo_theta = jnp.sum((self.alpha_theta - 1) * E_log_theta + 
                           self.alpha_theta * E_log_xi[:, jnp.newaxis] -
                           E_xi[:, jnp.newaxis] * E_theta)
        elbo_theta -= batch_size * self.d * jsp.gammaln(self.alpha_theta)
        elbo += scale * elbo_theta
        
        # p(ξ) - Gamma distribution: Γ(α_ξ, λ_ξ)
        # log p(ξ) = (α_ξ-1)*log(ξ) - λ_ξ*ξ + α_ξ*log(λ_ξ) - log(Γ(α_ξ))
        elbo_xi = jnp.sum((self.alpha_xi - 1) * E_log_xi - self.lambda_xi * E_xi)
        elbo_xi += batch_size * (self.alpha_xi * jnp.log(self.lambda_xi) - jsp.gammaln(self.alpha_xi))
        elbo += scale * elbo_xi
        
        # === Priors on global parameters (not scaled) ===
        # p(β | η) - Gamma distribution: Γ(α_β, η)
        # log p(β|η) = (α_β-1)*log(β) + α_β*log(η) - η*β - log(Γ(α_β))
        elbo_beta = jnp.sum((self.alpha_beta - 1) * self.E_log_beta +
                          self.alpha_beta * self.E_log_eta[:, jnp.newaxis] -
                          self.E_eta[:, jnp.newaxis] * self.E_beta)
        elbo_beta -= self.p * self.d * jsp.gammaln(self.alpha_beta)
        elbo += elbo_beta
        
        # p(η) - Gamma distribution: Γ(α_η, λ_η)
        # log p(η) = (α_η-1)*log(η) - λ_η*η + α_η*log(λ_η) - log(Γ(α_η))
        elbo_eta = jnp.sum((self.alpha_eta - 1) * self.E_log_eta - self.lambda_eta * self.E_eta)
        elbo_eta += self.p * (self.alpha_eta * jnp.log(self.lambda_eta) - jsp.gammaln(self.alpha_eta))
        elbo += elbo_eta
        
        # p(v) - Multivariate Gaussian: N(0, σ_v² I)
        # log p(v) = -d/2 * log(2πσ_v²) - 1/(2σ_v²) * (μ'μ + tr(Σ))
        # Using E[v'v] = E[v]'E[v] + tr(Var[v]) = μ'μ + tr(Σ)
        elbo_v = 0.0
        for k in range(self.kappa):
            elbo_v -= 0.5 * self.d * jnp.log(2 * jnp.pi * self.sigma_v**2)
            elbo_v -= 0.5 / self.sigma_v**2 * (
                jnp.sum(self.mu_v[k]**2) + jnp.trace(self.Sigma_v[k])
            )
        elbo += elbo_v
        
        # p(γ) - Multivariate Gaussian: N(0, σ_γ² I)
        # log p(γ) = -p_aux/2 * log(2πσ_γ²) - 1/(2σ_γ²) * (μ'μ + tr(Σ))
        elbo_gamma = 0.0
        if self.p_aux > 0:
            for k in range(self.kappa):
                elbo_gamma -= 0.5 * self.p_aux * jnp.log(2 * jnp.pi * self.sigma_gamma**2)
                elbo_gamma -= 0.5 / self.sigma_gamma**2 * (
                    jnp.sum(self.mu_gamma[k]**2) + jnp.trace(self.Sigma_gamma[k])
                )
        elbo += elbo_gamma
        
        # === Entropy terms ===
        # H[q(θ)] - Gamma entropy: H[Γ(a,b)] = a - log(b) + log(Γ(a)) + (1-a)ψ(a)
        # where ψ is the digamma function (derivative of log Γ)
        H_theta = jnp.sum(a_theta - jnp.log(b_theta) + jsp.gammaln(a_theta) +
                        (1 - a_theta) * jsp.digamma(a_theta))
        elbo += scale * H_theta
        
        # H[q(ξ)] - Gamma entropy
        H_xi = jnp.sum(a_xi - jnp.log(b_xi) + jsp.gammaln(a_xi) +
                     (1 - a_xi) * jsp.digamma(a_xi))
        elbo += scale * H_xi
        
        # H[q(β)] - Gamma entropy
        H_beta = jnp.sum(self.a_beta - jnp.log(self.b_beta) + jsp.gammaln(self.a_beta) +
                       (1 - self.a_beta) * jsp.digamma(self.a_beta))
        elbo += H_beta
        
        # H[q(η)] - Gamma entropy
        H_eta = jnp.sum(self.a_eta - jnp.log(self.b_eta) + jsp.gammaln(self.a_eta) +
                      (1 - self.a_eta) * jsp.digamma(self.a_eta))
        elbo += H_eta
        
        # H[q(v)] - Multivariate Gaussian entropy: H[N(μ,Σ)] = d/2*(1+log(2π)) + 1/2*log|Σ|
        # Using slogdet for numerical stability (returns sign and log of determinant)
        for k in range(self.kappa):
            sign, logdet = jnp.linalg.slogdet(self.Sigma_v[k])
            elbo += jnp.where(sign > 0, 0.5 * self.d * (1 + jnp.log(2 * jnp.pi)) + 0.5 * logdet, 0.0)
        
        # H[q(γ)] - Multivariate Gaussian entropy
        if self.p_aux > 0:
            for k in range(self.kappa):
                sign, logdet = jnp.linalg.slogdet(self.Sigma_gamma[k])
                elbo += jnp.where(sign > 0, 0.5 * self.p_aux * (1 + jnp.log(2 * jnp.pi)) + 0.5 * logdet, 0.0)
        
        return float(elbo)

    def compute_heldout_loglik(
        self,
        X_heldout: np.ndarray,
        y_heldout: np.ndarray,
        X_aux_heldout: Optional[np.ndarray] = None,
        n_iter: int = 30
    ) -> float:
        """
        Compute held-out log-likelihood for convergence monitoring (Blei-style).

        This provides an unbiased estimate of model fit by:
        1. Inferring θ for held-out samples using frozen global params
        2. Computing predictive log-likelihood for counts and labels

        Following Hoffman et al. (2013), held-out LL is more reliable than
        ELBO for tracking convergence since ELBO can increase due to
        tighter variational approximation rather than better model fit.

        Parameters
        ----------
        X_heldout : (n_heldout, p) count matrix
        y_heldout : (n_heldout,) or (n_heldout, κ) labels
        X_aux_heldout : (n_heldout, p_aux) auxiliary features
        n_iter : number of local VI iterations for θ inference

        Returns
        -------
        float : Average held-out log-likelihood per sample
        """
        # Convert to JAX arrays
        if sp.issparse(X_heldout):
            X_heldout = jnp.array(X_heldout.toarray())
        else:
            X_heldout = jnp.array(X_heldout)

        y_heldout = jnp.array(y_heldout)
        if y_heldout.ndim == 1:
            y_heldout = y_heldout[:, jnp.newaxis]

        n_heldout = X_heldout.shape[0]

        if X_aux_heldout is None:
            X_aux_heldout = jnp.zeros((n_heldout, self.p_aux if self.p_aux > 0 else 0))
        else:
            X_aux_heldout = jnp.array(X_aux_heldout)

        # Infer θ for held-out samples with frozen global params
        result = self.transform(
            np.array(X_heldout),
            y_new=np.array(y_heldout),
            X_aux_new=np.array(X_aux_heldout),
            n_iter=n_iter
        )
        E_theta = jnp.array(result['E_theta'])
        E_log_theta = jsp.digamma(jnp.array(result['a_theta'])) - jnp.log(jnp.array(result['b_theta']))

        # === Poisson log-likelihood for counts ===
        # log p(x|θ,β) = Σ_ij [x_ij * log(Σ_ℓ θ_iℓ β_jℓ) - Σ_ℓ θ_iℓ β_jℓ - log(x_ij!)]
        log_rates = E_log_theta[:, jnp.newaxis, :] + self.E_log_beta_effective[jnp.newaxis, :, :]
        log_sum_rates = logsumexp(log_rates, axis=2)  # (n_heldout, p)

        loglik_counts = jnp.sum(X_heldout * log_sum_rates)
        loglik_counts -= jnp.sum(E_theta.sum(axis=1, keepdims=True) * self.E_beta_effective.sum(axis=0, keepdims=True))
        loglik_counts -= jnp.sum(jsp.gammaln(X_heldout + 1))

        # === Bernoulli log-likelihood for labels ===
        # log p(y|θ,v,γ) using sigmoid approximation
        if self.p_aux > 0:
            logits = E_theta @ self.E_v_effective.T + X_aux_heldout @ self.E_gamma.T
        else:
            logits = E_theta @ self.E_v_effective.T

        # Binary cross-entropy: y*log(σ) + (1-y)*log(1-σ)
        # = y*logits - log(1 + exp(logits))
        logits_clipped = jnp.clip(logits, -500, 500)
        loglik_labels = jnp.sum(
            y_heldout * logits_clipped - jnp.log1p(jnp.exp(logits_clipped))
        )

        # Combine and normalize
        total_loglik = float(loglik_counts) + self.regression_weight * float(loglik_labels)
        return total_loglik / n_heldout

    # =========================================================================
    # MAIN FITTING LOOP
    # =========================================================================

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_aux: np.ndarray,
        max_epochs: int = 100,
        elbo_freq: int = 10,
        verbose: bool = True,
        early_stopping: bool = False,
        X_heldout: Optional[np.ndarray] = None,
        y_heldout: Optional[np.ndarray] = None,
        X_aux_heldout: Optional[np.ndarray] = None,
        heldout_freq: int = 5
    ):
        """
        Fit model using Stochastic Variational Inference.

        Handles both dense and sparse input matrices. Sparse matrices are
        converted to dense on GPU for computation.

        Convergence tracked via EMA + Welford's online algorithm:
        - elbo_ema_: Exponential moving average of batch ELBO
        - elbo_welford_mean_: Running mean (Welford)
        - elbo_welford_var_: Running variance (Welford)
        - convergence_history_: List of (epoch, ema, mean, std, rel_change)
        - heldout_loglik_history_: List of (epoch, heldout_ll) if held-out data provided

        Parameters
        ----------
        X : np.ndarray or scipy.sparse matrix
            Gene expression data (n_cells, n_genes)
        y : np.ndarray
            Labels (n_cells,) or (n_cells, n_outcomes)
        X_aux : np.ndarray
            Auxiliary features (n_cells, n_aux)
        early_stopping : bool
            If True, stop when EMA relative change < convergence_tol for
            convergence_window consecutive checks.
        X_heldout : np.ndarray, optional
            Held-out count data for convergence tracking (Blei-style)
        y_heldout : np.ndarray, optional
            Held-out labels for convergence tracking
        X_aux_heldout : np.ndarray, optional
            Held-out auxiliary features
        heldout_freq : int
            Frequency (in epochs) to compute held-out log-likelihood
        """
        # Convert sparse matrices to dense JAX arrays
        if sp.issparse(X):
            print(f"Converting sparse matrix ({X.format}) to dense JAX array...")
            X = jnp.array(X.toarray())
        else:
            X = jnp.array(X)
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y = jnp.array(y)
        
        X_aux = jnp.array(X_aux) if X_aux is not None else jnp.zeros((X.shape[0], 0))
        
        # Initialize
        self._initialize_global_parameters(X, y, X_aux)
        
        # Use ceiling division to ensure all samples are processed
        n_batches = max(1, (self.n + self.batch_size - 1) // self.batch_size)
        iteration = 0
        self.elbo_history_ = []
        
        # EMA + Welford initialization (O(1) memory)
        self.elbo_ema_ = None           # Exponential moving average
        self.elbo_ema_prev_ = None      # Previous EMA for relative change
        self.elbo_welford_n_ = 0        # Welford count
        self.elbo_welford_mean_ = 0.0   # Welford running mean
        self.elbo_welford_M2_ = 0.0     # Welford sum of squared deviations
        self.convergence_history_ = []  # (epoch, ema, mean, std, rel_change)
        self.last_elbo_ = None          # Track last computed ELBO for display
        consecutive_converged = 0

        # Held-out log-likelihood tracking (Blei-style)
        use_heldout = X_heldout is not None and y_heldout is not None
        self.heldout_loglik_history_ = []  # (epoch, heldout_ll)
        
        # Storage for final training set local parameters
        self.train_a_theta_ = None
        self.train_b_theta_ = None
        self.train_a_xi_ = None
        self.train_b_xi_ = None
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            # Shuffle data - use numpy for permutation then convert to JAX
            perm = np.random.permutation(self.n)
            
            # Storage for training parameters (overwritten each epoch)
            epoch_a_theta = jnp.zeros((self.n, self.d))
            epoch_b_theta = jnp.zeros((self.n, self.d))
            epoch_a_xi = jnp.zeros(self.n)
            epoch_b_xi = jnp.zeros(self.n)
            
            epoch_elbo = 0.0
            
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.n)
                idx = perm[start:end]
                
                X_batch = X[idx]
                y_batch = y[idx]
                X_aux_batch = X_aux[idx]
                
                actual_batch_size = end - start
                scale = self.n / actual_batch_size
                
                # 1. Optimize local parameters
                a_theta, b_theta, a_xi, b_xi, zeta = self._update_local_parameters(
                    X_batch, y_batch, X_aux_batch
                )
                
                # Store training set parameters (final epoch will be retained)
                epoch_a_theta = epoch_a_theta.at[idx].set(a_theta)
                epoch_b_theta = epoch_b_theta.at[idx].set(b_theta)
                epoch_a_xi = epoch_a_xi.at[idx].set(a_xi)
                epoch_b_xi = epoch_b_xi.at[idx].set(b_xi)
                
                E_theta = a_theta / b_theta
                E_log_theta = jsp.digamma(a_theta) - jnp.log(b_theta)
                
                # 2. Compute intermediate global parameters
                a_beta_hat, b_beta_hat = self._compute_intermediate_beta(
                    X_batch, E_theta, E_log_theta, scale
                )
                a_eta_hat, b_eta_hat = self._compute_intermediate_eta(E_theta, scale)
                mu_v_hat, Sigma_v_hat = self._compute_intermediate_v(
                    y_batch, X_aux_batch, E_theta, a_theta, b_theta, zeta, scale
                )
                mu_gamma_hat, Sigma_gamma_hat = self._compute_intermediate_gamma(
                    y_batch, X_aux_batch, E_theta, zeta, scale
                )
                
                # 3. SVI update with natural gradients
                rho_t = self._get_learning_rate(iteration)
                self._svi_update_global(
                    rho_t,
                    a_beta_hat, b_beta_hat,
                    a_eta_hat, b_eta_hat,
                    mu_v_hat, Sigma_v_hat,
                    mu_gamma_hat, Sigma_gamma_hat
                )

                # 4. Update spike-and-slab inclusion probabilities (if enabled)
                if self.use_spike_slab:
                    self._update_spike_slab(E_theta, self.E_beta, a_theta, b_theta)

                # Compute ELBO periodically
                if iteration % elbo_freq == 0:
                    elbo = self._compute_elbo(
                        X_batch, y_batch, X_aux_batch,
                        a_theta, b_theta, a_xi, b_xi, zeta, scale
                    )
                    if np.isfinite(elbo):
                        self.elbo_history_.append((iteration, elbo))
                        epoch_elbo = elbo
                        self.last_elbo_ = elbo  # Track for display
                        
                        # EMA update: O(1)
                        if self.elbo_ema_ is None:
                            self.elbo_ema_ = elbo
                        else:
                            self.elbo_ema_ = (self.ema_decay * self.elbo_ema_ + 
                                              (1 - self.ema_decay) * elbo)
                        
                        # Welford's online update: O(1)
                        self.elbo_welford_n_ += 1
                        delta = elbo - self.elbo_welford_mean_
                        self.elbo_welford_mean_ += delta / self.elbo_welford_n_
                        delta2 = elbo - self.elbo_welford_mean_
                        self.elbo_welford_M2_ += delta * delta2
                
                iteration += 1
            
            # End of epoch: compute convergence diagnostics
            if self.elbo_welford_n_ > 1:
                welford_var = self.elbo_welford_M2_ / (self.elbo_welford_n_ - 1)
                welford_std = np.sqrt(max(0, welford_var))
            else:
                welford_std = np.inf
            
            # Relative change in EMA
            if self.elbo_ema_prev_ is not None and self.elbo_ema_ is not None:
                rel_change = abs(self.elbo_ema_ - self.elbo_ema_prev_) / (abs(self.elbo_ema_prev_) + 1e-10)
            else:
                rel_change = np.inf
            
            # Store convergence diagnostics (sparse: every 5 epochs)
            if epoch % 5 == 0:
                self.elbo_ema_prev_ = self.elbo_ema_
                self.convergence_history_.append((
                    epoch,
                    self.elbo_ema_ if self.elbo_ema_ is not None else np.nan,
                    self.elbo_welford_mean_,
                    welford_std,
                    rel_change
                ))

            # Compute held-out log-likelihood (Blei-style convergence tracking)
            heldout_ll = None
            if use_heldout and epoch % heldout_freq == 0:
                heldout_ll = self.compute_heldout_loglik(
                    X_heldout, y_heldout, X_aux_heldout, n_iter=20
                )
                self.heldout_loglik_history_.append((epoch, heldout_ll))

            if verbose and epoch % 5 == 0:
                beta_diversity = np.std(np.array(self.E_beta), axis=1).mean()
                ema_str = f"{self.elbo_ema_:.2e}" if self.elbo_ema_ is not None else "N/A"
                rel_str = f"{rel_change:.2e}" if rel_change != np.inf else "N/A"
                elbo_str = f"{self.last_elbo_:.2e}" if self.last_elbo_ is not None else "N/A"
                heldout_str = f", HO-LL = {heldout_ll:.2f}" if heldout_ll is not None else ""
                print(f"Epoch {epoch}: ELBO = {elbo_str}, EMA = {ema_str}, "
                      f"Δrel = {rel_str}, ρ_t = {rho_t:.4f}, "
                      f"v = {np.array(self.mu_v).ravel()[:3]}, β_div = {beta_diversity:.3f}{heldout_str}")
            
            # Early stopping check
            if early_stopping and rel_change < self.convergence_tol:
                consecutive_converged += 1
                if consecutive_converged >= self.convergence_window:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch}: "
                              f"EMA rel_change < {self.convergence_tol} for "
                              f"{self.convergence_window} consecutive checks")
                    break
            else:
                consecutive_converged = 0
            
            # Store final epoch's training parameters
            self.train_a_theta_ = epoch_a_theta
            self.train_b_theta_ = epoch_b_theta
            self.train_a_xi_ = epoch_a_xi
            self.train_b_xi_ = epoch_b_xi
        
        self.training_time_ = time.time() - start_time
        
        # Final convergence summary
        if self.elbo_welford_n_ > 1:
            final_var = self.elbo_welford_M2_ / (self.elbo_welford_n_ - 1)
            final_std = np.sqrt(max(0, final_var))
        else:
            final_std = np.nan
        
        self.final_elbo_ema_ = self.elbo_ema_
        self.final_elbo_mean_ = self.elbo_welford_mean_
        self.final_elbo_std_ = final_std
        
        if verbose:
            print(f"\nTraining complete in {self.training_time_:.1f}s")
            print(f"Final ELBO: EMA = {self.final_elbo_ema_:.2e}, "
                  f"Mean = {self.final_elbo_mean_:.2e}, Std = {self.final_elbo_std_:.2e}")

            # Report final held-out LL if available
            if len(self.heldout_loglik_history_) > 0:
                final_heldout_ll = self.heldout_loglik_history_[-1][1]
                print(f"Final Held-out LL: {final_heldout_ll:.4f} (per sample)")

            # v learning diagnostics
            if hasattr(self, 'initial_mu_v_'):
                v_change = np.array(self.mu_v - self.initial_mu_v_)
                v_change_norm = np.linalg.norm(v_change)
                v_init_norm = np.linalg.norm(np.array(self.initial_mu_v_))
                v_final_norm = np.linalg.norm(np.array(self.mu_v))
                print(f"\nv Learning Diagnostics:")
                print(f"  Initial v norm: {v_init_norm:.4f}")
                print(f"  Final v norm:   {v_final_norm:.4f}")
                print(f"  v change norm:  {v_change_norm:.4f}")
                print(f"  Relative change: {v_change_norm / (v_init_norm + 1e-10):.4f}")
                print(f"  Max |v| change:  {np.abs(v_change).max():.4f}")
                print(f"  v range: [{np.array(self.mu_v).min():.4f}, {np.array(self.mu_v).max():.4f}]")
                print(f"  v std:   {np.array(self.mu_v).std():.4f}")
                if v_change_norm < 0.1:
                    print(f"  WARNING: v barely changed from initialization!")
                if np.array(self.mu_v).std() < 0.1:
                    print(f"  WARNING: v is essentially flat - not learning discrimination!")

        return self
    
    def transform(self, X_new: np.ndarray, y_new: np.ndarray = None, 
                  X_aux_new: np.ndarray = None, n_iter: int = 50) -> dict:
        """
        Infer θ for new samples with frozen global parameters (β, η, v, γ).
        
        VECTORIZED: Regression contributions computed in vectorized form.
        
        Parameters
        ----------
        X_new : (n_new, p) count matrix (dense or sparse)
        y_new : (n_new,) or (n_new, κ) labels (optional, for ζ updates)
        X_aux_new : (n_new, p_aux) auxiliary covariates
        n_iter : local VI iterations
        
        Returns
        -------
        dict with keys: 'E_theta', 'a_theta', 'b_theta', 'a_xi', 'b_xi'
        """
        # Convert sparse to dense if needed
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
            y_new = jnp.full((n_new, self.kappa), 0.5)  # Neutral labels
        else:
            y_new = jnp.array(y_new)
            y_new = y_new.reshape(-1, 1) if y_new.ndim == 1 else y_new
        
        # Initialize local parameters with factor diversity
        row_sums = X_new.sum(axis=1, keepdims=True) + 1
        factor_scales = jnp.linspace(0.5, 2.0, self.d)
        
        a_theta = jnp.full((n_new, self.d), self.alpha_theta + self.p * self.alpha_beta)
        b_theta = row_sums / self.d * factor_scales
        a_xi = jnp.full(n_new, self.alpha_xi + self.d * self.alpha_theta)
        b_xi = jnp.full(n_new, self.lambda_xi)
        
        # Frozen global expectations
        E_beta = self.a_beta / self.b_beta          # (p, d)
        E_log_beta = jsp.digamma(self.a_beta) - jnp.log(self.b_beta)
        E_v = self.mu_v                              # (κ, d)
        E_v_sq = self.mu_v**2 + jnp.diagonal(self.Sigma_v, axis1=1, axis2=2)
        E_gamma = self.mu_gamma                      # (κ, p_aux)
        
        beta_col_sums = E_beta.sum(axis=0)  # (d,)
        
        for it in range(n_iter):
            # Current local expectations
            E_theta = a_theta / b_theta              # (n_new, d)
            E_log_theta = jsp.digamma(a_theta) - jnp.log(b_theta)
            E_xi = a_xi / b_xi
            E_theta_sq = (a_theta * (a_theta + 1)) / b_theta**2
            
            # Update ξ
            a_xi = self.alpha_xi + self.d * self.alpha_theta
            b_xi = self.lambda_xi + E_theta.sum(axis=1)
            E_xi = a_xi / b_xi
            
            # Compute φ for auxiliary variable z
            log_phi = E_log_theta[:, jnp.newaxis, :] + E_log_beta[jnp.newaxis, :, :]  # (n, p, d)
            log_phi -= logsumexp(log_phi, axis=2, keepdims=True)
            phi = jnp.exp(log_phi)
            
            # Update ζ (JJ bound)
            if self.p_aux > 0:
                aux_contrib = X_aux_new @ E_gamma.T  # (n_new, κ)
            else:
                aux_contrib = 0.0
            theta_v = E_theta @ E_v.T            # (n_new, κ)
            E_A = theta_v + aux_contrib
            if self.p_aux > 0:
                E_A_sq = (E_theta_sq @ E_v_sq.T + 
                          2 * theta_v * aux_contrib + 
                          aux_contrib**2)
            else:
                E_A_sq = E_theta_sq @ E_v_sq.T
            zeta = jnp.sqrt(jnp.maximum(E_A_sq, 1e-10))
            lam = jnp.tanh(zeta / 2) / (4 * zeta + 1e-10)
            
            # Update θ
            # Shape: sum over genes of X * φ
            shape_contrib = jnp.einsum('ij,ijl->il', X_new, phi)  # (n_new, d)
            a_theta = self.alpha_theta + shape_contrib
            
            # Rate: ξ + sum_j β_jℓ + regression term R_iℓ
            b_theta = E_xi[:, jnp.newaxis] + beta_col_sums[jnp.newaxis, :]
            
            # === VECTORIZED regression contribution ===
            # For each factor ℓ, compute C^{(-ℓ)} vectorized over all outcomes
            # Broadcasting: (n_new, kappa, 1) - (n_new, 1, d) * (1, kappa, d)
            C_minus_ell = (
                E_A[:, :, jnp.newaxis] -  # (n_new, kappa, 1)
                E_theta[:, jnp.newaxis, :] * E_v[jnp.newaxis, :, :]  # (n_new, kappa, d)
            )
            
            # R for all outcomes and factors: (n_new, kappa, d)
            R = (
                -(y_new[:, :, jnp.newaxis] - 0.5) * E_v[jnp.newaxis, :, :] +
                2 * lam[:, :, jnp.newaxis] * E_v[jnp.newaxis, :, :] * C_minus_ell +
                2 * lam[:, :, jnp.newaxis] * E_v_sq[jnp.newaxis, :, :] * E_theta[:, jnp.newaxis, :]
            )
            
            # Sum over outcomes: (n_new, d)
            R_sum = R.sum(axis=1)
            
            b_theta = b_theta + R_sum
            b_theta = jnp.maximum(b_theta, 1e-10)
        
        E_theta_final = a_theta / b_theta
        
        return {
            'E_theta': np.array(E_theta_final),
            'a_theta': np.array(a_theta),
            'b_theta': np.array(b_theta),
            'a_xi': np.array(a_xi),
            'b_xi': np.array(b_xi)
        }

    def predict_proba(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                      n_iter: int = 50) -> np.ndarray:
        """
        Predict class probabilities for new samples.
        
        Parameters
        ----------
        X_new : (n_new, p) count matrix (dense or sparse)
        X_aux_new : (n_new, p_aux) auxiliary covariates
        n_iter : local VI iterations for θ inference
        
        Returns
        -------
        proba : (n_new,) or (n_new, κ) predicted probabilities
        """
        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, self.p_aux if self.p_aux > 0 else 0))
        else:
            X_aux_new = np.array(X_aux_new)
            
        result = self.transform(X_new, y_new=None, X_aux_new=X_aux_new, n_iter=n_iter)
        E_theta = jnp.array(result['E_theta'])
        
        # Compute logits: θ @ v^T + X_aux @ γ^T
        if self.p_aux > 0:
            logits = E_theta @ self.mu_v.T + jnp.array(X_aux_new) @ self.mu_gamma.T
        else:
            logits = E_theta @ self.mu_v.T
        proba = jsp.expit(logits)
        
        return np.array(proba).squeeze()

    def predict(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                n_iter: int = 50, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for new samples.

        Parameters
        ----------
        X_new : (n_new, p) count matrix (dense or sparse)
        X_aux_new : (n_new, p_aux) auxiliary covariates
        n_iter : local VI iterations for θ inference
        threshold : classification threshold

        Returns
        -------
        labels : (n_new,) or (n_new, κ) predicted labels
        """
        proba = self.predict_proba(X_new, X_aux_new, n_iter)
        return (proba >= threshold).astype(int)

    def _compute_memory_efficient_batch_size(self, target_gb: float = 0.5) -> int:
        """
        Compute optimal batch size for memory efficiency.

        The main memory bottleneck is the (batch, p, d) tensor in transform.
        We target approximately target_gb GB per batch.

        Parameters
        ----------
        target_gb : float
            Target memory usage in GB per batch

        Returns
        -------
        int : Optimal batch size
        """
        # Memory for (batch, p, d) tensor in float64
        bytes_per_sample = self.p * self.d * 8
        target_bytes = target_gb * 1024 * 1024 * 1024
        batch_size = max(1, int(target_bytes / bytes_per_sample))
        return batch_size

    def transform_batched(self, X_new: np.ndarray, y_new: np.ndarray = None,
                          X_aux_new: np.ndarray = None, n_iter: int = 50,
                          batch_size: int = None, verbose: bool = False) -> dict:
        """
        Memory-efficient batched inference for θ on new samples.

        This method processes samples in batches to avoid OOM errors when
        dealing with large datasets (n × p × d tensor can be huge).

        Parameters
        ----------
        X_new : (n_new, p) count matrix (dense or sparse)
        y_new : (n_new,) or (n_new, κ) labels (optional, for ζ updates)
        X_aux_new : (n_new, p_aux) auxiliary covariates
        n_iter : local VI iterations per batch
        batch_size : samples per batch (auto-computed if None)
        verbose : print progress

        Returns
        -------
        dict with keys: 'E_theta', 'a_theta', 'b_theta', 'a_xi', 'b_xi'
        """
        # Convert sparse to dense numpy if needed
        if sp.issparse(X_new):
            X_new = X_new.toarray()
        else:
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

        # Auto-compute batch size for ~500MB memory usage
        if batch_size is None:
            batch_size = self._compute_memory_efficient_batch_size(target_gb=0.5)

        if verbose:
            print(f"Using batch_size={batch_size} for memory efficiency "
                  f"(p={self.p}, d={self.d})")

        # Initialize output arrays
        E_theta_all = np.zeros((n_new, self.d))
        a_theta_all = np.zeros((n_new, self.d))
        b_theta_all = np.zeros((n_new, self.d))
        a_xi_all = np.zeros(n_new)
        b_xi_all = np.zeros(n_new)

        # Process in batches
        n_batches = (n_new + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_new)

            if verbose and n_batches > 1:
                print(f"  Processing batch {batch_idx + 1}/{n_batches} "
                      f"(samples {start_idx}-{end_idx})")

            # Extract batch
            X_batch = X_new[start_idx:end_idx]
            y_batch = y_new[start_idx:end_idx]
            X_aux_batch = X_aux_new[start_idx:end_idx]

            # Run unbatched transform on this small batch
            result = self.transform(X_batch, y_new=y_batch,
                                   X_aux_new=X_aux_batch, n_iter=n_iter)

            # Store results
            E_theta_all[start_idx:end_idx] = result['E_theta']
            a_theta_all[start_idx:end_idx] = result['a_theta']
            b_theta_all[start_idx:end_idx] = result['b_theta']
            a_xi_all[start_idx:end_idx] = result['a_xi']
            b_xi_all[start_idx:end_idx] = result['b_xi']

        return {
            'E_theta': E_theta_all,
            'a_theta': a_theta_all,
            'b_theta': b_theta_all,
            'a_xi': a_xi_all,
            'b_xi': b_xi_all
        }

    def predict_proba_batched(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                               n_iter: int = 50, batch_size: int = None,
                               verbose: bool = False) -> np.ndarray:
        """
        Memory-efficient batched prediction of class probabilities.

        This method processes samples in batches to avoid OOM errors.

        Parameters
        ----------
        X_new : (n_new, p) count matrix (dense or sparse)
        X_aux_new : (n_new, p_aux) auxiliary covariates
        n_iter : local VI iterations for θ inference
        batch_size : samples per batch (auto-computed if None)
        verbose : print progress

        Returns
        -------
        proba : (n_new,) or (n_new, κ) predicted probabilities
        """
        # Convert sparse to dense numpy if needed
        if sp.issparse(X_new):
            X_new = X_new.toarray()
        else:
            X_new = np.array(X_new)

        n_new = X_new.shape[0]

        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, self.p_aux if self.p_aux > 0 else 0))
        else:
            X_aux_new = np.array(X_aux_new)

        # Auto-compute batch size for ~500MB memory usage
        if batch_size is None:
            batch_size = self._compute_memory_efficient_batch_size(target_gb=0.5)

        if verbose:
            print(f"Using batch_size={batch_size} for memory efficiency")

        # Initialize output array
        proba_all = np.zeros(n_new) if self.kappa == 1 else np.zeros((n_new, self.kappa))

        # Process in batches
        n_batches = (n_new + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_new)

            if verbose and n_batches > 1:
                print(f"  Processing batch {batch_idx + 1}/{n_batches} "
                      f"(samples {start_idx}-{end_idx})")

            # Extract batch
            X_batch = X_new[start_idx:end_idx]
            X_aux_batch = X_aux_new[start_idx:end_idx]

            # Run unbatched predict_proba on this small batch
            proba_batch = self.predict_proba(X_batch, X_aux_batch, n_iter=n_iter)

            # Store results
            proba_all[start_idx:end_idx] = proba_batch

        return proba_all

    def predict_batched(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                        n_iter: int = 50, threshold: float = 0.5,
                        batch_size: int = None, verbose: bool = False) -> np.ndarray:
        """
        Memory-efficient batched prediction of class labels.

        Parameters
        ----------
        X_new : (n_new, p) count matrix (dense or sparse)
        X_aux_new : (n_new, p_aux) auxiliary covariates
        n_iter : local VI iterations for θ inference
        threshold : classification threshold
        batch_size : samples per batch (auto-computed if None)
        verbose : print progress

        Returns
        -------
        labels : (n_new,) or (n_new, κ) predicted labels
        """
        proba = self.predict_proba_batched(X_new, X_aux_new, n_iter, batch_size, verbose)
        return (proba >= threshold).astype(int)


if __name__ == "__main__":
    # Test with train/test split
    print("Testing JAX-optimized SVI model...")
    np.random.seed(42)
    n_train, n_test, p, d = 200, 50, 50, 5
    
    # Generate synthetic data
    theta_true_train = np.random.gamma(2, 1, (n_train, d))
    theta_true_test = np.random.gamma(2, 1, (n_test, d))
    beta_true = np.random.gamma(2, 1, (p, d))
    
    X_train = np.random.poisson(theta_true_train @ beta_true.T)
    X_test = np.random.poisson(theta_true_test @ beta_true.T)
    
    v_true = np.array([[1, -1, 0.5, 0, 0]])
    
    from scipy.special import expit
    logits_train = theta_true_train @ v_true.T
    y_train = (expit(logits_train) > 0.5).astype(float).ravel()
    
    logits_test = theta_true_test @ v_true.T
    y_test = (expit(logits_test) > 0.5).astype(float).ravel()
    
    X_aux_train = np.random.randn(n_train, 2)
    X_aux_test = np.random.randn(n_test, 2)
    
    # Train model
    model = SVICorrected(
        n_factors=d, 
        batch_size=64, 
        learning_rate=0.5,
        learning_rate_decay=0.6,
        learning_rate_min=0.01,
        local_iterations=20,
        random_state=42,
        use_jit=True,
        device='gpu'
    )
    model.fit(X_train, y_train, X_aux_train, max_epochs=100, verbose=True)
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Final v:\n{np.array(model.mu_v)}")
    print(f"True v:\n{v_true}")
    print(f"v correlation: {np.corrcoef(np.array(model.mu_v).ravel(), v_true.ravel())[0,1]:.3f}")
    
    # Test transform on held-out data
    print(f"\n{'='*50}")
    print("TEST SET INFERENCE")
    print(f"{'='*50}")
    
    result = model.transform(X_test, y_new=None, X_aux_new=X_aux_test, n_iter=50)
    E_theta_test = result['E_theta']
    
    # Check theta recovery (correlation with true theta per factor)
    print("θ recovery (correlation per factor):")
    for ell in range(d):
        corr = np.corrcoef(E_theta_test[:, ell], theta_true_test[:, ell])[0, 1]
        print(f"  Factor {ell}: {corr:.3f}")
    
    # Prediction performance
    y_pred_proba = model.predict_proba(X_test, X_aux_test, n_iter=50)
    y_pred = model.predict(X_test, X_aux_test, n_iter=50)
    
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest accuracy: {accuracy:.3f}")
    
    # AUC if sklearn available
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Test AUC: {auc:.3f}")
    except ImportError:
        pass


# Alias for backward compatibility - SVI is the standard name
SVI = SVICorrected
