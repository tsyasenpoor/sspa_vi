"""
Corrected Stochastic Variational Inference for Supervised Poisson Factorization
================================================================================

Key corrections from the original implementation:
1. Natural gradient updates for exponential family parameters
2. Proper intermediate parameter computation following Hoffman et al. (2013)
3. Consistent scaling for mini-batch -> full dataset extrapolation

For Gamma(a, b): natural params η = (a-1, -b), sufficient stats t(x) = (log x, x)
For Gaussian N(μ, Σ): natural params η = (Σ⁻¹μ, -½Σ⁻¹), sufficient stats t(x) = (x, xxᵀ)

SVI Update (Eq. 34 from Hoffman et al.):
    λ^(t) = (1 - ρ_t) λ^(t-1) + ρ_t * λ̂
where λ̂ = α + N · E_φ[t(x_i, z_i)] is the intermediate parameter.

References:
- Hoffman et al. (2013) "Stochastic Variational Inference", JMLR
- Your PSB paper derivations (A.24-A.60)
"""

import numpy as np
from scipy.special import digamma, gammaln, expit, logsumexp
from typing import Tuple, Optional
import time


class SVICorrected:
    """
    Corrected SVI implementation with proper natural gradient updates.
    
    Key architectural decisions:
    - Global params (β, η, v, γ): Updated via SVI with natural gradients
    - Local params (θ, ξ, ζ): Fully optimized per mini-batch
    """
    
    def __init__(
        self,
        n_factors: int,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.75,  # κ in ρ_t = (τ + t)^{-κ}
        learning_rate_delay: float = 1.0,   # τ
        learning_rate_min: float = 1e-4,
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
        
        # Convergence tracking
        ema_decay: float = 0.95,
        convergence_tol: float = 1e-4,
        convergence_window: int = 10,
        
        random_state: Optional[int] = None
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
        
        # RNG
        self.rng = np.random.default_rng(random_state)
        
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
    
    def _lambda_jj(self, zeta: np.ndarray) -> np.ndarray:
        """
        Jaakkola-Jordan auxiliary function: λ(ζ) = tanh(ζ/2) / (4ζ)
        For ζ→0: λ(0) = 1/8
        """
        result = np.zeros_like(zeta)
        nonzero = np.abs(zeta) > 1e-8
        result[nonzero] = np.tanh(zeta[nonzero] / 2) / (4 * zeta[nonzero])
        result[~nonzero] = 0.125
        return result
    
    # =========================================================================
    # NATURAL PARAMETER CONVERSIONS
    # =========================================================================
    
    def _gamma_to_natural(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gamma(a, b) → natural parameters (η₁, η₂) = (a-1, -b)
        """
        return a - 1, -b
    
    def _natural_to_gamma(self, eta1: np.ndarray, eta2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Natural parameters (η₁, η₂) → Gamma(a, b) = (η₁+1, -η₂)
        """
        a = eta1 + 1
        b = -eta2
        # Ensure valid parameters
        a = np.maximum(a, 1.001)
        b = np.maximum(b, 1e-6)
        return a, b
    
    def _gaussian_to_natural(self, mu: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        N(μ, Σ) → natural parameters (η₁, η₂) = (Σ⁻¹μ, -½Σ⁻¹)
        """
        try:
            Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(Sigma.shape[0]))
            eta1 = Sigma_inv @ mu
            eta2 = -0.5 * Sigma_inv
        except np.linalg.LinAlgError:
            # Fallback: diagonal approximation
            diag = np.diag(Sigma) + 1e-6
            eta1 = mu / diag
            eta2 = -0.5 * np.diag(1.0 / diag)
        return eta1, eta2
    
    def _natural_to_gaussian(self, eta1: np.ndarray, eta2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Natural parameters (η₁, η₂) → N(μ, Σ)
        η₁ = Σ⁻¹μ, η₂ = -½Σ⁻¹
        ⟹ Σ = -½ η₂⁻¹, μ = Σ η₁
        """
        try:
            # Add regularization for numerical stability
            eta2_reg = eta2 - 1e-6 * np.eye(eta2.shape[0])
            Sigma = -0.5 * np.linalg.inv(eta2_reg)
            # Ensure symmetry and positive definiteness
            Sigma = 0.5 * (Sigma + Sigma.T)
            eigvals = np.linalg.eigvalsh(Sigma)
            if np.min(eigvals) < 1e-6:
                Sigma += (1e-6 - np.min(eigvals) + 1e-8) * np.eye(Sigma.shape[0])
            mu = Sigma @ eta1
        except np.linalg.LinAlgError:
            # Fallback: return prior
            d = eta1.shape[0]
            Sigma = np.eye(d)
            mu = np.zeros(d)
        return mu, Sigma
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _initialize_global_parameters(self, X: np.ndarray, y: np.ndarray, X_aux: np.ndarray):
        """Initialize global variational parameters with factor diversity."""
        self.n, self.p = X.shape
        self.kappa = 1 if y.ndim == 1 else y.shape[1]
        self.p_aux = X_aux.shape[1] if X_aux is not None and X_aux.size > 0 else 0
        
        # β: Gene loadings, Gamma(a_β, b_β)
        # Initialize with factor-specific patterns to break symmetry
        col_means = X.mean(axis=0) + 1  # (p,)
        
        # Create factor-diverse initialization
        # Each factor has different genes highly loaded
        self.a_beta = np.full((self.p, self.d), self.alpha_beta + 0.1)
        self.b_beta = np.full((self.p, self.d), 1.0)
        
        # Assign each gene to a "preferred" factor with higher loading
        gene_assignments = np.arange(self.p) % self.d
        for j in range(self.p):
            preferred_factor = gene_assignments[j]
            self.a_beta[j, preferred_factor] += col_means[j] / self.d * 2
            # Add noise to other factors
            for ell in range(self.d):
                if ell != preferred_factor:
                    self.a_beta[j, ell] += col_means[j] / self.d * 0.5 * (1 + 0.2 * self.rng.random())
        
        # η: Gene activities, Gamma(a_η, b_η)
        self.a_eta = np.full(self.p, self.alpha_eta + self.d * self.alpha_beta)
        self.b_eta = np.full(self.p, self.lambda_eta)
        
        # v: Regression coefficients, N(μ_v, Σ_v)
        # Initialize with small random values
        self.mu_v = 0.1 * self.rng.standard_normal((self.kappa, self.d))
        self.Sigma_v = np.array([np.eye(self.d) * self.sigma_v**2 for _ in range(self.kappa)])
        
        # γ: Auxiliary coefficients, N(μ_γ, Σ_γ)
        if self.p_aux > 0:
            self.mu_gamma = np.zeros((self.kappa, self.p_aux))
            self.Sigma_gamma = np.array([np.eye(self.p_aux) * self.sigma_gamma**2 for _ in range(self.kappa)])
        else:
            self.mu_gamma = np.zeros((self.kappa, 0))
            self.Sigma_gamma = np.zeros((self.kappa, 0, 0))
        
        # Spike-and-slab indicators
        if self.use_spike_slab:
            self.rho_beta = np.full((self.p, self.d), self.pi_beta)
            self.rho_v = np.full((self.kappa, self.d), self.pi_v)
        else:
            self.rho_beta = np.ones((self.p, self.d))
            self.rho_v = np.ones((self.kappa, self.d))
        
        # Store natural parameters for SVI updates
        self._update_natural_params()
        self._compute_expectations()
        
        # Debug: check initial diversity
        print(f"Initial beta diversity: {np.std(self.E_beta, axis=1).mean():.4f}")
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
            self.mu_v[k], self.Sigma_v[k] = self._natural_to_gaussian(
                self.eta1_v[k], self.eta2_v[k]
            )
            self.mu_gamma[k], self.Sigma_gamma[k] = self._natural_to_gaussian(
                self.eta1_gamma[k], self.eta2_gamma[k]
            )
    
    def _compute_expectations(self):
        """Compute expected sufficient statistics from current parameters."""
        # E[β], E[log β]
        self.E_beta = self.a_beta / self.b_beta
        self.E_log_beta = digamma(self.a_beta) - np.log(self.b_beta)
        
        # E[η], E[log η]
        self.E_eta = self.a_eta / self.b_eta
        self.E_log_eta = digamma(self.a_eta) - np.log(self.b_eta)
        
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
    
    # =========================================================================
    # LOCAL PARAMETER UPDATES (Full optimization per mini-batch)
    # =========================================================================
    
    def _update_local_parameters(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        X_aux_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimize local parameters (θ, ξ, ζ) for a mini-batch.
        
        Returns: (a_theta, b_theta, a_xi, b_xi, zeta)
        """
        batch_size = X_batch.shape[0]
        
        # Initialize local params
        row_sums = X_batch.sum(axis=1, keepdims=True) + 1
        theta_init = row_sums / self.d
        
        a_theta = np.full((batch_size, self.d), self.alpha_theta) + theta_init
        b_theta = np.full((batch_size, self.d), 1.0)
        a_xi = np.full(batch_size, self.alpha_xi)
        b_xi = np.full(batch_size, self.lambda_xi)
        zeta = np.ones((batch_size, self.kappa))
        
        for _ in range(self.local_iterations):
            # Current expectations
            E_theta = a_theta / b_theta
            E_log_theta = digamma(a_theta) - np.log(b_theta)
            E_xi = a_xi / b_xi
            
            # Update φ (multinomial allocations) - implicit in z expectations
            # φ_ijℓ ∝ exp(E[log θ_iℓ] + E[log β_jℓ])
            log_phi = E_log_theta[:, np.newaxis, :] + self.E_log_beta_effective[np.newaxis, :, :]
            log_phi_max = log_phi.max(axis=2, keepdims=True)
            phi = np.exp(log_phi - log_phi_max)
            phi = phi / (phi.sum(axis=2, keepdims=True) + 1e-10)
            
            # E[z_ijℓ] = x_ij * φ_ijℓ
            E_z = X_batch[:, :, np.newaxis] * phi  # (batch, p, d)
            
            # Update θ: Gamma(a_θ, b_θ)
            # a_θ_iℓ = α_θ + Σ_j E[z_ijℓ]
            # b_θ_iℓ = E[ξ_i] + Σ_j E[β_jℓ] + R_iℓ (regression term from JJ bound)
            
            a_theta_new = self.alpha_theta + E_z.sum(axis=1)
            
            b_theta_new = E_xi[:, np.newaxis] + self.E_beta_effective.sum(axis=0)[np.newaxis, :]
            
            # Add regression contribution (from JJ bound, PSB Eq. A.23)
            for k in range(self.kappa):
                y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
                lam = self._lambda_jj(zeta[:, k])
                
                # C^{(-ℓ)}_{ik} = Σ_{m≠ℓ} E[θ_im] E[v_km] + x^aux_i · E[γ_k]
                # For each ℓ:
                theta_v = E_theta @ self.E_v_effective[k]  # (batch,)
                aux_term = X_aux_batch @ self.E_gamma[k] if self.p_aux > 0 else 0.0  # (batch,)
                
                # Vectorized over ℓ
                C_minus_ell = (theta_v[:, np.newaxis] - 
                              E_theta * self.E_v_effective[k][np.newaxis, :] +
                              aux_term if isinstance(aux_term, float) else aux_term[:, np.newaxis])  # (batch, d)
                
                E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
                
                # R_iℓ = -(y_ik - 0.5) v_kℓ + 2λ(ζ_ik) v_kℓ C^{(-ℓ)} + 2λ(ζ_ik) E[v²_kℓ] E[θ_iℓ]
                R = (-(y_k - 0.5)[:, np.newaxis] * self.E_v_effective[k][np.newaxis, :] +
                     2 * lam[:, np.newaxis] * self.E_v_effective[k][np.newaxis, :] * C_minus_ell +
                     2 * lam[:, np.newaxis] * E_v_sq[np.newaxis, :] * E_theta)
                
                b_theta_new += self.regression_weight * R
            
            b_theta_new = np.maximum(b_theta_new, 1e-6)
            a_theta_new = np.maximum(a_theta_new, 1.001)
            
            # Update ξ: Gamma(a_ξ, b_ξ)
            # a_ξ_i = α_ξ + d·α_θ
            # b_ξ_i = λ_ξ + Σ_ℓ E[θ_iℓ]
            E_theta_new = a_theta_new / b_theta_new
            a_xi_new = np.full(batch_size, self.alpha_xi + self.d * self.alpha_theta)
            b_xi_new = self.lambda_xi + E_theta_new.sum(axis=1)
            
            # Update ζ (JJ auxiliary)
            # ζ_ik = sqrt(E[A²_ik])
            for k in range(self.kappa):
                aux_contrib = X_aux_batch @ self.E_gamma[k] if self.p_aux > 0 else 0.0
                E_A = E_theta_new @ self.E_v_effective[k] + aux_contrib
                E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
                Var_theta = a_theta_new / (b_theta_new**2)
                E_A_sq = E_A**2 + (Var_theta * E_v_sq[np.newaxis, :]).sum(axis=1)
                zeta[:, k] = np.sqrt(np.maximum(E_A_sq, 1e-8))
            
            a_theta, b_theta = a_theta_new, b_theta_new
            a_xi, b_xi = a_xi_new, b_xi_new
        
        return a_theta, b_theta, a_xi, b_xi, zeta
    
    # =========================================================================
    # INTERMEDIATE GLOBAL PARAMETERS (as if mini-batch were full dataset)
    # =========================================================================
    
    def _compute_intermediate_beta(
        self,
        X_batch: np.ndarray,
        E_theta: np.ndarray,
        E_log_theta: np.ndarray,
        scale: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute intermediate β parameters using natural gradient.
        
        From Hoffman et al., the intermediate parameter is:
        λ̂ = α + N · E[t(x, z)]
        
        For β ~ Gamma(α_β, η_j), the complete conditional natural params are:
        η₁ = α_β - 1 + Σ_i z_ijℓ
        η₂ = -η_j - Σ_i θ_iℓ
        """
        # Compute E[z_ijℓ] = x_ij · φ_ijℓ
        log_phi = E_log_theta[:, np.newaxis, :] + self.E_log_beta_effective[np.newaxis, :, :]
        log_phi = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
        phi = np.exp(log_phi)
        
        # Sufficient statistics
        # Σ_i z_ijℓ for each gene j and factor ℓ
        z_sum = (X_batch[:, :, np.newaxis] * phi).sum(axis=0)  # (p, d)
        theta_sum = E_theta.sum(axis=0)  # (d,)
        
        # Intermediate natural parameters (pretend batch is full dataset)
        eta1_hat = (self.alpha_beta - 1) + scale * z_sum
        eta2_hat = -self.E_eta[:, np.newaxis] - scale * theta_sum[np.newaxis, :]
        
        # Convert to canonical
        a_beta_hat, b_beta_hat = self._natural_to_gamma(eta1_hat, eta2_hat)
        
        return a_beta_hat, b_beta_hat
    
    def _compute_intermediate_eta(
        self,
        E_theta: np.ndarray,
        scale: float
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            np.full(self.p, eta1_hat), eta2_hat
        )
        
        return a_eta_hat, b_eta_hat
    
    def _compute_intermediate_v(
        self,
        y_batch: np.ndarray,
        X_aux_batch: np.ndarray,
        E_theta: np.ndarray,
        a_theta: np.ndarray,
        b_theta: np.ndarray,
        zeta: np.ndarray,
        scale: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute intermediate v parameters.
        
        v_k ~ N(0, σ_v² I)
        From JJ bound, the approximate posterior is Gaussian with:
        - Precision: (1/σ_v²) I + 2 Σ_i λ(ζ_ik) E[θ_i⊗θ_i]
        - Mean precision: Σ_i (y_ik - 0.5) E[θ_i] - 2λ(ζ_ik) E[θ_i](x^aux · γ_k)
        
        For diagonal approximation (independent v_{kℓ}):
        - precision_ℓ = 1/σ_v² + 2 Σ_i λ(ζ_ik) E[θ²_iℓ]  
        - mean·precision_ℓ = Σ_i [(y_ik - 0.5) - 2λ(ζ_ik) C_{ik}] E[θ_iℓ]
        
        where C_{ik} = Σ_{m≠ℓ} E[θ_im] E[v_km] + x^aux · γ_k
        """
        mu_v_hat = np.zeros((self.kappa, self.d))
        Sigma_v_hat = np.zeros((self.kappa, self.d, self.d))
        
        for k in range(self.kappa):
            y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
            lam = self._lambda_jj(zeta[:, k])  # (batch,)
            
            # E[θ²] = E[θ]² + Var[θ]
            Var_theta = a_theta / (b_theta**2)
            E_theta_sq = E_theta**2 + Var_theta  # (batch, d)
            
            # Auxiliary contribution
            aux_contrib = X_aux_batch @ self.E_gamma[k] if self.p_aux > 0 else 0.0  # (batch,)
            
            # Full model contribution (before removing ℓ-th factor)
            full_theta_v = E_theta @ self.E_v_effective[k]  # (batch,)
            
            # Diagonal precision and mean for each factor ℓ
            precision = np.full(self.d, 1.0 / self.sigma_v**2)
            mean_contrib = np.zeros(self.d)
            
            for ell in range(self.d):
                # C_{ik}^{(-ℓ)} = full - E[θ_iℓ] E[v_kℓ] + aux
                C_minus_ell = full_theta_v - E_theta[:, ell] * self.E_v_effective[k, ell] + aux_contrib
                
                # Precision contribution: 2 Σ_i λ(ζ_ik) E[θ²_iℓ]
                precision[ell] += 2 * scale * np.sum(lam * E_theta_sq[:, ell])
                
                # Mean*precision contribution: Σ_i [(y_ik - 0.5) - 2λ(ζ_ik) C_{ik}^{(-ℓ)}] E[θ_iℓ]
                mean_contrib[ell] = scale * np.sum(
                    ((y_k - 0.5) - 2 * lam * C_minus_ell) * E_theta[:, ell]
                )
            
            # Convert to mean and variance
            Sigma_v_hat[k] = np.diag(1.0 / precision)
            mu_v_hat[k] = mean_contrib / precision
            
            # Clip for stability
            mu_v_hat[k] = np.clip(mu_v_hat[k], -10, 10)
        
        return mu_v_hat, Sigma_v_hat
    
    def _compute_intermediate_gamma(
        self,
        y_batch: np.ndarray,
        X_aux_batch: np.ndarray,
        E_theta: np.ndarray,
        zeta: np.ndarray,
        scale: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute intermediate γ parameters."""
        if self.p_aux == 0:
            return np.zeros((self.kappa, 0)), np.zeros((self.kappa, 0, 0))
        
        mu_gamma_hat = np.zeros((self.kappa, self.p_aux))
        Sigma_gamma_hat = np.zeros((self.kappa, self.p_aux, self.p_aux))
        
        for k in range(self.kappa):
            y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
            lam = self._lambda_jj(zeta[:, k])
            
            # Precision
            prec_prior = np.eye(self.p_aux) / self.sigma_gamma**2
            prec_lik = 2 * scale * X_aux_batch.T @ (lam[:, np.newaxis] * X_aux_batch)
            prec_hat = prec_prior + prec_lik
            
            # Mean
            theta_v = E_theta @ self.E_v_effective[k]
            mean_contrib = scale * X_aux_batch.T @ (y_k - 0.5 - 2 * lam * theta_v)
            
            try:
                Sigma_gamma_hat[k] = np.linalg.inv(prec_hat)
                mu_gamma_hat[k] = Sigma_gamma_hat[k] @ mean_contrib
            except np.linalg.LinAlgError:
                Sigma_gamma_hat[k] = self.Sigma_gamma[k]
                mu_gamma_hat[k] = self.mu_gamma[k]
        
        return mu_gamma_hat, Sigma_gamma_hat
    
    # =========================================================================
    # SVI UPDATES (Natural Gradient)
    # =========================================================================
    
    def _svi_update_global(
        self,
        rho_t: float,
        a_beta_hat: np.ndarray, b_beta_hat: np.ndarray,
        a_eta_hat: np.ndarray, b_eta_hat: np.ndarray,
        mu_v_hat: np.ndarray, Sigma_v_hat: np.ndarray,
        mu_gamma_hat: np.ndarray, Sigma_gamma_hat: np.ndarray
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
            self.mu_v[k] = (1 - rho_t) * self.mu_v[k] + rho_t * mu_v_hat[k]
            self.Sigma_v[k] = (1 - rho_t) * self.Sigma_v[k] + rho_t * Sigma_v_hat[k]
            # Ensure positive definiteness
            self.Sigma_v[k] = 0.5 * (self.Sigma_v[k] + self.Sigma_v[k].T)
            eigvals = np.linalg.eigvalsh(self.Sigma_v[k])
            if np.min(eigvals) < 1e-6:
                self.Sigma_v[k] += (1e-6 - np.min(eigvals) + 1e-8) * np.eye(self.d)
        
        # gamma: update canonical parameters directly
        if self.p_aux > 0:
            for k in range(self.kappa):
                self.mu_gamma[k] = (1 - rho_t) * self.mu_gamma[k] + rho_t * mu_gamma_hat[k]
                self.Sigma_gamma[k] = (1 - rho_t) * self.Sigma_gamma[k] + rho_t * Sigma_gamma_hat[k]
                self.Sigma_gamma[k] = 0.5 * (self.Sigma_gamma[k] + self.Sigma_gamma[k].T)
                eigvals = np.linalg.eigvalsh(self.Sigma_gamma[k])
                if np.min(eigvals) < 1e-6:
                    self.Sigma_gamma[k] += (1e-6 - np.min(eigvals) + 1e-8) * np.eye(self.p_aux)
        
        # Update expectations
        self._compute_expectations()
    
    # =========================================================================
    # ELBO COMPUTATION
    # =========================================================================
    
    def _compute_elbo(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        X_aux_batch: np.ndarray,
        a_theta: np.ndarray,
        b_theta: np.ndarray,
        a_xi: np.ndarray,
        b_xi: np.ndarray,
        zeta: np.ndarray,
        scale: float
    ) -> float:
        """
        Compute ELBO for mini-batch, scaled to full dataset.
        
        ELBO = E[log p(X, y, z, θ, ξ, β, η, v, γ)] - E[log q(all)]
        """
        batch_size = X_batch.shape[0]
        elbo = 0.0
        
        # Local expectations
        E_theta = a_theta / b_theta
        E_log_theta = digamma(a_theta) - np.log(b_theta)
        E_xi = a_xi / b_xi
        E_log_xi = digamma(a_xi) - np.log(b_xi)
        
        # === Poisson likelihood (via collapsed z) ===
        # E[log p(X | θ, β)] = Σ_ij x_ij log(Σ_ℓ θ_iℓ β_jℓ) - Σ_ij Σ_ℓ θ_iℓ β_jℓ
        # Using log-sum-exp for stability
        log_rates = E_log_theta[:, np.newaxis, :] + self.E_log_beta_effective[np.newaxis, :, :]
        log_sum_rates = logsumexp(log_rates, axis=2)  # (batch, p)
        elbo_x = np.sum(X_batch * log_sum_rates)
        elbo_x -= np.sum(E_theta.sum(axis=0) * self.E_beta_effective.sum(axis=0))
        elbo += scale * elbo_x
        
        # === Bernoulli likelihood (JJ bound) ===
        elbo_y = 0.0
        for k in range(self.kappa):
            y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
            lam = self._lambda_jj(zeta[:, k])
            
            aux_contrib = X_aux_batch @ self.E_gamma[k] if self.p_aux > 0 else 0.0
            E_A = E_theta @ self.E_v_effective[k] + aux_contrib
            E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
            Var_theta = a_theta / (b_theta**2)
            E_A_sq = E_A**2 + (Var_theta * E_v_sq[np.newaxis, :]).sum(axis=1)
            
            # JJ bound: (y - 0.5) E[A] - λ(ζ) E[A²] + λ(ζ) ζ² - ζ/2 - log(1 + exp(ζ))
            elbo_y += np.sum((y_k - 0.5) * E_A - lam * E_A_sq)
            elbo_y += np.sum(lam * zeta[:, k]**2 - 0.5 * zeta[:, k] - 
                            np.log1p(np.exp(np.clip(zeta[:, k], -500, 500))))
        elbo += scale * self.regression_weight * elbo_y
        
        # === Priors on local parameters ===
        # p(θ | ξ)
        elbo_theta = np.sum((self.alpha_theta - 1) * E_log_theta + 
                           self.alpha_theta * E_log_xi[:, np.newaxis] -
                           E_xi[:, np.newaxis] * E_theta)
        elbo_theta -= batch_size * self.d * gammaln(self.alpha_theta)
        elbo += scale * elbo_theta
        
        # p(ξ)
        elbo_xi = np.sum((self.alpha_xi - 1) * E_log_xi - self.lambda_xi * E_xi)
        elbo_xi += batch_size * (self.alpha_xi * np.log(self.lambda_xi) - gammaln(self.alpha_xi))
        elbo += scale * elbo_xi
        
        # === Priors on global parameters (not scaled) ===
        # p(β | η)
        elbo_beta = np.sum((self.alpha_beta - 1) * self.E_log_beta +
                          self.alpha_beta * self.E_log_eta[:, np.newaxis] -
                          self.E_eta[:, np.newaxis] * self.E_beta)
        elbo_beta -= self.p * self.d * gammaln(self.alpha_beta)
        elbo += elbo_beta
        
        # p(η)
        elbo_eta = np.sum((self.alpha_eta - 1) * self.E_log_eta - self.lambda_eta * self.E_eta)
        elbo_eta += self.p * (self.alpha_eta * np.log(self.lambda_eta) - gammaln(self.alpha_eta))
        elbo += elbo_eta
        
        # p(v)
        elbo_v = 0.0
        for k in range(self.kappa):
            elbo_v -= 0.5 * self.d * np.log(2 * np.pi * self.sigma_v**2)
            elbo_v -= 0.5 / self.sigma_v**2 * (
                np.sum(self.mu_v[k]**2) + np.trace(self.Sigma_v[k])
            )
        elbo += elbo_v
        
        # p(γ)
        elbo_gamma = 0.0
        if self.p_aux > 0:
            for k in range(self.kappa):
                elbo_gamma -= 0.5 * self.p_aux * np.log(2 * np.pi * self.sigma_gamma**2)
                elbo_gamma -= 0.5 / self.sigma_gamma**2 * (
                    np.sum(self.mu_gamma[k]**2) + np.trace(self.Sigma_gamma[k])
                )
        elbo += elbo_gamma
        
        # === Entropy terms ===
        # H[q(θ)]
        H_theta = np.sum(a_theta - np.log(b_theta) + gammaln(a_theta) +
                        (1 - a_theta) * digamma(a_theta))
        elbo += scale * H_theta
        
        # H[q(ξ)]
        H_xi = np.sum(a_xi - np.log(b_xi) + gammaln(a_xi) +
                     (1 - a_xi) * digamma(a_xi))
        elbo += scale * H_xi
        
        # H[q(β)]
        H_beta = np.sum(self.a_beta - np.log(self.b_beta) + gammaln(self.a_beta) +
                       (1 - self.a_beta) * digamma(self.a_beta))
        elbo += H_beta
        
        # H[q(η)]
        H_eta = np.sum(self.a_eta - np.log(self.b_eta) + gammaln(self.a_eta) +
                      (1 - self.a_eta) * digamma(self.a_eta))
        elbo += H_eta
        
        # H[q(v)] = 0.5 * d * (1 + log(2π)) + 0.5 * log|Σ|
        for k in range(self.kappa):
            sign, logdet = np.linalg.slogdet(self.Sigma_v[k])
            if sign > 0:
                elbo += 0.5 * self.d * (1 + np.log(2 * np.pi)) + 0.5 * logdet
        
        # H[q(γ)]
        if self.p_aux > 0:
            for k in range(self.kappa):
                sign, logdet = np.linalg.slogdet(self.Sigma_gamma[k])
                if sign > 0:
                    elbo += 0.5 * self.p_aux * (1 + np.log(2 * np.pi)) + 0.5 * logdet
        
        return elbo
    
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
        early_stopping: bool = False
    ):
        """
        Fit model using Stochastic Variational Inference.
        
        Convergence tracked via EMA + Welford's online algorithm:
        - elbo_ema_: Exponential moving average of batch ELBO
        - elbo_welford_mean_: Running mean (Welford)
        - elbo_welford_var_: Running variance (Welford)
        - convergence_history_: List of (epoch, ema, mean, std, rel_change)
        
        Parameters
        ----------
        early_stopping : bool
            If True, stop when EMA relative change < convergence_tol for 
            convergence_window consecutive checks.
        """
        # Ensure y is 2D
        if y.ndim == 1:
            y = y[:, np.newaxis]
        
        # Initialize
        self._initialize_global_parameters(X, y, X_aux)
        
        # Use ceiling division to ensure all samples are processed
        n_batches = max(1, (self.n + self.batch_size - 1) // self.batch_size)
        iteration = 0
        self.elbo_history_ = []
        
        # =====================================================================
        # EMA + Welford initialization (O(1) memory)
        # =====================================================================
        self.elbo_ema_ = None           # Exponential moving average
        self.elbo_ema_prev_ = None      # Previous EMA for relative change
        self.elbo_welford_n_ = 0        # Welford count
        self.elbo_welford_mean_ = 0.0   # Welford running mean
        self.elbo_welford_M2_ = 0.0     # Welford sum of squared deviations
        self.convergence_history_ = []  # (epoch, ema, mean, std, rel_change)
        self.last_elbo_ = None          # Track last computed ELBO for display
        consecutive_converged = 0
        
        # Storage for final training set local parameters
        self.train_a_theta_ = None
        self.train_b_theta_ = None
        self.train_a_xi_ = None
        self.train_b_xi_ = None
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            # Shuffle data
            perm = self.rng.permutation(self.n)
            
            # Storage for training parameters (overwritten each epoch)
            epoch_a_theta = np.zeros((self.n, self.d))
            epoch_b_theta = np.zeros((self.n, self.d))
            epoch_a_xi = np.zeros(self.n)
            epoch_b_xi = np.zeros(self.n)
            
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
                epoch_a_theta[idx] = a_theta
                epoch_b_theta[idx] = b_theta
                epoch_a_xi[idx] = a_xi
                epoch_b_xi[idx] = b_xi
                
                E_theta = a_theta / b_theta
                E_log_theta = digamma(a_theta) - np.log(b_theta)
                
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
                        
                        # =====================================================
                        # EMA update: O(1)
                        # =====================================================
                        if self.elbo_ema_ is None:
                            self.elbo_ema_ = elbo
                        else:
                            self.elbo_ema_ = (self.ema_decay * self.elbo_ema_ + 
                                              (1 - self.ema_decay) * elbo)
                        
                        # =====================================================
                        # Welford's online update: O(1)
                        # Mean and variance without storing history
                        # =====================================================
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
            
            self.elbo_ema_prev_ = self.elbo_ema_
            
            # Store convergence diagnostics (sparse: every 5 epochs or at elbo_freq boundaries)
            if epoch % 5 == 0:
                self.convergence_history_.append((
                    epoch,
                    self.elbo_ema_ if self.elbo_ema_ is not None else np.nan,
                    self.elbo_welford_mean_,
                    welford_std,
                    rel_change
                ))
            
            if verbose and epoch % 5 == 0:
                beta_diversity = np.std(self.E_beta, axis=1).mean()
                # Include EMA and relative change in output
                ema_str = f"{self.elbo_ema_:.2e}" if self.elbo_ema_ is not None else "N/A"
                rel_str = f"{rel_change:.2e}" if rel_change != np.inf else "N/A"
                # Use last_elbo_ to avoid showing 0 when no ELBO computed this epoch
                elbo_str = f"{self.last_elbo_:.2e}" if self.last_elbo_ is not None else "N/A"
                print(f"Epoch {epoch}: ELBO = {elbo_str}, EMA = {ema_str}, "
                      f"Δrel = {rel_str}, ρ_t = {rho_t:.4f}, "
                      f"v = {self.mu_v.ravel()[:3]}, β_div = {beta_diversity:.3f}")
            
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
        
        return self
    
    def transform(self, X_new: np.ndarray, y_new: np.ndarray = None, 
                  X_aux_new: np.ndarray = None, n_iter: int = 50) -> dict:
        """
        Infer θ for new samples with frozen global parameters (β, η, v, γ).
        
        Parameters
        ----------
        X_new : (n_new, p) count matrix
        y_new : (n_new,) or (n_new, κ) labels (optional, for ζ updates)
        X_aux_new : (n_new, p_aux) auxiliary covariates
        n_iter : local VI iterations
        
        Returns
        -------
        dict with keys: 'E_theta', 'a_theta', 'b_theta', 'a_xi', 'b_xi'
        """
        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, self.p_aux if self.p_aux > 0 else 0))
        if y_new is None:
            y_new = np.full((n_new, self.kappa), 0.5)  # Neutral labels
        y_new = y_new.reshape(-1, 1) if y_new.ndim == 1 else y_new
        
        # Initialize local parameters with factor diversity
        row_sums = X_new.sum(axis=1, keepdims=True) + 1
        factor_scales = np.linspace(0.5, 2.0, self.d)
        
        a_theta = np.full((n_new, self.d), self.alpha_theta + self.p * self.alpha_beta)
        b_theta = row_sums / self.d * factor_scales
        a_xi = np.full(n_new, self.alpha_xi + self.d * self.alpha_theta)
        b_xi = np.full(n_new, self.lambda_xi)
        
        # Frozen global expectations
        E_beta = self.a_beta / self.b_beta          # (p, d)
        E_log_beta = digamma(self.a_beta) - np.log(self.b_beta)
        E_v = self.mu_v                              # (κ, d)
        E_v_sq = self.mu_v**2 + np.diagonal(self.Sigma_v, axis1=1, axis2=2)
        E_gamma = self.mu_gamma                      # (κ, p_aux)
        
        beta_col_sums = E_beta.sum(axis=0)  # (d,)
        
        for it in range(n_iter):
            # Current local expectations
            E_theta = a_theta / b_theta              # (n_new, d)
            E_log_theta = digamma(a_theta) - np.log(b_theta)
            E_xi = a_xi / b_xi
            E_theta_sq = (a_theta * (a_theta + 1)) / b_theta**2
            
            # Update ξ
            a_xi = self.alpha_xi + self.d * self.alpha_theta
            b_xi = self.lambda_xi + E_theta.sum(axis=1)
            E_xi = a_xi / b_xi
            
            # Compute φ for auxiliary variable z
            log_phi = E_log_theta[:, np.newaxis, :] + E_log_beta[np.newaxis, :, :]  # (n, p, d)
            log_phi -= logsumexp(log_phi, axis=2, keepdims=True)
            phi = np.exp(log_phi)
            
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
            zeta = np.sqrt(np.maximum(E_A_sq, 1e-10))
            lam = np.tanh(zeta / 2) / (4 * zeta + 1e-10)
            
            # Update θ
            # Shape: sum over genes of X * φ
            shape_contrib = np.einsum('ij,ijl->il', X_new, phi)  # (n_new, d)
            a_theta = self.alpha_theta + shape_contrib
            
            # Rate: ξ + sum_j β_jℓ + regression term R_iℓ
            b_theta = E_xi[:, np.newaxis] + beta_col_sums[np.newaxis, :]
            
            # Add regression contribution from JJ bound
            for k in range(self.kappa):
                y_k = y_new[:, k]
                for ell in range(self.d):
                    C_minus_ell = (E_A[:, k] - E_theta[:, ell] * E_v[k, ell])
                    R_ell = (-(y_k - 0.5) * E_v[k, ell] + 
                             2 * lam[:, k] * E_v[k, ell] * C_minus_ell +
                             2 * lam[:, k] * E_v_sq[k, ell] * E_theta[:, ell])
                    b_theta[:, ell] += R_ell
            
            b_theta = np.maximum(b_theta, 1e-10)
        
        E_theta_final = a_theta / b_theta
        
        return {
            'E_theta': E_theta_final,
            'a_theta': a_theta,
            'b_theta': b_theta,
            'a_xi': a_xi,
            'b_xi': b_xi
        }

    def predict_proba(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                      n_iter: int = 50) -> np.ndarray:
        """
        Predict class probabilities for new samples.
        
        Parameters
        ----------
        X_new : (n_new, p) count matrix
        X_aux_new : (n_new, p_aux) auxiliary covariates
        n_iter : local VI iterations for θ inference
        
        Returns
        -------
        proba : (n_new,) or (n_new, κ) predicted probabilities
        """
        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, self.p_aux if self.p_aux > 0 else 0))
            
        result = self.transform(X_new, y_new=None, X_aux_new=X_aux_new, n_iter=n_iter)
        E_theta = result['E_theta']
        
        # Compute logits: θ @ v^T + X_aux @ γ^T
        if self.p_aux > 0:
            logits = E_theta @ self.mu_v.T + X_aux_new @ self.mu_gamma.T
        else:
            logits = E_theta @ self.mu_v.T
        proba = expit(logits)
        
        return proba.squeeze()

    def predict(self, X_new: np.ndarray, X_aux_new: np.ndarray = None,
                n_iter: int = 50, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for new samples.
        
        Parameters
        ----------
        X_new : (n_new, p) count matrix
        X_aux_new : (n_new, p_aux) auxiliary covariates
        n_iter : local VI iterations for θ inference
        threshold : classification threshold
        
        Returns
        -------
        labels : (n_new,) or (n_new, κ) predicted labels
        """
        proba = self.predict_proba(X_new, X_aux_new, n_iter)
        return (proba >= threshold).astype(int)


if __name__ == "__main__":
    # Test with train/test split
    np.random.seed(42)
    n_train, n_test, p, d = 200, 50, 50, 5
    
    # Generate synthetic data
    theta_true_train = np.random.gamma(2, 1, (n_train, d))
    theta_true_test = np.random.gamma(2, 1, (n_test, d))
    beta_true = np.random.gamma(2, 1, (p, d))
    
    X_train = np.random.poisson(theta_true_train @ beta_true.T)
    X_test = np.random.poisson(theta_true_test @ beta_true.T)
    
    v_true = np.array([[1, -1, 0.5, 0, 0]])
    
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
        random_state=42
    )
    model.fit(X_train, y_train, X_aux_train, max_epochs=100, verbose=True)
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Final v:\n{model.mu_v}")
    print(f"True v:\n{v_true}")
    print(f"v correlation: {np.corrcoef(model.mu_v.ravel(), v_true.ravel())[0,1]:.3f}")
    
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
