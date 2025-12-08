import numpy as np
from scipy.special import digamma, gammaln, expit, logsumexp
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
# from .utils import *
# from .data import *
# from .config import *

class VI:
    
    def __init__(
        self,
        n_factors: int,
        alpha_theta: float = 2.0,
        alpha_beta: float = 2.0,
        alpha_xi: float = 2.0,
        alpha_eta: float = 2.0,
        lambda_xi: float = 1.5,
        lambda_eta: float = 1.5,
        sigma_v: float = 0.1,
        sigma_gamma: float = 0.1,
        random_state: int = 42,
        # Spike-and-slab parameters
        pi_v: float = 0.05,  # Prior probability of v being active
        pi_beta: float = 0.05,  # Prior probability of beta being active
        spike_variance_v: float = 1e-6,  # Variance for spike in v
        spike_value_beta: float = 1e-6  # Small value for spike in beta
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
        self.rng = np.random.RandomState(random_state)
        self.regression_weight = 0.01
        
        # Spike-and-slab parameters
        self.pi_v = pi_v
        self.pi_beta = pi_beta
        self.spike_variance_v = spike_variance_v
        self.spike_value_beta = spike_value_beta
        
    def _initialize_parameters(self, X: np.ndarray, y: np.ndarray, X_aux: np.ndarray):
        """Initialize variational parameters with data-driven initialization."""
        self.n, self.p = X.shape
        self.kappa = y.shape[1] if y.ndim > 1 else 1
        self.p_aux = X_aux.shape[1]
        
        # Better initialization: use data statistics
        # Initialize theta and beta based on row/column sums of X
        row_sums = X.sum(axis=1, keepdims=True) + 1e-6
        col_sums = X.sum(axis=0, keepdims=True) + 1e-6
        total_sum = X.sum() + 1e-6
        
        # Normalize and add small random perturbation
        theta_init = (row_sums / self.d) * (1 + 0.1 * self.rng.randn(self.n, self.d))
        beta_init = (col_sums.T / self.d) * (1 + 0.1 * self.rng.randn(self.p, self.d))
        
        # Clip to ensure positive values
        theta_init = np.maximum(theta_init, 0.1)
        beta_init = np.maximum(beta_init, 0.1)
        
        # Initialize Gamma parameters from moments
        # For Gamma(a, b): mean = a/b, variance = a/b²
        # Set variance to be a fraction of mean²
        variance_scale = 0.1
        
        self.a_theta = (theta_init**2) / (variance_scale * theta_init**2) + self.alpha_theta
        self.b_theta = theta_init / (variance_scale * theta_init**2) + 0.1
        
        self.a_beta = (beta_init**2) / (variance_scale * beta_init**2) + self.alpha_beta
        self.b_beta = beta_init / (variance_scale * beta_init**2) + 0.1
        
        self.a_xi = self.alpha_xi + 0.1 * self.rng.exponential(size=self.n)
        self.b_xi = np.ones(self.n) * self.lambda_xi
        
        self.a_eta = self.alpha_eta + 0.1 * self.rng.exponential(size=self.p)
        self.b_eta = np.ones(self.p) * self.lambda_eta
        
        # Initialize Normal parameters - small values near zero
        base = self.rng.randn(self.kappa, self.d)
        scale_factors = 0.5 + 0.5 * np.arange(self.kappa) / self.kappa
        self.mu_v = base * scale_factors[:, np.newaxis]
        self.Sigma_v = np.tile(np.eye(self.d)[np.newaxis, :, :], (self.kappa, 1, 1))
        
        # Initialize spike-and-slab indicators for v
        # rho_v[k, ell] = q(s_v_kell = 1) - probability that v_kell is active (slab)
        self.rho_v = np.ones((self.kappa, self.d)) * self.pi_v

        self.mu_gamma = 0.01 * self.rng.randn(self.kappa, self.p_aux)
        self.Sigma_gamma = np.tile(np.eye(self.p_aux)[np.newaxis, :, :], (self.kappa, 1, 1))
        
        # Initialize spike-and-slab indicators for beta
        # rho_beta[j, ell] = q(s_beta_jell = 1) - probability that beta_jell is active (slab)
        self.rho_beta = np.ones((self.p, self.d)) * self.pi_beta
        
        # Initialize Jaakola-Jordan auxiliary parameters with better starting values
        self.zeta = np.ones((self.n, self.kappa)) * 0.1
        
    def _compute_expectations(self):
        """Compute all needed expectations."""
        # Gamma distributions: E[x] = a/b, E[log x] = psi(a) - log(b)
        self.E_theta = self.a_theta / self.b_theta
        self.E_log_theta = digamma(self.a_theta) - np.log(self.b_theta)
        
        # Spike-and-slab for beta: E[beta] = rho * E[beta | slab] + (1-rho) * spike_value
        self.E_beta_slab = self.a_beta / self.b_beta
        self.E_log_beta_slab = digamma(self.a_beta) - np.log(self.b_beta)
        self.E_beta = self.rho_beta * self.E_beta_slab + (1 - self.rho_beta) * self.spike_value_beta
        self.E_log_beta = self.rho_beta * self.E_log_beta_slab + (1 - self.rho_beta) * np.log(self.spike_value_beta + 1e-10)
        
        self.E_xi = self.a_xi / self.b_xi
        self.E_log_xi = digamma(self.a_xi) - np.log(self.b_xi)
        
        self.E_eta = self.a_eta / self.b_eta
        self.E_log_eta = digamma(self.a_eta) - np.log(self.b_eta)
        
        # Spike-and-slab for v: E[v] = rho * E[v | slab] + (1-rho) * 0
        self.E_v_slab = self.mu_v
        self.E_v = self.rho_v[:, :, np.newaxis] * self.mu_v[:, :, np.newaxis]
        self.E_v = self.E_v.squeeze(-1)  # Remove extra dimension
        
        # Normal distributions: expectations are just the means
        self.E_gamma = self.mu_gamma
        
    def _lambda_jj(self, zeta: np.ndarray) -> np.ndarray:
        """Jaakola-Jordan lambda function."""
        result = np.zeros_like(zeta)
        nonzero = np.abs(zeta) > 1e-10
        result[nonzero] = (1.0 / (4.0 * zeta[nonzero])) * np.tanh(zeta[nonzero] / 2.0)
        result[~nonzero] = 0.125
        return result
    
    def _allocate_counts(self, X: np.ndarray) -> np.ndarray:
        """
        Allocate counts using multinomial with expected log parameters.
        
        This implements the update from Equation (9) in the HPF paper:
        φ_uik ∝ exp{Ψ(a_θ_iℓ) - log(b_θ_iℓ) + Ψ(a_β_jℓ) - log(b_β_jℓ)}
        """
        # Compute unnormalized log probabilities: E[log θ_iℓ] + E[log β_jℓ]
        # Shape: (n, p, d)
        log_phi = (self.E_log_theta[:, np.newaxis, :] + 
                   self.E_log_beta[np.newaxis, :, :])
        
        # Normalize using logsumexp for numerical stability (softmax over ℓ dimension)
        log_phi_normalized = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
        
        # Exponentiate to get probabilities
        phi = np.exp(log_phi_normalized)
        
        # Expected counts: E[z_ijℓ | x_ij] = x_ij * φ_ijℓ
        z = X[:, :, np.newaxis] * phi
        
        return z
    
    def _update_theta(self, z: np.ndarray, y: np.ndarray, X_aux: np.ndarray):
        """Update theta using coordinate ascent (partially vectorized)."""
        E_theta_prev = self.E_theta.copy()
        
        # Shape: alpha_theta + sum_j z_ijl
        self.a_theta = self.alpha_theta + np.sum(z, axis=1)
        
        # Rate: E[xi_i] + sum_j E[beta_jl] + regression terms
        self.b_theta = self.E_xi[:, np.newaxis] + np.sum(self.E_beta, axis=0)[np.newaxis, :]
        
        # Add regression contribution
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = self._lambda_jj(self.zeta[:, k])
            
            # Vectorize over dimensions
            for ell in range(self.d):
                # Compute E[C_kℓ] = sum of other dimensions' contributions
                # Vectorized: (n, d) @ (d,) but exclude current dimension
                mask = np.ones(self.d, dtype=bool)
                mask[ell] = False
                E_C = self.E_theta[:, mask] @ self.E_v[k, mask] + X_aux @ self.E_gamma[k]
                
                # Variance term: E[v²_kℓ]
                E_v_sq_ell = self.mu_v[k, ell]**2 + self.Sigma_v[k, ell, ell]
                
                # Regression contribution (vectorized over samples)
                regression_contrib = (
                    -(y_k - 0.5) * self.E_v[k, ell]
                    + 2 * lam * self.E_v[k, ell] * E_C
                    + 2 * lam * E_theta_prev[:, ell] * E_v_sq_ell
                )
                
                self.b_theta[:, ell] += self.regression_weight * regression_contrib
        
        self.b_theta = np.clip(self.b_theta, 1e-6, 1e6)
        self.a_theta = np.clip(self.a_theta, 1.01, 1e6)
    
    def _update_beta(self, z: np.ndarray):
        """Update beta using coordinate ascent."""
        self.a_beta = self.alpha_beta + np.sum(z, axis=0)
        self.b_beta = self.E_eta[:, np.newaxis] + np.sum(self.E_theta, axis=0)[np.newaxis, :]
        
        self.b_beta = np.clip(self.b_beta, 1e-6, 1e6)
        self.a_beta = np.clip(self.a_beta, 1.01, 1e6)
    
    def _update_rho_beta(self, z: np.ndarray):
        """
        Update spike-and-slab indicators for beta.
        
        rho_beta[j, ell] = q(s_beta_jell = 1)
        
        Using mean-field approximation:
        log rho/(1-rho) = E[log p(z, beta | s=1)] - E[log p(z, beta | s=0)] + log(pi/(1-pi))
        """
        for j in range(self.p):
            for ell in range(self.d):
                # Log probability under slab (active)
                # E[log p(z_ijℓ | s=1)] + E[log p(beta_jℓ | s=1, eta)]
                log_prob_slab = 0.0
                
                # Contribution from Poisson likelihood
                for i in range(self.n):
                    if z[i, j, ell] > 1e-10:
                        log_prob_slab += z[i, j, ell] * self.E_log_beta_slab[j, ell]
                        log_prob_slab -= self.E_theta[i, ell] * self.E_beta_slab[j, ell]
                
                # Contribution from Gamma prior
                log_prob_slab += (self.alpha_beta - 1) * self.E_log_beta_slab[j, ell]
                log_prob_slab -= self.E_eta[j] * self.E_beta_slab[j, ell]
                
                # Log probability under spike (inactive)
                log_prob_spike = 0.0
                
                # Contribution from Poisson likelihood with spike value
                for i in range(self.n):
                    if z[i, j, ell] > 1e-10:
                        log_prob_spike += z[i, j, ell] * np.log(self.spike_value_beta + 1e-10)
                        log_prob_spike -= self.E_theta[i, ell] * self.spike_value_beta
                
                # Prior contribution
                log_odds = log_prob_slab - log_prob_spike + np.log(self.pi_beta / (1 - self.pi_beta + 1e-10))
                
                # Convert to probability using sigmoid
                self.rho_beta[j, ell] = expit(log_odds)
                
    def _update_rho_v(self, y: np.ndarray, X_aux: np.ndarray):
        """
        Update spike-and-slab indicators for v.
        
        rho_v[k, ell] = q(s_v_kell = 1)
        
        Using mean-field approximation:
        log rho/(1-rho) = E[log p(y, v | s=1)] - E[log p(y, v | s=0)] + log(pi/(1-pi))
        """
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = self._lambda_jj(self.zeta[:, k])
            
            for ell in range(self.d):
                # Log probability under slab (active): v_kell ~ N(mu, Sigma)
                log_prob_slab = 0.0
                
                # Contribution from Bernoulli likelihood (via Jaakola-Jordan bound)
                for i in range(self.n):
                    # E[A_ik] when v_kell is active
                    mask = np.ones(self.d, dtype=bool)
                    mask[ell] = False
                    E_A_active = self.E_theta[i, ell] * self.mu_v[k, ell]
                    if mask.any():
                        E_A_active += self.E_theta[i, mask] @ self.mu_v[k, mask]
                    E_A_active += X_aux[i] @ self.E_gamma[k]
                    
                    # Second moment contribution
                    E_A_sq_active = E_A_active**2
                    E_v_sq_ell = self.mu_v[k, ell]**2 + self.Sigma_v[k, ell, ell]
                    E_A_sq_active += (self.a_theta[i, ell] / self.b_theta[i, ell]**2) * E_v_sq_ell
                    
                    log_prob_slab += (y_k[i] - 0.5) * E_A_active - lam[i] * E_A_sq_active
                
                # Contribution from Normal prior
                log_prob_slab -= 0.5 * np.log(2 * np.pi * self.sigma_v**2)
                log_prob_slab -= 0.5 * (self.mu_v[k, ell]**2 + self.Sigma_v[k, ell, ell]) / self.sigma_v**2
                
                # Log probability under spike (inactive): v_kell = 0
                log_prob_spike = 0.0
                
                # Contribution from Bernoulli likelihood when v_kell = 0
                for i in range(self.n):
                    # E[A_ik] when v_kell is inactive (=0)
                    mask = np.ones(self.d, dtype=bool)
                    mask[ell] = False
                    E_A_inactive = 0.0
                    if mask.any():
                        E_A_inactive = self.E_theta[i, mask] @ self.mu_v[k, mask]
                    E_A_inactive += X_aux[i] @ self.E_gamma[k]
                    
                    E_A_sq_inactive = E_A_inactive**2
                    # No variance contribution from v_kell since it's 0
                    
                    log_prob_spike += (y_k[i] - 0.5) * E_A_inactive - lam[i] * E_A_sq_inactive
                
                # Prior contribution (spike has density at 0, represented by very small variance)
                log_prob_spike -= 0.5 * np.log(2 * np.pi * self.spike_variance_v)
                # At v=0: -0.5 * 0^2 / spike_variance_v = 0
                
                # Compute log odds
                log_odds = log_prob_slab - log_prob_spike + np.log(self.pi_v / (1 - self.pi_v + 1e-10))
                
                # Convert to probability using sigmoid
                self.rho_v[k, ell] = expit(log_odds)
    
    def _update_xi(self):
        """Update xi using coordinate ascent."""
        self.a_xi = np.full(self.n, self.alpha_xi + self.d * self.alpha_theta)
        self.b_xi = self.lambda_xi + np.sum(self.E_theta, axis=1)
        self.b_xi = np.clip(self.b_xi, 1e-6, 1e6)
    
    def _update_eta(self):
        """Update eta using coordinate ascent."""
        self.a_eta = np.full(self.p, self.alpha_eta + self.d * self.alpha_beta)
        self.b_eta = self.lambda_eta + np.sum(self.E_beta, axis=1)
        self.b_eta = np.clip(self.b_eta, 1e-6, 1e6)
    
    def _update_v(self, y: np.ndarray, X_aux: np.ndarray):
        """Update v using coordinate ascent (vectorized) with spike-and-slab."""
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = self._lambda_jj(self.zeta[:, k])
            
            # Precision matrix: (1/σ²_v)I + 2 Σ_i λ_ik diag(E[θ²_iℓ])
            # Vectorized: E[θ²] = E[θ]² + Var[θ] = E[θ]² + a/b²
            E_theta_sq = self.E_theta**2 + self.a_theta / (self.b_theta**2)
            # Weighted sum over samples: shape (d,)
            prec = (1.0 / self.sigma_v**2) * np.eye(self.d)
            prec += 2 * np.diag((lam[:, np.newaxis] * E_theta_sq).sum(axis=0))
            
            # Mean vector: vectorized computation
            # Shape: (n,) * (n, d) -> sum -> (d,)
            mean_contrib = ((y_k - 0.5)[:, np.newaxis] * self.E_theta).sum(axis=0)
            mean_contrib -= (2 * lam[:, np.newaxis] * self.E_theta * 
                           (X_aux @ self.E_gamma[k])[:, np.newaxis]).sum(axis=0)
            
            try:
                self.Sigma_v[k] = np.linalg.inv(prec)
                self.mu_v[k] = self.Sigma_v[k] @ mean_contrib
                self.mu_v[k] = np.clip(self.mu_v[k], -10, 10)
            except np.linalg.LinAlgError:
                pass
    
    def _update_gamma(self, y: np.ndarray, X_aux: np.ndarray):
        """Update gamma using coordinate ascent (vectorized)."""
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = self._lambda_jj(self.zeta[:, k])
            
            # Precision matrix: (1/σ²_γ)I + 2 Σ_i λ_ik x^aux_i (x^aux_i)^T
            # Vectorized: X_aux^T diag(2*lam) X_aux
            prec = (1.0 / self.sigma_gamma**2) * np.eye(self.p_aux)
            prec += 2 * X_aux.T @ (lam[:, np.newaxis] * X_aux)
            
            # Mean vector: vectorized
            # Shape: (p_aux, n) @ (n,) -> (p_aux,)
            mean_contrib = X_aux.T @ (y_k - 0.5)
            mean_contrib -= 2 * X_aux.T @ (lam * (self.E_theta @ self.E_v[k]))
            
            try:
                self.Sigma_gamma[k] = np.linalg.inv(prec)
                self.mu_gamma[k] = self.Sigma_gamma[k] @ mean_contrib
                self.mu_gamma[k] = np.clip(self.mu_gamma[k], -3, 3)
            except np.linalg.LinAlgError:
                pass
    
    def _update_zeta(self, y: np.ndarray, X_aux: np.ndarray):
        """Update Jaakola-Jordan auxiliary parameters (vectorized)."""
        for k in range(self.kappa):
            # E[A_ik] = E[θ_i]^T E[v_k] + x^aux_i^T E[γ_k]
            # Shape: (n,)
            E_A = self.E_theta @ self.E_v[k] + X_aux @ self.E_gamma[k]
            E_A_sq = E_A**2
            
            # Add variance terms: Σ_ℓ (E[v²_kℓ] * Var[θ_iℓ])
            # Var[θ_iℓ] = a_iℓ / b_iℓ²
            # E[v²_kℓ] = μ²_kℓ + Σ_kℓℓ
            E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])  # Shape: (d,)
            Var_theta = self.a_theta / (self.b_theta**2)  # Shape: (n, d)
            E_A_sq += (Var_theta * E_v_sq[np.newaxis, :]).sum(axis=1)
            
            self.zeta[:, k] = np.sqrt(np.maximum(E_A_sq, 1e-10))
    
    def _compute_elbo(self, X: np.ndarray, y: np.ndarray, X_aux: np.ndarray, debug: bool = False, iteration: int = 0) -> float:
        """
        Compute ELBO = E[log p(x, y, z)] - E[log q(z)]
        
        Using Poisson augmentation structure for tractable computation.
        """
        z = self._allocate_counts(X)
        elbo = 0.0
        elbo_components = {}
        
        # =====================================================================
        # E[log p(z | θ, β)] - Poisson latent variables
        # =====================================================================
        elbo_z = 0.0
        for i in range(self.n):
            for j in range(self.p):
                for ell in range(self.d):
                    if z[i, j, ell] > 1e-10:
                        # E[log Poisson(z_ijℓ | θ_iℓ β_jℓ)]
                        elbo_z += z[i, j, ell] * (self.E_log_theta[i, ell] + self.E_log_beta[j, ell])
                        elbo_z -= self.E_theta[i, ell] * self.E_beta[j, ell]
                        elbo_z -= gammaln(z[i, j, ell] + 1)
        elbo += elbo_z
        elbo_components['E[log p(z|theta,beta)]'] = elbo_z
        
        # =====================================================================
        # E[log p(y | θ, v, γ)] - Bernoulli with Jaakola-Jordan bound
        # =====================================================================
        elbo_y = 0.0
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = self._lambda_jj(self.zeta[:, k])
            
            for i in range(self.n):
                E_A = self.E_theta[i] @ self.E_v[k] + X_aux[i] @ self.E_gamma[k]
                E_A_sq = E_A**2
                # Add variance contribution
                E_A_sq += np.sum((self.mu_v[k]**2 + np.diag(self.Sigma_v[k])) * 
                                (self.a_theta[i] / self.b_theta[i]**2))
                
                # Jaakola-Jordan lower bound
                elbo_y += (y_k[i] - 0.5) * E_A - lam[i] * E_A_sq
        elbo += elbo_y
        elbo_components['E[log p(y|theta,v,gamma)]'] = elbo_y
        
        # =====================================================================
        # E[log p(θ | ξ)] and E[log p(ξ)]
        # =====================================================================
        elbo_theta_xi = 0.0
        for i in range(self.n):
            # E[log p(ξ_i)]
            elbo_theta_xi += (self.alpha_xi - 1) * self.E_log_xi[i]
            elbo_theta_xi -= self.lambda_xi * self.E_xi[i]
            elbo_theta_xi += self.alpha_xi * np.log(self.lambda_xi) - gammaln(self.alpha_xi)
            
            # E[log p(θ_i | ξ_i)]
            for ell in range(self.d):
                elbo_theta_xi += (self.alpha_theta - 1) * self.E_log_theta[i, ell]
                elbo_theta_xi += self.alpha_theta * self.E_log_xi[i]
                elbo_theta_xi -= self.E_xi[i] * self.E_theta[i, ell]
                elbo_theta_xi -= gammaln(self.alpha_theta)
        elbo += elbo_theta_xi
        elbo_components['E[log p(theta,xi)]'] = elbo_theta_xi
        
        # =====================================================================
        # E[log p(β | η)] and E[log p(η)] with spike-and-slab
        # =====================================================================
        elbo_beta_eta = 0.0
        for j in range(self.p):
            # E[log p(η_j)]
            elbo_beta_eta += (self.alpha_eta - 1) * self.E_log_eta[j]
            elbo_beta_eta -= self.lambda_eta * self.E_eta[j]
            elbo_beta_eta += self.alpha_eta * np.log(self.lambda_eta) - gammaln(self.alpha_eta)
            
            # E[log p(β_j | η_j, s_beta)] with spike-and-slab
            for ell in range(self.d):
                # Slab component (when s_beta = 1)
                slab_contrib = (self.alpha_beta - 1) * self.E_log_beta_slab[j, ell]
                slab_contrib += self.alpha_beta * self.E_log_eta[j]
                slab_contrib -= self.E_eta[j] * self.E_beta_slab[j, ell]
                slab_contrib -= gammaln(self.alpha_beta)
                
                # Weighted by probability of being in slab
                elbo_beta_eta += self.rho_beta[j, ell] * slab_contrib
                
                # Prior on indicator: E[log p(s_beta)]
                if self.rho_beta[j, ell] > 1e-10:
                    elbo_beta_eta += self.rho_beta[j, ell] * np.log(self.pi_beta + 1e-10)
                if (1 - self.rho_beta[j, ell]) > 1e-10:
                    elbo_beta_eta += (1 - self.rho_beta[j, ell]) * np.log(1 - self.pi_beta + 1e-10)
        elbo += elbo_beta_eta
        elbo_components['E[log p(beta,eta)]'] = elbo_beta_eta
        
        # =====================================================================
        # E[log p(v)] and E[log p(γ)] with spike-and-slab for v
        # =====================================================================
        elbo_v_gamma = 0.0
        for k in range(self.kappa):
            # E[log p(v_k | s_v)] with spike-and-slab
            for ell in range(self.d):
                # Slab component (when s_v = 1): N(0, σ²_v)
                slab_contrib = -0.5 * np.log(2 * np.pi * self.sigma_v**2)
                slab_contrib -= 0.5 * (self.mu_v[k, ell]**2 + self.Sigma_v[k, ell, ell]) / self.sigma_v**2
                
                # Spike component (when s_v = 0): N(0, spike_variance_v)
                spike_contrib = -0.5 * np.log(2 * np.pi * self.spike_variance_v)
                spike_contrib -= 0.5 * (self.mu_v[k, ell]**2 + self.Sigma_v[k, ell, ell]) / self.spike_variance_v
                
                # Weighted combination
                elbo_v_gamma += self.rho_v[k, ell] * slab_contrib
                elbo_v_gamma += (1 - self.rho_v[k, ell]) * spike_contrib
                
                # Prior on indicator: E[log p(s_v)]
                if self.rho_v[k, ell] > 1e-10:
                    elbo_v_gamma += self.rho_v[k, ell] * np.log(self.pi_v + 1e-10)
                if (1 - self.rho_v[k, ell]) > 1e-10:
                    elbo_v_gamma += (1 - self.rho_v[k, ell]) * np.log(1 - self.pi_v + 1e-10)
            
            # E[log p(γ_k)]
            elbo_v_gamma -= 0.5 * self.p_aux * np.log(2 * np.pi * self.sigma_gamma**2)
            elbo_v_gamma -= 0.5 * (np.sum(self.mu_gamma[k]**2) + np.trace(self.Sigma_gamma[k])) / self.sigma_gamma**2
        elbo += elbo_v_gamma
        elbo_components['E[log p(v,gamma)]'] = elbo_v_gamma
        
        # =====================================================================
        # -E[log q(z)] - Entropy terms
        # =====================================================================
        
        # Entropy of q(θ)
        entropy_theta = 0.0
        for i in range(self.n):
            for ell in range(self.d):
                entropy_theta += self.a_theta[i, ell] - np.log(self.b_theta[i, ell])
                entropy_theta += gammaln(self.a_theta[i, ell])
                entropy_theta += (1 - self.a_theta[i, ell]) * digamma(self.a_theta[i, ell])
        elbo += entropy_theta
        elbo_components['H[q(theta)]'] = entropy_theta
        
        # Entropy of q(β)
        entropy_beta = 0.0
        for j in range(self.p):
            for ell in range(self.d):
                entropy_beta += self.a_beta[j, ell] - np.log(self.b_beta[j, ell])
                entropy_beta += gammaln(self.a_beta[j, ell])
                entropy_beta += (1 - self.a_beta[j, ell]) * digamma(self.a_beta[j, ell])
        elbo += entropy_beta
        elbo_components['H[q(beta)]'] = entropy_beta
        
        # Entropy of q(ξ)
        entropy_xi = 0.0
        for i in range(self.n):
            entropy_xi += self.a_xi[i] - np.log(self.b_xi[i])
            entropy_xi += gammaln(self.a_xi[i])
            entropy_xi += (1 - self.a_xi[i]) * digamma(self.a_xi[i])
        elbo += entropy_xi
        elbo_components['H[q(xi)]'] = entropy_xi
        
        # Entropy of q(η)
        entropy_eta = 0.0
        for j in range(self.p):
            entropy_eta += self.a_eta[j] - np.log(self.b_eta[j])
            entropy_eta += gammaln(self.a_eta[j])
            entropy_eta += (1 - self.a_eta[j]) * digamma(self.a_eta[j])
        elbo += entropy_eta
        elbo_components['H[q(eta)]'] = entropy_eta
        
        # Entropy of q(v)
        entropy_v = 0.0
        for k in range(self.kappa):
            sign, logdet = np.linalg.slogdet(self.Sigma_v[k])
            if sign > 0:
                entropy_v += 0.5 * (self.d * (1 + np.log(2 * np.pi)) + logdet)
        elbo += entropy_v
        elbo_components['H[q(v)]'] = entropy_v
        
        # Entropy of q(γ)
        entropy_gamma = 0.0
        for k in range(self.kappa):
            sign, logdet = np.linalg.slogdet(self.Sigma_gamma[k])
            if sign > 0:
                entropy_gamma += 0.5 * (self.p_aux * (1 + np.log(2 * np.pi)) + logdet)
        elbo += entropy_gamma
        elbo_components['H[q(gamma)]'] = entropy_gamma
        
        # Entropy of q(s_beta) - Bernoulli entropy
        entropy_s_beta = 0.0
        for j in range(self.p):
            for ell in range(self.d):
                rho = self.rho_beta[j, ell]
                if rho > 1e-10 and rho < 1 - 1e-10:
                    entropy_s_beta -= rho * np.log(rho) + (1 - rho) * np.log(1 - rho)
        elbo += entropy_s_beta
        elbo_components['H[q(s_beta)]'] = entropy_s_beta
        
        # Entropy of q(s_v) - Bernoulli entropy
        entropy_s_v = 0.0
        for k in range(self.kappa):
            for ell in range(self.d):
                rho = self.rho_v[k, ell]
                if rho > 1e-10 and rho < 1 - 1e-10:
                    entropy_s_v -= rho * np.log(rho) + (1 - rho) * np.log(1 - rho)
        elbo += entropy_s_v
        elbo_components['H[q(s_v)]'] = entropy_s_v
        
        if debug:
            print(f"\n  === Iteration {iteration} ELBO Breakdown ===")
            for name, value in elbo_components.items():
                status = "✓" if np.isfinite(value) else "✗"
                print(f"    {status} {name:30s}: {value:12.2f}")
            print(f"    {'='*44}")
            print(f"    {'Total ELBO':30s}: {elbo:12.2f}")
        
        # Store components for tracking
        self.last_elbo_components_ = elbo_components
        
        if not np.isfinite(elbo):
            return -np.inf
        
        return elbo
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_aux: np.ndarray,
        max_iter: int = 100,
        tol: float = 10.0,  # Absolute tolerance: stop if ELBO change < 10.0
        rel_tol: float = 2e-4,  # Relative tolerance: 0.02% change
        elbo_freq: int = 5,  # Compute ELBO every N iterations (default 5 for speed)
        min_iter: int = 30,  # Minimum iterations before checking convergence
        patience: int = 3,  # Stop if no significant improvement for this many ELBO checks
        verbose: bool = True,
        # Damping parameters
        theta_damping: float = 0.5,  # Heavy damping for theta (most unstable)
        beta_damping: float = 0.7,   # Moderate damping for beta
        v_damping: float = 0.6,      # Moderate damping for v
        gamma_damping: float = 0.6,  # Moderate damping for gamma
        xi_damping: float = 0.8,     # Light damping for xi
        eta_damping: float = 0.8,    # Light damping for eta
        adaptive_damping: bool = True,  # Adjust damping based on ELBO
        debug: bool = False
    ):
        """
        Fit the model using coordinate ascent VI with damping.
        
        Parameters:
        -----------
        tol : float (default=10.0)
            Absolute tolerance for ELBO convergence. Stop when |ELBO_change| < tol.
        rel_tol : float (default=2e-4)
            Relative tolerance for ELBO convergence. Stop when |ELBO_change/ELBO| < rel_tol (0.02%).
        elbo_freq : int (default=5)
            Compute ELBO every N iterations. Set to 5-10 for speed on large datasets.
        min_iter : int (default=30)
            Minimum iterations before checking convergence criteria.
        patience : int (default=3)
            Number of consecutive ELBO checks with small improvements before stopping.
            This prevents premature stopping while avoiding endless tiny improvements.
        
        Damping parameters control how much of the new update to accept:
        - damping = 1.0: Full update (no damping)
        - damping = 0.5: Accept 50% of new update, keep 50% of old value
        - damping = 0.0: No update (keep old value)
        
        Lower damping = more stable but slower convergence
        """
        import time


        start_time = time.time()
        if y.ndim == 1:
            y = y[:, np.newaxis]
        
        self._initialize_parameters(X, y, X_aux)
        elbo_history = []
        patience_counter = 0  # Track consecutive small improvements
        
        # Track damping factors (for adaptive damping)
        damping_factors = {
            'theta': theta_damping,
            'beta': beta_damping,
            'v': v_damping,
            'gamma': gamma_damping,
            'xi': xi_damping,
            'eta': eta_damping
        }
        
        for iteration in range(max_iter):
            # ============================================
            # Compute expectations
            # ============================================
            self._compute_expectations()
            
            # ============================================
            # Allocate counts
            # ============================================
            z = self._allocate_counts(X)
            
            # ============================================
            # Update theta with damping
            # ============================================
            a_theta_old = self.a_theta.copy()
            b_theta_old = self.b_theta.copy()
            
            self._update_theta(z, y, X_aux)
            
            # Apply damping
            self.a_theta = (damping_factors['theta'] * self.a_theta + 
                        (1 - damping_factors['theta']) * a_theta_old)
            self.b_theta = (damping_factors['theta'] * self.b_theta + 
                        (1 - damping_factors['theta']) * b_theta_old)
            
            # ============================================
            # Update beta with damping
            # ============================================
            a_beta_old = self.a_beta.copy()
            b_beta_old = self.b_beta.copy()
            
            self._update_beta(z)
            
            # Apply damping
            self.a_beta = (damping_factors['beta'] * self.a_beta + 
                        (1 - damping_factors['beta']) * a_beta_old)
            self.b_beta = (damping_factors['beta'] * self.b_beta + 
                        (1 - damping_factors['beta']) * b_beta_old)
            
            # Update spike-and-slab indicators for beta (no damping)
            self._update_rho_beta(z)
            
            # ============================================
            # Update xi with damping
            # ============================================
            a_xi_old = self.a_xi.copy()
            b_xi_old = self.b_xi.copy()
            
            self._update_xi()
            
            # Apply damping
            self.a_xi = (damping_factors['xi'] * self.a_xi + 
                        (1 - damping_factors['xi']) * a_xi_old)
            self.b_xi = (damping_factors['xi'] * self.b_xi + 
                        (1 - damping_factors['xi']) * b_xi_old)
            
            # ============================================
            # Update eta with damping
            # ============================================
            a_eta_old = self.a_eta.copy()
            b_eta_old = self.b_eta.copy()
            
            self._update_eta()
            
            # Apply damping
            self.a_eta = (damping_factors['eta'] * self.a_eta + 
                        (1 - damping_factors['eta']) * a_eta_old)
            self.b_eta = (damping_factors['eta'] * self.b_eta + 
                        (1 - damping_factors['eta']) * b_eta_old)
            
            # ============================================
            # Update v with damping
            # ============================================
            mu_v_old = self.mu_v.copy()
            Sigma_v_old = self.Sigma_v.copy()
            
            self._update_v(y, X_aux)
            
            # Apply damping
            self.mu_v = (damping_factors['v'] * self.mu_v + 
                        (1 - damping_factors['v']) * mu_v_old)
            self.Sigma_v = (damping_factors['v'] * self.Sigma_v + 
                        (1 - damping_factors['v']) * Sigma_v_old)
            
            # Update spike-and-slab indicators for v (no damping)
            self._update_rho_v(y, X_aux)
            
            # ============================================
            # Update gamma with damping
            # ============================================
            mu_gamma_old = self.mu_gamma.copy()
            Sigma_gamma_old = self.Sigma_gamma.copy()
            
            self._update_gamma(y, X_aux)
            
            # Apply damping
            self.mu_gamma = (damping_factors['gamma'] * self.mu_gamma + 
                            (1 - damping_factors['gamma']) * mu_gamma_old)
            self.Sigma_gamma = (damping_factors['gamma'] * self.Sigma_gamma + 
                            (1 - damping_factors['gamma']) * Sigma_gamma_old)
            
            # ============================================
            # Update zeta (no damping - auxiliary variable)
            # ============================================
            self._update_zeta(y, X_aux)
            
            # ============================================
            # Compute ELBO (less frequently for speed)
            # ============================================
            compute_elbo = (iteration % elbo_freq == 0 or iteration == 0 or iteration == max_iter - 1)
            
            if compute_elbo:
                elbo = self._compute_elbo(X, y, X_aux, debug=debug, iteration=iteration+1)
                elbo_history.append((iteration, elbo))
            
            # ============================================
            # Adaptive damping adjustment
            # ============================================
            if adaptive_damping and len(elbo_history) > 1:
                _, elbo_curr = elbo_history[-1]
                _, elbo_prev = elbo_history[-2]
                elbo_change = elbo_curr - elbo_prev
                
                if elbo_change > 0:
                    # ELBO improved: trust updates more (increase damping toward 1.0)
                    for key in damping_factors:
                        damping_factors[key] = min(damping_factors[key] * 1.05, 1.0)
                else:
                    # ELBO decreased: be more conservative (decrease damping)
                    for key in damping_factors:
                        damping_factors[key] = max(damping_factors[key] * 0.9, 0.1)
            
            # ============================================
            # Debug output
            # ============================================
            if debug and (iteration % 1 == 0):
                print(f"\n{'='*60}")
                print(f"=== Iteration {iteration + 1}/{max_iter} ===")
                print(f"{'='*60}")
                print(f"Parameter ranges:")
                print(f"  E[theta]: [{self.E_theta.min():.4f}, {self.E_theta.max():.4f}]")
                print(f"  E[beta]:  [{self.E_beta.min():.4f}, {self.E_beta.max():.4f}]")
                print(f"  E[v]:     [{self.E_v.min():.4f}, {self.E_v.max():.4f}]")
                if self.E_gamma.size > 0:  # Check if E_gamma is not empty
                    if self.E_gamma.size > 0:
                        print(f"  E[gamma]: [{self.E_gamma.min():.4f}, {self.E_gamma.max():.4f}]")
            else:
                print(f"  E[gamma]: (empty array)")
                print(f"  a_theta:  [{self.a_theta.min():.4f}, {self.a_theta.max():.4f}]")
                print(f"  b_theta:  [{self.b_theta.min():.4f}, {self.b_theta.max():.4f}]")
                if adaptive_damping:
                    print(f"Damping factors:")
                    print(f"  theta={damping_factors['theta']:.3f}, "
                          f"v={damping_factors['v']:.3f}, gamma={damping_factors['gamma']:.3f}")
            
            # ============================================
            # Verbose output
            # ============================================
            if verbose and compute_elbo:
                _, elbo_curr = elbo_history[-1]
                iter_time = time.time() - start_time
                print(f"\nIteration {iteration + 1}/{max_iter}, ELBO: {elbo_curr:.2f} (time: {iter_time:.2f}s)")
                
                if len(elbo_history) > 1:
                    _, elbo_prev = elbo_history[-2]
                    elbo_change = elbo_curr - elbo_prev
                    change_symbol = "↑" if elbo_change > 0 else "↓"
                    rel_change = abs(elbo_change / (abs(elbo_prev) + 1e-10))
                    print(f"  ELBO change: {change_symbol} {elbo_change:.2f} (relative: {rel_change:.6f})")
                    
                    if elbo_change < 0:
                        print(f"  ⚠ WARNING: ELBO decreased by {abs(elbo_change):.2f}")
            
            # ============================================
            # Check convergence with patience mechanism
            # Only check after minimum iterations to avoid premature stopping
            # ============================================
            if iteration >= min_iter and len(elbo_history) > 1 and compute_elbo:
                _, elbo_curr = elbo_history[-1]
                _, elbo_prev = elbo_history[-2]
                elbo_change = elbo_curr - elbo_prev
                rel_change = abs(elbo_change / (abs(elbo_prev) + 1e-10))
                
                # Require BOTH absolute AND relative convergence
                abs_converged = abs(elbo_change) < tol
                rel_converged = rel_change < rel_tol
                
                if abs_converged and rel_converged:
                    patience_counter += 1
                    if verbose and debug:
                        print(f"  Patience: {patience_counter}/{patience} (small improvement)")
                else:
                    patience_counter = 0  # Reset if we see significant improvement
                
                # Stop if we've had small improvements for 'patience' consecutive checks
                if patience_counter >= patience:
                    if verbose:
                        total_time = time.time() - start_time
                        print(f"\n{'='*60}")
                        print(f"✓ Converged after {iteration + 1} iterations ({total_time:.2f}s)")
                        print(f"  Absolute change: {abs(elbo_change):.6f} < {tol}")
                        print(f"  Relative change: {rel_change:.6f} < {rel_tol}")
                        print(f"  Patience exhausted: {patience_counter}/{patience}")
                        print(f"{'='*60}")
                    break
        
        # Store results
        self.elbo_history_ = elbo_history
        self.training_time_ = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {self.training_time_:.2f}s")
            print(f"Final ELBO: {elbo_history[-1][1]:.2f}")
        
        return self
    
    def infer_theta(
        self,
        X: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infer theta (sample-specific factors) for new samples using learned global parameters.
        
        This is used for validation/test sets where we need to infer local parameters
        given the learned global parameters (beta, upsilon).
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_new, p)
            Gene expression count matrix for new samples
        max_iter : int
            Maximum iterations for inference
        tol : float
            Convergence tolerance for theta updates
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        E_theta_new : np.ndarray, shape (n_new, d)
            Expected values of theta for new samples
        a_theta_new : np.ndarray, shape (n_new, d)
            Shape parameters of theta posterior (for uncertainty quantification)
        b_theta_new : np.ndarray, shape (n_new, d)
            Rate parameters of theta posterior (for uncertainty quantification)
        """
        n_new = X.shape[0]
        
        if not hasattr(self, 'E_beta'):
            raise RuntimeError("Model must be fitted before inferring theta for new samples")
        
        # Initialize theta parameters for new samples
        # Use mean of training theta as starting point
        a_theta_new = np.tile(self.a_theta.mean(axis=0), (n_new, 1))
        b_theta_new = np.tile(self.b_theta.mean(axis=0), (n_new, 1))
        
        # Use mean of training xi for rate parameter initialization
        E_xi_mean = self.E_xi.mean()
        
        if verbose:
            print(f"Inferring theta for {n_new} new samples...")
        
        # Coordinate ascent to infer theta
        for iteration in range(max_iter):
            # Store old values for convergence check
            a_theta_old = a_theta_new.copy()
            
            # Compute expectations
            E_theta_new = a_theta_new / b_theta_new
            E_log_theta_new = digamma(a_theta_new) - np.log(b_theta_new)
            
            # Allocate counts using expected log parameters
            # log φ_ijℓ = E[log θ_iℓ] + E[log β_jℓ]
            log_phi = E_log_theta_new[:, np.newaxis, :] + self.E_log_beta[np.newaxis, :, :]
            log_phi_normalized = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
            phi = np.exp(log_phi_normalized)
            
            # Expected latent counts: z_ijℓ = x_ij * φ_ijℓ
            z_new = X[:, :, np.newaxis] * phi
            
            # Update theta parameters
            # Shape: α_θ + Σ_j z_ijℓ
            a_theta_new = self.alpha_theta + np.sum(z_new, axis=1)
            
            # Rate: E[ξ] + Σ_j E[β_jℓ]
            b_theta_new = E_xi_mean + np.sum(self.E_beta, axis=0)[np.newaxis, :]
            b_theta_new = np.maximum(b_theta_new, 1e-10)
            
            # Check convergence
            max_change = np.max(np.abs(a_theta_new - a_theta_old))
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration + 1}/{max_iter}, max_change: {max_change:.6f}")
            
            if max_change < tol:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
        
        # Final expectations
        E_theta_new = a_theta_new / b_theta_new
        
        return E_theta_new, a_theta_new, b_theta_new
    
    def predict_proba(
        self, 
        X: np.ndarray, 
        X_aux: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Predict probabilities for new samples.
        
        Uses the learned global parameters (beta, upsilon) to infer sample-specific
        factors (theta) and compute classification probabilities.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_new, p)
            Gene expression count matrix for new samples
        X_aux : np.ndarray, shape (n_new, p_aux)
            Auxiliary features for new samples
        max_iter : int
            Maximum iterations for theta inference
        tol : float
            Convergence tolerance for theta updates
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        probs : np.ndarray, shape (n_new, kappa)
            Predicted probabilities for each class
        """
        # Infer theta for new samples using learned global parameters
        E_theta_new, _, _ = self.infer_theta(X, max_iter=max_iter, tol=tol, verbose=verbose)
        
        # Compute linear predictions and probabilities
        # A_ik = θ_i^T v_k + x_i^aux^T γ_k
        linear_pred = E_theta_new @ self.E_v.T + X_aux @ self.E_gamma.T
        probs = expit(linear_pred)
        
        return probs

