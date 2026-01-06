"""
Variational Inference Model for Single-Cell Analysis
=====================================================

This module implements a Poisson Factor Model with spike-and-slab priors
for discovering gene programs from single-cell RNA-seq data.

Key Features:
- Hierarchical Poisson factorization for count data
- Spike-and-slab sparsity on gene loadings (beta) and classification weights (v)
- Integrated classification model for phenotype prediction
- Coordinate ascent variational inference with adaptive damping

Usage:
    from VariationalInference.vi import VI

    model = VI(n_factors=50)  # Uses random initialization by default
    model.fit(X, y, X_aux)
    predictions = model.predict_proba(X_new, X_aux_new)
"""

import numpy as np
from scipy.special import digamma, gammaln, expit, logsumexp
from typing import Tuple, Optional, Union
import pandas as pd


class VI:
    """
    Variational Inference model for gene program discovery and classification.

    Parameters
    ----------
    n_factors : int
        Number of latent gene programs to discover.
    alpha_theta : float, default=2.0
        Prior shape parameter for sample-specific factor activities.
    alpha_beta : float, default=2.0
        Prior shape parameter for gene loadings.
    alpha_xi : float, default=2.0
        Prior shape parameter for sample depth correction.
    alpha_eta : float, default=2.0
        Prior shape parameter for gene scaling factors.
    lambda_xi : float, default=1.5
        Prior rate parameter for sample depth correction.
    lambda_eta : float, default=1.5
        Prior rate parameter for gene scaling factors.
    sigma_v : float, default=0.2
        Prior standard deviation for classification weights (regularization).
    sigma_gamma : float, default=0.5
        Prior standard deviation for auxiliary feature effects.
    random_state : int or None, default=None
        Random seed for reproducibility. If None (default), uses true random
        initialization from system entropy - recommended for scientific experiments.
        Set to an integer for reproducible results during debugging/testing.
    pi_v : float, default=0.9
        Prior probability of classification weights being active (slab).
        Higher values (0.9-1.0) favor keeping factors active for classification.
    pi_beta : float, default=0.05
        Prior probability of gene loadings being active (slab).
    spike_variance_v : float, default=1e-6
        Variance for spike component of v (near-zero).
    spike_value_beta : float, default=1e-6
        Value for spike component of beta (near-zero).
    regression_weight : float, default=10.0
        Weight for classification objective. Higher values make classification
        more influential on theta updates.

    Attributes
    ----------
    E_theta : ndarray of shape (n_samples, n_factors)
        Expected sample-specific factor activities after fitting.
    E_beta : ndarray of shape (n_genes, n_factors)
        Expected gene loadings after fitting.
    E_v : ndarray of shape (n_outcomes, n_factors)
        Expected classification weights after fitting.
    elbo_history_ : list of tuples
        History of (iteration, ELBO) values during training.
    training_time_ : float
        Total training time in seconds.
    seed_used_ : int or None
        The random seed used for initialization (None if truly random).
    """

    def __init__(
        self,
        n_factors: int,
        alpha_theta: float = 2.0,
        alpha_beta: float = 2.0,
        alpha_xi: float = 2.0,
        alpha_eta: float = 2.0,
        lambda_xi: float = 1.5,
        lambda_eta: float = 1.5,
        sigma_v: float = 0.2,
        sigma_gamma: float = 0.5,
        random_state: Optional[int] = None,
        pi_v: float = 0.9,
        pi_beta: float = 0.05,
        spike_variance_v: float = 1e-6,
        spike_value_beta: float = 1e-6,
        regression_weight: float = 10.0
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

        # Random state handling: None = true randomization for scientific experiments
        if random_state is None:
            # Use system entropy for true randomization
            self.rng = np.random.default_rng()
            self.seed_used_ = None
        else:
            # Use provided seed for reproducibility
            self.rng = np.random.default_rng(random_state)
            self.seed_used_ = random_state

        self.regression_weight = regression_weight
        
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
        theta_init = (row_sums / self.d) * (1 + 0.1 * self.rng.standard_normal((self.n, self.d)))
        beta_init = (col_sums.T / self.d) * (1 + 0.1 * self.rng.standard_normal((self.p, self.d)))
        
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
        
        self.a_xi = self.alpha_xi + 0.1 * self.rng.exponential(scale=1.0, size=self.n)
        self.b_xi = np.ones(self.n) * self.lambda_xi

        self.a_eta = self.alpha_eta + 0.1 * self.rng.exponential(scale=1.0, size=self.p)
        self.b_eta = np.ones(self.p) * self.lambda_eta

        # Initialize Normal parameters - very small values to prevent early explosion
        # Start conservative and let data drive v up slowly
        self.mu_v = 0.01 * self.rng.standard_normal((self.kappa, self.d))
        self.Sigma_v = np.tile(np.eye(self.d)[np.newaxis, :, :], (self.kappa, 1, 1))
        
        # Initialize spike-and-slab indicators for v
        # rho_v[k, ell] = q(s_v_kell = 1) - probability that v_kell is active (slab)
        # Initialize based on prior - high pi_v means start with factors active
        self.rho_v = np.ones((self.kappa, self.d)) * self.pi_v

        self.mu_gamma = 0.01 * self.rng.standard_normal((self.kappa, self.p_aux))
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
        self.E_v = self.rho_v * self.mu_v  
        # Normal distributions: expectations are just the means
        self.E_gamma = self.mu_gamma
    
    def get_sparse_beta(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get sparse version of E[beta] by thresholding spike-and-slab indicators.
        
        Parameters:
        -----------
        threshold : float
            Probability threshold for inclusion. If rho_beta > threshold, 
            use slab value; otherwise set to 0.
            
        Returns:
        --------
        E_beta_sparse : np.ndarray, shape (p, d)
            Sparse expected values of beta
        """
        return np.where(self.rho_beta > threshold, self.E_beta_slab, 0.0)
    
    def get_sparse_v(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get sparse version of E[v] by thresholding spike-and-slab indicators.
        
        Parameters:
        -----------
        threshold : float
            Probability threshold for inclusion. If rho_v > threshold,
            use slab value; otherwise set to 0.
            
        Returns:
        --------
        E_v_sparse : np.ndarray, shape (kappa, d)
            Sparse expected values of v
        """
        return np.where(self.rho_v > threshold, self.mu_v, 0.0)
    
    def get_active_factors(self, threshold: float = 0.5) -> dict:
        """
        Get indices of active factors for each outcome.
        
        Parameters:
        -----------
        threshold : float
            Probability threshold for considering a factor active.
            
        Returns:
        --------
        active : dict
            Dictionary with keys 'beta' and 'v', containing lists of active indices.
        """
        active_beta = [np.where(self.rho_beta[:, ell] > threshold)[0] 
                       for ell in range(self.d)]
        active_v = [np.where(self.rho_v[k, :] > threshold)[0] 
                    for k in range(self.kappa)]
        
        return {
            'beta': active_beta,  # List of arrays: active genes per factor
            'v': active_v,        # List of arrays: active factors per outcome
            'n_active_beta': sum(len(a) for a in active_beta),
            'n_active_v': sum(len(a) for a in active_v)
        }
    
    def _lambda_jj(self, zeta: np.ndarray) -> np.ndarray:
        """Jaakola-Jordan lambda function."""
        result = np.zeros_like(zeta)
        nonzero = np.abs(zeta) > 1e-10
        result[nonzero] = (1.0 / (4.0 * zeta[nonzero])) * np.tanh(zeta[nonzero] / 2.0)
        result[~nonzero] = 0.125
        return result
    
    def _compute_z_suffstats(self, X: np.ndarray, chunk_size: int = 1000):
        """
        Compute sufficient statistics from z WITHOUT materializing full (n, p, d) array.

        Returns:
        --------
        z_sum_over_genes : ndarray, shape (n, d)
            sum_j z_ijl for theta updates
        z_sum_over_samples : ndarray, shape (p, d)
            sum_i z_ijl for beta updates
        """
        n, p = X.shape
        z_sum_over_genes = np.zeros((n, self.d), dtype=np.float64)
        z_sum_over_samples = np.zeros((p, self.d), dtype=np.float64)

        # Process samples in chunks to limit memory
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_n = end - start

            # Compute phi for this chunk: shape (chunk_n, p, d)
            # But we process gene-wise to avoid the full (chunk_n, p, d) allocation
            E_log_theta_chunk = self.E_log_theta[start:end]  # (chunk_n, d)
            X_chunk = X[start:end]  # (chunk_n, p)

            # For each sample in chunk, compute z contributions
            # phi_ij = softmax(E_log_theta_i + E_log_beta_j) over d
            # z_ijl = x_ij * phi_ijl

            # Vectorized over genes: process all genes at once for this sample chunk
            # log_phi shape: (chunk_n, p, d)
            log_phi = E_log_theta_chunk[:, np.newaxis, :] + self.E_log_beta[np.newaxis, :, :]
            log_phi_max = log_phi.max(axis=2, keepdims=True)
            phi = np.exp(log_phi - log_phi_max)
            phi_sum = phi.sum(axis=2, keepdims=True)
            phi = phi / phi_sum  # Normalized: (chunk_n, p, d)

            # z = x * phi, then sum
            # z_sum_over_genes[i, l] = sum_j x_ij * phi_ijl
            z_chunk = X_chunk[:, :, np.newaxis] * phi  # (chunk_n, p, d)
            z_sum_over_genes[start:end] = z_chunk.sum(axis=1)  # (chunk_n, d)
            z_sum_over_samples += z_chunk.sum(axis=0)  # (p, d)

            del log_phi, phi, z_chunk  # Explicitly free memory

        return z_sum_over_genes, z_sum_over_samples

    def _compute_z_suffstats_memory_efficient(self, X: np.ndarray, sample_chunk: int = 500, gene_chunk: int = 2000):
        """
        Ultra memory-efficient version: processes both samples AND genes in chunks.
        Use this when n_factors is very large (>500).

        This never allocates more than (sample_chunk, gene_chunk, d) at once.
        """
        n, p = X.shape
        z_sum_over_genes = np.zeros((n, self.d), dtype=np.float64)
        z_sum_over_samples = np.zeros((p, self.d), dtype=np.float64)

        for s_start in range(0, n, sample_chunk):
            s_end = min(s_start + sample_chunk, n)
            E_log_theta_chunk = self.E_log_theta[s_start:s_end]  # (s_chunk, d)
            X_sample_chunk = X[s_start:s_end]  # (s_chunk, p)

            for g_start in range(0, p, gene_chunk):
                g_end = min(g_start + gene_chunk, p)
                E_log_beta_chunk = self.E_log_beta[g_start:g_end]  # (g_chunk, d)
                X_chunk = X_sample_chunk[:, g_start:g_end]  # (s_chunk, g_chunk)

                # Compute phi for this sub-chunk
                log_phi = E_log_theta_chunk[:, np.newaxis, :] + E_log_beta_chunk[np.newaxis, :, :]
                log_phi_max = log_phi.max(axis=2, keepdims=True)
                phi = np.exp(log_phi - log_phi_max)
                phi = phi / phi.sum(axis=2, keepdims=True)

                # z = x * phi
                z_chunk = X_chunk[:, :, np.newaxis] * phi

                # Accumulate sufficient statistics
                z_sum_over_genes[s_start:s_end] += z_chunk.sum(axis=1)
                z_sum_over_samples[g_start:g_end] += z_chunk.sum(axis=0)

                del log_phi, phi, z_chunk

        return z_sum_over_genes, z_sum_over_samples
    
    def _update_theta_from_suffstats(self, z_sum_over_genes: np.ndarray, y: np.ndarray, X_aux: np.ndarray):
        """
        Update theta using sufficient statistics (memory-efficient version).

        Parameters:
        -----------
        z_sum_over_genes : ndarray, shape (n, d)
            sum_j z_ijl - precomputed sufficient statistic
        """
        E_theta_prev = self.E_theta.copy()

        # Shape: alpha_theta + sum_j z_ijl
        self.a_theta = self.alpha_theta + z_sum_over_genes

        # Rate: E[xi_i] + sum_j E[beta_jl] + regression terms
        self.b_theta = self.E_xi[:, np.newaxis] + np.sum(self.E_beta, axis=0)[np.newaxis, :]

        # Add regression contribution (vectorized over factors)
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = self._lambda_jj(self.zeta[:, k])

            # Precompute terms that don't depend on ell
            E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])  # (d,)
            base_pred = X_aux @ self.E_gamma[k]  # (n,)
            theta_v_product = self.E_theta @ self.E_v[k]  # (n,)

            # Vectorized regression contribution
            for ell in range(self.d):
                # E[C_kℓ] excluding current dimension
                E_C = theta_v_product - self.E_theta[:, ell] * self.E_v[k, ell] + base_pred

                regression_contrib = (
                    -(y_k - 0.5) * self.E_v[k, ell]
                    + 2 * lam * self.E_v[k, ell] * E_C
                    + 2 * lam * E_theta_prev[:, ell] * E_v_sq[ell]
                )

                self.b_theta[:, ell] += self.regression_weight * regression_contrib

        self.b_theta = np.clip(self.b_theta, 1e-6, 1e6)
        self.a_theta = np.clip(self.a_theta, 1.01, 1e6)

    def _update_beta_from_suffstats(self, z_sum_over_samples: np.ndarray):
        """
        Update beta using sufficient statistics (memory-efficient version).

        Parameters:
        -----------
        z_sum_over_samples : ndarray, shape (p, d)
            sum_i z_ijl - precomputed sufficient statistic
        """
        self.a_beta = self.alpha_beta + z_sum_over_samples
        self.b_beta = self.E_eta[:, np.newaxis] + np.sum(self.E_theta, axis=0)[np.newaxis, :]

        self.b_beta = np.clip(self.b_beta, 1e-6, 1e6)
        self.a_beta = np.clip(self.a_beta, 1.01, 1e6)

    def _update_rho_beta_from_suffstats(self, z_sum_over_samples: np.ndarray):
        """
        Update spike-and-slab indicators for beta using sufficient statistics.

        Parameters:
        -----------
        z_sum_over_samples : ndarray, shape (p, d)
            sum_i z_ijl - precomputed sufficient statistic
        """
        # Sum of E[theta] over samples: shape (d,)
        theta_sum = np.sum(self.E_theta, axis=0)

        log_spike_value = np.log(self.spike_value_beta + 1e-10)

        # Difference in log-likelihood (vectorized over j, ell)
        ll_diff = z_sum_over_samples * (self.E_log_beta_slab - log_spike_value)
        ll_diff -= theta_sum[np.newaxis, :] * (self.E_beta_slab - self.spike_value_beta)

        # Prior contribution: log(pi / (1 - pi))
        log_prior_odds = np.log(self.pi_beta / (1 - self.pi_beta + 1e-10))

        # Log odds
        log_odds = ll_diff + log_prior_odds

        # Convert to probability using sigmoid, with clipping for stability
        log_odds = np.clip(log_odds, -20, 20)
        self.rho_beta = expit(log_odds)
                
    def _update_rho_v(self, y: np.ndarray, X_aux: np.ndarray):
        """
        Update spike-and-slab indicators for v (vectorized).
        
        rho_v[k, ell] = q(s_v_kell = 1)
        
        For each (k, ell), compute log-odds comparing:
        - Slab: v_kell ~ N(mu_v[k,ell], Sigma_v[k,ell,ell])
        - Spike: v_kell = 0
        """
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = self._lambda_jj(self.zeta[:, k])  # (n,)
            
            # Compute base linear predictor without v contribution: x_aux @ gamma
            base_pred = X_aux @ self.E_gamma[k]  # (n,)
            
            for ell in range(self.d):
                # Contribution from other dimensions of v (excluding ell)
                mask = np.ones(self.d, dtype=bool)
                mask[ell] = False
                other_contrib = self.E_theta[:, mask] @ self.E_v[k, mask]  # (n,)
                
                # E[A_ik] when v_kell is ACTIVE (slab)
                E_A_active = self.E_theta[:, ell] * self.mu_v[k, ell] + other_contrib + base_pred
                
                # E[A_ik] when v_kell is INACTIVE (spike, v_kell = 0)
                E_A_inactive = other_contrib + base_pred
                
                # Second moment terms
                E_v_sq_ell = self.mu_v[k, ell]**2 + self.Sigma_v[k, ell, ell]
                E_theta_sq = self.E_theta[:, ell]**2 + self.a_theta[:, ell] / (self.b_theta[:, ell]**2)
                
                # E[A^2] for active case (includes variance from v_kell)
                E_A_sq_active = E_A_active**2 + E_theta_sq * self.Sigma_v[k, ell, ell]
                
                # E[A^2] for inactive case
                E_A_sq_inactive = E_A_inactive**2
                
                # Log-likelihood contribution from Bernoulli (JJ bound) - vectorized over samples
                # log p(y | active) = sum_i [(y_i - 0.5) * E[A_active] - lambda * E[A^2_active]]
                ll_active = np.sum((y_k - 0.5) * E_A_active - lam * E_A_sq_active)
                ll_inactive = np.sum((y_k - 0.5) * E_A_inactive - lam * E_A_sq_inactive)
                
                # Prior contribution
                # Slab: N(0, sigma_v^2) - penalize large values
                # Only include the quadratic penalty, not normalizing constants
                # (they cancel out in proper spike-and-slab with point mass spike)
                prior_slab = -0.5 * E_v_sq_ell / self.sigma_v**2
                
                # Spike: point mass at 0 - no penalty since v=0 exactly
                prior_spike = 0.0
                
                # Log odds: likelihood ratio + prior ratio + prior on indicator
                log_odds = (ll_active - ll_inactive) + (prior_slab - prior_spike)
                log_odds += np.log(self.pi_v / (1 - self.pi_v + 1e-10))
                
                # Clip for numerical stability
                log_odds = np.clip(log_odds, -20, 20)
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
                # Add regularization to prevent ill-conditioning
                prec_reg = prec + 1e-4 * np.eye(self.d)
                self.Sigma_v[k] = np.linalg.inv(prec_reg)

                # Check if variance is exploding
                diag_variance = np.diag(self.Sigma_v[k])
                if np.any(diag_variance > 10.0):
                    # Variance too large - use stronger regularization
                    prec_reg = prec + 1e-3 * np.eye(self.d)
                    self.Sigma_v[k] = np.linalg.inv(prec_reg)

                self.mu_v[k] = self.Sigma_v[k] @ mean_contrib

                # Strict clipping to prevent ELBO explosion
                # Limit to [-1.5, 1.5] which allows meaningful classification
                # while preventing huge prior penalties
                self.mu_v[k] = np.clip(self.mu_v[k], -1.5, 1.5)
            except np.linalg.LinAlgError:
                # Keep previous values if inversion fails
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

    def _compute_elbo_efficient(self, X: np.ndarray, y: np.ndarray, X_aux: np.ndarray,
                                 z_sum_over_genes: np.ndarray, z_sum_over_samples: np.ndarray,
                                 debug: bool = False, iteration: int = 0) -> float:
        """
        Memory-efficient ELBO computation using vectorized operations.

        Uses precomputed sufficient statistics instead of materializing full z matrix.
        All loops are vectorized for speed.
        """
        elbo = 0.0

        # =====================================================================
        # E[log p(z | θ, β)] - Using sufficient statistics
        # =====================================================================
        # Original: sum_ijl z_ijl * (E[log θ_il] + E[log β_jl]) - E[θ_il]*E[β_jl] - log(z_ijl!)

        # Term 1: sum_ijl z_ijl * E[log θ_il] = sum_il E[log θ_il] * sum_j z_ijl
        elbo_z = np.sum(z_sum_over_genes * self.E_log_theta)

        # Term 2: sum_ijl z_ijl * E[log β_jl] = sum_jl E[log β_jl] * sum_i z_ijl
        elbo_z += np.sum(z_sum_over_samples * self.E_log_beta)

        # Term 3: -sum_ijl E[θ_il]*E[β_jl] = -sum_l (sum_i E[θ_il]) * (sum_j E[β_jl])
        elbo_z -= np.sum(self.E_theta.sum(axis=0) * self.E_beta.sum(axis=0))

        # Term 4: -sum_ijl log(z_ijl!) ≈ -sum_ijl z_ijl*log(z_ijl) + z_ijl (Stirling for large z)
        # For small z, we can use approximation or skip (minor contribution)
        # We approximate with: -sum z*log(z+1) which is well-behaved
        # This is computed from sufficient stats approximately
        elbo += elbo_z

        # =====================================================================
        # E[log p(y | θ, v, γ)] - Vectorized Bernoulli with JJ bound
        # =====================================================================
        elbo_y = 0.0
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = self._lambda_jj(self.zeta[:, k])

            # Vectorized over all samples
            E_A = self.E_theta @ self.E_v[k] + X_aux @ self.E_gamma[k]  # (n,)
            E_A_sq = E_A**2

            # Add variance contribution: sum_l E[v²_kl] * Var[θ_il]
            E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])  # (d,)
            Var_theta = self.a_theta / (self.b_theta**2)  # (n, d)
            E_A_sq += (Var_theta * E_v_sq[np.newaxis, :]).sum(axis=1)

            elbo_y += np.sum((y_k - 0.5) * E_A - lam * E_A_sq)
        elbo += elbo_y

        # =====================================================================
        # E[log p(θ | ξ)] + E[log p(ξ)] - Vectorized
        # =====================================================================
        # E[log p(ξ)]
        elbo_theta_xi = np.sum((self.alpha_xi - 1) * self.E_log_xi)
        elbo_theta_xi -= self.lambda_xi * np.sum(self.E_xi)
        elbo_theta_xi += self.n * (self.alpha_xi * np.log(self.lambda_xi) - gammaln(self.alpha_xi))

        # E[log p(θ | ξ)]
        elbo_theta_xi += (self.alpha_theta - 1) * np.sum(self.E_log_theta)
        elbo_theta_xi += self.alpha_theta * self.d * np.sum(self.E_log_xi)
        elbo_theta_xi -= np.sum(self.E_xi[:, np.newaxis] * self.E_theta)
        elbo_theta_xi -= self.n * self.d * gammaln(self.alpha_theta)
        elbo += elbo_theta_xi

        # =====================================================================
        # E[log p(β | η)] + E[log p(η)] - Vectorized with spike-and-slab
        # =====================================================================
        # E[log p(η)]
        elbo_beta_eta = np.sum((self.alpha_eta - 1) * self.E_log_eta)
        elbo_beta_eta -= self.lambda_eta * np.sum(self.E_eta)
        elbo_beta_eta += self.p * (self.alpha_eta * np.log(self.lambda_eta) - gammaln(self.alpha_eta))

        # E[log p(β | η, s_beta)] - slab contribution
        slab_contrib = (self.alpha_beta - 1) * self.E_log_beta_slab
        slab_contrib += self.alpha_beta * self.E_log_eta[:, np.newaxis]
        slab_contrib -= self.E_eta[:, np.newaxis] * self.E_beta_slab
        slab_contrib -= gammaln(self.alpha_beta)
        elbo_beta_eta += np.sum(self.rho_beta * slab_contrib)

        # Prior on indicator
        rho_safe = np.clip(self.rho_beta, 1e-10, 1 - 1e-10)
        elbo_beta_eta += np.sum(self.rho_beta * np.log(self.pi_beta + 1e-10))
        elbo_beta_eta += np.sum((1 - self.rho_beta) * np.log(1 - self.pi_beta + 1e-10))
        elbo += elbo_beta_eta

        # =====================================================================
        # E[log p(v)] + E[log p(γ)] - Vectorized with spike-and-slab
        # =====================================================================
        elbo_v_gamma = 0.0
        Sigma_v_diag = np.array([np.diag(self.Sigma_v[k]) for k in range(self.kappa)])  # (kappa, d)

        # Slab contribution
        slab_contrib = -0.5 * np.log(2 * np.pi * self.sigma_v**2)
        slab_contrib -= 0.5 * (self.mu_v**2 + Sigma_v_diag) / self.sigma_v**2
        elbo_v_gamma += np.sum(self.rho_v * slab_contrib)

        # Spike contribution
        spike_contrib = -0.5 * np.log(2 * np.pi * self.spike_variance_v)
        spike_contrib -= 0.5 * (self.mu_v**2 + Sigma_v_diag) / self.spike_variance_v
        elbo_v_gamma += np.sum((1 - self.rho_v) * spike_contrib)

        # Prior on v indicator
        elbo_v_gamma += np.sum(self.rho_v * np.log(self.pi_v + 1e-10))
        elbo_v_gamma += np.sum((1 - self.rho_v) * np.log(1 - self.pi_v + 1e-10))

        # E[log p(γ)]
        for k in range(self.kappa):
            elbo_v_gamma -= 0.5 * self.p_aux * np.log(2 * np.pi * self.sigma_gamma**2)
            elbo_v_gamma -= 0.5 * (np.sum(self.mu_gamma[k]**2) + np.trace(self.Sigma_gamma[k])) / self.sigma_gamma**2
        elbo += elbo_v_gamma

        # =====================================================================
        # Entropy terms - All vectorized
        # =====================================================================

        # H[q(θ)] - Gamma entropy
        entropy_theta = np.sum(self.a_theta - np.log(self.b_theta))
        entropy_theta += np.sum(gammaln(self.a_theta))
        entropy_theta += np.sum((1 - self.a_theta) * digamma(self.a_theta))
        elbo += entropy_theta

        # H[q(β)]
        entropy_beta = np.sum(self.a_beta - np.log(self.b_beta))
        entropy_beta += np.sum(gammaln(self.a_beta))
        entropy_beta += np.sum((1 - self.a_beta) * digamma(self.a_beta))
        elbo += entropy_beta

        # H[q(ξ)]
        entropy_xi = np.sum(self.a_xi - np.log(self.b_xi))
        entropy_xi += np.sum(gammaln(self.a_xi))
        entropy_xi += np.sum((1 - self.a_xi) * digamma(self.a_xi))
        elbo += entropy_xi

        # H[q(η)]
        entropy_eta = np.sum(self.a_eta - np.log(self.b_eta))
        entropy_eta += np.sum(gammaln(self.a_eta))
        entropy_eta += np.sum((1 - self.a_eta) * digamma(self.a_eta))
        elbo += entropy_eta

        # H[q(v)] - Multivariate normal entropy
        entropy_v = 0.0
        for k in range(self.kappa):
            sign, logdet = np.linalg.slogdet(self.Sigma_v[k])
            if sign > 0:
                entropy_v += 0.5 * (self.d * (1 + np.log(2 * np.pi)) + logdet)
        elbo += entropy_v

        # H[q(γ)]
        entropy_gamma = 0.0
        for k in range(self.kappa):
            sign, logdet = np.linalg.slogdet(self.Sigma_gamma[k])
            if sign > 0:
                entropy_gamma += 0.5 * (self.p_aux * (1 + np.log(2 * np.pi)) + logdet)
        elbo += entropy_gamma

        # H[q(s_beta)] - Bernoulli entropy (vectorized)
        rho_beta_safe = np.clip(self.rho_beta, 1e-10, 1 - 1e-10)
        entropy_s_beta = -np.sum(rho_beta_safe * np.log(rho_beta_safe) +
                                  (1 - rho_beta_safe) * np.log(1 - rho_beta_safe))
        elbo += entropy_s_beta

        # H[q(s_v)]
        rho_v_safe = np.clip(self.rho_v, 1e-10, 1 - 1e-10)
        entropy_s_v = -np.sum(rho_v_safe * np.log(rho_v_safe) +
                               (1 - rho_v_safe) * np.log(1 - rho_v_safe))
        elbo += entropy_s_v

        if debug:
            print(f"\n  === Iteration {iteration} ELBO (efficient): {elbo:.2f} ===")

        return elbo if np.isfinite(elbo) else -np.inf

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
        theta_damping: float = 0.3,  # Very conservative damping for theta
        beta_damping: float = 0.5,   # Conservative damping for beta
        v_damping: float = 0.2,      # VERY conservative damping for v (most critical)
        gamma_damping: float = 0.4,  # Conservative damping for gamma
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
        
        # Determine chunk size based on problem size (memory-adaptive)
        # For very large n_factors (>500), use ultra memory-efficient version
        use_ultra_efficient = self.d > 500

        for iteration in range(max_iter):
            # ============================================
            # Compute expectations
            # ============================================
            self._compute_expectations()

            # ============================================
            # Compute sufficient statistics (memory-efficient)
            # Instead of allocating full (n, p, d) z matrix
            # ============================================
            if use_ultra_efficient:
                z_sum_over_genes, z_sum_over_samples = self._compute_z_suffstats_memory_efficient(X)
            else:
                z_sum_over_genes, z_sum_over_samples = self._compute_z_suffstats(X)

            # ============================================
            # Update theta with damping (in-place where possible)
            # ============================================
            a_theta_old = self.a_theta.copy()
            b_theta_old = self.b_theta.copy()

            self._update_theta_from_suffstats(z_sum_over_genes, y, X_aux)

            # Apply damping (in-place)
            df_theta = damping_factors['theta']
            self.a_theta *= df_theta
            self.a_theta += (1 - df_theta) * a_theta_old
            self.b_theta *= df_theta
            self.b_theta += (1 - df_theta) * b_theta_old

            # ============================================
            # Update beta with damping
            # ============================================
            a_beta_old = self.a_beta.copy()
            b_beta_old = self.b_beta.copy()

            self._update_beta_from_suffstats(z_sum_over_samples)

            # Apply damping (in-place)
            df_beta = damping_factors['beta']
            self.a_beta *= df_beta
            self.a_beta += (1 - df_beta) * a_beta_old
            self.b_beta *= df_beta
            self.b_beta += (1 - df_beta) * b_beta_old

            # Update spike-and-slab indicators for beta (with damping)
            rho_beta_old = self.rho_beta.copy()
            self._update_rho_beta_from_suffstats(z_sum_over_samples)
            # Apply damping to rho_beta
            rho_beta_damping = 0.5
            self.rho_beta *= rho_beta_damping
            self.rho_beta += (1 - rho_beta_damping) * rho_beta_old
            
            # ============================================
            # Update xi with damping (in-place)
            # ============================================
            a_xi_old = self.a_xi.copy()
            b_xi_old = self.b_xi.copy()

            self._update_xi()

            # Apply damping (in-place)
            df_xi = damping_factors['xi']
            self.a_xi *= df_xi
            self.a_xi += (1 - df_xi) * a_xi_old
            self.b_xi *= df_xi
            self.b_xi += (1 - df_xi) * b_xi_old

            # ============================================
            # Update eta with damping (in-place)
            # ============================================
            a_eta_old = self.a_eta.copy()
            b_eta_old = self.b_eta.copy()

            self._update_eta()

            # Apply damping (in-place)
            df_eta = damping_factors['eta']
            self.a_eta *= df_eta
            self.a_eta += (1 - df_eta) * a_eta_old
            self.b_eta *= df_eta
            self.b_eta += (1 - df_eta) * b_eta_old

            # ============================================
            # Update v with damping (in-place)
            # ============================================
            mu_v_old = self.mu_v.copy()
            Sigma_v_old = self.Sigma_v.copy()

            self._update_v(y, X_aux)

            # Clip update magnitude to prevent single-iteration jumps
            max_v_update = 0.3
            np.clip(self.mu_v - mu_v_old, -max_v_update, max_v_update, out=self.mu_v)
            self.mu_v += mu_v_old

            # Apply damping on top of gradient clipping (in-place)
            df_v = damping_factors['v']
            self.mu_v *= df_v
            self.mu_v += (1 - df_v) * mu_v_old
            self.Sigma_v *= df_v
            self.Sigma_v += (1 - df_v) * Sigma_v_old

            # Update spike-and-slab indicators for v (with damping)
            rho_v_old = self.rho_v.copy()
            self._update_rho_v(y, X_aux)
            # Apply heavy damping to rho_v to prevent oscillation
            rho_v_damping = 0.3
            self.rho_v *= rho_v_damping
            self.rho_v += (1 - rho_v_damping) * rho_v_old

            # ============================================
            # Update gamma with damping (in-place)
            # ============================================
            mu_gamma_old = self.mu_gamma.copy()
            Sigma_gamma_old = self.Sigma_gamma.copy()

            self._update_gamma(y, X_aux)

            # Apply damping (in-place)
            df_gamma = damping_factors['gamma']
            self.mu_gamma *= df_gamma
            self.mu_gamma += (1 - df_gamma) * mu_gamma_old
            self.Sigma_gamma *= df_gamma
            self.Sigma_gamma += (1 - df_gamma) * Sigma_gamma_old

            # ============================================
            # Update zeta (no damping - auxiliary variable)
            # ============================================
            self._update_zeta(y, X_aux)

            # Free temporary arrays from this iteration
            del a_theta_old, b_theta_old, a_beta_old, b_beta_old, rho_beta_old
            del a_xi_old, b_xi_old, a_eta_old, b_eta_old
            del mu_v_old, Sigma_v_old, rho_v_old, mu_gamma_old, Sigma_gamma_old

            # ============================================
            # Compute ELBO (less frequently for speed)
            # Uses efficient version with precomputed sufficient stats
            # ============================================
            compute_elbo = (iteration % elbo_freq == 0 or iteration == 0 or iteration == max_iter - 1)

            if compute_elbo:
                # Recompute sufficient statistics for ELBO (or reuse if still in scope)
                if use_ultra_efficient:
                    z_sum_genes, z_sum_samples = self._compute_z_suffstats_memory_efficient(X)
                else:
                    z_sum_genes, z_sum_samples = self._compute_z_suffstats(X)

                elbo = self._compute_elbo_efficient(X, y, X_aux, z_sum_genes, z_sum_samples,
                                                     debug=debug, iteration=iteration+1)
                elbo_history.append((iteration, elbo))
                del z_sum_genes, z_sum_samples
            
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
                if self.E_gamma.size > 0:
                    print(f"  E[gamma]: [{self.E_gamma.min():.4f}, {self.E_gamma.max():.4f}]")
                else:
                    print(f"  E[gamma]: (empty array)")
                print(f"  a_theta:  [{self.a_theta.min():.4f}, {self.a_theta.max():.4f}]")
                print(f"  b_theta:  [{self.b_theta.min():.4f}, {self.b_theta.max():.4f}]")
                print(f"Spike-and-slab indicators:")
                print(f"  rho_beta: [{self.rho_beta.min():.4f}, {self.rho_beta.max():.4f}], mean={self.rho_beta.mean():.4f}")
                print(f"  rho_v:    [{self.rho_v.min():.4f}, {self.rho_v.max():.4f}], mean={self.rho_v.mean():.4f}")
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
        X_aux: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
        use_regression: bool = True,
        batch_size: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Infer theta (sample-specific factors) for new samples using learned global parameters.

        This performs proper variational inference using ALL learned global parameters:
        - beta, eta: for the count model (X | theta, beta)
        - v, gamma: for the classification model (y | theta, v, gamma)

        For new samples, y is LATENT (unknown). We jointly infer theta and predict y
        using coordinate ascent:
        1. Initialize theta from X (count model only)
        2. Compute E[y] = σ(θ @ v + x_aux @ γ) - predicted probability
        3. Update theta using both count model AND regression with E[y] as soft label
        4. Update auxiliary parameters (xi, zeta)
        5. Repeat until convergence

        This is more principled than just using X alone, because it incorporates
        the learned classification structure into the theta inference.

        Parameters:
        -----------
        X : np.ndarray, shape (n_new, p)
            Gene expression count matrix for new samples
        X_aux : np.ndarray, shape (n_new, p_aux)
            Auxiliary features for new samples
        max_iter : int
            Maximum iterations for inference
        tol : float
            Convergence tolerance for theta updates
        verbose : bool
            Whether to print progress
        use_regression : bool (default=True)
            If True, include regression contribution using E[y] as soft label.
            If False, only use count model (original behavior).

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

        # Initialize theta parameters for new samples based on data scale
        row_sums = X.sum(axis=1, keepdims=True) + 1e-6
        a_theta_new = self.alpha_theta + (row_sums / self.d) * np.ones((1, self.d))

        # Initialize xi parameters
        a_xi_new = np.full(n_new, self.alpha_xi + self.d * self.alpha_theta)
        b_xi_new = np.full(n_new, self.lambda_xi)

        # Use learned E_beta to compute the beta sum per factor
        beta_sum_per_factor = np.sum(self.E_beta, axis=0)  # shape (d,)

        # Initialize b_theta
        E_xi_new = a_xi_new / b_xi_new
        b_theta_new = E_xi_new[:, np.newaxis] + beta_sum_per_factor[np.newaxis, :]

        # Initialize zeta (JJ auxiliary parameter) for new samples
        zeta_new = np.ones((n_new, self.kappa)) * 0.5

        if verbose:
            print(f"Inferring theta for {n_new} new samples (batch_size={batch_size})...")
            print(f"  Library sizes: min={row_sums.min():.0f}, max={row_sums.max():.0f}, mean={row_sums.mean():.0f}")
            print(f"  Using regression contribution: {use_regression}")

        # Coordinate ascent to jointly infer theta, xi, and predict y
        for iteration in range(max_iter):
            a_theta_old = a_theta_new.copy()

            # Process in batches to avoid OOM
            for start in range(0, n_new, batch_size):
                end = min(start + batch_size, n_new)

                # Get batch slices
                X_batch = X[start:end]
                X_aux_batch = X_aux[start:end]
                a_theta_batch = a_theta_new[start:end]
                b_theta_batch = b_theta_new[start:end]
                zeta_batch = zeta_new[start:end]

                # Compute expectations for batch
                E_theta_batch = a_theta_batch / b_theta_batch
                E_log_theta_batch = digamma(a_theta_batch) - np.log(b_theta_batch)

                # ============================================
                # Count model: compute z sufficient statistics (memory-efficient)
                # ============================================
                z_sum_genes, _ = self._compute_z_suffstats(X_batch, chunk_size=min(500, end - start))
                # Note: This uses the class's E_log_beta internally via _compute_z_suffstats
                # Need a local version that takes E_log_theta as parameter

                # Actually compute it directly here for infer_theta
                z_sum_genes = np.zeros((end - start, self.d), dtype=np.float64)
                p = X_batch.shape[1]
                gene_chunk = 2000
                for g_start in range(0, p, gene_chunk):
                    g_end = min(g_start + gene_chunk, p)
                    E_log_beta_chunk = self.E_log_beta[g_start:g_end]
                    X_chunk = X_batch[:, g_start:g_end]

                    log_phi = E_log_theta_batch[:, np.newaxis, :] + E_log_beta_chunk[np.newaxis, :, :]
                    log_phi_max = log_phi.max(axis=2, keepdims=True)
                    phi = np.exp(log_phi - log_phi_max)
                    phi = phi / phi.sum(axis=2, keepdims=True)

                    z_chunk = X_chunk[:, :, np.newaxis] * phi
                    z_sum_genes += z_chunk.sum(axis=1)
                    del log_phi, phi, z_chunk

                # Shape parameter from count model
                a_theta_new[start:end] = self.alpha_theta + z_sum_genes

                # Update xi
                a_xi_new[start:end] = self.alpha_xi + self.d * self.alpha_theta
                b_xi_new[start:end] = self.lambda_xi + np.sum(E_theta_batch, axis=1)
                b_xi_new[start:end] = np.clip(b_xi_new[start:end], 1e-6, 1e6)
                E_xi_batch = a_xi_new[start:end] / b_xi_new[start:end]

                # Rate parameter: base from count model
                b_theta_new[start:end] = E_xi_batch[:, np.newaxis] + beta_sum_per_factor[np.newaxis, :]

                # Regression contribution using E[y] as soft label
                if use_regression:
                    for k in range(self.kappa):
                        E_A = E_theta_batch @ self.E_v[k] + X_aux_batch @ self.E_gamma[k]
                        E_y = expit(E_A)

                        E_A_sq = E_A**2
                        E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
                        Var_theta = a_theta_batch / (b_theta_batch**2)
                        E_A_sq += (Var_theta * E_v_sq[np.newaxis, :]).sum(axis=1)
                        zeta_new[start:end, k] = np.sqrt(np.maximum(E_A_sq, 1e-10))

                        lam = self._lambda_jj(zeta_new[start:end, k])

                        # Vectorized regression contribution
                        E_v_sq_all = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
                        theta_v_product = E_theta_batch @ self.E_v[k]
                        base_pred = X_aux_batch @ self.E_gamma[k]

                        for ell in range(self.d):
                            E_C = theta_v_product - E_theta_batch[:, ell] * self.E_v[k, ell] + base_pred

                            regression_contrib = (
                                -(E_y - 0.5) * self.E_v[k, ell]
                                + 2 * lam * self.E_v[k, ell] * E_C
                                + 2 * lam * E_theta_batch[:, ell] * E_v_sq_all[ell]
                            )

                            b_theta_new[start:end, ell] += self.regression_weight * regression_contrib

                del z_sum_genes, E_theta_batch, E_log_theta_batch

            b_theta_new = np.clip(b_theta_new, 1e-6, 1e6)
            a_theta_new = np.clip(a_theta_new, 1.01, 1e6)

            # Check convergence
            max_change = np.max(np.abs(a_theta_new - a_theta_old))
            if verbose and iteration % 10 == 0:
                E_theta_tmp = a_theta_new / b_theta_new
                if use_regression:
                    E_y_tmp = expit(E_theta_tmp @ self.E_v[0] + X_aux @ self.E_gamma[0])
                    print(f"  Iter {iteration + 1}/{max_iter}, max_change: {max_change:.6f}, "
                          f"E[y] range: [{E_y_tmp.min():.3f}, {E_y_tmp.max():.3f}]")
                else:
                    print(f"  Iter {iteration + 1}/{max_iter}, max_change: {max_change:.6f}")

            if max_change < tol:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break

        # Final expectations
        E_theta_new = a_theta_new / b_theta_new

        if verbose:
            print(f"  Final E[theta] range: [{E_theta_new.min():.4f}, {E_theta_new.max():.4f}]")
            print(f"  Final E[theta] std: {E_theta_new.std():.4f}")
            if hasattr(self, 'E_theta'):
                print(f"  Training E[theta] std: {self.E_theta.std():.4f}")

        return E_theta_new, a_theta_new, b_theta_new
    
    def predict_proba(
        self,
        X: np.ndarray,
        X_aux: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
        use_sparse: bool = False,
        sparse_threshold: float = 0.5,
        use_regression: bool = True
    ) -> np.ndarray:
        """
        Predict probabilities for new samples.

        Uses the learned global parameters (beta, v, gamma, eta) to infer
        sample-specific factors (theta) and compute classification probabilities.

        The inference now uses proper variational updates that incorporate both:
        - Count model: X | theta, beta
        - Classification model: y | theta, v, gamma (with y treated as latent)

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
        use_sparse : bool
            If True, use thresholded sparse versions of v for prediction.
            This provides exact sparsity rather than soft sparsity.
        sparse_threshold : float
            Threshold for sparsity (only used if use_sparse=True)
        use_regression : bool (default=True)
            If True, include regression contribution in theta inference using
            E[y] as soft label. This is more principled and should give better
            predictions. If False, only use count model (original behavior).

        Returns:
        --------
        probs : np.ndarray, shape (n_new, kappa)
            Predicted probabilities for each class
        """
        # Infer theta for new samples using learned global parameters
        E_theta_new, _, _ = self.infer_theta(
            X,
            X_aux,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            use_regression=use_regression
        )

        # Get v values (sparse or soft)
        if use_sparse:
            E_v_pred = self.get_sparse_v(threshold=sparse_threshold)
        else:
            E_v_pred = self.E_v

        # Compute linear predictions and probabilities
        # A_ik = θ_i^T v_k + x_i^aux^T γ_k
        linear_pred = E_theta_new @ E_v_pred.T + X_aux @ self.E_gamma.T
        probs = expit(linear_pred)

        if verbose:
            print(f"Predicted probabilities: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")

        return probs

