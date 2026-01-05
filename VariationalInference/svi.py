"""
Stochastic Variational Inference Model for Single-Cell Analysis
================================================================

This module implements Stochastic Variational Inference (SVI) for the Poisson
Factor Model with spike-and-slab priors. SVI enables scalable inference on
large datasets by using mini-batches and natural gradients.

Algorithm (from Hoffman et al., 2013):
1. Initialize global parameters λ^(0) randomly
2. Set step-size schedule ρ_t appropriately
3. Repeat:
   - Sample a mini-batch of data points uniformly
   - Compute local variational parameters (theta) for the mini-batch
   - Compute intermediate global parameters as if mini-batch were full dataset
   - Update global parameters: λ^(t) = (1 - ρ_t)λ^(t-1) + ρ_t λ̂

Key Differences from Batch VI:
- Processes mini-batches instead of full dataset
- Uses learning rate schedule for global parameter updates
- Local parameters (theta) are computed fresh for each mini-batch
- Global parameters (beta, v, gamma, eta) are updated incrementally

Usage:
    from VariationalInference.svi import SVI

    model = SVI(n_factors=50, batch_size=128)
    model.fit(X, y, X_aux)
    predictions = model.predict_proba(X_new, X_aux_new)
"""

import numpy as np
from scipy.special import digamma, gammaln, expit, logsumexp
from typing import Tuple, Optional, Union, List
import time


class SVI:
    """
    Stochastic Variational Inference model for gene program discovery.

    This class implements SVI which scales better to large datasets by:
    1. Processing mini-batches instead of all samples
    2. Using natural gradient updates with learning rate schedule
    3. Separating local (theta) and global (beta, v, gamma) parameters

    Parameters
    ----------
    n_factors : int
        Number of latent gene programs to discover.
    batch_size : int, default=128
        Size of mini-batches for stochastic updates.
    learning_rate : float, default=0.01
        Initial learning rate for global parameter updates.
    learning_rate_decay : float, default=0.75
        Decay exponent for learning rate schedule (kappa in ρ_t = (τ + t)^(-κ)).
    learning_rate_delay : float, default=1.0
        Delay parameter for learning rate schedule (tau in ρ_t = (τ + t)^(-κ)).
    local_iterations : int, default=5
        Number of iterations to optimize local parameters per mini-batch.
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
        Prior standard deviation for classification weights.
    sigma_gamma : float, default=0.5
        Prior standard deviation for auxiliary feature effects.
    random_state : int or None, default=None
        Random seed for reproducibility.
    pi_v : float, default=0.2
        Prior probability of classification weights being active.
    pi_beta : float, default=0.05
        Prior probability of gene loadings being active.
    spike_variance_v : float, default=1e-6
        Variance for spike component of v.
    spike_value_beta : float, default=1e-6
        Value for spike component of beta.

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
    epoch_end_elbo_history_ : list of tuples
        History of (epoch, ELBO) values computed at the END of each epoch.
        Used for accurate epoch-over-epoch convergence tracking.
    training_time_ : float
        Total training time in seconds.
    """

    def __init__(
        self,
        n_factors: int,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.75,
        learning_rate_delay: float = 1.0,
        local_iterations: int = 5,
        alpha_theta: float = 2.0,
        alpha_beta: float = 2.0,
        alpha_xi: float = 2.0,
        alpha_eta: float = 2.0,
        lambda_xi: float = 1.5,
        lambda_eta: float = 1.5,
        sigma_v: float = 0.2,
        sigma_gamma: float = 0.5,
        random_state: Optional[int] = None,
        pi_v: float = 0.2,
        pi_beta: float = 0.05,
        spike_variance_v: float = 1e-6,
        spike_value_beta: float = 1e-6
    ):
        self.d = n_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_delay = learning_rate_delay
        self.local_iterations = local_iterations

        self.alpha_theta = alpha_theta
        self.alpha_beta = alpha_beta
        self.alpha_xi = alpha_xi
        self.alpha_eta = alpha_eta
        self.lambda_xi = lambda_xi
        self.lambda_eta = lambda_eta
        self.sigma_v = sigma_v
        self.sigma_gamma = sigma_gamma

        # Random state handling
        if random_state is None:
            self.rng = np.random.default_rng()
            self.seed_used_ = None
        else:
            self.rng = np.random.default_rng(random_state)
            self.seed_used_ = random_state

        self.regression_weight = 0.01

        # Spike-and-slab parameters
        self.pi_v = pi_v
        self.pi_beta = pi_beta
        self.spike_variance_v = spike_variance_v
        self.spike_value_beta = spike_value_beta

    def _get_learning_rate(self, iteration: int) -> float:
        """
        Compute learning rate for current iteration.

        Uses Robbins-Monro schedule: ρ_t = (τ + t)^(-κ)

        Parameters
        ----------
        iteration : int
            Current iteration number (0-indexed).

        Returns
        -------
        rho : float
            Learning rate for this iteration.
        """
        return (self.learning_rate_delay + iteration) ** (-self.learning_rate_decay)

    def _initialize_global_parameters(self, X: np.ndarray, y: np.ndarray, X_aux: np.ndarray):
        """
        Initialize global variational parameters with data-driven initialization.

        Global parameters in SVI:
        - beta: gene loadings (n_genes, n_factors)
        - eta: gene scaling factors (n_genes,)
        - v: classification weights (n_outcomes, n_factors)
        - gamma: auxiliary feature effects (n_outcomes, n_aux)
        - rho_beta, rho_v: spike-and-slab indicators
        """
        self.n, self.p = X.shape
        self.kappa = y.shape[1] if y.ndim > 1 else 1
        self.p_aux = X_aux.shape[1]

        # Initialize beta based on column statistics
        col_sums = X.sum(axis=0, keepdims=True) + 1e-6
        beta_init = (col_sums.T / self.d) * (1 + 0.1 * self.rng.standard_normal((self.p, self.d)))
        beta_init = np.maximum(beta_init, 0.1)

        variance_scale = 0.1
        self.a_beta = (beta_init**2) / (variance_scale * beta_init**2) + self.alpha_beta
        self.b_beta = beta_init / (variance_scale * beta_init**2) + 0.1

        # Initialize eta
        self.a_eta = self.alpha_eta + 0.1 * self.rng.exponential(scale=1.0, size=self.p)
        self.b_eta = np.ones(self.p) * self.lambda_eta

        # Initialize v (classification weights)
        self.mu_v = 0.01 * self.rng.standard_normal((self.kappa, self.d))
        self.Sigma_v = np.tile(np.eye(self.d)[np.newaxis, :, :], (self.kappa, 1, 1))

        # Initialize spike-and-slab indicators
        self.rho_v = np.ones((self.kappa, self.d)) * 0.5
        self.rho_beta = np.ones((self.p, self.d)) * self.pi_beta

        # Initialize gamma (auxiliary effects)
        self.mu_gamma = 0.01 * self.rng.standard_normal((self.kappa, self.p_aux))
        self.Sigma_gamma = np.tile(np.eye(self.p_aux)[np.newaxis, :, :], (self.kappa, 1, 1))

        # Compute initial global expectations
        self._compute_global_expectations()

    def _compute_global_expectations(self):
        """Compute expectations for global parameters."""
        # Beta expectations with spike-and-slab
        self.E_beta_slab = self.a_beta / self.b_beta
        self.E_log_beta_slab = digamma(self.a_beta) - np.log(self.b_beta)
        self.E_beta = self.rho_beta * self.E_beta_slab + (1 - self.rho_beta) * self.spike_value_beta
        self.E_log_beta = self.rho_beta * self.E_log_beta_slab + (1 - self.rho_beta) * np.log(self.spike_value_beta + 1e-10)

        # Eta expectations
        self.E_eta = self.a_eta / self.b_eta
        self.E_log_eta = digamma(self.a_eta) - np.log(self.b_eta)

        # V expectations with spike-and-slab
        self.E_v_slab = self.mu_v
        self.E_v = self.rho_v * self.mu_v

        # Gamma expectations
        self.E_gamma = self.mu_gamma

    def _initialize_local_parameters(self, X_batch: np.ndarray, batch_size: int):
        """
        Initialize local variational parameters for a mini-batch.

        Local parameters in SVI:
        - theta: sample-specific factor activities (batch_size, n_factors)
        - xi: sample depth correction (batch_size,)
        - zeta: JJ auxiliary parameters (batch_size, n_outcomes)
        """
        row_sums = X_batch.sum(axis=1, keepdims=True) + 1e-6

        theta_init = (row_sums / self.d) * (1 + 0.1 * self.rng.standard_normal((batch_size, self.d)))
        theta_init = np.maximum(theta_init, 0.1)

        variance_scale = 0.1
        a_theta = (theta_init**2) / (variance_scale * theta_init**2) + self.alpha_theta
        b_theta = theta_init / (variance_scale * theta_init**2) + 0.1

        a_xi = self.alpha_xi + 0.1 * self.rng.exponential(scale=1.0, size=batch_size)
        b_xi = np.ones(batch_size) * self.lambda_xi

        zeta = np.ones((batch_size, self.kappa)) * 0.1

        return a_theta, b_theta, a_xi, b_xi, zeta

    def _compute_local_expectations(self, a_theta, b_theta, a_xi, b_xi):
        """Compute expectations for local parameters."""
        E_theta = a_theta / b_theta
        E_log_theta = digamma(a_theta) - np.log(b_theta)
        E_xi = a_xi / b_xi
        E_log_xi = digamma(a_xi) - np.log(b_xi)
        return E_theta, E_log_theta, E_xi, E_log_xi

    def _lambda_jj(self, zeta: np.ndarray) -> np.ndarray:
        """Jaakola-Jordan lambda function."""
        result = np.zeros_like(zeta)
        nonzero = np.abs(zeta) > 1e-10
        result[nonzero] = (1.0 / (4.0 * zeta[nonzero])) * np.tanh(zeta[nonzero] / 2.0)
        result[~nonzero] = 0.125
        return result

    def _allocate_counts(self, X_batch: np.ndarray, E_log_theta, E_log_beta) -> np.ndarray:
        """Allocate counts using multinomial with expected log parameters."""
        log_phi = E_log_theta[:, np.newaxis, :] + E_log_beta[np.newaxis, :, :]
        log_phi_normalized = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
        phi = np.exp(log_phi_normalized)
        z = X_batch[:, :, np.newaxis] * phi
        return z

    def _compute_z_suffstats_batch(self, X_batch: np.ndarray, E_log_theta, E_log_beta,
                                    gene_chunk: int = 2000):
        """
        Compute sufficient statistics from z WITHOUT materializing full array.
        Processes genes in chunks to limit memory.

        Returns:
        --------
        z_sum_over_genes : ndarray, shape (batch_size, d)
            sum_j z_ijl for theta updates
        z_sum_over_samples : ndarray, shape (p, d)
            sum_i z_ijl for beta updates
        """
        batch_size = X_batch.shape[0]
        p = E_log_beta.shape[0]

        z_sum_over_genes = np.zeros((batch_size, self.d), dtype=np.float64)
        z_sum_over_samples = np.zeros((p, self.d), dtype=np.float64)

        for g_start in range(0, p, gene_chunk):
            g_end = min(g_start + gene_chunk, p)
            E_log_beta_chunk = E_log_beta[g_start:g_end]  # (g_chunk, d)
            X_chunk = X_batch[:, g_start:g_end]  # (batch, g_chunk)

            # Compute phi for this gene chunk
            log_phi = E_log_theta[:, np.newaxis, :] + E_log_beta_chunk[np.newaxis, :, :]
            log_phi_max = log_phi.max(axis=2, keepdims=True)
            phi = np.exp(log_phi - log_phi_max)
            phi = phi / phi.sum(axis=2, keepdims=True)

            # Use einsum to compute sufficient statistics directly without
            # materializing the full z_chunk array (batch_size x gene_chunk x d)
            # z_ijl = X_ij * phi_ijl, then sum appropriately
            z_sum_over_genes += np.einsum('ij,ijd->id', X_chunk, phi)
            z_sum_over_samples[g_start:g_end] = np.einsum('ij,ijd->jd', X_chunk, phi)

            del log_phi, phi

        return z_sum_over_genes, z_sum_over_samples

    def _update_local_theta(self, z: np.ndarray, y_batch: np.ndarray, X_aux_batch: np.ndarray,
                           a_theta, b_theta, E_theta, E_xi, zeta):
        """Update local theta parameters for mini-batch."""
        batch_size = z.shape[0]

        # Shape: alpha_theta + sum_j z_ijl
        a_theta_new = self.alpha_theta + np.sum(z, axis=1)

        # Rate: E[xi_i] + sum_j E[beta_jl] + regression terms
        b_theta_new = E_xi[:, np.newaxis] + np.sum(self.E_beta, axis=0)[np.newaxis, :]

        # Add regression contribution
        for k in range(self.kappa):
            y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
            lam = self._lambda_jj(zeta[:, k])

            for ell in range(self.d):
                mask = np.ones(self.d, dtype=bool)
                mask[ell] = False
                E_C = E_theta[:, mask] @ self.E_v[k, mask] + X_aux_batch @ self.E_gamma[k]

                E_v_sq_ell = self.mu_v[k, ell]**2 + self.Sigma_v[k, ell, ell]

                regression_contrib = (
                    -(y_k - 0.5) * self.E_v[k, ell]
                    + 2 * lam * self.E_v[k, ell] * E_C
                    + 2 * lam * E_theta[:, ell] * E_v_sq_ell
                )

                b_theta_new[:, ell] += self.regression_weight * regression_contrib

        b_theta_new = np.clip(b_theta_new, 1e-6, 1e6)
        a_theta_new = np.clip(a_theta_new, 1.01, 1e6)

        return a_theta_new, b_theta_new

    def _update_local_theta_from_suffstats(self, z_sum_over_genes: np.ndarray,
                                            y_batch: np.ndarray, X_aux_batch: np.ndarray,
                                            a_theta, b_theta, E_theta, E_xi, zeta):
        """
        Update local theta parameters using sufficient statistics (memory-efficient).

        Parameters:
        -----------
        z_sum_over_genes : ndarray, shape (batch_size, d)
            sum_j z_ijl - precomputed sufficient statistic
        """
        # Shape: alpha_theta + sum_j z_ijl
        a_theta_new = self.alpha_theta + z_sum_over_genes

        # Rate: E[xi_i] + sum_j E[beta_jl] + regression terms
        b_theta_new = E_xi[:, np.newaxis] + np.sum(self.E_beta, axis=0)[np.newaxis, :]

        # Add regression contribution (vectorized)
        for k in range(self.kappa):
            y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
            lam = self._lambda_jj(zeta[:, k])

            E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
            base_pred = X_aux_batch @ self.E_gamma[k]
            theta_v_product = E_theta @ self.E_v[k]

            # Vectorized over all factors ell (eliminates inner loop)
            # E_C_ell = theta_v_product - E_theta[:, ell] * E_v[k, ell] + base_pred
            E_C_all = theta_v_product[:, np.newaxis] - E_theta * self.E_v[k] + base_pred[:, np.newaxis]

            regression_contrib = (
                -(y_k - 0.5)[:, np.newaxis] * self.E_v[k]
                + 2 * lam[:, np.newaxis] * self.E_v[k] * E_C_all
                + 2 * lam[:, np.newaxis] * E_theta * E_v_sq
            )

            b_theta_new += self.regression_weight * regression_contrib

        b_theta_new = np.clip(b_theta_new, 1e-6, 1e6)
        a_theta_new = np.clip(a_theta_new, 1.01, 1e6)

        return a_theta_new, b_theta_new

    def _update_local_xi(self, E_theta, batch_size):
        """Update local xi parameters for mini-batch."""
        a_xi = np.full(batch_size, self.alpha_xi + self.d * self.alpha_theta)
        b_xi = self.lambda_xi + np.sum(E_theta, axis=1)
        b_xi = np.clip(b_xi, 1e-6, 1e6)
        return a_xi, b_xi

    def _update_local_zeta(self, E_theta, X_aux_batch, zeta):
        """Update local zeta (JJ auxiliary) parameters."""
        for k in range(self.kappa):
            E_A = E_theta @ self.E_v[k] + X_aux_batch @ self.E_gamma[k]
            E_A_sq = E_A**2

            E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
            # Approximate Var[theta] from current parameters
            zeta[:, k] = np.sqrt(np.maximum(E_A_sq + 0.1, 1e-10))

        return zeta

    def _compute_intermediate_beta(self, z: np.ndarray, E_theta, scale_factor: float):
        """
        Compute intermediate beta parameters as if mini-batch were full dataset.

        The key SVI idea: scale the sufficient statistics by N/batch_size
        """
        # Scale by N/batch_size to pretend mini-batch is full dataset
        a_beta_hat = self.alpha_beta + scale_factor * np.sum(z, axis=0)
        b_beta_hat = self.E_eta[:, np.newaxis] + scale_factor * np.sum(E_theta, axis=0)[np.newaxis, :]

        b_beta_hat = np.clip(b_beta_hat, 1e-6, 1e6)
        a_beta_hat = np.clip(a_beta_hat, 1.01, 1e6)

        return a_beta_hat, b_beta_hat

    def _compute_intermediate_beta_from_suffstats(self, z_sum_over_samples: np.ndarray,
                                                   E_theta, scale_factor: float):
        """
        Memory-efficient version using precomputed sufficient statistics.

        Parameters:
        -----------
        z_sum_over_samples : ndarray, shape (p, d)
            sum_i z_ijl - precomputed sufficient statistic
        """
        a_beta_hat = self.alpha_beta + scale_factor * z_sum_over_samples
        b_beta_hat = self.E_eta[:, np.newaxis] + scale_factor * np.sum(E_theta, axis=0)[np.newaxis, :]

        b_beta_hat = np.clip(b_beta_hat, 1e-6, 1e6)
        a_beta_hat = np.clip(a_beta_hat, 1.01, 1e6)

        return a_beta_hat, b_beta_hat

    def _compute_intermediate_eta(self, E_theta, scale_factor: float):
        """Compute intermediate eta parameters."""
        a_eta_hat = np.full(self.p, self.alpha_eta + self.d * self.alpha_beta)

        # E_beta_slab expectations for eta update
        E_beta_for_eta = self.rho_beta * self.E_beta_slab + (1 - self.rho_beta) * self.spike_value_beta
        b_eta_hat = self.lambda_eta + np.sum(E_beta_for_eta, axis=1)
        b_eta_hat = np.clip(b_eta_hat, 1e-6, 1e6)

        return a_eta_hat, b_eta_hat

    def _compute_intermediate_rho_beta(self, z: np.ndarray, E_theta, scale_factor: float):
        """Compute intermediate rho_beta (spike-and-slab indicators for beta)."""
        z_sum = scale_factor * np.sum(z, axis=0)
        theta_sum = scale_factor * np.sum(E_theta, axis=0)

        log_spike_value = np.log(self.spike_value_beta + 1e-10)

        ll_diff = z_sum * (self.E_log_beta_slab - log_spike_value)
        ll_diff -= theta_sum[np.newaxis, :] * (self.E_beta_slab - self.spike_value_beta)

        log_prior_odds = np.log(self.pi_beta / (1 - self.pi_beta + 1e-10))
        log_odds = ll_diff + log_prior_odds
        log_odds = np.clip(log_odds, -20, 20)

        return expit(log_odds)

    def _compute_intermediate_rho_beta_from_suffstats(self, z_sum_over_samples: np.ndarray,
                                                       E_theta, scale_factor: float):
        """
        Memory-efficient version using precomputed sufficient statistics.

        Parameters:
        -----------
        z_sum_over_samples : ndarray, shape (p, d)
            sum_i z_ijl - precomputed sufficient statistic
        """
        z_sum = scale_factor * z_sum_over_samples
        theta_sum = scale_factor * np.sum(E_theta, axis=0)

        log_spike_value = np.log(self.spike_value_beta + 1e-10)

        ll_diff = z_sum * (self.E_log_beta_slab - log_spike_value)
        ll_diff -= theta_sum[np.newaxis, :] * (self.E_beta_slab - self.spike_value_beta)

        log_prior_odds = np.log(self.pi_beta / (1 - self.pi_beta + 1e-10))
        log_odds = ll_diff + log_prior_odds
        log_odds = np.clip(log_odds, -20, 20)

        return expit(log_odds)

    def _compute_intermediate_v(self, y_batch: np.ndarray, X_aux_batch: np.ndarray,
                                E_theta, a_theta_batch, b_theta_batch, zeta, scale_factor: float):
        """Compute intermediate v parameters."""
        mu_v_hat = np.zeros_like(self.mu_v)
        Sigma_v_hat = np.zeros_like(self.Sigma_v)

        for k in range(self.kappa):
            y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
            lam = self._lambda_jj(zeta[:, k])

            E_theta_sq = E_theta**2 + a_theta_batch / (b_theta_batch**2)

            # Scale the precision contributions
            prec = (1.0 / self.sigma_v**2) * np.eye(self.d)
            prec += 2 * scale_factor * np.diag((lam[:, np.newaxis] * E_theta_sq).sum(axis=0))

            mean_contrib = scale_factor * ((y_k - 0.5)[:, np.newaxis] * E_theta).sum(axis=0)
            mean_contrib -= scale_factor * (2 * lam[:, np.newaxis] * E_theta *
                           (X_aux_batch @ self.E_gamma[k])[:, np.newaxis]).sum(axis=0)

            try:
                prec_reg = prec + 1e-4 * np.eye(self.d)
                Sigma_v_hat[k] = np.linalg.inv(prec_reg)
                mu_v_hat[k] = Sigma_v_hat[k] @ mean_contrib
                mu_v_hat[k] = np.clip(mu_v_hat[k], -1.5, 1.5)
            except np.linalg.LinAlgError:
                Sigma_v_hat[k] = self.Sigma_v[k]
                mu_v_hat[k] = self.mu_v[k]

        return mu_v_hat, Sigma_v_hat

    def _compute_intermediate_gamma(self, y_batch: np.ndarray, X_aux_batch: np.ndarray,
                                    E_theta, zeta, scale_factor: float):
        """Compute intermediate gamma parameters."""
        mu_gamma_hat = np.zeros_like(self.mu_gamma)
        Sigma_gamma_hat = np.zeros_like(self.Sigma_gamma)

        for k in range(self.kappa):
            y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
            lam = self._lambda_jj(zeta[:, k])

            prec = (1.0 / self.sigma_gamma**2) * np.eye(self.p_aux)
            prec += 2 * scale_factor * X_aux_batch.T @ (lam[:, np.newaxis] * X_aux_batch)

            mean_contrib = scale_factor * X_aux_batch.T @ (y_k - 0.5)
            mean_contrib -= scale_factor * 2 * X_aux_batch.T @ (lam * (E_theta @ self.E_v[k]))

            try:
                Sigma_gamma_hat[k] = np.linalg.inv(prec)
                mu_gamma_hat[k] = Sigma_gamma_hat[k] @ mean_contrib
                mu_gamma_hat[k] = np.clip(mu_gamma_hat[k], -3, 3)
            except np.linalg.LinAlgError:
                Sigma_gamma_hat[k] = self.Sigma_gamma[k]
                mu_gamma_hat[k] = self.mu_gamma[k]

        return mu_gamma_hat, Sigma_gamma_hat

    def _compute_intermediate_rho_v(self, y_batch: np.ndarray, X_aux_batch: np.ndarray,
                                    E_theta, a_theta_batch, b_theta_batch, zeta, scale_factor: float):
        """Compute intermediate rho_v (spike-and-slab indicators for v)."""
        rho_v_hat = np.zeros_like(self.rho_v)

        for k in range(self.kappa):
            y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
            lam = self._lambda_jj(zeta[:, k])

            base_pred = X_aux_batch @ self.E_gamma[k]

            # Vectorized over all factors ell (eliminates inner loop)
            # Compute full contribution once
            full_contrib = E_theta @ self.E_v[k]  # (batch,)

            # other_contrib_ell = full_contrib - E_theta[:, ell] * E_v[k, ell]
            other_contrib_all = full_contrib[:, np.newaxis] - E_theta * self.E_v[k]  # (batch, d)

            E_A_active_all = E_theta * self.mu_v[k] + other_contrib_all + base_pred[:, np.newaxis]
            E_A_inactive_all = other_contrib_all + base_pred[:, np.newaxis]

            E_v_sq_all = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])  # (d,)
            E_theta_sq_all = E_theta**2 + a_theta_batch / (b_theta_batch**2)  # (batch, d)
            Sigma_v_diag = np.diag(self.Sigma_v[k])  # (d,)

            E_A_sq_active_all = E_A_active_all**2 + E_theta_sq_all * Sigma_v_diag
            E_A_sq_inactive_all = E_A_inactive_all**2

            # Sum over samples (axis=0) for each factor
            y_shifted = (y_k - 0.5)[:, np.newaxis]
            lam_exp = lam[:, np.newaxis]

            ll_active_all = scale_factor * np.sum(y_shifted * E_A_active_all - lam_exp * E_A_sq_active_all, axis=0)
            ll_inactive_all = scale_factor * np.sum(y_shifted * E_A_inactive_all - lam_exp * E_A_sq_inactive_all, axis=0)

            prior_slab_all = -0.5 * E_v_sq_all / self.sigma_v**2

            log_odds_all = (ll_active_all - ll_inactive_all) + prior_slab_all
            log_odds_all += np.log(self.pi_v / (1 - self.pi_v + 1e-10))
            log_odds_all = np.clip(log_odds_all, -20, 20)
            rho_v_hat[k] = expit(log_odds_all)

        return rho_v_hat

    def _update_global_parameters(self, rho_t: float,
                                  a_beta_hat, b_beta_hat,
                                  a_eta_hat, b_eta_hat,
                                  rho_beta_hat,
                                  mu_v_hat, Sigma_v_hat,
                                  mu_gamma_hat, Sigma_gamma_hat,
                                  rho_v_hat):
        """
        Update global parameters using SVI update rule:
        λ^(t) = (1 - ρ_t) λ^(t-1) + ρ_t λ̂
        """
        # Update beta
        self.a_beta = (1 - rho_t) * self.a_beta + rho_t * a_beta_hat
        self.b_beta = (1 - rho_t) * self.b_beta + rho_t * b_beta_hat

        # Update eta
        self.a_eta = (1 - rho_t) * self.a_eta + rho_t * a_eta_hat
        self.b_eta = (1 - rho_t) * self.b_eta + rho_t * b_eta_hat

        # Update rho_beta
        self.rho_beta = (1 - rho_t) * self.rho_beta + rho_t * rho_beta_hat

        # Update v
        self.mu_v = (1 - rho_t) * self.mu_v + rho_t * mu_v_hat
        self.Sigma_v = (1 - rho_t) * self.Sigma_v + rho_t * Sigma_v_hat

        # Update gamma
        self.mu_gamma = (1 - rho_t) * self.mu_gamma + rho_t * mu_gamma_hat
        self.Sigma_gamma = (1 - rho_t) * self.Sigma_gamma + rho_t * Sigma_gamma_hat

        # Update rho_v
        self.rho_v = (1 - rho_t) * self.rho_v + rho_t * rho_v_hat

        # Recompute global expectations
        self._compute_global_expectations()

    def _compute_elbo_batch(self, X_batch: np.ndarray, y_batch: np.ndarray,
                            X_aux_batch: np.ndarray, E_theta, E_log_theta,
                            a_theta, b_theta, E_xi, E_log_xi, a_xi, b_xi,
                            zeta, scale_factor: float, gene_chunk: int = 2000) -> float:
        """
        Compute ELBO contribution from a mini-batch (scaled to full dataset).
        Memory-efficient version that processes genes in chunks.
        """
        batch_size = X_batch.shape[0]
        elbo = 0.0

        # E[log p(z | θ, β)] - scaled (memory-efficient chunked computation)
        elbo_z = 0.0
        for g_start in range(0, self.p, gene_chunk):
            g_end = min(g_start + gene_chunk, self.p)
            E_log_beta_chunk = self.E_log_beta[g_start:g_end]  # (g_chunk, d)
            E_beta_chunk = self.E_beta[g_start:g_end]  # (g_chunk, d)
            X_chunk = X_batch[:, g_start:g_end]  # (batch, g_chunk)

            # Compute phi for this gene chunk
            log_phi = E_log_theta[:, np.newaxis, :] + E_log_beta_chunk[np.newaxis, :, :]
            log_phi_max = log_phi.max(axis=2, keepdims=True)
            phi = np.exp(log_phi - log_phi_max)
            phi = phi / phi.sum(axis=2, keepdims=True)

            # z = x * phi
            z_chunk = X_chunk[:, :, np.newaxis] * phi  # (batch, g_chunk, d)

            # ELBO contribution from this chunk (vectorized)
            # sum_ijl z_ijl * (E_log_theta_il + E_log_beta_jl)
            elbo_z += np.sum(z_chunk * (E_log_theta[:, np.newaxis, :] + E_log_beta_chunk[np.newaxis, :, :]))
            # - sum_ijl E_theta_il * E_beta_jl (can be computed without z)
            elbo_z -= np.sum(E_theta[:, np.newaxis, :] * E_beta_chunk[np.newaxis, :, :])
            # - sum_ijl gammaln(z_ijl + 1)
            elbo_z -= np.sum(gammaln(z_chunk + 1))

            del log_phi, phi, z_chunk  # Free memory

        elbo += scale_factor * elbo_z

        # E[log p(y | θ, v, γ)] - scaled (vectorized)
        elbo_y = 0.0
        for k in range(self.kappa):
            y_k = y_batch[:, k] if y_batch.ndim > 1 else y_batch
            lam = self._lambda_jj(zeta[:, k])

            E_A = E_theta @ self.E_v[k] + X_aux_batch @ self.E_gamma[k]
            E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
            Var_theta = a_theta / (b_theta**2)
            E_A_sq = E_A**2 + np.sum(Var_theta * E_v_sq[np.newaxis, :], axis=1)

            elbo_y += np.sum((y_k - 0.5) * E_A - lam * E_A_sq)
        elbo += scale_factor * elbo_y

        # E[log p(θ | ξ)] + E[log p(ξ)] - scaled (vectorized)
        elbo_theta_xi = np.sum((self.alpha_xi - 1) * E_log_xi)
        elbo_theta_xi -= self.lambda_xi * np.sum(E_xi)
        elbo_theta_xi += batch_size * (self.alpha_xi * np.log(self.lambda_xi) - gammaln(self.alpha_xi))

        elbo_theta_xi += np.sum((self.alpha_theta - 1) * E_log_theta)
        elbo_theta_xi += self.alpha_theta * self.d * np.sum(E_log_xi)
        elbo_theta_xi -= np.sum(E_xi[:, np.newaxis] * E_theta)
        elbo_theta_xi -= batch_size * self.d * gammaln(self.alpha_theta)
        elbo += scale_factor * elbo_theta_xi

        # Local entropy (scaled, vectorized)
        entropy_theta = np.sum(a_theta - np.log(b_theta) + gammaln(a_theta) +
                               (1 - a_theta) * digamma(a_theta))
        elbo += scale_factor * entropy_theta

        entropy_xi = np.sum(a_xi - np.log(b_xi) + gammaln(a_xi) +
                            (1 - a_xi) * digamma(a_xi))
        elbo += scale_factor * entropy_xi

        # Global terms (not scaled - computed once, vectorized)
        # E[log p(β | η)] + E[log p(η)]
        elbo_beta_eta = np.sum((self.alpha_eta - 1) * self.E_log_eta)
        elbo_beta_eta -= self.lambda_eta * np.sum(self.E_eta)
        elbo_beta_eta += self.p * (self.alpha_eta * np.log(self.lambda_eta) - gammaln(self.alpha_eta))

        slab_contrib = ((self.alpha_beta - 1) * self.E_log_beta_slab +
                        self.alpha_beta * self.E_log_eta[:, np.newaxis] -
                        self.E_eta[:, np.newaxis] * self.E_beta_slab - gammaln(self.alpha_beta))
        elbo_beta_eta += np.sum(self.rho_beta * slab_contrib)

        # Spike-and-slab prior terms
        rho_safe = np.clip(self.rho_beta, 1e-10, 1 - 1e-10)
        elbo_beta_eta += np.sum(self.rho_beta * np.log(self.pi_beta + 1e-10))
        elbo_beta_eta += np.sum((1 - self.rho_beta) * np.log(1 - self.pi_beta + 1e-10))
        elbo += elbo_beta_eta

        # E[log p(v)] + E[log p(γ)] (vectorized)
        elbo_v_gamma = 0.0
        E_v_sq = self.mu_v**2 + np.diagonal(self.Sigma_v, axis1=1, axis2=2)
        slab_contrib_v = -0.5 * np.log(2 * np.pi * self.sigma_v**2) - 0.5 * E_v_sq / self.sigma_v**2
        spike_contrib_v = -0.5 * np.log(2 * np.pi * self.spike_variance_v) - 0.5 * E_v_sq / self.spike_variance_v
        elbo_v_gamma += np.sum(self.rho_v * slab_contrib_v + (1 - self.rho_v) * spike_contrib_v)
        elbo_v_gamma += np.sum(self.rho_v * np.log(self.pi_v + 1e-10))
        elbo_v_gamma += np.sum((1 - self.rho_v) * np.log(1 - self.pi_v + 1e-10))

        for k in range(self.kappa):
            elbo_v_gamma -= 0.5 * self.p_aux * np.log(2 * np.pi * self.sigma_gamma**2)
            elbo_v_gamma -= 0.5 * (np.sum(self.mu_gamma[k]**2) + np.trace(self.Sigma_gamma[k])) / self.sigma_gamma**2
        elbo += elbo_v_gamma

        # Global entropy terms (vectorized)
        entropy_beta = np.sum(self.a_beta - np.log(self.b_beta) + gammaln(self.a_beta) +
                              (1 - self.a_beta) * digamma(self.a_beta))
        elbo += entropy_beta

        entropy_eta = np.sum(self.a_eta - np.log(self.b_eta) + gammaln(self.a_eta) +
                             (1 - self.a_eta) * digamma(self.a_eta))
        elbo += entropy_eta

        entropy_v = 0.0
        for k in range(self.kappa):
            sign, logdet = np.linalg.slogdet(self.Sigma_v[k])
            if sign > 0:
                entropy_v += 0.5 * (self.d * (1 + np.log(2 * np.pi)) + logdet)
        elbo += entropy_v

        entropy_gamma = 0.0
        for k in range(self.kappa):
            sign, logdet = np.linalg.slogdet(self.Sigma_gamma[k])
            if sign > 0:
                entropy_gamma += 0.5 * (self.p_aux * (1 + np.log(2 * np.pi)) + logdet)
        elbo += entropy_gamma

        # Spike-and-slab entropy (vectorized)
        rho_beta_safe = np.clip(self.rho_beta, 1e-10, 1 - 1e-10)
        entropy_s_beta = -np.sum(rho_beta_safe * np.log(rho_beta_safe) +
                                  (1 - rho_beta_safe) * np.log(1 - rho_beta_safe))
        elbo += entropy_s_beta

        rho_v_safe = np.clip(self.rho_v, 1e-10, 1 - 1e-10)
        entropy_s_v = -np.sum(rho_v_safe * np.log(rho_v_safe) +
                               (1 - rho_v_safe) * np.log(1 - rho_v_safe))
        elbo += entropy_s_v

        return elbo if np.isfinite(elbo) else -np.inf

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_aux: np.ndarray,
        max_epochs: int = 100,
        elbo_freq: int = 10,
        min_epochs: int = 10,
        patience: int = 5,
        tol: float = 10.0,
        rel_tol: float = 2e-4,
        verbose: bool = True,
        debug: bool = False
    ):
        """
        Fit the model using Stochastic Variational Inference.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_genes)
            Gene expression count matrix.
        y : np.ndarray, shape (n_samples,) or (n_samples, n_outcomes)
            Binary labels for classification.
        X_aux : np.ndarray, shape (n_samples, n_aux_features)
            Auxiliary features (e.g., sex, batch).
        max_epochs : int, default=100
            Maximum number of epochs (passes through data).
        elbo_freq : int, default=10
            Compute ELBO every N mini-batch iterations.
        min_epochs : int, default=10
            Minimum epochs before checking convergence.
        patience : int, default=5
            Early stopping patience.
        tol : float, default=10.0
            Absolute ELBO tolerance for convergence.
        rel_tol : float, default=2e-4
            Relative ELBO tolerance for convergence.
        verbose : bool, default=True
            Whether to print progress.
        debug : bool, default=False
            Whether to print detailed debug information.

        Returns
        -------
        self
        """
        start_time = time.time()

        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Initialize global parameters
        self._initialize_global_parameters(X, y, X_aux)

        n_samples = X.shape[0]
        batch_size = min(self.batch_size, n_samples)
        n_batches = max(1, n_samples // batch_size)
        scale_factor = n_samples / batch_size

        elbo_history = []
        epoch_end_elbo_history = []  # Track ELBO at the END of each epoch for proper convergence
        patience_counter = 0
        iteration = 0

        if verbose:
            print(f"\nSVI Training Configuration:")
            print(f"  Samples: {n_samples}, Genes: {self.p}, Factors: {self.d}")
            print(f"  Batch size: {batch_size}, Batches per epoch: {n_batches}")
            print(f"  Learning rate schedule: (τ + t)^(-κ) with τ={self.learning_rate_delay}, κ={self.learning_rate_decay}")

        for epoch in range(max_epochs):
            # Shuffle data at the start of each epoch
            perm = self.rng.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            X_aux_shuffled = X_aux[perm]

            epoch_start = time.time()

            for batch_idx in range(n_batches):
                # Get mini-batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                actual_batch_size = end_idx - start_idx

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                X_aux_batch = X_aux_shuffled[start_idx:end_idx]

                actual_scale = n_samples / actual_batch_size

                # Get learning rate for this iteration
                rho_t = self._get_learning_rate(iteration)

                # Initialize local parameters for this batch
                a_theta, b_theta, a_xi, b_xi, zeta = self._initialize_local_parameters(
                    X_batch, actual_batch_size
                )

                # Optimize local parameters (memory-efficient: no full z array)
                for local_iter in range(self.local_iterations):
                    E_theta, E_log_theta, E_xi, E_log_xi = self._compute_local_expectations(
                        a_theta, b_theta, a_xi, b_xi
                    )

                    # Compute z sufficient statistics without full array allocation
                    z_sum_genes, _ = self._compute_z_suffstats_batch(
                        X_batch, E_log_theta, self.E_log_beta
                    )

                    # Update local parameters using sufficient statistics
                    a_theta, b_theta = self._update_local_theta_from_suffstats(
                        z_sum_genes, y_batch, X_aux_batch, a_theta, b_theta, E_theta, E_xi, zeta
                    )
                    a_xi, b_xi = self._update_local_xi(E_theta, actual_batch_size)
                    zeta = self._update_local_zeta(E_theta, X_aux_batch, zeta)

                    E_theta, E_log_theta, E_xi, E_log_xi = self._compute_local_expectations(
                        a_theta, b_theta, a_xi, b_xi
                    )

                    del z_sum_genes  # Free memory

                # Compute intermediate global parameters (memory-efficient)
                z_sum_genes, z_sum_samples = self._compute_z_suffstats_batch(
                    X_batch, E_log_theta, self.E_log_beta
                )

                a_beta_hat, b_beta_hat = self._compute_intermediate_beta_from_suffstats(
                    z_sum_samples, E_theta, actual_scale
                )
                a_eta_hat, b_eta_hat = self._compute_intermediate_eta(E_theta, actual_scale)
                rho_beta_hat = self._compute_intermediate_rho_beta_from_suffstats(
                    z_sum_samples, E_theta, actual_scale
                )

                del z_sum_genes, z_sum_samples  # Free memory

                mu_v_hat, Sigma_v_hat = self._compute_intermediate_v(
                    y_batch, X_aux_batch, E_theta, a_theta, b_theta, zeta, actual_scale
                )
                mu_gamma_hat, Sigma_gamma_hat = self._compute_intermediate_gamma(
                    y_batch, X_aux_batch, E_theta, zeta, actual_scale
                )
                rho_v_hat = self._compute_intermediate_rho_v(
                    y_batch, X_aux_batch, E_theta, a_theta, b_theta, zeta, actual_scale
                )

                # Update global parameters with learning rate
                self._update_global_parameters(
                    rho_t,
                    a_beta_hat, b_beta_hat,
                    a_eta_hat, b_eta_hat,
                    rho_beta_hat,
                    mu_v_hat, Sigma_v_hat,
                    mu_gamma_hat, Sigma_gamma_hat,
                    rho_v_hat
                )

                # Compute ELBO periodically
                if iteration % elbo_freq == 0:
                    elbo = self._compute_elbo_batch(
                        X_batch, y_batch, X_aux_batch,
                        E_theta, E_log_theta, a_theta, b_theta,
                        E_xi, E_log_xi, a_xi, b_xi, zeta, actual_scale
                    )
                    elbo_history.append((iteration, elbo))

                    if debug:
                        print(f"  Iter {iteration}: ELBO={elbo:.2f}, ρ_t={rho_t:.4f}")

                iteration += 1

            epoch_time = time.time() - epoch_start

            # End of epoch: compute epoch-end ELBO for accurate epoch-over-epoch comparison
            # Use the last batch's local parameters to compute epoch-end ELBO
            epoch_end_elbo = self._compute_elbo_batch(
                X_batch, y_batch, X_aux_batch,
                E_theta, E_log_theta, a_theta, b_theta,
                E_xi, E_log_xi, a_xi, b_xi, zeta, actual_scale
            )
            epoch_end_elbo_history.append((epoch, epoch_end_elbo))
            elbo_history.append((iteration - 1, epoch_end_elbo))  # Also add to elbo_history for completeness

            # Report epoch progress
            if verbose:
                total_time = time.time() - start_time
                print(f"\nEpoch {epoch + 1}/{max_epochs}, ELBO: {epoch_end_elbo:.2f}, "
                      f"ρ_t: {rho_t:.4f} (epoch: {epoch_time:.2f}s, total: {total_time:.2f}s)")

                if len(epoch_end_elbo_history) > 1:
                    _, prev_epoch_elbo = epoch_end_elbo_history[-2]
                    elbo_change = epoch_end_elbo - prev_epoch_elbo
                    change_symbol = "↑" if elbo_change > 0 else "↓"
                    rel_change = abs(elbo_change / (abs(prev_epoch_elbo) + 1e-10))
                    print(f"  Epoch ELBO change: {change_symbol} {elbo_change:.2f} (relative: {rel_change:.6f})")

            # Check convergence using epoch-end ELBOs for accurate epoch-over-epoch comparison
            if epoch >= min_epochs and len(epoch_end_elbo_history) > 1:
                _, elbo_curr = epoch_end_elbo_history[-1]
                _, elbo_prev = epoch_end_elbo_history[-2]
                elbo_change = elbo_curr - elbo_prev
                rel_change = abs(elbo_change / (abs(elbo_prev) + 1e-10))

                if abs(elbo_change) < tol and rel_change < rel_tol:
                    patience_counter += 1
                    if verbose:
                        print(f"  Patience: {patience_counter}/{patience}")
                else:
                    patience_counter = 0

                if patience_counter >= patience:
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"Converged after {epoch + 1} epochs")
                        print(f"{'='*60}")
                    break

        # Store full-data theta for training samples
        self._compute_full_theta(X, y, X_aux)

        self.elbo_history_ = elbo_history
        self.epoch_end_elbo_history_ = epoch_end_elbo_history  # Epoch-end ELBOs for accurate convergence tracking
        self.training_time_ = time.time() - start_time

        if verbose:
            print(f"\nTraining completed in {self.training_time_:.2f}s")
            if elbo_history:
                print(f"Final ELBO: {elbo_history[-1][1]:.2f}")

        return self

    def _compute_full_theta(self, X: np.ndarray, y: np.ndarray, X_aux: np.ndarray,
                             batch_size: int = 1000, n_iter: int = 10):
        """
        Compute theta for all training samples after fitting.
        Uses batched processing to avoid OOM with large datasets.

        Parameters:
        -----------
        batch_size : int
            Number of samples to process at once (default: 1000)
        n_iter : int
            Number of iterations for theta inference (default: 10, reduced from 20)
        """
        n_samples = X.shape[0]

        # Initialize
        row_sums = X.sum(axis=1, keepdims=True) + 1e-6
        theta_init = (row_sums / self.d) * np.ones((1, self.d))
        theta_init = np.maximum(theta_init, 0.1)

        variance_scale = 0.1
        self.a_theta = (theta_init**2) / (variance_scale * theta_init**2) + self.alpha_theta
        self.b_theta = theta_init / (variance_scale * theta_init**2) + 0.1

        self.a_xi = self.alpha_xi + 0.1 * np.ones(n_samples)
        self.b_xi = np.ones(n_samples) * self.lambda_xi
        self.zeta = np.ones((n_samples, self.kappa)) * 0.1

        # Iterate to convergence using batched updates
        for iteration in range(n_iter):
            # Process in batches to limit memory
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                # Get batch data
                X_batch = X[start:end]
                y_batch = y[start:end] if y.ndim > 1 else y[start:end, np.newaxis]
                X_aux_batch = X_aux[start:end]
                a_theta_batch = self.a_theta[start:end]
                b_theta_batch = self.b_theta[start:end]
                a_xi_batch = self.a_xi[start:end]
                b_xi_batch = self.b_xi[start:end]
                zeta_batch = self.zeta[start:end]

                # Compute local expectations for batch
                E_theta_batch = a_theta_batch / b_theta_batch
                E_log_theta_batch = digamma(a_theta_batch) - np.log(b_theta_batch)
                E_xi_batch = a_xi_batch / b_xi_batch

                # Compute z sufficient statistics (memory-efficient)
                z_sum_genes, _ = self._compute_z_suffstats_batch(
                    X_batch, E_log_theta_batch, self.E_log_beta
                )

                # Update theta for this batch
                a_theta_new, b_theta_new = self._update_local_theta_from_suffstats(
                    z_sum_genes, y_batch, X_aux_batch,
                    a_theta_batch, b_theta_batch, E_theta_batch, E_xi_batch, zeta_batch
                )

                # Update xi for this batch
                E_theta_new = a_theta_new / b_theta_new
                a_xi_new, b_xi_new = self._update_local_xi(E_theta_new, end - start)

                # Update zeta for this batch
                zeta_new = self._update_local_zeta(E_theta_new, X_aux_batch, zeta_batch)

                # Store back
                self.a_theta[start:end] = a_theta_new
                self.b_theta[start:end] = b_theta_new
                self.a_xi[start:end] = a_xi_new
                self.b_xi[start:end] = b_xi_new
                self.zeta[start:end] = zeta_new

                # Free memory
                del z_sum_genes, E_theta_batch, E_log_theta_batch, E_xi_batch
                del a_theta_new, b_theta_new, a_xi_new, b_xi_new, zeta_new

        # Store final expectations
        self.E_theta = self.a_theta / self.b_theta
        self.E_log_theta = digamma(self.a_theta) - np.log(self.b_theta)
        self.E_xi = self.a_xi / self.b_xi
        self.E_log_xi = digamma(self.a_xi) - np.log(self.b_xi)

    def get_sparse_beta(self, threshold: float = 0.5) -> np.ndarray:
        """Get sparse version of E[beta] by thresholding spike-and-slab indicators."""
        return np.where(self.rho_beta > threshold, self.E_beta_slab, 0.0)

    def get_sparse_v(self, threshold: float = 0.5) -> np.ndarray:
        """Get sparse version of E[v] by thresholding spike-and-slab indicators."""
        return np.where(self.rho_v > threshold, self.mu_v, 0.0)

    def get_active_factors(self, threshold: float = 0.5) -> dict:
        """Get indices of active factors for each outcome."""
        active_beta = [np.where(self.rho_beta[:, ell] > threshold)[0]
                       for ell in range(self.d)]
        active_v = [np.where(self.rho_v[k, :] > threshold)[0]
                    for k in range(self.kappa)]

        return {
            'beta': active_beta,
            'v': active_v,
            'n_active_beta': sum(len(a) for a in active_beta),
            'n_active_v': sum(len(a) for a in active_v)
        }

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
        Infer theta for new samples using learned global parameters.
        Uses batched processing to avoid OOM with large datasets.

        Same interface as VI.infer_theta for compatibility.
        """
        n_new = X.shape[0]

        if not hasattr(self, 'E_beta'):
            raise RuntimeError("Model must be fitted before inferring theta")

        row_sums = X.sum(axis=1, keepdims=True) + 1e-6
        a_theta_new = self.alpha_theta + (row_sums / self.d) * np.ones((1, self.d))

        a_xi_new = np.full(n_new, self.alpha_xi + self.d * self.alpha_theta)
        b_xi_new = np.full(n_new, self.lambda_xi)

        beta_sum_per_factor = np.sum(self.E_beta, axis=0)
        E_xi_new = a_xi_new / b_xi_new
        b_theta_new = E_xi_new[:, np.newaxis] + beta_sum_per_factor[np.newaxis, :]

        zeta_new = np.ones((n_new, self.kappa)) * 0.5

        if verbose:
            print(f"Inferring theta for {n_new} new samples (batch_size={batch_size})...")

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

                E_theta_batch = a_theta_batch / b_theta_batch
                E_log_theta_batch = digamma(a_theta_batch) - np.log(b_theta_batch)

                # Compute z sufficient statistics using chunked processing
                z_sum_genes, _ = self._compute_z_suffstats_batch(
                    X_batch, E_log_theta_batch, self.E_log_beta
                )

                # Update a_theta for this batch
                a_theta_new[start:end] = self.alpha_theta + z_sum_genes

                # Update xi
                a_xi_new[start:end] = self.alpha_xi + self.d * self.alpha_theta
                b_xi_new[start:end] = self.lambda_xi + np.sum(E_theta_batch, axis=1)
                b_xi_new[start:end] = np.clip(b_xi_new[start:end], 1e-6, 1e6)
                E_xi_batch = a_xi_new[start:end] / b_xi_new[start:end]

                # Update b_theta
                b_theta_new[start:end] = E_xi_batch[:, np.newaxis] + beta_sum_per_factor[np.newaxis, :]

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

                        # Vectorized regression contribution over all factors ell
                        E_v_sq_all = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
                        theta_v_product = E_theta_batch @ self.E_v[k]
                        base_pred = X_aux_batch @ self.E_gamma[k]

                        # E_C_ell = theta_v_product - E_theta[:, ell] * E_v[k, ell] + base_pred
                        E_C_all = theta_v_product[:, np.newaxis] - E_theta_batch * self.E_v[k] + base_pred[:, np.newaxis]

                        regression_contrib = (
                            -(E_y - 0.5)[:, np.newaxis] * self.E_v[k]
                            + 2 * lam[:, np.newaxis] * self.E_v[k] * E_C_all
                            + 2 * lam[:, np.newaxis] * E_theta_batch * E_v_sq_all
                        )

                        b_theta_new[start:end] += self.regression_weight * regression_contrib

                del z_sum_genes, E_theta_batch, E_log_theta_batch

            b_theta_new = np.clip(b_theta_new, 1e-6, 1e6)
            a_theta_new = np.clip(a_theta_new, 1.01, 1e6)

            max_change = np.max(np.abs(a_theta_new - a_theta_old))
            if verbose and iteration % 10 == 0:
                print(f"  Iter {iteration + 1}/{max_iter}, max_change: {max_change:.6f}")

            if max_change < tol:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break

        E_theta_new = a_theta_new / b_theta_new
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

        Same interface as VI.predict_proba for compatibility.
        """
        E_theta_new, _, _ = self.infer_theta(
            X, X_aux, max_iter=max_iter, tol=tol,
            verbose=verbose, use_regression=use_regression
        )

        if use_sparse:
            E_v_pred = self.get_sparse_v(threshold=sparse_threshold)
        else:
            E_v_pred = self.E_v

        linear_pred = E_theta_new @ E_v_pred.T + X_aux @ self.E_gamma.T
        probs = expit(linear_pred)

        if verbose:
            print(f"Predicted probabilities: min={probs.min():.4f}, max={probs.max():.4f}")

        return probs
