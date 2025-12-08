import numpy as np
from scipy.special import digamma, gammaln, expit, logsumexp
from scipy.sparse import csr_matrix
from typing import Tuple, Optional
import numba
import pandas as pd
from sklearn.model_selection import train_test_split


# Numba-compiled helper functions for hot paths
@numba.jit(nopython=True, cache=True)
def lambda_jj_numba(zeta: np.ndarray) -> np.ndarray:
    """Jaakola-Jordan lambda function - JIT compiled."""
    result = np.zeros_like(zeta)
    for i in range(zeta.shape[0]):
        if abs(zeta[i]) > 1e-10:
            result[i] = (1.0 / (4.0 * zeta[i])) * np.tanh(zeta[i] / 2.0)
        else:
            result[i] = 0.125
    return result


@numba.jit(nopython=True, cache=True)
def compute_E_C_vectorized(E_theta: np.ndarray, E_v: np.ndarray, 
                          X_aux_i: np.ndarray, E_gamma: np.ndarray,
                          d: int) -> np.ndarray:
    """Compute E[C_kℓ] efficiently for all samples and factors."""
    n = E_theta.shape[0]
    E_C = np.zeros((n, d))
    
    # Contribution from all factors except current
    for ell in range(d):
        for i in range(n):
            for ell_prime in range(d):
                if ell_prime != ell:
                    E_C[i, ell] += E_theta[i, ell_prime] * E_v[ell_prime]
            # Add auxiliary contribution (same for all ell)
            if ell == 0:
                E_C[i, ell] += np.dot(X_aux_i, E_gamma)
            else:
                E_C[i, ell] += np.dot(X_aux_i, E_gamma)
    return E_C


@numba.jit(nopython=True, cache=True)
def update_theta_regression_term(E_theta: np.ndarray, E_v: np.ndarray,
                                 y_k: np.ndarray, lam: np.ndarray,
                                 X_aux_row: np.ndarray, E_gamma: np.ndarray,
                                 Sigma_v_diag: np.ndarray) -> np.ndarray:
    """Compute regression contribution to theta rate parameter."""
    n, d = E_theta.shape
    regression_contrib = np.zeros((n, d))
    
    for ell in range(d):
        for i in range(n):
            E_C = 0.0
            for ell_prime in range(d):
                if ell_prime != ell:
                    E_C += E_theta[i, ell_prime] * E_v[ell_prime]
            E_C += np.dot(X_aux_row, E_gamma)
            
            E_v_sq_ell = E_v[ell] * E_v[ell] + Sigma_v_diag[ell]
            regression_contrib[i, ell] = (
                -(y_k[i] - 0.5) * E_v[ell]
                + 2 * lam[i] * E_v[ell] * E_C
                + 2 * lam[i] * E_theta[i, ell] * E_v_sq_ell
            )
    return regression_contrib


class VI:
    """Optimized Variational Inference with numba JIT and vectorization."""
    
    def __init__(
        self,
        n_factors: int,
        alpha_theta: float = 2.0,  # FIXED: Must be > 1 for well-behaved Gamma
        alpha_beta: float = 2.0,   # FIXED: Must be > 1 for well-behaved Gamma
        alpha_xi: float = 2.0,     # FIXED: Increased from 1.5
        alpha_eta: float = 2.0,    # FIXED: Increased from 1.5
        lambda_xi: float = 1.5,
        lambda_eta: float = 1.5,
        sigma_v: float = 0.1,
        sigma_gamma: float = 0.1,
        random_state: int = 42
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
        self.regression_weight = 0.001
        
    def _initialize_parameters(self, X, y: np.ndarray, X_aux: np.ndarray):
        """Initialize variational parameters with data-driven initialization."""
        self.n, self.p = X.shape
        self.kappa = y.shape[1] if y.ndim > 1 else 1
        self.p_aux = X_aux.shape[1]
        
        # CONSERVATIVE initialization for sparse matrix
        row_sums = np.array(X.sum(axis=1)).flatten() + 1e-6
        col_sums = np.array(X.sum(axis=0)).flatten() + 1e-6
        
        # Scale down dramatically - typical scRNA-seq has low counts
        total_counts = np.sum(row_sums)
        mean_count_per_entry = total_counts / (X.nnz if hasattr(X, 'nnz') else np.count_nonzero(X))
        
        # Target: rates should be close to observed counts (1-10, not 1000s)
        scale_factor = max(0.1, mean_count_per_entry / 10.0)  # Much more conservative
        
        theta_init = (row_sums[:, np.newaxis] / self.d) * scale_factor * (1 + 0.01 * self.rng.randn(self.n, self.d))
        beta_init = (col_sums[:, np.newaxis] / self.d) * scale_factor * (1 + 0.01 * self.rng.randn(self.p, self.d))
        
        # Keep small but positive
        theta_init = np.clip(theta_init, 0.01, 1.0)  # Much smaller range
        beta_init = np.clip(beta_init, 0.01, 1.0)
        
        # Conservative Gamma parameterization: E[X] = a/b, Var[X] = a/b^2
        # Want E[θ] ≈ theta_init, with reasonable variance
        variance_scale = 1.0  # Higher variance for more flexibility
        
        self.a_theta = (theta_init**2) / (variance_scale * theta_init) + self.alpha_theta
        self.b_theta = theta_init / (variance_scale * theta_init) + 0.01
        
        self.a_beta = (beta_init**2) / (variance_scale * beta_init) + self.alpha_beta
        self.b_beta = beta_init / (variance_scale * beta_init) + 0.01
        
        self.a_xi = self.alpha_xi + 0.1 * self.rng.exponential(size=self.n)
        self.b_xi = np.ones(self.n) * self.lambda_xi
        
        self.a_eta = self.alpha_eta + 0.1 * self.rng.exponential(size=self.p)
        self.b_eta = np.ones(self.p) * self.lambda_eta
        
        self.mu_v = 0.01 * self.rng.randn(self.kappa, self.d)
        self.Sigma_v = np.tile(np.eye(self.d)[np.newaxis, :, :], (self.kappa, 1, 1))
        
        self.mu_gamma = 0.01 * self.rng.randn(self.kappa, self.p_aux)
        self.Sigma_gamma = np.tile(np.eye(self.p_aux)[np.newaxis, :, :], (self.kappa, 1, 1))
        
        self.zeta = np.ones((self.n, self.kappa)) * 0.1
        
        # CRITICAL: Clip immediately after initialization to prevent negative entropy
        self.a_theta = np.clip(self.a_theta, 1.01, 1e6)
        self.b_theta = np.clip(self.b_theta, 1e-6, 1e6)
        self.a_beta = np.clip(self.a_beta, 1.01, 1e6)
        self.b_beta = np.clip(self.b_beta, 1e-6, 1e6)
        self.a_xi = np.clip(self.a_xi, 1.01, 1e6)
        self.b_xi = np.clip(self.b_xi, 1e-6, 1e6)
        self.a_eta = np.clip(self.a_eta, 1.01, 1e6)
        self.b_eta = np.clip(self.b_eta, 1e-6, 1e6)
        
    def _compute_expectations(self):
        """Compute all needed expectations."""
        self.E_theta = self.a_theta / self.b_theta
        self.E_log_theta = digamma(self.a_theta) - np.log(self.b_theta)
        
        self.E_beta = self.a_beta / self.b_beta
        self.E_log_beta = digamma(self.a_beta) - np.log(self.b_beta)
        
        self.E_xi = self.a_xi / self.b_xi
        self.E_log_xi = digamma(self.a_xi) - np.log(self.b_xi)
        
        self.E_eta = self.a_eta / self.b_eta
        self.E_log_eta = digamma(self.a_eta) - np.log(self.b_eta)
        
        self.E_v = self.mu_v
        self.E_gamma = self.mu_gamma
        
    def _allocate_counts_sparse(self, X):
        """MEMORY-EFFICIENT count allocation - only store non-zero entries."""
        # Pre-compute log probabilities only for observed entries
        nz_entries = []
        log_phi_nz = []
        
        if hasattr(X, 'indptr'):  # Sparse CSR matrix
            for i in range(self.n):
                start_idx = X.indptr[i]
                end_idx = X.indptr[i + 1]
                if end_idx > start_idx:
                    cols = X.indices[start_idx:end_idx]
                    vals = X.data[start_idx:end_idx]
                    
                    # Compute log probabilities for this row's non-zero entries
                    log_phi_row = (self.E_log_theta[i, np.newaxis, :] + 
                                  self.E_log_beta[cols, :])  # (nnz_row, d)
                    
                    # Normalize
                    log_phi_row_norm = log_phi_row - logsumexp(log_phi_row, axis=1, keepdims=True)
                    
                    # Store entries and probabilities
                    for idx, (j, x_ij) in enumerate(zip(cols, vals)):
                        nz_entries.append((i, j, x_ij))
                        log_phi_nz.append(log_phi_row_norm[idx])
        else:  # Dense matrix
            nz_i, nz_j = np.nonzero(X)
            for i, j in zip(nz_i, nz_j):
                x_ij = X[i, j]
                log_phi_ij = (self.E_log_theta[i, :] + self.E_log_beta[j, :])
                log_phi_ij_norm = log_phi_ij - logsumexp(log_phi_ij)
                nz_entries.append((i, j, x_ij))
                log_phi_nz.append(log_phi_ij_norm)
        
        return nz_entries, np.array(log_phi_nz)
    
    def _compute_factor_sums_sparse(self, nz_entries, log_phi_nz):
        """Compute factor sums efficiently from sparse representation."""
        # Initialize sums
        z_sum_rows = np.zeros((self.n, self.d))  # Sum over columns for each row
        z_sum_cols = np.zeros((self.p, self.d))  # Sum over rows for each column
        
        # Compute sums from non-zero entries
        phi_nz = np.exp(log_phi_nz)
        
        # CRITICAL CHECK: phi should be probabilities (sum to 1 for each entry)
        if len(phi_nz) > 0:
            phi_sums = phi_nz.sum(axis=1) if phi_nz.ndim > 1 else np.array([phi_nz.sum()])
            if not np.allclose(phi_sums, 1.0, atol=1e-6):
                print(f"WARNING: phi not normalized! Sample sums: {phi_sums[:5]}")
        
        for idx, (i, j, x_ij) in enumerate(nz_entries):
            z_ijk = x_ij * phi_nz[idx]  # (d,) vector
            z_sum_rows[i] += z_ijk
            z_sum_cols[j] += z_ijk
        
        return z_sum_rows, z_sum_cols
    
    def _update_theta_sparse(self, z_sum_rows: np.ndarray, y: np.ndarray, X_aux: np.ndarray):
        """Memory-efficient theta update using sparse sums."""
        E_theta_prev = self.E_theta.copy()
        
        # Shape update - use pre-computed sums
        self.a_theta = self.alpha_theta + z_sum_rows
        self.a_theta = np.clip(self.a_theta, 1.01, 1e6)  # Clip immediately
        
        # Rate base with clipping
        E_xi_safe = np.clip(self.E_xi, 1e-10, 1e6)
        E_beta_sum = np.clip(np.sum(self.E_beta, axis=0), 1e-10, 1e6)
        self.b_theta = E_xi_safe[:, np.newaxis] + E_beta_sum[np.newaxis, :]
        
        # Add regression contribution with careful numerical handling
        if self.regression_weight > 0:
            for k in range(self.kappa):
                y_k = y[:, k] if y.ndim > 1 else y
                lam = lambda_jj_numba(self.zeta[:, k])
                lam = np.clip(lam, 0, 1.0)  # Bound lambda
                
                # Vectorized computation for all factors
                for ell in range(self.d):
                    # Compute E_C without current factor
                    mask = np.ones(self.d, dtype=bool)
                    mask[ell] = False
                    E_C = self.E_theta[:, mask] @ self.E_v[k, mask]
                    if X_aux.shape[1] > 0:
                        E_C += X_aux @ self.E_gamma[k]
                    
                    E_v_ell = np.clip(self.E_v[k, ell], -10, 10)
                    E_v_sq_ell = self.mu_v[k, ell]**2 + self.Sigma_v[k, ell, ell]
                    E_v_sq_ell = np.clip(E_v_sq_ell, 0, 100)
                    
                    regression_contrib = (
                        -(y_k - 0.5) * E_v_ell
                        + 2 * lam * E_v_ell * E_C
                        + 2 * lam * E_theta_prev[:, ell] * E_v_sq_ell
                    )
                    regression_contrib = np.clip(regression_contrib, -1e3, 1e3)
                    
                    self.b_theta[:, ell] += self.regression_weight * regression_contrib
        
        # Final clipping - AGGRESSIVE to prevent explosion
        self.b_theta = np.clip(self.b_theta, 1e-3, 1e3)
        self.a_theta = np.clip(self.a_theta, 1.01, 1e3)
    
    def _update_theta(self, z: np.ndarray, y: np.ndarray, X_aux: np.ndarray):
        """Update theta using coordinate ascent."""
        self._update_theta_vectorized(z, y, X_aux)
    
    def _update_beta_sparse(self, z_sum_cols: np.ndarray):
        """Memory-efficient beta update using sparse sums."""
        self.a_beta = self.alpha_beta + z_sum_cols
        self.a_beta = np.clip(self.a_beta, 1.01, 1e6)
        
        E_eta_safe = np.clip(self.E_eta, 1e-10, 1e6)
        E_theta_sum = np.clip(np.sum(self.E_theta, axis=0), 1e-10, 1e6)
        self.b_beta = E_eta_safe[:, np.newaxis] + E_theta_sum[np.newaxis, :]
        
        self.b_beta = np.clip(self.b_beta, 1e-3, 1e3)
        self.a_beta = np.clip(self.a_beta, 1.01, 1e3)
    
    def _update_xi(self):
        """Update xi using coordinate ascent with numerical safeguards."""
        self.a_xi = np.full(self.n, self.alpha_xi + self.d * self.alpha_theta)
        self.a_xi = np.clip(self.a_xi, 1.01, 1e6)
        
        E_theta_sum = np.clip(np.sum(self.E_theta, axis=1), 1e-10, 1e6)
        self.b_xi = self.lambda_xi + E_theta_sum
        self.b_xi = np.clip(self.b_xi, 1e-6, 1e6)
    
    def _update_eta(self):
        """Update eta using coordinate ascent with numerical safeguards."""
        self.a_eta = np.full(self.p, self.alpha_eta + self.d * self.alpha_beta)
        self.a_eta = np.clip(self.a_eta, 1.01, 1e6)
        
        E_beta_sum = np.clip(np.sum(self.E_beta, axis=1), 1e-10, 1e6)
        self.b_eta = self.lambda_eta + E_beta_sum
        self.b_eta = np.clip(self.b_eta, 1e-6, 1e6)
    
    def _update_v(self, y: np.ndarray, X_aux: np.ndarray):
        """Vectorized v update with numerical safeguards."""
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = lambda_jj_numba(self.zeta[:, k])
            lam = np.clip(lam, 0, 1.0)
            
            # Vectorized precision computation with safeguards
            E_theta_sq = self.E_theta**2 + self.a_theta / (self.b_theta**2)
            E_theta_sq = np.clip(E_theta_sq, 0, 1e3)
            
            prec = (1.0 / self.sigma_v**2) * np.eye(self.d)
            prec += 2 * np.diag((lam[:, np.newaxis] * E_theta_sq).sum(axis=0))
            prec = 0.5 * (prec + prec.T)  # Ensure symmetry
            
            # Vectorized mean computation with safeguards
            mean_contrib = ((y_k - 0.5)[:, np.newaxis] * self.E_theta).sum(axis=0)
            if X_aux.shape[1] > 0:
                X_aux_gamma = X_aux @ self.E_gamma[k]
                X_aux_gamma = np.clip(X_aux_gamma, -100, 100)
                mean_contrib -= (2 * lam[:, np.newaxis] * self.E_theta * X_aux_gamma[:, np.newaxis]).sum(axis=0)
            
            mean_contrib = np.clip(mean_contrib, -1e3, 1e3)
            
            try:
                # Add small diagonal for numerical stability
                prec_reg = prec + 1e-6 * np.eye(self.d)
                self.Sigma_v[k] = np.linalg.inv(prec_reg)
                self.mu_v[k] = self.Sigma_v[k] @ mean_contrib
                self.mu_v[k] = np.clip(self.mu_v[k], -10, 10)
            except np.linalg.LinAlgError:
                # If inversion fails, use previous values
                pass
    
    def _update_gamma(self, y: np.ndarray, X_aux: np.ndarray):
        """Vectorized gamma update with numerical safeguards."""
        if X_aux.shape[1] == 0:
            return
            
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = lambda_jj_numba(self.zeta[:, k])
            lam = np.clip(lam, 0, 1.0)
            
            # Vectorized precision with safeguards
            prec = (1.0 / self.sigma_gamma**2) * np.eye(self.p_aux)
            X_aux_T_lam_X = X_aux.T @ (lam[:, np.newaxis] * X_aux)
            prec += 2 * X_aux_T_lam_X
            prec = 0.5 * (prec + prec.T)  # Ensure symmetry
            
            # Vectorized mean with safeguards
            mean_contrib = X_aux.T @ (y_k - 0.5)
            theta_v = self.E_theta @ self.E_v[k]
            theta_v = np.clip(theta_v, -100, 100)
            mean_contrib -= 2 * X_aux.T @ (lam * theta_v)
            
            mean_contrib = np.clip(mean_contrib, -1e3, 1e3)
            
            try:
                # Add small diagonal for numerical stability
                prec_reg = prec + 1e-6 * np.eye(self.p_aux)
                self.Sigma_gamma[k] = np.linalg.inv(prec_reg)
                self.mu_gamma[k] = self.Sigma_gamma[k] @ mean_contrib
                self.mu_gamma[k] = np.clip(self.mu_gamma[k], -10, 10)
            except np.linalg.LinAlgError:
                # If inversion fails, use previous values
                pass
    
    def _update_zeta(self, y: np.ndarray, X_aux: np.ndarray):
        """Vectorized zeta update with numerical safeguards."""
        for k in range(self.kappa):
            # Vectorized computation
            E_A = self.E_theta @ self.E_v[k]
            if X_aux.shape[1] > 0:
                E_A += X_aux @ self.E_gamma[k]
            E_A = np.clip(E_A, -100, 100)
            E_A_sq = E_A**2
            
            E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
            E_v_sq = np.clip(E_v_sq, 0, 100)
            Var_theta = self.a_theta / (self.b_theta**2)
            Var_theta = np.clip(Var_theta, 0, 100)
            E_A_sq += (Var_theta * E_v_sq[np.newaxis, :]).sum(axis=1)
            
            self.zeta[:, k] = np.sqrt(np.clip(E_A_sq, 1e-10, 1e4))
    
    def _compute_elbo(self, X, y: np.ndarray, X_aux: np.ndarray, 
                     debug: bool = False, iteration: int = 0) -> float:
        """Compute ELBO with numerical safeguards for large sparse matrices."""
        elbo = 0.0
        elbo_components = {}
        
        # E[log p(X | θ, β)] - OPTIMIZED Poisson likelihood computation
        elbo_x = 0.0
        sample_rates = []
        sample_counts = []
        
        # Single pass through sparse entries - compute both likelihood and debug info
        if hasattr(X, 'indptr'):  # Sparse matrix
            for i in range(self.n):
                start_idx = X.indptr[i]
                end_idx = X.indptr[i + 1]
                if end_idx > start_idx:
                    cols = X.indices[start_idx:end_idx]
                    x_vals = X.data[start_idx:end_idx]
                    
                    # Vectorized rate computation for this row
                    rates_row = np.sum(self.E_theta[i, np.newaxis, :] * self.E_beta[cols, :], axis=1)
                    rates_row = np.clip(rates_row, 1e-10, 1e6)
                    
                    # Vectorized likelihood computation
                    mask_pos = x_vals > 0
                    if np.any(mask_pos):
                        x_pos = x_vals[mask_pos]
                        rates_pos = rates_row[mask_pos]
                        
                        elbo_x += np.sum(x_pos * np.log(rates_pos) - rates_pos - gammaln(x_pos + 1))
                        
                        # Collect debug samples (first iteration only)
                        # if debug and iteration == 1 and len(sample_rates) < 20:
                        #     sample_rates.extend(rates_pos[:min(5, len(rates_pos))])
                        #     sample_counts.extend(x_pos[:min(5, len(x_pos))])
        else:  # Dense matrix
            nz_i, nz_j = np.nonzero(X)
            if len(nz_i) > 0:
                # Vectorized computation for all non-zero entries
                rates_nz = np.sum(self.E_theta[nz_i, :] * self.E_beta[nz_j, :], axis=1)
                rates_nz = np.clip(rates_nz, 1e-10, 1e6)
                x_nz = X[nz_i, nz_j]
                
                elbo_x += np.sum(x_nz * np.log(rates_nz) - rates_nz - gammaln(x_nz + 1))
                
                # if debug and iteration == 1:
                #     sample_rates = rates_nz[:20].tolist()
                #     sample_counts = x_nz[:20].tolist()
        
        # CRITICAL: Subtract sum of all rates (Poisson normalization term)
        # This is: -Σ_u Σ_i θ_u^T β_i = -Σ_k (Σ_u θ_uk)(Σ_i β_ik)
        theta_sum = np.sum(self.E_theta, axis=0)  # shape: (d,)
        beta_sum = np.sum(self.E_beta, axis=0)     # shape: (d,)
        total_rate = np.dot(theta_sum, beta_sum)
        elbo_x -= total_rate
        
        # if debug and (iteration == 1 or iteration == 6):
        #     print(f"    [POISSON DEBUG] Total rate normalization: -{total_rate:.2f}")
        #     print(f"    [POISSON DEBUG] theta_sum: {theta_sum}")
        #     print(f"    [POISSON DEBUG] beta_sum: {beta_sum}")
        
        elbo += elbo_x
        elbo_components['E[log p(X|theta,beta)]'] = elbo_x
        
        # Debug info - show what's happening with rates and parameters
        # if debug and (iteration == 1 or iteration == 6):
        #     if sample_rates and sample_counts:
        #         print(f"    Sample observed counts: {sample_counts[:10]}")
        #         print(f"    Sample predicted rates: {[f'{r:.2f}' for r in sample_rates[:10]]}")
        #         print(f"    Mean rate: {np.mean(sample_rates):.2f}, Mean count: {np.mean(sample_counts):.2f}")
        #         total_nonzeros = X.nnz if hasattr(X, 'nnz') else np.count_nonzero(X)
        #         print(f"    Total non-zero entries: {total_nonzeros:,}")
        #         print(f"    Approx contribution per entry: {elbo_x / total_nonzeros:.2f}")
            
        #     # Check parameter ranges
        #     print(f"    E[theta] range: [{self.E_theta.min():.6f}, {self.E_theta.max():.6f}]")
        #     print(f"    E[beta] range: [{self.E_beta.min():.6f}, {self.E_beta.max():.6f}]")
        #     print(f"    E[xi] range: [{self.E_xi.min():.6f}, {self.E_xi.max():.6f}]")
        #     print(f"    E[eta] range: [{self.E_eta.min():.6f}, {self.E_eta.max():.6f}]")
            
        #     # CRITICAL DEBUG: Check Gamma parameter ranges
        #     print(f"    a_theta range: [{self.a_theta.min():.6f}, {self.a_theta.max():.6f}]")
        #     print(f"    b_theta range: [{self.b_theta.min():.6f}, {self.b_theta.max():.6f}]")
        #     print(f"    a_beta range: [{self.a_beta.min():.6f}, {self.a_beta.max():.6f}]")
        #     print(f"    b_beta range: [{self.b_beta.min():.6f}, {self.b_beta.max():.6f}]")
            
        #     # Check if Gamma parameters make sense: E[X] = a/b
        #     theta_check = np.allclose(self.E_theta, self.a_theta / self.b_theta)
        #     beta_check = np.allclose(self.E_beta, self.a_beta / self.b_beta)
        #     print(f"    Gamma consistency: theta={theta_check}, beta={beta_check}")
            
        #     # Sample a few manual calculations
        #     print(f"    Sample E[theta][0,0] = {self.E_theta[0,0]:.6f} vs a/b = {self.a_theta[0,0]/self.b_theta[0,0]:.6f}")
        #     print(f"    Sample E[beta][0,0] = {self.E_beta[0,0]:.6f} vs a/b = {self.a_beta[0,0]/self.b_beta[0,0]:.6f}")
            
        #     # Check if parameters are exploding
        #     if np.any(self.E_theta > 1e6) or np.any(self.E_beta > 1e6):
        #         print(f"    ⚠️  WARNING: Parameters are exploding!")
            
        #     # Sanity check: compute a few rates manually
        #     if hasattr(X, 'indptr') and X.indptr[1] > X.indptr[0]:
        #         i, j = 0, X.indices[X.indptr[0]]
        #         x_ij = X.data[X.indptr[0]]
        #         rate_manual = np.sum(self.E_theta[i] * self.E_beta[j])
        #         print(f"    Manual check: x[{i},{j}]={x_ij:.1f}, rate={rate_manual:.4f}, ll_contrib={x_ij * np.log(rate_manual) - rate_manual - gammaln(x_ij + 1):.2f}")
        
        # E[log p(y | θ, v, γ)] - vectorized with safeguards
        elbo_y = 0.0
        for k in range(self.kappa):
            y_k = y[:, k] if y.ndim > 1 else y
            lam = lambda_jj_numba(self.zeta[:, k])
            lam = np.clip(lam, 0, 1.0)
            
            E_A = self.E_theta @ self.E_v[k]
            if X_aux.shape[1] > 0:
                E_A += X_aux @ self.E_gamma[k]
            E_A = np.clip(E_A, -100, 100)
            
            # Vectorized computation of E[A^2]
            E_v_sq = self.mu_v[k]**2 + np.diag(self.Sigma_v[k])
            E_v_sq = np.clip(E_v_sq, 0, 100)
            Var_theta = self.a_theta / (self.b_theta**2)
            Var_theta = np.clip(Var_theta, 0, 100)
            E_A_sq = E_A**2 + np.sum(Var_theta * E_v_sq[np.newaxis, :], axis=1)
            E_A_sq = np.clip(E_A_sq, 0, 1e4)
            
            elbo_y += np.sum((y_k - 0.5) * E_A - lam * E_A_sq)
        
        elbo += elbo_y
        elbo_components['E[log p(y|theta,v,gamma)]'] = elbo_y
        
        # Priors - with corrected formula for scalars
        # E[log p(θ, ξ)] - CORRECTED FORMULA
        elbo_theta_xi = (
            # p(ξ) terms: (α_ξ-1)*E[log ξ] - λ_ξ*E[ξ] + normalizing constant
            np.sum((self.alpha_xi - 1) * self.E_log_xi) -
            np.sum(self.lambda_xi * self.E_xi) +
            self.n * (self.alpha_xi * np.log(self.lambda_xi) - gammaln(self.alpha_xi)) +
            # p(θ|ξ) terms: (α_θ-1)*E[log θ] + α_θ*E[log ξ] - E[ξ]*E[θ] - normalizing constant
            np.sum((self.alpha_theta - 1) * self.E_log_theta) +
            self.d * self.alpha_theta * np.sum(self.E_log_xi) -
            np.sum(self.E_xi[:, np.newaxis] * self.E_theta) -
            self.n * self.d * gammaln(self.alpha_theta)
        )
        
        # DEBUG: Break down the theta/xi prior computation
        # if debug and (iteration == 1 or iteration == 6):
        #     xi_prior = (np.sum((self.alpha_xi - 1) * self.E_log_xi) - 
        #                np.sum(self.lambda_xi * self.E_xi) +
        #                self.n * (self.alpha_xi * np.log(self.lambda_xi) - gammaln(self.alpha_xi)))
        #     theta_given_xi = (np.sum((self.alpha_theta - 1) * self.E_log_theta) +
        #                      np.sum(self.alpha_theta * self.E_log_xi[:, np.newaxis]) -
        #                      np.sum(self.E_xi[:, np.newaxis] * self.E_theta) -
        #                      self.n * self.d * gammaln(self.alpha_theta))
        #     print(f"    [PRIOR DEBUG] p(ξ): {xi_prior:.2f}, p(θ|ξ): {theta_given_xi:.2f}")
        #     print(f"    [PRIOR DEBUG] E[log ξ] range: [{self.E_log_xi.min():.4f}, {self.E_log_xi.max():.4f}]")
        #     print(f"    [PRIOR DEBUG] E[log θ] range: [{self.E_log_theta.min():.4f}, {self.E_log_theta.max():.4f}]")
            
        #     # DETAILED BREAKDOWN of p(θ|ξ) terms
        #     term1 = np.sum((self.alpha_theta - 1) * self.E_log_theta)
        #     term2 = np.sum(self.alpha_theta * self.E_log_xi[:, np.newaxis]) 
        #     term3 = -np.sum(self.E_xi[:, np.newaxis] * self.E_theta)
        #     term4 = -self.n * self.d * gammaln(self.alpha_theta)
        #     print(f"    [PRIOR BREAKDOWN] (α_θ-1)*E[log θ]: {term1:.2f}")
        #     print(f"    [PRIOR BREAKDOWN] α_θ*E[log ξ]: {term2:.2f}")  
        #     print(f"    [PRIOR BREAKDOWN] -E[ξ]*E[θ]: {term3:.2f}")
        #     print(f"    [PRIOR BREAKDOWN] -n*d*Γ(α_θ): {term4:.2f}")
        #     print(f"    [PRIOR CHECK] α_θ={self.alpha_theta}, n={self.n}, d={self.d}")
            
        #     # CRITICAL DEBUG: Check the math manually
        #     print(f"    [MATH CHECK] (α_θ-1) = {self.alpha_theta - 1}")
        #     print(f"    [MATH CHECK] E[log θ] sample: {self.E_log_theta[0, :5]}")
        #     print(f"    [MATH CHECK] Should be negative: {(self.alpha_theta - 1) * self.E_log_theta[0, 0]:.6f}")
        #     print(f"    [MATH CHECK] Total θ terms: {self.n * self.d}")
        #     print(f"    [MATH CHECK] Expected magnitude: ~{(self.alpha_theta - 1) * np.mean(self.E_log_theta) * self.n * self.d:.2f}")
        elbo += elbo_theta_xi
        elbo_components['E[log p(theta,xi)]'] = elbo_theta_xi
        
        # E[log p(β, η)] - CORRECTED FORMULA  
        elbo_beta_eta = (
            # p(η) terms
            np.sum((self.alpha_eta - 1) * self.E_log_eta) -
            np.sum(self.lambda_eta * self.E_eta) +
            self.p * (self.alpha_eta * np.log(self.lambda_eta) - gammaln(self.alpha_eta)) +
            # p(β|η) terms
            np.sum((self.alpha_beta - 1) * self.E_log_beta) +
            self.d * self.alpha_beta * np.sum(self.E_log_eta)  -
            np.sum(self.E_eta[:, np.newaxis] * self.E_beta) -
            self.p * self.d * gammaln(self.alpha_beta)
        )
        
        # DEBUG: Break down the beta/eta prior computation
        # if debug and (iteration == 1 or iteration == 6):
        #     eta_prior = (np.sum((self.alpha_eta - 1) * self.E_log_eta) - 
        #                 np.sum(self.lambda_eta * self.E_eta) +
        #                 self.p * (self.alpha_eta * np.log(self.lambda_eta) - gammaln(self.alpha_eta)))
        #     beta_given_eta = (np.sum((self.alpha_beta - 1) * self.E_log_beta) +
        #                      np.sum(self.alpha_beta * self.E_log_eta[:, np.newaxis]) -
        #                      np.sum(self.E_eta[:, np.newaxis] * self.E_beta) -
        #                      self.p * self.d * gammaln(self.alpha_beta))
        #     print(f"    [PRIOR DEBUG] p(η): {eta_prior:.2f}, p(β|η): {beta_given_eta:.2f}")
        #     print(f"    [PRIOR DEBUG] E[log η] range: [{self.E_log_eta.min():.4f}, {self.E_log_eta.max():.4f}]")
        #     print(f"    [PRIOR DEBUG] E[log β] range: [{self.E_log_beta.min():.4f}, {self.E_log_beta.max():.4f}]")
        elbo += elbo_beta_eta
        elbo_components['E[log p(beta,eta)]'] = elbo_beta_eta
        
        # Priors on v, gamma - vectorized
        Sigma_v_traces = np.array([np.trace(self.Sigma_v[k]) for k in range(self.kappa)])
        elbo_v_gamma = (-0.5 * self.kappa * self.d * np.log(2 * np.pi * self.sigma_v**2) -
                       0.5 * (np.sum(self.mu_v**2) + np.sum(Sigma_v_traces)) / self.sigma_v**2)
        
        if X_aux.shape[1] > 0:
            Sigma_gamma_traces = np.array([np.trace(self.Sigma_gamma[k]) for k in range(self.kappa)])
            elbo_v_gamma += (-0.5 * self.kappa * self.p_aux * np.log(2 * np.pi * self.sigma_gamma**2) -
                           0.5 * (np.sum(self.mu_gamma**2) + np.sum(Sigma_gamma_traces)) / self.sigma_gamma**2)
        
        elbo += elbo_v_gamma
        elbo_components['E[log p(v,gamma)]'] = elbo_v_gamma
        
        # Entropy terms - CORRECT FORMULA for Gamma(a,b) distributions
        # H[q(θ)] = sum_i sum_k [a_ik - log(b_ik) + gammaln(a_ik) + (1-a_ik)*digamma(a_ik)]
        a_safe = np.clip(self.a_theta, 1.01, 1e6)
        b_safe = np.clip(self.b_theta, 1e-6, 1e6)
        entropy_theta = (np.sum(a_safe - np.log(b_safe)) +
                        np.sum(gammaln(a_safe)) +
                        np.sum((1 - a_safe) * digamma(a_safe)))
        elbo += entropy_theta
        elbo_components['H[q(theta)]'] = entropy_theta
        
        a_safe = np.clip(self.a_beta, 1.01, 1e6)
        b_safe = np.clip(self.b_beta, 1e-6, 1e6)
        entropy_beta = (np.sum(a_safe - np.log(b_safe)) +
                       np.sum(gammaln(a_safe)) +
                       np.sum((1 - a_safe) * digamma(a_safe)))
        elbo += entropy_beta
        elbo_components['H[q(beta)]'] = entropy_beta
        
        a_safe = np.clip(self.a_xi, 1.01, 1e6)
        b_safe = np.clip(self.b_xi, 1e-6, 1e6)
        entropy_xi = (np.sum(a_safe - np.log(b_safe)) +
                     np.sum(gammaln(a_safe)) +
                     np.sum((1 - a_safe) * digamma(a_safe)))
        elbo += entropy_xi
        elbo_components['H[q(xi)]'] = entropy_xi
        
        a_safe = np.clip(self.a_eta, 1.01, 1e6)
        b_safe = np.clip(self.b_eta, 1e-6, 1e6)
        entropy_eta = (np.sum(a_safe - np.log(b_safe)) +
                      np.sum(gammaln(a_safe)) +
                      np.sum((1 - a_safe) * digamma(a_safe)))
        elbo += entropy_eta
        elbo_components['H[q(eta)]'] = entropy_eta
        
        # Compute matrix entropies safely
        entropy_v = 0.0
        for k in range(self.kappa):
            sign, logdet = np.linalg.slogdet(self.Sigma_v[k])
            if sign > 0 and np.isfinite(logdet) and logdet < 100:
                entropy_v += 0.5 * (self.d * (1 + np.log(2 * np.pi)) + logdet)
        elbo += entropy_v
        elbo_components['H[q(v)]'] = entropy_v
        
        entropy_gamma = 0.0
        if X_aux.shape[1] > 0:
            for k in range(self.kappa):
                sign, logdet = np.linalg.slogdet(self.Sigma_gamma[k])
                if sign > 0 and np.isfinite(logdet) and logdet < 100:
                    entropy_gamma += 0.5 * (self.p_aux * (1 + np.log(2 * np.pi)) + logdet)
        elbo += entropy_gamma
        elbo_components['H[q(gamma)]'] = entropy_gamma
        
        # CRITICAL ENTROPY DEBUG - Check why entropy values are astronomically large
        if debug:
            print(f"\n  === Iteration {iteration} ELBO Breakdown ===")
            for name, value in elbo_components.items():
                status = "✓" if np.isfinite(value) and (not name.startswith('H[') or value >= 0) else "✗"
                print(f"    {status} {name:30s}: {value:12.2f}")
            print(f"    {'='*44}")
            print(f"    {'Total ELBO':30s}: {elbo:12.2f}")
        
        self.last_elbo_components_ = elbo_components
        
        if not np.isfinite(elbo):
            return -np.inf

        return elbo
    
    def fit(self, X, y: np.ndarray, X_aux: np.ndarray,
            max_iter: int = 100, tol: float = 10.0, rel_tol: float = 2e-4,
            elbo_freq: int = 1, min_iter: int = 30, patience: int = 3,
            verbose: bool = True, theta_damping: float = 0.5,
            beta_damping: float = 0.7, v_damping: float = 0.6,
            gamma_damping: float = 0.6, xi_damping: float = 0.8,
            eta_damping: float = 0.8, adaptive_damping: bool = True,
            debug: bool = False):
        """Fit with optimized updates for sparse or dense X."""
        import time
        
        start_time = time.time()
        if y.ndim == 1:
            y = y[:, np.newaxis]
        
        self._initialize_parameters(X, y, X_aux)
        elbo_history = []
        patience_counter = 0
        
        # MODERATE damping to prevent oscillations while preserving monotonicity
        # Use careful damping that helps stability without breaking convergence
        damping_factors = {
            'theta': min(theta_damping, 0.7), 'beta': min(beta_damping, 0.7), 
            'v': min(v_damping, 0.8), 'gamma': min(gamma_damping, 0.8), 
            'xi': min(xi_damping, 0.8), 'eta': min(eta_damping, 0.8)
        }
        
        for iteration in range(max_iter):
            # CRITICAL: Compute expectations with current parameters
            self._compute_expectations()
            
            # MEMORY-EFFICIENT sparse count allocation
            nz_entries, log_phi_nz = self._allocate_counts_sparse(X)
            z_sum_rows, z_sum_cols = self._compute_factor_sums_sparse(nz_entries, log_phi_nz)
            
            # DEBUG: Check count sums
            # if debug and (iteration == 0 or iteration == 5):
            #     print(f"\n    [DEBUG Iter {iteration+1}] z_sum_rows range: [{z_sum_rows.min():.4f}, {z_sum_rows.max():.4f}]")
            #     print(f"    [DEBUG Iter {iteration+1}] z_sum_cols range: [{z_sum_cols.min():.4f}, {z_sum_cols.max():.4f}]")
            #     print(f"    [DEBUG Iter {iteration+1}] Total allocated: {z_sum_rows.sum():.2f}, should be ≈ total counts: {sum(x for _, _, x in nz_entries):.2f}")
            
            # MONOTONICITY DEBUG: Track ELBO before parameter updates
            # if debug and iteration >= 2:
            #     elbo_before_updates = self._compute_elbo(X, y, X_aux, debug=False, iteration=iteration+1)
            #     print(f"\n    [MONO DEBUG] ELBO before parameter updates: {elbo_before_updates:.2f}")
            
            # Apply MODERATE damping to prevent oscillations
            # Even with coordinate ascent, the mean-field approximation in regression
            # requires damping for stability
            damping_factors = {
                'theta': 0.5, 'beta': 0.5,  # More aggressive damping
                'xi': 0.8, 'eta': 0.8,
                'v': 0.7, 'gamma': 0.7  # Add missing v and gamma damping
            }

            # Update theta with damping
            a_theta_old, b_theta_old = self.a_theta.copy(), self.b_theta.copy()
            self._update_theta_sparse(z_sum_rows, y, X_aux)
            damp = damping_factors['theta']
            self.a_theta = damp * self.a_theta + (1 - damp) * a_theta_old
            self.b_theta = damp * self.b_theta + (1 - damp) * b_theta_old

            # Recompute expectations after theta update
            self._compute_expectations()
            
            # MONOTONICITY DEBUG: Check ELBO after theta update
            # if debug and iteration >= 2:
            #     elbo_after_theta = self._compute_elbo(X, y, X_aux, debug=False, iteration=iteration+1)
            #     print(f"    [MONO DEBUG] ELBO after theta update: {elbo_after_theta:.2f} (change: {elbo_after_theta - elbo_before_updates:.2f})")

            # Update beta with damping
            a_beta_old, b_beta_old = self.a_beta.copy(), self.b_beta.copy()
            self._update_beta_sparse(z_sum_cols)
            damp = damping_factors['beta']
            self.a_beta = damp * self.a_beta + (1 - damp) * a_beta_old
            self.b_beta = damp * self.b_beta + (1 - damp) * b_beta_old

            # Recompute expectations after beta update
            self._compute_expectations()
            
            # MONOTONICITY DEBUG: Check ELBO after beta update
            # if debug and iteration >= 2:
            #     elbo_after_beta = self._compute_elbo(X, y, X_aux, debug=False, iteration=iteration+1)
            #     print(f"    [MONO DEBUG] ELBO after beta update: {elbo_after_beta:.2f} (change: {elbo_after_beta - elbo_after_theta:.2f})")

            # Update xi with damping
            a_xi_old, b_xi_old = self.a_xi.copy(), self.b_xi.copy()
            self._update_xi()
            damp = damping_factors['xi']
            self.a_xi = damp * self.a_xi + (1 - damp) * a_xi_old
            self.b_xi = damp * self.b_xi + (1 - damp) * b_xi_old

            # Update eta with damping
            a_eta_old, b_eta_old = self.a_eta.copy(), self.b_eta.copy()
            self._update_eta()
            damp = damping_factors['eta']
            self.a_eta = damp * self.a_eta + (1 - damp) * a_eta_old
            self.b_eta = damp * self.b_eta + (1 - damp) * b_eta_old
            
            # Damped updates for v, gamma
            mu_v_old = self.mu_v.copy()
            Sigma_v_old = self.Sigma_v.copy()
            self._update_v(y, X_aux)
            damp = damping_factors['v']
            self.mu_v = damp * self.mu_v + (1 - damp) * mu_v_old
            self.Sigma_v = damp * self.Sigma_v + (1 - damp) * Sigma_v_old
            
            mu_gamma_old = self.mu_gamma.copy()
            Sigma_gamma_old = self.Sigma_gamma.copy()
            self._update_gamma(y, X_aux)
            damp = damping_factors['gamma']
            self.mu_gamma = damp * self.mu_gamma + (1 - damp) * mu_gamma_old
            self.Sigma_gamma = damp * self.Sigma_gamma + (1 - damp) * Sigma_gamma_old
            
            self._update_zeta(y, X_aux)
            
            # CRITICAL FIX: Recompute expectations after all parameter updates
            self._compute_expectations()
            
            compute_elbo = (iteration % elbo_freq == 0 or iteration == 0 or iteration == max_iter - 1)
            
            if compute_elbo:
                elbo = self._compute_elbo(X, y, X_aux, debug=debug, iteration=iteration+1)
                elbo_history.append((iteration, elbo))
            
            if adaptive_damping and len(elbo_history) > 1:
                _, elbo_curr = elbo_history[-1]
                _, elbo_prev = elbo_history[-2]
                elbo_change = elbo_curr - elbo_prev
                
                if elbo_change > 0:
                    for key in damping_factors:
                        damping_factors[key] = min(damping_factors[key] * 1.05, 1.0)
                else:
                    for key in damping_factors:
                        damping_factors[key] = max(damping_factors[key] * 0.9, 0.1)
            
            if verbose and compute_elbo:
                _, elbo_curr = elbo_history[-1]
                iter_time = time.time() - start_time
                print(f"Iteration {iteration + 1}/{max_iter}, ELBO: {elbo_curr:.2f} (time: {iter_time:.2f}s)")
                
                if len(elbo_history) > 1:
                    _, elbo_prev = elbo_history[-2]
                    elbo_change = elbo_curr - elbo_prev
                    change_symbol = "↑" if elbo_change > 0 else "↓"
                    rel_change = abs(elbo_change / (abs(elbo_prev) + 1e-10))
                    print(f"  ELBO change: {change_symbol} {elbo_change:.2f} (relative: {rel_change:.6f})")
                    
                    # CRITICAL: Check for monotonicity violation
                    if elbo_change < -1e-6:  # Allow tiny numerical errors
                        print(f"  ⚠️  MONOTONICITY VIOLATION! ELBO decreased by {abs(elbo_change):.2f}")
                        print(f"  🔍 DEBUG: This should NEVER happen in coordinate ascent VI!")
                        
                        # Stop and debug - this indicates a serious bug
                        if iteration > 5:  # Give a few iterations to settle
                            print(f"  🛑 STOPPING TRAINING - Need to debug monotonicity violation!")
                            print(f"  📊 Check parameter updates, entropy calculations, or ELBO computation!")
                            break
            
            if iteration >= min_iter and len(elbo_history) > 1 and compute_elbo:
                _, elbo_curr = elbo_history[-1]
                _, elbo_prev = elbo_history[-2]
                elbo_change = elbo_curr - elbo_prev
                rel_change = abs(elbo_change / (abs(elbo_prev) + 1e-10))
                
                if abs(elbo_change) < tol and rel_change < rel_tol:
                    patience_counter += 1
                else:
                    patience_counter = 0
                
                if patience_counter >= patience:
                    if verbose:
                        total_time = time.time() - start_time
                        print(f"\n{'='*60}")
                        print(f"✓ Converged after {iteration + 1} iterations ({total_time:.2f}s)")
                        print(f"{'='*60}")
                    break
        
        self.elbo_history_ = elbo_history
        self.training_time_ = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {self.training_time_:.2f}s")
            if elbo_history:
                print(f"Final ELBO: {elbo_history[-1][1]:.2f}")
        
        return self
    
    def infer_theta(self, X, max_iter: int = 50,
                   tol: float = 1e-4, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Infer theta for new samples with sparse or dense X - OPTIMIZED."""
        n_new = X.shape[0]
        
        if not hasattr(self, 'E_beta'):
            raise RuntimeError("Model must be fitted before inferring theta")
        
        a_theta_new = np.tile(self.a_theta.mean(axis=0), (n_new, 1))
        b_theta_new = np.tile(self.b_theta.mean(axis=0), (n_new, 1))
        E_xi_mean = self.E_xi.mean()
        
        if verbose:
            print(f"Inferring theta for {n_new} new samples...")
        
        for iteration in range(max_iter):
            a_theta_old = a_theta_new.copy()
            
            E_theta_new = a_theta_new / b_theta_new
            E_log_theta_new = digamma(a_theta_new) - np.log(b_theta_new)
            
            # Compute phi more efficiently
            log_phi = E_log_theta_new[:, np.newaxis, :] + self.E_log_beta[np.newaxis, :, :]
            log_phi_normalized = log_phi - logsumexp(log_phi, axis=2, keepdims=True)
            
            # Sparse/dense compatible count allocation
            z_new = np.zeros((n_new, self.p, self.d))
            if hasattr(X, 'indptr'):  # Sparse CSR matrix
                phi_flat = np.exp(log_phi_normalized.reshape(-1, self.d))
                for i in range(n_new):
                    start_idx = X.indptr[i]
                    end_idx = X.indptr[i + 1]
                    if end_idx > start_idx:
                        cols = X.indices[start_idx:end_idx]
                        x_vals = X.data[start_idx:end_idx]
                        z_new[i, cols, :] = x_vals[:, np.newaxis] * phi_flat[i * self.p + cols]
            else:  # Dense numpy array - vectorized
                nz_i, nz_j = np.nonzero(X)
                if len(nz_i) > 0:
                    phi_nz = np.exp(log_phi_normalized[nz_i, nz_j, :])
                    x_nz = X[nz_i, nz_j]
                    z_new[nz_i, nz_j, :] = x_nz[:, np.newaxis] * phi_nz
            
            a_theta_new = self.alpha_theta + np.sum(z_new, axis=1)
            b_theta_new = E_xi_mean + np.sum(self.E_beta, axis=0)[np.newaxis, :]
            b_theta_new = np.maximum(b_theta_new, 1e-10)
            
            max_change = np.max(np.abs(a_theta_new - a_theta_old))
            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration + 1}/{max_iter}, max_change: {max_change:.6f}")
            
            if max_change < tol:
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break
        
        E_theta_new = a_theta_new / b_theta_new
        return E_theta_new, a_theta_new, b_theta_new
    
    def predict_proba(self, X, X_aux: np.ndarray,
                     max_iter: int = 50, tol: float = 1e-4,
                     verbose: bool = False) -> np.ndarray:
        """Predict probabilities for new samples with sparse or dense X."""
        E_theta_new, _, _ = self.infer_theta(X, max_iter=max_iter, tol=tol, verbose=verbose)
        linear_pred = E_theta_new @ self.E_v.T + X_aux @ self.E_gamma.T
        probs = expit(linear_pred)
        return probs