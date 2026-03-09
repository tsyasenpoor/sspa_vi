"""
Stochastic Variational Inference for Supervised Poisson Factorization (DRGP)
=============================================================================

Global params (β, η, v, γ) updated via SVI (Hoffman et al. 2013).
Local params (θ, ξ, φ) fully optimized per mini-batch.

SVI update rule (Hoffman et al. 2013, Eq 26):
  λ^(t) = (1 - ρ_t) λ^(t-1) + ρ_t * λ̂
where λ̂ is the intermediate parameter computed as if the mini-batch
were the entire dataset (scale = N / batch_size).

Initialization and Poisson factorization core follow scHPF exactly.
Supervision via Jaakkola-Jordan logistic regression bound.

References:
- Hoffman, Blei, Wang, Paisley (2013) "Stochastic Variational Inference"
- Gopalan, Hofman, Blei (2014) "Scalable Recommendation with HPF"
- Levitin et al. (2019) "scHPF", Molecular Systems Biology
"""

import numpy as np
import scipy.sparse as sp
from scipy.special import digamma, gammaln, expit
from scipy.special import logsumexp as sp_logsumexp
from typing import Tuple, Optional, Dict, Any, List
import time


class SVI:
    """
    SVI for Supervised Poisson Factorization.

    Same model as CAVI but with stochastic optimization of global parameters.
    Local parameters (θ, ξ) are fully optimized per mini-batch.
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
        # SVI-specific
        batch_size: int = 128,
        learning_rate_delay: float = 1.0,    # τ
        learning_rate_decay: float = 0.75,   # κ
        local_iterations: int = 10,
        # General
        random_state: Optional[int] = None,
        mode: str = 'unmasked',
        pathway_mask: Optional[np.ndarray] = None,
        pathway_names: Optional[List[str]] = None,
        n_pathway_factors: Optional[int] = None,
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

        self.batch_size = batch_size
        self.tau = learning_rate_delay
        self.kappa = learning_rate_decay
        self.local_iterations = local_iterations

        self.mode = mode
        self.pathway_mask = pathway_mask
        self.pathway_names = pathway_names
        self.n_pathway_factors = n_pathway_factors

        if mode in ['masked', 'pathway_init'] and pathway_mask is None:
            raise ValueError(f"pathway_mask required for mode='{mode}'")
        if mode == 'combined':
            if pathway_mask is None or n_pathway_factors is None:
                raise ValueError("pathway_mask and n_pathway_factors required")

        if random_state is not None:
            np.random.seed(random_state)
            self.seed_used_ = random_state
        else:
            self.seed_used_ = None

        self.n = self.p = self.n_outcomes = self.p_aux = None
        self.bp = self.dp = None

    # =================================================================
    # Helpers (same as CAVI)
    # =================================================================

    @staticmethod
    def _mean_var_ratio(X, axis):
        if sp.issparse(X):
            sums = np.asarray(X.sum(axis=axis)).ravel()
        else:
            sums = np.asarray(X.sum(axis=axis)).ravel()
        return float(np.mean(sums) / max(np.var(sums), 1e-10))

    @staticmethod
    def _lambda_jj(zeta):
        safe = np.maximum(np.abs(zeta), 1e-8)
        return np.tanh(safe / 2.0) / (4.0 * safe)

    # =================================================================
    # Initialization (identical to CAVI — scHPF pattern)
    # =================================================================

    def _initialize(self, X, y, X_aux):
        if sp.issparse(X):
            self.n, self.p = X.shape
        else:
            self.n, self.p = X.shape

        self.n_outcomes = 1 if y.ndim == 1 else y.shape[1]
        self.p_aux = X_aux.shape[1] if X_aux is not None and X_aux.size > 0 else 0

        K = self.K

        # Empirical bp, dp (scalars)
        self.bp = self.ap * self._mean_var_ratio(X, axis=1)
        self.dp = self.cp * self._mean_var_ratio(X, axis=0)
        if self.bp > 1000 * self.dp:
            self.dp = self.bp / 1000
            print(f"Clipping dp to {self.dp:.4f}")

        bp, dp = self.bp, self.dp

        # Global params: β, η
        self.a_beta = np.random.uniform(0.5 * self.c, 1.5 * self.c, (self.p, K))
        self.b_beta = np.random.uniform(0.5 * dp, 1.5 * dp, (self.p, K))
        self.a_eta = np.full(self.p, self.cp + K * self.c)
        self.b_eta = np.random.uniform(0.5 * dp, 1.5 * dp, self.p)

        # Pathway mask
        self._init_beta_mask()

        # Regression params: v, γ
        self.mu_v = np.random.randn(self.n_outcomes, K) * 0.01
        self.sigma_v_diag = np.full((self.n_outcomes, K), self.sigma_v ** 2)

        if self.p_aux > 0:
            self.mu_gamma = np.zeros((self.n_outcomes, self.p_aux))
            self.Sigma_gamma = np.stack([
                np.eye(self.p_aux) * self.sigma_gamma ** 2
                for _ in range(self.n_outcomes)
            ])
        else:
            self.mu_gamma = np.zeros((self.n_outcomes, 0))
            self.Sigma_gamma = np.zeros((self.n_outcomes, 0, 0))

        print(f"SVI Init: n={self.n}, p={self.p}, K={K}")
        print(f"  bp={bp:.4f}, dp={dp:.4f}")
        print(f"  E[beta] range: [{self.E_beta.min():.4f}, {self.E_beta.max():.4f}]")

    def _init_beta_mask(self):
        """Same as CAVI."""
        if self.mode == 'masked' and self.pathway_mask is not None:
            pm = self.pathway_mask.T
            if pm.shape[1] < self.K:
                pad = np.zeros((self.p, self.K - pm.shape[1]))
                self.beta_mask = np.hstack([pm, pad])
            else:
                self.beta_mask = pm[:, :self.K]
            small_a, large_b = self.c * 0.01, 100.0
            self.a_beta = np.where(self.beta_mask > 0.5, self.a_beta, small_a)
            self.b_beta = np.where(self.beta_mask > 0.5, self.b_beta, large_b)
        elif self.mode == 'combined' and self.pathway_mask is not None:
            pm = self.pathway_mask.T
            npath = self.n_pathway_factors
            self.beta_mask = np.ones((self.p, self.K))
            mask_part = pm[:, :npath] if pm.shape[1] >= npath else \
                np.hstack([pm, np.zeros((self.p, npath - pm.shape[1]))])
            self.beta_mask[:, :npath] = mask_part
            small_a, large_b = self.c * 0.01, 100.0
            for k in range(npath):
                self.a_beta[:, k] = np.where(self.beta_mask[:, k] > 0.5,
                                              self.a_beta[:, k], small_a)
                self.b_beta[:, k] = np.where(self.beta_mask[:, k] > 0.5,
                                              self.b_beta[:, k], large_b)
        else:
            self.beta_mask = None

    def _enforce_beta_mask(self):
        if self.beta_mask is None:
            return
        small_a, large_b = self.c * 0.01, 100.0
        if self.mode == 'masked':
            self.a_beta = np.where(self.beta_mask > 0.5, self.a_beta, small_a)
            self.b_beta = np.where(self.beta_mask > 0.5, self.b_beta, large_b)
        elif self.mode == 'combined':
            for k in range(self.n_pathway_factors):
                self.a_beta[:, k] = np.where(self.beta_mask[:, k] > 0.5,
                                              self.a_beta[:, k], small_a)
                self.b_beta[:, k] = np.where(self.beta_mask[:, k] > 0.5,
                                              self.b_beta[:, k], large_b)

    # =================================================================
    # Expected values
    # =================================================================

    @property
    def E_beta(self):
        return self.a_beta / self.b_beta

    @property
    def E_log_beta(self):
        return digamma(self.a_beta) - np.log(self.b_beta)

    @property
    def E_eta(self):
        return self.a_eta / self.b_eta

    # =================================================================
    # Local inference (full optimization per mini-batch)
    # =================================================================

    def _optimize_local(self, X_batch, y_batch, X_aux_batch, use_regression):
        """
        Fully optimize θ, ξ for a mini-batch (freeze globals).

        Returns: a_theta, b_theta, a_xi, b_xi, zeta
        """
        B = X_batch.shape[0]
        K = self.K

        # Initialize local params (scHPF style)
        a_theta = np.random.uniform(0.5 * self.a, 1.5 * self.a, (B, K))
        b_theta = np.full((B, K), self.bp)
        a_xi = np.full(B, self.ap + K * self.a)
        b_xi = np.full(B, self.bp)
        zeta = np.ones((B, self.n_outcomes))

        E_log_beta = self.E_log_beta
        E_beta = self.E_beta
        beta_col_sums = E_beta.sum(axis=0)
        E_v = self.mu_v
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag

        if sp.issparse(X_batch):
            X_batch = np.asarray(X_batch.todense())

        for it in range(self.local_iterations):
            E_theta = a_theta / b_theta
            E_log_theta = digamma(a_theta) - np.log(b_theta)
            E_xi = a_xi / b_xi

            # φ: (B, p, K) — only compute for nonzero entries if B is large
            # For simplicity, dense here (OK for batch_size ≤ ~256)
            log_phi = E_log_theta[:, None, :] + E_log_beta[None, :, :]
            log_phi -= sp_logsumexp(log_phi, axis=2, keepdims=True)
            phi = np.exp(log_phi)

            # z_sum for θ shape: (B, K)
            z_sum = np.einsum('ij,ijk->ik', X_batch, phi)

            # θ update
            a_theta = self.a + z_sum
            b_theta = E_xi[:, None] + beta_col_sums[None, :]

            # Regression correction to θ rate
            if use_regression and self.regression_weight > 0:
                R = self._regression_correction_batch(
                    y_batch, X_aux_batch, E_theta, E_v, E_v_sq, zeta)
                b_theta = b_theta + self.regression_weight * R
                b_theta = np.maximum(b_theta, 1e-6)

            # ξ update
            E_theta_new = a_theta / b_theta
            b_xi = self.bp + E_theta_new.sum(axis=1)

            # ζ update (JJ)
            if use_regression:
                E_A = E_theta_new @ E_v.T
                if self.p_aux > 0:
                    E_A = E_A + X_aux_batch @ self.mu_gamma.T
                Var_theta = a_theta / (b_theta ** 2)
                E_A_sq = E_A ** 2 + Var_theta @ E_v_sq.T
                zeta = np.sqrt(np.maximum(E_A_sq, 1e-8))

        return a_theta, b_theta, a_xi, b_xi, zeta

    def _regression_correction_batch(self, y_batch, X_aux_batch,
                                      E_theta, E_v, E_v_sq, zeta):
        """JJ regression correction to θ rate for a batch."""
        y_exp = y_batch if y_batch.ndim > 1 else y_batch[:, None]
        lam = self._lambda_jj(zeta)

        theta_v = E_theta @ E_v.T
        if self.p_aux > 0:
            theta_v = theta_v + X_aux_batch @ self.mu_gamma.T

        C_minus = theta_v[:, :, None] - E_theta[:, None, :] * E_v[None, :, :]

        R = (
            -(y_exp[:, :, None] - 0.5) * E_v[None, :, :]
            + 2 * lam[:, :, None] * E_v[None, :, :] * C_minus
            + 2 * lam[:, :, None] * E_v_sq[None, :, :] * E_theta[:, None, :]
        )
        return R.sum(axis=1)

    # =================================================================
    # Intermediate global parameters
    # =================================================================

    def _intermediate_beta(self, X_batch, E_theta, E_log_theta, scale):
        """Compute β̂ as if batch were full dataset."""
        if sp.issparse(X_batch):
            X_batch = np.asarray(X_batch.todense())

        E_log_beta = self.E_log_beta
        log_phi = E_log_theta[:, None, :] + E_log_beta[None, :, :]
        log_phi -= sp_logsumexp(log_phi, axis=2, keepdims=True)
        phi = np.exp(log_phi)

        z_sum = np.einsum('ij,ijk->jk', X_batch, phi)  # (p, K)
        theta_sum = E_theta.sum(axis=0)  # (K,)

        # Intermediate params (as if N copies of this batch)
        a_beta_hat = self.c + scale * z_sum
        b_beta_hat = self.E_eta[:, None] + scale * theta_sum[None, :]

        return a_beta_hat, b_beta_hat

    def _intermediate_eta(self):
        """Compute η̂."""
        a_eta_hat = np.full(self.p, self.cp + self.K * self.c)
        b_eta_hat = self.dp + self.E_beta.sum(axis=1)
        return a_eta_hat, b_eta_hat

    def _intermediate_v(self, y_batch, X_aux_batch, E_theta,
                        a_theta, b_theta, zeta, scale):
        """Compute v̂ (same equations as CAVI but with scale)."""
        y_exp = y_batch if y_batch.ndim > 1 else y_batch[:, None]
        lam = self._lambda_jj(zeta)
        Var_theta = a_theta / (b_theta ** 2)
        E_theta_sq = E_theta ** 2 + Var_theta

        theta_v = E_theta @ self.mu_v.T
        if self.p_aux > 0:
            theta_v = theta_v + X_aux_batch @ self.mu_gamma.T

        C_minus = theta_v[:, :, None] - E_theta[:, None, :] * self.mu_v[None, :, :]

        precision = 1.0 / (self.sigma_v ** 2) + \
                    2 * scale * np.einsum('ik,id->kd', lam, E_theta_sq)

        term1 = scale * np.einsum('ik,id->kd', y_exp - 0.5, E_theta)
        term2 = 2 * scale * np.einsum('ik,ikd,id->kd', lam, C_minus, E_theta)
        mean_prec = term1 - term2

        mu_v_hat = np.clip(mean_prec / precision, -10, 10)
        sigma_v_hat = 1.0 / precision

        return mu_v_hat, sigma_v_hat

    def _intermediate_gamma(self, y_batch, X_aux_batch, E_theta,
                            zeta, scale):
        """Compute γ̂."""
        if self.p_aux == 0:
            return self.mu_gamma.copy(), self.Sigma_gamma.copy()

        y_exp = y_batch if y_batch.ndim > 1 else y_batch[:, None]
        lam = self._lambda_jj(zeta)

        mu_gamma_hat = np.zeros_like(self.mu_gamma)
        Sigma_gamma_hat = np.zeros_like(self.Sigma_gamma)

        for k in range(self.n_outcomes):
            prec_prior = np.eye(self.p_aux) / (self.sigma_gamma ** 2)
            weighted_X = X_aux_batch * (2 * scale * lam[:, k])[:, None]
            prec = prec_prior + weighted_X.T @ X_aux_batch

            theta_v_k = E_theta @ self.mu_v[k]
            residual = (y_exp[:, k] - 0.5) - 2 * lam[:, k] * theta_v_k
            mean_prec = scale * X_aux_batch.T @ residual

            Sigma_gamma_hat[k] = np.linalg.inv(prec)
            mu_gamma_hat[k] = Sigma_gamma_hat[k] @ mean_prec

        return mu_gamma_hat, Sigma_gamma_hat

    # =================================================================
    # SVI global update (Robbins-Monro in canonical space)
    # =================================================================

    def _svi_update(self, rho, a_hat, b_hat, a_old, b_old):
        """Weighted average: (1-ρ)*old + ρ*hat."""
        return (1 - rho) * a_old + rho * a_hat, (1 - rho) * b_old + rho * b_hat

    # =================================================================
    # Held-out LL
    # =================================================================

    def compute_heldout_ll(self, X_val, n_iter=20):
        """Mean Poisson LL per validation sample."""
        if sp.issparse(X_val):
            X_coo = X_val.tocoo()
        else:
            X_coo = sp.coo_matrix(X_val)

        n_val = X_val.shape[0]
        a_theta = np.random.uniform(0.5 * self.a, 1.5 * self.a, (n_val, self.K))
        b_theta = np.full((n_val, self.K), self.bp)
        a_xi = np.full(n_val, self.ap + self.K * self.a)
        b_xi = np.full(n_val, self.bp)

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
            z_sum = np.zeros((n_val, self.K))
            np.add.at(z_sum, row, Xphi)

            a_theta = self.a + z_sum
            b_theta = E_xi[:, None] + beta_col_sums[None, :]
            b_xi = self.bp + E_theta.sum(axis=1)

        E_log_theta = digamma(a_theta) - np.log(b_theta)
        E_theta = a_theta / b_theta
        log_rates = E_log_theta[row] + E_log_beta[col]
        log_sum_rates = sp_logsumexp(log_rates, axis=1)

        ll = np.sum(data * log_sum_rates)
        ll -= np.sum(E_theta.sum(axis=0) * self.E_beta.sum(axis=0))
        ll -= np.sum(gammaln(data + 1))

        return ll / n_val

    # =================================================================
    # fit()
    # =================================================================

    def fit(self, X_train, y_train, X_aux_train=None,
            X_val=None, y_val=None, X_aux_val=None,
            max_epochs=500, check_freq=5,
            v_warmup=50, verbose=True,
            heldout_patience=50):
        """
        Fit via SVI.

        One epoch = one pass through all mini-batches.
        Global params updated after each mini-batch.
        """
        t0 = time.time()

        if X_aux_train is None:
            X_aux_train = np.zeros((X_train.shape[0], 0))
        y = np.asarray(y_train, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        X_aux = np.asarray(X_aux_train, dtype=np.float64)

        if X_val is not None and X_aux_val is None:
            X_aux_val = np.zeros((X_val.shape[0], 0))

        self._initialize(X_train, y, X_aux)

        N = self.n
        B = min(self.batch_size, N)
        n_batches = (N + B - 1) // B
        scale = float(N) / B

        self.elbo_history_ = []
        self.holl_history_ = []

        best_holl = -np.inf
        best_params = None
        patience_counter = 0

        global_step = 0

        for epoch in range(max_epochs):
            in_warmup = (epoch < v_warmup)
            use_regression = not in_warmup

            # Shuffle data
            perm = np.random.permutation(N)

            for batch_idx in range(n_batches):
                start = batch_idx * B
                end = min(start + B, N)
                idx = perm[start:end]
                actual_B = len(idx)
                actual_scale = float(N) / actual_B

                # Extract batch
                if sp.issparse(X_train):
                    X_b = X_train.tocsr()[idx].toarray()
                else:
                    X_b = X_train[idx]
                y_b = y[idx]
                X_aux_b = X_aux[idx]

                # Step size: ρ_t = (τ + t)^{-κ}
                rho_t = (self.tau + global_step) ** (-self.kappa)
                global_step += 1

                # 1. Optimize local parameters
                a_theta, b_theta, a_xi, b_xi, zeta = \
                    self._optimize_local(X_b, y_b, X_aux_b, use_regression)

                E_theta = a_theta / b_theta
                E_log_theta = digamma(a_theta) - np.log(b_theta)

                # 2. Compute intermediate global parameters
                a_beta_hat, b_beta_hat = self._intermediate_beta(
                    X_b, E_theta, E_log_theta, actual_scale)

                # 3. SVI update for β
                self.a_beta, self.b_beta = self._svi_update(
                    rho_t, a_beta_hat, b_beta_hat, self.a_beta, self.b_beta)
                self._enforce_beta_mask()

                # 4. Update η (not stochastic — depends only on β)
                _, b_eta_hat = self._intermediate_eta()
                self.b_eta = b_eta_hat  # Deterministic given β

                # 5. Update v, γ (after warmup)
                if use_regression:
                    mu_v_hat, sigma_v_hat = self._intermediate_v(
                        y_b, X_aux_b, E_theta, a_theta, b_theta,
                        zeta, actual_scale)
                    self.mu_v = (1 - rho_t) * self.mu_v + rho_t * mu_v_hat
                    self.sigma_v_diag = (1 - rho_t) * self.sigma_v_diag + rho_t * sigma_v_hat

                    if self.p_aux > 0:
                        mu_g_hat, Sig_g_hat = self._intermediate_gamma(
                            y_b, X_aux_b, E_theta, zeta, actual_scale)
                        self.mu_gamma = (1 - rho_t) * self.mu_gamma + rho_t * mu_g_hat

            # End of epoch — check convergence
            if epoch % check_freq == 0:
                holl = None
                if X_val is not None:
                    holl = self.compute_heldout_ll(X_val)
                    self.holl_history_.append((epoch, holl))

                    if holl > best_holl:
                        best_holl = holl
                        best_params = self._checkpoint()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                if verbose:
                    phase = "WU" if in_warmup else "FT"
                    v_str = f"v={self.mu_v.ravel()[:3]}"
                    holl_str = f"  HO-LL={holl:.2f}" if holl is not None else ""
                    beta_range = f"E[β]=[{self.E_beta.min():.3f},{self.E_beta.max():.3f}]"
                    print(f"Epoch {epoch:4d} [{phase}]: ρ={rho_t:.5f}  "
                          f"{beta_range}  {v_str}{holl_str}")

                # Early stopping
                if patience_counter >= heldout_patience and epoch >= v_warmup + 20:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no HO-LL improvement for {heldout_patience} checks)")
                    break

        elapsed = time.time() - t0
        if verbose:
            print(f"\nTraining complete in {elapsed:.1f}s")
            if best_params is not None:
                print(f"Best HO-LL: {best_holl:.4f}")

        if best_params is not None:
            self._restore(best_params)

        return self

    # =================================================================
    # Checkpoint / restore
    # =================================================================

    def _checkpoint(self):
        return {
            'a_beta': self.a_beta.copy(), 'b_beta': self.b_beta.copy(),
            'a_eta': self.a_eta.copy(), 'b_eta': self.b_eta.copy(),
            'mu_v': self.mu_v.copy(), 'sigma_v_diag': self.sigma_v_diag.copy(),
            'mu_gamma': self.mu_gamma.copy(), 'Sigma_gamma': self.Sigma_gamma.copy(),
        }

    def _restore(self, cp):
        for k, v in cp.items():
            setattr(self, k, v)

    # =================================================================
    # predict / transform
    # =================================================================

    def predict_proba(self, X_new, X_aux_new=None, n_iter=20):
        if sp.issparse(X_new):
            X_coo = X_new.tocoo()
        else:
            X_coo = sp.coo_matrix(X_new)

        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, max(self.p_aux, 0)))

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

        E_theta = a_theta / b_theta
        logits = E_theta @ self.mu_v.T
        if self.p_aux > 0:
            logits = logits + X_aux_new @ self.mu_gamma.T

        return expit(logits).squeeze()

    def transform(self, X_new, y_new=None, X_aux_new=None, n_iter=20, **kwargs):
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