"""Full Gibbs sampler for supervised Poisson factorization (DRGP model).

Stages A-D in one class:
  A. Raikov-augmented HPF Gibbs (Cemgil 2009; Gopalan-Blei 2015) for theta, beta,
     xi, eta. Conjugate Gamma updates given multinomial z.
  B. Spike-and-slab on beta via auxiliary r_jℓ ∈ {0,1} + Beta-Bernoulli pi.
     Inactive factors are zeroed out for that gene.
  C. PG-augmented Gibbs (Polson-Scott-Windle 2013) for the logistic head:
     omega_ik ~ PG(1, |psi|), upsilon_k Gaussian conditional, gamma_k Gaussian
     conditional, s_kℓ via Park-Casella inverse-Gaussian (Bayesian Lasso).
  D. Slice-sampled theta under supervision (Neal 2003 step-out + shrinkage).
     The supervised augmentation introduces a quadratic in theta that breaks
     conjugacy; slice sampling is the rigorous fix.

Notation matches `vi_cavi.py` and the paper (theta, beta, xi, eta, upsilon/v,
gamma) so the outputs can flow into the same downstream evaluators.

Initial implementation: NumPy, kappa=1 (binary supervision). The code is
written to be extensible to kappa>1 in the future.

References:
  Cemgil 2009 — NMF Gibbs via Raikov augmentation
  Gopalan, Hofman, Blei 2015 — Scalable HPF
  Polson, Scott, Windle 2013 — Bayesian logistic via Pólya-Gamma
  Park & Casella 2008 — Bayesian Lasso via scale-mixture-of-normals
  Neal 2003 — Slice sampling
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
from polyagamma import random_polyagamma


# =============================================================================
# Helpers
# =============================================================================

def _safe_log(x, floor=1e-30):
    return np.log(np.maximum(x, floor))


def _slice_sample_1d(log_f, x_init, w=1.0, max_step=50, max_shrink=100,
                     lower=1e-6, upper=1e6, rng=None):
    """Univariate slice sampler (Neal 2003): step-out + shrinkage.

    log_f(x) returns log-density up to a constant (need not be normalized).
    Returns a new sample from p(x) ∝ exp(log_f(x)), constrained to (lower, upper).
    """
    if rng is None:
        rng = np.random.default_rng()

    log_y = log_f(x_init) + np.log(rng.random())
    L = x_init - w * rng.random()
    R = L + w
    L = max(L, lower); R = min(R, upper)
    # step-out left
    for _ in range(max_step):
        if L <= lower or log_f(L) <= log_y:
            break
        L = max(L - w, lower)
    # step-out right
    for _ in range(max_step):
        if R >= upper or log_f(R) <= log_y:
            break
        R = min(R + w, upper)
    # shrinkage
    for _ in range(max_shrink):
        x_new = L + (R - L) * rng.random()
        if log_f(x_new) > log_y:
            return x_new
        if x_new < x_init:
            L = x_new
        else:
            R = x_new
    # fallback
    return x_init


# =============================================================================
# Sampler
# =============================================================================

@dataclass
class GibbsConfig:
    """Hyperparameters for the Gibbs sampler. Defaults match vi_cavi.py."""
    n_factors: int = 20
    a: float = 0.3          # theta Gamma shape
    ap: float = 1.0         # xi Gamma shape prior
    c: float = 0.3          # beta Gamma shape
    cp: float = 1.0         # eta Gamma shape prior
    b_v: float = 2.0        # Laplace scale for upsilon (Bayesian Lasso)
    sigma_gamma: float = 1.0  # Normal std for gamma (aux head)
    alpha_pi: float = 1.0   # Beta prior alpha for pi
    beta_pi_scale: float = 5.0  # Beta prior beta = alpha_pi * scale (favors sparsity)
    use_spike_slab: bool = True
    supervised: bool = True
    use_intercept: bool = True
    # slice sampler
    slice_w: float = 0.5
    slice_max_step: int = 30
    slice_max_shrink: int = 100
    # data-driven rate priors (set automatically in _initialize)
    bp: float = -1.0
    dp: float = -1.0


class GibbsSampler:
    """Full Gibbs sampler for supervised Poisson factorization with spike-slab β."""

    def __init__(self, cfg: GibbsConfig, random_state: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(random_state)
        # state filled by _initialize
        self.theta = None     # (n, K)
        self.beta_tilde = None  # (p, K) — sampled active values
        self.r_beta = None    # (p, K) — spike-slab indicators
        self.pi = None        # scalar
        self.xi = None        # (n,)
        self.eta = None       # (p,)
        self.upsilon = None   # (K,)  [kappa=1 binary]
        self.gamma = None     # (p_aux_with_intercept,)
        self.s = None         # (K,)  — Bayesian Lasso local scales
        self.omega = None     # (n,)  — PG aux
        # data
        self.X_coo = None
        self.rows = None      # (nnz,) cell indices
        self.cols = None      # (nnz,) gene indices
        self.x_vals = None    # (nnz,) counts
        self.z = None         # (nnz, K) latent allocations
        self.n = self.p = self.K = self.p_aux = None
        self.y = None
        self.X_aux = None
        # chains (post-burn)
        self.chains_ = {}

    # ---------------- initialization ----------------------------------------

    def _initialize(self, X, y, X_aux):
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        self.X_coo = X.tocoo()
        self.rows = self.X_coo.row.astype(np.int32)
        self.cols = self.X_coo.col.astype(np.int32)
        self.x_vals = self.X_coo.data.astype(np.int32)
        self.n, self.p = X.shape
        self.K = self.cfg.n_factors
        self.y = np.asarray(y).astype(np.int32).ravel()

        # Standardize aux + prepend intercept
        if X_aux is None or (hasattr(X_aux, "size") and X_aux.size == 0):
            aux = np.zeros((self.n, 0))
        else:
            aux = np.asarray(X_aux, dtype=np.float64)
            mu = aux.mean(0, keepdims=True)
            sd = aux.std(0, keepdims=True)
            sd[sd == 0] = 1.0
            aux = (aux - mu) / sd
        if self.cfg.use_intercept:
            aux = np.hstack([np.ones((self.n, 1)), aux])
        self.X_aux = aux
        self.p_aux = aux.shape[1]

        # Empirical rate priors (mirror scHPF / vi_cavi)
        def _mean_var_ratio(M, axis):
            s = np.asarray(M.sum(axis=axis)).ravel().astype(np.float64)
            return float(s.mean() / max(s.var(), 1e-10))
        self.cfg.bp = self.cfg.ap * _mean_var_ratio(X, axis=1)
        self.cfg.dp = self.cfg.cp * _mean_var_ratio(X, axis=0)
        # Floors/ceilings to avoid pathological priors
        for attr in ("bp", "dp"):
            v = getattr(self.cfg, attr)
            v = float(np.clip(v, 1e-4, 1e4))
            setattr(self.cfg, attr, v)

        K, n, p = self.K, self.n, self.p
        rng = self.rng

        # Gamma states init: random around the prior mean
        self.theta = rng.gamma(self.cfg.a, 1.0 / self.cfg.bp, size=(n, K))
        self.beta_tilde = rng.gamma(self.cfg.c, 1.0 / self.cfg.dp, size=(p, K))
        self.xi = rng.gamma(self.cfg.ap + K * self.cfg.a, 1.0 / self.cfg.bp, size=n)
        self.eta = rng.gamma(self.cfg.cp + K * self.cfg.c, 1.0 / self.cfg.dp, size=p)

        # Spike-slab
        if self.cfg.use_spike_slab:
            self.pi = 0.5
            # Bernoulli init with pi=0.5
            self.r_beta = rng.binomial(1, self.pi, size=(p, K)).astype(np.int8)
        else:
            self.pi = 1.0
            self.r_beta = np.ones((p, K), dtype=np.int8)

        # Latent z allocations: init by current rates
        self.z = np.zeros((len(self.x_vals), K), dtype=np.int32)
        self._sample_z()

        # Supervised head
        self.upsilon = rng.normal(0, 0.1, size=K)
        self.gamma = np.zeros(self.p_aux)
        if self.cfg.use_intercept and self.p_aux > 0:
            n_pos = int(self.y.sum())
            n_neg = self.n - n_pos
            if n_pos > 0 and n_neg > 0:
                self.gamma[0] = float(np.log(n_pos / n_neg))
        self.s = np.full(K, 2.0 * self.cfg.b_v ** 2)  # E[s] under Exp(λ²/2) = 1/λ²·2 = 2 b²
        self.omega = np.full(n, 0.25)  # PG(1,0) mean = 0.25

    # ---------------- Stage A: Raikov z + Gamma updates ---------------------

    def _sample_z(self):
        """Vectorized Multinomial split via binomial-thinning.

        For each nnz entry m=(i,j) with x_{ij} = x:
            z_{m,1} ~ Bin(x, p_1)
            z_{m,2} ~ Bin(x - z_{m,1}, p_2 / (1 - p_1))
            ...
            z_{m,K} = x - sum_{k<K} z_{m,k}

        Each Binomial step is a single np.random.binomial call across all
        nnz entries — O(nnz·K) total but with K ~ 50 vectorized numpy ops
        instead of nnz Python calls (100-200× speedup for large nnz).

        Inactive factors (r_{jℓ}=0) contribute zero rate and receive z=0.
        """
        K = self.K
        nnz = len(self.x_vals)
        # rates: (nnz, K)
        rates = (self.theta[self.rows]
                 * self.beta_tilde[self.cols]
                 * self.r_beta[self.cols].astype(np.float64))
        rate_sum = rates.sum(axis=1)
        # Cells with no active factor: rate_sum=0 — assign uniform within active set.
        # In practice this only happens transiently; protect with a small eps.
        no_active = rate_sum <= 0
        if no_active.any():
            rates[no_active] = 1.0  # uniform fallback over all K
            rate_sum[no_active] = float(K)

        z = np.zeros((nnz, K), dtype=np.int32)
        x_remaining = self.x_vals.astype(np.int32).copy()
        rate_remaining = rate_sum.copy()
        for k in range(K - 1):
            p_k = rates[:, k] / np.maximum(rate_remaining, 1e-12)
            # Numerical floor/ceil to keep within [0, 1]
            np.clip(p_k, 0.0, 1.0, out=p_k)
            draws = self.rng.binomial(x_remaining, p_k).astype(np.int32)
            z[:, k] = draws
            x_remaining -= draws
            rate_remaining -= rates[:, k]
        z[:, K - 1] = x_remaining
        self.z = z

    def _aggregate_z(self):
        """Return z_sum_theta (n,K), z_sum_beta (p,K)."""
        z_sum_theta = np.zeros((self.n, self.K), dtype=np.float64)
        z_sum_beta = np.zeros((self.p, self.K), dtype=np.float64)
        np.add.at(z_sum_theta, self.rows, self.z)
        np.add.at(z_sum_beta, self.cols, self.z)
        return z_sum_theta, z_sum_beta

    def _sample_xi(self):
        """xi_i | θ ~ Gamma(ap + K*a, bp + Σ_ℓ θ_iℓ)."""
        shape = self.cfg.ap + self.K * self.cfg.a
        rate = self.cfg.bp + self.theta.sum(axis=1)
        self.xi = self.rng.gamma(shape, 1.0 / rate)

    def _sample_eta(self):
        """eta_j | β ~ Gamma(cp + K*c, dp + Σ_ℓ β̃_jℓ · r_jℓ).
        Inactive entries contribute 0 (β=0). For the active subset only.
        """
        active = self.r_beta.astype(np.float64)
        shape = self.cfg.cp + active.sum(axis=1) * self.cfg.c  # per-gene active K
        rate = self.cfg.dp + (self.beta_tilde * active).sum(axis=1)
        # ensure shape > 0 even if no active factor
        shape = np.maximum(shape, self.cfg.cp + 1e-6)
        self.eta = self.rng.gamma(shape, 1.0 / rate)

    def _sample_beta_tilde(self, z_sum_beta):
        """β̃_jℓ | rest ~ Gamma(c + z_sum_jℓ · r_jℓ, η_j + r_jℓ · Σ_i θ_iℓ).

        For inactive (r=0): β̃ is sampled from prior (not used in likelihood).
        """
        active = self.r_beta.astype(np.float64)
        theta_col_sum = self.theta.sum(axis=0)  # (K,)
        # active case: posterior using z_sum and θ_col_sum
        shape_active = self.cfg.c + z_sum_beta  # uses z which is 0 for inactive
        rate_active = self.eta[:, None] + theta_col_sum[None, :]
        # inactive case: prior Gamma(c, η_j)
        shape_inactive = self.cfg.c * np.ones((self.p, self.K))
        rate_inactive = self.eta[:, None] * np.ones((self.p, self.K))
        # combine
        shape = np.where(self.r_beta == 1, shape_active, shape_inactive)
        rate = np.where(self.r_beta == 1, rate_active, rate_inactive)
        self.beta_tilde = self.rng.gamma(shape, 1.0 / np.maximum(rate, 1e-12))

    def _effective_beta(self):
        """β as used in the likelihood: β̃ · r."""
        return self.beta_tilde * self.r_beta

    # ---------------- Stage B: spike-slab updates ---------------------------

    def _sample_r_beta(self):
        """Sample r_jℓ ∈ {0,1} marginally over z, conditional on θ, β̃, π.

        Bayes factor: log P(x_{·j} | r_jℓ=1, rest) - log P(x_{·j} | r_jℓ=0, rest).

        Poisson likelihood: each gene j contributes ∑_i [x_ij log rate_ij - rate_ij].
        Adding factor ℓ to the rate changes both terms.
        """
        if not self.cfg.use_spike_slab:
            return
        log_pi = np.log(np.clip(self.pi, 1e-9, 1 - 1e-9))
        log_1mpi = np.log(np.clip(1 - self.pi, 1e-9, 1 - 1e-9))
        # Precompute current per-cell rate per gene: rate_ij = Σ_m θ_im · β̃_jm · r_jm
        beta_eff = self._effective_beta()  # (p, K)
        # Rate_ij for each gene j, cell i: theta @ beta_eff[j].T → (n, p)
        # n*p can be big (1k × 2k = 2M), but we want to keep it small per call.
        # We compute rates_ij for the FULL matrix once per sweep (still 2M floats).
        rate_full = self.theta @ beta_eff.T  # (n, p)
        # For x_ij we need a vectorized form. Cache x_csr.
        x_csr = sp.csr_matrix((self.x_vals, (self.rows, self.cols)),
                              shape=(self.n, self.p)).toarray()  # 1k×2k = 2M; fine.
        # Loop over factors (K=20) only.
        for k in range(self.K):
            theta_k = self.theta[:, k]                  # (n,)
            beta_k_active = self.beta_tilde[:, k]       # (p,)
            # Delta rate when r_jk flips 1->0 or 0->1 for gene j:
            delta_jk = np.outer(theta_k, beta_k_active)  # (n, p): θ_ik β̃_jk
            # rates if r_jk = 1: rate_full + (1 - r_jk_current)·delta_jk
            # rates if r_jk = 0: rate_full - r_jk_current·delta_jk
            cur_r = self.r_beta[:, k].astype(np.float64)  # (p,)
            rate_with    = rate_full + (1.0 - cur_r)[None, :] * delta_jk  # r=1 hypothesis
            rate_without = rate_full - cur_r[None, :] * delta_jk          # r=0 hypothesis
            # log-likelihood per gene (sum over cells)
            #   ll_j = Σ_i [x_ij log(rate_ij) - rate_ij] (drop x! const)
            with np.errstate(invalid="ignore", divide="ignore"):
                log_rw = _safe_log(rate_with)
                log_rwo = _safe_log(rate_without)
                # exclude impossible rate_without=0 if x_ij>0
                ll_with    = (x_csr * log_rw - rate_with).sum(axis=0)    # (p,)
                ll_without = (x_csr * log_rwo - rate_without).sum(axis=0)
            log_p1 = log_pi + ll_with
            log_p0 = log_1mpi + ll_without
            # softmax to probability of r=1
            mx = np.maximum(log_p1, log_p0)
            p1 = np.exp(log_p1 - mx) / (np.exp(log_p1 - mx) + np.exp(log_p0 - mx))
            new_r = (self.rng.random(self.p) < p1).astype(np.int8)
            # update rate_full incrementally so we don't recompute next k
            delta_apply = (new_r.astype(np.float64) - cur_r)[None, :] * delta_jk
            rate_full = rate_full + delta_apply
            self.r_beta[:, k] = new_r

    def _sample_pi(self):
        """π | r ~ Beta(α + Σr, β + (pK - Σr))."""
        if not self.cfg.use_spike_slab:
            return
        n_active = int(self.r_beta.sum())
        n_total = self.p * self.K
        alpha = self.cfg.alpha_pi
        beta_p = self.cfg.alpha_pi * self.cfg.beta_pi_scale
        self.pi = float(self.rng.beta(alpha + n_active, beta_p + (n_total - n_active)))

    # ---------------- Stage C: PG-augmented supervised head ------------------

    def _psi(self):
        """Linear predictor ψ_i = θ_i^T υ + (x_aux_i)^T γ.  Shape (n,)."""
        out = self.theta @ self.upsilon
        if self.p_aux > 0:
            out = out + self.X_aux @ self.gamma
        return out

    def _sample_omega(self):
        """ω_i | y, ψ_i ~ PG(1, |ψ_i|)."""
        psi = self._psi()
        # polyagamma's random_polyagamma uses |z|; sign doesn't matter for PG(1, |z|)
        self.omega = np.asarray(random_polyagamma(z=np.abs(psi), size=self.n))

    def _sample_upsilon(self):
        """υ | ω, γ, θ, y, s ~ N(μ, Σ).
        Σ⁻¹ = diag(1/s) + Θ^T Ω Θ
        μ = Σ · Θ^T (κ - Ω · X_aux γ)
        """
        kappa = self.y - 0.5
        Theta = self.theta
        omega = self.omega
        # Aux contribution (offset)
        aux_off = self.X_aux @ self.gamma if self.p_aux > 0 else np.zeros(self.n)
        b = Theta.T @ (kappa - omega * aux_off)  # (K,)
        # Σ⁻¹
        prec = (Theta.T * omega) @ Theta + np.diag(1.0 / np.maximum(self.s, 1e-12))
        L = np.linalg.cholesky(prec)
        # solve L L^T μ = b
        mu = np.linalg.solve(L.T, np.linalg.solve(L, b))
        # sample
        eta = self.rng.standard_normal(self.K)
        self.upsilon = mu + np.linalg.solve(L.T, eta)

    def _sample_gamma(self):
        """γ | ω, υ, θ, y ~ N(μ, Σ).
        Σ⁻¹ = (1/σ_γ²) I + X_aux^T Ω X_aux
        μ = Σ · X_aux^T (κ - Ω · Θ υ)
        """
        if self.p_aux == 0:
            return
        kappa = self.y - 0.5
        omega = self.omega
        theta_v = self.theta @ self.upsilon  # (n,)
        # Σ⁻¹
        prec = (self.X_aux.T * omega) @ self.X_aux + np.eye(self.p_aux) / (self.cfg.sigma_gamma ** 2)
        L = np.linalg.cholesky(prec)
        b = self.X_aux.T @ (kappa - omega * theta_v)
        mu = np.linalg.solve(L.T, np.linalg.solve(L, b))
        eta = self.rng.standard_normal(self.p_aux)
        self.gamma = mu + np.linalg.solve(L.T, eta)

    def _sample_s(self):
        """s_ℓ | υ_ℓ via Park-Casella: 1/s_ℓ ~ InverseGaussian(μ, λ)
        where μ = 1/(b_v · |υ_ℓ|), λ = 1/b_v². Then s_ℓ = 1/(sample).
        """
        b_v = self.cfg.b_v
        for k in range(self.K):
            v = abs(self.upsilon[k])
            if v < 1e-9:
                inv_s = self.rng.wald(mean=1e9, scale=1.0 / b_v ** 2)
            else:
                inv_s = self.rng.wald(mean=1.0 / (b_v * v), scale=1.0 / b_v ** 2)
            self.s[k] = 1.0 / max(inv_s, 1e-12)

    # ---------------- Stage D: slice-sampled θ -------------------------------

    def _sample_theta(self, z_sum_theta):
        """Sample θ_iℓ given supervision.

        Log-density (up to constant):
            log p(θ) = (a_θ - 1) log θ - b_θ θ - A θ² + B θ
        where:
            a_θ = a + z_sum_theta[i, ℓ]
            b_θ = ξ_i + Σ_j β̃_jℓ · r_jℓ
            A   = (1/2) ω_i · υ_ℓ²
            B   = υ_ℓ · (κ_i - ω_i · C_iℓ)
            C_iℓ = θ_i^T υ + (x_aux)^T γ - θ_iℓ · υ_ℓ      (the "other-factors" predictor)

        If unsupervised, this collapses to a Gamma(a_θ, b_θ) and we sample directly.
        """
        beta_eff = self._effective_beta()  # (p, K)
        beta_col_sum = beta_eff.sum(axis=0)  # (K,)
        a_theta_mat = self.cfg.a + z_sum_theta  # (n, K)
        b_theta_mat = self.xi[:, None] + beta_col_sum[None, :]  # (n, K)

        if not self.cfg.supervised:
            self.theta = self.rng.gamma(a_theta_mat, 1.0 / np.maximum(b_theta_mat, 1e-12))
            return

        # Supervised: slice sample per (i, ℓ)
        kappa_y = self.y - 0.5  # (n,)
        omega = self.omega       # (n,)
        ups = self.upsilon       # (K,)
        ups2 = ups ** 2          # (K,)
        aux_off = self.X_aux @ self.gamma if self.p_aux > 0 else np.zeros(self.n)

        # full theta @ ups
        full_proj = self.theta @ ups + aux_off  # (n,)

        cfg = self.cfg
        for i in range(self.n):
            for k in range(self.K):
                a_th = a_theta_mat[i, k]
                b_th = b_theta_mat[i, k]
                A = 0.5 * omega[i] * ups2[k]
                # C_iℓ = full_proj_i - θ_iℓ * ups_ℓ  (note: includes aux)
                C = full_proj[i] - self.theta[i, k] * ups[k]
                B = ups[k] * (kappa_y[i] - omega[i] * C)

                def log_f(t, a_th=a_th, b_th=b_th, A=A, B=B):
                    if t <= 0:
                        return -np.inf
                    return (a_th - 1) * np.log(t) - b_th * t - A * t * t + B * t

                old = self.theta[i, k]
                new = _slice_sample_1d(
                    log_f, old,
                    w=cfg.slice_w,
                    max_step=cfg.slice_max_step,
                    max_shrink=cfg.slice_max_shrink,
                    lower=1e-6, upper=1e4,
                    rng=self.rng,
                )
                # update full_proj for the change
                full_proj[i] += (new - old) * ups[k]
                self.theta[i, k] = new

    # ---------------- main loop ---------------------------------------------

    def fit(self, X, y, X_aux=None, n_burn=200, n_keep=500,
            store_chains=("upsilon", "gamma"),
            print_every=10, verbose=True):
        """Run the Gibbs chain."""
        self._initialize(X, y, X_aux)
        n_iter = n_burn + n_keep

        # Storage for chains
        self.chains_ = {k: [] for k in store_chains}
        if "theta_post_mean" not in self.chains_:
            theta_running_sum = np.zeros_like(self.theta)
            beta_running_sum = np.zeros_like(self.beta_tilde)
            r_beta_running_sum = np.zeros_like(self.r_beta, dtype=np.float64)
            n_kept = 0

        t0 = time.time()
        for it in range(n_iter):
            # Stage A + B
            self._sample_z()
            z_sum_theta, z_sum_beta = self._aggregate_z()
            self._sample_xi()
            self._sample_eta()
            self._sample_beta_tilde(z_sum_beta)
            if self.cfg.use_spike_slab:
                self._sample_r_beta()
                self._sample_pi()

            # Stage C
            if self.cfg.supervised:
                self._sample_omega()
                self._sample_upsilon()
                self._sample_gamma()
                self._sample_s()

            # Stage D (or plain Gamma if not supervised)
            self._sample_theta(z_sum_theta)

            # Collect
            if it >= n_burn:
                for k in self.chains_:
                    self.chains_[k].append(np.array(getattr(self, k)).copy())
                theta_running_sum += self.theta
                beta_running_sum += self.beta_tilde
                r_beta_running_sum += self.r_beta
                n_kept += 1

            if verbose and (it % print_every == 0 or it == n_iter - 1):
                elapsed = time.time() - t0
                n_active = int(self.r_beta.sum()) if self.cfg.use_spike_slab else self.p * self.K
                msg = (f"iter {it:4d}/{n_iter}  "
                       f"||v||={np.linalg.norm(self.upsilon):.3f}  "
                       f"|v|_max={np.abs(self.upsilon).max():.3f}  "
                       f"|theta|_mean={self.theta.mean():.3f}  "
                       f"active_β={n_active}/{self.p * self.K}  "
                       f"pi={self.pi:.3f}  "
                       f"elapsed={elapsed:.1f}s")
                print(msg, flush=True)

        # finalize chains
        for k in self.chains_:
            self.chains_[k] = np.stack(self.chains_[k], axis=0)
        self.theta_post_mean = theta_running_sum / n_kept
        self.beta_tilde_post_mean = beta_running_sum / n_kept
        self.r_beta_post_mean = r_beta_running_sum / n_kept
        return self

    # ---------------- prediction ---------------------------------------------

    def _infer_theta_new(self, X_new, n_iter=20):
        """Mean-field-style theta inference on new data, holding β, η, υ, γ fixed.

        Run a brief Gibbs chain on θ_new with the supervision term DROPPED
        (we don't have y_new). Use β_post_mean for likelihood.
        """
        if not sp.issparse(X_new):
            X_new = sp.csr_matrix(X_new)
        X_new_coo = X_new.tocoo()
        rows = X_new_coo.row.astype(np.int32)
        cols = X_new_coo.col.astype(np.int32)
        x_vals = X_new_coo.data.astype(np.int32)
        n_new = X_new.shape[0]

        beta_eff = self.beta_tilde_post_mean * self.r_beta_post_mean  # use posterior means
        beta_col_sum = beta_eff.sum(axis=0)

        # init theta_new from prior
        theta_new = self.rng.gamma(self.cfg.a, 1.0 / self.cfg.bp, size=(n_new, self.K))
        xi_new = self.rng.gamma(self.cfg.ap + self.K * self.cfg.a,
                                1.0 / self.cfg.bp, size=n_new)
        z_new = np.zeros((len(x_vals), self.K), dtype=np.int32)

        for _ in range(n_iter):
            # vectorized z_new (binomial thinning, same trick as _sample_z)
            rates = theta_new[rows] * beta_eff[cols]  # (nnz_new, K)
            rate_sum = rates.sum(axis=1)
            no_active = rate_sum <= 0
            if no_active.any():
                rates[no_active] = 1.0; rate_sum[no_active] = float(self.K)
            x_rem = x_vals.copy()
            rate_rem = rate_sum.copy()
            for k in range(self.K - 1):
                p_k = rates[:, k] / np.maximum(rate_rem, 1e-12)
                np.clip(p_k, 0.0, 1.0, out=p_k)
                draws = self.rng.binomial(x_rem, p_k).astype(np.int32)
                z_new[:, k] = draws
                x_rem -= draws
                rate_rem -= rates[:, k]
            z_new[:, self.K - 1] = x_rem
            z_sum = np.zeros((n_new, self.K))
            np.add.at(z_sum, rows, z_new)
            # xi
            xi_new = self.rng.gamma(self.cfg.ap + self.K * self.cfg.a,
                                    1.0 / (self.cfg.bp + theta_new.sum(axis=1)))
            # theta — unsupervised Gamma (no y, no slice)
            shape = self.cfg.a + z_sum
            rate = xi_new[:, None] + beta_col_sum[None, :]
            theta_new = self.rng.gamma(shape, 1.0 / rate)
        return theta_new

    def predict_proba(self, X_new, X_aux_new=None, n_iter=20):
        """Predict P(y=1 | X_new) using posterior-mean υ, γ and inferred θ_new."""
        theta_new = self._infer_theta_new(X_new, n_iter=n_iter)
        # standardize and intercept aux (must match training pipeline)
        if X_aux_new is None or (hasattr(X_aux_new, "size") and X_aux_new.size == 0):
            aux = np.zeros((theta_new.shape[0], 0))
        else:
            aux = np.asarray(X_aux_new, dtype=np.float64)
            # NOTE: should reuse training mean/sd; for now standardize on the new set as approx
            mu = aux.mean(0, keepdims=True)
            sd = aux.std(0, keepdims=True); sd[sd == 0] = 1.0
            aux = (aux - mu) / sd
        if self.cfg.use_intercept:
            aux = np.hstack([np.ones((theta_new.shape[0], 1)), aux])

        ups_mean = self.chains_["upsilon"].mean(axis=0)  # (K,)
        gam_mean = self.chains_["gamma"].mean(axis=0) if "gamma" in self.chains_ else self.gamma

        logits = theta_new @ ups_mean
        if aux.shape[1] > 0:
            logits = logits + aux @ gam_mean
        return 1.0 / (1.0 + np.exp(-logits))

    # ---------------- diagnostics --------------------------------------------

    def upsilon_credible_intervals(self, alpha=0.05):
        """Return (lower, upper, mean) over post-burn samples for υ_ℓ."""
        chain = self.chains_["upsilon"]  # (n_keep, K)
        lo = np.quantile(chain, alpha / 2, axis=0)
        hi = np.quantile(chain, 1 - alpha / 2, axis=0)
        mn = chain.mean(axis=0)
        return lo, hi, mn
