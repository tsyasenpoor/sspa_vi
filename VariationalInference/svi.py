"""Stochastic Variational Inference for Pólya-Gamma-augmented Supervised PF.

This file implements the full SVI algorithm of Hoffman, Blei, Wang & Paisley
(2013) — "Stochastic Variational Inference" — applied to the supervised
Poisson factorization model with Pólya-Gamma (PSW 2013) augmentation on the
logistic regression head. No Jaakkola-Jordan code here at all — PG only.

Model
-----
For cells i=1..n, genes j=1..p, factors ℓ=1..K, outcomes k=1..kappa:

    x_ij | θ_i, β_j         ~ Poisson( Σ_ℓ θ_iℓ · β_jℓ · r_jℓ )
    θ_iℓ | ξ_i              ~ Gamma( a, ξ_i )
    ξ_i                     ~ Gamma( ap, bp )
    β_jℓ | r_jℓ=1, η_j      ~ Gamma( c, η_j )
    β_jℓ | r_jℓ=0           = 0  (spike)
    r_jℓ | π                ~ Bernoulli( π )
    π                       ~ Beta( α_π, β_π )
    η_j                     ~ Gamma( cp, dp )
    y_ik | θ_i, v_k, γ_k    ~ Bernoulli( σ( θ_i·v_k + x_aux_i·γ_k ) )
    v_kℓ | s_kℓ             ~ N( 0, s_kℓ )
    s_kℓ                    ~ Exp( 1/(2·b_v²) )       [Bayesian Lasso]
    γ_k                     ~ N( 0, σ_γ²·I )

PG augmentation introduces ω_ik ~ PG(1, 0), with conditional ω_ik|ψ_ik ~ PG(1, |ψ_ik|)
where ψ_ik = θ_i·v_k + x_aux_i·γ_k.

Locals / Globals split (per Hoffman 2013 SVI)
---------------------------------------------
* LOCAL  (per cell i):  θ_i, ξ_i, ω_i  (and the multinomial split z_{ij·})
* GLOBAL (shared):      β, η, r, π, v, γ, s

Algorithm (one stochastic outer iteration)
------------------------------------------
For t = 1, 2, ...
    ρ_t = (τ_0 + t) ** (-κ_lr)                                 # step size
    1. Sample minibatch B ⊂ {1..n} of size S.
    2. LOCAL: for i ∈ B, run n_local_iter sweeps of:
        z_{ij·} expectation under q(θ_i)·q(β_j)·q(r)
        θ_iℓ Gamma update (linearized form-matching for PG quadratic term)
        ξ_i Gamma update
        ω_i sample from PG(1, |E[ψ_i]|)  -- PG-Gibbs flavour (not pure MFVB)
       Persist (a_θi, b_θi, a_ξi, b_ξi) in the warm-start cache for next visit.
    3. SUFFICIENT STATS from B, scaled by n/|B| to estimate full-batch SS.
    4. STOCHASTIC NATURAL GRADIENT on global natural params:
            λ_t  ← (1 - ρ_t) λ_{t-1}  +  ρ_t · λ̂(ss_t)
       Updated globals: β, η, r, π, v, γ. s is Park-Casella on v.
    5. (Optional) Track HOLL on validation set every check_freq iterations.

References
----------
Hoffman, Blei, Wang & Paisley (2013). Stochastic Variational Inference. JMLR 14.
Polson, Scott, Windle (2013). Bayesian Inference for Logistic Models. JASA 108.
Cemgil (2009). Bayesian Inference for NMF. CIN 2009.
Gopalan, Hofman, Blei (2015). Scalable Recommendation with HPF. UAI.
Park & Casella (2008). The Bayesian Lasso. JASA 103.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

# polyagamma is CPU-only; PG samples cross host↔device exactly once per outer iter.
try:
    from polyagamma import random_polyagamma
    _HAS_POLYAGAMMA = True
except ImportError:
    random_polyagamma = None
    _HAS_POLYAGAMMA = False

# Reuse the same array-namespace shim as vi_cavi.py so this file is GPU-ready
# whenever JAX is installed. xp resolves to jnp under JAX and numpy otherwise.
try:
    from .jax_backend import (
        xp, USE_JAX, HAS_GPU, to_device, to_numpy,
        digamma,
        log_expit, expit as _expit,
        scatter_add_to, phi_chunk_core,
    )
except ImportError:
    from jax_backend import (
        xp, USE_JAX, HAS_GPU, to_device, to_numpy,
        digamma,
        log_expit, expit as _expit,
        scatter_add_to, phi_chunk_core,
    )


# ----------------------------------------------------------------
# JIT-fused inner CAVI step (the local CAVI hot path).
# ----------------------------------------------------------------
# The whole body of one local CAVI inner iter — softmax over multinomial
# responsibilities, segment_sum scatter to per-cell, supervised PG correction,
# and the Gamma θ/ξ updates — is fused into a single compiled GPU kernel.
# JAX caches a separate compile per unique input shape, so we pad nnz_b to a
# fixed bucket inside _local_step (see _NNZ_BUCKET below) to keep the number
# of distinct compiles small (~10 across a long run).
if USE_JAX:
    import jax
    from jax import jit as _jit
    from functools import partial as _partial
    import jax.numpy as _jnp

    @_partial(_jit, static_argnames=("S", "K", "supervised", "has_aux"))
    def _jit_inner_cavi_step(
        a_th, b_th, a_xi, b_xi,
        omega_b, kappa_y_b, aux_b,
        rb_pad, jb_pad, xb_pad,
        E_log_beta_r, E_beta_col_sum,
        E_v, E_v_sq, E_gamma,
        a_const, ap_const, bp_const,
        S, K, supervised, has_aux,
    ):
        """One local CAVI iter, fully fused. Padding entries have xb_pad=0 so
        they contribute nothing to the scatter and are safe to leave in."""
        E_log_th_b = digamma(a_th) - _jnp.log(b_th)
        E_th_b = a_th / b_th
        E_xi_b = a_xi / b_xi

        # Multinomial responsibilities (fused softmax) × x at NNZ positions.
        z_vals = phi_chunk_core(E_log_th_b[rb_pad], E_log_beta_r[jb_pad], xb_pad)
        # rb_pad is sorted (CSR→COO) + 0-padding at end → still segment_sum-safe.
        z_sum_theta_b = jax.ops.segment_sum(z_vals, rb_pad, num_segments=S)

        if supervised:
            C_full_b = E_th_b @ E_v.T
            if has_aux:
                C_full_b = C_full_b + aux_b @ E_gamma.T
            term1 = -(kappa_y_b @ E_v)
            term2 = (omega_b @ E_v_sq) * E_th_b
            C_minus = C_full_b[:, :, None] - (E_th_b[:, None, :] * E_v[None, :, :])
            term3 = _jnp.sum(omega_b[:, :, None] * E_v[None, :, :] * C_minus, axis=1)
            R_il = term1 + term2 + term3
        else:
            R_il = _jnp.zeros_like(E_th_b)

        a_th_new = a_const + z_sum_theta_b
        b_th_new = _jnp.maximum(
            E_xi_b[:, None] + E_beta_col_sum[None, :] + R_il, 1e-3
        )
        a_xi_new = _jnp.full((S,), ap_const + K * a_const, dtype=_jnp.float32)
        E_th_row_sum = (a_th_new / b_th_new).sum(axis=1)
        b_xi_new = _jnp.maximum(bp_const + E_th_row_sum, 1e-3)

        return a_th_new, b_th_new, a_xi_new, b_xi_new, z_vals
else:
    _jit_inner_cavi_step = None

# Bucket size for nnz_b padding. nnz_b varies by ±tens-of-k per batch; rounding
# up to a 64k bucket gives ~10 unique padded sizes across a 1500-iter run, so
# we pay ~10 one-shot JIT compiles and reuse cached kernels afterwards. Each
# padded batch wastes at most BUCKET/avg_nnz_b ≈ 64k/2M = ~3% extra work.
_NNZ_BUCKET = 65536


def _pad_nnz(rb, jb, xb, bucket=_NNZ_BUCKET):
    """Pad rb, jb, xb to next multiple of bucket with safe dummies (xb=0)."""
    n = rb.shape[0]
    n_pad = ((n + bucket - 1) // bucket) * bucket
    if n_pad == n:
        return rb, jb, xb
    pad = n_pad - n
    rb = np.concatenate([rb, np.zeros(pad, dtype=rb.dtype)])
    jb = np.concatenate([jb, np.zeros(pad, dtype=jb.dtype)])
    xb = np.concatenate([xb, np.zeros(pad, dtype=xb.dtype)])
    return rb, jb, xb


def _auto_ell_chunk(nnz_b, target_gb=None):
    """Pick ell-chunk size so per-intermediate (nnz_b, Lc) array stays under target.

    Three intermediates are alive simultaneously in _update_r_pi (rw, rwo, log_ratio),
    so we budget for ~3 such arrays.
    """
    if target_gb is None:
        if HAS_GPU:
            try:
                import jax
                dev = [d for d in jax.devices() if d.platform == "gpu"][0]
                mem_bytes = dev.memory_stats()["bytes_limit"]
                target_gb = mem_bytes / (1024 ** 3) * 0.05
            except Exception:
                target_gb = 1.0
        else:
            target_gb = 2.0
    # nnz_b * Lc * 4 bytes per intermediate, 3 intermediates
    max_lc = int(target_gb * (1024 ** 3) / max(nnz_b * 4 * 3, 1))
    return max(8, max_lc)


# ============================================================================
# Config
# ============================================================================

@dataclass
class SVIConfig:
    """All hyperparameters for SVIPG.

    Sensible defaults are chosen to match vi_cavi.py's PG-CAVI mode so that
    a full-batch SVI run (batch_size = n_train) reproduces PG-CAVI within MC
    noise.
    """
    n_factors: int = 20

    # Gamma priors (HPF parameterisation)
    a: float = 0.3
    c: float = 0.3
    ap: float = 1.0
    cp: float = 1.0

    # Regression head priors
    b_v: float = 2.0                # Laplace scale for v
    sigma_gamma: float = 1.0        # Normal std for γ
    use_intercept: bool = True

    # Spike-slab on β
    use_spike_slab: bool = True
    alpha_pi: float = 1.0           # Beta(α, β·scale)
    beta_pi_scale: float = 5.0

    # SVI hyperparameters (Hoffman 2013)
    batch_size: int = 64
    tau0: float = 100.0             # delay — large value damps the very first
                                    # high-variance natural-gradient steps.
                                    # Hoffman recommends tau0 ≥ batch_size.
    kappa_lr: float = 0.7           # forgetting rate, κ ∈ (0.5, 1]
    n_local_iter: int = 3           # local CAVI sweeps per minibatch
    n_pg_subsweeps: int = 1         # ω samples drawn per local sweep
    pg_ema_alpha: float = 0.5       # EMA on E[ω] across visits to dampen MC noise
    # v warm-up: skip supervised correction to θ + skip updates to v, γ
    # for the first v_warmup outer iters.  Mirrors vi_cavi.py behaviour and
    # prevents the chicken-and-egg failure where v stays at 0 because θ has
    # not yet found a disease-aligned direction.
    v_warmup: int = 50

    # bp, dp are set data-driven in _initialize
    bp: float = -1.0
    dp: float = -1.0


# ============================================================================
# Helpers
# ============================================================================

def _log(x, floor=1e-30):
    return xp.log(xp.maximum(x, floor))


def _gamma_E_log(a, b):
    """E[log X] for X ~ Gamma(shape=a, rate=b)."""
    return digamma(a) - xp.log(b)


def _prepend_intercept(X_aux, use_intercept):
    if not use_intercept:
        return X_aux
    if X_aux is None or X_aux.size == 0:
        raise ValueError("intercept requested but X_aux is empty; provide either with intercept off or non-empty aux")
    ones = xp.ones((X_aux.shape[0], 1), dtype=X_aux.dtype)
    return xp.concatenate([ones, X_aux], axis=1)


# ============================================================================
# SVIPG
# ============================================================================

class SVIPG:
    """SVI for PG-augmented supervised Poisson factorization."""

    # ----------------------------------------------------------------
    # init / state
    # ----------------------------------------------------------------

    def __init__(self, cfg: SVIConfig, random_state: Optional[int] = None,
                 kappa: int = 1):
        if not _HAS_POLYAGAMMA:
            raise RuntimeError("polyagamma is required: pip install polyagamma")
        self.cfg = cfg
        self.kappa = int(kappa)        # number of binary outcomes
        self.seed = random_state
        self.rng = np.random.default_rng(random_state)
        self._pg_rng = np.random.default_rng(
            None if random_state is None else random_state + 1)

        # filled in by _initialize
        self.n = self.p = self.p_aux = None
        # global natural-parameter arrays
        self.a_beta = None    # (p, K)
        self.b_beta = None    # (p, K)
        self.a_eta = None     # (p,)
        self.b_eta = None     # (p,)
        self.rho_r = None     # (p, K) Bern posterior probability
        self.alpha_pi = None  # scalar Beta posterior α
        self.beta_pi = None   # scalar Beta posterior β
        # v, γ in mean/cov form (recomputed from natural params after each step)
        self.mu_v = None      # (kappa, K)
        self.Sigma_v = None   # (kappa, K, K)
        self.mu_gamma = None  # (kappa, p_aux)
        self.Sigma_gamma = None  # (kappa, p_aux, p_aux)
        # Bayesian-Lasso scales (per outcome, per factor) — sampled
        self.s_v = None       # (kappa, K)
        # ω posterior mean per cell per outcome (EMA-smoothed across visits)
        self.omega_pg_mean = None     # (n, kappa) — only entries for cells visited so far are meaningful

        # local cache — warm-start across minibatch visits
        self.a_theta_cache = None     # (n, K)
        self.b_theta_cache = None     # (n, K)
        self.a_xi_cache = None        # (n,)
        self.b_xi_cache = None        # (n,)

        # diagnostics
        self.holl_history_ = []        # list of (iter, holl)
        self.t_step = 0

    def _initialize(self, X, y, X_aux):
        cfg = self.cfg
        # ---- shape / type fixing --------------------------------------
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        self.X = X
        self.n, self.p = X.shape
        y2 = y if y.ndim > 1 else y[:, None]
        self.y = xp.asarray(y2.astype(np.float32))      # (n, kappa)
        if X_aux is None or (hasattr(X_aux, "size") and X_aux.size == 0):
            X_aux_dev = xp.zeros((self.n, 0), dtype=np.float32)
        else:
            # Standardise aux columns (training set only); reuse mean/sd at predict.
            arr = np.asarray(X_aux, dtype=np.float64)
            mu = arr.mean(0, keepdims=True)
            sd = arr.std(0, keepdims=True); sd[sd == 0] = 1.0
            self._aux_mu = mu; self._aux_sd = sd
            arr_std = (arr - mu) / sd
            X_aux_dev = xp.asarray(arr_std.astype(np.float32))
        if cfg.use_intercept:
            X_aux_dev = _prepend_intercept(X_aux_dev, True)
        self.X_aux = X_aux_dev
        self.p_aux = self.X_aux.shape[1]

        # ---- empirical rate priors (scHPF / Cemgil) -------------------
        def _mvr(M, axis):
            s = np.asarray(M.sum(axis=axis)).ravel().astype(np.float64)
            return float(s.mean() / max(s.var(), 1e-10))
        cfg.bp = float(np.clip(cfg.ap * _mvr(X, axis=1), 1e-4, 1e4))
        cfg.dp = float(np.clip(cfg.cp * _mvr(X, axis=0), 1e-4, 1e4))

        # ---- Gamma global states (β, η) -------------------------------
        K = cfg.n_factors
        rng = self.rng
        # Initialise q(β) at prior mean: a_β = c, b_β = dp (so E[β]=c/dp).
        self.a_beta = xp.asarray(rng.uniform(0.5 * cfg.c, 1.5 * cfg.c,
                                             size=(self.p, K)).astype(np.float32))
        self.b_beta = xp.full((self.p, K), float(cfg.dp), dtype=xp.float32)
        # q(η_j) shape is constant if no spike-slab. With spike-slab, the shape
        # depends on the *expected* number of active factors per gene (Σ_ℓ ρ_jℓ).
        self.a_eta = xp.full(self.p, cfg.cp + K * cfg.c, dtype=xp.float32)
        self.b_eta = xp.full(self.p, float(cfg.dp + K * cfg.c / cfg.dp), dtype=xp.float32)

        # ---- spike-slab on β ------------------------------------------
        if cfg.use_spike_slab:
            self.rho_r = xp.full((self.p, K), 0.5, dtype=xp.float32)
            self.alpha_pi = float(cfg.alpha_pi)
            self.beta_pi = float(cfg.alpha_pi * cfg.beta_pi_scale)
        else:
            self.rho_r = xp.ones((self.p, K), dtype=xp.float32)
            self.alpha_pi = float(cfg.alpha_pi)
            self.beta_pi = float(cfg.alpha_pi * cfg.beta_pi_scale)

        # ---- regression head (v, γ) -----------------------------------
        kappa = self.kappa
        self.mu_v = xp.asarray(rng.normal(0, 0.05, size=(kappa, K)).astype(np.float32))
        self.Sigma_v = xp.broadcast_to(
            (xp.eye(K, dtype=xp.float32) * (cfg.b_v ** 2))[None, :, :],
            (kappa, K, K),
        )
        self.Sigma_v = xp.array(self.Sigma_v)   # ensure writable
        if self.p_aux > 0:
            self.mu_gamma = xp.zeros((kappa, self.p_aux), dtype=xp.float32)
            # init intercept (column 0) at empirical log-odds when use_intercept
            if cfg.use_intercept:
                y_np = np.asarray(to_numpy(self.y))
                for k in range(kappa):
                    n_pos = int(y_np[:, k].sum())
                    n_neg = self.n - n_pos
                    if n_pos > 0 and n_neg > 0:
                        self.mu_gamma = (
                            self.mu_gamma.at[k, 0].set(float(np.log(n_pos / n_neg)))
                            if USE_JAX else self._mat_set(self.mu_gamma, (k, 0), np.log(n_pos / n_neg))
                        )
            self.Sigma_gamma = xp.broadcast_to(
                (xp.eye(self.p_aux, dtype=xp.float32) * (cfg.sigma_gamma ** 2))[None, :, :],
                (kappa, self.p_aux, self.p_aux),
            )
            self.Sigma_gamma = xp.array(self.Sigma_gamma)
        else:
            self.mu_gamma = xp.zeros((kappa, 0), dtype=xp.float32)
            self.Sigma_gamma = xp.zeros((kappa, 0, 0), dtype=xp.float32)

        # ---- Bayesian-Lasso scales -----------------------------------
        # E[s_kℓ] under Exp(1/(2 b_v²)) = 2 b_v² — init at prior mean
        self.s_v = xp.full((kappa, K), 2.0 * cfg.b_v ** 2, dtype=xp.float32)

        # ---- ω posterior-mean cache (per cell per outcome) ----------
        # PG(1, 0) has mean 1/4; start there.
        self.omega_pg_mean = xp.full((self.n, kappa), 0.25, dtype=xp.float32)

        # ---- local cache --------------------------------------------
        # Init θ_i at prior moments: a_θi = a, b_θi = bp
        self.a_theta_cache = xp.full((self.n, K), cfg.a, dtype=xp.float32)
        self.b_theta_cache = xp.full((self.n, K), float(cfg.bp), dtype=xp.float32)
        self.a_xi_cache = xp.full(self.n, cfg.ap + K * cfg.a, dtype=xp.float32)
        self.b_xi_cache = xp.full(self.n, float(cfg.bp), dtype=xp.float32)

        # CSR for fast row slicing of x_i in local CAVI
        self.X_csr = X.tocsr()

    # ----------------------------------------------------------------
    # Helpers for in-place sets compatible with both numpy & jnp
    # ----------------------------------------------------------------

    @staticmethod
    def _mat_set(arr, idx, val):
        """Generic 'arr[idx] = val' that also works for JAX arrays via .at."""
        if USE_JAX:
            return arr.at[idx].set(val)
        arr[idx] = val
        return arr

    @staticmethod
    def _row_set(arr, k, row):
        if USE_JAX:
            return arr.at[k].set(row)
        arr[k] = row
        return arr

    # ----------------------------------------------------------------
    # E[β], E[log β], E[r], etc.  (cheap, used in local CAVI)
    # ----------------------------------------------------------------

    def _E_beta(self):
        """E[β_jℓ · r_jℓ] = ρ_jℓ · (a_β/b_β).  Inactive β contribute 0."""
        return self.rho_r * (self.a_beta / self.b_beta)

    def _E_log_beta(self):
        """E[log β_jℓ] under q(β) — used only when r_jℓ=1."""
        return digamma(self.a_beta) - xp.log(self.b_beta)

    def _E_eta(self):
        return self.a_eta / self.b_eta

    # ----------------------------------------------------------------
    # LOCAL updates: per-cell CAVI on (z, θ, ξ) + PG sample of ω
    # ----------------------------------------------------------------

    def _local_step(self, batch_idx, supervised=True):
        """Run n_local_iter CAVI sweeps on the locals for cells in batch_idx.

        Updates self.a_theta_cache, self.b_theta_cache, self.a_xi_cache,
        self.b_xi_cache for those cells, and updates self.omega_pg_mean
        via EMA across visits.

        Returns the minibatch-collected statistics needed for global updates:
            E_theta_b      (S, K)
            E_log_theta_b  (S, K)
            z_local_per_j  (S, K) — actually z_{i,·,ℓ} summed over j ... no,
                                    we need per-(i, j, ℓ).  We aggregate to
                                    ss_z_per_factor_gene below.
            ss_z_gene       (p, K) — Σ_{i∈B} Σ_j z_{ijℓ}·[j==fixed]
                                    actually we need (p, K): Σ_i z_{ijℓ} for each j
            ss_theta_total (K,)    — Σ_{i∈B} E[θ_iℓ]  (for β's b update)
            omega_b        (S, kappa) — current E[ω_ib] per cell in batch
            kappa_y_b      (S, kappa) — y_b - 0.5
            E_theta_outer  (kappa, K, K) — Σ_{i∈B} ω_ib·E[θ_i]·E[θ_i]^T
            E_theta_aux    (kappa, K)    — Σ_{i∈B} E[θ_i]·(κ_ib - ω_ib·aux_i·γ_k)
            x_aux_outer    (kappa, p_aux, p_aux) — Σ ω·aux·aux^T
            x_aux_lin      (kappa, p_aux)        — Σ aux·(κ - ω·E[θ]·v_k)
        """
        cfg = self.cfg
        K = cfg.n_factors
        S = len(batch_idx)
        kappa = self.kappa

        # ---- pull state for the batch --------------------------------
        idx_dev = to_device(np.asarray(batch_idx, dtype=np.int64))
        a_th = self.a_theta_cache[idx_dev]    # (S, K)
        b_th = self.b_theta_cache[idx_dev]    # (S, K)
        a_xi = self.a_xi_cache[idx_dev]       # (S,)
        b_xi = self.b_xi_cache[idx_dev]       # (S,)
        aux_b = self.X_aux[idx_dev]           # (S, p_aux)
        y_b = self.y[idx_dev]                 # (S, kappa)

        # ---- pre-compute global expectations needed in local CAVI ---
        # E[β]·r and E[log β] depend only on globals — compute once per minibatch.
        E_beta_r = self._E_beta()                                 # (p, K)
        E_log_beta = self._E_log_beta()                           # (p, K)
        E_log_r = xp.log(xp.maximum(self.rho_r, 1e-30))            # (p, K)
        E_beta_col_sum = E_beta_r.sum(axis=0)                      # (K,)
        # Pre-fuse E[log β] + E[log r] so the inner CAVI loop reads a single
        # (p, K) tensor instead of two — saves one gather + one add per inner
        # iter, and lets phi_chunk_core fuse the rest of the softmax pipeline.
        E_log_beta_r = E_log_beta + E_log_r                       # (p, K)

        # For supervised correction in θ update we need E[v], E[v²] etc.
        E_v = self.mu_v                                           # (kappa, K)
        E_v_sq = self.mu_v ** 2 + xp.stack([
            xp.diag(self.Sigma_v[k]) for k in range(kappa)
        ], axis=0)                                                # (kappa, K)
        E_gamma = self.mu_gamma                                   # (kappa, p_aux)

        # current ω_b (EMA-smoothed mean from prior visits)
        omega_b = self.omega_pg_mean[idx_dev]                     # (S, kappa)
        kappa_y_b = y_b - 0.5                                     # (S, kappa)

        # We will collect z_{ijℓ} aggregated to (p, K) at end of local CAVI.
        # In CAVI z is the expected count, so we work with x_ij · φ_ijℓ where
        # φ is the multinomial responsibility.  We never materialise (S, p, K)
        # in dense form — instead we iterate over the nonzeros of X_b.

        # Slice X for the batch — CSR row slicing is fast.
        X_b_sp = self.X_csr[np.asarray(batch_idx)]   # (S, p) sparse
        X_b_coo = X_b_sp.tocoo()
        # local row indices within batch (0..S-1)
        rb = X_b_coo.row.astype(np.int32)
        jb = X_b_coo.col.astype(np.int32)
        xb = X_b_coo.data.astype(np.float32)         # (nnz_b,)
        # Pad to next BUCKET multiple so the JIT'd inner step compiles per
        # bucketed shape rather than per actual nnz_b (drops compile churn).
        rb, jb, xb = _pad_nnz(rb, jb, xb)
        rb_dev = to_device(rb); jb_dev = to_device(jb); xb_dev = to_device(xb)

        # ---- local CAVI iterations ----------------------------------
        # When JAX is available, dispatch to the JIT-fused single-kernel step.
        # Otherwise fall back to the eager NumPy path.
        if USE_JAX and _jit_inner_cavi_step is not None:
            has_aux = bool(self.p_aux > 0)
            for _it in range(cfg.n_local_iter):
                a_th, b_th, a_xi, b_xi, z_vals = _jit_inner_cavi_step(
                    a_th, b_th, a_xi, b_xi,
                    omega_b, kappa_y_b, aux_b,
                    rb_dev, jb_dev, xb_dev,
                    E_log_beta_r, E_beta_col_sum,
                    E_v, E_v_sq, E_gamma,
                    float(cfg.a), float(cfg.ap), float(cfg.bp),
                    int(S), int(K), bool(supervised), has_aux,
                )
        else:
            for _it in range(cfg.n_local_iter):
                E_log_th_b = digamma(a_th) - xp.log(b_th)
                E_th_b = a_th / b_th
                E_xi_b = a_xi / b_xi
                z_vals = phi_chunk_core(
                    E_log_th_b[rb_dev], E_log_beta_r[jb_dev], xb_dev,
                )
                z_sum_theta_b = scatter_add_to(
                    xp.zeros((S, K), dtype=xp.float32),
                    rb_dev, z_vals, sorted_indices=True,
                )
                if supervised:
                    C_full_b = (E_th_b @ E_v.T) + (aux_b @ E_gamma.T)
                    term1 = -(kappa_y_b @ E_v)
                    term2 = (omega_b @ E_v_sq) * E_th_b
                    C_minus = C_full_b[:, :, None] - (E_th_b[:, None, :] * E_v[None, :, :])
                    term3 = xp.sum(omega_b[:, :, None] * E_v[None, :, :] * C_minus, axis=1)
                    R_il = term1 + term2 + term3
                else:
                    R_il = xp.zeros_like(E_th_b)
                a_th = cfg.a + z_sum_theta_b
                b_th = xp.maximum(E_xi_b[:, None] + E_beta_col_sum[None, :] + R_il, 1e-3)
                a_xi = xp.full(S, cfg.ap + K * cfg.a, dtype=xp.float32)
                E_th_row_sum = (a_th / b_th).sum(axis=1)
                b_xi = xp.maximum(cfg.bp + E_th_row_sum, 1e-3)

        # --- ω_b: sample from PG(1, |E[ψ]|) (only if supervised) ---
        E_th_b_final = a_th / b_th                                # (S, K)
        if supervised:
            psi_b = (E_th_b_final @ E_v.T)                        # (S, kappa)
            if self.p_aux > 0:
                psi_b = psi_b + aux_b @ E_gamma.T
            psi_host = np.asarray(to_numpy(psi_b), dtype=np.float64)
            omega_samples = []
            for _ in range(self.cfg.n_pg_subsweeps):
                sample = np.asarray(
                    random_polyagamma(z=np.abs(psi_host).ravel(), size=psi_host.size),
                    dtype=np.float32,
                ).reshape(psi_host.shape)
                omega_samples.append(sample)
            omega_new = xp.asarray(np.mean(omega_samples, axis=0))   # (S, kappa)
            alpha = self.cfg.pg_ema_alpha
            omega_b = alpha * omega_new + (1.0 - alpha) * omega_b
        # else: omega_b remains at prior mean (PG(1,0) → 0.25)

        # ---- write locals + omega back to caches ---------------------
        self.a_theta_cache = self._scatter_rows(self.a_theta_cache, idx_dev, a_th)
        self.b_theta_cache = self._scatter_rows(self.b_theta_cache, idx_dev, b_th)
        self.a_xi_cache = self._scatter_idx(self.a_xi_cache, idx_dev, a_xi)
        self.b_xi_cache = self._scatter_idx(self.b_xi_cache, idx_dev, b_xi)
        self.omega_pg_mean = self._scatter_rows(self.omega_pg_mean, idx_dev, omega_b)

        # ---- aggregate sufficient statistics for global updates ------
        # ss_z_gene[j, ℓ] = Σ_{i∈B} z_{ijℓ} (final z, after CAVI). Single 2D scatter-add.
        ss_z_gene = xp.zeros((self.p, K), dtype=xp.float32)
        if USE_JAX:
            ss_z_gene = ss_z_gene.at[jb_dev].add(z_vals)
        else:
            np.add.at(ss_z_gene, np.asarray(jb_dev), np.asarray(z_vals))

        # ss_theta_total[ℓ] = Σ_{i∈B} E[θ_iℓ]
        ss_theta_total = (a_th / b_th).sum(axis=0)                # (K,)
        E_th_b_final = a_th / b_th                                # (S, K)

        # For v_k update — per-outcome k:
        #   prec_v += Σ_{i∈B} ω_ik · E[θ_i] E[θ_i]^T
        #   mu_lin += Σ_{i∈B} E[θ_i] · (κ_ik − ω_ik · aux_i^T E[γ_k])
        ss_v_outer = xp.zeros((kappa, K, K), dtype=xp.float32)
        ss_v_lin = xp.zeros((kappa, K), dtype=xp.float32)
        # also for γ:
        ss_g_outer = xp.zeros((kappa, self.p_aux, self.p_aux), dtype=xp.float32)
        ss_g_lin = xp.zeros((kappa, self.p_aux), dtype=xp.float32)
        for k in range(kappa):
            om_k = omega_b[:, k]                                  # (S,)
            kap_k = kappa_y_b[:, k]                               # (S,)
            # v
            ss_v_outer = self._row_set(ss_v_outer, k,
                                        (E_th_b_final.T * om_k) @ E_th_b_final)
            aux_off = aux_b @ E_gamma[k] if self.p_aux > 0 else xp.zeros(S, dtype=xp.float32)
            ss_v_lin = self._row_set(ss_v_lin, k,
                                      E_th_b_final.T @ (kap_k - om_k * aux_off))
            # γ
            if self.p_aux > 0:
                ss_g_outer = self._row_set(ss_g_outer, k,
                                            (aux_b.T * om_k) @ aux_b)
                theta_v_k = E_th_b_final @ E_v[k]
                ss_g_lin = self._row_set(ss_g_lin, k,
                                          aux_b.T @ (kap_k - om_k * theta_v_k))

        return {
            "S": S,
            "ss_z_gene": ss_z_gene,
            "ss_theta_total": ss_theta_total,
            "ss_v_outer": ss_v_outer,
            "ss_v_lin": ss_v_lin,
            "ss_g_outer": ss_g_outer,
            "ss_g_lin": ss_g_lin,
            "E_theta_b": E_th_b_final,
            "omega_b": omega_b,
            # NNZ indices for _update_r_pi (avoid recomputing in r-pi step)
            "rb_dev": rb_dev,
            "jb_dev": jb_dev,
            "xb_dev": xb_dev,
        }

    @staticmethod
    def _scatter_rows(target, rows, new_rows):
        """target[rows, :] = new_rows."""
        if USE_JAX:
            return target.at[rows].set(new_rows)
        target[np.asarray(rows)] = np.asarray(new_rows)
        return target

    @staticmethod
    def _scatter_idx(target, idx, new_vals):
        """target[idx] = new_vals (1-D target)."""
        if USE_JAX:
            return target.at[idx].set(new_vals)
        target[np.asarray(idx)] = np.asarray(new_vals)
        return target

    # ----------------------------------------------------------------
    # GLOBAL stochastic-natural-gradient updates
    # ----------------------------------------------------------------

    def _update_globals(self, ss, rho_t, supervised=True):
        """Apply stochastic natural gradient with step size rho_t to global params.

        If supervised=False (during v_warmup), v / γ / s are NOT updated —
        only the Poisson-side globals (β, η, r, π) move.
        """
        cfg = self.cfg
        scale = self.n / ss["S"]

        # --- β ---------------------------------------------------------
        # Full-batch natural params (active component, given r_jℓ):
        #     a_β = c + Σ_i z_{ijℓ}        b_β = E[η_j] + Σ_i E[θ_iℓ]
        E_eta = self._E_eta()                                  # (p,)
        a_beta_hat = cfg.c + scale * ss["ss_z_gene"]           # (p, K)
        b_beta_hat = E_eta[:, None] + scale * ss["ss_theta_total"][None, :]
        self.a_beta = (1.0 - rho_t) * self.a_beta + rho_t * a_beta_hat
        self.b_beta = (1.0 - rho_t) * self.b_beta + rho_t * xp.maximum(b_beta_hat, 1e-3)

        # --- η (no minibatch dependence — depends only on β) -----------
        # b_η_hat = dp + Σ_ℓ E[β_jℓ · r_jℓ]
        E_beta_r = self._E_beta()                              # (p, K)
        a_eta_hat = cfg.cp + cfg.c * self.rho_r.sum(axis=1)
        b_eta_hat = cfg.dp + E_beta_r.sum(axis=1)
        self.a_eta = (1.0 - rho_t) * self.a_eta + rho_t * a_eta_hat
        self.b_eta = (1.0 - rho_t) * self.b_eta + rho_t * xp.maximum(b_eta_hat, 1e-3)

        # --- spike-slab r, π -------------------------------------------
        if cfg.use_spike_slab:
            self._update_r_pi(ss, rho_t)

        if not supervised:
            # Skip v / γ / s updates during Poisson-only warmup
            return

        # --- v (per outcome) -------------------------------------------
        # Natural params η_1 = Σ⁻¹μ, η_2 = -½Σ⁻¹.  Hoffman 2013 SVI: minibatch
        # sufficient statistics are scaled by n/|B| to estimate the full-batch
        # natural parameter, then blended with previous via rho.
        kappa = self.kappa
        for k in range(kappa):
            Dinv = xp.diag(1.0 / xp.maximum(self.s_v[k], 1e-12))
            eta2_hat = -0.5 * (Dinv + scale * ss["ss_v_outer"][k])
            eta1_hat = scale * ss["ss_v_lin"][k]
            # current natural params
            Sigma_inv_old = xp.linalg.inv(self.Sigma_v[k])
            eta2_old = -0.5 * Sigma_inv_old
            eta1_old = Sigma_inv_old @ self.mu_v[k]
            eta2_new = (1.0 - rho_t) * eta2_old + rho_t * eta2_hat
            eta1_new = (1.0 - rho_t) * eta1_old + rho_t * eta1_hat
            # recover Σ, μ
            Sigma_v_new = xp.linalg.inv(-2.0 * eta2_new + xp.eye(cfg.n_factors, dtype=xp.float32) * 1e-6)
            mu_v_new = Sigma_v_new @ eta1_new
            self.Sigma_v = self._row_set(self.Sigma_v, k, Sigma_v_new)
            self.mu_v = self._row_set(self.mu_v, k, mu_v_new)

        # --- γ (per outcome) -------------------------------------------
        if self.p_aux > 0:
            for k in range(kappa):
                Ig = xp.eye(self.p_aux, dtype=xp.float32) * (1.0 / (cfg.sigma_gamma ** 2))
                eta2_hat = -0.5 * (Ig + scale * ss["ss_g_outer"][k])
                eta1_hat = scale * ss["ss_g_lin"][k]
                Sg_inv_old = xp.linalg.inv(self.Sigma_gamma[k])
                eta2_old = -0.5 * Sg_inv_old
                eta1_old = Sg_inv_old @ self.mu_gamma[k]
                eta2_new = (1.0 - rho_t) * eta2_old + rho_t * eta2_hat
                eta1_new = (1.0 - rho_t) * eta1_old + rho_t * eta1_hat
                Sigma_g_new = xp.linalg.inv(-2.0 * eta2_new + xp.eye(self.p_aux, dtype=xp.float32) * 1e-6)
                mu_g_new = Sigma_g_new @ eta1_new
                self.Sigma_gamma = self._row_set(self.Sigma_gamma, k, Sigma_g_new)
                self.mu_gamma = self._row_set(self.mu_gamma, k, mu_g_new)

        # --- s (Park-Casella) ------------------------------------------
        self._update_s()

    def _update_r_pi(self, ss, rho_t):
        """Stochastic natural gradient on spike-slab indicators r_jℓ + Beta π.

        log_LR[j, ℓ] ≈ Σ_i x_ij·log(rate_with / rate_without) − (rate_with − rate_without)
            rate_with  = rate_b[i,j] + (1−ρ_jℓ)·θ_iℓ·β_jℓ
            rate_without = rate_b[i,j] − ρ_jℓ·θ_iℓ·β_jℓ
            ⇒ rate_with − rate_without = θ_iℓ·β_jℓ  (closed form for the rate term)

        Implementation: stay GPU-resident, evaluate the log-ratio only at the NNZ
        positions of x_ij (everything else multiplies by 0), then scatter-add by
        gene index. Chunk over ℓ to bound (nnz_b, Lc) temporaries.
        """
        cfg = self.cfg
        K = cfg.n_factors
        p = self.p

        E_th_b = ss["E_theta_b"]                                  # (S, K)
        E_beta = self.a_beta / self.b_beta                         # (p, K)
        rho_r = self.rho_r                                         # (p, K)
        E_beta_r = E_beta * rho_r                                  # (p, K)

        # NNZ indices for the minibatch (stashed by _local_step).
        rb_dev = ss["rb_dev"]
        jb_dev = ss["jb_dev"]
        xb_dev = ss["xb_dev"]

        # rate_b is (S, p) — for S=512, p=13k it is ~26MB, cheap on GPU. Gather
        # to NNZ-only (nnz_b,) so the chunked loop never materialises (S, p) again.
        rate_b = E_th_b @ E_beta_r.T                               # (S, p)
        rate_b = xp.maximum(rate_b, 1e-10)
        rate_at_nnz = rate_b[rb_dev, jb_dev]                       # (nnz_b,)

        log_pi_ratio = float(np.log(self.alpha_pi / max(self.beta_pi, 1e-12)))
        scale = self.n / ss["S"]

        # term_rate is dense and closed-form: β_jℓ · Σ_i θ_iℓ (no (S,p) materialisation).
        ss_theta_total = ss["ss_theta_total"]                      # (K,)
        term_rate = E_beta * ss_theta_total[None, :]               # (p, K)

        # term_log: chunked (nnz_b, Lc) computation, scatter-add by jb_dev.
        nnz_b = int(rb_dev.shape[0])
        Lc = _auto_ell_chunk(nnz_b)
        jb_np = np.asarray(jb_dev) if not USE_JAX else None
        term_log_chunks = []
        for ell_start in range(0, K, Lc):
            ell_end = min(ell_start + Lc, K)
            Lc_c = ell_end - ell_start
            theta_c = E_th_b[:, ell_start:ell_end][rb_dev]         # (nnz_b, Lc_c)
            beta_c = E_beta[:, ell_start:ell_end][jb_dev]          # (nnz_b, Lc_c)
            rho_c = rho_r[:, ell_start:ell_end][jb_dev]            # (nnz_b, Lc_c)
            c = theta_c * beta_c
            rw = rate_at_nnz[:, None] + (1.0 - rho_c) * c
            rwo = rate_at_nnz[:, None] - rho_c * c
            log_ratio = xp.log(xp.maximum(rw, 1e-12)) - xp.log(xp.maximum(rwo, 1e-12))
            weighted = xb_dev[:, None] * log_ratio                 # (nnz_b, Lc_c)
            slab = xp.zeros((p, Lc_c), dtype=xp.float32)
            if USE_JAX:
                slab = slab.at[jb_dev].add(weighted)
            else:
                np.add.at(slab, jb_np, weighted)
            term_log_chunks.append(slab)
        term_log = xp.concatenate(term_log_chunks, axis=1)         # (p, K)

        log_LR = scale * (term_log - term_rate) + log_pi_ratio
        rho_r_hat = _expit(log_LR)
        self.rho_r = (1.0 - rho_t) * self.rho_r + rho_t * rho_r_hat

        # π posterior: stochastic natural gradient on Beta natural params
        sum_r = float(to_numpy(self.rho_r.sum()))
        total_r = float(self.p * K)
        alpha_hat = cfg.alpha_pi + sum_r
        beta_hat = cfg.alpha_pi * cfg.beta_pi_scale + (total_r - sum_r)
        self.alpha_pi = (1.0 - rho_t) * self.alpha_pi + rho_t * alpha_hat
        self.beta_pi = (1.0 - rho_t) * self.beta_pi + rho_t * beta_hat

    def _update_s(self):
        """Park-Casella inverse-Gaussian update for Bayesian-Lasso scales.

        Vectorized: numpy.Generator.wald accepts array means.
        """
        bv = self.cfg.b_v
        bv2_inv = 1.0 / (bv * bv) if bv > 0 else 1e6
        mu_v_h = np.abs(np.asarray(to_numpy(self.mu_v), dtype=np.float64))
        # mean[k, ℓ] = 1/(bv * |v_kℓ|), with a huge fallback when v ≈ 0
        means = np.where(mu_v_h < 1e-9, 1e9, 1.0 / (bv * np.maximum(mu_v_h, 1e-12)))
        inv_s = self._pg_rng.wald(mean=means, scale=bv2_inv)
        s_new = (1.0 / np.maximum(inv_s, 1e-12)).astype(np.float32)
        self.s_v = to_device(s_new)

    # ----------------------------------------------------------------
    # HOLL on validation set
    # ----------------------------------------------------------------

    def _holl(self, X_val, y_val, X_aux_val, n_iter=20):
        """Held-out predictive log-likelihood under current globals.

        Infers θ_val on GPU via _infer_theta (chunked NNZ scatter), then computes
        Bernoulli LL using mu_v / mu_gamma. All on device.
        """
        E_th = self._infer_theta(X_val, n_iter=n_iter)   # xp (nv, K) — device array
        nv = int(E_th.shape[0])
        logits = E_th @ self.mu_v.T                       # (nv, kappa)
        if self.p_aux > 0:
            aux_v_h = self._aux_standardise(X_aux_val, nv)
            if aux_v_h is not None and aux_v_h.shape[1] > 0:
                logits = logits + to_device(aux_v_h) @ self.mu_gamma.T
        y2 = np.asarray(y_val, dtype=np.float32)
        if y2.ndim == 1:
            y2 = y2[:, None]
        y2_dev = to_device(y2)
        log_sig = log_expit(logits)
        log_1msig = log_expit(-logits)
        ll_per = y2_dev * log_sig + (1.0 - y2_dev) * log_1msig
        return float(to_numpy(ll_per.mean()))

    # ----------------------------------------------------------------
    # Predict
    # ----------------------------------------------------------------

    def _infer_theta(self, X_new, n_iter=20):
        """Run unsupervised local CAVI to infer E[θ] for new cells under frozen globals.

        GPU-resident with chunked NNZ scatter — the (nnz, K) intermediates are
        partitioned into chunks of `nnz_chunk` rows so peak memory stays bounded.
        Returns an xp array on the active device (caller converts as needed).
        """
        if not sp.issparse(X_new):
            X_new = sp.csr_matrix(X_new)
        n_new, _ = X_new.shape
        cfg = self.cfg
        K = cfg.n_factors

        a_th = xp.full((n_new, K), cfg.a, dtype=xp.float32)
        b_th = xp.full((n_new, K), float(cfg.bp), dtype=xp.float32)
        a_xi = xp.full((n_new,), cfg.ap + K * cfg.a, dtype=xp.float32)
        b_xi = xp.full((n_new,), float(cfg.bp), dtype=xp.float32)

        E_log_beta = self._E_log_beta()            # (p, K) — already on device
        E_beta_r = self._E_beta()                   # (p, K)
        E_beta_col_sum = E_beta_r.sum(axis=0)       # (K,)

        X_coo = X_new.tocoo()
        rb = X_coo.row.astype(np.int32)
        jb = X_coo.col.astype(np.int32)
        xb = X_coo.data.astype(np.float32)
        rb_dev = to_device(rb)
        jb_dev = to_device(jb)
        xb_dev = to_device(xb)
        nnz = rb.shape[0]

        # Chunk size: target ~256MB per (chunk, K) intermediate.
        # Five such intermediates are alive at peak (logits, mx_shift, e, phi, z_vals);
        # JAX will free intermediates between ops, so a 256MB budget gives ~1GB peak.
        nnz_chunk = max(20_000, int(256 * 1024 * 1024 / max(K * 4, 1)))
        nnz_chunk = min(nnz_chunk, nnz)

        for _ in range(n_iter):
            E_log_th = digamma(xp.maximum(a_th, 1e-9)) - xp.log(b_th)   # (n_new, K)
            z_sum = xp.zeros((n_new, K), dtype=xp.float32)
            for s in range(0, nnz, nnz_chunk):
                e = min(s + nnz_chunk, nnz)
                rb_c = rb_dev[s:e]
                jb_c = jb_dev[s:e]
                xb_c = xb_dev[s:e]
                z_vals = phi_chunk_core(E_log_th[rb_c], E_log_beta[jb_c], xb_c)
                # rb_c is a slice of the globally-sorted rb_dev → still sorted.
                z_sum = scatter_add_to(z_sum, rb_c, z_vals, sorted_indices=True)
            a_th = cfg.a + z_sum
            b_th = (a_xi / b_xi)[:, None] + E_beta_col_sum[None, :]
            E_th_row_sum = (a_th / b_th).sum(axis=1)
            b_xi = xp.maximum(cfg.bp + E_th_row_sum, 1e-3)
        return a_th / b_th

    def _aux_standardise(self, X_aux_new, n_new):
        """Apply training-set standardisation + intercept to new aux features."""
        if self.p_aux == 0:
            return None
        if X_aux_new is None or (hasattr(X_aux_new, "size") and X_aux_new.size == 0):
            aux = np.zeros((n_new, 0), dtype=np.float32)
        else:
            aux = ((np.asarray(X_aux_new, dtype=np.float64) - self._aux_mu) / self._aux_sd).astype(np.float32)
        if self.cfg.use_intercept:
            aux = np.hstack([np.ones((n_new, 1), dtype=np.float32), aux])
        return aux

    def transform(self, X_new, X_aux_new=None, n_iter=20):
        """Return {'E_theta': (n,K), 'proba': (n,kappa)} for new cells (numpy)."""
        E_th_dev = self._infer_theta(X_new, n_iter=n_iter)
        E_th = np.asarray(to_numpy(E_th_dev), dtype=np.float32)
        n_new = E_th.shape[0]
        mu_v_h = np.asarray(to_numpy(self.mu_v))
        mu_g_h = np.asarray(to_numpy(self.mu_gamma))
        logits = E_th @ mu_v_h.T
        aux_v = self._aux_standardise(X_aux_new, n_new)
        if aux_v is not None and aux_v.shape[1] > 0:
            logits = logits + aux_v @ mu_g_h.T
        proba = 1.0 / (1.0 + np.exp(-logits))
        return {"E_theta": E_th, "proba": proba}

    def predict_proba(self, X_new, X_aux_new=None, n_iter=20):
        """P(y=1 | X_new) using current globals; θ_new inferred via local CAVI."""
        return self.transform(X_new, X_aux_new=X_aux_new, n_iter=n_iter)["proba"]

    # ----------------------------------------------------------------
    # FIT
    # ----------------------------------------------------------------

    def fit(self, X, y, X_aux=None,
            X_val=None, y_val=None, X_aux_val=None,
            max_iter=2000, check_freq=50,
            verbose=True, print_every=10):
        """Run the SVI loop.

        Parameters
        ----------
        X, y : training data (sparse counts, binary labels)
        X_aux : auxiliary features (n, p_aux) (excluding intercept)
        X_val, y_val, X_aux_val : optional validation set for HOLL tracking
        max_iter : number of stochastic outer iterations
        check_freq : compute HOLL every this many iters
        """
        self._initialize(X, y, X_aux)

        cfg = self.cfg
        t0 = time.time()

        for t in range(max_iter):
            self.t_step = t
            rho_t = (cfg.tau0 + t) ** (-cfg.kappa_lr)

            # 1. minibatch
            S = min(cfg.batch_size, self.n)
            batch_idx = self.rng.choice(self.n, size=S, replace=False).astype(np.int64)
            supervised = (t >= cfg.v_warmup)

            # 2. local CAVI + (if supervised) ω sample
            ss = self._local_step(batch_idx, supervised=supervised)

            # 3. + 4. global stochastic-natural-gradient update
            self._update_globals(ss, rho_t, supervised=supervised)

            # diagnostics
            if verbose and (t % print_every == 0 or t == max_iter - 1):
                mu_v_h = np.asarray(to_numpy(self.mu_v))
                msg = (f"iter {t:4d}/{max_iter}  rho={rho_t:.4f}  "
                       f"||v||={np.linalg.norm(mu_v_h):.3f}  "
                       f"|v|_max={np.abs(mu_v_h).max():.3f}  "
                       f"E[π]={self.alpha_pi/(self.alpha_pi+self.beta_pi):.3f}  "
                       f"elapsed={time.time()-t0:.1f}s")
                print(msg, flush=True)

            if X_val is not None and (t % check_freq == 0 or t == max_iter - 1):
                holl = self._holl(X_val, y_val, X_aux_val, n_iter=20)
                self.holl_history_.append((t, holl))
                if verbose:
                    print(f"  HOLL t={t}: {holl:.5f}", flush=True)

        return self

    # ----------------------------------------------------------------
    # Save outputs (CAVI-compatible format)
    # ----------------------------------------------------------------

    def save(
        self,
        out_dir,
        gene_list=None,
        splits=None,
        label_columns=None,
        aux_columns=None,
        val_test_data=None,
        cell_metadata=None,
        prefix: str = "svi",
        n_transform_iter: int = 20,
    ):
        """Save outputs in a layout aligned with VI's `save_results` (utils.py).

        Produces the same set of CSV/NPZ/JSON files as VI so the same downstream
        analysis utilities work.  All parameter values are taken as the **final
        iterate** λ_T of the Hoffman & Blei 2013 SVI loop (single-iterate; the
        Robbins-Monro step-size schedule ρ_t = (τ_0 + t)^(-κ) is the variance-
        control mechanism, per the paper).

        Files produced (prefix=svi by default):
            svi_model_params.npz         — essential params for re-loading
            svi_gene_programs.csv.gz     — programs × genes, with v_weight_<label> cols
            svi_r_beta.csv.gz            — spike-slab posterior probs (if enabled)
            svi_theta_train.csv.gz       — E[θ] cached at last visit of training cells
            svi_theta_val.csv.gz         — E[θ] for val cells (inferred via local CAVI)
            svi_theta_test.csv.gz        — E[θ] for test cells (inferred via local CAVI)
            svi_gamma_weights.csv.gz     — kappa × p_aux, with intercept + aux col names
            svi_gamma_variance.csv.gz    — diag(Sigma_gamma)
            svi_v_weights.csv.gz         — kappa × K (mirror of beta_df's v_weight cols)
            svi_v_variance.csv.gz        — diag(Sigma_v) per outcome
            svi_holl_history.csv         — held-out log-lik trajectory
            svi_summary.json.gz          — hyperparams, history, dimensions, perf
            v_vector_seed*.npy           — flat v for stability tracking
        """
        import pandas as pd, json, gzip
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

        ext = ".csv.gz"
        compression = "gzip"

        K = self.cfg.n_factors
        prog_labels = [f"GP{i+1}" for i in range(K)]

        # ---- gene names ---------------------------------------------------
        if gene_list is None:
            gene_names = [f"gene_{j}" for j in range(self.p)]
        else:
            gene_names = list(gene_list)
            if len(gene_names) != self.p:
                # fall back to numeric labels if a mismatch slips through
                gene_names = [f"gene_{j}" for j in range(self.p)]

        # ---- label / aux names -------------------------------------------
        if label_columns is not None and len(label_columns) == self.kappa:
            outcome_names = list(label_columns)
        else:
            outcome_names = [f"outcome_{k}" for k in range(self.kappa)]

        # gamma column names — intercept (if any) + aux columns
        n_gamma_cols = self.p_aux
        has_intercept = bool(self.cfg.use_intercept) and n_gamma_cols > 0
        if aux_columns is not None:
            if has_intercept and len(aux_columns) == n_gamma_cols - 1:
                gamma_col_names = ["intercept"] + list(aux_columns)
            elif len(aux_columns) == n_gamma_cols:
                gamma_col_names = list(aux_columns)
            else:
                gamma_col_names = (["intercept"] if has_intercept else []) + \
                    [f"aux_{j}" for j in range(n_gamma_cols - (1 if has_intercept else 0))]
        else:
            gamma_col_names = (["intercept"] if has_intercept else []) + \
                [f"aux_{j}" for j in range(n_gamma_cols - (1 if has_intercept else 0))]

        # ---- pull final-iterate params to host ---------------------------
        E_beta = np.asarray(to_numpy(self._E_beta()))                     # (p, K)
        E_log_beta = np.asarray(to_numpy(self._E_log_beta()))             # (p, K)
        mu_v = np.asarray(to_numpy(self.mu_v))                            # (kappa, K)
        Sigma_v = np.asarray(to_numpy(self.Sigma_v))                      # (kappa, K, K)
        mu_gamma = np.asarray(to_numpy(self.mu_gamma))                    # (kappa, p_aux)
        Sigma_gamma = np.asarray(to_numpy(self.Sigma_gamma))              # (kappa, p_aux, p_aux)
        s_v = np.asarray(to_numpy(self.s_v))                              # (kappa, K)
        rho_r = np.asarray(to_numpy(self.rho_r))                          # (p, K)

        # ---- 1. gene_programs CSV (programs × genes + v_weight cols) ----
        beta_df = pd.DataFrame(E_beta.T, index=prog_labels, columns=gene_names)
        for k in range(self.kappa):
            beta_df.insert(k, f"v_weight_{outcome_names[k]}", mu_v[k])
        beta_df.to_csv(out / f"{prefix}_gene_programs{ext}", compression=compression)

        # ---- 2. r_beta CSV (programs × genes) ---------------------------
        if self.cfg.use_spike_slab:
            r_df = pd.DataFrame(rho_r.T, index=prog_labels, columns=gene_names)
            r_df.to_csv(out / f"{prefix}_r_beta{ext}", compression=compression)

        # ---- 3. v weights CSV (kappa × K) -------------------------------
        v_df = pd.DataFrame(mu_v,
                            index=outcome_names,
                            columns=prog_labels)
        v_df.index.name = "label"
        v_df.to_csv(out / f"{prefix}_v_weights{ext}", compression=compression)

        # ---- 4. v variance CSV (diag of Sigma_v per outcome) ------------
        v_var = np.stack([np.diag(Sigma_v[k]) for k in range(self.kappa)])
        v_var_df = pd.DataFrame(v_var,
                                index=outcome_names,
                                columns=prog_labels)
        v_var_df.index.name = "label"
        v_var_df.to_csv(out / f"{prefix}_v_variance{ext}", compression=compression)

        # ---- 5. gamma weights / variance CSVs ---------------------------
        if mu_gamma.size > 0:
            g_df = pd.DataFrame(mu_gamma,
                                index=outcome_names,
                                columns=gamma_col_names)
            g_df.index.name = "label"
            g_df.to_csv(out / f"{prefix}_gamma_weights{ext}", compression=compression)

            gamma_var = np.stack([np.diag(Sigma_gamma[k]) for k in range(self.kappa)])
            gv_df = pd.DataFrame(gamma_var,
                                 index=outcome_names,
                                 columns=gamma_col_names)
            gv_df.index.name = "label"
            gv_df.to_csv(out / f"{prefix}_gamma_variance{ext}", compression=compression)

        # ---- 6. theta CSVs (train always; val/test if val_test_data) ----
        E_theta_train = np.asarray(to_numpy(self.a_theta_cache /
                                            xp.maximum(self.b_theta_cache, 1e-12)))
        train_idx = (splits["train"] if splits is not None and "train" in splits
                     else [f"cell_{i}" for i in range(E_theta_train.shape[0])])

        def _add_meta(df):
            if cell_metadata is not None:
                common = df.index.intersection(cell_metadata.index)
                if len(common) > 0:
                    for col in cell_metadata.columns:
                        df.insert(0, col, cell_metadata.reindex(df.index)[col])
            return df

        theta_train_df = pd.DataFrame(E_theta_train,
                                      index=pd.Index(train_idx, name="cell_id"),
                                      columns=prog_labels)
        _add_meta(theta_train_df)
        theta_train_df.to_csv(out / f"{prefix}_theta_train{ext}", compression=compression)

        if val_test_data is not None and splits is not None:
            for split_name in ("val", "test"):
                X_key = f"X_{split_name}"
                if X_key not in val_test_data or split_name not in splits:
                    continue
                X_split = val_test_data[X_key]
                X_aux_split = val_test_data.get(f"X_aux_{split_name}")
                E_th = self._infer_theta(X_split, n_iter=n_transform_iter)
                df = pd.DataFrame(E_th,
                                  index=pd.Index(splits[split_name], name="cell_id"),
                                  columns=prog_labels)
                _add_meta(df)
                df.to_csv(out / f"{prefix}_theta_{split_name}{ext}", compression=compression)

        # ---- 7. holl history CSV ---------------------------------------
        if self.holl_history_:
            pd.DataFrame(self.holl_history_, columns=["iter", "holl"]).to_csv(
                out / f"{prefix}_holl_history.csv", index=False)

        # ---- 8. essential model params NPZ -----------------------------
        essential = {
            "n_factors": K,
            "a": self.cfg.a, "c": self.cfg.c,
            "b_v": self.cfg.b_v, "sigma_gamma": self.cfg.sigma_gamma,
            "E_beta": E_beta, "E_log_beta": E_log_beta,
            "mu_v": mu_v, "Sigma_v": Sigma_v, "sigma_v_diag": v_var,
            "mu_gamma": mu_gamma, "Sigma_gamma": Sigma_gamma,
            "s_v": s_v, "rho_r": rho_r,
            "alpha_pi": float(self.alpha_pi), "beta_pi": float(self.beta_pi),
            "a_beta": np.asarray(to_numpy(self.a_beta)),
            "b_beta": np.asarray(to_numpy(self.b_beta)),
            "a_eta": np.asarray(to_numpy(self.a_eta)),
            "b_eta": np.asarray(to_numpy(self.b_eta)),
            "aux_mu": getattr(self, "_aux_mu", None),
            "aux_sd": getattr(self, "_aux_sd", None),
            "n": self.n, "p": self.p, "p_aux": self.p_aux,
            "kappa": self.kappa,
            "use_intercept": bool(self.cfg.use_intercept),
            "holl_history": np.array(self.holl_history_) if self.holl_history_ else np.zeros((0, 2)),
            "t_step": self.t_step,
        }
        # Strip Nones (np.savez_compressed can't take them)
        essential = {k: v for k, v in essential.items() if v is not None}
        np.savez_compressed(out / f"{prefix}_model_params.npz", **essential)

        # ---- 9. flat v vector for stability tracking -------------------
        if self.seed is not None:
            np.save(out / f"v_vector_seed{self.seed}.npy", mu_v.flatten())

        # ---- 10. summary JSON ------------------------------------------
        summary = {
            "config": self.cfg.__dict__,
            "model_class": "SVIPG",
            "algorithm_reference": "Hoffman, Blei, Wang & Paisley (2013) Algorithm 1 (single final iterate; Robbins-Monro step size).",
            "kappa": self.kappa,
            "label_columns": outcome_names,
            "aux_columns": list(aux_columns) if aux_columns is not None else None,
            "data_shapes": {"n": self.n, "p": self.p, "p_aux": self.p_aux},
            "training": {
                "n_outer_iter": int(self.t_step) + 1,
                "final_holl": (self.holl_history_[-1][1]
                               if self.holl_history_ else None),
                "v_norm": float(np.linalg.norm(mu_v)),
                "v_max_abs": float(np.abs(mu_v).max()),
                "gamma_norm": float(np.linalg.norm(mu_gamma)) if mu_gamma.size else None,
                "E_pi": float(self.alpha_pi / (self.alpha_pi + self.beta_pi))
                        if self.cfg.use_spike_slab else None,
                "rho_r_mean_active": float((rho_r > 0.5).mean())
                                     if self.cfg.use_spike_slab else None,
            },
            "holl_history": self.holl_history_,
        }

        def _to_jsonable(obj):
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(x) for x in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if hasattr(obj, "__array__"):
                arr = np.array(obj)
                return arr.item() if arr.ndim == 0 else arr.tolist()
            if obj is None or isinstance(obj, (int, float, str, bool)):
                return obj
            return str(obj)

        summary = _to_jsonable(summary)
        with gzip.open(out / f"{prefix}_summary.json.gz", "wt") as f:
            json.dump(summary, f, indent=2)


# ============================================================================
# CLI
# ============================================================================

def _load_h5ad(path, label_col, aux_cols):
    import anndata as ad
    a = ad.read_h5ad(path)
    X = a.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    y = a.obs[label_col].values.astype(np.int32)
    aux = a.obs[aux_cols].values.astype(np.float32) if aux_cols else None
    return X, y, aux, list(a.var_names), list(a.obs_names)


def main():
    import argparse
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to .h5ad")
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--aux-columns", nargs="+", default=[])
    ap.add_argument("--n-factors", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tau0", type=float, default=1.0)
    ap.add_argument("--kappa-lr", type=float, default=0.7)
    ap.add_argument("--n-local-iter", type=int, default=3)
    ap.add_argument("--n-pg-subsweeps", type=int, default=1)
    ap.add_argument("--pg-ema-alpha", type=float, default=0.5)
    ap.add_argument("--v-warmup", type=int, default=50)
    ap.add_argument("--a", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=0.3)
    ap.add_argument("--b-v", type=float, default=2.0)
    ap.add_argument("--sigma-gamma", type=float, default=1.0)
    ap.add_argument("--use-spike-slab", action="store_true", default=True)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--check-freq", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    X, y, aux, gene_names, cell_names = _load_h5ad(
        args.data, args.label_column, args.aux_columns)
    # split
    idx = np.arange(X.shape[0])
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, stratify=y, random_state=args.seed)
    val_idx, te_idx = train_test_split(te_idx, test_size=0.5,
                                       stratify=y[te_idx], random_state=args.seed)
    Xtr = X[tr_idx]; ytr = y[tr_idx]; aux_tr = aux[tr_idx] if aux is not None else None
    Xva = X[val_idx]; yva = y[val_idx]; aux_va = aux[val_idx] if aux is not None else None
    Xte = X[te_idx]; yte = y[te_idx]; aux_te = aux[te_idx] if aux is not None else None

    cfg = SVIConfig(
        n_factors=args.n_factors,
        a=args.a, c=args.c,
        b_v=args.b_v, sigma_gamma=args.sigma_gamma,
        batch_size=args.batch_size, tau0=args.tau0, kappa_lr=args.kappa_lr,
        n_local_iter=args.n_local_iter, n_pg_subsweeps=args.n_pg_subsweeps,
        pg_ema_alpha=args.pg_ema_alpha,
        v_warmup=args.v_warmup,
        use_spike_slab=args.use_spike_slab,
    )
    print("backend:", "JAX/GPU" if USE_JAX and HAS_GPU else "JAX/CPU" if USE_JAX else "NumPy")

    model = SVIPG(cfg, random_state=args.seed, kappa=1)
    model.fit(Xtr, ytr, X_aux=aux_tr, X_val=Xva, y_val=yva, X_aux_val=aux_va,
              max_iter=args.max_iter, check_freq=args.check_freq, verbose=True)

    proba_te = model.predict_proba(Xte, X_aux_new=aux_te, n_iter=20).ravel()
    auc = roc_auc_score(yte, proba_te)
    print(f"\nTest AUC: {auc:.4f}")

    model.save(args.output_dir)


if __name__ == "__main__":
    main()
