"""Pólya-Gamma CAVI supervised-regression kernel — design-agnostic.

Shared by:
  - flat CAVI (design = E[θ_i], units = cells)
  - hierarchical CAVIHierarchical (design = E[Θ_g], units = patients)

Single home for the supervised math: υ posterior, γ posterior, the R
correction on the design's Gamma rate, the PG tilt, and L_sup. Sibling
models call these functions — never copy-paste — so the two paths cannot
silently drift.

## Tempering convention (generalized-Bayes power posterior)

    L = E_q[log p(X|θ,β)] + rw · E_q[log p(y|design,υ,γ)]
        + log-priors - entropy

The supervised contributions (R_lin, R_quad in design's b-rate; data
terms in υ and γ updates; L_sup ELBO term) are multiplied by

    effective_rw = ramp * regression_weight

at the call sites. Priors and the PG tilt stay at weight 1.

PG augmentation is computed at count 1 (`E[ω] = (1/(2c)) tanh(c/2)`) and
the rw multiplier on the quadratic sites does the scaling. This is
numerically identical to strict PG(rw, c) tempering and avoids the rw²
double-count that would result from setting the PG count to rw. The
tilt `c = sqrt(E[ψ²])` is rw-independent.

## Conventions

All shapes are parameterized over `units` — flat = n cells, sc = G patients.

  E_design      (units, K)     first moment of q(design)
  Var_design    (units, K)     variance of q(design); = E/b for Gamma
  mu_v          (kappa, K)
  sigma_v_diag  (kappa, K)
  mu_gamma      (kappa, p_aux)
  X_aux         (units, p_aux) intercept prepended upstream; (units, 0) if p_aux=0
  y             (units, kappa)
  wbar          (units, kappa) output of pg_tilt
  c             (units, kappa) output of pg_tilt
  sample_weights(units, kappa)
  effective_rw  float           = ramp * regression_weight
  ramp          float in [0,1]  needed for under-relaxation step sizes
  row_chunk     int             upper bound on per-chunk rows; halved on OOM

Each function returns the chunk size it ended up using (callers update state).
"""
from __future__ import annotations

import numpy as np

try:
    from .jax_backend import (
        xp, USE_JAX, log_expit, omega_bar,
    )
except ImportError:
    from jax_backend import (  # type: ignore[no-redef]
        xp, USE_JAX, log_expit, omega_bar,
    )


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "resource_exhausted" in msg
        or "out of memory" in msg
        or "cuda_error_out_of_memory" in msg
    )


def _min_chunk(units: int) -> int:
    return max(1024, units // 256)


def _has_aux(X_aux) -> bool:
    return X_aux is not None and X_aux.shape[1] > 0


# =========================================================================
# pg_tilt — PG augmentation: c = sqrt(E[A²]); wbar = omega_bar(c). rw-independent.
# =========================================================================
def pg_tilt(E_design, Var_design, mu_v, sigma_v_diag, mu_gamma, X_aux,
            row_chunk):
    """Returns (c, wbar, row_chunk_used).

    c²_{ik} = (E[design_i]^T μ_v_k + X_aux_i μ_γ_k^T)²
              + Σ_ℓ Var[design_iℓ] (μ_v²_kℓ + σ²_v_kℓ)
    wbar_{ik} = (1/(2 c_{ik})) tanh(c_{ik}/2)

    `Var[design]` cross-term ONLY (paper notes §3.6; Var[υ], Var[γ]
    cross-terms intentionally omitted for paper parity).
    """
    units = E_design.shape[0]
    min_chunk = _min_chunk(units)
    has_aux = _has_aux(X_aux)
    E_v_sq = mu_v ** 2 + sigma_v_diag                       # (kappa, K)

    chunks = []
    i0 = 0
    while i0 < units:
        i1 = min(i0 + row_chunk, units)
        try:
            E_A_c = E_design[i0:i1] @ mu_v.T                # (chunk, kappa)
            if has_aux:
                E_A_c = E_A_c + X_aux[i0:i1] @ mu_gamma.T
            E_A_sq_c = xp.square(E_A_c) + Var_design[i0:i1] @ E_v_sq.T
            chunks.append(xp.sqrt(xp.maximum(E_A_sq_c, 1e-12)))
            i0 = i1
        except Exception as exc:
            if _is_oom_error(exc) and row_chunk > min_chunk:
                row_chunk = max(min_chunk, row_chunk // 2)
                continue
            raise

    c = xp.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    return c, omega_bar(c), row_chunk


# =========================================================================
# pg_R_correction — supervised contributions to the design's Gamma rate.
# =========================================================================
def pg_R_correction(E_design, mu_v, sigma_v_diag, mu_gamma, X_aux, wbar,
                    y, sample_weights, effective_rw, row_chunk):
    """Returns (R_lin, R_quad, row_chunk_used). Each (units, K), already × effective_rw.

    R_lin (rate-shift, linear in design):
      R_lin_{iℓ} = effective_rw · Σ_k W_ik [
          -(y_ik - 0.5) v_kℓ
          + ω̄_ik (E[design_i]^T v_k + x_aux_i^T γ_k) v_kℓ
          - ω̄_ik E[design_iℓ] v²_kℓ
      ]

    R_quad (rate-shift coefficient on a_design):
      R_quad_{iℓ} = effective_rw · 0.5 · Σ_k W_ik ω̄_ik E[v²_kℓ]

    Per PG_CAVI_implementation_notes.md §3.2 the 0.5 is load-bearing —
    if you start from a JJ checkpoint and only swap `lam → wbar/2`
    without keeping this 0.5, the self-term is 2× too large.

    The design's b-rate update solves the quadratic
        b² - (b_Poisson + R_lin) b - R_quad · a = 0
    so the supervised correction sits inside both b_base and disc.
    """
    units = E_design.shape[0]
    y_exp = y if y.ndim > 1 else y[:, None]
    E_v_sq = mu_v ** 2 + sigma_v_diag         # (kappa, K) — for R_quad
    E_v_sq_col = xp.square(mu_v)              # (kappa, K) — for R_lin self-term
    min_chunk = _min_chunk(units)
    has_aux = _has_aux(X_aux)

    R_lin_chunks = []
    R_quad_chunks = []
    i0 = 0
    while i0 < units:
        i1 = min(i0 + row_chunk, units)
        try:
            E_design_c = E_design[i0:i1]
            W_c = sample_weights[i0:i1]
            W_wbar_c = W_c * wbar[i0:i1]
            design_v_c = E_design_c @ mu_v.T
            if has_aux:
                design_v_c = design_v_c + X_aux[i0:i1] @ mu_gamma.T

            R_lin_c = -(W_c * (y_exp[i0:i1] - 0.5)) @ mu_v        # (chunk, K)
            R_lin_c = R_lin_c + (W_wbar_c * design_v_c) @ mu_v
            R_lin_c = R_lin_c - E_design_c * (W_wbar_c @ E_v_sq_col)
            R_quad_c = 0.5 * W_wbar_c @ E_v_sq                    # (chunk, K)

            R_lin_chunks.append(R_lin_c * effective_rw)
            R_quad_chunks.append(R_quad_c * effective_rw)
            i0 = i1
        except Exception as exc:
            if _is_oom_error(exc) and row_chunk > min_chunk:
                row_chunk = max(min_chunk, row_chunk // 2)
                continue
            raise

    R_lin = xp.concatenate(R_lin_chunks, axis=0) if len(R_lin_chunks) > 1 else R_lin_chunks[0]
    R_quad = xp.concatenate(R_quad_chunks, axis=0) if len(R_quad_chunks) > 1 else R_quad_chunks[0]
    return R_lin, R_quad, row_chunk


# =========================================================================
# pg_update_v — υ posterior (diagonal Gaussian, Laplace-Lasso prior).
# =========================================================================
def pg_update_v(E_design, Var_design, mu_v, sigma_v_diag, mu_gamma,
                X_aux, y, wbar, sample_weights, b_v, K,
                effective_rw, ramp, row_chunk):
    """Returns (new_mu_v, new_sigma_v_diag, row_chunk_used).

    Tempered: data sufficient statistics × effective_rw. Laplace prior at weight 1.
    Under-relaxed by α_v = min(0.1, 10/K) · ramp (trust region α scales with ramp,
    not rw — supervision strength shouldn't change step size).

    precision_kℓ = 1/b_v² + effective_rw · 2 · Σ_i W_ik (ω̄_ik/2) E[design²_iℓ]
                                  = 1/b_v² + effective_rw · Σ_i W_ik ω̄_ik E[design²_iℓ]
    mean numerator (raw)
      = effective_rw · [Σ_i W_ik (y_ik - 0.5) E[design_iℓ]
                       - 2 Σ_i W_ik (ω̄_ik/2) ((design_v_ik) E[design_iℓ] - v_kℓ E[design_iℓ]²)]
    """
    units = E_design.shape[0]
    kappa = mu_v.shape[0]
    y_exp = y if y.ndim > 1 else y[:, None]
    lam = wbar / 2.0                                  # PG-CAVI: 2*lam = wbar at tilt optimum
    W = sample_weights
    E_v = mu_v
    prior_precision = xp.full_like(mu_v, 1.0 / (b_v ** 2))
    min_chunk = _min_chunk(units)
    has_aux = _has_aux(X_aux)

    prec_sum  = xp.zeros((kappa, K))
    term1_sum = xp.zeros((kappa, K))
    parta_sum = xp.zeros((kappa, K))
    partb_sum = xp.zeros((kappa, K))

    i0 = 0
    while i0 < units:
        i1 = min(i0 + row_chunk, units)
        try:
            E_design_c = E_design[i0:i1]
            E_design_sq_c = xp.square(E_design_c) + Var_design[i0:i1]    # E[design²]
            design_v_c = E_design_c @ E_v.T
            if has_aux:
                design_v_c = design_v_c + X_aux[i0:i1] @ mu_gamma.T
            W_lam_c = W[i0:i1] * lam[i0:i1]
            W_y_c   = W[i0:i1] * (y_exp[i0:i1] - 0.5)

            prec_sum  = prec_sum  + W_lam_c.T @ E_design_sq_c
            term1_sum = term1_sum + W_y_c.T   @ E_design_c
            parta_sum = parta_sum + (W_lam_c * design_v_c).T @ E_design_c
            partb_sum = partb_sum + W_lam_c.T @ xp.square(E_design_c)
            i0 = i1
        except Exception as exc:
            if _is_oom_error(exc) and row_chunk > min_chunk:
                row_chunk = max(min_chunk, row_chunk // 2)
                continue
            raise

    # Tempering: scale data terms by effective_rw (= ramp * rw)
    prec_sum  = prec_sum  * effective_rw
    term1_sum = term1_sum * effective_rw
    parta_sum = parta_sum * effective_rw
    partb_sum = partb_sum * effective_rw

    data_prec = 2 * prec_sum
    precision = prior_precision + data_prec
    precision = xp.maximum(precision, 1e-12)
    term1 = term1_sum
    term2 = 2.0 * (parta_sum - E_v * partb_sum)
    mean_prec = term1 - term2

    mu_v_new = mean_prec / precision
    delta_v = 3.0 * xp.sqrt(xp.maximum(sigma_v_diag, 1e-8))
    mu_v_new = xp.clip(mu_v_new, mu_v - delta_v, mu_v + delta_v)
    alpha_v = min(0.1, 10.0 / K) * ramp
    new_mu_v = (1.0 - alpha_v) * mu_v + alpha_v * mu_v_new

    sigma_v_new = 1.0 / precision
    sigma_v_floor = 0.01 * (b_v ** 2)
    sigma_v_new = xp.maximum(sigma_v_new, sigma_v_floor)
    new_sigma_v_diag = (1.0 - alpha_v) * sigma_v_diag + alpha_v * sigma_v_new

    return new_mu_v, new_sigma_v_diag, row_chunk


# =========================================================================
# pg_update_gamma — γ posterior (Gaussian).
# =========================================================================
def pg_update_gamma(E_design, mu_v, mu_gamma, Sigma_gamma, sigma_gamma_param,
                    X_aux, y, wbar, sample_weights, K,
                    effective_rw, ramp, p_aux, row_chunk):
    """In-place style update for γ. Returns (new_mu_gamma, new_Sigma_gamma, row_chunk_used).

    `Sigma_gamma` shape (kappa, p_aux, p_aux). `sigma_gamma_param` is the scalar
    Gaussian prior std (prior precision I/sigma_gamma_param²).

    Tempered: weighted X.T X and X.T y sums × effective_rw. Prior weight 1.
    Trust region: |γ_k - γ_k^old| ≤ 3·sigma_gamma_param (prior-scale).
    Under-relaxed by α_γ = min(0.3, 30/K) · ramp.

    Mirrors the existing flat `_update_gamma` exactly when units = n, design = θ.
    """
    if p_aux == 0:
        return mu_gamma, Sigma_gamma, row_chunk

    units = E_design.shape[0]
    kappa = mu_v.shape[0]
    y_exp = y if y.ndim > 1 else y[:, None]
    lam = wbar / 2.0
    W = sample_weights

    # Pre-compute design @ μ_v.T in chunks (memory parity with flat code).
    design_v = xp.zeros((units, kappa))
    for i0 in range(0, units, row_chunk):
        i1 = min(i0 + row_chunk, units)
        E_design_c = E_design[i0:i1]
        if USE_JAX:
            design_v = design_v.at[i0:i1].set(E_design_c @ mu_v.T)
        else:
            design_v[i0:i1] = E_design_c @ mu_v.T

    alpha_gamma = min(0.3, 30.0 / K) * ramp
    new_mu_gamma = mu_gamma
    new_Sigma_gamma = Sigma_gamma
    for k in range(kappa):
        prec_prior = xp.eye(p_aux) / (sigma_gamma_param ** 2)
        W_lam_k = W[:, k] * lam[:, k]
        # Tempering: data accumulators × effective_rw (= ramp × rw).
        weighted_X = X_aux * (2.0 * effective_rw * W_lam_k)[:, None]
        prec = prec_prior + weighted_X.T @ X_aux

        theta_v_k = design_v[:, k]
        residual = effective_rw * (
            W[:, k] * (y_exp[:, k] - 0.5) - 2.0 * W_lam_k * theta_v_k
        )
        mean_prec = X_aux.T @ residual

        Sigma_new = xp.linalg.inv(prec) if USE_JAX else np.linalg.inv(prec)
        mu_gamma_new = Sigma_new @ mean_prec

        # Trust region: prior-scale clip (matches flat code).
        delta_gamma = 3.0 * sigma_gamma_param
        mu_gamma_new = xp.clip(
            mu_gamma_new,
            mu_gamma[k] - delta_gamma,
            mu_gamma[k] + delta_gamma,
        )

        Sigma_damped = (1.0 - alpha_gamma) * Sigma_gamma[k] + alpha_gamma * Sigma_new
        mu_damped = (1.0 - alpha_gamma) * mu_gamma[k] + alpha_gamma * mu_gamma_new

        if USE_JAX:
            new_Sigma_gamma = new_Sigma_gamma.at[k].set(Sigma_damped)
            new_mu_gamma = new_mu_gamma.at[k].set(mu_damped)
        else:
            new_Sigma_gamma[k] = Sigma_damped
            new_mu_gamma[k] = mu_damped

    return new_mu_gamma, new_Sigma_gamma, row_chunk


# =========================================================================
# pg_Lsup — supervised contribution to ELBO (BEFORE × rw).
# =========================================================================
def pg_Lsup(E_design, Var_design, mu_v, sigma_v_diag, mu_gamma,
            X_aux, y, c, wbar, sample_weights, row_chunk):
    """Returns (Lsup_raw, row_chunk_used). Caller multiplies by effective_rw.

      L_sup = Σ_ik W_ik [ (y_ik - 0.5) E[A_ik]  -  (ω̄_ik/2) E[A²_ik]
                          + (ω̄_ik/2) c²_ik  -  c_ik/2  +  log σ(c_ik) ]

    At the c-tilt optimum (after pg_tilt), c² = E[A²] and the last three
    terms collapse to log σ(c) - c/2 (JJ form). Keep the full form for
    monotonicity-debugging when the tilt is slightly stale.
    """
    units = E_design.shape[0]
    y_exp = y if y.ndim > 1 else y[:, None]
    lam = wbar / 2.0
    E_v_sq = mu_v ** 2 + sigma_v_diag
    min_chunk = _min_chunk(units)
    has_aux = _has_aux(X_aux)

    Lsup_raw = 0.0
    i0 = 0
    while i0 < units:
        i1 = min(i0 + row_chunk, units)
        try:
            E_design_c = E_design[i0:i1]
            Var_design_c = Var_design[i0:i1]
            E_A_c = E_design_c @ mu_v.T
            if has_aux:
                E_A_c = E_A_c + X_aux[i0:i1] @ mu_gamma.T
            E_A_sq_c = E_A_c ** 2 + Var_design_c @ E_v_sq.T

            lam_c = lam[i0:i1]
            W_c = sample_weights[i0:i1]
            c_c = c[i0:i1]
            Lsup_raw = Lsup_raw + xp.sum(
                W_c * ((y_exp[i0:i1] - 0.5) * E_A_c - lam_c * E_A_sq_c)
            )
            Lsup_raw = Lsup_raw + xp.sum(
                W_c * (lam_c * c_c ** 2 - 0.5 * c_c + log_expit(c_c))
            )
            i0 = i1
        except Exception as exc:
            if _is_oom_error(exc) and row_chunk > min_chunk:
                row_chunk = max(min_chunk, row_chunk // 2)
                continue
            raise

    return Lsup_raw, row_chunk
