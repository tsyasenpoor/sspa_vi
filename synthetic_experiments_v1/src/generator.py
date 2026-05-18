"""
Ancestral sampler for DRGP synthetic experiments.

Generative model (0-indexed):
  X_i        ~ covariate distributions (q = 5)
  S          : (p, K) binary support; disjoint by default,
               with optional Jaccard-controlled overlap on `overlap_pair`
  beta_{jk}  = 0 if S_{jk} = 0 else Gamma(a_beta, b_beta)
  Delta      : (K, q) mediation matrix; zero except entries in `delta_spec`
  theta_{ik} ~ Gamma(a_theta + max(X_i . Delta_k, 0), b_theta)
  Mu         = Theta @ Beta.T
  Y_{ij}     ~ Poisson(Mu_{ij})   or   NB(mu=Mu_{ij}, dispersion=phi)
  v_k        = 0 for k not in `rel_idx`, else sign * U(1.0, 2.5)
  alpha_l    ~ N(0, alpha_sd^2)
  eta_i      = Theta_i . v + X_i . alpha + xi_0   (xi_0 calibrated so mean sigmoid ~= 0.5)
  y_i        ~ Bernoulli(sigmoid(eta_i))

OOD test-set construction:
  Call `generate(..., freeze_params=GroundTruth_train)` with a shifted covariate
  distribution. Beta, v, alpha, Delta, S, rel_idx, xi_0 are reused from train;
  only X, Theta, Mu, Y, y are resampled.

Fast-dev defaults are provided in `FAST_DEV_KWARGS`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize


FAST_DEV_KWARGS = dict(
    n=100, p=500, K_true=3, K_rel=2, q=5,
    m_choices=(20, 30, 40),
    delta_spec={(0, 2): 1.5, (1, 1): 1.0},
)


@dataclass
class GroundTruth:
    Y: np.ndarray            # (n, p) counts
    y: np.ndarray            # (n,)   binary phenotype
    X: np.ndarray            # (n, q) covariates
    Theta: np.ndarray        # (n, K) program activations
    Beta: np.ndarray         # (p, K) gene loadings
    S: np.ndarray            # (p, K) binary support
    v: np.ndarray            # (K,)   regression weights
    alpha: np.ndarray        # (q,)   direct covariate effects
    Delta: np.ndarray        # (K, q) mediation matrix
    xi_0: float              # intercept
    rel_idx: np.ndarray      # (K_rel,) 0-indexed disease-relevant programs
    config: dict             # snapshot of generator kwargs


def _solve_jaccard_overlap(m: int, J: float) -> int:
    """|A ∩ B| for equal-size sets of size m at target Jaccard J: c = 2Jm/(1+J)."""
    if J <= 0.0:
        return 0
    return int(round(2.0 * J * m / (1.0 + J)))


def _build_supports(
    p: int,
    K: int,
    m_choices: tuple,
    overlap_pair: Optional[tuple],
    jaccard: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build (p, K) binary support. Disjoint except on `overlap_pair` (if set)."""
    S = np.zeros((p, K), dtype=np.int32)
    pool = list(range(p))
    rng.shuffle(pool)

    m = rng.choice(m_choices, size=K)

    if overlap_pair is not None and jaccard > 0.0:
        k1, k2 = overlap_pair
        if not (0 <= k1 < K and 0 <= k2 < K and k1 != k2):
            raise ValueError(f"overlap_pair {overlap_pair} invalid for K={K}")
        m_pair = int(np.mean([m[k1], m[k2]]))
        m[k1] = m[k2] = m_pair
        c = _solve_jaccard_overlap(m_pair, jaccard)
        n_uniq = m_pair - c
        needed = c + 2 * n_uniq
        if len(pool) < needed:
            raise ValueError(
                f"Gene pool too small for overlap pair: need {needed}, have {len(pool)}"
            )
        shared = pool[:c]
        uniq_k1 = pool[c:c + n_uniq]
        uniq_k2 = pool[c + n_uniq:c + 2 * n_uniq]
        pool = pool[c + 2 * n_uniq:]
        S[shared, k1] = 1
        S[shared, k2] = 1
        S[uniq_k1, k1] = 1
        S[uniq_k2, k2] = 1
        remaining = [k for k in range(K) if k not in (k1, k2)]
    else:
        remaining = list(range(K))

    for k in remaining:
        if len(pool) < m[k]:
            raise ValueError(
                f"Gene pool exhausted at program {k}: need {m[k]}, have {len(pool)}. "
                f"Increase p or decrease m_choices."
            )
        idx = pool[:m[k]]
        pool = pool[m[k]:]
        S[idx, k] = 1

    return S


def _calibrate_intercept(eta_no_intercept: np.ndarray, target_rate: float) -> tuple[float, bool]:
    """Brent solve for xi_0 such that mean sigmoid(eta + xi_0) = target_rate.

    Returns (xi_0, ok). ok=False signals fallback to 0.0 (silent miscalibration).
    """
    def f(xi_0: float) -> float:
        z = eta_no_intercept + xi_0
        # Numerically stable mean sigmoid
        return float(np.mean(np.where(z >= 0,
                                      1.0 / (1.0 + np.exp(-z)),
                                      np.exp(z) / (1.0 + np.exp(z))))) - target_rate
    try:
        return optimize.brentq(f, -30.0, 30.0), True
    except ValueError:
        return 0.0, False


def generate(
    n: int = 500,
    p: int = 5000,
    K_true: int = 10,
    K_rel: int = 3,
    q: int = 5,
    # Boosted defaults (post-review): preserve heavy tail (shape<1) on beta and
    # theta; lower rates so E[lib] = K * m_bar * E[theta] * E[beta] = 10*50*1*2 = 1000.
    a_beta: float = 0.3,
    b_beta: float = 0.15,
    a_theta: float = 0.3,
    b_theta: float = 0.3,
    v_magnitude_range: tuple = (1.0, 2.5),
    alpha_sd: float = 0.3,
    target_phenotype_rate: float = 0.5,
    likelihood: str = "poisson",
    nb_dispersion: float = 0.5,
    m_choices: tuple = (30, 50, 80),
    overlap_pair: Optional[tuple] = None,
    jaccard: float = 0.0,
    delta_spec: Optional[dict] = None,
    asthma_rate: float = 0.2,
    seed: int = 0,
    freeze_params: Optional[GroundTruth] = None,
    rel_idx: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> GroundTruth:
    """Generate one synthetic dataset.

    delta_spec : dict mapping (program_idx, covariate_idx) -> effect size.
                 Default puts asthma effect on program 0, age effect on program 1.
                 Covariate order: [sex, age, asthma, smoker, bmi].

    freeze_params : if given, reuse Beta, v, alpha, Delta, S, rel_idx, xi_0 from
                    this GroundTruth and only resample X, Theta, Y, y.
                    Used for OOD test-set construction (C2).

    rel_idx : explicit 0-indexed array of disease-relevant programs.
              Default: np.arange(K_rel).
    """
    config = dict(
        n=n, p=p, K_true=K_true, K_rel=K_rel, q=q,
        a_beta=a_beta, b_beta=b_beta, a_theta=a_theta, b_theta=b_theta,
        v_magnitude_range=v_magnitude_range, alpha_sd=alpha_sd,
        target_phenotype_rate=target_phenotype_rate,
        likelihood=likelihood, nb_dispersion=nb_dispersion,
        m_choices=tuple(m_choices), overlap_pair=overlap_pair,
        jaccard=jaccard, delta_spec=delta_spec, asthma_rate=asthma_rate,
        seed=seed, frozen=freeze_params is not None,
    )

    rng = np.random.default_rng(seed)

    # ---- Stage 1: Covariates (always resampled, even with frozen params) ----
    x_sex    = rng.binomial(1, 0.5, size=n).astype(np.float64)
    x_age    = rng.standard_normal(size=n)
    x_asthma = rng.binomial(1, asthma_rate, size=n).astype(np.float64)
    x_smoker = rng.binomial(1, 0.3, size=n).astype(np.float64)
    x_bmi    = rng.standard_normal(size=n)
    X = np.stack([x_sex, x_age, x_asthma, x_smoker, x_bmi], axis=1)
    assert X.shape == (n, q), f"q must be 5; got q={q}"

    # ---- Frozen-params branch: reuse structural quantities ----
    if freeze_params is not None:
        fp = freeze_params
        if fp.Beta.shape[0] != p:
            raise ValueError(f"freeze_params has p={fp.Beta.shape[0]}, expected {p}")
        K_true = fp.Beta.shape[1]
        S = fp.S
        Beta = fp.Beta
        Delta = fp.Delta
        v = fp.v
        alpha = fp.alpha
        rel_idx_used = fp.rel_idx
        xi_0 = fp.xi_0
    else:
        # ---- Stage 2: Supports ----
        S = _build_supports(p, K_true, m_choices, overlap_pair, jaccard, rng)

        # ---- Stage 3: Loadings ----
        Beta = np.zeros((p, K_true))
        mask_bool = S.astype(bool)
        Beta[mask_bool] = rng.gamma(shape=a_beta, scale=1.0 / b_beta, size=mask_bool.sum())

        # ---- Stage 4: Mediation matrix ----
        Delta = np.zeros((K_true, q))
        if delta_spec is None:
            delta_spec = {(0, 2): 1.5, (1, 1): 1.0}  # asthma->prog0, age->prog1
        for (k, ell), val in delta_spec.items():
            if not (0 <= k < K_true and 0 <= ell < q):
                raise ValueError(f"delta_spec entry ({k},{ell}) out of bounds")
            Delta[k, ell] = val

        # ---- v and alpha (computed before Theta so we can calibrate xi_0) ----
        rel_idx_used = np.arange(K_rel) if rel_idx is None else np.asarray(rel_idx)
        if rel_idx_used.size != K_rel:
            raise ValueError(f"rel_idx size {rel_idx_used.size} != K_rel {K_rel}")
        if rel_idx_used.max() >= K_true or rel_idx_used.min() < 0:
            raise ValueError(f"rel_idx out of bounds for K_true={K_true}")
        v = np.zeros(K_true)
        signs = rng.choice([-1.0, +1.0], size=K_rel)
        mags = rng.uniform(*v_magnitude_range, size=K_rel)
        v[rel_idx_used] = signs * mags
        alpha = rng.normal(0.0, alpha_sd, size=q)
        xi_0 = None  # calibrated after Theta is drawn

    # ---- Stage 4b: Program activations ----
    XD = X @ Delta.T                                  # (n, K)
    shape_param = a_theta + np.maximum(XD, 0.0)
    Theta = rng.gamma(shape=shape_param, scale=1.0 / b_theta)
    assert Theta.shape == (n, K_true)

    # ---- Stage 5: Counts ----
    Mu = Theta @ Beta.T                               # (n, p)
    if likelihood == "poisson":
        Y = rng.poisson(Mu)
    elif likelihood == "nb":
        # variance = mu + phi * mu^2 ; numpy parameterization (n, p) with mean=n(1-p)/p
        # => n_param = 1/phi, p_param = 1/(1 + phi*mu)
        n_param = 1.0 / float(nb_dispersion)
        p_param = 1.0 / (1.0 + float(nb_dispersion) * Mu)
        Y = rng.negative_binomial(n_param, p_param)
    else:
        raise ValueError(f"Unknown likelihood: {likelihood}")
    Y = Y.astype(np.int32)

    # ---- Stage 6: Phenotype ----
    if freeze_params is None:
        eta_no_intercept = Theta @ v + X @ alpha
        xi_0, ok = _calibrate_intercept(eta_no_intercept, target_phenotype_rate)
        if verbose and not ok:
            print(f"[generator] WARNING: xi_0 bracketing failed; fallback xi_0=0.0")
    eta = Theta @ v + X @ alpha + xi_0
    # Stable sigmoid
    prob = np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)),
                              np.exp(eta) / (1.0 + np.exp(eta)))
    y = rng.binomial(1, prob).astype(np.int32)

    return GroundTruth(
        Y=Y, y=y, X=X, Theta=Theta, Beta=Beta, S=S,
        v=v, alpha=alpha, Delta=Delta, xi_0=float(xi_0),
        rel_idx=rel_idx_used, config=config,
    )
