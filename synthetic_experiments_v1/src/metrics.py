"""
Metrics for DRGP synthetic experiments.

All metrics take ground-truth + estimates, apply Hungarian matching where
appropriate, and return scalars / arrays.

Convention: Hungarian matching is on Beta (gene-level loadings) using cosine
similarity. Returned permutation `pi` has length K_fit, where pi[k] = j means
fitted column k matches true column j. Unmatched fitted columns get pi[k] = -1.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score


_EPS = 1e-12


def _l2_normalize_cols(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=0, keepdims=True) + _EPS)


def hungarian_match(Beta_true: np.ndarray, Beta_hat: np.ndarray) -> np.ndarray:
    """Hungarian assignment on cosine(Beta_hat[:,k], Beta_true[:,j]).

    Returns pi : (K_fit,) int32. pi[k] = j if fitted col k matches true col j; -1 if unmatched.
    Works for K_fit >= K_true and K_fit < K_true.
    """
    Bt = _l2_normalize_cols(Beta_true)            # (p, K_true)
    Bh = _l2_normalize_cols(Beta_hat)             # (p, K_fit)
    cos = Bh.T @ Bt                               # (K_fit, K_true)
    row_ind, col_ind = linear_sum_assignment(-cos)
    pi = -np.ones(Bh.shape[1], dtype=np.int32)
    pi[row_ind] = col_ind
    return pi


def matched_cosine(Beta_true: np.ndarray, Beta_hat: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Per-matched-program cosine similarity. Output length K_true; NaN for unmatched."""
    K_true = Beta_true.shape[1]
    out = np.full(K_true, np.nan)
    for k_fit in range(Beta_hat.shape[1]):
        k_true = int(pi[k_fit])
        if k_true < 0:
            continue
        a = Beta_hat[:, k_fit]
        b = Beta_true[:, k_true]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        out[k_true] = float(a @ b) / (denom + _EPS) if denom > 0 else 0.0
    return out


def matched_jaccard_topm(S_true: np.ndarray, Beta_hat: np.ndarray, pi: np.ndarray, m: int = 50) -> np.ndarray:
    """Top-m gene Jaccard between |Beta_hat[:,k_fit]| and true support."""
    K_true = S_true.shape[1]
    out = np.full(K_true, np.nan)
    for k_fit in range(Beta_hat.shape[1]):
        k_true = int(pi[k_fit])
        if k_true < 0:
            continue
        top = set(np.argsort(-np.abs(Beta_hat[:, k_fit]))[:m].tolist())
        true_supp = set(np.flatnonzero(S_true[:, k_true]).tolist())
        if not top and not true_supp:
            out[k_true] = 1.0
        else:
            inter = len(top & true_supp)
            union = len(top | true_supp)
            out[k_true] = inter / max(union, 1)
    return out


def support_auprc(S_true: np.ndarray, R_beta: np.ndarray, pi: np.ndarray) -> float:
    """AUPRC of posterior inclusion probabilities vs binary support, pooled over matched columns."""
    K_fit = R_beta.shape[1]
    y_true_all, y_score_all = [], []
    for k_fit in range(K_fit):
        k_true = int(pi[k_fit])
        if k_true < 0:
            continue
        y_true_all.append(S_true[:, k_true])
        y_score_all.append(R_beta[:, k_fit])
    if not y_true_all:
        return float("nan")
    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def fdr_at_threshold(S_true: np.ndarray, R_beta: np.ndarray, pi: np.ndarray, thr: float = 0.5) -> float:
    """Empirical FDR at threshold thr on matched columns."""
    K_fit = R_beta.shape[1]
    y_true_all, y_score_all = [], []
    for k_fit in range(K_fit):
        k_true = int(pi[k_fit])
        if k_true < 0:
            continue
        y_true_all.append(S_true[:, k_true])
        y_score_all.append(R_beta[:, k_fit])
    if not y_true_all:
        return float("nan")
    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)
    called = y_score > thr
    if called.sum() == 0:
        return float("nan")
    return float(((called) & (y_true == 0)).sum()) / float(called.sum())


def _align_v_to_true(v_true: np.ndarray, v_hat: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (v_hat_aligned, mask) of length K_true; entries for unmatched true programs are 0 (mask=False)."""
    K_true = v_true.shape[0]
    v_hat_aligned = np.zeros(K_true)
    mask = np.zeros(K_true, dtype=bool)
    for k_fit in range(v_hat.shape[0]):
        k_true = int(pi[k_fit])
        if k_true < 0:
            continue
        v_hat_aligned[k_true] = v_hat[k_fit]
        mask[k_true] = True
    return v_hat_aligned, mask


def v_spearman(v_true: np.ndarray, v_hat: np.ndarray, pi: np.ndarray) -> float:
    """Spearman rho of |v_true| vs |v_hat_aligned| on matched true programs."""
    v_hat_aligned, mask = _align_v_to_true(v_true, v_hat, pi)
    if mask.sum() < 2:
        return float("nan")
    r, _ = spearmanr(np.abs(v_true[mask]), np.abs(v_hat_aligned[mask]))
    return float(r) if np.isfinite(r) else float("nan")


def v_kendall(v_true: np.ndarray, v_hat: np.ndarray, pi: np.ndarray) -> float:
    v_hat_aligned, mask = _align_v_to_true(v_true, v_hat, pi)
    if mask.sum() < 2:
        return float("nan")
    t, _ = kendalltau(np.abs(v_true[mask]), np.abs(v_hat_aligned[mask]))
    return float(t) if np.isfinite(t) else float("nan")


def precision_at_k(v_true: np.ndarray, v_hat: np.ndarray, pi: np.ndarray, K_rel: int = 3) -> float:
    """Fraction of top-K_rel |v_hat| that match a truly relevant program (v_true != 0)."""
    top_fit = np.argsort(-np.abs(v_hat))[:K_rel]
    rel_true = set(np.flatnonzero(v_true).tolist())
    hits = sum(1 for kf in top_fit if int(pi[kf]) in rel_true)
    return hits / float(K_rel)


def held_out_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def is_valid_permutation(pi: np.ndarray) -> bool:
    """Hungarian outputs at most one fitted col per true col. matched values (>=0) must be unique."""
    matched = pi[pi >= 0]
    return matched.size == np.unique(matched).size


def delta_pearson(Delta_true: np.ndarray, Delta_hat: np.ndarray, pi: np.ndarray) -> float:
    """Pearson correlation between true and recovered Delta entries after column permutation."""
    K_true, q = Delta_true.shape
    K_fit = Delta_hat.shape[0]
    a, b = [], []
    for k_fit in range(K_fit):
        k_true = int(pi[k_fit])
        if k_true < 0:
            continue
        a.append(Delta_hat[k_fit])
        b.append(Delta_true[k_true])
    if not a:
        return float("nan")
    a = np.concatenate(a)
    b = np.concatenate(b)
    if np.std(a) < _EPS or np.std(b) < _EPS:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r) if np.isfinite(r) else float("nan")
