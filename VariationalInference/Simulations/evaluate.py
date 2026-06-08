"""Per-result evaluation: Hungarian, recovery, ranking, prediction, liability, splitting."""
from __future__ import annotations
import gzip, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.stats import spearmanr
import anndata as ad


def _l2_normalize(x: np.ndarray, axis: int = 0) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def hungarian_match(beta_hat: np.ndarray, beta_star: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Assign each truth column to one inferred column minimizing 1 - cos.

    beta_hat: (p, K_hat); beta_star: (p, L_truth). Returns (assign of len L_truth,
    cost matrix L_truth × K_hat)."""
    bh = _l2_normalize(beta_hat, axis=0)
    bs = _l2_normalize(beta_star, axis=0)
    cost = 1.0 - bs.T @ bh                     # (L_truth, K_hat)
    row, col = linear_sum_assignment(cost)
    assign = np.full(cost.shape[0], -1, dtype=np.int64)
    assign[row] = col
    return assign, cost


def recovery_cosine(beta_hat: np.ndarray, beta_star: np.ndarray,
                    assign: np.ndarray) -> np.ndarray:
    bh = _l2_normalize(beta_hat, axis=0)
    bs = _l2_normalize(beta_star, axis=0)
    return np.array([float(bs[:, l] @ bh[:, assign[l]]) for l in range(len(assign))])


def recovery_jaccard_oracle(beta_hat: np.ndarray, S_star: list[np.ndarray],
                            assign: np.ndarray) -> list[float]:
    """Oracle top-|S*_l| cut on the matched factor — same rule for DRGP and baselines."""
    out = []
    for l, S_l in enumerate(S_star):
        k = assign[l]
        size = len(S_l)
        top = np.argsort(np.abs(beta_hat[:, k]))[-size:]
        inter = len(set(top.tolist()) & set(S_l.tolist()))
        union = size + size - inter
        out.append(inter / union if union > 0 else 0.0)
    return out


def recovery_jaccard_drgp_native(rho: np.ndarray, S_star: list[np.ndarray],
                                 assign: np.ndarray, thresh: float = 0.5) -> list[float]:
    if rho.size == 0:
        return [float("nan")] * len(S_star)
    out = []
    for l, S_l in enumerate(S_star):
        k = assign[l]
        S_hat = set(np.flatnonzero(rho[:, k] > thresh).tolist())
        S_true = set(S_l.tolist())
        union = S_hat | S_true
        out.append(len(S_hat & S_true) / len(union) if union else 0.0)
    return out


def _scale_aware_importance(*, method_family: str, K: int, mu_v=None,
                            theta_train=None, beta_LR_std=None, H_train=None) -> np.ndarray:
    """DRGP: |mu_v,k| * sd_train(theta_k).  Baselines: |beta_LR,k| (head fit on standardized H)."""
    if method_family == "drgp":
        sd_th = theta_train.std(axis=0, ddof=0).astype(np.float64)
        return np.abs(mu_v.ravel().astype(np.float64)) * sd_th
    if method_family == "baseline":
        return np.abs(beta_LR_std.astype(np.float64))
    raise ValueError(method_family)


def ranking_metrics(*, mu_v=None, theta_train=None, beta_LR_std=None,
                    H_train=None, assign: np.ndarray,
                    causal_mask: np.ndarray) -> dict:
    """assign[l] = matched factor κ. causal_mask[l] = True if truth slot l is causal."""
    fam = "drgp" if mu_v is not None else "baseline"
    K = (len(mu_v) if fam == "drgp" else len(beta_LR_std))
    imp = _scale_aware_importance(
        method_family=fam, K=K, mu_v=mu_v, theta_train=theta_train,
        beta_LR_std=beta_LR_std, H_train=H_train,
    )
    matched_causal = [int(assign[l]) for l in range(len(assign)) if causal_mask[l]]
    decoy_slots = [int(assign[l]) for l in range(len(assign)) if not causal_mask[l]]
    positive_factors = set(matched_causal)
    y_lbl = np.array([int(k in positive_factors) for k in range(K)])
    if y_lbl.any() and (y_lbl == 0).any():
        rank_auc = float(roc_auc_score(y_lbl, imp))
    else:
        rank_auc = float("nan")
    order = np.argsort(-imp)
    rank_of = {int(k): i for i, k in enumerate(order)}
    decoy_rank = int(rank_of[decoy_slots[0]]) if decoy_slots else -1
    return dict(ranking_auc=rank_auc, decoy_rank=decoy_rank,
                importance=imp.tolist())


def splitting_metrics(beta_hat: np.ndarray, S_star_pathway: list[np.ndarray]
                      ) -> tuple[list[float], list[float]]:
    """Concentration + Coverage on the K_path pathway truths (design §7.7)."""
    K = beta_hat.shape[1]
    concentration, coverage = [], []
    abs_b = np.abs(beta_hat)
    for l, S_l in enumerate(S_star_pathway):
        size = len(S_l)
        S_true = set(S_l.tolist())
        best = 0.0
        union_for_coverage: set[int] = set()
        for k in range(K):
            top_k = set(np.argsort(abs_b[:, k])[-size:].tolist())
            inter = len(top_k & S_true)
            union = len(top_k | S_true)
            j = inter / union if union else 0.0
            if j > best:
                best = j
            union_for_coverage |= top_k
        concentration.append(best)
        coverage.append(len(S_true & union_for_coverage) / max(len(S_true), 1))
    return concentration, coverage


def run(result_dir: str, h5ad_path: str) -> dict:
    """Walk one result dir, compute the full §7 metric panel, append to metrics.json."""
    rd = Path(result_dir)
    A = ad.read_h5ad(h5ad_path)
    metrics = json.loads((rd / "metrics.json").read_text())
    method = metrics["method"]; family = "drgp" if method.startswith("drgp_") else "baseline"

    u = np.asarray(A.uns["u"])                            # (p, L_cols)
    delta = float(A.uns["delta"])
    beta_star = delta * u                                  # (p, L_cols)
    L = beta_star.shape[1]
    causal_mask = np.array([float(A.uns["v_star"][l]) != 0.0 for l in range(L)])
    S_star = [np.asarray(A.uns["S_ell"][str(l)]) for l in range(L)]

    with gzip.open(rd / "result.pkl.gz", "rb") as f:
        payload = pickle.load(f)

    if family == "drgp":
        beta_hat = payload["mu_beta"]                      # (p, K)
        rho = payload.get("rho", np.zeros((beta_hat.shape[0], 0)))
        K = beta_hat.shape[1]
    else:
        beta_hat = payload["beta"].T                       # (p, K)
        K = beta_hat.shape[1]

    assign, _ = hungarian_match(beta_hat, beta_star)
    cosines = recovery_cosine(beta_hat, beta_star, assign)
    jac_oracle = recovery_jaccard_oracle(beta_hat, S_star, assign)
    metrics_out = dict(metrics)
    metrics_out["matched_cosine_per_l"] = cosines.tolist()
    metrics_out["jaccard_oracle_per_l"] = jac_oracle
    if family == "drgp" and rho.size:
        metrics_out["jaccard_drgp_native_per_l"] = recovery_jaccard_drgp_native(rho, S_star, assign)

    if family == "drgp":
        rm = ranking_metrics(mu_v=payload["mu_v"].ravel(),
                             theta_train=payload["theta_tr"],
                             assign=assign, causal_mask=causal_mask)
    else:
        rm = ranking_metrics(beta_LR_std=payload["beta_LR_std"],
                             H_train=payload["H_train"],
                             assign=assign, causal_mask=causal_mask)
    metrics_out["ranking_auc"] = rm["ranking_auc"]
    metrics_out["decoy_rank"] = rm["decoy_rank"]

    fp = pd.read_parquet(rd / "fold_predictions.parquet")
    te_idx = fp["cell_idx"].to_numpy()
    A_int = fp["A_integrated"].to_numpy()
    liability = A.obs["liability"].to_numpy()[te_idx]
    pi_true = A.obs["pi_true"].to_numpy()[te_idx]
    metrics_out["spearman_liability"] = float(spearmanr(A_int, liability).correlation)
    metrics_out["calibration_mae"] = float(np.mean(np.abs(
        1.0 / (1.0 + np.exp(-A_int)) - pi_true)))

    n_path = A.uns["mask_M"].shape[1]
    S_path = [S_star[l] for l in range(1, n_path + 1)]
    conc, cov = splitting_metrics(beta_hat, S_path)
    metrics_out["splitting_concentration"] = conc
    metrics_out["splitting_coverage"] = cov

    (rd / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    return metrics_out


def stability_run(truth_idx: int, method_family: str, mode_or_method: str,
                  K: int) -> dict:
    """Within-truth pairwise matched cosine across the R_seed inner seeds at the stability cell.
    Stratifies by causal-matched factors vs nuisance per pair."""
    from itertools import combinations
    from .truths import load_truth
    from . import config as _cfg
    base = _cfg.SIM_ROOT / "results"
    cell = _cfg.STABILITY_CELL
    fits = []
    for s in _cfg.INNER_SEEDS_STABILITY:
        rd = base / method_family / mode_or_method / f"truth{truth_idx}" / \
             f"h{cell['h2']}_r{cell['r']}_K{cell['K']}_seed{s}"
        if not (rd / "result.pkl.gz").exists():
            continue
        with gzip.open(rd / "result.pkl.gz", "rb") as f:
            p = pickle.load(f)
        beta = p["mu_beta"] if method_family == "drgp" else p["beta"].T
        fits.append((s, beta))
    causal_subset_cosines = []
    other_cosines = []
    truth = load_truth(truth_idx)
    u = truth["u"]
    beta_star = u
    causal_idx_per_fit = []
    for s, beta in fits:
        assign, _ = hungarian_match(beta, beta_star)
        is_causal = np.array([float(truth["v_star"][l]) != 0.0 for l in range(beta_star.shape[1])])
        causal_factor_set = set(int(assign[l]) for l in range(len(assign)) if is_causal[l])
        causal_idx_per_fit.append((s, beta, causal_factor_set))
    for (sa, ba, ca), (sb, bb, cb) in combinations(causal_idx_per_fit, 2):
        bha = _l2_normalize(ba, axis=0); bhb = _l2_normalize(bb, axis=0)
        cost = 1.0 - bha.T @ bhb
        row, col = linear_sum_assignment(cost)
        for r_idx, c_idx in zip(row, col):
            cos = float(1.0 - cost[r_idx, c_idx])
            if r_idx in ca and c_idx in cb:
                causal_subset_cosines.append(cos)
            else:
                other_cosines.append(cos)
    return dict(
        causal_subset_mean_cos=float(np.mean(causal_subset_cosines)) if causal_subset_cosines else float("nan"),
        other_mean_cos=float(np.mean(other_cosines)) if other_cosines else float("nan"),
        n_pairs=len(causal_subset_cosines) + len(other_cosines),
        truth_idx=truth_idx, method=f"{method_family}_{mode_or_method}", K=K,
    )
