"""Study #3: K-fold patient cross-validation for the patient pseudo-bulk AUC.

Partitions the 80 patients into K disjoint folds; for each fold the factorizer is fit on the
other folds' cells, the held-out patients' pseudo-bulk profiles are projected in, and an L2-LR
head (fit on the training pseudo-bulk) predicts them. Every patient is predicted exactly once,
so pooling the held-out predictions yields a single ~80-patient AUC with a DeLong CI per
(truth, seed) -- removing the 16-patient single-fold noise that forced pooling over 5x5.

Uniform across methods: only the (fit -> projector) step differs. DRGP modes wrap CAVI (same
config as runner_drgp); nmf/schpf/spectra reuse runner_unsup._fit_*. gene_lr has no factor
model and is excluded. Writes cv_metrics.json + cv_predictions.parquet.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from . import config
from ._runner_utils import pseudobulk_mean, derive_seeds
from .runner_unsup import _fit_nmf, _fit_schpf, _fit_spectra
from .runner_drgp import _build_pathway_mask


def patient_kfold(patient_ids: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return a list of patient-id arrays, one per fold (disjoint, ~equal size)."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(patient_ids))
    return [np.asarray(f) for f in np.array_split(uniq, n_folds)]


def delong_auc_ci(y: np.ndarray, s: np.ndarray, alpha: float = 0.05):
    """Single-AUC value + DeLong CI via the structural components (exact, O(mn); fine for ~80)."""
    y = np.asarray(y); s = np.asarray(s)
    pos = s[y == 1]; neg = s[y == 0]; m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), (float("nan"), float("nan"))
    D = pos[:, None] - neg[None, :]
    psi = (D > 0).astype(float) + 0.5 * (D == 0)
    auc = float(psi.mean())
    v10 = psi.mean(axis=1)            # (m,) component over positives
    v01 = psi.mean(axis=0)            # (n,) component over negatives
    var = (v10.var(ddof=1) / m if m > 1 else 0.0) + (v01.var(ddof=1) / n if n > 1 else 0.0)
    se = float(np.sqrt(var)); z = float(norm.ppf(1 - alpha / 2))
    return auc, (auc - z * se, auc + z * se)


def _drgp_projector(X_tr, y_tr, patient_ids_tr, mask_M, mode, K, fit_seed):
    from VariationalInference.vi_cavi import CAVI
    pathway_mask = _build_pathway_mask(mask_M, mode, K)
    n_path = pathway_mask.shape[0] if mode == "combined" else None
    model = CAVI(
        n_factors=K, a=config.CAVI_A, c=config.CAVI_C, b_v=config.CAVI_B_V, sigma_gamma=1.0,
        regression_weight=config.REGRESSION_WEIGHT,
        supervised_update_weight=config.SUPERVISED_UPDATE_WEIGHT,
        calibrate_b_v=config.CALIBRATE_B_V, regression_design=config.REGRESSION_DESIGN,
        use_class_weights=True, random_state=fit_seed, mode=mode,
        pathway_mask=pathway_mask, n_pathway_factors=n_path,
    )
    aux = np.ones((X_tr.shape[0], 1), dtype=np.float32)
    model.fit(X_tr, y_tr, X_aux_train=aux, max_iter=config.CAVI_MAX_ITER,
              check_freq=config.CAVI_CHECK_FREQ, tol=config.CAVI_TOL,
              v_warmup=config.CAVI_V_WARMUP, early_stopping=config.EARLY_STOPPING,
              n_patients=len(np.unique(patient_ids_tr)), patient_ids=patient_ids_tr)

    def proj(Xsp):
        th = np.asarray(model.transform(Xsp, n_iter=20, supervised=False)["E_theta"])
        if config.REGRESSION_DESIGN == "normalized":
            return th / np.maximum(th.sum(axis=1, keepdims=True), 1e-8)
        return th
    return proj


def _baseline_projector(X_tr, mask_M, method, K, fit_seed):
    if method == "nmf":
        _, _, _, proj = _fit_nmf(X_tr, X_tr[:1], K, fit_seed)
    elif method == "schpf":
        _, _, _, proj = _fit_schpf(X_tr, X_tr[:1], K, fit_seed)
    elif method == "spectra":
        _, _, _, proj = _fit_spectra(X_tr, X_tr[:1], mask_M, K, fit_seed)
    else:
        raise ValueError(method)
    return proj


def run(h5ad_path: str, method: str, K: int, inner_seed: int, out_dir: str,
        n_folds: int = 5, verbose: bool = False) -> dict:
    """method: 'drgp_<mode>' or one of nmf/schpf/spectra."""
    A = ad.read_h5ad(h5ad_path)
    truth_idx = int(A.uns["truth_idx"]); h2 = float(A.uns["h2"]); r = float(A.uns["r"])
    is_drgp = method.startswith("drgp_")
    mode = method.split("drgp_")[1] if is_drgp else "lr"
    seeds = derive_seeds(truth_idx=truth_idx, h2=h2, r=r, inner_seed=inner_seed, K=K, method=method)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    y = A.obs["y"].to_numpy().astype(np.float32)
    pid = A.obs["patient_id"].to_numpy()
    mask_M = A.uns["mask_M"]
    folds = patient_kfold(pid, n_folds, seeds["split_seed"])

    pooled_prob, pooled_y, pooled_pid = [], [], []
    for fi, test_patients in enumerate(folds):
        te_mask = np.isin(pid, test_patients)
        tr_idx = np.flatnonzero(~te_mask); te_idx = np.flatnonzero(te_mask)
        if is_drgp:
            proj = _drgp_projector(X[tr_idx], y[tr_idx], pid[tr_idx], mask_M, mode, K,
                                   seeds["fit_seed"] + fi)
        else:
            proj = _baseline_projector(X[tr_idx], mask_M, method, K, seeds["fit_seed"] + fi)
        Xpb_tr, ypb_tr, _ = pseudobulk_mean(X, pid, tr_idx, y)
        Xpb_te, ypb_te, pid_te = pseudobulk_mean(X, pid, te_idx, y)
        Htr = np.asarray(proj(sp.csr_matrix(Xpb_tr)), dtype=np.float64)
        Hte = np.asarray(proj(sp.csr_matrix(Xpb_te)), dtype=np.float64)
        mu = Htr.mean(0); sd = Htr.std(0); sd[sd < 1e-8] = 1.0
        head = LogisticRegressionCV(Cs=config.LR_C_GRID, cv=min(config.LR_CV_FOLDS, 3),
                                    penalty="l2", solver="lbfgs", max_iter=config.LR_MAX_ITER,
                                    scoring="roc_auc", n_jobs=1).fit((Htr - mu) / sd, ypb_tr)
        prob = head.predict_proba((Hte - mu) / sd)[:, 1]
        pooled_prob.append(prob); pooled_y.append(ypb_te); pooled_pid.append(pid_te)
        if verbose:
            print(f"  fold {fi}: {len(te_idx)} cells / {len(pid_te)} patients held out")

    prob = np.concatenate(pooled_prob); yv = np.concatenate(pooled_y)
    pids = np.concatenate(pooled_pid)
    auc, (lo, hi) = delong_auc_ci(yv, prob)
    metrics = {
        "method": method, "mode": mode, "K": int(K), "truth_idx": truth_idx,
        "h2": h2, "r": r, "rho": float(A.uns.get("rho", -1.0)), "inner_seed": int(inner_seed),
        "n_folds": int(n_folds), "n_patients": int(len(yv)),
        "cv_patient_auc": float(auc), "cv_patient_auc_lo": float(lo), "cv_patient_auc_hi": float(hi),
    }
    (out / "cv_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"patient_id": pids, "y_true": yv, "prob": prob}).to_parquet(
        out / "cv_predictions.parquet")
    (out / "done.flag").write_text("ok\n")
    if verbose:
        print(f"  pooled CV patient AUC = {auc:.3f} [{lo:.3f}, {hi:.3f}]  (n={len(yv)})")
    return metrics
