"""Gene-level baseline: L1-LR on log1p library-normalized counts."""
from __future__ import annotations
import gzip, json, pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from . import config
from ._runner_utils import (derive_seeds, patient_grouped_split,
                            pseudobulk_mean, patient_liability)
from .runner_unsup import _libnorm_log1p
from scipy.stats import spearmanr


def run(h5ad_path: str, inner_seed: int, out_dir: str) -> dict:
    A = ad.read_h5ad(h5ad_path)
    truth_idx = int(A.uns["truth_idx"]); h2 = float(A.uns["h2"]); r = float(A.uns["r"])
    seeds = derive_seeds(truth_idx=truth_idx, h2=h2, r=r, inner_seed=inner_seed,
                         K=0, method="gene_lr")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    y = A.obs["y"].to_numpy().astype(np.float32)
    patient_ids = A.obs["patient_id"].to_numpy()
    tr_idx, te_idx = patient_grouped_split(patient_ids, n_test_patients=config.N_TEST_PATIENTS,
                                           seed=seeds["split_seed"])
    Xtr_dense = _libnorm_log1p(X[tr_idx])
    Xte_dense = _libnorm_log1p(X[te_idx])
    head = LogisticRegressionCV(
        Cs=config.LR_C_GRID, cv=config.LR_CV_FOLDS, penalty="l1",
        solver="saga", max_iter=config.LR_MAX_ITER, scoring="roc_auc", n_jobs=1,
    ).fit(Xtr_dense, y[tr_idx])
    logit = head.decision_function(Xte_dense)
    proba = head.predict_proba(Xte_dense)[:, 1]
    cell_auc = float(roc_auc_score(y[te_idx], proba))
    y_pred = (proba > 0.5).astype(int)

    # ---- Patient pseudo-bulk: mean raw counts -> libnorm-log1p -> own L1-LR head ----------
    Xpb_tr, ypb_tr, _ = pseudobulk_mean(X, patient_ids, tr_idx, y)
    Xpb_te, ypb_te, pid_te = pseudobulk_mean(X, patient_ids, te_idx, y)
    Xpb_tr_n = _libnorm_log1p(sp.csr_matrix(Xpb_tr))
    Xpb_te_n = _libnorm_log1p(sp.csr_matrix(Xpb_te))
    pat_head = LogisticRegressionCV(
        Cs=config.LR_C_GRID, cv=min(config.LR_CV_FOLDS, 3), penalty="l1",
        solver="saga", max_iter=config.LR_MAX_ITER, scoring="roc_auc", n_jobs=1,
    ).fit(Xpb_tr_n, ypb_tr)
    pat_logit = pat_head.decision_function(Xpb_te_n)
    pat_proba = pat_head.predict_proba(Xpb_te_n)[:, 1]
    pat_auc = (float(roc_auc_score(ypb_te, pat_proba))
               if len(np.unique(ypb_te)) == 2 else float("nan"))
    liab_pb_te = patient_liability(pid_te, np.asarray(A.uns["liability_patient"]))
    pat_spearman = float(spearmanr(pat_logit, liab_pb_te).correlation)

    metrics = {
        "method": "gene_lr", "mode": "raw", "K": 0,
        "truth_idx": truth_idx, "h2": h2, "r": r, "rho": float(A.uns.get("rho", -1.0)),
        "inner_seed": int(inner_seed),
        "cell_auc_integrated": cell_auc, "cell_auc_posthoc": cell_auc,
        "patient_auc_integrated": pat_auc, "patient_auc_posthoc": pat_auc,
        "patient_spearman_liability": pat_spearman,
        "cell_f1": float(f1_score(y[te_idx], y_pred)),
        "cell_acc": float(accuracy_score(y[te_idx], y_pred)),
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"cell_idx": te_idx, "y_true": y[te_idx], "A_integrated": logit,
                  "A_posthoc": logit, "proba": proba}).to_parquet(out / "fold_predictions.parquet")
    with gzip.open(out / "result.pkl.gz", "wb") as f:
        pickle.dump(dict(beta_LR=head.coef_.ravel(),
                         tr_idx=tr_idx, te_idx=te_idx), f)
    (out / "done.flag").write_text("ok\n")
    return metrics
