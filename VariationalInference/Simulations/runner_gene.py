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
from ._runner_utils import derive_seeds, patient_grouped_split
from .runner_unsup import _libnorm_log1p


def run(h5ad_path: str, inner_seed: int, out_dir: str) -> dict:
    A = ad.read_h5ad(h5ad_path)
    truth_idx = int(A.uns["truth_idx"]); h2 = float(A.uns["h2"]); r = float(A.uns["r"])
    seeds = derive_seeds(truth_idx=truth_idx, h2=h2, r=r, inner_seed=inner_seed,
                         K=0, method="gene_lr")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    y = A.obs["y"].to_numpy().astype(np.float32)
    patient_ids = A.obs["patient_id"].to_numpy()
    tr_idx, te_idx = patient_grouped_split(patient_ids, n_test_patients=8,
                                           seed=seeds["split_seed"])
    Xtr_dense = _libnorm_log1p(X[tr_idx])
    Xte_dense = _libnorm_log1p(X[te_idx])
    head = LogisticRegressionCV(
        Cs=config.LR_C_GRID, cv=config.LR_CV_FOLDS, penalty="l1",
        solver="saga", max_iter=config.LR_MAX_ITER, scoring="roc_auc", n_jobs=1,
    ).fit(Xtr_dense, y[tr_idx])
    proba = head.predict_proba(Xte_dense)[:, 1]
    cell_auc = float(roc_auc_score(y[te_idx], proba))
    y_pred = (proba > 0.5).astype(int)
    metrics = {
        "method": "gene_lr", "mode": "raw", "K": 0,
        "truth_idx": truth_idx, "h2": h2, "r": r, "inner_seed": int(inner_seed),
        "cell_auc_integrated": cell_auc, "cell_auc_posthoc": cell_auc,
        "cell_f1": float(f1_score(y[te_idx], y_pred)),
        "cell_acc": float(accuracy_score(y[te_idx], y_pred)),
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"cell_idx": te_idx, "y_true": y[te_idx], "A_integrated": proba,
                  "A_posthoc": proba}).to_parquet(out / "fold_predictions.parquet")
    with gzip.open(out / "result.pkl.gz", "wb") as f:
        pickle.dump(dict(beta_LR=head.coef_.ravel(),
                         tr_idx=tr_idx, te_idx=te_idx), f)
    (out / "done.flag").write_text("ok\n")
    return metrics
