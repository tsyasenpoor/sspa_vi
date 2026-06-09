"""One unsupervised-factorizer fit + L1-LR head."""
from __future__ import annotations
import gzip, json, pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from . import config
from ._runner_utils import derive_seeds, patient_grouped_split
from .projection import poisson_foldin_solve, nmf_nnls_project, standardize_factors


def _libnorm_log1p(X: sp.csr_matrix, target: float | None = None) -> np.ndarray:
    depth = np.asarray(X.sum(axis=1)).ravel()
    if target is None:
        target = float(np.median(depth))
    sf = target / np.maximum(depth, 1.0)
    Xn = X.multiply(sf[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    return Xn.toarray().astype(np.float32)


def _fit_nmf(X_train_sp: sp.csr_matrix, X_test_sp: sp.csr_matrix, K: int, seed: int):
    Xtr = _libnorm_log1p(X_train_sp)
    Xte = _libnorm_log1p(X_test_sp)
    m = NMF(n_components=K, init="nndsvd", solver="mu", beta_loss="kullback-leibler",
            max_iter=500, random_state=seed)
    H_tr = m.fit_transform(Xtr)              # (n_tr, K)
    W = m.components_                        # (K, p)
    H_te = nmf_nnls_project(Xte, W)
    return H_tr, H_te, W


def _fit_schpf(X_train_sp: sp.csr_matrix, X_test_sp: sp.csr_matrix, K: int, seed: int):
    from schpf import scHPF
    m = scHPF(nfactors=K, random_state=seed)
    m.fit(X_train_sp)
    H_tr = np.asarray(m.cell_score(), dtype=np.float32)         # (n_tr, K)
    H_te = np.asarray(m.project(X_test_sp).cell_score(), dtype=np.float32)
    beta = np.asarray(m.gene_score()).T                          # (K, p)
    return H_tr, H_te, beta


def _fit_spectra(X_train_sp: sp.csr_matrix, X_test_sp: sp.csr_matrix, mask_M: np.ndarray,
                 K: int, seed: int):
    from Spectra import Spectra as SpectraModel
    gene_sets = {f"path_{k}": np.flatnonzero(mask_M[:, k]).tolist()
                 for k in range(mask_M.shape[1])}
    np.random.seed(seed)
    m = SpectraModel(n_factors=K, gene_sets=gene_sets)
    m.fit(X_train_sp.toarray() if sp.issparse(X_train_sp) else X_train_sp)
    beta = np.asarray(getattr(m, "factors_genes", getattr(m, "beta", None)),
                      dtype=np.float64)                          # (K, p)
    H_tr = poisson_foldin_solve(np.asarray(X_train_sp.toarray(), dtype=np.float32), beta)
    H_te = poisson_foldin_solve(np.asarray(X_test_sp.toarray(), dtype=np.float32), beta)
    return H_tr, H_te, beta


def run(h5ad_path: str, method: str, K: int, inner_seed: int, out_dir: str) -> dict:
    assert method in ("nmf", "schpf", "spectra")
    A = ad.read_h5ad(h5ad_path)
    truth_idx = int(A.uns["truth_idx"]); h2 = float(A.uns["h2"]); r = float(A.uns["r"])
    seeds = derive_seeds(truth_idx=truth_idx, h2=h2, r=r, inner_seed=inner_seed,
                         K=K, method=f"{method}_lr")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    y = A.obs["y"].to_numpy().astype(np.float32)
    patient_ids = A.obs["patient_id"].to_numpy()
    tr_idx, te_idx = patient_grouped_split(patient_ids, n_test_patients=8,
                                           seed=seeds["split_seed"])
    X_tr, X_te = X[tr_idx], X[te_idx]; y_tr, y_te = y[tr_idx], y[te_idx]

    if method == "nmf":
        H_tr, H_te, beta = _fit_nmf(X_tr, X_te, K, seeds["fit_seed"])
    elif method == "schpf":
        H_tr, H_te, beta = _fit_schpf(X_tr, X_te, K, seeds["fit_seed"])
    else:
        H_tr, H_te, beta = _fit_spectra(X_tr, X_te, A.uns["mask_M"], K, seeds["fit_seed"])

    Htr_std, Hte_std, mean_tr, sd_tr = standardize_factors(H_tr, H_te)
    head = LogisticRegressionCV(
        Cs=config.LR_C_GRID, cv=config.LR_CV_FOLDS, penalty="l1",
        solver="saga", max_iter=config.LR_MAX_ITER, scoring="roc_auc", n_jobs=1,
    ).fit(Htr_std, y_tr)
    logit = head.decision_function(Hte_std)
    proba = head.predict_proba(Hte_std)[:, 1]
    cell_auc = float(roc_auc_score(y_te, proba))
    y_pred = (proba > 0.5).astype(int)

    metrics = {
        "method": f"{method}_lr", "mode": "lr", "K": int(K),
        "truth_idx": truth_idx, "h2": h2, "r": r, "inner_seed": int(inner_seed),
        "cell_auc_integrated": cell_auc,
        "cell_auc_posthoc": cell_auc,         # head IS the only logit
        "cell_f1": float(f1_score(y_te, y_pred)),
        "cell_acc": float(accuracy_score(y_te, y_pred)),
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"cell_idx": te_idx, "y_true": y_te, "A_integrated": logit,
                  "A_posthoc": logit, "proba": proba}).to_parquet(out / "fold_predictions.parquet")
    with gzip.open(out / "result.pkl.gz", "wb") as f:
        pickle.dump(dict(H_train=H_tr, H_test=H_te, beta=beta,
                         beta_LR_std=head.coef_.ravel(),
                         mean_tr=mean_tr, sd_tr=sd_tr,
                         tr_idx=tr_idx, te_idx=te_idx), f)
    (out / "done.flag").write_text("ok\n")
    return metrics
