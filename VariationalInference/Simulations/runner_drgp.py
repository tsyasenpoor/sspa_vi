"""One DRGP fit on one .h5ad. Wraps CAVI directly (not via quick_reference subprocess)
so we own the post-processing (integrated + posthoc AUC, Poisson-only fold-in)."""
from __future__ import annotations
import gzip
import json
import pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from VariationalInference.vi_cavi import CAVI
from . import config
from ._runner_utils import derive_seeds, patient_grouped_split


def _build_pathway_mask(mask_M: np.ndarray, mode: str, K: int) -> np.ndarray | None:
    """vi_cavi expects (n_pathways, n_genes)."""
    if mode == "unmasked":
        return None
    if mode in ("masked", "pathway_init"):
        return mask_M.T.astype(np.float32)        # (K_path, p)
    if mode == "combined":
        return mask_M.T.astype(np.float32)        # combined needs n_pathway_factors flag
    raise ValueError(mode)


def _integrated_logit(theta: np.ndarray, mu_v: np.ndarray) -> np.ndarray:
    """A_i = theta_i * mu_v.T (single-label collapse - mu_v is (kappa, K)). Take first col."""
    A = theta @ mu_v.T                                  # (n, kappa=1)
    return np.asarray(A).ravel().astype(np.float32)


def run(h5ad_path: str, mode: str, K: int, inner_seed: int,
        out_dir: str, regression_weight: float | None = None,
        max_iter: int | None = None) -> dict:
    A = ad.read_h5ad(h5ad_path)
    truth_idx = int(A.uns["truth_idx"]); h2 = float(A.uns["h2"]); r = float(A.uns["r"])
    method = f"drgp_{mode}"
    seeds = derive_seeds(truth_idx=truth_idx, h2=h2, r=r, inner_seed=inner_seed,
                         K=K, method=method)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    y = A.obs["y"].to_numpy().astype(np.float32)
    patient_ids = A.obs["patient_id"].to_numpy()
    n_test = max(1, int(round(config.N_PATIENTS / 5)))    # 8 patients
    tr_idx, te_idx = patient_grouped_split(patient_ids, n_test_patients=n_test,
                                           seed=seeds["split_seed"])
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    pathway_mask = _build_pathway_mask(A.uns["mask_M"], mode, K)
    n_pathway = None
    if mode == "combined":
        n_pathway = pathway_mask.shape[0]

    rw = config.REGRESSION_WEIGHT if regression_weight is None else regression_weight
    model = CAVI(
        n_factors=K, a=config.CAVI_A, c=config.CAVI_C,
        b_v=1.0, sigma_gamma=1.0,
        regression_weight=rw,
        use_class_weights=True,
        random_state=seeds["fit_seed"],
        mode=mode,
        pathway_mask=pathway_mask,
        n_pathway_factors=n_pathway,
    )
    model.fit(
        X_tr, y_tr,
        max_iter=max_iter or config.CAVI_MAX_ITER,
        check_freq=config.CAVI_CHECK_FREQ,
        tol=config.CAVI_TOL,
        v_warmup=config.CAVI_V_WARMUP,
        early_stopping=config.EARLY_STOPPING,
        n_patients=len(np.unique(patient_ids[tr_idx])),
        patient_ids=patient_ids[tr_idx],
        verbose=False,
    )

    theta_te = model.transform(X_te, n_iter=20)
    theta_tr = np.asarray(model.E_theta)
    mu_v = np.asarray(model.mu_v)                       # (kappa, K)
    A_int_te = _integrated_logit(theta_te, mu_v)
    A_int_tr = _integrated_logit(theta_tr, mu_v)
    cell_auc_integrated = float(roc_auc_score(y_te, A_int_te))

    head = LogisticRegressionCV(
        Cs=config.LR_C_GRID, cv=config.LR_CV_FOLDS, penalty="l1",
        solver="saga", max_iter=config.LR_MAX_ITER, scoring="roc_auc",
        n_jobs=1,
    ).fit(theta_tr, y_tr)
    posthoc_pred = head.predict_proba(theta_te)[:, 1]
    cell_auc_posthoc = float(roc_auc_score(y_te, posthoc_pred))

    y_pred = (A_int_te > 0).astype(int)
    metrics = {
        "method": method, "mode": mode, "K": int(K),
        "truth_idx": truth_idx, "h2": h2, "r": r, "inner_seed": int(inner_seed),
        "regression_weight": float(rw),
        "cell_auc_integrated": cell_auc_integrated,
        "cell_auc_posthoc": cell_auc_posthoc,
        "cell_f1": float(f1_score(y_te, y_pred)),
        "cell_acc": float(accuracy_score(y_te, y_pred)),
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pd.DataFrame({
        "cell_idx": te_idx, "y_true": y_te,
        "A_integrated": A_int_te, "A_posthoc": posthoc_pred,
        "theta_norm_te": np.linalg.norm(theta_te, axis=1),
    }).to_parquet(out / "fold_predictions.parquet")

    payload = dict(mu_beta=np.asarray(model.mu_beta), mu_v=mu_v,
                   rho=np.asarray(getattr(model, "rho", np.zeros(0))),
                   theta_tr=theta_tr.astype(np.float32),
                   theta_te=theta_te.astype(np.float32),
                   tr_idx=tr_idx, te_idx=te_idx,
                   posthoc_coef=head.coef_.ravel())
    with gzip.open(out / "result.pkl.gz", "wb") as f:
        pickle.dump(payload, f)
    (out / "done.flag").write_text("ok\n")
    return metrics
