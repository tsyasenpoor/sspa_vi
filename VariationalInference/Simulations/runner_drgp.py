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


def _integrated_logit(theta: np.ndarray, mu_v: np.ndarray,
                      X_aux: np.ndarray | None = None,
                      mu_gamma: np.ndarray | None = None) -> np.ndarray:
    """A_i = theta_i * mu_v.T + X_aux_i * mu_gamma.T (single-label collapse)."""
    A = theta @ mu_v.T                                  # (n, kappa=1)
    if X_aux is not None and mu_gamma is not None:
        A = A + X_aux @ mu_gamma.T
    return np.asarray(A).ravel().astype(np.float32)


def run(h5ad_path: str, mode: str, K: int, inner_seed: int,
        out_dir: str, regression_weight: float | None = None,
        max_iter: int | None = None, early_stopping: str | None = None,
        verbose: bool = False) -> dict:
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
    intercept_aux_tr = np.ones((X_tr.shape[0], 1), dtype=np.float32)
    intercept_aux_te = np.ones((X_te.shape[0], 1), dtype=np.float32)
    model.fit(
        X_tr, y_tr,
        X_aux_train=intercept_aux_tr,
        max_iter=max_iter or config.CAVI_MAX_ITER,
        check_freq=config.CAVI_CHECK_FREQ,
        tol=config.CAVI_TOL,
        v_warmup=config.CAVI_V_WARMUP,
        early_stopping=early_stopping or config.EARLY_STOPPING,
        n_patients=len(np.unique(patient_ids[tr_idx])),
        patient_ids=patient_ids[tr_idx],
        verbose=verbose,
    )

    # Poisson-only fold-in theta for BOTH train and test cells. theta_tr_pois is
    # the regime-consistent training theta (no R_quad/R_lin) — used to fit the
    # regime-consistent posthoc head so the head is applied to and trained on
    # the same theta regime.
    theta_te = np.asarray(
        model.transform(X_te, n_iter=20, supervised=False)["E_theta"])
    theta_tr_pois = np.asarray(
        model.transform(X_tr, n_iter=20, supervised=False)["E_theta"])
    theta_tr = np.asarray(model.E_theta)                # supervised-corrected E[theta]
    mu_v = np.asarray(model.mu_v)                       # (kappa, K)
    mu_gamma = np.asarray(model.mu_gamma)

    # Integrated logit WITH gamma (held-out, for §7.5 calibration vs LR baselines).
    A_int_te = _integrated_logit(theta_te, mu_v, intercept_aux_te, mu_gamma)
    A_int_tr = _integrated_logit(theta_tr, mu_v, intercept_aux_tr, mu_gamma)
    cell_auc_integrated = float(roc_auc_score(y_te, A_int_te))
    cell_auc_integrated_train = float(roc_auc_score(y_tr, A_int_tr))

    # Theta-only logit (γ contribution excluded) — measures whether (β, υ) carry
    # discriminative signal independent of the γ intercept absorbing labels.
    # This is the engagement-gate metric: γ-memorization is invisible to it.
    A_theta_only_te = (theta_te @ mu_v.T).ravel().astype(np.float32)
    A_theta_only_tr = (theta_tr @ mu_v.T).ravel().astype(np.float32)
    cell_auc_theta_only = float(roc_auc_score(y_te, A_theta_only_te))
    cell_auc_theta_only_train = float(roc_auc_score(y_tr, A_theta_only_tr))

    # Posthoc head fit on the SUPERVISED theta_tr (matches the LR baselines pattern
    # for §7.5 — both heads ingest the joint-fit cell scores). Has the supervised vs
    # Poisson-only regime mismatch when scored on theta_te.
    head = LogisticRegressionCV(
        Cs=config.LR_C_GRID, cv=config.LR_CV_FOLDS, penalty="l1",
        solver="saga", max_iter=config.LR_MAX_ITER, scoring="roc_auc",
        n_jobs=1,
    ).fit(theta_tr, y_tr)
    posthoc_pred = head.predict_proba(theta_te)[:, 1]
    cell_auc_posthoc = float(roc_auc_score(y_te, posthoc_pred))
    cell_auc_posthoc_train = float(roc_auc_score(y_tr, head.predict_proba(theta_tr)[:, 1]))

    # Regime-consistent posthoc head: fit on Poisson-only theta_tr_pois, score on
    # Poisson-only theta_te. Removes the supervised-vs-Poisson regime mismatch
    # without paying for per-fold refits (which don't fix it anyway).
    head_consistent = LogisticRegressionCV(
        Cs=config.LR_C_GRID, cv=config.LR_CV_FOLDS, penalty="l1",
        solver="saga", max_iter=config.LR_MAX_ITER, scoring="roc_auc",
        n_jobs=1,
    ).fit(theta_tr_pois, y_tr)
    proba_consistent = head_consistent.predict_proba(theta_te)[:, 1]
    cell_auc_consistent = float(roc_auc_score(y_te, proba_consistent))

    y_pred = (A_int_te > 0).astype(int)
    metrics = {
        "method": method, "mode": mode, "K": int(K),
        "truth_idx": truth_idx, "h2": h2, "r": r, "inner_seed": int(inner_seed),
        "regression_weight": float(rw),
        "cell_auc_integrated": cell_auc_integrated,
        "cell_auc_integrated_train": cell_auc_integrated_train,
        "cell_auc_theta_only": cell_auc_theta_only,
        "cell_auc_theta_only_train": cell_auc_theta_only_train,
        "cell_auc_posthoc": cell_auc_posthoc,
        "cell_auc_posthoc_train": cell_auc_posthoc_train,
        "cell_auc_consistent": cell_auc_consistent,
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

    # E_beta is the posterior mean of beta (p, K); r_beta is the spike-slab
    # inclusion prob (p, K), only present when use_spike_slab=True (mode != masked).
    # We keep the result-pickle keys as `mu_beta`/`rho` so evaluate.py reads stay stable.
    rho = (np.asarray(model.r_beta) if getattr(model, "use_spike_slab", False)
           else np.zeros((0, 0), dtype=np.float32))
    payload = dict(mu_beta=np.asarray(model.E_beta), mu_v=mu_v,
                   rho=rho,
                   theta_tr=theta_tr.astype(np.float32),
                   theta_te=theta_te.astype(np.float32),
                   tr_idx=tr_idx, te_idx=te_idx,
                   posthoc_coef=head.coef_.ravel())
    with gzip.open(out / "result.pkl.gz", "wb") as f:
        pickle.dump(payload, f)
    (out / "done.flag").write_text("ok\n")
    return metrics
