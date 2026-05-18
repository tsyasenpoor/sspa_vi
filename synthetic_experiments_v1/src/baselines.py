"""
Method wrappers for the DRGP synthetic experiments.

Each fitter takes a GroundTruth instance and K_fit, returns:
    {
        "Beta_hat":  (p, K_fit)         gene loadings
        "Theta_hat": (n, K_fit)         sample activations
        "v_hat":     (K_fit,)           regression weights (factor part only)
        "gamma_hat": (q,) or None       direct covariate weights (DRGP only)
        "R_beta":    (p, K_fit) or None spike-and-slab posterior inclusion (DRGP only)
        "predict":   callable(GT) -> (n_new,) probability scores
        "elapsed_s": float
        "extra":     dict (model object, raw posteriors, etc., kept for debug)
    }
"""
from __future__ import annotations

import sys
import time
from typing import Callable, Optional

import numpy as np

# Ensure BRay package is on the path so vi_cavi import works.
_BRAY_ROOT = "/labs/Aguiar/SSPA_BRAY/BRay"
if _BRAY_ROOT not in sys.path:
    sys.path.insert(0, _BRAY_ROOT)


# ----------------------------------------------------------------------
# Train / val split helper for DRGP early stopping
# ----------------------------------------------------------------------
def _train_val_split(n: int, val_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n)))
    return idx[n_val:], idx[:n_val]


# ----------------------------------------------------------------------
# DRGP (vi_cavi.CAVI)
# ----------------------------------------------------------------------
def fit_drgp(
    gt,
    K_fit: int,
    mode: str = "unmasked",
    pathway_mask: Optional[np.ndarray] = None,
    n_pathway_factors: Optional[int] = None,
    max_iter: int = 600,
    check_freq: int = 5,
    tol: float = 1e-3,
    early_stopping: str = "heldout_ll",
    b_v: float = 1.0,
    sigma_gamma: float = 1.0,
    regression_weight: float = 1.0,
    use_intercept: bool = True,
    val_frac: float = 0.1,
    random_state: int = 0,
    verbose: bool = False,
) -> dict:
    """Fit Supervised Poisson Factorization via vi_cavi.CAVI.

    `regression_weight` is set to 1.0 so the in-fit auto-scale (`*= nnz/n`)
    fires once; do NOT pre-scale (per CLAUDE.md).
    """
    from VariationalInference.vi_cavi import CAVI  # noqa: E402

    t0 = time.time()
    rng = np.random.default_rng(random_state)
    n_total = gt.X.shape[0]
    if val_frac <= 0.0:
        idx_tr = np.arange(n_total)
        idx_val = np.array([], dtype=np.int64)
        X_val_arg = y_val_arg = X_aux_val_arg = None
        if early_stopping == "heldout_ll":
            early_stopping = "elbo"  # heldout_ll needs a val set
    else:
        idx_tr, idx_val = _train_val_split(n_total, val_frac, rng)
        X_val_arg = gt.Y[idx_val].astype(np.float32)
        y_val_arg = gt.y[idx_val].astype(np.float32)
        X_aux_val_arg = gt.X[idx_val].astype(np.float32)

    X_train = gt.Y[idx_tr].astype(np.float32)
    y_train = gt.y[idx_tr].astype(np.float32)
    X_aux_train = gt.X[idx_tr].astype(np.float32)

    model = CAVI(
        n_factors=K_fit,
        mode=mode,
        pathway_mask=pathway_mask,
        n_pathway_factors=n_pathway_factors,
        b_v=b_v,
        sigma_gamma=sigma_gamma,
        regression_weight=regression_weight,
        use_intercept=use_intercept,
        random_state=random_state,
    )
    model.fit(
        X_train=X_train, y_train=y_train, X_aux_train=X_aux_train,
        X_val=X_val_arg, y_val=y_val_arg, X_aux_val=X_aux_val_arg,
        max_iter=max_iter, check_freq=check_freq, tol=tol,
        early_stopping=early_stopping, verbose=verbose,
    )

    # Posterior means on the FULL cohort: re-infer theta on every row.
    # (Cleanest path for matching/eval.)
    a_th, b_th = model._infer_theta_sparse(
        _coo(gt.Y), n_new=gt.Y.shape[0], n_iter=20,
        X_aux_new=model._prepend_intercept(
            np.asarray(gt.X, dtype=np.float32), n=gt.Y.shape[0]),
    )
    Theta_hat = _to_numpy(a_th / b_th)

    Beta_hat = _to_numpy(model.E_beta)                   # (p, K_fit)
    R_beta = _to_numpy(model.r_beta) if hasattr(model, "r_beta") else None

    # mu_v shape: (kappa, K_fit). Single-label here => kappa=1.
    v_hat = _to_numpy(model.mu_v).reshape(-1)
    if v_hat.size != K_fit:
        # CT mode shouldn't apply here, but defensively slice
        v_hat = v_hat[:K_fit]

    # mu_gamma shape: (kappa, p_aux_including_intercept).
    # First entry is intercept (if use_intercept=True); aux covariates follow.
    mu_gamma = _to_numpy(model.mu_gamma)
    if model.use_intercept and mu_gamma.shape[1] >= 1:
        gamma_hat = mu_gamma[0, 1:]
    else:
        gamma_hat = mu_gamma[0] if mu_gamma.size > 0 else None

    def predict(gt_new) -> np.ndarray:
        probs = model.predict_proba(
            gt_new.Y.astype(np.float32),
            X_aux_new=gt_new.X.astype(np.float32),
            n_iter=20,
        )
        return _to_numpy(probs).reshape(-1)

    return {
        "Beta_hat": Beta_hat,
        "Theta_hat": Theta_hat,
        "v_hat": v_hat,
        "gamma_hat": gamma_hat,
        "R_beta": R_beta,
        "predict": predict,
        "elapsed_s": time.time() - t0,
        "extra": {"model": model, "idx_tr": idx_tr, "idx_val": idx_val},
    }


# ----------------------------------------------------------------------
# NMF + L1-LR
# ----------------------------------------------------------------------
def fit_nmf_lr(gt, K_fit: int, random_state: int = 0, **_) -> dict:
    from sklearn.decomposition import NMF
    from sklearn.linear_model import LogisticRegressionCV

    t0 = time.time()
    nmf = NMF(n_components=K_fit, init="nndsvda", max_iter=500,
              random_state=random_state)
    Theta_hat = nmf.fit_transform(gt.Y.astype(float))      # (n, K)
    Beta_hat = nmf.components_.T                           # (p, K)

    # Combine factor features + aux X for OOD-style prediction
    feats = np.hstack([Theta_hat, gt.X])
    lr = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=10, cv=5, max_iter=5000,
        random_state=random_state,
    )
    lr.fit(feats, gt.y)
    full_coef = lr.coef_.ravel()
    v_hat = full_coef[:K_fit]
    gamma_hat = full_coef[K_fit:]

    def predict(gt_new) -> np.ndarray:
        Theta_new = nmf.transform(gt_new.Y.astype(float))
        return lr.predict_proba(np.hstack([Theta_new, gt_new.X]))[:, 1]

    return {
        "Beta_hat": Beta_hat,
        "Theta_hat": Theta_hat,
        "v_hat": v_hat,
        "gamma_hat": gamma_hat,
        "R_beta": None,
        "predict": predict,
        "elapsed_s": time.time() - t0,
        "extra": {"nmf": nmf, "lr": lr},
    }


# ----------------------------------------------------------------------
# PCA + L1-LR
# ----------------------------------------------------------------------
def fit_pca_lr(gt, K_fit: int, random_state: int = 0, **_) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegressionCV

    t0 = time.time()
    Y_log = np.log1p(gt.Y.astype(float))
    pca = PCA(n_components=K_fit, random_state=random_state)
    Theta_hat = pca.fit_transform(Y_log)
    Beta_hat = pca.components_.T

    feats = np.hstack([Theta_hat, gt.X])
    lr = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=10, cv=5, max_iter=5000,
        random_state=random_state,
    )
    lr.fit(feats, gt.y)
    full_coef = lr.coef_.ravel()
    v_hat = full_coef[:K_fit]
    gamma_hat = full_coef[K_fit:]

    def predict(gt_new) -> np.ndarray:
        Theta_new = pca.transform(np.log1p(gt_new.Y.astype(float)))
        return lr.predict_proba(np.hstack([Theta_new, gt_new.X]))[:, 1]

    return {
        "Beta_hat": Beta_hat,
        "Theta_hat": Theta_hat,
        "v_hat": v_hat,
        "gamma_hat": gamma_hat,
        "R_beta": None,
        "predict": predict,
        "elapsed_s": time.time() - t0,
        "extra": {"pca": pca, "lr": lr},
    }


# ----------------------------------------------------------------------
# Plain L1-LR on log1p(Y) (no factorization)
# ----------------------------------------------------------------------
def fit_plain_lr(gt, K_fit: int = 0, random_state: int = 0, **_) -> dict:
    from sklearn.linear_model import LogisticRegressionCV

    t0 = time.time()
    feats = np.hstack([np.log1p(gt.Y.astype(float)), gt.X])
    lr = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=10, cv=5, max_iter=5000,
        random_state=random_state,
    )
    lr.fit(feats, gt.y)

    def predict(gt_new) -> np.ndarray:
        f = np.hstack([np.log1p(gt_new.Y.astype(float)), gt_new.X])
        return lr.predict_proba(f)[:, 1]

    return {
        "Beta_hat": None,
        "Theta_hat": None,
        "v_hat": None,
        "gamma_hat": None,
        "R_beta": None,
        "predict": predict,
        "elapsed_s": time.time() - t0,
        "extra": {"lr": lr},
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _coo(Y):
    """Convert dense to scipy COO for vi_cavi internals."""
    import scipy.sparse as sp
    return sp.coo_matrix(Y.astype(np.float32))


def _to_numpy(x):
    """Bring jax arrays back to numpy when JAX backend is active."""
    if hasattr(x, "device_buffer") or hasattr(x, "addressable_data"):
        return np.asarray(x)
    if hasattr(x, "block_until_ready"):
        try:
            return np.asarray(x.block_until_ready())
        except Exception:
            return np.asarray(x)
    return np.asarray(x)


_SCHPF_ROOT = "/labs/Aguiar/SSPA_BRAY/scHPF"
if _SCHPF_ROOT not in sys.path:
    sys.path.insert(0, _SCHPF_ROOT)


def fit_schpf_lr(gt, K_fit: int, random_state: int = 0,
                  max_iter: int = 200, min_iter: int = 20, **_) -> dict:
    """scHPF (Levitin lab hierarchical Poisson factorization) + L1-LR.

    Trains scHPF on raw counts, extracts cell_score (theta-like) and
    gene_score (beta-like), then fits L1-LR on [cell_score, X_aux] for
    the regression head. Test predictions project new cells via scHPF.project.
    """
    import scipy.sparse as sp
    from schpf import scHPF                                                  # noqa: E402
    from sklearn.linear_model import LogisticRegressionCV

    t0 = time.time()
    np.random.seed(random_state)
    model = scHPF(nfactors=K_fit, min_iter=min_iter, max_iter=max_iter,
                   verbose=False)
    model.fit(sp.coo_matrix(gt.Y.astype(np.int32)))
    Theta_hat = np.asarray(model.cell_score())                  # (n, K)
    Beta_hat = np.asarray(model.gene_score())                   # (p, K)

    feats = np.hstack([Theta_hat, gt.X])
    lr = LogisticRegressionCV(
        penalty="l1", solver="saga", Cs=10, cv=5, max_iter=5000,
        random_state=random_state,
    )
    lr.fit(feats, gt.y)
    full_coef = lr.coef_.ravel()
    v_hat = full_coef[:K_fit]
    gamma_hat = full_coef[K_fit:]

    def predict(gt_new) -> np.ndarray:
        # Project new cells; gene distributions frozen
        proj = model.project(sp.coo_matrix(gt_new.Y.astype(np.int32)),
                              recalc_bp=False, replace=False)
        Theta_new = np.asarray(proj.cell_score())
        return lr.predict_proba(np.hstack([Theta_new, gt_new.X]))[:, 1]

    return {
        "Beta_hat": Beta_hat,
        "Theta_hat": Theta_hat,
        "v_hat": v_hat,
        "gamma_hat": gamma_hat,
        "R_beta": None,
        "predict": predict,
        "elapsed_s": time.time() - t0,
        "extra": {"schpf": model, "lr": lr},
    }


METHOD_FITTERS: dict[str, Callable[..., dict]] = {
    "drgp_unmasked": lambda gt, K_fit, **kw: fit_drgp(gt, K_fit, mode="unmasked", **kw),
    "drgp_masked":   lambda gt, K_fit, **kw: fit_drgp(gt, K_fit, mode="masked", **kw),
    "drgp_combined": lambda gt, K_fit, **kw: fit_drgp(gt, K_fit, mode="combined", **kw),
    "schpf_lr": fit_schpf_lr,
    "nmf_lr": fit_nmf_lr,
    "pca_lr": fit_pca_lr,
    "plain_lr": fit_plain_lr,
}
