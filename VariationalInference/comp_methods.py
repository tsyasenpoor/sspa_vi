from sklearn.svm import SVC
import os
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib


def _to_dense(x):
    """Convert torch tensor, scipy sparse, or numpy array → dense numpy."""
    if hasattr(x, 'numpy'):          # torch tensor
        x = x.numpy()
    if sp.issparse(x):               # scipy sparse
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


# ── algorithm registry ──────────────────────────────────────────────────────
_CLASSIFIERS = {
    "svm": lambda: make_pipeline(StandardScaler(), SVC(probability=True)),
    "lr":  lambda: make_pipeline(StandardScaler(), LogisticRegression()),
    "lrl": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="saga")),
    "lrr": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l2")),
    # mf variants share the same classifiers, NMF handled separately
    "mflr":  lambda: make_pipeline(StandardScaler(), LogisticRegression()),
    "mflrl": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="saga")),
    "mflrr": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l2")),
}


def train_alg(
    algorithm, x_data_train, x_aux_data_train, y_data_train, save_path, latent_dim
):
    """Train a baseline classifier. Returns (model, nmf) for mf* methods, model otherwise.

    Parameters
    ----------
    algorithm : str
        One of: svm, lr, lrl, lrr, mflr, mflrl, mflrr.
    x_data_train : array-like
        Gene expression matrix (n_cells, n_genes). Accepts numpy, scipy sparse, or torch tensor.
    x_aux_data_train : array-like
        Auxiliary covariate matrix (n_cells, n_aux).
    y_data_train : array-like
        Binary label vector (n_cells,).
    save_path : str
        Directory to save model artifacts.
    latent_dim : int
        Number of NMF components (only used for mf* methods).

    Returns
    -------
    model : sklearn Pipeline
        Fitted classifier.
    nmf : NMF or None
        Fitted NMF object (only for mf* methods, None otherwise).
    """
    X_gex = _to_dense(x_data_train)
    X_aux = _to_dense(x_aux_data_train)
    y = _to_dense(y_data_train).ravel().astype(int)

    if algorithm not in _CLASSIFIERS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(_CLASSIFIERS.keys())}")

    nmf_obj = None
    if algorithm.startswith("mf"):
        nmf_obj = NMF(n_components=latent_dim)
        X_latent = nmf_obj.fit_transform(X_gex)
        X_train = np.concatenate((X_latent, X_aux), axis=1)
    else:
        X_train = np.concatenate((X_gex, X_aux), axis=1)

    model = _CLASSIFIERS[algorithm]()
    model.fit(X_train, y)

    train_accuracy = model.score(X_train, y)
    print(f"[{algorithm}] Training accuracy: {train_accuracy:.4f}")

    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model, os.path.join(save_path, f"{algorithm}_model.pkl"))
    if nmf_obj is not None:
        joblib.dump(nmf_obj, os.path.join(save_path, f"{algorithm}_nmf.pkl"))

    return model, nmf_obj


def eval_alg(
    model, algorithm, x_data_test, x_aux_data_test, y_data_test, save_path, latent_dim,
    nmf=None
):
    """Evaluate a trained baseline classifier on held-out data.

    Parameters
    ----------
    model : sklearn Pipeline
        Fitted classifier from train_alg.
    algorithm : str
        Algorithm name (must match train_alg).
    x_data_test : array-like
        Gene expression matrix (n_cells, n_genes).
    x_aux_data_test : array-like
        Auxiliary covariate matrix (n_cells, n_aux).
    y_data_test : array-like
        True labels (n_cells,).
    save_path : str
        Directory to save evaluation results.
    latent_dim : int
        Number of NMF components (only used for mf* methods if nmf is None).
    nmf : NMF or None
        Pre-fitted NMF from train_alg. If None and algorithm is mf*, a new NMF is fitted
        on test data (NOT recommended — pass the training NMF).
    """
    print(f"evaluating {algorithm}")

    X_gex = _to_dense(x_data_test)
    X_aux = _to_dense(x_aux_data_test)
    y_true = _to_dense(y_data_test).ravel().astype(int)

    if algorithm.startswith("mf"):
        if nmf is not None:
            X_latent = nmf.transform(X_gex)
        else:
            print(f"  WARNING: No training NMF provided for {algorithm}; re-fitting on test data.")
            nmf_new = NMF(n_components=latent_dim)
            X_latent = nmf_new.fit_transform(X_gex)
        X_test = np.concatenate((X_latent, X_aux), axis=1)
    else:
        X_test = np.concatenate((X_gex, X_aux), axis=1)

    y_pred = model.predict(X_test)
    y_pred_proba_full = model.predict_proba(X_test)
    n_classes = y_pred_proba_full.shape[1]

    test_accuracy = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred, average="weighted")
    test_recall = recall_score(y_true, y_pred, average="weighted")
    test_f1 = f1_score(y_true, y_pred, average="weighted")
    test_confusion_matrix = confusion_matrix(y_true, y_pred)
    if n_classes == 2:
        test_roc_auc = roc_auc_score(y_true, y_pred_proba_full[:, 1])
    else:
        test_roc_auc = roc_auc_score(y_true, y_pred_proba_full, multi_class='ovr', average='weighted')

    print(f"  accuracy:  {test_accuracy:.4f}")
    print(f"  precision: {test_precision:.4f}")
    print(f"  recall:    {test_recall:.4f}")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  ROC AUC:   {test_roc_auc:.4f}")
    print(f"  confusion matrix:\n{test_confusion_matrix}")

    results = {
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "confusion_matrix": test_confusion_matrix,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(results, os.path.join(save_path, f"{algorithm}_results.pkl"))
    return results
