"""Fixed-loadings inductive projection and feature standardization."""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls


def poisson_foldin_solve(X_test: np.ndarray, beta: np.ndarray,
                         n_iter: int = 20, eps: float = 1e-9) -> np.ndarray:
    """Multiplicative-update Poisson fold-in for held-out cells. Same kernel scHPF uses.

    X_test: (n_test, p)  beta: (K, p).  Returns (n_test, K) theta_hat, nonneg.
    """
    n, _ = X_test.shape
    K = beta.shape[0]
    theta = np.ones((n, K), dtype=np.float64) * (X_test.sum(axis=1, keepdims=True) /
                                                  np.maximum(beta.sum(axis=1).sum(), 1.0))
    beta = beta.astype(np.float64) + eps
    for _ in range(n_iter):
        lam = theta @ beta + eps
        ratio = X_test / lam
        num = ratio @ beta.T
        den = beta.sum(axis=1)[None, :] + eps
        theta = theta * num / den
    return theta.astype(np.float32)


def nmf_nnls_project(X_test: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Row-wise NNLS project of X_test onto a fixed dictionary W (K, p).
    Returns H (n_test, K)."""
    n = X_test.shape[0]
    K = W.shape[0]
    H = np.zeros((n, K), dtype=np.float32)
    Wt = W.T
    for i in range(n):
        H[i], _ = nnls(Wt, X_test[i].astype(np.float64))
    return H


def standardize_factors(H_train: np.ndarray, H_test: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (H_train_std, H_test_std, mean_train, sd_train).
    Test data uses train statistics only — no leakage."""
    mean = H_train.mean(axis=0)
    sd = H_train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    Htr = ((H_train - mean) / sd).astype(np.float32)
    Hte = ((H_test - mean) / sd).astype(np.float32)
    return Htr, Hte, mean.astype(np.float32), sd
