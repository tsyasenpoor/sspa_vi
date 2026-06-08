import numpy as np
from VariationalInference.Simulations.projection import (
    poisson_foldin_solve, nmf_nnls_project, standardize_factors,
)

def test_poisson_foldin_recovers_known_theta():
    rng = np.random.default_rng(0)
    K, G = 4, 200
    beta = rng.gamma(1.0, 1.0, size=(K, G))
    theta_true = rng.gamma(1.0, 1.0, size=(20, K))
    lam = theta_true @ beta
    X = rng.poisson(lam).astype(np.float32)
    theta_hat = poisson_foldin_solve(X, beta, n_iter=300)
    cos = (theta_hat * theta_true).sum(axis=1) / (
        np.linalg.norm(theta_hat, axis=1) * np.linalg.norm(theta_true, axis=1) + 1e-9)
    assert cos.mean() > 0.95

def test_nmf_nnls_project_nonneg():
    rng = np.random.default_rng(1)
    K, G = 4, 200
    W = rng.gamma(1.0, 1.0, size=(K, G))
    X = rng.gamma(1.0, 1.0, size=(20, G))
    H = nmf_nnls_project(X, W)
    assert H.shape == (20, K)
    assert (H >= -1e-8).all()

def test_standardize_factors_train_stats_only():
    rng = np.random.default_rng(2)
    H_train = rng.normal(loc=3.0, scale=2.0, size=(100, 5))
    H_test = rng.normal(loc=10.0, scale=4.0, size=(50, 5))
    Ht_train, Ht_test, mean, sd = standardize_factors(H_train, H_test)
    np.testing.assert_allclose(Ht_train.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(Ht_train.std(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(Ht_test.mean(axis=0), (H_test.mean(axis=0) - mean) / sd, atol=1e-6)
