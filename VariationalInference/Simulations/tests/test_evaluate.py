import numpy as np
from VariationalInference.Simulations.evaluate import (
    hungarian_match, recovery_cosine, recovery_jaccard_oracle,
)

def test_hungarian_match_recovers_identity_under_permutation():
    rng = np.random.default_rng(0)
    beta_star = rng.gamma(1.0, 1.0, size=(100, 5))
    perm = np.array([3, 1, 4, 0, 2])
    beta_hat = beta_star[:, perm] + rng.normal(scale=0.01, size=(100, 5))
    assign, cost = hungarian_match(beta_hat, beta_star)
    # assign[ell] = factor index matched to truth column ell
    np.testing.assert_array_equal(np.argsort(assign), perm)

def test_recovery_cosine_high_on_planted():
    rng = np.random.default_rng(1)
    beta_star = rng.gamma(1.0, 1.0, size=(50, 3))
    beta_hat = np.hstack([beta_star, rng.normal(size=(50, 5))])
    assign, _ = hungarian_match(beta_hat, beta_star)
    cos = recovery_cosine(beta_hat, beta_star, assign)
    assert (cos > 0.99).all()

def test_recovery_jaccard_oracle_exact():
    p = 20
    beta_star = np.zeros((p, 2))
    beta_star[:5, 0] = 1.0          # |S*| = 5
    beta_star[5:12, 1] = 1.0        # |S*| = 7
    beta_hat = beta_star + 0.001 * np.random.default_rng(0).normal(size=beta_star.shape)
    assign = np.array([0, 1])
    S_star = [np.arange(5), np.arange(5, 12)]
    jac = recovery_jaccard_oracle(beta_hat, S_star, assign)
    assert all(j > 0.99 for j in jac)


def test_ranking_importance_decoy_at_bottom_drgp():
    from VariationalInference.Simulations.evaluate import ranking_metrics
    K = 5
    mu_v = np.array([2.0, -1.5, 0.05, 1.0, 0.1])    # factor 2 is the decoy
    theta = np.zeros((100, K))
    theta[:, 0] = np.random.default_rng(0).normal(size=100) * 2.0   # high SD
    theta[:, 1] = np.random.default_rng(1).normal(size=100) * 1.5
    theta[:, 2] = np.random.default_rng(2).normal(size=100) * 3.0   # decoy SD high
    theta[:, 3] = np.random.default_rng(3).normal(size=100) * 1.0
    theta[:, 4] = np.random.default_rng(4).normal(size=100) * 0.5
    assign = np.array([2, 0, 1, 3])
    causal_mask = np.array([False, True, True, False, True])
    res = ranking_metrics(mu_v=mu_v, theta_train=theta, beta_LR_std=None,
                          H_train=None, assign=assign, causal_mask=causal_mask)
    assert res["decoy_rank"] > 0
