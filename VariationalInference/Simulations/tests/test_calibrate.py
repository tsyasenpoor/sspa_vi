import numpy as np
from scipy import sparse
from VariationalInference.Simulations.calibrate import ss_type_from_means
from VariationalInference.Simulations.calibrate_subdominance_r import ss_type_from_baseline


def test_ss_type_from_means_matches_sparse_kernel_on_means():
    """ss_type computed on a dense mean matrix equals the sparse-kernel value when
    the sparse input is the same mean matrix cast to counts (round-trip sanity)."""
    rng = np.random.default_rng(0)
    G, N, T = 200, 600, 4
    mu = rng.gamma(2.0, 1.0, size=(G, N)).astype(np.float64)
    types = rng.integers(0, T, size=N)
    counts = sparse.csr_matrix(mu.T.astype(np.float64))   # cells×genes; same matrix as means
    dense_ss = ss_type_from_means(mu, types)
    sparse_ss = ss_type_from_baseline(counts, types)
    # Both compute the same library-norm + log1p transform, then between-type SS.
    np.testing.assert_allclose(dense_ss, sparse_ss, rtol=1e-3)
