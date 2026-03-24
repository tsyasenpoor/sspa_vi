"""
Coordinate Ascent VI for Supervised Poisson Factorization (DRGP)
================================================================

Poisson factorization core follows scHPF (Levitin et al., MSB 2019)
exactly. Supervision via Jaakkola-Jordan logistic regression bound
from the DRGP manuscript appendix.

scHPF update equations (Gopalan et al. 2014, Eqs 7-9):
  phi_{ij} ~ exp{ psi(a^theta_{ik}) - log(b^theta_{ik}) + psi(a^beta_{jk}) - log(b^beta_{jk}) }
  a^beta_{jk} = c + Sum_i x_{ij} phi_{ijk}
  b^beta_{jk} = E[eta_j] + Sum_i E[theta_{ik}]
  a^theta_{ik} = a + Sum_j x_{ij} phi_{ijk}
  b^theta_{ik} = E[xi_i] + Sum_j E[beta_{jk}]  (+ JJ regression correction)

Hierarchical priors:
  xi_i ~ Gamma(a', b')     ->  a^xi_i = a' + K*a  (constant)
                                b^xi_i = b' + Sum_k E[theta_{ik}]
  eta_j ~ Gamma(c', d')    ->  a^eta_j = c' + K*c  (constant)
                                b^eta_j = d' + Sum_k E[beta_{jk}]

Key differences from previous vi_cavi.py:
1. bp, dp are SCALARS (scHPF), not per-cell/gene vectors
2. Random init: U(0.5*prior, 1.5*prior) for both shape AND rate
3. First iteration uses random Dirichlet phi (scHPF symmetry breaking)
4. No damping -- pure coordinate ascent
5. No diversity noise, no annealing, no warmup phases
6. All parameters (theta, v, gamma, zeta) learned jointly from iteration 0
7. No spike-and-slab

GPU/JAX acceleration:
- When JAX is installed the hot numerical paths (phi, ELBO, updates)
  run on GPU if available, otherwise on CPU via XLA.
- Falls back transparently to NumPy/SciPy when JAX is absent.

References:
- Gopalan, Hofman, Blei (2014) "Scalable Recommendation with HPF"
- Levitin et al. (2019) "scHPF", Molecular Systems Biology
- Jaakkola & Jordan (2000) variational logistic bound
- Hoffman et al. (2013) "Stochastic Variational Inference"
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Any, List
import time

try:
    from .jax_backend import (
        xp, USE_JAX, HAS_GPU, to_device, to_numpy,
        digamma, gammaln, logsumexp_rows, softmax_rows,
        log_expit, lambda_jj, scatter_add_to, phi_chunk_core,
        expit as _expit, backend_info,
    )
except ImportError:
    from jax_backend import (
        xp, USE_JAX, HAS_GPU, to_device, to_numpy,
        digamma, gammaln, logsumexp_rows, softmax_rows,
        log_expit, lambda_jj, scatter_add_to, phi_chunk_core,
        expit as _expit, backend_info,
    )


def _auto_chunk_size(nnz, K, target_gb=None):
    """Auto-tune chunk size to target a given work-array memory budget.

    With K factors, each chunk of C entries uses C * K * 4 bytes (float32).
    On GPU, uses a conservative fraction of VRAM to leave headroom for
    scatter promotions, parameter tensors, and allocator fragmentation.
    On CPU, targets ~4GB by default.
    """
    if target_gb is None:
        if HAS_GPU:
            # Use ~3% of GPU memory for work arrays. In practice, scatter-add
            # and dtype promotion can transiently require additional buffers,
            # so this budget must remain conservative at large K.
            try:
                import jax
                dev = [d for d in jax.devices() if d.platform == "gpu"][0]
                mem_bytes = dev.memory_stats()["bytes_limit"]
                target_gb = mem_bytes / (1024 ** 3) * 0.03
            except Exception:
                target_gb = 2.0  # conservative GPU default
        else:
            target_gb = 4.0
    max_by_mem = int(target_gb * (1024 ** 3) / (K * 4))
    # Floor: keep at least ~200MB work array (but never below 20k nnz)
    # to avoid excessive loop overhead when K is very large.
    floor = max(20_000, int(0.2 * (1024 ** 3) / (K * 4)))
    return max(floor, min(max_by_mem, nnz))


def _row_chunk_size(n, K, n_intermediates=2, target_gb=None):
    """Return number of rows to process at a time for (n, K) operations.

    Limits the peak memory of row-wise temporaries to *target_gb*.

    Parameters
    ----------
    n : int
        Total number of rows (cells).
    K : int
        Number of factors.
    n_intermediates : int
        Number of simultaneous (chunk, K) temporaries at peak.
    target_gb : float or None
        Memory budget in GiB.  ``None`` auto-selects based on backend.
    """
    if target_gb is None:
        if HAS_GPU:
            try:
                import jax
                dev = [d for d in jax.devices() if d.platform == "gpu"][0]
                mem_bytes = dev.memory_stats()["bytes_limit"]
                # Use ~3% of VRAM for row-chunked temporaries (conservative
                # to leave headroom for allocator fragmentation and other ops)
                target_gb = mem_bytes / (1024 ** 3) * 0.03
            except Exception:
                target_gb = 0.5
        else:
            target_gb = 2.0
    bytes_per_row = n_intermediates * K * 4  # float32
    max_rows = max(1024, int(target_gb * (1024 ** 3) / bytes_per_row))
    return min(max_rows, n)


def _is_oom_error(exc: Exception) -> bool:
    """Best-effort detection of JAX/XLA OOM errors."""
    msg = str(exc).lower()
    return (
        "resource_exhausted" in msg
        or "out of memory" in msg
        or "cuda_error_out_of_memory" in msg
        or "allocator" in msg and "memory" in msg
    )


class CAVI:
    """
    CAVI for Supervised Poisson Factorization.

    Parameters
    ----------
    n_factors : int
        Number of latent factors K.
    a : float
        Gamma shape prior for theta (cell loadings). Default 0.3 (scHPF).
    ap : float
        Gamma shape prior for xi (cell capacity). Default 1.0.
    c : float
        Gamma shape prior for beta (gene loadings). Default 0.3 (scHPF).
    cp : float
        Gamma shape prior for eta (gene capacity). Default 1.0.
    sigma_v : float
        Gaussian prior std for v (regression weights). Used when v_prior='normal'.
    b_v : float
        Laplace prior scale for v (regression weights). Used when v_prior='laplace'.
        Smaller b_v = stronger sparsity. Var[v] = 2*b_v^2.
    v_prior : str
        Prior distribution for v: 'normal' (Gaussian) or 'laplace' (Bayesian Lasso).
    sigma_gamma : float
        Gaussian prior std for gamma (auxiliary covariate weights).
    regression_weight : float
        Scalar weight for the classification term.
    use_class_weights : bool
        If True, apply balanced class weights per label to the Bernoulli
        regression loss.  Positive samples for label k are weighted by
        n / (2 * n_pos_k) and negatives by n / (2 * n_neg_k), so the
        total weighted count per label stays the same but rare classes
        contribute proportionally more.
    mode : str
        'unmasked', 'masked', 'pathway_init', 'combined'.
    """

    def __init__(
        self,
        n_factors: int,
        a: float = 0.3,
        ap: float = 1.0,
        c: float = 0.3,
        cp: float = 1.0,
        sigma_v: float = 1.0,
        b_v: float = 1.0,
        v_prior: str = 'normal',
        sigma_gamma: float = 1.0,
        regression_weight: float = 1.0,
        use_class_weights: bool = True,
        use_intercept: bool = True,
        random_state: Optional[int] = None,
        mode: str = 'unmasked',
        pathway_mask: Optional[np.ndarray] = None,
        pathway_names: Optional[List[str]] = None,
        n_pathway_factors: Optional[int] = None,
        nnz_chunk_size: int = 1_000_000,
        **_ignored,
    ):
        self.K = n_factors
        self.a = a
        self.ap = ap
        self.c = c
        self.cp = cp
        self.sigma_v = sigma_v
        self.b_v = b_v
        self.v_prior = v_prior.lower()
        if self.v_prior not in ('normal', 'laplace'):
            raise ValueError(f"v_prior must be 'normal' or 'laplace', got '{v_prior}'")
        self.sigma_gamma = sigma_gamma
        self.regression_weight = regression_weight
        self.use_class_weights = use_class_weights
        # Cap on JJ auxiliary zeta.  Keeps lambda_JJ(zeta) >= lambda_JJ(zeta_max) > 0,
        # preventing the quadratic braking on theta from vanishing.
        # At zeta_max=4: lambda_min ~ 0.060.  The JJ bound remains valid for any zeta.
        self.zeta_max = 4.0

        self.use_intercept = use_intercept
        self.nnz_chunk_size = nnz_chunk_size
        self.mode = mode
        self.pathway_mask = pathway_mask
        self.pathway_names = pathway_names
        self.n_pathway_factors = n_pathway_factors

        if mode in ['masked', 'pathway_init'] and pathway_mask is None:
            raise ValueError(f"pathway_mask required for mode='{mode}'")
        if mode == 'combined':
            if pathway_mask is None or n_pathway_factors is None:
                raise ValueError("pathway_mask and n_pathway_factors required for combined mode")

        if random_state is not None:
            np.random.seed(random_state)
            self.seed_used_ = random_state
        else:
            self.seed_used_ = None

        # Will be set in fit()
        self.n = self.p = self.kappa = self.p_aux = None
        self.bp = self.dp = None

    # =================================================================
    # scHPF empirical hyperparameters
    # =================================================================

    @staticmethod
    def _mean_var_ratio(X, axis):
        """ap * mean / var along axis, as in scHPF."""
        if sp.issparse(X):
            sums = np.asarray(X.sum(axis=axis)).ravel()
        else:
            sums = np.asarray(X.sum(axis=axis)).ravel()
        return float(np.mean(sums) / max(np.var(sums), 1e-10))

    # =================================================================
    # Intercept helper
    # =================================================================

    def _prepend_intercept(self, X_aux, n=None):
        """Prepend a column of ones to X_aux if use_intercept is True."""
        if not self.use_intercept:
            return X_aux
        if X_aux is None or (hasattr(X_aux, 'size') and X_aux.size == 0):
            if n is None:
                raise ValueError("n required when X_aux is None with use_intercept=True")
            return np.ones((n, 1), dtype=np.float32)
        X_aux = np.asarray(X_aux, dtype=np.float32)
        ones = np.ones((X_aux.shape[0], 1), dtype=X_aux.dtype)
        return np.hstack([ones, X_aux])

    # =================================================================
    # Initialization (scHPF pattern)
    # =================================================================

    def _initialize(self, X, y, X_aux):
        """
        scHPF initialization:
        1. Empirical bp, dp (scalars)
        2. Random Gamma params: U(0.5*prior, 1.5*prior)
        3. xi.shape, eta.shape set to constants
        4. Transfer all arrays to device (GPU if available)
        """
        if sp.issparse(X):
            self.n, self.p = X.shape
        else:
            self.n, self.p = X.shape

        self.kappa = 1 if y.ndim == 1 else y.shape[1]
        self.p_aux = X_aux.shape[1] if X_aux is not None and X_aux.size > 0 else 0

        K = self.K

        # --- Empirical hyperparameters (scHPF: scalar bp, dp) ---
        self.bp = self.ap * self._mean_var_ratio(X, axis=1)
        self.dp = self.cp * self._mean_var_ratio(X, axis=0)
        # Clip if bp >> dp (scHPF)
        if self.bp > 1000 * self.dp:
            old_dp = self.dp
            self.dp = self.bp / 1000
            print(f"Clipping dp: was {old_dp:.4f} now {self.dp:.4f}")

        bp, dp = self.bp, self.dp

        # --- xi: Gamma(ap + K*a, b_xi) ---
        # shape is CONSTANT = ap + K*a
        self.a_xi = np.full(self.n, self.ap + K * self.a)
        self.b_xi = np.random.uniform(0.5 * bp, 1.5 * bp, self.n)

        # --- theta: Gamma(a_theta, b_theta) ---
        self.a_theta = np.random.uniform(0.5 * self.a, 1.5 * self.a,
                                         (self.n, K))
        self.b_theta = np.random.uniform(max(0.5 * bp, 0.5e-2), max(1.5 * bp, 1.5e-2), (self.n, K))

        # --- eta: Gamma(cp + K*c, b_eta) ---
        self.a_eta = np.full(self.p, self.cp + K * self.c)
        self.b_eta = np.random.uniform(0.5 * dp, 1.5 * dp, self.p)

        # --- beta: Gamma(a_beta, b_beta) ---
        self.a_beta = np.random.uniform(0.5 * self.c, 1.5 * self.c,
                                        (self.p, K))
        self.b_beta = np.random.uniform(0.5 * dp, 1.5 * dp, (self.p, K))

        # --- Apply pathway mask if needed ---
        self._init_beta_mask()

        # Boolean mask: True where beta_{jk} is an active latent variable.
        # Used to exclude masked entries from phi normalization, ELBO terms,
        # and Poisson rate computation.  Shape (p, K).
        if self.beta_mask is not None:
            self._active_beta = self.beta_mask > 0.5  # numpy bool
            self._n_active_beta = int(self._active_beta.sum())
        else:
            self._active_beta = None  # means all active
            self._n_active_beta = self.p * K

        # --- v: init small random ---
        self.mu_v = np.random.randn(self.kappa, K) * 0.01
        if self.v_prior == 'normal':
            # N(0, sigma_v^2/K) prior -- K-scaled so total logit theta*v has variance ~ sigma_v^2
            self.sigma_v_diag = np.full((self.kappa, K), (self.sigma_v ** 2) / K)
        else:
            # Laplace (Bayesian Lasso): scale b_v so the prior precision is
            # meaningful relative to the data precision.  The data contributes
            # O(N * lambda * E[theta^2]) to the v precision; the prior contributes
            # O(1/(b_v * |v|)).  Without scaling, b_v=1 is overwhelmed for large N
            # (prior < 1% of total precision for N > 100K).
            #
            # Following Park & Casella (2008), the optimal Lasso penalty scales as
            # sqrt(N).  We apply:  b_v_eff = b_v * sqrt(K) / sqrt(N)
            # so that b_v=1.0 gives sensible shrinkage regardless of dataset size.
            b_v_eff = self.b_v * np.sqrt(K) / np.sqrt(self.n)
            print(f"  [Laplace] b_v auto-scaled: {self.b_v:.4f} -> {b_v_eff:.6f} "
                  f"(N={self.n}, K={K})")
            self.b_v = b_v_eff
            # Init sigma_v_diag from Laplace variance = 2*b_v^2
            self.sigma_v_diag = np.full((self.kappa, K), 2.0 * self.b_v ** 2)

        # --- gamma: N(0, sigma_gamma^2) ---
        if self.p_aux > 0:
            self.mu_gamma = np.zeros((self.kappa, self.p_aux))
            self.Sigma_gamma = np.stack([
                np.eye(self.p_aux) * self.sigma_gamma ** 2
                for _ in range(self.kappa)
            ])
            # Initialize intercept column (col 0) to empirical log-odds
            if self.use_intercept:
                y_2d = y if y.ndim > 1 else y[:, None]
                for k in range(self.kappa):
                    n_pos = np.sum(y_2d[:, k] > 0.5)
                    n_neg = self.n - n_pos
                    if n_pos > 0 and n_neg > 0:
                        self.mu_gamma[k, 0] = np.log(n_pos / n_neg)
                        print(f"  intercept[{k}] init = {self.mu_gamma[k, 0]:.4f} "
                              f"(log-odds: {n_pos}/{n_neg})")
        else:
            self.mu_gamma = np.zeros((self.kappa, 0))
            self.Sigma_gamma = np.zeros((self.kappa, 0, 0))

        # --- JJ auxiliary ---
        self.zeta = np.ones((self.n, self.kappa))

        # --- Oscillation tracking for v update ---
        self._v_prev_mu = None
        self._v_raw_prev = None

        # --- Class weights for imbalanced labels ---
        y_2d = y if y.ndim > 1 else y[:, None]
        if self.use_class_weights:
            self._sample_weights = np.ones((self.n, self.kappa), dtype=np.float32)
            for k in range(self.kappa):
                n_pos = np.sum(y_2d[:, k] > 0.5)
                n_neg = self.n - n_pos
                if n_pos > 0 and n_neg > 0:
                    w_pos = self.n / (2.0 * n_pos)
                    w_neg = self.n / (2.0 * n_neg)
                    self._sample_weights[:, k] = np.where(
                        y_2d[:, k] > 0.5, w_pos, w_neg
                    )
        else:
            self._sample_weights = np.ones((self.n, self.kappa), dtype=np.float32)

        if self.use_class_weights:
            for k in range(self.kappa):
                n_pos = np.sum(y_2d[:, k] > 0.5)
                n_neg = self.n - n_pos
                w_pos = self.n / (2.0 * n_pos) if n_pos > 0 else 1.0
                w_neg = self.n / (2.0 * n_neg) if n_neg > 0 else 1.0
                print(f"  class_weights[{k}]: pos={n_pos} neg={n_neg} "
                      f"w_pos={w_pos:.3f} w_neg={w_neg:.3f}")

        # --- Store sparse structure for O(nnz*K) phi ---
        if sp.issparse(X):
            X_coo = X.tocoo()
        else:
            X_coo = sp.coo_matrix(X)
        row = X_coo.row.astype(np.int32)
        col = X_coo.col.astype(np.int32)
        data = X_coo.data.astype(np.float32)
        self._nnz = len(data)

        # Pre-sort by row for cache locality and segment_sum on GPU
        row_order = np.argsort(row, kind='mergesort')
        self._X_row = row[row_order]
        self._X_col = col[row_order]
        self._X_data = data[row_order]

        # Auto-tune chunk size based on K (target ~4GB work arrays)
        self._effective_chunk = _auto_chunk_size(self._nnz, K)
        n_chunks = (self._nnz + self._effective_chunk - 1) // self._effective_chunk
        print(f"  Chunk size: {self._effective_chunk:,} "
              f"({n_chunks} chunks for {self._nnz:,} nnz)")

        # Compute gammaln(data+1) sum in chunks to avoid a full-nnz GPU array
        # (saves ~3 GiB at float32 / ~6 GiB at float64 for 828M nnz).
        _gammaln_sum = 0.0
        _gl_chunk = 500_000
        for _gl_start in range(0, self._nnz, _gl_chunk):
            _gl_end = min(_gl_start + _gl_chunk, self._nnz)
            _gl_data = to_device(self._X_data[_gl_start:_gl_end])
            _gammaln_sum += float(xp.sum(gammaln(_gl_data + 1)))
            del _gl_data
        self._gammaln_data_sum = _gammaln_sum

        # Initialize E_theta/E_beta caches
        self._E_theta_cache = None
        self._E_beta_cache = None

        # Cache digamma/gammaln for constant-shape parameters (a_xi, a_eta)
        self._digamma_a_xi = digamma(self.a_xi)
        self._gammaln_a_xi = gammaln(self.a_xi)
        self._digamma_a_eta = digamma(self.a_eta)
        self._gammaln_a_eta = gammaln(self.a_eta)

        # ── Transfer all numpy arrays to device (GPU if available) ──
        self._to_device()

        # Adaptive row chunk size for (n, K) operations -- shared across
        # _update_zeta, _update_theta, _update_v, _compute_elbo.
        # Starts at auto-tuned value and shrinks on OOM.
        self._row_chunk = _row_chunk_size(self.n, self.K, n_intermediates=4)

        # Pre-compute digamma caches (now on device)
        self._refresh_log_caches()

        print(f"Initialized: n={self.n}, p={self.p}, K={K}, nnz={self._nnz}")
        print(f"  bp={bp:.4f}, dp={dp:.4f}")
        print(f"  E[theta] range: [{float(self.E_theta.min()):.4f}, {float(self.E_theta.max()):.4f}]")
        print(f"  E[beta] range: [{float(self.E_beta.min()):.4f}, {float(self.E_beta.max()):.4f}]")

    def _to_device(self):
        """Transfer numpy parameter arrays to JAX device (no-op without JAX).

        Sparse structure arrays (_X_row, _X_col, _X_data) are kept on CPU
        to save ~9 GiB of GPU memory.  They are transferred per-chunk
        during phi / ELBO computation (JAX handles this transparently).
        """
        if not USE_JAX:
            return
        _cpu_only = {'_X_row', '_X_col', '_X_data'}
        for name in list(vars(self)):
            if name in _cpu_only:
                continue
            val = getattr(self, name)
            if isinstance(val, np.ndarray):
                setattr(self, name, to_device(val))

    def _init_beta_mask(self):
        """Build beta_mask array for pathway modes.

        Modes
        -----
        masked : Hard constraint — beta priors are suppressed (near-zero)
            for gene-factor pairs outside the pathway mask.  The mask is
            re-enforced every iteration via ``_enforce_beta_mask``.
        pathway_init : Soft warm-start — beta shape params (a_beta) are
            boosted where the pathway mask is active, giving the model an
            informed starting point.  No mask is enforced during training,
            so beta is free to deviate from the pathway structure.
        combined : First ``n_pathway_factors`` factors are hard-constrained
            by the pathway mask (like masked); remaining factors are free
            (like unmasked) for de novo gene program discovery.
        """
        if self.mode == 'masked' and self.pathway_mask is not None:
            # pathway_mask: (n_pathways, n_genes) -> transpose to (n_genes, K)
            # pad or truncate to K columns
            pm = self.pathway_mask.T  # (p, n_pathways)
            if pm.shape[1] < self.K:
                pad = np.zeros((self.p, self.K - pm.shape[1]))
                self.beta_mask = np.hstack([pm, pad])
            else:
                self.beta_mask = pm[:, :self.K]
            # Zero out beta where mask is 0
            small_a = self.c * 0.01
            large_b = 100.0
            self.a_beta = np.where(self.beta_mask > 0.5, self.a_beta, small_a)
            self.b_beta = np.where(self.beta_mask > 0.5, self.b_beta, large_b)

        elif self.mode == 'pathway_init' and self.pathway_mask is not None:
            # Soft warm-start: boost a_beta where pathway is active so
            # E[beta] = a_beta / b_beta starts higher for pathway genes.
            # No mask is stored — beta evolves freely during training.
            pm = self.pathway_mask.T  # (p, n_pathways)
            if pm.shape[1] < self.K:
                pw_indicator = np.hstack([
                    pm, np.zeros((self.p, self.K - pm.shape[1]))
                ])
            else:
                pw_indicator = pm[:, :self.K]
            # Where pathway is active: multiply a_beta by a boost factor
            # so initial E[beta] is ~boost_factor× higher for pathway genes.
            boost = 5.0
            self.a_beta = np.where(pw_indicator > 0.5,
                                   self.a_beta * boost, self.a_beta)
            self.beta_mask = None  # no enforcement during training

        elif self.mode == 'combined' and self.pathway_mask is not None:
            pm = self.pathway_mask.T
            npath = self.n_pathway_factors
            self.beta_mask = np.ones((self.p, self.K))
            # First npath factors are pathway-constrained
            if pm.shape[1] >= npath:
                mask_part = pm[:, :npath]
            else:
                mask_part = np.hstack([pm, np.zeros((self.p, npath - pm.shape[1]))])
            self.beta_mask[:, :npath] = mask_part
            small_a = self.c * 0.01
            large_b = 100.0
            for k in range(npath):
                self.a_beta[:, k] = np.where(
                    self.beta_mask[:, k] > 0.5, self.a_beta[:, k], small_a)
                self.b_beta[:, k] = np.where(
                    self.beta_mask[:, k] > 0.5, self.b_beta[:, k], large_b)
        else:
            self.beta_mask = None

    def _enforce_beta_mask(self):
        """Re-enforce mask after beta update.

        Masked entries are treated as absent from the model: their a_beta
        and b_beta are pinned to small/large values so E[beta] ≈ 0.
        The boolean ``_active_beta`` (on device) controls which entries
        participate in phi normalization, ELBO terms, and rate sums.
        """
        if self._active_beta is None:
            return
        small_a = self.c * 0.01
        large_b = 100.0
        self.a_beta = xp.where(self._active_beta, self.a_beta, small_a)
        self.b_beta = xp.where(self._active_beta, self.b_beta, large_b)

    # =================================================================
    # Expected values (properties for convenience)
    # =================================================================

    @property
    def E_theta(self):
        if self._E_theta_cache is None:
            self._E_theta_cache = self.a_theta / self.b_theta
        return self._E_theta_cache

    @property
    def E_log_theta(self):
        return digamma(self.a_theta) - xp.log(self.b_theta)

    @property
    def E_beta(self):
        if self._E_beta_cache is None:
            self._E_beta_cache = self.a_beta / self.b_beta
        return self._E_beta_cache

    @property
    def E_log_beta(self):
        return digamma(self.a_beta) - xp.log(self.b_beta)

    def _invalidate_theta_cache(self):
        """Invalidate E_theta cache after a_theta or b_theta changes."""
        self._E_theta_cache = None

    def _invalidate_beta_cache(self):
        """Invalidate E_beta cache after a_beta or b_beta changes."""
        self._E_beta_cache = None

    def _refresh_log_caches(self):
        """Recompute cached digamma arrays after theta/beta updates.

        E_log_theta is NOT cached here: it is computed per-chunk inside
        _compute_phi_sparse to avoid a persistent (n, K) GPU array
        (~5 GiB for large n).  Only E_log_beta (p, K) is cached since
        p is typically small.

        Digamma values are recomputed on demand in _compute_elbo.

        For masked models, masked entries get E_log_beta = -inf so that
        they receive exactly zero responsibility in softmax (phi) and
        contribute zero to logsumexp in the Poisson likelihood.
        """
        _dig_beta = digamma(self.a_beta)
        self._E_log_beta_cache = _dig_beta - xp.log(self.b_beta)
        del _dig_beta
        # Masked entries: set to -inf so exp(-inf) = 0 in softmax/logsumexp
        if self._active_beta is not None:
            self._E_log_beta_cache = xp.where(
                self._active_beta, self._E_log_beta_cache,
                xp.asarray(-xp.inf, dtype=self._E_log_beta_cache.dtype)
            )

    @property
    def E_xi(self):
        return self.a_xi / self.b_xi

    @property
    def E_eta(self):
        return self.a_eta / self.b_eta

    # =================================================================
    # phi computation (sparse, O(nnz*K))
    # =================================================================

    def _compute_phi_sparse(self, random_init=False):
        """
        Compute Xphi using only nonzero entries, processed in chunks to
        bound peak memory at O(chunk_size * K) instead of O(nnz * K).

        Returns:
            z_sum_beta: (p, K) = Sum_i x_{ij} phi_{ijk}
            z_sum_theta: (n, K) = Sum_j x_{ij} phi_{ijk}
        """
        K = self.K

        if random_init:
            return self._random_init_z_sums()

        nnz = self._nnz
        base_chunk = self._effective_chunk
        adaptive_chunk = base_chunk
        # Never shrink below ~50MB work array (or 2k nnz), otherwise overhead
        # dominates and convergence becomes impractically slow.
        min_chunk = max(2_000, int(0.05 * (1024 ** 3) / (K * 4)))
        z_sum_beta = xp.zeros((self.p, K))
        z_sum_theta = xp.zeros((self.n, K))

        start = 0
        while start < nnz:
            end = min(start + adaptive_chunk, nnz)
            row_c = self._X_row[start:end]
            col_c = self._X_col[start:end]
            data_c = self._X_data[start:end]

            try:
                # Compute E_log_theta per chunk to avoid caching a full
                # (n, K) array on GPU (saves ~5 GiB at n=593K, K=1197).
                E_log_theta_rows = digamma(self.a_theta[row_c]) - xp.log(self.b_theta[row_c])
                Xphi = phi_chunk_core(
                    E_log_theta_rows,
                    self._E_log_beta_cache[col_c],
                    data_c,
                )
                del E_log_theta_rows

                z_sum_beta_new = scatter_add_to(z_sum_beta, col_c, Xphi,
                                                sorted_indices=False)
                z_sum_theta_new = scatter_add_to(z_sum_theta, row_c, Xphi,
                                                 sorted_indices=True)
                z_sum_beta = z_sum_beta_new
                z_sum_theta = z_sum_theta_new
                del Xphi

                start = end
                if adaptive_chunk < base_chunk:
                    adaptive_chunk = min(base_chunk, int(adaptive_chunk * 1.25))

            except Exception as exc:
                if _is_oom_error(exc) and adaptive_chunk > min_chunk:
                    new_chunk = max(min_chunk, adaptive_chunk // 2)
                    if new_chunk == adaptive_chunk:
                        raise
                    print(
                        f"  [OOM guard] shrinking phi chunk from {adaptive_chunk:,} "
                        f"to {new_chunk:,} at nnz [{start:,}:{end:,}]"
                    )
                    adaptive_chunk = new_chunk
                    continue
                raise

        return z_sum_beta, z_sum_theta

    def _random_init_z_sums(self):
        """Fast random initialization of z_sums using row/col sum Dirichlet.

        Instead of sampling a K-dimensional Dirichlet for every nonzero
        entry (O(nnz*K) Gamma draws -- hours for large datasets), we
        sample one Dirichlet per cell and per gene and scale by the
        respective row/col sums.  This is O((n+p)*K) and takes seconds.

        The per-entry Dirichlet init produces:
            z_sum_theta_{ik} = Sum_j x_{ij} * phi_{ijk}
        where phi_{ij} ~ Dir(1,...,1) independently.  In expectation,
        z_sum_theta_{ik} = row_sum_i / K.  Our approximation samples
        a single Dirichlet per cell and scales by row_sum, preserving
        the same mean and similar variance structure.
        """
        K = self.K
        data_np = to_numpy(self._X_data)
        row_np = to_numpy(self._X_row)
        col_np = to_numpy(self._X_col)
        row_sums = np.bincount(row_np, weights=data_np, minlength=self.n)
        col_sums = np.bincount(col_np, weights=data_np, minlength=self.p)

        # Dirichlet(1,...,1) = normalized Gamma(1,1) = normalized Exponential
        # Exponential is much faster to sample than Gamma for large K.
        z_theta_np = np.random.exponential(1.0, (self.n, K)).astype(np.float32)
        z_theta_np /= z_theta_np.sum(axis=1, keepdims=True)
        z_theta_np *= row_sums.astype(np.float32)[:, None]
        z_sum_theta = to_device(z_theta_np)
        del z_theta_np

        z_beta_np = np.random.exponential(1.0, (self.p, K)).astype(np.float32)
        z_beta_np /= z_beta_np.sum(axis=1, keepdims=True)
        z_beta_np *= col_sums.astype(np.float32)[:, None]
        z_sum_beta = to_device(z_beta_np)
        del z_beta_np

        return z_sum_beta, z_sum_theta

    # =================================================================
    # Theta-Beta Scale Balancing
    # =================================================================

    def _rescale_factors(self):
        """Rescale theta and beta per factor to maintain comparable scales.

        In Poisson factorization X ≈ theta @ beta.T, the product theta*beta
        is identifiable but the individual scales are not.  When beta is much
        larger than theta (common with many genes), the regression coupling
        logit = theta @ v requires v to compensate, causing v magnitude
        explosion.

        This method rescales each factor k so that mean(E[theta_k]) and
        mean(E[beta_k]) are at their geometric mean, and compensates v
        (and its variance) to keep logits invariant.

        For a Gamma(a, b) variational factor, E = a/b.  Scaling E by s
        while preserving the shape a is done by dividing b by s.
        """
        for k in range(self.K):
            # Compute mean E[theta_k] in chunks to avoid full (n,K) materialization
            theta_sum = 0.0
            for i0 in range(0, self.n, self._row_chunk):
                i1 = min(i0 + self._row_chunk, self.n)
                theta_sum += float(
                    (self.a_theta[i0:i1, k] / self.b_theta[i0:i1, k]).sum()
                )
            mean_theta_k = theta_sum / self.n

            mean_beta_k = float(
                (self.a_beta[:, k] / self.b_beta[:, k]).mean()
            )

            if mean_theta_k < 1e-30 or mean_beta_k < 1e-30:
                continue

            target = np.sqrt(mean_theta_k * mean_beta_k)
            s_theta = target / mean_theta_k   # > 1 when theta is small
            s_beta = target / mean_beta_k     # > 1 when beta is small

            # Limit per-step rescaling to avoid destabilizing other updates
            s_theta = np.clip(s_theta, 0.5, 2.0)
            s_beta = 1.0 / s_theta  # Ensure product theta*beta unchanged

            # Rescale theta: E[theta_k] *= s_theta  (divide b by s_theta)
            self.b_theta = self.b_theta.at[:, k].set(self.b_theta[:, k] / s_theta)
            # Rescale beta: E[beta_k] *= s_beta  (divide b by s_beta)
            self.b_beta = self.b_beta.at[:, k].set(self.b_beta[:, k] / s_beta)

            # Compensate v to keep logit = theta @ v invariant:
            # new_theta = s_theta * old_theta, so new_v = old_v / s_theta
            self.mu_v = self.mu_v.at[:, k].set(self.mu_v[:, k] / s_theta)
            self.sigma_v_diag = self.sigma_v_diag.at[:, k].set(self.sigma_v_diag[:, k] / (s_theta ** 2))

        # Invalidate caches after modifying b_theta, b_beta
        self._invalidate_theta_cache()
        self._invalidate_beta_cache()

    # =================================================================
    # CAVI Updates
    # =================================================================

    def _update_beta(self, z_sum_beta):
        """beta shape and rate (scHPF Eq 8, gene side).

        For masked models, only active entries are updated.  Masked entries
        remain pinned at their suppressed values — we never compute the
        standard Gamma update for them and then overwrite, because that
        post-hoc projection is not a valid coordinate-ascent step.
        """
        # Compute theta_sum in chunks to avoid materializing E_theta cache
        # when z_sum_theta may still be alive (saves one (n, K) array).
        theta_sum = xp.zeros(self.K)
        for i0 in range(0, self.n, self._row_chunk):
            i1 = min(i0 + self._row_chunk, self.n)
            theta_sum = theta_sum + (self.a_theta[i0:i1] / self.b_theta[i0:i1]).sum(axis=0)

        new_a = self.c + z_sum_beta
        new_b = self.E_eta[:, None] + theta_sum[None, :]
        # Floor: prevent pathologically small shape/rate values
        new_a = xp.maximum(new_a, 1e-6)
        new_b = xp.maximum(new_b, 1e-8)

        if self._active_beta is not None:
            # Only update active entries; masked entries keep their pinned values.
            self.a_beta = xp.where(self._active_beta, new_a, self.a_beta)
            self.b_beta = xp.where(self._active_beta, new_b, self.b_beta)
        else:
            self.a_beta = new_a
            self.b_beta = new_b
        self._invalidate_beta_cache()

    def _update_eta(self):
        """eta rate (scHPF): b^eta_j = d' + Sum_k E[beta_{jk}].

        Only active beta entries contribute (masked entries are absent).
        """
        # a^eta is constant = cp + K*c (set in init, never changes)
        if self._active_beta is not None:
            self.b_eta = self.dp + xp.where(
                self._active_beta, self.E_beta, 0.0).sum(axis=1)
        else:
            self.b_eta = self.dp + self.E_beta.sum(axis=1)

    def _update_theta(self, z_sum_theta, y, X_aux):
        """
        theta shape and rate (scHPF Eq 7, cell side + JJ regression).

        Solves the quadratic for b_theta in one step:
            b^2 - b_base*b - c_quad*a_theta = 0
            b = (b_base + sqrt(b_base^2 + 4*c_quad*a_theta)) / 2
        which is always positive (given c_quad > 0, guaranteed by the zeta cap).

        Row-chunked when regression is active to avoid multiple full (n, K)
        intermediates (R_linear, R_quad, b_base, disc) living simultaneously.
        """
        # a^theta_{ik} = a + Sum_j x_{ij} phi_{ijk}
        self.a_theta = self.a + z_sum_theta

        # b_base = E[xi_i] + Sum_j E[beta_{jk}] + regression_weight * R_linear
        # Active-only beta sums (masked entries are absent from the model)
        if self._active_beta is not None:
            beta_sum = xp.where(self._active_beta, self.E_beta, 0.0).sum(axis=0)
        else:
            beta_sum = self.E_beta.sum(axis=0)  # (K,)

        if self.regression_weight > 0:
            # Pre-compute tiny (kappa, K) quantities for _regression_rate_parts
            y_exp = y if y.ndim > 1 else y[:, None]        # (n, kappa)
            lam = lambda_jj(self.zeta)                       # (n, kappa)
            W = self._sample_weights                         # (n, kappa)
            # NOTE: do NOT use self.E_theta here — it materializes the full
            # (n, K) cache.  Row slices are computed on-the-fly below.
            E_v = self.mu_v                                  # (kappa, K)
            E_v_sq = E_v ** 2 + self.sigma_v_diag            # (kappa, K)
            E_v_sq_col = xp.square(E_v)                      # (kappa, K)
            rw = self.regression_weight

            min_chunk = max(1024, self.n // 256)
            b_theta_chunks = []
            i0 = 0
            while i0 < self.n:
                i1 = min(i0 + self._row_chunk, self.n)
                try:
                    # --- inline _regression_rate_parts for this chunk ---
                    E_theta_c = self.a_theta[i0:i1] / self.b_theta[i0:i1]
                    W_c = W[i0:i1]
                    W_lam_c = W_c * lam[i0:i1]
                    theta_v_c = E_theta_c @ E_v.T                             # (chunk, kappa)
                    if self.p_aux > 0:
                        theta_v_c = theta_v_c + X_aux[i0:i1] @ self.mu_gamma.T

                    R_lin_c = -(W_c * (y_exp[i0:i1] - 0.5)) @ E_v            # (chunk, K)
                    R_lin_c = R_lin_c + 2.0 * ((W_lam_c * theta_v_c) @ E_v)
                    R_lin_c = R_lin_c - 2.0 * E_theta_c * (W_lam_c @ E_v_sq_col)
                    R_quad_c = (2.0 * W_lam_c) @ E_v_sq                       # (chunk, K)

                    # --- solve quadratic for b_theta ---
                    b_poisson_c = self.E_xi[i0:i1, None] + beta_sum[None, :]  # (chunk, K)
                    b_base_c = b_poisson_c + rw * R_lin_c
                    c_quad_c = rw * R_quad_c
                    disc_c = xp.sqrt(xp.square(b_base_c) + 4.0 * c_quad_c * self.a_theta[i0:i1])
                    b_theta_c = (b_base_c + disc_c) / 2.0

                    # Floor at 10% of b_poisson
                    b_theta_c = xp.maximum(b_theta_c, 0.1 * b_poisson_c)
                    # Floor at bp
                    b_theta_c = xp.maximum(b_theta_c, self.bp)
                    # Absolute floor: in combined mode, beta_sum can be tiny
                    # for pathway factors (few active genes), making b_poisson
                    # and bp both near-zero.  Without this floor, b_theta
                    # collapses → E[theta] explodes → destabilizes everything.
                    b_theta_c = xp.maximum(b_theta_c, 1e-2)
                    b_theta_chunks.append(b_theta_c)
                    i0 = i1
                except Exception as exc:
                    if _is_oom_error(exc) and self._row_chunk > min_chunk:
                        self._row_chunk = max(min_chunk, self._row_chunk // 2)
                        print(f"  [OOM guard] row chunk → {self._row_chunk:,}")
                        continue
                    raise

            self.b_theta = xp.concatenate(b_theta_chunks, axis=0) if len(b_theta_chunks) > 1 else b_theta_chunks[0]
            self._theta_inner_iters = 1

            # Re-tighten zeta for updated theta
            self._invalidate_theta_cache()
            self._update_zeta(X_aux)
        else:
            self.b_theta = self.E_xi[:, None] + beta_sum[None, :]
            self.b_theta = xp.maximum(self.b_theta, 1e-2)
            self._theta_inner_iters = 0
        self._invalidate_theta_cache()

    def _update_xi(self):
        """xi rate (scHPF): b^xi_i = b' + Sum_k E[theta_{ik}]."""
        # a^xi is constant = ap + K*a (set in init)
        # Compute E[theta].sum(axis=1) in chunks to avoid caching full (n,K).
        theta_row_sum_chunks = []
        for i0 in range(0, self.n, self._row_chunk):
            i1 = min(i0 + self._row_chunk, self.n)
            theta_row_sum_chunks.append(
                (self.a_theta[i0:i1] / self.b_theta[i0:i1]).sum(axis=1)
            )
        theta_row_sum = xp.concatenate(theta_row_sum_chunks) if len(theta_row_sum_chunks) > 1 else theta_row_sum_chunks[0]
        self.b_xi = self.bp + theta_row_sum

    def _update_zeta(self, X_aux):
        """JJ auxiliary: zeta_{ik} = min(sqrt(E[A^2_{ik}]), zeta_max).

        Capping zeta keeps lambda_JJ(zeta) bounded from below, which prevents the
        quadratic braking on theta from vanishing.  The JJ lower bound is valid
        for ANY zeta, so capping gives a slightly looser but still valid bound.
        Each CAVI update still provably increases this capped-zeta ELBO.

        Row-chunked with adaptive OOM guard.
        """
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K) -- tiny
        min_chunk = max(1024, self.n // 256)

        zeta_chunks = []
        i0 = 0
        while i0 < self.n:
            i1 = min(i0 + self._row_chunk, self.n)
            try:
                b_theta_c = self.b_theta[i0:i1]
                E_theta_c = self.a_theta[i0:i1] / b_theta_c
                Var_theta_c = E_theta_c / b_theta_c      # a/(b^2) = (a/b)/b
                E_A_c = E_theta_c @ self.mu_v.T
                if self.p_aux > 0:
                    E_A_c = E_A_c + X_aux[i0:i1] @ self.mu_gamma.T
                E_A_sq_c = xp.square(E_A_c) + Var_theta_c @ E_v_sq.T
                zeta_chunks.append(
                    xp.minimum(xp.sqrt(xp.maximum(E_A_sq_c, 1e-8)), self.zeta_max)
                )
                i0 = i1
            except Exception as exc:
                if _is_oom_error(exc) and self._row_chunk > min_chunk:
                    self._row_chunk = max(min_chunk, self._row_chunk // 2)
                    print(f"  [OOM guard] row chunk → {self._row_chunk:,}")
                    continue
                raise
        self.zeta = xp.concatenate(zeta_chunks, axis=0) if len(zeta_chunks) > 1 else zeta_chunks[0]

    def _regression_rate_parts(self, y, X_aux):
        """
        Compute the JJ regression correction to theta rate, split into linear
        and quadratic parts.

        The full correction is R = R_linear + R_quad_coeff * E[theta], where:
          R_linear_{il} = Sum_k [-(y_{ik} - 0.5) v_{kl}
                                + 2*lambda(zeta_{ik}) v_{kl} C^{(-l)}_{ik}]
          R_quad_coeff_{il} = Sum_k [2*lambda(zeta_{ik}) E[v^2_{kl}]]

        Memory-efficient: avoids (n, kappa, K) C_minus intermediate by
        algebraic decomposition into (n, kappa) @ (kappa, K) matmuls.

        Returns
        -------
        R_linear : (n, K) -- terms not depending on E[theta_{il}]
        R_quad_coeff : (n, K) -- coefficient of E[theta_{il}], always >= 0
        """
        y_exp = y if y.ndim > 1 else y[:, None]  # (n, kappa)
        lam = lambda_jj(self.zeta)                 # (n, kappa)
        W = self._sample_weights                    # (n, kappa)
        E_theta = self.E_theta                      # (n, K)
        E_v = self.mu_v                             # (kappa, K)
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag # (kappa, K)

        # Full linear predictor: (n, kappa) -- BLAS
        theta_v = E_theta @ E_v.T
        if self.p_aux > 0:
            theta_v = theta_v + X_aux @ self.mu_gamma.T

        W_lam = W * lam  # (n, kappa)

        # BLAS matmuls to avoid (n, kappa, K) C_minus intermediate:
        E_v_sq_col = xp.square(E_v)                             # (kappa, K) -- tiny
        R_linear = -(W * (y_exp - 0.5)) @ E_v                   # (n, K)
        R_linear = R_linear + 2.0 * ((W_lam * theta_v) @ E_v)   # fuse add
        R_linear = R_linear - 2.0 * E_theta * (W_lam @ E_v_sq_col)  # fuse subtract

        # R_quad via single BLAS matmul
        R_quad_coeff = (2.0 * W_lam) @ E_v_sq                   # (n, K)

        return R_linear, R_quad_coeff

    def _update_v(self, y, X_aux, iteration=0):
        """
        v posterior (diagonal Gaussian, JJ bound).

        For v_prior='normal':
            precision_{kl} = 1/sigma^2_eff + 2 Sum_i lambda(zeta_{ik}) E[theta^2_{il}]
            sigma^2_eff = sigma^2_v / K

        For v_prior='laplace' (Bayesian Lasso):
            precision_{kl} = E_q[1/s_{kl}] + 2 Sum_i lambda(zeta_{ik}) E[theta^2_{il}]
            where E_q[1/s_{kl}] is computed from the inverse Gaussian
            variational posterior on the scale mixture variable s_{kl}.

        Memory-efficient: row-chunked to avoid full (n, K) temporaries
        (E_theta_sq, Var_theta) while accumulating (kappa, K) sufficient
        statistics via BLAS matmuls.
        """
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = lambda_jj(self.zeta)
        W = self._sample_weights                    # (n, kappa)
        E_v = self.mu_v

        if self.v_prior == 'normal':
            sigma_v_eff_sq = (self.sigma_v ** 2) / self.K
            prior_precision = 1.0 / sigma_v_eff_sq
        else:
            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K)
            omega = xp.sqrt(xp.maximum(E_v_sq, 1e-12))
            # Laplace (Bayesian Lasso) prior precision from the inverse-
            # Gaussian scale mixture.  For large |v| (omega >> 1) this
            # vanishes as 1/(b_v*omega), removing all regularization and
            # letting v explode.  Floor at 1.0 to ensure sigma2_v never
            # exceeds 1.0 per component, providing meaningful shrinkage
            # comparable to a unit-variance Gaussian prior.
            prior_precision_raw = 1.0 / (self.b_v * omega) + 1.0 / (omega ** 2)
            prior_precision = xp.maximum(prior_precision_raw, 1.0)

        # Accumulate (kappa, K) sufficient statistics in chunks to avoid
        # materializing full (n, K) intermediates (Var_theta, E_theta_sq).
        # Uses self._row_chunk with adaptive OOM guard.
        min_chunk = max(1024, self.n // 256)
        prec_sum = xp.zeros((self.kappa, self.K))   # W_lam.T @ E_theta_sq
        term1_sum = xp.zeros((self.kappa, self.K))   # W_y.T @ E_theta
        parta_sum = xp.zeros((self.kappa, self.K))   # (W_lam * theta_v).T @ E_theta
        partb_sum = xp.zeros((self.kappa, self.K))   # W_lam.T @ E_theta_plain_sq

        i0 = 0
        while i0 < self.n:
            i1 = min(i0 + self._row_chunk, self.n)
            try:
                b_theta_c = self.b_theta[i0:i1]
                E_theta_c = self.a_theta[i0:i1] / b_theta_c            # (chunk, K)
                E_theta_psq_c = xp.square(E_theta_c)                   # (chunk, K)
                Var_theta_c = E_theta_c / b_theta_c                    # a/b^2 = (a/b)/b
                E_theta_sq_c = E_theta_psq_c + Var_theta_c             # (chunk, K)

                theta_v_c = E_theta_c @ E_v.T                          # (chunk, kappa)
                if self.p_aux > 0:
                    theta_v_c = theta_v_c + X_aux[i0:i1] @ self.mu_gamma.T

                W_lam_c = W[i0:i1] * lam[i0:i1]                       # (chunk, kappa)
                W_y_c = W[i0:i1] * (y_exp[i0:i1] - 0.5)              # (chunk, kappa)

                prec_sum = prec_sum + W_lam_c.T @ E_theta_sq_c
                term1_sum = term1_sum + W_y_c.T @ E_theta_c
                parta_sum = parta_sum + (W_lam_c * theta_v_c).T @ E_theta_c
                partb_sum = partb_sum + W_lam_c.T @ E_theta_psq_c
                i0 = i1
            except Exception as exc:
                if _is_oom_error(exc) and self._row_chunk > min_chunk:
                    self._row_chunk = max(min_chunk, self._row_chunk // 2)
                    print(f"  [OOM guard] row chunk → {self._row_chunk:,}")
                    continue
                raise

        precision = prior_precision + 2 * prec_sum                  # (kappa, K)
        term1 = term1_sum
        term2 = 2.0 * (parta_sum - E_v * partb_sum)
        mean_prec = term1 - term2

        sigma_v_diag_new = 1.0 / precision

        if self.v_prior == 'normal':
            # Clip v per-element
            v_clip = min(5.0, 10.0 / np.sqrt(self.K))
            mu_v_new = xp.clip(mean_prec / precision, -v_clip, v_clip)

            # Adaptive damping — scale with K so that the per-iteration logit
            # change ≈ K * alpha * |v| stays bounded regardless of K.
            # K=50 → 0.15 (unchanged); K=348 → ~0.043.
            alpha_max = min(0.15, 7.5 / self.K)
            alpha = min(alpha_max, 0.05 + (alpha_max - 0.05) * (iteration / max(200, iteration)))
            mu_v_candidate = (1.0 - alpha) * self.mu_v + alpha * mu_v_new

            # --- Period-2 oscillation detection and correction ---
            if self._v_prev_mu is None:
                self._v_prev_mu = xp.zeros_like(self.mu_v)
                self._v_raw_prev = xp.zeros_like(self.mu_v)

            sign_flip_now = (self.mu_v * mu_v_candidate) < 0
            sign_flip_prev = (self._v_prev_mu * self.mu_v) < 0
            oscillating = sign_flip_now & sign_flip_prev

            avg_raw = 0.5 * (mu_v_new + self._v_raw_prev)
            corrected = (1.0 - alpha) * self.mu_v + alpha * avg_raw
            mu_v_candidate = xp.where(oscillating, corrected, mu_v_candidate)

            self._v_prev_mu = xp.array(self.mu_v)
            self._v_raw_prev = xp.array(mu_v_new)
            self.mu_v = xp.clip(mu_v_candidate, -v_clip, v_clip)
            # Use exact posterior variance — damping sigma creates a one-way
            # ratchet that exponentially collapses variance over iterations
            # because the data precision only grows as theta scales up.
            self.sigma_v_diag = sigma_v_diag_new
        else:
            # Laplace: the adaptive E[1/s] shrinkage already regularises v,
            # so the clip can be much looser than for the normal prior.
            # The normal branch uses 10/sqrt(K) assuming all K factors are
            # active; Laplace drives most weights to ~0, so only a handful
            # are large and the effective sum is << K.
            # However, in combined mode the pathway factors are mask-
            # constrained (not sparsified), keeping effective K ≈ full K.
            # Use K-dependent bounds so that the total logit change per
            # iteration (≈ K * max_step) stays bounded regardless of K.
            v_clip = min(5.0, 10.0 / np.sqrt(self.K))
            mu_v_new = xp.clip(mean_prec / precision, -v_clip, v_clip)

            # Damping: K-scale alpha_max so combined mode (K=403) doesn't
            # diverge while small-K models still converge quickly.
            # K=50 → 0.06; K=348 → 0.009; K=403 → 0.007.
            alpha_max = min(0.06, 3.0 / self.K)
            alpha = min(alpha_max, 0.05 * alpha_max / 0.06 + (alpha_max - 0.05 * alpha_max / 0.06) * (iteration / max(200, iteration)))
            mu_v_candidate = (1.0 - alpha) * self.mu_v + alpha * mu_v_new

            # Per-element step cap: scale with K so total logit change
            # (K * max_step) stays bounded at ~2.5 regardless of K.
            # K=50 → 0.05; K=348 → 0.0072; K=403 → 0.0062.
            max_step = min(0.05, 2.5 / self.K)
            delta = mu_v_candidate - self.mu_v
            delta = xp.clip(delta, -max_step, max_step)
            mu_v_candidate = self.mu_v + delta

            # Period-2 oscillation detection and correction
            if self._v_prev_mu is None:
                self._v_prev_mu = xp.zeros_like(self.mu_v)
                self._v_raw_prev = xp.zeros_like(self.mu_v)

            # Sign-flip detection (original)
            sign_flip_now = (self.mu_v * mu_v_candidate) < 0
            sign_flip_prev = (self._v_prev_mu * self.mu_v) < 0
            oscillating = sign_flip_now & sign_flip_prev

            # Period-2 cycle detection: if candidate ≈ v_{t-1} the
            # system is bouncing between two attractors.  Average the
            # raw proposals to escape the cycle.
            period2 = xp.abs(mu_v_candidate - self._v_prev_mu) < 0.05 * (xp.abs(self.mu_v) + 1e-6)
            oscillating = oscillating | period2

            avg_raw = 0.5 * (mu_v_new + self._v_raw_prev)
            corrected = (1.0 - alpha) * self.mu_v + alpha * avg_raw
            mu_v_candidate = xp.where(oscillating, corrected, mu_v_candidate)

            self._v_prev_mu = xp.array(self.mu_v)
            self._v_raw_prev = xp.array(mu_v_new)
            self.mu_v = xp.clip(mu_v_candidate, -v_clip, v_clip)

            # Floor sigma_v_diag to prevent posterior variance collapse.
            # With N~7000 cells the data precision can drive sigma→1e-15,
            # making the posterior degenerate and mu updates erratic.
            sigma_v_diag_new = xp.maximum(sigma_v_diag_new, 1e-4)
            # Laplace prior precision depends on mu_v through omega, so
            # sigma and mu must move in lockstep.  Damping both at the
            # same rate keeps q(v) coherent.
            self.sigma_v_diag = (1.0 - alpha) * self.sigma_v_diag + alpha * sigma_v_diag_new

    def _update_gamma(self, y, X_aux, iteration=0):
        """gamma posterior (Gaussian, JJ bound)."""
        if self.p_aux == 0:
            return
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = lambda_jj(self.zeta)
        W = self._sample_weights                    # (n, kappa)

        # Pre-compute theta @ mu_v.T in chunks to avoid caching full E_theta
        theta_v = xp.zeros((self.n, self.kappa))
        for i0 in range(0, self.n, self._row_chunk):
            i1 = min(i0 + self._row_chunk, self.n)
            E_theta_c = self.a_theta[i0:i1] / self.b_theta[i0:i1]
            if USE_JAX:
                theta_v = theta_v.at[i0:i1].set(E_theta_c @ self.mu_v.T)
            else:
                theta_v[i0:i1] = E_theta_c @ self.mu_v.T

        # Match v damping rate to keep the coupled system stable.
        alpha_max = min(0.06, 3.0 / self.K)
        alpha = min(alpha_max, 0.05 * alpha_max / 0.06 + (alpha_max - 0.05 * alpha_max / 0.06) * (iteration / max(200, iteration)))

        for k in range(self.kappa):
            prec_prior = xp.eye(self.p_aux) / (self.sigma_gamma ** 2)
            # Precision: 1/sigma^2 I + 2 Sum_i W_{ik} lambda(zeta_{ik}) x^aux_i x^aux_i^T
            W_lam_k = W[:, k] * lam[:, k]  # (n,) -- class-weighted lambda
            weighted_X = X_aux * (2 * W_lam_k)[:, None]
            prec_lik = weighted_X.T @ X_aux
            prec = prec_prior + prec_lik

            # Residual: W*(y - 0.5) - 2*W*lambda(zeta) theta*v
            theta_v_k = theta_v[:, k]
            residual = W[:, k] * (y_exp[:, k] - 0.5) - 2 * W_lam_k * theta_v_k
            mean_prec = X_aux.T @ residual

            if USE_JAX:
                self.Sigma_gamma = self.Sigma_gamma.at[k].set(xp.linalg.inv(prec))
            else:
                self.Sigma_gamma[k] = np.linalg.inv(prec)
            mu_gamma_new = self.Sigma_gamma[k] @ mean_prec
            # Clip gamma to prevent explosion (K-scaled like v_clip)
            gamma_clip = min(5.0, 10.0 / np.sqrt(self.K))
            mu_gamma_new = xp.clip(mu_gamma_new, -gamma_clip, gamma_clip)
            new_mu_k = (1.0 - alpha) * self.mu_gamma[k] + alpha * mu_gamma_new
            # Per-element step cap (same K-scaling as v update)
            gamma_step = min(0.05, 2.5 / self.K)
            delta_k = new_mu_k - self.mu_gamma[k]
            delta_k = xp.clip(delta_k, -gamma_step, gamma_step)
            new_mu_k = self.mu_gamma[k] + delta_k
            if USE_JAX:
                self.mu_gamma = self.mu_gamma.at[k].set(new_mu_k)
            else:
                self.mu_gamma[k] = new_mu_k

    # =================================================================
    # ELBO
    # =================================================================

    def _compute_elbo(self, X_dense, y, X_aux):
        """Compute ELBO = E[log p] - E[log q].

        Row-chunked for all theta-related (n, K) terms to avoid
        materializing multiple full (n, K) temporaries simultaneously.
        """
        E_beta = self.E_beta
        E_log_beta = self._E_log_beta_cache
        E_xi = self.E_xi
        E_eta = self.E_eta
        # Reuse cached digamma for constant-shape a_xi, a_eta
        E_log_xi = self._digamma_a_xi - xp.log(self.b_xi)
        E_log_eta = self._digamma_a_eta - xp.log(self.b_eta)

        elbo = 0.0

        # === Poisson likelihood (collapsed z) -- chunked to avoid (nnz, K) ===
        # E_log_theta is needed for random-access by sparse row indices,
        # so we compute it in row chunks and concatenate, then delete after use.
        min_chunk = max(1024, self.n // 256)
        elt_chunks = []
        i0 = 0
        while i0 < self.n:
            i1 = min(i0 + self._row_chunk, self.n)
            try:
                elt_chunks.append(
                    digamma(self.a_theta[i0:i1]) - xp.log(self.b_theta[i0:i1])
                )
                i0 = i1
            except Exception as exc:
                if _is_oom_error(exc) and self._row_chunk > min_chunk:
                    self._row_chunk = max(min_chunk, self._row_chunk // 2)
                    print(f"  [OOM guard] row chunk → {self._row_chunk:,}")
                    continue
                raise
        E_log_theta = xp.concatenate(elt_chunks, axis=0) if len(elt_chunks) > 1 else elt_chunks[0]
        del elt_chunks

        poisson_ll = 0.0
        nnz = self._nnz
        chunk = self._effective_chunk
        for start in range(0, nnz, chunk):
            end = min(start + chunk, nnz)
            row_c = self._X_row[start:end]
            col_c = self._X_col[start:end]
            data_c = self._X_data[start:end]
            log_rates_c = E_log_theta[row_c] + E_log_beta[col_c]  # (chunk, K)
            log_sum_c = logsumexp_rows(log_rates_c).ravel()
            poisson_ll += xp.dot(data_c, log_sum_c)
            del log_rates_c

        # Compute E[theta].sum(axis=0) in chunks to avoid (n,K) cache
        theta_col_sum = xp.zeros(self.K)
        for i0 in range(0, self.n, self._row_chunk):
            i1 = min(i0 + self._row_chunk, self.n)
            theta_col_sum = theta_col_sum + (self.a_theta[i0:i1] / self.b_theta[i0:i1]).sum(axis=0)
        # Use only active beta entries in the rate term (masked = absent = 0)
        if self._active_beta is not None:
            beta_col_sum = xp.where(self._active_beta, E_beta, 0.0).sum(axis=0)
        else:
            beta_col_sum = E_beta.sum(axis=0)
        poisson_ll -= xp.sum(theta_col_sum * beta_col_sum)
        poisson_ll -= self._gammaln_data_sum  # cached at init
        elbo += poisson_ll

        # === Row-chunked theta terms: JJ regression LL, prior, entropy ===
        y_exp = y if y.ndim > 1 else y[:, None]
        lam = lambda_jj(self.zeta)
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag   # tiny (kappa, K)
        W = self._sample_weights  # (n, kappa)

        regression_ll = 0.0
        theta_prior = 0.0
        theta_entropy = 0.0
        gammaln_a = gammaln(self.a)

        i0 = 0
        while i0 < self.n:
            i1 = min(i0 + self._row_chunk, self.n)
            try:
                a_theta_c = self.a_theta[i0:i1]
                b_theta_c = self.b_theta[i0:i1]
                E_theta_c = a_theta_c / b_theta_c
                E_log_theta_c = E_log_theta[i0:i1]

                # --- JJ regression likelihood ---
                Var_theta_c = E_theta_c / b_theta_c                  # a/b^2 = (a/b)/b
                E_A_c = E_theta_c @ self.mu_v.T                      # (chunk, kappa)
                if self.p_aux > 0:
                    E_A_c = E_A_c + X_aux[i0:i1] @ self.mu_gamma.T
                E_A_sq_c = E_A_c ** 2 + Var_theta_c @ E_v_sq.T       # (chunk, kappa)
                lam_c = lam[i0:i1]
                W_c = W[i0:i1]
                zeta_c = self.zeta[i0:i1]
                regression_ll += xp.sum(
                    W_c * ((y_exp[i0:i1] - 0.5) * E_A_c - lam_c * E_A_sq_c)
                )
                regression_ll += xp.sum(
                    W_c * (lam_c * zeta_c ** 2 - 0.5 * zeta_c + log_expit(zeta_c))
                )

                # --- Prior p(theta|xi) ---
                theta_prior += xp.sum(
                    (self.a - 1) * E_log_theta_c
                    + self.a * E_log_xi[i0:i1, None]
                    - E_xi[i0:i1, None] * E_theta_c
                )

                # --- Entropy -E[log q(theta)] ---
                psi_a_c = digamma(a_theta_c)
                theta_entropy += xp.sum(
                    a_theta_c - xp.log(b_theta_c)
                    + gammaln(a_theta_c)
                    + (1 - a_theta_c) * psi_a_c
                )
                i0 = i1
            except Exception as exc:
                if _is_oom_error(exc) and self._row_chunk > min_chunk:
                    self._row_chunk = max(min_chunk, self._row_chunk // 2)
                    print(f"  [OOM guard] row chunk → {self._row_chunk:,}")
                    continue
                raise

        del E_log_theta  # free the full (n, K) array

        elbo += self.regression_weight * regression_ll
        elbo += theta_prior
        elbo -= self.n * self.K * gammaln_a
        elbo += theta_entropy

        # === Prior: p(beta|eta) — active entries only ===
        # Masked entries are absent from the model, so they contribute
        # nothing to the prior or its normalization constant.
        _beta_prior_terms = ((self.c - 1) * E_log_beta
                             + self.c * E_log_eta[:, None]
                             - E_eta[:, None] * E_beta)
        if self._active_beta is not None:
            elbo += xp.sum(xp.where(self._active_beta, _beta_prior_terms, 0.0))
        else:
            elbo += xp.sum(_beta_prior_terms)
        del _beta_prior_terms
        elbo -= self._n_active_beta * gammaln(self.c)

        # === Prior: p(xi) ===
        elbo += xp.sum((self.ap - 1) * E_log_xi
                       + self.ap * xp.log(self.bp) - self.bp * E_xi)
        elbo -= self.n * gammaln(self.ap)

        # === Prior: p(eta) ===
        elbo += xp.sum((self.cp - 1) * E_log_eta
                       + self.cp * xp.log(self.dp) - self.dp * E_eta)
        elbo -= self.p * gammaln(self.cp)

        # === Prior: p(v) ===
        if self.v_prior == 'normal':
            sigma_v_eff_sq = (self.sigma_v ** 2) / self.K
            elbo -= 0.5 * xp.sum(self.mu_v ** 2 + self.sigma_v_diag) / sigma_v_eff_sq
            elbo -= 0.5 * self.kappa * self.K * xp.log(2 * xp.pi * sigma_v_eff_sq)
        else:
            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag  # (kappa, K)
            omega = xp.sqrt(xp.maximum(E_v_sq, 1e-12))
            E_inv_s = 1.0 / (self.b_v * omega) + 1.0 / (omega ** 2)

            mu_s = self.b_v / omega
            E_log_s = xp.log(xp.maximum(mu_s, 1e-12))
            elbo += xp.sum(-0.5 * xp.log(2 * xp.pi) - 0.5 * E_log_s
                           - 0.5 * E_v_sq * E_inv_s)

            rate_s = 1.0 / (2.0 * self.b_v ** 2)
            elbo += xp.sum(xp.log(rate_s) - mu_s * rate_s)

            lambda_s = 1.0 / (self.b_v ** 2)
            elbo += 0.5 * xp.sum(xp.log(2 * xp.pi * xp.e * mu_s ** 3
                                        / xp.maximum(lambda_s, 1e-12)))

        # === Entropy: -E[log q] ===
        # q(beta) — active entries only (masked entries are not latent variables)
        psi_a_beta = digamma(self.a_beta)
        _beta_entropy = (self.a_beta - xp.log(self.b_beta)
                         + gammaln(self.a_beta)
                         + (1 - self.a_beta) * psi_a_beta)
        if self._active_beta is not None:
            elbo += xp.sum(xp.where(self._active_beta, _beta_entropy, 0.0))
        else:
            elbo += xp.sum(_beta_entropy)
        del psi_a_beta, _beta_entropy
        # q(xi)
        elbo += xp.sum(self.a_xi - xp.log(self.b_xi)
                       + self._gammaln_a_xi
                       + (1 - self.a_xi) * self._digamma_a_xi)
        # q(eta)
        elbo += xp.sum(self.a_eta - xp.log(self.b_eta)
                       + self._gammaln_a_eta
                       + (1 - self.a_eta) * self._digamma_a_eta)
        # q(v) Gaussian entropy
        elbo += 0.5 * xp.sum(xp.log(2 * xp.pi * xp.e * self.sigma_v_diag))

        return float(elbo), float(poisson_ll), float(regression_ll)

    # =================================================================
    # Held-out log-likelihood (scHPF pattern: mean negative Poisson LL)
    # =================================================================

    def compute_heldout_ll(self, X_val, y_val=None, X_aux_val=None, n_iter=20):
        """
        Held-out log-likelihood on validation data.

        Returns
        -------
        total_ll : float
            Mean total LL per sample (Poisson + weighted regression if labels provided).
        poisson_ll : float
            Mean Poisson LL per sample.
        regression_ll : float or None
            Mean regression LL per sample (before weighting), or None if no labels.
        """
        if sp.issparse(X_val):
            X_val_coo = X_val.tocoo()
        else:
            X_val_coo = sp.coo_matrix(X_val)

        n_val = X_val.shape[0]

        # Infer theta for validation cells (freeze globals) via shared method
        if X_aux_val is None:
            X_aux_val = np.zeros((n_val, 0))
        X_aux_val = self._prepend_intercept(
            np.asarray(X_aux_val, dtype=np.float32), n=n_val)
        X_aux_v_dev = to_device(X_aux_val)

        a_theta_v, b_theta_v = self._infer_theta_sparse(
            X_val_coo, n_val, n_iter, X_aux_new=X_aux_v_dev)

        E_log_beta = self._E_log_beta_cache
        E_beta = self.E_beta

        row = to_device(X_val_coo.row.astype(np.int32))
        col = to_device(X_val_coo.col.astype(np.int32))
        data_np = X_val_coo.data if X_val_coo.data.dtype == np.float32 else X_val_coo.data.astype(np.float32)
        data = to_device(data_np)
        nnz_val = len(data_np)
        chunk = _auto_chunk_size(nnz_val, self.K)

        # Poisson LL per sample -- chunked
        E_log_theta_v = digamma(a_theta_v) - xp.log(b_theta_v)
        E_theta_v = a_theta_v / b_theta_v
        poisson_ll = 0.0
        gammaln_term = float(xp.sum(gammaln(data + 1)))
        for start in range(0, nnz_val, chunk):
            end = min(start + chunk, nnz_val)
            row_c = row[start:end]
            col_c = col[start:end]
            data_c = data[start:end]
            log_rates_c = E_log_theta_v[row_c] + E_log_beta[col_c]
            log_sum_c = logsumexp_rows(log_rates_c).ravel()
            poisson_ll += xp.dot(data_c, log_sum_c)
            del log_rates_c

        # Use only active beta entries in the rate term (masked = absent = 0)
        if self._active_beta is not None:
            _beta_col_sum = xp.where(self._active_beta, E_beta, 0.0).sum(axis=0)
        else:
            _beta_col_sum = E_beta.sum(axis=0)
        poisson_ll -= xp.sum(E_theta_v.sum(axis=0) * _beta_col_sum)
        poisson_ll -= gammaln_term
        poisson_ll_per_sample = float(poisson_ll) / n_val

        # Regression LL on validation data (if labels provided)
        regression_ll_per_sample = None
        if y_val is not None:
            y_v = to_device(np.asarray(y_val, dtype=np.float32))
            if y_v.ndim == 1:
                y_v = y_v[:, None]
            E_A = E_theta_v @ self.mu_v.T
            if self.p_aux > 0:
                E_A = E_A + X_aux_v_dev @ self.mu_gamma.T

            E_v_sq = self.mu_v ** 2 + self.sigma_v_diag
            Var_theta_v = a_theta_v / (b_theta_v ** 2)
            E_A_sq = E_A ** 2 + Var_theta_v @ E_v_sq.T

            zeta_val = xp.sqrt(xp.maximum(E_A_sq, 1e-8))
            lam = lambda_jj(zeta_val)

            reg_ll = xp.sum((y_v - 0.5) * E_A - lam * E_A_sq)
            reg_ll += xp.sum(lam * zeta_val ** 2 - 0.5 * zeta_val
                             + log_expit(zeta_val))
            regression_ll_per_sample = float(reg_ll / n_val)

        total_ll = poisson_ll_per_sample
        if regression_ll_per_sample is not None:
            total_ll += self.regression_weight * regression_ll_per_sample

        return float(total_ll), float(poisson_ll_per_sample), regression_ll_per_sample

    # =================================================================
    # fit()
    # =================================================================

    def fit(self, X_train, y_train, X_aux_train=None,
            X_val=None, y_val=None, X_aux_val=None,
            max_iter=600, check_freq=5, tol=0.001,
            v_warmup=50, verbose=True,
            early_stopping='heldout_ll'):
        """
        Fit the supervised Poisson factorization model.

        Parameters
        ----------
        X_train : sparse or dense (n, p)
        y_train : (n,) or (n, kappa)
        X_aux_train : (n, p_aux) or None
        X_val, y_val, X_aux_val : validation data (optional)
        max_iter : int
        check_freq : int
        tol : float -- convergence if |pct_change| < tol twice in a row
        v_warmup : int -- deprecated, kept for CLI compatibility (ignored)
        verbose : bool
        early_stopping : str
            'heldout_ll' -- stop on held-out LL / regression LL plateau (default).
            'elbo' -- stop only on ELBO convergence.
            'none' -- disable all early stopping, run all iterations.
        """
        t0 = time.time()

        # Print backend info
        info = backend_info()
        print(f"Backend: {info['device']}"
              + (f"  (JAX {info.get('jax_version', '')})" if info['backend'] == 'jax' else ""))

        if X_aux_train is None:
            X_aux_train = np.zeros((X_train.shape[0], 0))
        y = np.asarray(y_train, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        X_aux = np.asarray(X_aux_train, dtype=np.float32)

        # Prepend intercept column (column of 1s) to X_aux
        X_aux = self._prepend_intercept(X_aux, n=X_train.shape[0])

        # Initialize (creates numpy arrays, then transfers to device)
        self._initialize(X_train, y, X_aux)

        # Transfer training labels / aux to device for hot-path computations
        y = to_device(y)
        X_aux = to_device(X_aux)

        # Dense X for ELBO (if small enough) -- otherwise use sparse
        if sp.issparse(X_train):
            X_dense = None  # Will use sparse ELBO
        else:
            X_dense = X_train

        # Validation setup
        if X_val is not None and X_aux_val is None:
            X_aux_val = np.zeros((X_val.shape[0], 0))
        if X_val is not None:
            X_aux_val = np.asarray(X_aux_val, dtype=np.float32)

        self.elbo_history_ = []
        self.holl_history_ = []
        loss_list = []
        pct_changes = []

        best_holl = -np.inf
        best_params = None

        # Regression early stopping: stop if Reg degrades for too long
        best_reg_ll = -np.inf
        best_reg_params = None
        best_reg_iter = 0
        reg_patience = 10  # stop after this many consecutive checks of Reg degradation

        # HO-LL total early stopping
        best_holl_iter = 0
        best_holl_params = None
        holl_patience = 25

        for t in range(max_iter):
            diag = verbose and (t % check_freq == 0)
            t_iter_start = time.time()

            # 1. Compute phi and z_sums (sparse)
            # Free E_theta cache to make room for z_sum_theta (both are n×K).
            # Phi uses a_theta/b_theta directly; cache is recomputed lazily.
            self._invalidate_theta_cache()
            random_phi = (t == 0)  # scHPF: random Dirichlet on first iter
            z_sum_beta, z_sum_theta = self._compute_phi_sparse(random_init=random_phi)
            if diag:
                print(f"  [timing t={t}] phi: {time.time() - t_iter_start:.1f}s")

            # 2. Update beta, eta (gene side first -- scHPF order)
            self._update_beta(z_sum_beta)
            del z_sum_beta
            self._update_eta()
            if diag:
                print(f"  [diag t={t}] after beta,eta: "
                      f"E[beta]=[{float(self.E_beta.min()):.4e},{float(self.E_beta.max()):.4e}] "
                      f"E[eta]=[{float(self.E_eta.min()):.4e},{float(self.E_eta.max()):.4e}]")

            # 2b. Rescale factors — DISABLED.
            # _rescale_factors() is not a valid CAVI coordinate-ascent step:
            # it modifies b_theta, b_beta, mu_v, sigma_v_diag simultaneously
            # without optimizing any variational objective, breaking ELBO
            # monotonicity.  The clipping of s_theta to [0.5, 2.0] followed
            # by s_beta = 1/s_theta also breaks the invariant theta*beta
            # when clipping activates.  Empirically this causes sigma2_v
            # collapse (divided by s_theta^2 every iteration) and ELBO
            # divergence after ~25 iterations.
            # self._rescale_factors()
            self._refresh_log_caches()  # still needed: beta/eta just changed

            # 3. Update zeta (JJ bound tightening)
            self._update_zeta(X_aux)
            if diag:
                lam_diag = lambda_jj(self.zeta)
                print(f"  [diag t={t}] after zeta:   "
                      f"zeta=[{float(self.zeta.min()):.4e},{float(self.zeta.max()):.4e}] "
                      f"lambda(zeta)=[{float(lam_diag.min()):.4e},"
                      f"{float(lam_diag.max()):.4e}]")

            # 4. Update theta, xi (cell side -- with full regression from start)
            self._update_theta(z_sum_theta, y, X_aux)
            del z_sum_theta
            self._update_xi()
            if diag:
                # Compute E[theta] stats without caching full (n,K)
                eth_min, eth_max, eth_sum = float('inf'), float('-inf'), 0.0
                for _i0 in range(0, self.n, self._row_chunk):
                    _i1 = min(_i0 + self._row_chunk, self.n)
                    _etc = self.a_theta[_i0:_i1] / self.b_theta[_i0:_i1]
                    eth_min = min(eth_min, float(_etc.min()))
                    eth_max = max(eth_max, float(_etc.max()))
                    eth_sum += float(_etc.sum())
                eth_mean = eth_sum / (self.n * self.K)
                inner_info = f" inner={self._theta_inner_iters}" if self._theta_inner_iters > 0 else ""
                print(f"  [diag t={t}] after theta,xi: "
                      f"E[theta]=[{eth_min:.4e},{eth_max:.4e}] mean={eth_mean:.4e} "
                      f"b_theta=[{float(self.b_theta.min()):.4e},{float(self.b_theta.max()):.4e}] "
                      f"a_theta=[{float(self.a_theta.min()):.4e},{float(self.a_theta.max()):.4e}]{inner_info}")
                if self._theta_inner_iters > 0:
                    lam_diag2 = lambda_jj(self.zeta)
                    print(f"  [diag t={t}] post-theta zeta: "
                          f"zeta=[{float(self.zeta.min()):.4e},{float(self.zeta.max()):.4e}] "
                          f"lambda(zeta)=[{float(lam_diag2.min()):.4e},"
                          f"{float(lam_diag2.max()):.4e}]")

            # Refresh digamma caches after theta/beta changed
            self._refresh_log_caches()

            # 5. Update v, gamma
            self._update_v(y, X_aux, iteration=t)
            self._update_gamma(y, X_aux, iteration=t)

            # 5b. Re-tighten zeta after v/gamma changed
            self._update_zeta(X_aux)

            if diag:
                t_updates = time.time() - t_iter_start
                if self.p_aux > 0:
                    print(f"  [diag t={t}] after v,gamma: "
                          f"mu_v=[{float(self.mu_v.min()):.4e},{float(self.mu_v.max()):.4e}] "
                          f"sigma2_v=[{float(self.sigma_v_diag.min()):.4e},{float(self.sigma_v_diag.max()):.4e}] "
                          f"mu_gamma=[{float(self.mu_gamma.min()):.4e},{float(self.mu_gamma.max()):.4e}]")
                else:
                    print(f"  [diag t={t}] after v:   "
                          f"mu_v=[{float(self.mu_v.min()):.4e},{float(self.mu_v.max()):.4e}] "
                          f"sigma2_v=[{float(self.sigma_v_diag.min()):.4e},{float(self.sigma_v_diag.max()):.4e}]")
                print(f"  [timing t={t}] updates: {t_updates:.1f}s")

            # 6. Check convergence
            if t % check_freq == 0:
                elbo, pois_ll, reg_ll = self._compute_elbo(X_dense, y, X_aux)
                self.elbo_history_.append((t, elbo))

                # ELBO monotonicity check
                if diag and len(self.elbo_history_) >= 2:
                    prev_elbo = self.elbo_history_[-2][1]
                    delta = elbo - prev_elbo
                    if delta < -1.0:  # allow tiny numerical noise
                        print(f"  [WARN t={t}] ELBO DECREASED by {delta:.4e} "
                              f"({prev_elbo:.4e} -> {elbo:.4e})")

                # Held-out LL (with breakdown)
                holl = None
                holl_pois = None
                holl_reg = None
                if X_val is not None:
                    holl, holl_pois, holl_reg = self.compute_heldout_ll(
                        X_val, y_val=y_val, X_aux_val=X_aux_val)
                    self.holl_history_.append((t, holl, holl_pois,
                                              holl_reg if holl_reg is not None else 0.0))
                    if holl > best_holl:
                        best_holl = holl
                        best_holl_iter = t
                        best_holl_params = self._checkpoint()
                        best_params = best_holl_params  # keep alias

                # Track best training regression LL for early stopping
                if reg_ll > best_reg_ll:
                    best_reg_ll = reg_ll
                    best_reg_params = self._checkpoint()
                    best_reg_iter = t

                # Loss = mean negative ELBO (tracks full objective including regression)
                curr_loss = -elbo / self.n
                loss_list.append(curr_loss)

                if len(loss_list) >= 2:
                    prev = loss_list[-2]
                    pct = 100 * (curr_loss - prev) / max(abs(prev), 1e-10)
                    pct_changes.append(pct)
                else:
                    pct_changes.append(100.0)

                if verbose:
                    if holl is not None:
                        holl_parts = [f"  HO-LL={holl:.2f}"]
                        holl_parts.append(f"  HO-Pois={holl_pois:.2f}")
                        if holl_reg is not None:
                            holl_parts.append(f"  HO-Reg={holl_reg:.4e}")
                        holl_str = "".join(holl_parts)
                    else:
                        holl_str = ""
                    t_total = time.time() - t_iter_start
                    mu_v_preview = to_numpy(self.mu_v).ravel()[:3]
                    print(f"Iter {t:4d}: ELBO={elbo:.4e}  "
                          f"Pois={pois_ll:.4e}  Reg={reg_ll:.4e}  "
                          f"v={mu_v_preview}{holl_str}  "
                          f"[{t_total:.1f}s]")

                # Early stopping checks (gated by early_stopping mode)
                if early_stopping == 'heldout_ll':
                    # HO-LL total early stopping (preferred when validation available)
                    if X_val is not None and holl is not None:
                        iters_since_best = t - best_holl_iter
                        if iters_since_best >= holl_patience and t >= 30:
                            if verbose:
                                print(f"HO-LL early stop at iter {t}: "
                                      f"HO-LL hasn't improved in {iters_since_best} iters "
                                      f"(best HO-LL={best_holl:.4f} at iter {best_holl_iter})")
                            break

                    # Regression early stopping (training, fallback if no validation)
                    elif X_val is None:
                        iters_since_best = t - best_reg_iter
                        if iters_since_best >= reg_patience and t >= 30:
                            if verbose:
                                print(f"Regression early stop at iter {t}: "
                                      f"Reg hasn't improved in {iters_since_best} iters "
                                      f"(best Reg={best_reg_ll:.4e} at iter {best_reg_iter})")
                            break

                if early_stopping in ('heldout_ll', 'elbo'):
                    # Convergence check on full ELBO (not just Poisson term)
                    if len(loss_list) >= 3 and t >= 30:
                        c1 = abs(pct_changes[-1]) < tol
                        c2 = abs(pct_changes[-2]) < tol
                        if c1 and c2:
                            if verbose:
                                print(f"Converged at iter {t}")
                            break

        elapsed = time.time() - t0
        if verbose:
            print(f"\nTraining complete in {elapsed:.1f}s")
            if best_params is not None:
                print(f"Best HO-LL: {best_holl:.4f}")

        # Restore best checkpoint (skip when early stopping is disabled)
        if early_stopping != 'none':
            if best_holl_params is not None:
                if verbose:
                    print(f"Restoring best HO-LL checkpoint (iter {best_holl_iter}, "
                          f"HO-LL={best_holl:.4f})")
                self._restore(best_holl_params)
            elif best_reg_params is not None:
                if verbose:
                    print(f"Restoring best regression checkpoint (iter {best_reg_iter})")
                self._restore(best_reg_params)
        elif verbose:
            print("Early stopping disabled -- keeping final parameters.")

        return self

    # =================================================================
    # Checkpoint/restore
    # =================================================================

    def _checkpoint(self):
        return {
            'a_beta': to_numpy(self.a_beta).copy(),
            'b_beta': to_numpy(self.b_beta).copy(),
            'a_eta': to_numpy(self.a_eta).copy(),
            'b_eta': to_numpy(self.b_eta).copy(),
            'a_theta': to_numpy(self.a_theta).copy(),
            'b_theta': to_numpy(self.b_theta).copy(),
            'a_xi': to_numpy(self.a_xi).copy(),
            'b_xi': to_numpy(self.b_xi).copy(),
            'mu_v': to_numpy(self.mu_v).copy(),
            'sigma_v_diag': to_numpy(self.sigma_v_diag).copy(),
            'mu_gamma': to_numpy(self.mu_gamma).copy(),
            'Sigma_gamma': to_numpy(self.Sigma_gamma).copy(),
            'zeta': to_numpy(self.zeta).copy(),
        }

    def _restore(self, cp):
        for k, v in cp.items():
            setattr(self, k, to_device(v))
        self._invalidate_theta_cache()
        self._invalidate_beta_cache()
        self._refresh_log_caches()

    # =================================================================
    # predict / transform (API compat)
    # =================================================================

    def _infer_theta_sparse(self, X_coo, n_new, n_iter=20, X_aux_new=None):
        """Infer theta for new data using chunked sparse phi.

        Includes the label-independent quadratic regularization from the JJ
        bound (R_quad = 2 * lambda(zeta) @ E[v^2]) to match the training
        regime where theta is updated with the full regression correction.
        Without this, test-time theta is purely Poisson, creating a train-test
        distribution shift.

        Returns a_theta, b_theta.
        """
        a_theta = to_device(np.random.uniform(0.5 * self.a, 1.5 * self.a, (n_new, self.K)))
        b_theta = to_device(np.full((n_new, self.K), self.bp))
        a_xi = to_device(np.full(n_new, self.ap + self.K * self.a))
        b_xi = to_device(np.full(n_new, self.bp))

        # Use cached E_log_beta (has -inf for masked entries) to ensure
        # masked components get zero responsibility in phi_chunk_core.
        E_log_beta = self._E_log_beta_cache
        # Active-only beta sums for the Poisson rate term
        if self._active_beta is not None:
            beta_col_sums = xp.where(self._active_beta, self.E_beta, 0.0).sum(axis=0)
        else:
            beta_col_sums = self.E_beta.sum(axis=0)

        row = to_device(X_coo.row.astype(np.int32))
        col = to_device(X_coo.col.astype(np.int32))
        data_np = X_coo.data if X_coo.data.dtype == np.float32 else X_coo.data.astype(np.float32)
        data = to_device(data_np)
        nnz = len(data_np)
        chunk = _auto_chunk_size(nnz, self.K)
        K = self.K

        # Pre-compute label-independent regression quantities (tiny, (kappa, K))
        E_v = self.mu_v                                  # (kappa, K)
        E_v_sq = E_v ** 2 + self.sigma_v_diag            # (kappa, K)
        has_regression = self.regression_weight > 0
        rw = self.regression_weight

        for _ in range(n_iter):
            E_log_theta = digamma(a_theta) - xp.log(b_theta)
            E_theta = a_theta / b_theta
            E_xi = a_xi / b_xi

            # Chunked phi + scatter
            z_sum = xp.zeros((n_new, K))
            for start in range(0, nnz, chunk):
                end = min(start + chunk, nnz)
                row_c = row[start:end]
                col_c = col[start:end]
                data_c = data[start:end]
                Xphi = phi_chunk_core(E_log_theta[row_c], E_log_beta[col_c], data_c)
                z_sum = scatter_add_to(z_sum, row_c, Xphi,
                                       sorted_indices=False)
                del Xphi

            a_theta = self.a + z_sum
            b_poisson = E_xi[:, None] + beta_col_sums[None, :]

            if has_regression:
                # Compute zeta for new data to get lambda(zeta)
                theta_v = E_theta @ E_v.T                    # (n_new, kappa)
                if X_aux_new is not None and self.p_aux > 0:
                    theta_v = theta_v + X_aux_new @ self.mu_gamma.T
                Var_theta = E_theta / b_theta
                E_A_sq = xp.square(theta_v) + Var_theta @ E_v_sq.T
                zeta_new = xp.minimum(
                    xp.sqrt(xp.maximum(E_A_sq, 1e-8)), self.zeta_max
                )
                lam = lambda_jj(zeta_new)                    # (n_new, kappa)

                # Quadratic regularization: R_quad_coeff = 2 * lam @ E_v_sq
                R_quad = (2.0 * lam) @ E_v_sq                # (n_new, K)

                # Solve quadratic: b^2 - b_poisson*b - rw*R_quad*a_theta = 0
                c_quad = rw * R_quad
                disc = xp.sqrt(xp.square(b_poisson) + 4.0 * c_quad * a_theta)
                b_theta = (b_poisson + disc) / 2.0
                b_theta = xp.maximum(b_theta, 0.1 * b_poisson)
                b_theta = xp.maximum(b_theta, self.bp)
            else:
                b_theta = b_poisson

            b_xi = self.bp + E_theta.sum(axis=1)

        return a_theta, b_theta

    def predict_proba(self, X_new, X_aux_new=None, n_iter=20):
        """Predict P(y=1 | X_new).

        Uses the probit approximation to account for posterior variance in
        both theta and v, improving probability calibration:
            P(y=1) ≈ sigmoid(E[logit] / sqrt(1 + pi * Var[logit] / 3))
        """
        if sp.issparse(X_new):
            X_coo = X_new.tocoo()
        else:
            X_coo = sp.coo_matrix(X_new)

        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, 0))
        X_aux_new = self._prepend_intercept(
            np.asarray(X_aux_new, dtype=np.float32), n=n_new)
        X_aux_new = to_device(X_aux_new)

        a_theta, b_theta = self._infer_theta_sparse(
            X_coo, n_new, n_iter, X_aux_new=X_aux_new)

        E_theta = a_theta / b_theta
        logits = E_theta @ self.mu_v.T
        if self.p_aux > 0:
            logits = logits + X_aux_new @ self.mu_gamma.T

        # Probit approximation: shrink logits by posterior variance of the
        # linear predictor A = theta @ v.  Var[A] ≈ Var[theta] @ E[v^2] +
        # E[theta]^2 @ Var[v] (independent posteriors).
        Var_theta = E_theta / b_theta                          # a/b^2 = (a/b)/b
        E_v_sq = self.mu_v ** 2 + self.sigma_v_diag            # (kappa, K)
        var_logits = Var_theta @ E_v_sq.T + xp.square(E_theta) @ self.sigma_v_diag.T
        if self.p_aux > 0:
            # Add gamma variance contribution (diagonal of Sigma_gamma)
            gamma_var = xp.stack([xp.diag(self.Sigma_gamma[k])
                                  for k in range(self.kappa)])  # (kappa, p_aux)
            var_logits = var_logits + xp.square(X_aux_new) @ gamma_var.T

        scale = xp.sqrt(1.0 + (np.pi / 3.0) * var_logits)
        logits_calibrated = logits / scale

        return to_numpy(_expit(logits_calibrated)).squeeze()

    def transform(self, X_new, y_new=None, X_aux_new=None, n_iter=20, **kwargs):
        """Infer theta for new data. Returns dict with E_theta, a_theta, b_theta."""
        if sp.issparse(X_new):
            X_coo = X_new.tocoo()
        else:
            X_coo = sp.coo_matrix(X_new)

        n_new = X_new.shape[0]
        if X_aux_new is None:
            X_aux_new = np.zeros((n_new, 0))
        X_aux_new = self._prepend_intercept(
            np.asarray(X_aux_new, dtype=np.float32), n=n_new)
        X_aux_dev = to_device(X_aux_new)
        a_theta, b_theta = self._infer_theta_sparse(
            X_coo, n_new, n_iter, X_aux_new=X_aux_dev)

        return {
            'E_theta': to_numpy(a_theta / b_theta),
            'a_theta': to_numpy(a_theta),
            'b_theta': to_numpy(b_theta),
        }
