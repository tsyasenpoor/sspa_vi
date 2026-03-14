"""JAX/GPU acceleration backend for SSPA-VI.

Auto-detects JAX availability and GPU presence.  Provides a unified
array namespace ``xp`` and JIT-compiled hot-path functions that fall
back to NumPy/SciPy equivalents when JAX is not installed.

Usage::

    from .jax_backend import (
        xp, USE_JAX, to_device, to_numpy,
        digamma, gammaln, logsumexp_rows, softmax_rows,
        log_expit, lambda_jj, scatter_add_to, phi_chunk_core,
        expit, backend_info,
    )
"""

import numpy as np
from functools import partial

# ── Backend detection ───────────────────────────────────────────────
USE_JAX = False
HAS_GPU = False
_DEVICE_STR = "CPU (NumPy)"

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import jit
    import jax.scipy.special as _jsp

    USE_JAX = True
    _devices = jax.devices()
    HAS_GPU = any(d.platform == "gpu" for d in _devices)
    _DEVICE_STR = f"GPU ({_devices})" if HAS_GPU else f"CPU/JAX ({_devices})"
except ImportError:
    pass


def backend_info() -> dict:
    """Return a dict describing the active compute backend."""
    info = {"backend": "jax" if USE_JAX else "numpy", "device": _DEVICE_STR}
    if USE_JAX:
        info["jax_version"] = jax.__version__
    return info


# ── Array namespace ─────────────────────────────────────────────────
xp = jnp if USE_JAX else np


# ── Device transfer helpers ─────────────────────────────────────────

def to_device(x):
    """Move a numpy array to the active JAX device (no-op without JAX)."""
    if USE_JAX and not isinstance(x, jnp.ndarray):
        return jnp.asarray(x)
    return x


def to_numpy(x):
    """Move an array to host numpy."""
    return np.asarray(x)


# ── Hot-path primitives ────────────────────────────────────────────
# Two complete implementations: JAX (JIT-compiled) and NumPy fallback.

if USE_JAX:
    # ─── JAX path ───────────────────────────────────────────────────

    @jit
    def digamma(x):
        return _jsp.digamma(x)

    @jit
    def gammaln(x):
        return _jsp.gammaln(x)

    @jit
    def logsumexp_rows(a):
        """Row-wise logsumexp."""
        return _jsp.logsumexp(a, axis=1, keepdims=True)

    @jit
    def softmax_rows(work):
        """Pure-functional row-wise softmax."""
        shifted = work - work.max(axis=1, keepdims=True)
        e = jnp.exp(shifted)
        return e / e.sum(axis=1, keepdims=True)

    @jit
    def log_expit(x):
        """log(sigmoid(x))."""
        return jax.nn.log_sigmoid(x)

    @jit
    def expit(x):
        """Logistic sigmoid."""
        return jax.nn.sigmoid(x)

    @jit
    def lambda_jj(zeta):
        """JJ bound helper: tanh(z/2) / (4z)."""
        safe = jnp.maximum(jnp.abs(zeta), 1e-8)
        return jnp.tanh(safe / 2.0) / (4.0 * safe)

    @partial(jit, static_argnums=(2,))
    def _segment_sum_sorted(values, indices, num_segments):
        """segment_sum for pre-sorted indices (much faster than scatter)."""
        return jax.ops.segment_sum(values, indices, num_segments=num_segments)

    def scatter_add_to(target, indices, values, *, sorted_indices=False):
        """Accumulate *values* into *target* at *indices*.

        When *sorted_indices* is True, uses segment_sum which is significantly
        faster on GPU than atomic scatter for non-unique indices.
        Returns a **new** array (JAX arrays are immutable).
        """
        if sorted_indices:
            return target + _segment_sum_sorted(values, indices, int(target.shape[0]))
        return target.at[indices].add(values)

    @jit
    def phi_chunk_core(E_log_theta_rows, E_log_beta_cols, data_c):
        """Fused log-rates -> softmax -> scale by data."""
        work = E_log_theta_rows + E_log_beta_cols
        shifted = work - work.max(axis=1, keepdims=True)
        e = jnp.exp(shifted)
        phi = e / e.sum(axis=1, keepdims=True)
        return phi * data_c[:, None]

else:
    # ─── NumPy / SciPy fallback ────────────────────────────────────

    from scipy.special import (
        digamma,
        gammaln,
        logsumexp as _scipy_lse,
        log_expit,
        expit,
    )

    def logsumexp_rows(a):
        """Row-wise logsumexp (scipy C implementation)."""
        return _scipy_lse(a, axis=1, keepdims=True)

    def softmax_rows(work):
        """In-place row-wise softmax (overwrites *work*)."""
        work -= work.max(axis=1, keepdims=True)
        np.exp(work, out=work)
        work /= work.sum(axis=1, keepdims=True)
        return work

    def lambda_jj(zeta):
        safe = np.maximum(np.abs(zeta), 1e-8)
        return np.tanh(safe / 2.0) / (4.0 * safe)

    def scatter_add_to(target, indices, values, *, sorted_indices=False):
        """Accumulate via sparse CSC matmul (no Python K-loop).

        Builds a CSC matrix M of shape (n_out, chunk) from *indices*
        in O(chunk) time (no sorting needed), then computes
        target += M @ values  in a single BLAS call.
        """
        from scipy.sparse import csc_matrix
        chunk = len(indices)
        n_out = target.shape[0]
        # CSC format: each column j has exactly one entry at row indices[j]
        M = csc_matrix(
            (np.ones(chunk, dtype=values.dtype), indices,
             np.arange(chunk + 1, dtype=np.int32)),
            shape=(n_out, chunk),
        )
        target += M @ values
        return target

    def phi_chunk_core(E_log_theta_rows, E_log_beta_cols, data_c):
        """log-rates -> in-place softmax -> scale by data."""
        work = E_log_theta_rows + E_log_beta_cols
        work -= work.max(axis=1, keepdims=True)
        np.exp(work, out=work)
        work /= work.sum(axis=1, keepdims=True)
        work *= data_c[:, None]
        return work
