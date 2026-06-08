"""Calibrate the subdominance ratio r (spec v2, eq. 3.5) from the scDesign3 baseline panel.

    r(delta) = delta^2 * SS_pert(1) / SS_type ,  delta(r*) = sqrt(r* * SS_type / SS_pert(1)).

SS_type   : between-cell-type, gene-centered SS of log-normalized BASELINE counts (nuisance the
            factorization competes for; per-gene baseline removed since it is absorbed by eta_j).
SS_pert(1): gene-centered SS of the program perturbation at unit effect, in (iota+1)-space
            (rank iota+1; never materializes the n x p perturbation).
"""
from __future__ import annotations
import numpy as np
from scipy import sparse


def ss_type_from_baseline(
    counts: sparse.csr_matrix,   # X^(0): n x p nonneg counts (cells x genes)
    cell_type: np.ndarray,       # (n,) int labels in [0, T)
    target_depth: float | None = None,
) -> float:
    """Denominator of eq. (3.5): sum_t n_t sum_j (mu_tj - gene_mean_j)^2 on log1p(CPM) scale."""
    n, _ = counts.shape
    depth = np.asarray(counts.sum(axis=1)).ravel()
    sf = (np.median(depth) if target_depth is None else target_depth) / np.maximum(depth, 1.0)
    L = counts.multiply(sf[:, None]).tocsr()                 # library-normalize (zeros stay zero)
    L = L.copy(); L.data = np.log1p(L.data)                  # log1p on nonzeros; log1p(0)=0
    T = int(cell_type.max()) + 1
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), cell_type)), shape=(n, T))
    nt = np.asarray(onehot.sum(axis=0)).ravel()              # (T,)
    mu_tj = np.asarray((onehot.T @ L).todense()) / np.maximum(nt[:, None], 1.0)   # (T, p)
    gene_mean = (nt @ mu_tj) / n                             # (p,)
    C = mu_tj - gene_mean[None, :]                           # (T, p)
    return float((nt[:, None] * C**2).sum())


def ss_pert_unit(
    A: np.ndarray,               # (n, L) activation deviations (theta* - base)/bbar, L = iota+1 (incl. decoy)
    U: np.ndarray,               # (p, L) unit per-gene pattern u_{jl} * 1[j in S_l]
) -> float:
    """Numerator of eq. (3.5) at delta=1, gene-centered, computed in L-space (rank L)."""
    n = A.shape[0]
    M_A = A.T @ A                                            # (L, L)
    M_U = U.T @ U                                            # (L, L)
    fro2 = float((M_A * M_U).sum())                          # ||P||_F^2 = <A^T A, U^T U>
    a_bar = A.mean(axis=0)                                   # (L,)
    center = float(n * (a_bar @ M_U @ a_bar))                # n * sum_j Pbar_j^2
    return fro2 - center


def delta_for_r(r_target: float | np.ndarray, ss_type: float, ss_pert_1: float) -> np.ndarray:
    """Invert r(delta) = delta^2 * SS_pert(1)/SS_type for the global effect-size scalar delta."""
    return np.sqrt(np.asarray(r_target, float) * ss_type / ss_pert_1)


# ---- usage --------------------------------------------------------------------------------------
# counts, cell_type : the scDesign3 8k x 10k baseline panel + its type labels
# A                 : build from the patient/responder assignment (sec 3.4):
#                       A[i, l] = D[g[i]] * 1[t[i] in T_l] * b[i, l] / bbar_l         (decoy = column 0)
# U                 : U[:, l] = u_jl over carrier set S_l, u ~ Unif[0.5, 1.5], else 0
#
# ss_t   = ss_type_from_baseline(counts, cell_type)
# ss_p1  = ss_pert_unit(A, U)                 # average over a few A-draws for a stable target
# deltas = delta_for_r(np.array([0.05, 0.20, 0.50]), ss_t, ss_p1)   # -> {low, mid, high}
# delta_jl = deltas[level] * U                # final per-gene log-fold effects for sec 3.5
