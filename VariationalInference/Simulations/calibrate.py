"""Calibration entry points: ss_type from mean matrix, ss_pert per truth, solve_delta."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from . import config
from .calibrate_subdominance_r import ss_type_from_baseline, ss_pert_unit, delta_for_r


def ss_type_from_means(mu_gene_by_cell: np.ndarray, cell_type: np.ndarray,
                       target_depth: float | None = None) -> float:
    """Design §4: ss_type denominator computed from scDesign3 means rather than the
    noisy realization, on log1p(library-normalized) scale. Input is dense (genes × cells)."""
    G, N = mu_gene_by_cell.shape
    depth = mu_gene_by_cell.sum(axis=0)
    if target_depth is None:
        target_depth = float(np.median(depth))
    sf = target_depth / np.maximum(depth, 1.0)
    L = np.log1p(mu_gene_by_cell * sf[None, :])           # (G, N)
    T = int(cell_type.max()) + 1
    onehot = np.zeros((N, T), dtype=np.float64)
    onehot[np.arange(N), cell_type] = 1.0
    nt = onehot.sum(axis=0)                               # (T,)
    mu_tj = (onehot.T @ L.T) / np.maximum(nt[:, None], 1.0)  # (T, G)
    gene_mean = (nt @ mu_tj) / N                          # (G,)
    C = mu_tj - gene_mean[None, :]                        # (T, G)
    return float((nt[:, None] * C ** 2).sum())


def _load_counts_and_types():
    counts_df = pd.read_csv(config.BASELINE_COUNTS_CSV, index_col=0)
    counts = sp.csr_matrix(counts_df.to_numpy(dtype=np.float64).T)   # cells×genes
    meta = pd.read_csv(config.BASELINE_META_CSV, index_col=0)
    types = meta["majorType"].map(config.TYPE_TO_INT).to_numpy()
    return counts, types


def compute_ss_type() -> float:
    """Use ss_type_from_baseline on simulated_counts.csv because scDesign3 was fit
    intercept-only (mu_formula='1'), so mu_mat has no cell-type variance; the
    realized counts carry the type structure via the copula. Per-type averaging
    over ~1300 cells/type makes the per-cell noise negligible vs between-type
    structure (we're measuring the SS of TYPE-MEANS, not of cells)."""
    counts, types = _load_counts_and_types()
    val = float(ss_type_from_baseline(counts, types))
    config.SIM_ROOT.mkdir(parents=True, exist_ok=True)
    (config.SIM_ROOT / "ss_type.json").write_text(json.dumps({"ss_type": val}, indent=2))
    return val


def load_ss_type() -> float:
    return float(json.loads((config.SIM_ROOT / "ss_type.json").read_text())["ss_type"])


from .truths import load_truth


def _build_A_draw(truth: dict, draw_seed: int) -> np.ndarray:
    """Build A (n_cells × L_cols) under §3.4 with a single (D, pi, b, patient_assign) draw.

    Self-contained — does not re-use dataset.py because we run this BEFORE dataset.py
    exists in the pipeline order. Returns (theta_star - theta_base)/bbar in L_cols."""
    rng = np.random.default_rng(draw_seed)
    meta = pd.read_csv(config.BASELINE_META_CSV, index_col=0)
    types = meta["majorType"].map(config.TYPE_TO_INT).to_numpy()
    N = len(types)
    G = config.N_PATIENTS
    # Balanced D
    D = np.zeros(G, dtype=np.int8); D[: G // 2] = 1; rng.shuffle(D)
    # Composition
    pi_global = np.bincount(types, minlength=config.T) / N
    alpha = config.DIRICHLET_A0 * pi_global
    pi_g = rng.dirichlet(alpha, size=G)                # (G, T)
    # Greedy assignment
    quota = np.round(pi_g * (N / G)).astype(int)
    quota[:, -1] += (N // G) - quota.sum(axis=1)       # tie-up
    pat = np.full(N, -1, dtype=np.int32)
    cells_by_type = {t: rng.permutation(np.flatnonzero(types == t)).tolist()
                     for t in range(config.T)}
    for g in range(G):
        for t in range(config.T):
            take = quota[g, t]
            if take <= 0: continue
            cells = cells_by_type[t][:take]
            del cells_by_type[t][:take]
            pat[cells] = g
    # Any orphan cells (rounding) -> stuff into the smallest patient
    orphans = np.flatnonzero(pat < 0)
    if orphans.size:
        for c in orphans:
            counts = np.bincount(pat[pat >= 0], minlength=G)
            pat[c] = int(np.argmin(counts))
    # Activity
    L = config.L_COLS
    A = np.zeros((N, L), dtype=np.float64)
    bbar = config.ALPHA_B / config.LAMBDA_B            # mean of Gamma(α, λ) shape/rate
    for l in range(L):
        T_l = set(truth["T_ell"][l].tolist())
        responder = np.isin(types, list(T_l))
        case_resp = (D[pat] == 1) & responder
        b = rng.gamma(config.ALPHA_B, 1.0 / config.LAMBDA_B, size=case_resp.sum())
        A[case_resp, l] = b / bbar
    return A


def compute_ss_pert(truth_idx: int, n_A_draws: int = 5, seed: int = 0) -> float:
    truth = load_truth(truth_idx)
    U = truth["u"].astype(np.float64)
    vals = []
    rng = np.random.default_rng(seed)
    for k in range(n_A_draws):
        A = _build_A_draw(truth, int(rng.integers(0, 2**31 - 1)))
        vals.append(ss_pert_unit(A, U))
    return float(np.mean(vals))


def solve_delta(truth_idx: int, r_targets: list[float] | None = None,
                n_A_draws: int = 5) -> dict[float, float]:
    r_targets = r_targets or config.R_VALUES
    ss_t = load_ss_type()
    ss_p1 = compute_ss_pert(truth_idx, n_A_draws=n_A_draws)
    deltas = delta_for_r(np.array(r_targets, dtype=np.float64), ss_t, ss_p1)
    return {float(r): float(d) for r, d in zip(r_targets, deltas)}


def run_calibration(G_truth: int | None = None) -> None:
    G_truth = G_truth or config.G_TRUTH
    out: dict[str, dict[str, float]] = {}
    for t in range(G_truth):
        per_r = solve_delta(t)
        out[str(t)] = {f"{r:.4f}": d for r, d in per_r.items()}
        print(f"  truth {t}: {per_r}")
    (config.SIM_ROOT / "delta_calibration.json").write_text(json.dumps(out, indent=2))


def load_delta(truth_idx: int, r: float) -> float:
    d = json.loads((config.SIM_ROOT / "delta_calibration.json").read_text())
    return float(d[str(truth_idx)][f"{r:.4f}"])
