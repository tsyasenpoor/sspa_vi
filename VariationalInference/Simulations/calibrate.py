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


def ss_type_from_means_log1pmu(mu_gene_by_cell: np.ndarray, cell_type: np.ndarray) -> float:
    """SS_type on log1p(mu0) -- the principled matched scale for r.

    The additive perturbation log lambda = log mu0 + delta*sum_l u(theta*-base) lives on the
    log scale, while the original ss_type_from_means uses log1p(library-normalized mu), a
    different transform, so r mixed units. NOTE: the obvious fix -- SS on raw log mu0 -- is
    degenerate here: mu0 is extremely sparse (median 0.019, 20% < 1e-3, many exact zeros), so
    log mu0 is dominated by the floor on near-zero means (between-type SS ~1e3x larger; r(0.15)
    -> ~1e-4). log1p(mu0) is finite at zero and is the count-level salience scale on which the
    perturbation's count effect actually lands; on it the current r is essentially unchanged
    (r=0.15 -> 0.131), so the existing datasets do NOT need regeneration -- relabel r instead."""
    G, N = mu_gene_by_cell.shape
    L = np.log1p(np.maximum(mu_gene_by_cell, 0.0))        # (G, N) finite at zero
    T = int(cell_type.max()) + 1
    onehot = np.zeros((N, T), dtype=np.float64)
    onehot[np.arange(N), cell_type] = 1.0
    nt = onehot.sum(axis=0)
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
    """Compute ss_type from scDesign3's fitted means (mu_mat) — the design's original
    intent (§4 of the implementation design). Requires a scDesign3 fit with cell-type
    in the marginal (mu_formula contains majorType); otherwise mu has no type variance
    and the SS collapses to numerical noise. Verified by the premise check before this
    switch (see /labs/Aguiar/SSPA_BRAY/scdesign3_covid19_cellTypeMarginal_8kcells_10kgenes)."""
    import h5py
    with h5py.File(config.NB_PARAMS_H5, "r") as f:
        mu = np.asarray(f["mu_mat"], dtype=np.float64)
    meta = pd.read_csv(config.BASELINE_META_CSV, index_col=0)
    types = meta["majorType"].map(config.TYPE_TO_INT).to_numpy()
    N = len(types)
    mu_gc = mu if mu.shape[1] == N else mu.T          # ensure genes × cells
    val = float(ss_type_from_means(mu_gc, types))
    config.SIM_ROOT.mkdir(parents=True, exist_ok=True)
    (config.SIM_ROOT / "ss_type.json").write_text(json.dumps({"ss_type": val}, indent=2))
    return val


def load_ss_type() -> float:
    return float(json.loads((config.SIM_ROOT / "ss_type.json").read_text())["ss_type"])


from .truths import load_truth


def _build_A_draw(truth: dict, draw_seed: int) -> np.ndarray:
    """Build A (n_cells x L_cols) using the SAME composition+activity functions as dataset.py
    so the calibration A matches the generation A. Calibrated at the headline perturbation
    fraction (PERTURB_FRAC_HEADLINE); lower rho at run time then genuinely dilutes signal."""
    from .dataset import patient_composition, activity
    rng = np.random.default_rng(draw_seed)
    meta = pd.read_csv(config.BASELINE_META_CSV, index_col=0)
    types = meta["majorType"].map(config.TYPE_TO_INT).to_numpy()
    g_i, K = patient_composition(types, rng)
    theta_star = activity(truth, types, g_i, K, rng, perturb_frac=config.PERTURB_FRAC_HEADLINE)
    # A is the activation deviation — exact match to dataset's perturb_and_sample's A_dev
    return (theta_star - config.THETA_BASE).astype(np.float64)


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
