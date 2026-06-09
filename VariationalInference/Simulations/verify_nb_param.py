"""Phase A.2 gate. Disambiguate sigma_mat as size (theta) vs dispersion (1/theta).

Discriminator: per-(gene, cell-type) CONDITIONAL variance. Marginal mean
and marginal variance are insufficient (design §3 step 5).

Procedure:
  1. Load mu_mat, sigma_mat, simulated_counts.csv, cell-type labels.
  2. For each candidate rule (`size`: n=sigma_mean_j; `dispersion`: n=1/sigma_mean_j):
     resample 200 randomly-selected genes at lambda = mu_mat across all 8000 cells;
     compute within-(gene, cell-type) variance per (gene, type) cell, ratio against
     simulated_counts.csv's within-(gene, type) variance.
  3. Pick the rule whose median log-ratio is closest to 0; assert |median| < 0.25
     and IQR < 0.5. Write the winning rule to config.NB_SIZE_FROM_SIGMA.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from . import config


def _load_baseline():
    with h5py.File(config.NB_PARAMS_H5, "r") as f:
        mu = np.asarray(f["mu_mat"], dtype=np.float64)            # (G, N) genes×cells
        sg = np.asarray(f["sigma_mat"], dtype=np.float64)         # (G, N)
    counts = pd.read_csv(config.BASELINE_COUNTS_CSV, index_col=0).to_numpy(np.int32)  # (G, N)
    meta = pd.read_csv(config.BASELINE_META_CSV, index_col=0)
    types = meta["majorType"].map(config.TYPE_TO_INT).to_numpy()  # (N,)
    return mu, sg, counts, types


def _within_type_variance(M: np.ndarray, types: np.ndarray) -> np.ndarray:
    """For (gene, cell) matrix M, return (gene, type) within-type sample variance."""
    G, _ = M.shape
    T = int(types.max()) + 1
    out = np.zeros((G, T), dtype=np.float64)
    for t in range(T):
        sl = M[:, types == t]
        out[:, t] = sl.var(axis=1, ddof=1)
    return out


def _resample(mu_sub: np.ndarray, sigma_mean_sub: np.ndarray, rule: str,
              rng: np.random.Generator) -> np.ndarray:
    if rule == "size":
        n = np.broadcast_to(sigma_mean_sub[:, None], mu_sub.shape)
    elif rule == "dispersion":
        n = np.broadcast_to(1.0 / np.maximum(sigma_mean_sub[:, None], 1e-6), mu_sub.shape)
    else:
        raise ValueError(rule)
    p = n / (n + np.maximum(mu_sub, 1e-6))
    return rng.negative_binomial(n=n, p=p).astype(np.int32)


def run(n_genes_check: int = 200, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    mu, sg, counts, types = _load_baseline()
    G, N = mu.shape
    valid_genes = np.where(~np.isnan(sg).all(axis=1))[0]
    gene_idx = rng.choice(valid_genes, size=n_genes_check, replace=False)
    mu_sub = mu[gene_idx]
    sigma_mean_sub = sg[gene_idx].mean(axis=1)             # per-gene mean over cells
    ref_var = _within_type_variance(counts[gene_idx], types)  # (n_genes, T)
    results: dict[str, dict] = {}
    for rule in ("size", "dispersion"):
        sim = _resample(mu_sub, sigma_mean_sub, rule, rng)
        sim_var = _within_type_variance(sim, types)
        log_ratio = np.log(np.maximum(sim_var, 1e-3)) - np.log(np.maximum(ref_var, 1e-3))
        med = float(np.median(log_ratio))
        iqr = float(np.subtract(*np.percentile(log_ratio, [75, 25])))
        results[rule] = dict(median_log_ratio=med, iqr=iqr)
        print(f"  {rule:11s}: median log(sim_var/ref_var) = {med:+.3f}  IQR = {iqr:.3f}")
    winner = min(results, key=lambda k: abs(results[k]["median_log_ratio"]))
    print(f"\n  WINNER: NB_SIZE_FROM_SIGMA = '{winner}'")
    assert abs(results[winner]["median_log_ratio"]) < 0.25, \
        f"NB gate FAILED: median log-ratio = {results[winner]['median_log_ratio']:.3f}"
    assert results[winner]["iqr"] < 0.5, \
        f"NB gate FAILED: IQR = {results[winner]['iqr']:.3f}"

    config.SIM_ROOT.mkdir(parents=True, exist_ok=True)
    (config.SIM_ROOT / "nb_param_gate.json").write_text(json.dumps(
        {"winner": winner, "details": results}, indent=2))

    # Overlap diagnostic — carrier sets should not heavily land on NaN-dispersion genes
    try:
        from .truths import draw_truth
        nan_genes = set(np.flatnonzero(np.isnan(sg).all(axis=1)).tolist())
        for t_idx in range(config.G_TRUTH):
            t = draw_truth(t_idx)
            overlaps = []
            for l in range(config.L_COLS):
                S = set(t["S"][l].tolist())
                overlaps.append(len(S & nan_genes))
            print(f"  truth {t_idx} carrier ∩ NaN-dispersion (per ℓ): {overlaps}")
    except Exception as e:
        print(f"  carrier/NaN overlap diagnostic skipped: {e}")

    return winner


if __name__ == "__main__":
    run()
