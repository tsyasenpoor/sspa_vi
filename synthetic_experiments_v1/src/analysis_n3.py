"""
N3 analysis: cross-seed stability.

Two pieces:
  1. Pairwise cross-seed cosine on Beta_hat per method (Hungarian-matched).
     Requires save_posteriors=true (N3 sweep).
  2. Cross-seed variance of v_hat after Hungarian alignment to ground truth.
     Uses N3 sweep raw if available; falls back to existing C1 raw.

Outputs:
  results/aggregated/N3_cross_seed_stability/{long.csv, pairwise_cos.csv}
  figures/N3/{pairwise_cos_heatmap_drgp.png, pairwise_cos_heatmap_nmf.png,
              v_variance_bars.png}
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _hungarian_cos(B1: np.ndarray, B2: np.ndarray) -> float:
    """Match columns of B1 to B2 by cosine; return mean cosine of matched pairs."""
    eps = 1e-12
    B1n = B1 / (np.linalg.norm(B1, axis=0, keepdims=True) + eps)
    B2n = B2 / (np.linalg.norm(B2, axis=0, keepdims=True) + eps)
    cos = B1n.T @ B2n
    row_ind, col_ind = linear_sum_assignment(-cos)
    return float(cos[row_ind, col_ind].mean())


def load_method_fits(raw_dir: str, method: str) -> list[dict]:
    """Returns list of {seed, Beta_hat, v_hat, pi, v_true, rel_idx} for the method."""
    out = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "cond*_seed*.npz"))):
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        if method not in d.files:
            continue
        m = d[method].item()
        if not isinstance(m, dict) or m.get("error"):
            continue
        if m.get("Beta_hat") is None:
            continue
        out.append({
            "seed": int(d["seed"]),
            "Beta_hat": np.asarray(m["Beta_hat"]),
            "v_hat": np.asarray(m["v_hat"]) if m.get("v_hat") is not None else None,
            "pi": np.asarray(m["pi"]),
            "v_true": np.asarray(d.get("v_true", [])) if "v_true" in d.files else None,
            "rel_idx": np.asarray(d.get("rel_idx", [])) if "rel_idx" in d.files else None,
        })
    return out


def pairwise_cos_matrix(fits: list[dict]) -> np.ndarray:
    n = len(fits)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            v = _hungarian_cos(fits[i]["Beta_hat"], fits[j]["Beta_hat"])
            M[i, j] = v; M[j, i] = v
    return M


def plot_cos_heatmap(M: np.ndarray, label: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="matched cos sim")
    ax.set_xlabel("seed j"); ax.set_ylabel("seed i")
    tri = M[np.triu_indices_from(M, k=1)]
    ax.set_title(f"N3: {label}  cross-seed Beta cos\n"
                  f"mean={tri.mean():.3f}  std={tri.std():.3f}  "
                  f"min={tri.min():.3f}")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def _v_aligned_to_true(v_true: np.ndarray, v_hat: np.ndarray, pi: np.ndarray) -> np.ndarray:
    K_true = v_true.size
    out = np.full(K_true, np.nan)
    for k_fit in range(v_hat.size):
        k_true = int(pi[k_fit])
        if 0 <= k_true < K_true:
            out[k_true] = v_hat[k_fit]
    return out


def plot_v_variance(fits_drgp: list[dict], fits_nmf: list[dict], out_path: str) -> None:
    K_true = fits_drgp[0]["v_true"].size if fits_drgp else 10
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.35
    pos = np.arange(K_true)

    def _stats(fits):
        if not fits:
            return None
        arr = np.stack([_v_aligned_to_true(f["v_true"], f["v_hat"], f["pi"])
                        for f in fits])  # (n_seeds, K_true)
        return arr

    arr_d = _stats(fits_drgp)
    arr_n = _stats(fits_nmf)

    if arr_d is not None:
        med = np.nanmedian(arr_d, axis=0)
        q25 = np.nanquantile(arr_d, 0.25, axis=0)
        q75 = np.nanquantile(arr_d, 0.75, axis=0)
        err = np.vstack([med - q25, q75 - med])
        ax.bar(pos - width/2, med, width=width, yerr=err,
               color="#2b6aa6", alpha=0.85, label="DRGP", capsize=3)
    if arr_n is not None:
        med = np.nanmedian(arr_n, axis=0)
        q25 = np.nanquantile(arr_n, 0.25, axis=0)
        q75 = np.nanquantile(arr_n, 0.75, axis=0)
        err = np.vstack([med - q25, q75 - med])
        ax.bar(pos + width/2, med, width=width, yerr=err,
               color="#aa5c2b", alpha=0.85, label="NMF+LR", capsize=3)

    if fits_drgp:
        rel_set = set(fits_drgp[0]["rel_idx"].tolist())
        v_true = fits_drgp[0]["v_true"]
        ax.scatter(pos, v_true, color="black", marker="x", s=40, zorder=4,
                   label="v_true (seed 0)")
        for k in rel_set:
            ax.axvspan(k - 0.4, k + 0.4, color="lightyellow", alpha=0.3, zorder=0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(pos); ax.set_xticklabels([str(k) for k in pos])
    ax.set_xlabel("program k (true index, Hungarian-aligned)")
    ax.set_ylabel("v_hat (median ± IQR across seeds)")
    ax.set_title("N3: cross-seed variance of v_hat (after Hungarian to ground truth)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/N3_cross_seed_stability")
    ap.add_argument("--out-dir", default="results/aggregated/N3_cross_seed_stability")
    ap.add_argument("--figure-dir", default="figures/N3")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    fits_drgp = load_method_fits(args.raw_dir, "drgp_unmasked")
    fits_nmf  = load_method_fits(args.raw_dir, "nmf_lr")
    print(f"loaded DRGP fits: {len(fits_drgp)}  NMF fits: {len(fits_nmf)}")

    rows = []

    if fits_drgp:
        M = pairwise_cos_matrix(fits_drgp)
        tri = M[np.triu_indices_from(M, k=1)]
        rows.append({"method": "drgp_unmasked",
                     "pairwise_cos_mean": float(tri.mean()),
                     "pairwise_cos_std":  float(tri.std()),
                     "pairwise_cos_min":  float(tri.min()),
                     "n_seeds": len(fits_drgp)})
        np.save(Path(args.out_dir) / "pairwise_cos_drgp.npy", M)
        plot_cos_heatmap(M, "DRGP", str(Path(args.figure_dir) / "pairwise_cos_heatmap_drgp.png"))
    if fits_nmf:
        M = pairwise_cos_matrix(fits_nmf)
        tri = M[np.triu_indices_from(M, k=1)]
        rows.append({"method": "nmf_lr",
                     "pairwise_cos_mean": float(tri.mean()),
                     "pairwise_cos_std":  float(tri.std()),
                     "pairwise_cos_min":  float(tri.min()),
                     "n_seeds": len(fits_nmf)})
        np.save(Path(args.out_dir) / "pairwise_cos_nmf.npy", M)
        plot_cos_heatmap(M, "NMF+LR", str(Path(args.figure_dir) / "pairwise_cos_heatmap_nmf.png"))

    if fits_drgp or fits_nmf:
        plot_v_variance(fits_drgp, fits_nmf, str(Path(args.figure_dir) / "v_variance_bars.png"))

    summary = pd.DataFrame(rows)
    summary.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print("\nSummary:")
    print(summary.round(4).to_string(index=False))
    print(f"\nFigures: {args.figure_dir}/")


if __name__ == "__main__":
    main()
