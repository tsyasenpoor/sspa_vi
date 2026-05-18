"""
N1 analysis: K misspecification (standalone sweep).

Reuses A1 raw .npz files. Plots cos_mean vs K_fit for DRGP, NMF, PCA.
Also computes the "noise factor" count: number of fitted programs with
near-zero max activation, i.e. extras that the model effectively zeroed out.

Noise-factor count requires Theta_hat which A1 does NOT save. Per-program
posterior inclusion r_beta_col_sum is in A1 raw for DRGP (R_beta saved),
so we use that as a proxy: a fitted program with sum(r_beta_col > 0.5)
near zero is a "noise factor". For NMF/PCA we use the fitted Beta column
norm as a proxy (small norm => unused factor).

Outputs:
  results/aggregated/N1_K_misspec/{long.csv, summary.csv}
  figures/N1/{cos_vs_K_per_method.png, noise_factor_count_K.png}
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


PALETTE = {"drgp_unmasked": "#2b6aa6", "nmf_lr": "#aa5c2b", "pca_lr": "#666666"}


def aggregate(raw_dir: str, config_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    cond_meta = {
        i: {"K_fit": c["K_fit"], "pi_label": c.get("pi_label", float("nan"))}
        for i, c in enumerate(cfg["conditions"])
    }
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "cond*_seed*.npz"))):
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        rec = {k: d[k] for k in d.files}
        cidx = int(rec["condition_idx"])
        meta = cond_meta.get(cidx, {})
        if meta.get("pi_label") != 0.10:
            continue  # focus on default sparsity for the K sweep
        for method in cfg["methods"]:
            if method not in rec:
                continue
            m = rec[method].item() if rec[method].dtype == object else rec[method]
            if not isinstance(m, dict) or m.get("error"):
                continue
            # Noise factor count: DRGP uses R_beta column sums; baselines use
            # Beta_hat column norm.
            n_noise = float("nan")
            if "R_beta" in m and m["R_beta"] is not None:
                R = np.asarray(m["R_beta"])
                # Count fitted columns whose support is essentially empty
                col_active_count = (R > 0.5).sum(axis=0)
                n_noise = int((col_active_count < 2).sum())
            rows.append({
                "method": method,
                "seed": int(rec["seed"]),
                "K_fit": meta["K_fit"],
                "cos_mean": float(m.get("cos_mean", float("nan"))),
                "v_spearman": float(m.get("v_spearman", float("nan"))),
                "support_auprc": float(m.get("support_auprc", float("nan"))),
                "n_noise_factors": n_noise,
            })
    return pd.DataFrame(rows)


def plot_cos_vs_K(df: pd.DataFrame, out_path: str, K_true: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in [m for m in PALETTE if m in df["method"].unique()]:
        sub = df[df["method"] == method].dropna(subset=["cos_mean"])
        if sub.empty:
            continue
        g = sub.groupby("K_fit")["cos_mean"]
        med = g.median(); q25 = g.quantile(0.25); q75 = g.quantile(0.75)
        xs = med.index.values
        ax.plot(xs, med.values, "-o", color=PALETTE[method], label=method)
        ax.fill_between(xs, q25.values, q75.values, alpha=0.18, color=PALETTE[method])
    ax.axvline(K_true, color="black", linestyle="--", linewidth=0.8,
                label=f"K_true = {K_true}")
    ax.set_xlabel("K_fit"); ax.set_ylabel("cos_mean (matched)")
    ax.set_title("N1: factor recovery vs K_fit (pi=0.10)")
    ax.set_ylim(0.0, 1.05); ax.grid(alpha=0.25); ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def plot_noise_factors(df: pd.DataFrame, out_path: str, K_true: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sub = df[df["method"] == "drgp_unmasked"].dropna(subset=["n_noise_factors"])
    if sub.empty:
        return
    K_fits = sorted(sub["K_fit"].unique())
    medians = [sub[sub["K_fit"] == K]["n_noise_factors"].median() for K in K_fits]
    expected_extra = [max(K - K_true, 0) for K in K_fits]
    width = 0.35
    pos = np.arange(len(K_fits))
    ax.bar(pos - width/2, medians, width=width, label="observed noise factors",
            color=PALETTE["drgp_unmasked"], alpha=0.85)
    ax.bar(pos + width/2, expected_extra, width=width, label="K_fit - K_true",
            color="lightgray", edgecolor="black")
    ax.set_xticks(pos); ax.set_xticklabels([str(K) for K in K_fits])
    ax.set_xlabel("K_fit"); ax.set_ylabel("# DRGP factors with sum(r_beta>0.5) < 2")
    ax.set_title("N1: DRGP spike-and-slab cleans extra factors (pi=0.10)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/A1_factor_recovery")
    ap.add_argument("--config",  default="configs/A1_factor_recovery.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/N1_K_misspec")
    ap.add_argument("--figure-dir", default="figures/N1")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)
    print(f"Wrote {len(df)} rows to long.csv (filtered to pi=0.10)")

    g = df.groupby(["method", "K_fit"]).agg(
        cos_mean_med=("cos_mean", "median"),
        v_spearman_med=("v_spearman", "median"),
        n_noise_factors_med=("n_noise_factors", "median"),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print(g.round(3).to_string(index=False))

    plot_cos_vs_K(df, str(Path(args.figure_dir) / "cos_vs_K_per_method.png"))
    plot_noise_factors(df, str(Path(args.figure_dir) / "noise_factor_count_K.png"))
    print(f"\nFigures: {args.figure_dir}/")


if __name__ == "__main__":
    main()
