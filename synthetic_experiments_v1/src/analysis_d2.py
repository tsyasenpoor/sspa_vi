"""
D2 analysis: sample complexity.

Reuses C2 raw data. Plots three metrics vs n:
  - A1: cos_mean        (matched factor recovery)
  - B1: v_spearman      (program ranking)
  - C2: ood_auroc       (OOD predictive performance)

One line per method, IQR shading.

Output:
  results/aggregated/D2_sample_complexity/{long.csv, summary.csv}
  figures/D2/sample_complexity_three_panel.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the C2 aggregator -- both have the same npz layout and 'n_label' field.
from src.analysis_c2 import aggregate                                       # noqa: E402

PALETTE = {
    "drgp_unmasked": "#2b6aa6",
    "nmf_lr":        "#aa5c2b",
    "pca_lr":        "#666666",
    "plain_lr":      "#5b8a3a",
}


def _panel(ax, df, metric, ylabel, ylim=None):
    methods = [m for m in PALETTE if m in df["method"].unique()]
    for method in methods:
        sub = df[df["method"] == method].dropna(subset=[metric])
        if sub.empty:
            continue
        g = sub.groupby("n_label")[metric]
        med = g.median()
        q25 = g.quantile(0.25); q75 = g.quantile(0.75)
        xs = med.index.values
        ax.plot(xs, med.values, "-o", color=PALETTE[method], label=method)
        ax.fill_between(xs, q25.values, q75.values, alpha=0.18, color=PALETTE[method])
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 250, 500])
    ax.set_xticklabels(["50", "100", "250", "500"])
    ax.set_xlabel("n (training samples)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/C2_sample_complexity_ood")
    ap.add_argument("--config",  default="configs/C2_sample_complexity_ood.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/D2_sample_complexity")
    ap.add_argument("--figure-dir", default="figures/D2")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)

    g = df.groupby(["method", "n_label"]).agg(
        cos_mean_med=("cos_mean", "median"),
        v_spearman_med=("v_spearman", "median"),
        ood_auroc_med=("ood_auroc", "median"),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print(g.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _panel(axes[0], df, "cos_mean",   "matched cos sim",       ylim=(0.0, 1.05))
    _panel(axes[1], df, "v_spearman", "v_spearman (|v|)",      ylim=(0.0, 1.05))
    _panel(axes[2], df, "ood_auroc",  "OOD AUROC",             ylim=(0.5, 1.0))
    fig.suptitle("D2 — sample complexity (sweep over n)")
    plt.tight_layout()
    plt.savefig(Path(args.figure_dir) / "sample_complexity_three_panel.png", dpi=180)
    plt.close()
    print(f"Figure: {args.figure_dir}/sample_complexity_three_panel.png")


if __name__ == "__main__":
    main()
