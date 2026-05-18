#!/usr/bin/env python3
"""
Quick figures for compare_pbmc outputs.

For now: Axis C stability box plot. Axes A and B added as their CSVs land.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_ROOT = Path("/labs/Aguiar/SSPA_BRAY/results/method_comparison")
TABLES = OUT_ROOT / "tables"
FIGS = OUT_ROOT / "figures"

PALETTE = {
    "drgp_masked":   "#2b6aa6",
    "drgp_unmasked": "#5fa8d3",
    "gsva_lr":       "#1b9e77",
    "gsva_lr_pb":    "#1b9e77",
    "muster_lr":     "#d97b00",
    "muster_lr_pb":  "#d97b00",
    "muster_phase1": "#d95f02",
    "muster_internal_cv": "#cccccc",
}


def plot_axis_c(subset: str) -> None:
    p = TABLES / f"axis_c_stability_{subset}.csv"
    if not p.exists():
        print(f"missing: {p}")
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, outcome in zip(axes, ("t2dm", "cvda")):
        sub = df[df["outcome"] == outcome]
        methods = sorted(sub["method"].unique())
        data = [sub[sub["method"] == m]["jaccard"].values for m in methods]
        bp = ax.boxplot(
            data, tick_labels=methods, showmeans=True, showfliers=True,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=5))
        for box, m in zip(bp["boxes"], methods):
            box.set_facecolor(PALETTE.get(m, "#888"))
            box.set_alpha(0.7)
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data):
            if len(vals) == 0:
                continue
            x = (i + 1) + rng.uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(x, vals, s=8, alpha=0.35, color="black", zorder=3)
        ax.set_title(f"{subset} — {outcome}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
        for n, vals in zip(methods, data):
            med = np.median(vals) if len(vals) else float("nan")
            print(f"  {subset}/{outcome}: {n} median Jaccard@50 = {med:.3f}  (n_pairs={len(vals)})")
    axes[0].set_ylabel("Pairwise Jaccard@50 (top-K signature overlap)")
    fig.suptitle("Axis C — stability of identified signature across replicates",
                 y=1.02)
    fig.tight_layout()
    out = FIGS / f"axis_c_stability_{subset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_axis_a_per_cell(subset: str) -> None:
    p = TABLES / f"axis_a_per_cell_{subset}.csv"
    if not p.exists():
        print(f"missing: {p}")
        return
    df = pd.read_csv(p)
    # Restrict to held-out / test / CV splits for honest comparison
    df = df[df["split"].isin(("test", "cv", "internal"))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, outcome in zip(axes, ("t2dm", "cvda")):
        sub = df[df["outcome"] == outcome]
        methods = sorted(sub["method"].unique())
        data = [sub[sub["method"] == m]["auc"].values for m in methods]
        bp = ax.boxplot(
            data, tick_labels=methods, showmeans=True, showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5))
        for box, m in zip(bp["boxes"], methods):
            box.set_facecolor(PALETTE.get(m, "#888"))
            box.set_alpha(0.7)
        ax.set_title(f"{subset} — {outcome}")
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, ls="--", c="gray", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        for label in ax.get_xticklabels():
            label.set_rotation(20)
    axes[0].set_ylabel("AUC (CV / held-out)")
    fig.suptitle(f"Axis A — predictive AUC (per-cell, {subset})", y=1.02)
    fig.tight_layout()
    out = FIGS / f"axis_a_per_cell_{subset}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_axis_b(subset: str) -> None:
    p = TABLES / f"axis_b_pathway_overlap_{subset}.csv"
    if not p.exists():
        print(f"missing: {p}")
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, outcome in zip(axes, ("t2dm", "cvda")):
        sub = df[df["outcome"] == outcome]
        pivoted = sub.pivot_table(
            index="K", columns="method_b", values="jaccard", aggfunc="mean")
        pivoted.plot(kind="bar", ax=ax, alpha=0.8, edgecolor="black")
        ax.set_title(f"{subset} — {outcome}")
        ax.set_xlabel("Top-K")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Jaccard vs drgp_masked top-K")
    fig.suptitle(f"Axis B — pathway-importance overlap with DRGP-masked, {subset}",
                 y=1.02)
    fig.tight_layout()
    out = FIGS / f"axis_b_pathway_overlap_{subset}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="allPBMC")
    ap.add_argument("--axes", default="A,B,C")
    args = ap.parse_args()
    axes = set(a.strip().upper() for a in args.axes.split(","))
    if "C" in axes:
        plot_axis_c(args.subset)
    if "A" in axes:
        plot_axis_a_per_cell(args.subset)
    if "B" in axes:
        plot_axis_b(args.subset)


if __name__ == "__main__":
    main()
