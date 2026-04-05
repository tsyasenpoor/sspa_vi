#!/usr/bin/env python
"""
Plot scalability benchmark results: wall time, peak memory, and predictive
performance vs dataset size (patients and cells).
================================================================================

Reads the aggregated CSV from aggregate_scalability_results.py and produces:
  1. Wall time + peak memory vs dataset size (dual x-axis: patients & cells)
  2. Val AUC vs dataset size by method and label
  3. Val F1  vs dataset size by method and label

Usage:
    python /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/analysis-utils/plot_scalability_benchmarks.py \
        --input  /labs/Aguiar/SSPA_BRAY/results/scalability_benchmark_patient_level/all_metrics.csv \
        --output-dir /labs/Aguiar/SSPA_BRAY/results/scalability_benchmark_patient_level/plots
    python /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/analysis-utils/plot_scalability_benchmarks.py \
        --input  /labs/Aguiar/SSPA_BRAY/results/ibd_benchmark/summary/all_metrics.csv \
        --output-dir /labs/Aguiar/SSPA_BRAY/results/ibd_benchmark/summary/plots
    """

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
METHOD_DISPLAY = {
    "drgp_unmasked": "DRGP (unmasked)",
    "baselines/lr": "LR",
    "baselines/lrl": "LR-L1",
    "baselines/lrr": "LR-Ridge",
    "baselines/mflr": "MF+LR",
    "baselines/mflrl": "MF+LR-L1",
    "baselines/mflrr": "MF+LR-Ridge",
    "baselines/svm": "SVM",
    "schpf/schpf_lr": "scHPF+LR",
    "schpf/schpf_lrl": "scHPF+LR-L1",
    "schpf/schpf_lrr": "scHPF+LR-Ridge",
    "spectra_sup/spectra_lr": "Spectra+LR",
    "spectra_sup/spectra_lrl": "Spectra+LR-L1",
    "spectra_sup/spectra_lrr": "Spectra+LR-Ridge",
}

METHOD_FAMILY = {}
for _m in METHOD_DISPLAY:
    if _m.startswith("drgp"):
        METHOD_FAMILY[_m] = "DRGP"
    elif _m.startswith("baselines/mf"):
        METHOD_FAMILY[_m] = "MF+Classifier"
    elif _m.startswith("baselines"):
        METHOD_FAMILY[_m] = "Raw Classifier"
    elif _m.startswith("schpf"):
        METHOD_FAMILY[_m] = "scHPF"
    elif _m.startswith("spectra"):
        METHOD_FAMILY[_m] = "Spectra"

FAMILY_COLORS = {
    "DRGP": "#E69F00",           # Okabe-Ito orange (our method)
    "Raw Classifier": "#999999", # gray
    "MF+Classifier": "#666666",  # dark gray
    "scHPF": "#AAAAAA",          # light gray
    "Spectra": "#444444",        # charcoal
}
FAMILY_MARKERS = {
    "DRGP": "D",
    "Raw Classifier": "o",
    "MF+Classifier": "s",
    "scHPF": "^",
    "Spectra": "v",
}


def _display(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def _build_patient_to_cells(df: pd.DataFrame) -> dict[int, int]:
    """Build a mapping from n_patients -> mean total cells across seeds."""
    if "n_cells_total" not in df.columns:
        return {}
    sub = df[df["n_cells_total"].notna()][["ratio", "seed", "n_cells_total"]].drop_duplicates()
    if sub.empty:
        return {}
    return (
        sub.groupby("ratio")["n_cells_total"]
        .mean()
        .round()
        .astype(int)
        .to_dict()
    )


# ---------------------------------------------------------------------------
# Benchmark: wall time + peak memory
# ---------------------------------------------------------------------------

def plot_benchmark_panel(df: pd.DataFrame, out_dir: Path, p2c: dict):
    """Bar charts with seed-level scatter: one col per patient count, two rows."""
    cols = ["ratio", "seed", "method", "wall_time_s", "peak_rss_mb"]
    bm = df[df["wall_time_s"].notna()][cols].drop_duplicates()
    if bm.empty:
        print("  No benchmark data to plot.")
        return

    bench_display = {
        "drgp_unmasked": "DRGP",
        "schpf": "scHPF",
        "spectra_sup": "Spectra",
        "baselines": "Baselines",
    }
    bench_colors = {
        "drgp_unmasked": "#E69F00",
        "schpf": "#AAAAAA",
        "spectra_sup": "#444444",
        "baselines": "#999999",
    }

    def _bench_group(m):
        if m.startswith("drgp"):
            return m
        elif m.startswith("schpf"):
            return "schpf"
        elif m.startswith("spectra"):
            return "spectra_sup"
        elif m.startswith("baselines"):
            return "baselines"
        return m

    bm = bm.copy()
    bm["group"] = bm["method"].apply(_bench_group)

    # One entry per (seed, group, ratio)
    grouped = (
        bm.groupby(["ratio", "seed", "group"])
        .agg({"wall_time_s": "first", "peak_rss_mb": "first"})
        .reset_index()
    )

    ratios = sorted(grouped["ratio"].unique())
    all_groups = sorted(grouped["group"].unique(), key=lambda g: bench_display.get(g, g))
    n_ratios = len(ratios)

    metrics_info = [
        ("wall_time_s", 60.0, "Wall Time (minutes)"),
        ("peak_rss_mb", 1024.0, "Peak Memory (GB)"),
    ]

    fig, axes = plt.subplots(
        len(metrics_info), n_ratios,
        figsize=(4.5 * n_ratios, 5 * len(metrics_info)),
        squeeze=False, sharey="row",
    )

    for row, (col_name, divisor, ylabel) in enumerate(metrics_info):
        for col, ratio in enumerate(ratios):
            ax = axes[row, col]
            sub = grouped[grouped["ratio"] == ratio]
            groups_here = [g for g in all_groups if g in sub["group"].values]

            if not groups_here:
                ax.set_visible(False)
                continue

            positions = np.arange(len(groups_here))
            colors = [bench_colors.get(g, "#7f7f7f") for g in groups_here]

            for i, g in enumerate(groups_here):
                vals = sub[sub["group"] == g][col_name].values / divisor
                mean_val = vals.mean()
                ax.bar(
                    i, mean_val, width=0.55, color=colors[i],
                    edgecolor="black", linewidth=0.5, alpha=0.75,
                )
                # Overlay individual seed points with jitter
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(
                    np.full_like(vals, i) + jitter, vals,
                    color="black", s=18, zorder=3, alpha=0.7,
                    edgecolors="white", linewidths=0.4,
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(
                [bench_display.get(g, g) for g in groups_here],
                fontsize=8, rotation=35, ha="right",
            )
            ax.grid(True, axis="y", alpha=0.3)

            n_cells_str = f" ({p2c[ratio]:,} cells)" if ratio in p2c else ""
            if row == 0:
                ax.set_title(
                    f"{int(ratio)} patients{n_cells_str}",
                    fontsize=10, fontweight="bold",
                )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=11)

    fig.suptitle(
        "Wall Time and Peak Memory by Method and Dataset Size",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_time_memory.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "benchmark_time_memory.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved benchmark_time_memory.png/pdf")


# ---------------------------------------------------------------------------
# Performance: AUC / F1
# ---------------------------------------------------------------------------

def _plot_metric(
    df: pd.DataFrame, metric: str, out_dir: Path, p2c: dict,
):
    """Grouped box plots: one column per patient count, one row per label.

    Within each panel, methods are shown as individual boxes so the discrete
    dataset sizes are not connected by misleading trend lines.
    """
    val = df[(df["split"] == "val") & df[metric].notna()].copy()
    if val.empty:
        print(f"  No val {metric.upper()} data to plot.")
        return

    labels = sorted(val["label"].unique())
    ratios = sorted(val["ratio"].unique())
    n_labels = len(labels)
    n_ratios = len(ratios)

    fig, axes = plt.subplots(
        n_labels, n_ratios,
        figsize=(4.5 * n_ratios, 5 * n_labels),
        squeeze=False,
    )

    # Determine a consistent method order across all panels (sorted by family
    # then display name) so colours and positions are comparable.
    all_methods = sorted(val["method"].unique(), key=lambda m: (
        METHOD_FAMILY.get(m, "ZZZ"), _display(m),
    ))

    for row, label_name in enumerate(labels):
        for col, ratio in enumerate(ratios):
            ax = axes[row, col]
            sub = val[(val["label"] == label_name) & (val["ratio"] == ratio)]

            # Only include methods that have data for this (label, ratio).
            methods_here = [m for m in all_methods if m in sub["method"].values]
            box_data = [sub[sub["method"] == m][metric].values for m in methods_here]
            families = [METHOD_FAMILY.get(m, "Other") for m in methods_here]
            colors = [FAMILY_COLORS.get(f, "#7f7f7f") for f in families]

            if not box_data:
                ax.set_visible(False)
                continue

            positions = np.arange(len(methods_here))
            bp = ax.boxplot(
                box_data, positions=positions, widths=0.55,
                patch_artist=True, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white",
                               markeredgecolor="black", markersize=4),
                medianprops=dict(color="black", linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.5),
            )
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)

            ax.set_xticks(positions)
            ax.set_xticklabels(
                [_display(m) for m in methods_here],
                fontsize=7, rotation=45, ha="right",
            )
            ax.grid(True, axis="y", alpha=0.3)

            n_cells_str = f" ({p2c[ratio]:,} cells)" if ratio in p2c else ""
            ax.set_title(
                f"{int(ratio)} patients{n_cells_str}",
                fontsize=10, fontweight="bold",
            )
            if col == 0:
                ax.set_ylabel(f"Val {metric.upper()}", fontsize=11)

        # Row label on the left-most axis
        axes[row, 0].annotate(
            label_name, xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=13, fontweight="bold", rotation=90,
            va="center", ha="center",
        )

    fig.suptitle(
        f"Val {metric.upper()} by Method and Dataset Size",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fname = f"val_{metric}_vs_size"
    fig.savefig(out_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png/pdf")


# ---------------------------------------------------------------------------
# Bar charts for single-dataset benchmarks (no ratio/scalability dimension)
# ---------------------------------------------------------------------------

BENCH_DISPLAY = {
    "drgp_unmasked": "DRGP",
    "schpf": "scHPF",
    "spectra_sup": "Spectra",
    "baselines": "Baselines",
}
BENCH_COLORS = {
    "drgp_unmasked": "#E69F00",      # orange (our method)
    "schpf": "#AAAAAA",              # light gray
    "spectra_sup": "#444444",        # charcoal
    "baselines": "#999999",          # gray
}


def _bench_group(m: str) -> str:
    if m.startswith("drgp"):
        return m
    elif m.startswith("schpf"):
        return "schpf"
    elif m.startswith("spectra"):
        return "spectra_sup"
    elif m.startswith("baselines"):
        return "baselines"
    return m


def plot_benchmark_bars(df: pd.DataFrame, out_dir: Path):
    """Bar chart with seed-level scatter for wall time and peak memory."""
    cols = ["seed", "method", "wall_time_s", "peak_rss_mb"]
    bm = df[df["wall_time_s"].notna()][[c for c in cols if c in df.columns]].drop_duplicates()
    if bm.empty:
        print("  No benchmark data to plot.")
        return

    bm = bm.copy()
    bm["group"] = bm["method"].apply(_bench_group)

    # One entry per (seed, group)
    grouped = (
        bm.groupby(["seed", "group"])
        .agg({"wall_time_s": "first", "peak_rss_mb": "first"})
        .reset_index()
    )

    groups_sorted = sorted(grouped["group"].unique(), key=lambda g: BENCH_DISPLAY.get(g, g))
    colors = [BENCH_COLORS.get(g, "#7f7f7f") for g in groups_sorted]
    x = np.arange(len(groups_sorted))

    metrics_info = [
        ("wall_time_s", 60.0, "Wall Time (minutes)", "Wall Time"),
        ("peak_rss_mb", 1024.0, "Peak Memory (GB)", "Peak Memory"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (col_name, divisor, ylabel, title) in zip(axes, metrics_info):
        for i, g in enumerate(groups_sorted):
            vals = grouped[grouped["group"] == g][col_name].values / divisor
            mean_val = vals.mean()
            ax.bar(i, mean_val, width=0.55, color=colors[i],
                   edgecolor="black", linewidth=0.5, alpha=0.75)
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color="black", s=18, zorder=3, alpha=0.7,
                       edgecolors="white", linewidths=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels([BENCH_DISPLAY.get(g, g) for g in groups_sorted], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Benchmark Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "benchmark_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved benchmark_bars.png/pdf")


def _plot_metric_bars(df: pd.DataFrame, metric: str, out_dir: Path):
    """Box plots of val <metric> by method with seed-level scatter, one subplot per label."""
    val = df[(df["split"] == "val") & df[metric].notna()].copy()
    if val.empty:
        print(f"  No val {metric.upper()} data to plot.")
        return

    labels = sorted(val["label"].unique())
    n_labels = len(labels)
    fig, axes = plt.subplots(1, n_labels, figsize=(6 * n_labels, 5), squeeze=False)

    for idx, label_name in enumerate(labels):
        ax = axes[0, idx]
        sub = val[val["label"] == label_name]

        # Sort methods by family then display name
        methods = sorted(sub["method"].unique(), key=lambda m: (
            METHOD_FAMILY.get(m, "ZZZ"), _display(m),
        ))
        box_data = [sub[sub["method"] == m][metric].values for m in methods]
        families = [METHOD_FAMILY.get(m, "Other") for m in methods]
        colors = [FAMILY_COLORS.get(f, "#7f7f7f") for f in families]

        positions = np.arange(len(methods))
        bp = ax.boxplot(
            box_data, positions=positions, widths=0.55,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=4),
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        # Overlay individual seed points
        rng = np.random.default_rng(42)
        for i, vals in enumerate(box_data):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full_like(vals, i) + jitter, vals,
                color="black", s=18, zorder=3, alpha=0.7,
                edgecolors="white", linewidths=0.4,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [_display(m) for m in methods],
            fontsize=8, rotation=35, ha="right",
        )
        ax.set_ylabel(f"Val {metric.upper()}", fontsize=12)
        ax.set_title(f"{label_name}", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Val {metric.upper()} by Method", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fname = f"val_{metric}_bars"
    fig.savefig(out_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png/pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot scalability benchmark results")
    p.add_argument(
        "--input", "-i",
        default="/labs/Aguiar/SSPA_BRAY/results/scalability_benchmark_patient_level/summary/all_metrics.csv",
    )
    p.add_argument(
        "--output-dir", "-o",
        default="/labs/Aguiar/SSPA_BRAY/results/scalability_benchmark_patient_level/summary/plots",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize label names
    label_map = {
        "covid-19 severity": "Severity",
        "severity": "Severity",
        "outcome": "Outcome",
    }
    df["label"] = df["label"].str.lower().map(lambda x: label_map.get(x, x))

    # Detect mode: scalability (multiple ratios) vs single-dataset (no ratio)
    valid_ratios = df["ratio"].dropna().unique()
    is_scalability = len(valid_ratios) > 1

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Labels: {sorted(df['label'].unique())}")
    print(f"Mode: {'scalability' if is_scalability else 'single-dataset (bar charts)'}")
    print()

    if is_scalability:
        p2c = _build_patient_to_cells(df)
        if p2c:
            print(f"Patient -> Cell mapping: { {int(k): v for k, v in sorted(p2c.items())} }")

        print("Plotting benchmark (time + memory) ...")
        plot_benchmark_panel(df, out_dir, p2c)

        print("Plotting val AUC ...")
        _plot_metric(df, "auc", out_dir, p2c)

        print("Plotting val F1 ...")
        _plot_metric(df, "f1", out_dir, p2c)
    else:
        print("Plotting benchmark bars ...")
        plot_benchmark_bars(df, out_dir)

        print("Plotting val AUC bars ...")
        _plot_metric_bars(df, "auc", out_dir)

        print("Plotting val F1 bars ...")
        _plot_metric_bars(df, "f1", out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
