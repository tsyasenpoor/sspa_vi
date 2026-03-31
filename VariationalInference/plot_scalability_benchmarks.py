#!/usr/bin/env python
"""
Plot scalability benchmark results: wall time, peak memory, and predictive
performance vs dataset size (patients and cells).
================================================================================

Reads the aggregated CSV from aggregate_scalability_results.py and produces:
  1. Wall time + peak memory vs dataset size (dual x-axis: patients & cells)
  2. Test AUC vs dataset size by method and label
  3. Test F1  vs dataset size by method and label

Usage:
    python plot_scalability_benchmarks.py \
        --input  .../summary/all_metrics.csv \
        --output-dir .../summary/plots
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
    "drgp_unmasked_fix": "DRGP (unmasked, fix)",
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
    "DRGP": "#d62728",
    "Raw Classifier": "#1f77b4",
    "MF+Classifier": "#ff7f0e",
    "scHPF": "#2ca02c",
    "Spectra": "#9467bd",
}
FAMILY_MARKERS = {
    "DRGP": "D",
    "Raw Classifier": "o",
    "MF+Classifier": "s",
    "scHPF": "^",
    "Spectra": "v",
}


def _style(method: str):
    family = METHOD_FAMILY.get(method, "Other")
    return {
        "color": FAMILY_COLORS.get(family, "#7f7f7f"),
        "marker": FAMILY_MARKERS.get(family, "x"),
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


def _add_cell_axis(ax: plt.Axes, patient_to_cells: dict[int, int]):
    """Add a secondary x-axis showing cell counts."""
    if not patient_to_cells:
        return
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    patient_ticks = sorted(patient_to_cells.keys())
    ax2.set_xticks(patient_ticks)
    ax2.set_xticklabels(
        [f"{patient_to_cells[p]:,}" for p in patient_ticks],
        fontsize=9,
    )
    ax2.set_xlabel("Number of Cells (total)", fontsize=11)


# ---------------------------------------------------------------------------
# Benchmark: wall time + peak memory
# ---------------------------------------------------------------------------

def plot_benchmark_panel(df: pd.DataFrame, out_dir: Path, p2c: dict):
    """Side-by-side wall time and peak memory vs number of patients."""
    cols = ["ratio", "seed", "method", "wall_time_s", "peak_rss_mb"]
    bm = df[df["wall_time_s"].notna()][cols].drop_duplicates()
    if bm.empty:
        print("  No benchmark data to plot.")
        return

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

    grouped = (
        bm.groupby(["ratio", "seed", "group"])
        .agg({"wall_time_s": "first", "peak_rss_mb": "first"})
        .reset_index()
    )
    stats = (
        grouped.groupby(["group", "ratio"])
        .agg(
            time_mean=("wall_time_s", "mean"),
            time_std=("wall_time_s", "std"),
            mem_mean=("peak_rss_mb", "mean"),
            mem_std=("peak_rss_mb", "std"),
        )
        .reset_index()
    )
    stats["time_std"] = stats["time_std"].fillna(0)
    stats["mem_std"] = stats["mem_std"].fillna(0)

    bench_display = {
        "drgp_unmasked": "DRGP (unmasked)",
        "drgp_unmasked_fix": "DRGP (unmasked, fix)",
        "schpf": "scHPF",
        "spectra_sup": "Spectra",
        "baselines": "Baselines (raw)",
    }
    bench_colors = {
        "drgp_unmasked": "#d62728",
        "drgp_unmasked_fix": "#e377c2",
        "schpf": "#2ca02c",
        "spectra_sup": "#9467bd",
        "baselines": "#1f77b4",
    }
    bench_markers = {
        "drgp_unmasked": "D",
        "drgp_unmasked_fix": "d",
        "schpf": "^",
        "spectra_sup": "v",
        "baselines": "o",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for grp in sorted(stats["group"].unique()):
        g = stats[stats["group"] == grp].sort_values("ratio")
        n_patients = g["ratio"].values  # ratio column IS patient count now

        label = bench_display.get(grp, grp)
        color = bench_colors.get(grp, "#7f7f7f")
        marker = bench_markers.get(grp, "x")

        ax1.errorbar(
            n_patients, g["time_mean"] / 60, yerr=g["time_std"] / 60,
            label=label, color=color, marker=marker, markersize=8,
            capsize=4, linewidth=2, alpha=0.85,
        )
        ax2.errorbar(
            n_patients, g["mem_mean"] / 1024, yerr=g["mem_std"] / 1024,
            label=label, color=color, marker=marker, markersize=8,
            capsize=4, linewidth=2, alpha=0.85,
        )

    for ax, ylabel, title in [
        (ax1, "Wall Time (minutes)", "Wall Time vs Dataset Size"),
        (ax2, "Peak Memory (GB)", "Peak Memory vs Dataset Size"),
    ]:
        ax.set_xlabel("Number of Patients", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        _add_cell_axis(ax, p2c)

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
    ylim: tuple = (0.0, 1.0),
):
    """Generic: test <metric> vs number of patients, one subplot per label."""
    test = df[(df["split"] == "test") & df[metric].notna()].copy()
    if test.empty:
        print(f"  No test {metric.upper()} data to plot.")
        return

    labels = sorted(test["label"].unique())
    n_labels = len(labels)
    fig, axes = plt.subplots(1, n_labels, figsize=(7 * n_labels, 6), squeeze=False)

    for idx, label_name in enumerate(labels):
        ax = axes[0, idx]
        sub = test[test["label"] == label_name]
        stats = (
            sub.groupby(["method", "ratio"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        stats["std"] = stats["std"].fillna(0)

        for method in sorted(stats["method"].unique()):
            g = stats[stats["method"] == method].sort_values("ratio")
            n_patients = g["ratio"].values
            style = _style(method)
            ax.errorbar(
                n_patients, g["mean"], yerr=g["std"],
                label=_display(method), capsize=3, linewidth=1.5,
                markersize=6, alpha=0.8, **style,
            )

        ax.set_xlabel("Number of Patients", fontsize=12)
        ax.set_ylabel(f"Test {metric.upper()}", fontsize=12)
        ax.set_title(f"Test {metric.upper()} — {label_name}", fontsize=13, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        _add_cell_axis(ax, p2c)

    fig.tight_layout()
    fname = f"test_{metric}_vs_size"
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

    # Build patient-count -> cell-count mapping for dual x-axis
    p2c = _build_patient_to_cells(df)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Patients: {sorted(df['ratio'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Labels: {sorted(df['label'].unique())}")
    if p2c:
        print(f"Patient -> Cell mapping: { {int(k): v for k, v in sorted(p2c.items())} }")
    print()

    print("Plotting benchmark (time + memory) ...")
    plot_benchmark_panel(df, out_dir, p2c)

    print("Plotting test AUC ...")
    _plot_metric(df, "auc", out_dir, p2c, ylim=(0.4, 1.0))

    print("Plotting test F1 ...")
    _plot_metric(df, "f1", out_dir, p2c, ylim=(0.0, 1.0))

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
