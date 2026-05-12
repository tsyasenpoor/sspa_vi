#!/usr/bin/env python
"""
Plot scalability benchmark results: wall time, peak memory, and predictive
performance vs dataset size (patients and cells).
================================================================================

Reads the aggregated CSV from aggregate_scalability_results.py and produces:
  1. Wall time + peak memory vs dataset size (dual x-axis: patients & cells)
  2. Val AUC vs dataset size by method and label
  3. Val F1  vs dataset size by method and label

When the CSV contains a ``cell_type`` column (biorepo layout), produces instead:
  1. AUC boxplots per label, x-axis = cell type
  2. F1  boxplots per label, x-axis = cell type

Usage:
    python /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/analysis-utils/plot_scalability_benchmarks.py \
        --input  /labs/Aguiar/SSPA_BRAY/results/scalability_benchmark_cell_level_no_split_seeds/all_metrics.csv \
        --output-dir /labs/Aguiar/SSPA_BRAY/results/scalability_benchmark_cell_level_no_split_seeds/plots
    python /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/analysis-utils/plot_scalability_benchmarks.py \
        --input  /labs/Aguiar/SSPA_BRAY/results/ibd_benchmark/summary/all_metrics.csv \
        --output-dir /labs/Aguiar/SSPA_BRAY/results/ibd_benchmark/summary/plots
    # Biorepo per-cell-type:
    python /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/analysis-utils/plot_scalability_benchmarks.py \
        --input  /labs/Aguiar/SSPA_BRAY/results/biorepo_vi_unmasked/all_metrics.csv \
        --output-dir /labs/Aguiar/SSPA_BRAY/results/biorepo_vi_unmasked/plots
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
    "drgp_masked": "DRGP (masked)",
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
        "drgp_unmasked": "DRGP (unmasked)",
        "drgp_masked": "DRGP (masked)",
        "schpf": "scHPF",
        "spectra_sup": "Spectra",
        "baselines": "Baselines",
    }
    bench_colors = {
        "drgp_unmasked": "#E69F00",
        "drgp_masked": "#D55E00",
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
                fontsize=14, rotation=35, ha="right",
            )
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=14)

            n_cells_str = f" ({p2c[ratio]:,} cells)" if ratio in p2c else ""
            if row == 0:
                ax.set_title(
                    f"{int(ratio)} patients{n_cells_str}",
                    fontsize=16, fontweight="bold",
                )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=16)

    fig.suptitle(
        "Wall Time and Peak Memory by Method and Dataset Size",
        fontsize=18, fontweight="bold", y=1.01,
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
        figsize=(6 * n_ratios, 6.5 * n_labels),
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

            # Build positions with gaps between family groups
            positions = []
            pos = 0
            for i, fam in enumerate(families):
                if i > 0 and fam != families[i - 1]:
                    pos += 0.6  # gap between groups
                positions.append(pos)
                pos += 1

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

            # Variant labels (e.g. "LR", "LR-L1") on tick, rotated 45 deg
            display_names = [_display(m) for m in methods_here]
            variant_labels = [_VARIANT_LABEL.get(d, d) for d in display_names]
            ax.set_xticks(positions)
            ax.set_xticklabels(variant_labels, fontsize=13, rotation=45, ha="right")

            # Group labels: one centered label per family group below the variant labels
            groups = []
            g_start = 0
            for i in range(1, len(families)):
                if families[i] != families[i - 1]:
                    groups.append((families[g_start], positions[g_start], positions[i - 1]))
                    g_start = i
            groups.append((families[g_start], positions[g_start], positions[-1]))

            trans = ax.get_xaxis_transform()
            for fam, x_start, x_end in groups:
                x_mid = (x_start + x_end) / 2
                fam_label = _FAMILY_DISPLAY.get(fam, fam)
                ax.text(
                    x_mid, -0.34, fam_label,
                    transform=trans, ha="center", va="top",
                    fontsize=14, fontweight="bold",
                )
                # Bracket line
                if x_start != x_end:
                    ax.annotate(
                        "", xy=(x_start - 0.25, -0.26), xytext=(x_end + 0.25, -0.26),
                        xycoords=trans, textcoords=trans,
                        arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8),
                    )

            ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="y", labelsize=13)

            n_cells_str = f" ({p2c[ratio]:,} cells)" if ratio in p2c else ""
            ax.set_title(
                f"{int(ratio)} patients{n_cells_str}",
                fontsize=16, fontweight="bold",
            )
            if col == 0:
                ax.set_ylabel(f"Val {metric.upper()}", fontsize=15)

        # Row label on the left-most axis
        axes[row, 0].annotate(
            label_name, xy=(-0.28, 0.5), xycoords="axes fraction",
            fontsize=18, fontweight="bold", rotation=90,
            va="center", ha="center",
        )

    fig.suptitle(
        f"Val {metric.upper()} by Method and Dataset Size",
        fontsize=20, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fname = f"val_{metric}_vs_size"
    fig.savefig(out_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png/pdf")


# ---------------------------------------------------------------------------
# Bar charts for single-dataset benchmarks (no ratio/scalability dimension)
# ---------------------------------------------------------------------------

BENCH_DISPLAY = {
    "drgp_unmasked": "DRGP (unmasked)",
    "drgp_masked": "DRGP (masked)",
    "schpf": "scHPF",
    "spectra_sup": "Spectra",
    "baselines": "Baselines",
}
BENCH_COLORS = {
    "drgp_unmasked": "#E69F00",      # orange (our method)
    "drgp_masked": "#D55E00",        # vermillion (our method, masked)
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

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

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
        ax.set_xticklabels([BENCH_DISPLAY.get(g, g) for g in groups_sorted],
                           fontsize=15, rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=17, fontweight="bold")
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Benchmark Comparison", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "benchmark_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved benchmark_bars.png/pdf")


_VARIANT_LABEL = {
    "DRGP (unmasked)": "unmasked",
    "DRGP (masked)": "masked",
    "LR": "LR",
    "LR-L1": "LR-L1",
    "LR-Ridge": "LR-Ridge",
    "MF+LR": "LR",
    "MF+LR-L1": "LR-L1",
    "MF+LR-Ridge": "LR-Ridge",
    "SVM": "SVM",
    "Spectra+LR": "LR",
    "Spectra+LR-L1": "LR-L1",
    "Spectra+LR-Ridge": "LR-Ridge",
    "scHPF+LR": "LR",
    "scHPF+LR-L1": "LR-L1",
    "scHPF+LR-Ridge": "LR-Ridge",
}

_FAMILY_DISPLAY = {
    "DRGP": "DRGP",
    "Raw Classifier": "Raw",
    "MF+Classifier": "MF+Clf",
    "Spectra": "Spectra",
    "scHPF": "scHPF",
}


def _plot_metric_bars(df: pd.DataFrame, metric: str, out_dir: Path):
    """Box plots of val <metric> by method with seed-level scatter, one subplot per label.

    Uses grouped x-axis: classifier variant on the tick, family name as a
    centered group label below — so 'Spectra' and 'scHPF' appear once each.
    """
    val = df[(df["split"] == "val") & df[metric].notna()].copy()
    if val.empty:
        print(f"  No val {metric.upper()} data to plot.")
        return

    labels = sorted(val["label"].unique())
    n_labels = len(labels)
    fig, axes = plt.subplots(1, n_labels, figsize=(8 * n_labels, 5.5), squeeze=False)

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

        # Build positions with gaps between family groups
        positions = []
        pos = 0
        for i, fam in enumerate(families):
            if i > 0 and fam != families[i - 1]:
                pos += 0.6  # gap between groups
            positions.append(pos)
            pos += 1

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
                np.full_like(vals, positions[i]) + jitter, vals,
                color="black", s=18, zorder=3, alpha=0.7,
                edgecolors="white", linewidths=0.4,
            )

        # Variant labels (e.g. "LR", "LR-L1") on tick, rotated 45 deg
        display_names = [_display(m) for m in methods]
        variant_labels = [_VARIANT_LABEL.get(d, d) for d in display_names]
        ax.set_xticks(positions)
        ax.set_xticklabels(variant_labels, fontsize=11, rotation=45, ha="right")

        # Group labels: one centered label per family group below the variant labels
        # Collect group spans
        groups = []
        g_start = 0
        for i in range(1, len(families)):
            if families[i] != families[i - 1]:
                groups.append((families[g_start], positions[g_start], positions[i - 1]))
                g_start = i
        groups.append((families[g_start], positions[g_start], positions[-1]))

        trans = ax.get_xaxis_transform()
        for fam, x_start, x_end in groups:
            x_mid = (x_start + x_end) / 2
            fam_label = _FAMILY_DISPLAY.get(fam, fam)
            ax.text(
                x_mid, -0.22, fam_label,
                transform=trans, ha="center", va="top",
                fontsize=12, fontweight="bold",
            )
            # Bracket line
            if x_start != x_end:
                ax.annotate(
                    "", xy=(x_start - 0.25, -0.14), xytext=(x_end + 0.25, -0.14),
                    xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.8),
                )

        ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
        ax.set_ylabel(f"Val {metric.upper()}", fontsize=14)
        ax.set_title(f"{label_name}", fontsize=15, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Val {metric.upper()} by Method", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fname = f"val_{metric}_bars"
    fig.savefig(out_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png/pdf")


# ---------------------------------------------------------------------------
# Biorepo per-cell-type plots
# ---------------------------------------------------------------------------

# Preferred display order for cell types (allPBMC last as the aggregate)
_CELLTYPE_ORDER = ["cd4t", "cd8t", "Bcell", "NKcell", "monocytes", "allPBMC"]
_CELLTYPE_DISPLAY = {
    "cd4t": "CD4 T",
    "cd8t": "CD8 T",
    "Bcell": "B cell",
    "NKcell": "NK cell",
    "monocytes": "Monocytes",
    "allPBMC": "All PBMC",
}
_CELLTYPE_COLORS = {
    "cd4t":      "#E69F00",
    "cd8t":      "#56B4E9",
    "Bcell":     "#009E73",
    "NKcell":    "#F0E442",
    "monocytes": "#0072B2",
    "allPBMC":   "#CC79A7",
}


def _plot_metric_by_celltype(df: pd.DataFrame, metric: str, out_dir: Path):
    """Box plots of {metric} by cell type, one subplot per label.

    Shows val and test splits side-by-side within each cell-type group.
    Seed-level points overlaid as scatter.
    """
    sub = df[df[metric].notna() & df["split"].isin(["val", "test"])].copy()
    if sub.empty:
        print(f"  No {metric.upper()} data (val/test) to plot.")
        return

    labels = sorted(sub["label"].unique())
    n_labels = len(labels)

    # Determine cell-type order: use preferred order for known types, append unknowns
    all_cts = sub["cell_type"].unique().tolist()
    ct_order = [c for c in _CELLTYPE_ORDER if c in all_cts] + \
               sorted(c for c in all_cts if c not in _CELLTYPE_ORDER)

    splits = ["val", "test"]
    n_splits = len(splits)
    rng = np.random.default_rng(42)

    panel_w = max(7, 2.2 * len(ct_order) * n_splits)
    fig, axes = plt.subplots(
        n_labels, 1,
        figsize=(panel_w, 5.5 * n_labels),
        squeeze=False,
    )

    for col, label_name in enumerate(labels):
        ax = axes[col, 0]
        lsub = sub[sub["label"] == label_name]

        # Build positions: pairs (val, test) per cell type, with a gap between types
        positions = []
        box_data = []
        x_tick_pos = []
        x_tick_labels = []
        colors = []
        pos = 0.0
        split_labels_used = []

        for ct in ct_order:
            ct_sub = lsub[lsub["cell_type"] == ct]
            if ct_sub.empty:
                continue
            ct_color = _CELLTYPE_COLORS.get(ct, "#999999")
            ct_display = _CELLTYPE_DISPLAY.get(ct, ct)

            ct_positions = []
            for sp in splits:
                vals = ct_sub[ct_sub["split"] == sp][metric].values
                if len(vals) == 0:
                    continue
                box_data.append(vals)
                positions.append(pos)
                ct_positions.append(pos)
                colors.append(ct_color)
                split_labels_used.append(sp)
                pos += 1.0

            if ct_positions:
                x_tick_pos.append(np.mean(ct_positions))
                x_tick_labels.append(ct_display)
                pos += 0.6  # gap between cell types

        if not box_data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            box_data, positions=positions, widths=0.6,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=4),
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)

        # Seed-level scatter
        for i, vals in enumerate(box_data):
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                np.full_like(vals, positions[i], dtype=float) + jitter, vals,
                color="black", s=20, zorder=3, alpha=0.7,
                edgecolors="white", linewidths=0.4,
            )

        # Val / test split marker annotations below x-axis
        trans = ax.get_xaxis_transform()
        pos_idx = 0
        for ct in ct_order:
            ct_sub = lsub[lsub["cell_type"] == ct]
            if ct_sub.empty:
                continue
            for sp in splits:
                if ct_sub[ct_sub["split"] == sp].empty:
                    continue
                ax.text(
                    positions[pos_idx], -0.10, sp,
                    transform=trans, ha="center", va="top",
                    fontsize=9, color="0.4",
                )
                pos_idx += 1

        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels(x_tick_labels, fontsize=13, rotation=30, ha="right")
        ax.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)
        ax.set_ylabel(metric.upper(), fontsize=14)
        ax.set_title(label_name, fontsize=15, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{metric.upper()} by Cell Type", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fname = f"celltype_{metric}"
    fig.savefig(out_dir / f"{fname}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png/pdf")


def _plot_metric_by_celltype_combined(df: pd.DataFrame, metric: str, out_dir: Path):
    """Single panel: all labels overlaid, x=cell type, y=metric (test split only).

    Useful for a quick one-figure summary across labels.
    """
    sub = df[(df["split"] == "test") & df[metric].notna()].copy()
    if sub.empty:
        print(f"  No test {metric.upper()} data to plot.")
        return

    labels = sorted(sub["label"].unique())
    all_cts = sub["cell_type"].unique().tolist()
    ct_order = [c for c in _CELLTYPE_ORDER if c in all_cts] + \
               sorted(c for c in all_cts if c not in _CELLTYPE_ORDER)

    label_colors = plt.cm.tab10(np.linspace(0, 0.9, len(labels)))
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(ct_order) * len(labels)), 5))

    n_labels = len(labels)
    width = 0.7 / n_labels
    x_base = np.arange(len(ct_order))

    for li, label_name in enumerate(labels):
        lsub = sub[sub["label"] == label_name]
        offsets = (np.arange(n_labels) - (n_labels - 1) / 2) * width
        color = label_colors[li]

        vals_list = []
        for ct in ct_order:
            vals = lsub[lsub["cell_type"] == ct][metric].values
            vals_list.append(vals)

        bp = ax.boxplot(
            vals_list,
            positions=x_base + offsets[li],
            widths=width * 0.85,
            patch_artist=True, showmeans=False,
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_label(label_name)

        for xi, vals in enumerate(vals_list):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-width * 0.3, width * 0.3, size=len(vals))
            ax.scatter(
                np.full_like(vals, x_base[xi] + offsets[li], dtype=float) + jitter,
                vals, color="black", s=16, zorder=3, alpha=0.65,
                edgecolors="none",
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [_CELLTYPE_DISPLAY.get(c, c) for c in ct_order],
        fontsize=13, rotation=30, ha="right",
    )
    ax.set_ylabel(f"Test {metric.upper()}", fontsize=14)
    ax.set_title(f"Test {metric.upper()} by Cell Type and Label", fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    # Legend: one entry per label
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=label_colors[li], alpha=0.7, label=lab)
        for li, lab in enumerate(labels)
    ]
    ax.legend(handles=handles, title="Label", fontsize=11, title_fontsize=11)

    fig.tight_layout()
    fname = f"celltype_{metric}_combined"
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

    # Detect mode
    is_biorepo = "cell_type" in df.columns and df["cell_type"].notna().any()
    valid_ratios = df["ratio"].dropna().unique()
    is_scalability = len(valid_ratios) > 1

    if is_biorepo:
        mode = "biorepo (per cell type)"
    elif is_scalability:
        mode = "scalability"
    else:
        mode = "single-dataset (bar charts)"

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Labels: {sorted(df['label'].unique())}")
    if is_biorepo:
        print(f"Cell types: {sorted(df['cell_type'].dropna().unique())}")
    print(f"Mode: {mode}")
    print()

    if is_biorepo:
        print("Plotting AUC by cell type ...")
        _plot_metric_by_celltype(df, "auc", out_dir)

        print("Plotting F1 by cell type ...")
        _plot_metric_by_celltype(df, "f1", out_dir)

        print("Plotting combined AUC (test, all labels) ...")
        _plot_metric_by_celltype_combined(df, "auc", out_dir)

        print("Plotting combined F1 (test, all labels) ...")
        _plot_metric_by_celltype_combined(df, "f1", out_dir)

    elif is_scalability:
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
