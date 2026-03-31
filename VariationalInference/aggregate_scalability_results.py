#!/usr/bin/env python
"""
Aggregate scalability benchmark results across all methods, ratios, and seeds.
==============================================================================

Walks the results directory tree and collects:
  - Performance metrics (accuracy, F1, AUC, precision, recall)
  - Benchmarking metrics (wall time, peak memory)

Outputs a unified CSV for analysis.

Usage:
    python aggregate_scalability_results.py \
        --results-root /labs/Aguiar/SSPA_BRAY/results/scalability_benchmark/methods \
        --output /labs/Aguiar/SSPA_BRAY/results/scalability_benchmark/summary/all_metrics.csv
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_benchmark(method_dir: Path) -> dict:
    """Load benchmark.json if it exists."""
    bm_path = method_dir / "benchmark.json"
    if bm_path.exists():
        with open(bm_path) as f:
            return json.load(f)
    return {}


def _load_cell_counts(seed_dir: Path) -> dict:
    """Extract cell counts from vi_summary.json.gz in any DRGP dir under seed_dir."""
    for method_dir in seed_dir.iterdir():
        if not method_dir.is_dir():
            continue
        summary_path = method_dir / "vi_summary.json.gz"
        if summary_path.exists():
            with gzip.open(summary_path, "rt") as f:
                d = json.load(f)
            shapes = d.get("data_shapes", {})
            if shapes:
                return {
                    "n_cells_train": shapes.get("n_train"),
                    "n_cells_val": shapes.get("n_val"),
                    "n_cells_test": shapes.get("n_test"),
                    "n_cells_total": sum(
                        shapes.get(k, 0) for k in ("n_train", "n_val", "n_test")
                    ),
                }
    return {}


def _parse_drgp(method_dir: Path, method_name: str, ratio: float, seed: int) -> list[dict]:
    """Parse DRGP results from vi_metrics.csv."""
    rows = []
    metrics_path = method_dir / "vi_metrics.csv"
    if not metrics_path.exists():
        return rows

    bm = _load_benchmark(method_dir)
    df = pd.read_csv(metrics_path)

    for _, row in df.iterrows():
        rows.append({
            "ratio": ratio,
            "seed": seed,
            "method": method_name,
            "label": row.get("label", "unknown"),
            "split": row.get("split", "unknown"),
            "accuracy": row.get("accuracy"),
            "f1": row.get("f1"),
            "auc": row.get("auc"),
            "precision": row.get("precision"),
            "recall": row.get("recall"),
            "wall_time_s": bm.get("wall_time_seconds"),
            "peak_rss_mb": bm.get("peak_rss_mb"),
        })
    return rows


def _parse_spectra_or_schpf_baselines(
    baseline_dir: Path, fit_dir_name: str, method_name: str, ratio: float, seed: int
) -> list[dict]:
    """Parse Spectra or scHPF baseline results from summary JSON."""
    rows = []

    # Find the summary JSON
    for summary_name in ["spectra_baselines_summary.json", "schpf_baselines_summary.json"]:
        summary_path = baseline_dir / summary_name
        if summary_path.exists():
            break
    else:
        return rows

    # Load benchmark from the fitting directory (sibling of baselines dir)
    fit_dir = baseline_dir.parent / fit_dir_name
    bm = _load_benchmark(fit_dir) if fit_dir.exists() else {}

    with open(summary_path) as f:
        summary = json.load(f)

    results = summary.get("results", {})
    for key, res in results.items():
        # Parse key like "spectra_lr_severity" or "schpf_lrl_outcome"
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            alg_name, label = parts
        else:
            alg_name, label = key, "unknown"

        for split_name in ["val", "test"]:
            if split_name in res:
                m = res[split_name]
                rows.append({
                    "ratio": ratio,
                    "seed": seed,
                    "method": f"{method_name}/{alg_name}",
                    "label": label,
                    "split": split_name,
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "auc": m.get("roc_auc"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "wall_time_s": bm.get("wall_time_seconds"),
                    "peak_rss_mb": bm.get("peak_rss_mb"),
                })

        if "train_accuracy" in res:
            rows.append({
                "ratio": ratio,
                "seed": seed,
                "method": f"{method_name}/{alg_name}",
                "label": label,
                "split": "train",
                "accuracy": res["train_accuracy"],
                "f1": None, "auc": None, "precision": None, "recall": None,
                "wall_time_s": bm.get("wall_time_seconds"),
                "peak_rss_mb": bm.get("peak_rss_mb"),
            })

    return rows


def _parse_baselines(baselines_dir: Path, ratio: float, seed: int) -> list[dict]:
    """Parse run_baselines.py results from summary.json per label subdirectory."""
    rows = []
    bm = _load_benchmark(baselines_dir)

    for label_dir in baselines_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label_tag = label_dir.name  # "severity" or "outcome"

        summary_path = label_dir / "summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        results = summary.get("results", {})
        for alg_name, res in results.items():
            if res.get("status") != "success":
                rows.append({
                    "ratio": ratio,
                    "seed": seed,
                    "method": f"baselines/{alg_name}",
                    "label": label_tag,
                    "split": "failed",
                    "accuracy": None, "f1": None, "auc": None,
                    "precision": None, "recall": None,
                    "wall_time_s": bm.get("wall_time_seconds"),
                    "peak_rss_mb": bm.get("peak_rss_mb"),
                })
                continue

            rows.append({
                "ratio": ratio,
                "seed": seed,
                "method": f"baselines/{alg_name}",
                "label": label_tag,
                "split": "val",
                "accuracy": res.get("val_accuracy"),
                "f1": res.get("val_f1"),
                "auc": res.get("val_auc"),
                "precision": None,
                "recall": None,
                "wall_time_s": bm.get("wall_time_seconds"),
                "peak_rss_mb": bm.get("peak_rss_mb"),
            })

    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate scalability benchmark results")
    p.add_argument("--results-root",
                   default="/labs/Aguiar/SSPA_BRAY/results/scalability_benchmark/methods",
                   help="Root directory of method results")
    p.add_argument("--output", "-o",
                   default="/labs/Aguiar/SSPA_BRAY/results/scalability_benchmark/summary/all_metrics.csv",
                   help="Output CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {results_root} ...")

    all_rows = []

    for ratio_dir in sorted(results_root.iterdir()):
        if not ratio_dir.is_dir():
            continue
        # Support "ratio_0.15", "15p" (patient count), and bare number dirs
        rname = ratio_dir.name
        if rname.startswith("ratio_"):
            ratio = float(rname.replace("ratio_", ""))
        elif rname.endswith("p"):
            try:
                ratio = int(rname[:-1])  # treat as patient count directly
            except ValueError:
                continue
        else:
            continue

        for seed_dir in sorted(ratio_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.replace("seed_", ""))
            cell_counts = _load_cell_counts(seed_dir)
            rows_before = len(all_rows)

            for method_dir in sorted(seed_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                name = method_dir.name

                if name.startswith("drgp_"):
                    all_rows.extend(_parse_drgp(method_dir, name, ratio, seed))
                elif name == "spectra_sup_baselines":
                    all_rows.extend(_parse_spectra_or_schpf_baselines(
                        method_dir, "spectra_sup", "spectra_sup", ratio, seed))
                elif name == "schpf_baselines":
                    all_rows.extend(_parse_spectra_or_schpf_baselines(
                        method_dir, "schpf", "schpf", ratio, seed))
                elif name == "baselines":
                    all_rows.extend(_parse_baselines(method_dir, ratio, seed))

            # Attach cell counts to all rows from this seed_dir
            if cell_counts:
                for row in all_rows[rows_before:]:
                    row.update(cell_counts)

    if not all_rows:
        print("No results found.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")

    # Summary: mean test AUC
    print("\n" + "=" * 90)
    print("PERFORMANCE: Mean test AUC by method and ratio (across seeds)")
    print("=" * 90)

    test_df = df[(df["split"] == "test") & df["auc"].notna()]
    if len(test_df) > 0:
        pivot = test_df.groupby(["method", "label", "ratio"])["auc"].agg(["mean", "std"]).round(4)
        print(pivot.to_string())

    # Summary: benchmark metrics
    print("\n" + "=" * 90)
    print("BENCHMARK: Wall time (s) and peak RSS (MB) by method and ratio")
    print("=" * 90)

    bm_cols = ["ratio", "seed", "method", "wall_time_s", "peak_rss_mb"]
    bm_df = df[df["wall_time_s"].notna()][bm_cols].drop_duplicates()
    if len(bm_df) > 0:
        bm_pivot = bm_df.groupby(["method", "ratio"]).agg({
            "wall_time_s": ["mean", "std"],
            "peak_rss_mb": ["mean", "std"],
        }).round(1)
        print(bm_pivot.to_string())

    # Failed jobs
    failed = df[df["split"] == "failed"]
    if len(failed) > 0:
        print("\n" + "=" * 90)
        print("FAILED RUNS:")
        print("=" * 90)
        print(failed[["ratio", "seed", "method", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
