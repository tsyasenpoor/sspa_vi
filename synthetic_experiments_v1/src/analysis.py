"""
Aggregate raw .npz results across seeds and conditions, compute summaries,
run paired Wilcoxon tests, and produce paper-ready figures.

Usage:
  python -m src.analysis --raw-dir results/raw/A1_factor_recovery \
                          --config configs/A1_factor_recovery.yaml \
                          --out-dir results/aggregated/A1_factor_recovery \
                          --figure-dir figures/A1
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
from scipy.stats import wilcoxon


def _load_record(npz_path: str) -> dict | None:
    """Load one .npz; return None if malformed."""
    try:
        d = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  skip {os.path.basename(npz_path)}: {e}")
        return None
    rec = {k: d[k] for k in d.files}
    return rec


def _flat_row(rec: dict, method: str, cond_meta: dict) -> dict | None:
    """Flatten a single (record, method) entry into a long-form row."""
    if method not in rec:
        return None
    m_obj = rec[method].item() if rec[method].dtype == object else rec[method]
    if not isinstance(m_obj, dict):
        return None
    if "error" in m_obj and m_obj.get("error"):
        return {
            "method": method, "error": True,
            "seed": int(rec["seed"]),
            "condition_idx": int(rec["condition_idx"]),
            **{f"cond_{k}": v for k, v in cond_meta.items()},
        }
    row = {
        "method": method,
        "error": False,
        "seed": int(rec["seed"]),
        "condition_idx": int(rec["condition_idx"]),
        "K_fit": int(rec["K_fit"].item()) if rec["K_fit"].ndim == 0 else int(rec["K_fit"]),
        "pi_label": float(rec["pi_label"].item()) if rec["pi_label"].ndim == 0 else float(rec["pi_label"]),
        "wall_time_s": float(rec["wall_time_s"].item()) if "wall_time_s" in rec else float("nan"),
        "elapsed_s":   float(m_obj.get("elapsed_s", float("nan"))),
        "cos_mean":    float(m_obj.get("cos_mean", float("nan"))),
        "jaccard_top50_mean": float(m_obj.get("jaccard_top50_mean", float("nan"))),
        "support_auprc":      float(m_obj.get("support_auprc", float("nan"))),
        "fdr_at_0p5":         float(m_obj.get("fdr_at_0p5", float("nan"))),
        "v_spearman":         float(m_obj.get("v_spearman", float("nan"))),
        "v_kendall":          float(m_obj.get("v_kendall", float("nan"))),
        "precision_at_rel":   float(m_obj.get("precision_at_rel", float("nan"))),
        "ood_auroc":          float(m_obj.get("ood_auroc", float("nan"))),
        "max_abs_beta":       float(m_obj.get("max_abs_beta", float("nan"))),
    }
    for k, v in cond_meta.items():
        row[f"cond_{k}"] = v
    return row


def aggregate(raw_dir: str, config_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    cond_meta_per_idx = {
        i: {"K_fit": c["K_fit"], "pi_label": c.get("pi_label", float("nan"))}
        for i, c in enumerate(cfg["conditions"])
    }
    rows = []
    paths = sorted(glob.glob(os.path.join(raw_dir, "cond*_seed*.npz")))
    for path in paths:
        rec = _load_record(path)
        if rec is None:
            continue
        cidx = int(rec["condition_idx"])
        meta = cond_meta_per_idx.get(cidx, {})
        for method in cfg["methods"]:
            r = _flat_row(rec, method, meta)
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Median + IQR per (method, K_fit, pi_label)."""
    g = df[~df["error"]].groupby(["method", "K_fit", "pi_label"])[metric]
    return g.agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75),
                   "count"]).rename(columns={
        "<lambda_0>": "q25", "<lambda_1>": "q75", "count": "n_seeds",
    }).reset_index()


def paired_wilcoxon(df: pd.DataFrame, metric: str, base: str) -> pd.DataFrame:
    """Per (K_fit, pi_label), paired Wilcoxon of each method vs base across seeds."""
    out = []
    for (Kf, pi), sub in df[~df["error"]].groupby(["K_fit", "pi_label"]):
        base_df = sub[sub["method"] == base][["seed", metric]].set_index("seed")
        for method in sub["method"].unique():
            if method == base:
                continue
            md = sub[sub["method"] == method][["seed", metric]].set_index("seed")
            joined = base_df.join(md, lsuffix="_base", rsuffix="_test").dropna()
            if len(joined) < 5:
                out.append({"K_fit": Kf, "pi_label": pi, "method": method,
                            "n_pairs": len(joined), "p_value": float("nan")})
                continue
            try:
                stat, p = wilcoxon(joined[f"{metric}_test"], joined[f"{metric}_base"])
            except ValueError:
                p = float("nan")
            out.append({
                "K_fit": Kf, "pi_label": pi, "method": method,
                "n_pairs": len(joined),
                "median_test": joined[f"{metric}_test"].median(),
                "median_base": joined[f"{metric}_base"].median(),
                "p_value": p,
            })
    return pd.DataFrame(out)


def plot_a1_heatmap(df: pd.DataFrame, out_path: str, method: str = "drgp_unmasked",
                    metric: str = "cos_mean") -> None:
    sub = df[(df["method"] == method) & (~df["error"])]
    if sub.empty:
        print(f"  no data for {method}")
        return
    pivot = sub.pivot_table(index="K_fit", columns="pi_label", values=metric, aggfunc="median")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_xlabel(r"sparsity $\pi$"); ax.set_ylabel(r"$K_{\mathrm{fit}}$")
    ax.set_title(f"A1: median {metric} ({method})")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if pivot.values[i, j] < 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_method_comparison(df: pd.DataFrame, out_path: str,
                            metric: str = "cos_mean",
                            K_fit_filter: int | None = None) -> None:
    sub = df[~df["error"]].copy()
    if K_fit_filter is not None:
        sub = sub[sub["K_fit"] == K_fit_filter]
    if sub.empty:
        return
    methods = sorted(sub["method"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [sub[sub["method"] == m][metric].dropna().values for m in methods]
    ax.boxplot(data, labels=methods, showmeans=True)
    ax.set_ylabel(metric)
    title_suffix = f" (K_fit={K_fit_filter})" if K_fit_filter else ""
    ax.set_title(f"A1: {metric} across methods{title_suffix}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--figure-dir", required=True)
    ap.add_argument("--metric", default="cos_mean")
    ap.add_argument("--baseline-method", default="nmf_lr",
                    help="Method to use as paired-Wilcoxon baseline.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)
    print(f"Wrote {len(df)} rows to long.csv")

    for metric in ("cos_mean", "jaccard_top50_mean", "support_auprc",
                    "v_spearman", "precision_at_rel", "ood_auroc"):
        t = summary_table(df, metric)
        t.to_csv(Path(args.out_dir) / f"summary_{metric}.csv", index=False)

    wt = paired_wilcoxon(df, args.metric, args.baseline_method)
    wt.to_csv(Path(args.out_dir) / f"wilcoxon_{args.metric}_vs_{args.baseline_method}.csv", index=False)

    plot_a1_heatmap(df, str(Path(args.figure_dir) / "A1_heatmap_cos_mean.png"),
                     metric="cos_mean")
    plot_a1_heatmap(df, str(Path(args.figure_dir) / "A1_heatmap_support_auprc.png"),
                     method="drgp_unmasked", metric="support_auprc")
    plot_method_comparison(df, str(Path(args.figure_dir) / "A1_method_comparison_K10.png"),
                            metric="cos_mean", K_fit_filter=10)
    plot_method_comparison(df, str(Path(args.figure_dir) / "A1_method_comparison_ood_K10.png"),
                            metric="ood_auroc", K_fit_filter=10)
    print(f"Figures in {args.figure_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
