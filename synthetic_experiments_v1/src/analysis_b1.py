"""
B1 analysis: Disease-Relevant Program Ranking under different DRGP modes.

Primary metric: v_spearman (Spearman rho of |v_hat| vs |v_true| after Hungarian).
Secondary: precision@K_rel, Kendall's tau, OOD AUROC.

Compares 4 conditions:
  unmasked         : K_fit=10, no mask. Baseline.
  masked_oracle    : K_fit=10, mask covers all programs. Best case for masked.
  masked_missing   : K_fit=7, mask covers [3..9] only. Plan's flagged failure case.
  combined_rescue  : K_fit=10, mask covers [3..9] + 3 free factors. Rescue test.

Output:
  results/aggregated/B1_program_ranking/{long.csv, summary.csv}
  figures/B1/{v_spearman_by_mode.png, precision_at_3_by_mode.png,
              ood_auroc_by_mode.png}
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def aggregate(raw_dir: str, config_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    cond_meta = {
        i: {"K_fit": c["K_fit"],
            "mode_label": c.get("mode_label", "unknown"),
            "methods": c.get("methods", [])}
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
        for method in meta["methods"]:
            if method not in rec:
                continue
            m = rec[method].item() if rec[method].dtype == object else rec[method]
            if not isinstance(m, dict) or m.get("error"):
                continue
            rows.append({
                "method": method,
                "mode_label": meta["mode_label"],
                "seed": int(rec["seed"]),
                "condition_idx": cidx,
                "K_fit": meta["K_fit"],
                "cos_mean": float(m.get("cos_mean", float("nan"))),
                "v_spearman": float(m.get("v_spearman", float("nan"))),
                "v_kendall":  float(m.get("v_kendall", float("nan"))),
                "precision_at_rel": float(m.get("precision_at_rel", float("nan"))),
                "support_auprc": float(m.get("support_auprc", float("nan"))),
                "ood_auroc": float(m.get("ood_auroc", float("nan"))),
                "elapsed_s": float(m.get("elapsed_s", float("nan"))),
            })
    return pd.DataFrame(rows)


# Stable ordering of modes for plotting (left-to-right tells the story)
MODE_ORDER = ["unmasked", "masked_oracle", "masked_missing", "combined_rescue"]
PALETTE = {
    "unmasked":         "#2b6aa6",
    "masked_oracle":    "#5e933d",
    "masked_missing":   "#aa5c2b",
    "combined_rescue":  "#7a3d8c",
}


def plot_metric_by_mode(df: pd.DataFrame, metric: str, ylabel: str,
                          title: str, out_path: str, ymin: float = 0.0,
                          ymax: float = 1.05) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    modes_present = [m for m in MODE_ORDER if m in df["mode_label"].unique()]
    data = [df[df["mode_label"] == m][metric].dropna().values for m in modes_present]
    positions = list(range(len(modes_present)))
    bp = ax.boxplot(data, positions=positions, widths=0.55,
                    patch_artist=True, showmeans=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=5))
    for box, mode in zip(bp["boxes"], modes_present):
        box.set_facecolor(PALETTE.get(mode, "#888"))
        box.set_alpha(0.65)
    # Also overlay individual seed dots with a small horizontal jitter
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, data):
        if len(vals) == 0:
            continue
        x_jit = pos + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x_jit, vals, s=10, alpha=0.35, color="black", zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(modes_present, rotation=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/B1_program_ranking")
    ap.add_argument("--config", default="configs/B1_program_ranking.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/B1_program_ranking")
    ap.add_argument("--figure-dir", default="figures/B1")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)
    print(f"Wrote {len(df)} rows to long.csv")

    g = df.groupby(["mode_label", "method"]).agg(
        v_spearman_med=("v_spearman", "median"),
        v_spearman_q25=("v_spearman", lambda s: s.quantile(0.25)),
        v_spearman_q75=("v_spearman", lambda s: s.quantile(0.75)),
        precision_at_rel_med=("precision_at_rel", "median"),
        ood_auroc_med=("ood_auroc", "median"),
        cos_mean_med=("cos_mean", "median"),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print(g.round(3).to_string(index=False))

    plot_metric_by_mode(
        df, "v_spearman", "Spearman rho(|v_true|, |v_hat|)",
        "B1: program-ranking quality by DRGP mode",
        str(Path(args.figure_dir) / "v_spearman_by_mode.png"),
        ymin=-0.2, ymax=1.05,
    )
    plot_metric_by_mode(
        df, "precision_at_rel", "precision@K_rel",
        "B1: precision@K_rel by DRGP mode",
        str(Path(args.figure_dir) / "precision_at_rel_by_mode.png"),
        ymin=0.0, ymax=1.05,
    )
    plot_metric_by_mode(
        df, "ood_auroc", "OOD AUROC",
        "B1: OOD predictive performance by DRGP mode",
        str(Path(args.figure_dir) / "ood_auroc_by_mode.png"),
        ymin=0.5, ymax=1.0,
    )

    # Wilcoxon: pairwise comparisons of v_spearman across modes, paired on seed
    print("\nPaired Wilcoxon: v_spearman across mode pairs (matched on seed)")
    modes = [m for m in MODE_ORDER if m in df["mode_label"].unique()]
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a, b = modes[i], modes[j]
            da = df[df["mode_label"] == a][["seed", "v_spearman"]].set_index("seed")
            db = df[df["mode_label"] == b][["seed", "v_spearman"]].set_index("seed")
            joined = da.join(db, lsuffix=f"_{a}", rsuffix=f"_{b}").dropna()
            if len(joined) < 5:
                continue
            try:
                stat, p = wilcoxon(joined[f"v_spearman_{a}"], joined[f"v_spearman_{b}"])
            except Exception:
                p = float("nan")
            diff = (joined[f"v_spearman_{a}"] - joined[f"v_spearman_{b}"]).median()
            print(f"  {a:18s} - {b:18s}  med diff={diff:+.4f}  n={len(joined)}  p={p:.4g}")

    print(f"\nFigures: {args.figure_dir}/")


if __name__ == "__main__":
    main()
