"""
D3 analysis: pathway-mask corruption robustness.

Compares masked vs combined modes under eta in {0, 0.1, 0.25, 0.5}.
Primary metric: cos_mean under each mode.
Plan's expected behavior: combined degrades more gracefully.
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


PALETTE = {
    "masked":   "#aa5c2b",
    "combined": "#2b6aa6",
}


def aggregate(raw_dir: str, config_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    cond_meta = {
        i: {"eta_label": c.get("eta_label", float("nan")),
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
        # Derive mode family ('masked' or 'combined') from the label prefix
        mode_family = "masked" if meta["mode_label"].startswith("masked") else "combined"
        for method in meta["methods"]:
            if method not in rec:
                continue
            m = rec[method].item() if rec[method].dtype == object else rec[method]
            if not isinstance(m, dict) or m.get("error"):
                continue
            rows.append({
                "method": method,
                "mode_family": mode_family,
                "mode_label": meta["mode_label"],
                "eta": meta["eta_label"],
                "seed": int(rec["seed"]),
                "cos_mean": float(m.get("cos_mean", float("nan"))),
                "v_spearman": float(m.get("v_spearman", float("nan"))),
                "ood_auroc": float(m.get("ood_auroc", float("nan"))),
            })
    return pd.DataFrame(rows)


def _panel(ax, df, metric, ylabel, ylim=None):
    for mode_family in ("masked", "combined"):
        sub = df[df["mode_family"] == mode_family].dropna(subset=[metric])
        if sub.empty:
            continue
        g = sub.groupby("eta")[metric]
        med = g.median(); q25 = g.quantile(0.25); q75 = g.quantile(0.75)
        xs = med.index.values
        ax.plot(xs, med.values, "-o", color=PALETTE[mode_family], label=mode_family)
        ax.fill_between(xs, q25.values, q75.values, alpha=0.18, color=PALETTE[mode_family])
    ax.set_xlabel(r"mask corruption $\eta$ (fraction of in-mask genes swapped)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/D3_mask_corruption")
    ap.add_argument("--config",  default="configs/D3_mask_corruption.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/D3_mask_corruption")
    ap.add_argument("--figure-dir", default="figures/D3")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)

    g = df.groupby(["mode_family", "eta"]).agg(
        cos_mean_med=("cos_mean", "median"),
        cos_mean_q25=("cos_mean", lambda s: s.quantile(0.25)),
        cos_mean_q75=("cos_mean", lambda s: s.quantile(0.75)),
        v_spearman_med=("v_spearman", "median"),
        ood_auroc_med=("ood_auroc", "median"),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print(g.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    _panel(axes[0], df, "cos_mean",   "matched cos sim",  ylim=(0.0, 1.05))
    _panel(axes[1], df, "v_spearman", "v_spearman (|v|)", ylim=(0.0, 1.05))
    _panel(axes[2], df, "ood_auroc",  "OOD AUROC",        ylim=(0.5, 1.0))
    fig.suptitle("D3 — pathway-mask corruption robustness")
    plt.tight_layout()
    plt.savefig(Path(args.figure_dir) / "d3_three_panel.png", dpi=180)
    plt.close()
    print(f"\nFigure: {args.figure_dir}/d3_three_panel.png")


if __name__ == "__main__":
    main()
