"""
D1 analysis: Likelihood misspecification (Poisson vs Negative-Binomial).

Reports A1 cos_mean and B1 v_spearman as functions of nb_dispersion phi
(phi=0 = pure Poisson; phi>0 = NB with variance = mu + phi*mu^2).
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
        i: {"phi_label": c.get("phi_label", float("nan"))}
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
        for method in cfg["methods"]:
            if method not in rec:
                continue
            m = rec[method].item() if rec[method].dtype == object else rec[method]
            if not isinstance(m, dict) or m.get("error"):
                continue
            rows.append({
                "method": method,
                "seed": int(rec["seed"]),
                "phi": meta["phi_label"],
                "cos_mean": float(m.get("cos_mean", float("nan"))),
                "v_spearman": float(m.get("v_spearman", float("nan"))),
                "support_auprc": float(m.get("support_auprc", float("nan"))),
                "ood_auroc": float(m.get("ood_auroc", float("nan"))),
            })
    return pd.DataFrame(rows)


def _panel(ax, df, metric, ylabel, ylim=None):
    for method in [m for m in PALETTE if m in df["method"].unique()]:
        sub = df[df["method"] == method].dropna(subset=[metric])
        if sub.empty:
            continue
        g = sub.groupby("phi")[metric]
        med = g.median(); q25 = g.quantile(0.25); q75 = g.quantile(0.75)
        xs = med.index.values
        ax.plot(xs, med.values, "-o", color=PALETTE[method], label=method)
        ax.fill_between(xs, q25.values, q75.values, alpha=0.18, color=PALETTE[method])
    ax.set_xlabel(r"NB dispersion $\phi$ (0 = Poisson)")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/D1_nb_likelihood")
    ap.add_argument("--config",  default="configs/D1_nb_likelihood.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/D1_nb_likelihood")
    ap.add_argument("--figure-dir", default="figures/D1")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)

    g = df.groupby(["method", "phi"]).agg(
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
    fig.suptitle("D1 — likelihood misspecification (NB dispersion sweep)")
    plt.tight_layout()
    plt.savefig(Path(args.figure_dir) / "d1_three_panel.png", dpi=180)
    plt.close()
    print(f"\nFigure: {args.figure_dir}/d1_three_panel.png")


if __name__ == "__main__":
    main()
