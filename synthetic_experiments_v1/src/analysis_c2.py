"""
C2 analysis: predictive performance in n << p + OOD.

Primary plot: OOD AUROC vs n, one line per method, with 25-75 IQR shading.
Also: paired Wilcoxon DRGP vs each baseline at each n.
"""
from __future__ import annotations

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
            "n_label": c.get("n_label", float("nan"))}
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
                "condition_idx": cidx,
                "K_fit": meta["K_fit"],
                "n_label": meta["n_label"],
                "cos_mean": float(m.get("cos_mean", float("nan"))),
                "v_spearman": float(m.get("v_spearman", float("nan"))),
                "support_auprc": float(m.get("support_auprc", float("nan"))),
                "ood_auroc": float(m.get("ood_auroc", float("nan"))),
                "elapsed_s": float(m.get("elapsed_s", float("nan"))),
            })
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/C2_sample_complexity_ood")
    ap.add_argument("--config", default="configs/C2_sample_complexity_ood.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/C2_sample_complexity_ood")
    ap.add_argument("--figure-dir", default="figures/C2")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)
    print(f"Wrote {len(df)} rows to long.csv")

    g = df.groupby(["method", "n_label"]).agg(
        ood_auroc_med=("ood_auroc", "median"),
        ood_auroc_q25=("ood_auroc", lambda s: s.quantile(0.25)),
        ood_auroc_q75=("ood_auroc", lambda s: s.quantile(0.75)),
        cos_mean_med=("cos_mean", "median"),
        v_spearman_med=("v_spearman", "median"),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print(g.round(3).to_string(index=False))

    # Plot OOD AUROC vs n
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = {"drgp_unmasked": "#2b6aa6", "nmf_lr": "#aa5c2b",
                "pca_lr": "#888888",      "plain_lr": "#5b8a3a"}
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].dropna(subset=["ood_auroc"])
        if sub.empty:
            continue
        g2 = sub.groupby("n_label")["ood_auroc"]
        med = g2.median()
        q25 = g2.quantile(0.25); q75 = g2.quantile(0.75)
        xs = med.index.values
        ax.plot(xs, med.values, "-o", label=method, color=palette.get(method, "k"))
        ax.fill_between(xs, q25.values, q75.values, alpha=0.18,
                         color=palette.get(method, "k"))
    ax.set_xlabel("n (training samples)")
    ax.set_ylabel("OOD AUROC")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 250, 500])
    ax.set_xticklabels(["50", "100", "250", "500"])
    ax.set_title("C2: OOD AUROC vs n (asthma 0.2 -> 0.6, frozen Beta/v/Delta)")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(args.figure_dir) / "ood_auroc_vs_n.png", dpi=180)
    plt.close()

    # Paired Wilcoxon DRGP vs each baseline at each n
    print("\nPaired Wilcoxon: DRGP vs baseline on OOD AUROC")
    for baseline in ("nmf_lr", "pca_lr", "plain_lr"):
        print(f"\n  vs {baseline}:")
        for n_label in sorted(df["n_label"].unique()):
            d = df[(df["method"] == "drgp_unmasked") & (df["n_label"] == n_label)][["seed", "ood_auroc"]].set_index("seed")
            b = df[(df["method"] == baseline) & (df["n_label"] == n_label)][["seed", "ood_auroc"]].set_index("seed")
            j = d.join(b, lsuffix="_d", rsuffix="_b").dropna()
            if len(j) < 5:
                continue
            diff = (j["ood_auroc_d"] - j["ood_auroc_b"]).median()
            try:
                stat, p = wilcoxon(j["ood_auroc_d"], j["ood_auroc_b"])
            except Exception:
                p = float("nan")
            print(f"    n={int(n_label):4d}  med(DRGP-{baseline})={diff:+.4f}  "
                  f"n_pairs={len(j)}  p={p:.4g}")

    print(f"\nFigure: {args.figure_dir}/ood_auroc_vs_n.png")


if __name__ == "__main__":
    main()
