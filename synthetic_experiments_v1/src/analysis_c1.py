"""
C1 analysis: Disease-relevant program identification vs. unsupervised baselines.

Primary metric (per plan §C1):
  AUROC of the binary classifier "is program k disease-relevant?" built from
  per-method |v_hat|. Pooled across (seed x k_true) pairs:
    - score : |v_hat_aligned[k_true]| after Hungarian matching
    - label : 1 if k_true in rel_idx else 0

plain_lr has no per-program v_hat (raw gene features) -> excluded from
the primary metric; reported only for OOD AUROC.

Secondary: number of false positives (programs in top-K_rel by |v_hat| that
are not truly disease-relevant). Per-seed; summarised as median.

Output:
  results/aggregated/C1_method_ranking/{long.csv, summary.csv, per_method_auroc.csv}
  figures/C1/{ranking_auroc_box.png, top3_false_positives_box.png,
              ood_auroc_box.png}
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _align_v_to_true(v_true: np.ndarray, v_hat: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K_true = v_true.shape[0]
    v_hat_aligned = np.full(K_true, np.nan)
    for k_fit in range(v_hat.shape[0]):
        k_true = int(pi[k_fit])
        if 0 <= k_true < K_true:
            v_hat_aligned[k_true] = v_hat[k_fit]
    return v_hat_aligned


def aggregate(raw_dir: str, config_path: str,
              extra_dirs: list[str] | None = None,
              extra_methods: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (long_df, per_program_df). long_df = one row per (seed, method,
    per-program). per_program_df has (seed, k_true, method, abs_v_hat, is_rel).

    extra_dirs: optional secondary raw dirs (e.g. the Spectra parallel run);
    extra_methods: method labels to look for in those dirs. Records are merged
    by (seed, condition_idx) at the DataFrame level."""
    cfg = yaml.safe_load(open(config_path))
    methods = list(cfg["methods"]) + list(extra_methods or [])
    K_rel = cfg["generator_defaults"].get("K_rel", 3)

    raw_dirs = [raw_dir] + list(extra_dirs or [])
    paths = []
    for d in raw_dirs:
        paths.extend(sorted(glob.glob(os.path.join(d, "cond*_seed*.npz"))))

    rows_long, rows_pp = [], []
    for path in paths:
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        rec = {k: d[k] for k in d.files}
        seed = int(rec["seed"])
        v_true = np.asarray(rec.get("v_true", []))
        rel_idx = set(int(i) for i in np.asarray(rec.get("rel_idx", [])).tolist())
        K_true = v_true.size
        if K_true == 0:
            continue

        for method in methods:
            if method not in rec:
                continue
            m = rec[method].item() if rec[method].dtype == object else rec[method]
            if not isinstance(m, dict) or m.get("error"):
                continue
            v_hat = m.get("v_hat")
            pi = m.get("pi")
            row = {
                "method": method,
                "seed": seed,
                "cos_mean": float(m.get("cos_mean", float("nan"))),
                "v_spearman": float(m.get("v_spearman", float("nan"))),
                "ood_auroc": float(m.get("ood_auroc", float("nan"))),
                "elapsed_s": float(m.get("elapsed_s", float("nan"))),
            }
            if v_hat is not None and pi is not None:
                v_hat = np.asarray(v_hat)
                pi = np.asarray(pi)
                v_hat_aligned = _align_v_to_true(v_true, v_hat, pi)
                # Per-program contribution to the pooled C1 metric
                for k_true in range(K_true):
                    abs_v = float(abs(v_hat_aligned[k_true])) if not np.isnan(v_hat_aligned[k_true]) else float("nan")
                    rows_pp.append({
                        "method": method, "seed": seed, "k_true": k_true,
                        "abs_v_hat_aligned": abs_v,
                        "is_disease_relevant": int(k_true in rel_idx),
                    })
                # Top-K_rel false positives (programs in top-K_rel of |v_hat_aligned|
                # that are NOT in rel_idx).
                finite_mask = ~np.isnan(v_hat_aligned)
                if finite_mask.sum() >= K_rel:
                    order = np.argsort(-np.abs(np.nan_to_num(v_hat_aligned, nan=-np.inf)))
                    top = order[:K_rel].tolist()
                    fp = sum(1 for k in top if k not in rel_idx)
                    row["top_K_false_positives"] = fp
                else:
                    row["top_K_false_positives"] = float("nan")
            else:
                row["top_K_false_positives"] = float("nan")
            rows_long.append(row)
    return pd.DataFrame(rows_long), pd.DataFrame(rows_pp)


def compute_method_auroc(df_pp: pd.DataFrame) -> pd.DataFrame:
    """Pooled AUROC per method using |v_hat_aligned| as score and is_rel as label."""
    rows = []
    for method, sub in df_pp.groupby("method"):
        sub = sub.dropna(subset=["abs_v_hat_aligned"])
        if sub.empty:
            rows.append({"method": method, "auroc": float("nan"), "n": 0,
                         "n_pos": 0, "n_neg": 0})
            continue
        y = sub["is_disease_relevant"].values
        s = sub["abs_v_hat_aligned"].values
        if len(np.unique(y)) < 2:
            rows.append({"method": method, "auroc": float("nan"), "n": len(y),
                         "n_pos": int(y.sum()), "n_neg": int((1-y).sum())})
            continue
        auroc = float(roc_auc_score(y, s))
        rows.append({"method": method, "auroc": auroc, "n": len(y),
                     "n_pos": int(y.sum()), "n_neg": int((1-y).sum())})
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def compute_per_seed_auroc(df_pp: pd.DataFrame) -> pd.DataFrame:
    """Per-seed AUROC for each method (10 programs => 3 pos / 7 neg per seed)."""
    rows = []
    for (method, seed), sub in df_pp.groupby(["method", "seed"]):
        sub = sub.dropna(subset=["abs_v_hat_aligned"])
        if sub.empty:
            continue
        y = sub["is_disease_relevant"].values
        s = sub["abs_v_hat_aligned"].values
        if len(np.unique(y)) < 2:
            continue
        rows.append({"method": method, "seed": seed,
                     "auroc": float(roc_auc_score(y, s))})
    return pd.DataFrame(rows)


def plot_metric_box(df: pd.DataFrame, metric: str, ylabel: str, title: str,
                     out_path: str, ymin: float = 0.0, ymax: float = 1.05) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    methods_present = sorted(df["method"].unique())
    data = [df[df["method"] == m][metric].dropna().values for m in methods_present]
    bp = ax.boxplot(data, labels=methods_present, showmeans=True, showfliers=False,
                     patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5),
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=5))
    palette = {
        "drgp_unmasked": "#2b6aa6",
        "schpf_lr":      "#9c2b6a",
        "spectra_lr":    "#d97b00",
        "nmf_lr":        "#aa5c2b",
        "pca_lr":        "#888888",
        "plain_lr":      "#5b8a3a",
    }
    for box, m in zip(bp["boxes"], methods_present):
        box.set_facecolor(palette.get(m, "#888"))
        box.set_alpha(0.65)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        x_jit = (i + 1) + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x_jit, vals, s=10, alpha=0.35, color="black", zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/C1_method_ranking")
    ap.add_argument("--config", default="configs/C1_method_ranking.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/C1_method_ranking")
    ap.add_argument("--figure-dir", default="figures/C1")
    ap.add_argument("--extra-raw-dirs", nargs="*",
                     default=["results/raw/C1_method_ranking_spectra"],
                     help="Additional raw dirs whose records get merged in.")
    ap.add_argument("--extra-methods", nargs="*", default=["spectra_lr"],
                     help="Method labels to look for in --extra-raw-dirs.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df_long, df_pp = aggregate(args.raw_dir, args.config,
                                extra_dirs=args.extra_raw_dirs,
                                extra_methods=args.extra_methods)
    df_long.to_csv(Path(args.out_dir) / "long.csv", index=False)
    df_pp.to_csv(Path(args.out_dir) / "per_program.csv", index=False)
    print(f"Wrote {len(df_long)} long rows, {len(df_pp)} per-program rows")

    # Pooled AUROC per method
    df_auroc = compute_method_auroc(df_pp)
    df_auroc.to_csv(Path(args.out_dir) / "per_method_auroc.csv", index=False)
    print("\nPooled AUROC per method (primary C1 metric):")
    print(df_auroc.round(4).to_string(index=False))

    # Per-seed AUROC (for boxplot)
    df_seed_auroc = compute_per_seed_auroc(df_pp)
    df_seed_auroc.to_csv(Path(args.out_dir) / "per_seed_auroc.csv", index=False)

    # Summary stats per method
    g = df_long.groupby("method").agg(
        cos_mean_med=("cos_mean", "median"),
        v_spearman_med=("v_spearman", "median"),
        ood_auroc_med=("ood_auroc", "median"),
        top_K_fp_med=("top_K_false_positives", "median"),
        elapsed_med=("elapsed_s", "median"),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print("\nPer-method summary:")
    print(g.round(3).to_string(index=False))

    # Pairwise Wilcoxon vs DRGP on per-seed ranking AUROC
    print("\nPaired Wilcoxon on per-seed ranking AUROC vs drgp_unmasked")
    drgp = df_seed_auroc[df_seed_auroc["method"] == "drgp_unmasked"][["seed", "auroc"]].set_index("seed")
    for method in df_seed_auroc["method"].unique():
        if method == "drgp_unmasked":
            continue
        b = df_seed_auroc[df_seed_auroc["method"] == method][["seed", "auroc"]].set_index("seed")
        j = drgp.join(b, lsuffix="_drgp", rsuffix=f"_{method}").dropna()
        if len(j) < 5:
            continue
        diff = (j["auroc_drgp"] - j[f"auroc_{method}"]).median()
        try:
            stat, p = wilcoxon(j["auroc_drgp"], j[f"auroc_{method}"])
        except Exception:
            p = float("nan")
        print(f"  DRGP vs {method:14s}  med(DRGP - {method})={diff:+.4f}  n={len(j)}  p={p:.4g}")

    # Figures
    plot_metric_box(df_seed_auroc, "auroc",
                     "AUROC: 'is program k disease-relevant?' (per seed)",
                     "C1: ranking AUROC by method",
                     str(Path(args.figure_dir) / "ranking_auroc_box.png"),
                     ymin=0.4, ymax=1.05)
    plot_metric_box(df_long, "top_K_false_positives",
                     "# false positives in top-3 |v_hat|",
                     "C1: false positives among top-3 ranked programs",
                     str(Path(args.figure_dir) / "top3_false_positives_box.png"),
                     ymin=-0.5, ymax=3.5)
    plot_metric_box(df_long, "ood_auroc", "OOD AUROC",
                     "C1: OOD predictive performance",
                     str(Path(args.figure_dir) / "ood_auroc_box.png"),
                     ymin=0.5, ymax=1.0)

    print(f"\nFigures: {args.figure_dir}/")


if __name__ == "__main__":
    main()
