"""
D4-specific analysis: overlap identifiability stress test.

Primary metric:  cos_overlap_pair = mean(cos_per_prog[[0, 1]])
                 — cosine similarity on the two overlapping programs only.
                 (cos_mean averages over all K_true matched programs, which
                  dilutes the overlap effect when 8/10 programs are disjoint.)

Secondary metric: v_spearman across all programs
                 — does supervision rescue ranking when loadings degrade?

Auxiliary: support_auprc on overlap pair (DRGP only).

Output:
  results/aggregated/D4_overlap_identifiability/{long.csv, summary.csv}
  figures/D4/{cos_overlap_vs_jaccard.png, v_spearman_vs_jaccard.png,
              support_auprc_vs_jaccard.png}
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


def aggregate_d4(raw_dir: str, config_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    # Overlap pair indices come from generator_defaults; default to (0, 1) for legacy D4
    overlap_pair = cfg["generator_defaults"].get("overlap_pair") or [0, 1]
    overlap_pair = tuple(int(x) for x in overlap_pair)
    print(f"  overlap_pair = {overlap_pair}")
    cond_meta = {
        i: {"K_fit": c["K_fit"],
            "jaccard_label": c.get("jaccard_label", float("nan"))}
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

            cos_per_prog = np.asarray(m.get("cos_per_prog", []))
            if cos_per_prog.size > max(overlap_pair):
                cos_overlap = float(np.nanmean(cos_per_prog[list(overlap_pair)]))
            else:
                cos_overlap = float("nan")

            # Support AUPRC restricted to overlap pair (DRGP only, R_beta available)
            support_auprc_overlap = float("nan")
            if "R_beta" in m and m["R_beta"] is not None and "pi" in m:
                R_beta = np.asarray(m["R_beta"])
                pi = np.asarray(m["pi"])
                S = _reconstruct_S_from_seed(cfg, cidx, int(rec["seed"]))
                # Find which fitted columns matched true programs 0 and 1
                from sklearn.metrics import average_precision_score
                y_true_all, y_score_all = [], []
                for k_fit in range(R_beta.shape[1]):
                    k_true = int(pi[k_fit])
                    if k_true in overlap_pair:
                        y_true_all.append(S[:, k_true])
                        y_score_all.append(R_beta[:, k_fit])
                if y_true_all:
                    y_true = np.concatenate(y_true_all)
                    y_score = np.concatenate(y_score_all)
                    if len(np.unique(y_true)) >= 2:
                        support_auprc_overlap = float(average_precision_score(y_true, y_score))

            rows.append({
                "experiment": "D4",
                "method": method,
                "seed": int(rec["seed"]),
                "condition_idx": cidx,
                "K_fit": meta["K_fit"],
                "jaccard_label": meta["jaccard_label"],
                "cos_overlap_pair": cos_overlap,
                "cos_mean_all": float(np.nanmean(cos_per_prog)) if cos_per_prog.size else float("nan"),
                "v_spearman": float(m.get("v_spearman", float("nan"))),
                "v_kendall":  float(m.get("v_kendall", float("nan"))),
                "precision_at_rel": float(m.get("precision_at_rel", float("nan"))),
                "support_auprc_all": float(m.get("support_auprc", float("nan"))),
                "support_auprc_overlap": support_auprc_overlap,
                "ood_auroc": float(m.get("ood_auroc", float("nan"))),
                "elapsed_s": float(m.get("elapsed_s", float("nan"))),
            })
    return pd.DataFrame(rows)


def _reconstruct_S_from_seed(cfg: dict, cond_idx: int, seed: int) -> np.ndarray:
    """Re-run the generator to get the true support mask S."""
    from src.generator import generate
    cond = cfg["conditions"][cond_idx]
    gen_kwargs = dict(cfg["generator_defaults"])
    gen_kwargs.update(cond.get("generator_overrides", {}))
    # YAML lists for overlap_pair need to become tuples
    if isinstance(gen_kwargs.get("overlap_pair"), list):
        gen_kwargs["overlap_pair"] = tuple(gen_kwargs["overlap_pair"])
    gen_kwargs["seed"] = seed
    gt = generate(**gen_kwargs)
    return gt.S


def plot_metric_vs_jaccard(df: pd.DataFrame, metric: str, ylabel: str,
                             title: str, out_path: str, ymin: float = 0.0,
                             ymax: float = 1.05,
                             overlap_pair_str: str = "0, 1") -> None:
    """Plot with distinct marker shapes / linestyles + small marker x-offset so
    methods that share a y value remain visible (DRGP and NMF both hit cos=1)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    methods = sorted(df["method"].unique())
    style = {
        "drgp_unmasked": dict(color="#2b6aa6", ls="-",  marker="o", ms=7,  mfc="#2b6aa6", mec="black", mew=0.7, x_off=-0.012),
        "nmf_lr":        dict(color="#aa5c2b", ls="--", marker="s", ms=6,  mfc="none",    mec="#aa5c2b", mew=1.6, x_off=+0.012),
        "pca_lr":        dict(color="#666666", ls=":",  marker="^", ms=6,  mfc="#666666", mec="black", mew=0.6, x_off=0.0),
    }

    # Detect lines that overlap entirely; annotate
    medians_per_method = {}
    for method in methods:
        sub = df[df["method"] == method].dropna(subset=[metric])
        if sub.empty:
            continue
        g = sub.groupby("jaccard_label")[metric]
        medians_per_method[method] = (g.median(), g.quantile(0.25), g.quantile(0.75))

    for method in methods:
        if method not in medians_per_method:
            continue
        med, q25, q75 = medians_per_method[method]
        s = style.get(method, dict(color="k", ls="-", marker="o", ms=6,
                                    mfc="k", mec="k", mew=0.5, x_off=0.0))
        xs = med.index.values
        xs_jit = xs + s["x_off"]
        ax.plot(xs, med.values, ls=s["ls"], color=s["color"],
                linewidth=2.0, label=method, zorder=2)
        ax.plot(xs_jit, med.values, ls="none", marker=s["marker"],
                ms=s["ms"], mfc=s["mfc"], mec=s["mec"], mew=s["mew"], zorder=3)
        ax.fill_between(xs, q25.values, q75.values, alpha=0.18,
                        color=s["color"], zorder=1)

    ax.set_xlabel(f"Jaccard overlap on pair (programs {overlap_pair_str})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/D4_overlap_identifiability")
    ap.add_argument("--config", default="configs/D4_overlap_identifiability.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/D4_overlap_identifiability")
    ap.add_argument("--figure-dir", default="figures/D4")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate_d4(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)
    print(f"Wrote {len(df)} rows to long.csv")

    # Stash the overlap pair string for plot titles
    cfg = yaml.safe_load(open(args.config))
    op = cfg["generator_defaults"].get("overlap_pair") or [0, 1]
    overlap_pair_str = f"{int(op[0])}, {int(op[1])}"

    # Summary table
    g = df.groupby(["method", "jaccard_label"]).agg(
        cos_overlap_med=("cos_overlap_pair", "median"),
        cos_overlap_q25=("cos_overlap_pair", lambda s: s.quantile(0.25)),
        cos_overlap_q75=("cos_overlap_pair", lambda s: s.quantile(0.75)),
        v_spearman_med=("v_spearman", "median"),
        v_spearman_q25=("v_spearman", lambda s: s.quantile(0.25)),
        v_spearman_q75=("v_spearman", lambda s: s.quantile(0.75)),
        support_auprc_overlap_med=("support_auprc_overlap", "median"),
        ood_auroc_med=("ood_auroc", "median"),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print(g.round(3).to_string(index=False))

    # Plots
    plot_metric_vs_jaccard(
        df, "cos_overlap_pair", "cos(beta_hat, beta_true) on overlap pair",
        "Loading recovery on overlap pair",
        str(Path(args.figure_dir) / "cos_overlap_vs_jaccard.png"),
        overlap_pair_str=overlap_pair_str,
    )
    plot_metric_vs_jaccard(
        df, "v_spearman", "Spearman rho(|v_true|, |v_hat|)",
        "Supervised ranking recovery under overlap",
        str(Path(args.figure_dir) / "v_spearman_vs_jaccard.png"),
        ymin=0.0, ymax=1.05,
        overlap_pair_str=overlap_pair_str,
    )
    plot_metric_vs_jaccard(
        df[df["method"] == "drgp_unmasked"], "support_auprc_overlap",
        "AUPRC of r_beta on overlap pair",
        "DRGP support recovery on overlap pair",
        str(Path(args.figure_dir) / "support_auprc_overlap_vs_jaccard.png"),
        ymin=0.0, ymax=1.05,
        overlap_pair_str=overlap_pair_str,
    )
    plot_metric_vs_jaccard(
        df, "ood_auroc", "OOD AUROC",
        "OOD predictive performance under overlap",
        str(Path(args.figure_dir) / "ood_auroc_vs_jaccard.png"),
        ymin=0.5, ymax=1.05,
        overlap_pair_str=overlap_pair_str,
    )

    # Paired Wilcoxon DRGP vs NMF on cos_overlap_pair at each J
    print("\nPaired Wilcoxon: DRGP vs NMF on cos_overlap_pair")
    for J in sorted(df["jaccard_label"].unique()):
        d = df[(df["method"] == "drgp_unmasked") & (df["jaccard_label"] == J)][["seed", "cos_overlap_pair"]].set_index("seed")
        n = df[(df["method"] == "nmf_lr") & (df["jaccard_label"] == J)][["seed", "cos_overlap_pair"]].set_index("seed")
        j = d.join(n, lsuffix="_d", rsuffix="_n").dropna()
        if len(j) < 5:
            continue
        try:
            stat, p = wilcoxon(j["cos_overlap_pair_d"], j["cos_overlap_pair_n"])
        except Exception:
            p = float("nan")
        diff = (j["cos_overlap_pair_d"] - j["cos_overlap_pair_n"]).median()
        print(f"  J={J:.1f}  med(DRGP-NMF)={diff:+.4f}  n={len(j)}  p={p:.4g}")

    print(f"\nFigures: {args.figure_dir}/")


if __name__ == "__main__":
    main()
