"""
A2 analysis: Spike-and-Slab Support Recovery.

Uses A1 raw .npz files (which already save R_beta for DRGP). For each fit:
  - Regenerate the true binary support mask S from the seed (deterministic).
  - Use Hungarian pi to align fitted R_beta columns to true support columns.
  - Compute AUPRC over the pooled (gene x matched-column) flattened arrays.
  - Compute FDR at threshold 0.5 (the principled spike-and-slab cutoff).
  - Compute the calibration curve: empirical FDR vs nominal 1 - r_beta threshold.

Per plan A2:
  Primary metric: AUPRC of r_beta vs binary S (pooled over matched columns).
  Secondary metric: FDR at r_beta > 0.5.
  Expected: AUPRC >= 0.80 at well-specified K_fit = K_true.

Output:
  results/aggregated/A2_support_recovery/{long.csv, summary.csv}
  figures/A2/{auprc_heatmap.png, pr_curves_K10.png, fdr_calibration_K10.png,
              auprc_vs_K.png}
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
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pool_matched(S_true: np.ndarray, R_beta: np.ndarray, pi: np.ndarray):
    """Flatten matched fitted columns paired with their true-support binary vectors."""
    K_true = S_true.shape[1]
    y_true, y_score = [], []
    for k_fit in range(R_beta.shape[1]):
        k_true = int(pi[k_fit])
        if 0 <= k_true < K_true:
            y_true.append(S_true[:, k_true])
            y_score.append(R_beta[:, k_fit])
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(y_score)


def aggregate(raw_dir: str, config_path: str) -> tuple[pd.DataFrame, dict]:
    cfg = yaml.safe_load(open(config_path))
    cond_meta = {
        i: {"K_fit": c["K_fit"],
            "pi_label": c.get("pi_label", float("nan")),
            "generator_overrides": c.get("generator_overrides", {})}
        for i, c in enumerate(cfg["conditions"])
    }

    from src.generator import generate

    rows = []
    # Stash per-(K_fit, pi) pooled scores+labels for the PR / calibration plots
    pooled = {}

    for path in sorted(glob.glob(os.path.join(raw_dir, "cond*_seed*.npz"))):
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        rec = {k: d[k] for k in d.files}
        cidx = int(rec["condition_idx"])
        seed = int(rec["seed"])
        meta = cond_meta.get(cidx, {})
        if "drgp_unmasked" not in rec:
            continue
        m = rec["drgp_unmasked"].item() if rec["drgp_unmasked"].dtype == object else rec["drgp_unmasked"]
        if not isinstance(m, dict) or m.get("error"):
            continue
        if "R_beta" not in m or "pi" not in m:
            continue
        R_beta = np.asarray(m["R_beta"])
        pi = np.asarray(m["pi"])

        # Regenerate the dataset to recover the true support S
        gen_kwargs = dict(cfg["generator_defaults"])
        gen_kwargs.update(meta["generator_overrides"])
        if isinstance(gen_kwargs.get("overlap_pair"), list):
            gen_kwargs["overlap_pair"] = tuple(gen_kwargs["overlap_pair"])
        gen_kwargs["seed"] = seed
        gt = generate(**gen_kwargs)
        S_true = gt.S

        y_true, y_score = _pool_matched(S_true, R_beta, pi)
        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            continue

        auprc = float(average_precision_score(y_true, y_score))
        prevalence = float(y_true.mean())

        # FDR at r_beta > 0.5 (the principled threshold)
        called = y_score > 0.5
        if called.sum() > 0:
            fdr_at_0p5 = float(((called) & (y_true == 0)).sum()) / float(called.sum())
            recall_at_0p5 = float(((called) & (y_true == 1)).sum()) / float(max((y_true == 1).sum(), 1))
        else:
            fdr_at_0p5 = float("nan")
            recall_at_0p5 = 0.0

        # Per-program AUPRC (for diagnostics)
        per_prog = []
        for k_fit in range(R_beta.shape[1]):
            k_true = int(pi[k_fit])
            if 0 <= k_true < S_true.shape[1]:
                y_t = S_true[:, k_true]
                if len(np.unique(y_t)) >= 2:
                    per_prog.append(float(average_precision_score(y_t, R_beta[:, k_fit])))
        per_prog_mean = float(np.mean(per_prog)) if per_prog else float("nan")

        rows.append({
            "method": "drgp_unmasked",
            "seed": seed,
            "condition_idx": cidx,
            "K_fit": meta["K_fit"],
            "pi_label": meta["pi_label"],
            "support_auprc_pooled": auprc,
            "support_auprc_perprog_mean": per_prog_mean,
            "fdr_at_0p5": fdr_at_0p5,
            "recall_at_0p5": recall_at_0p5,
            "prevalence": prevalence,
        })

        # Pool a small subset (every 10th seed) for global PR / calibration plots
        key = (meta["K_fit"], meta["pi_label"])
        if seed % 10 == 0:
            pooled.setdefault(key, []).append((y_true.astype(np.int8), y_score.astype(np.float32)))

    return pd.DataFrame(rows), pooled


def plot_auprc_heatmap(df: pd.DataFrame, out_path: str) -> None:
    sub = df.dropna(subset=["support_auprc_pooled"])
    if sub.empty:
        return
    pivot = sub.pivot_table(index="K_fit", columns="pi_label",
                             values="support_auprc_pooled", aggfunc="median")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_xlabel(r"sparsity $\pi$"); ax.set_ylabel(r"$K_{\mathrm{fit}}$")
    ax.set_title("A2: median pooled AUPRC of r_beta")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if pivot.values[i, j] < 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def plot_pr_curves(pooled: dict, out_path: str, K_filter: int = 10) -> None:
    """PR curves for K_fit=K_filter, one panel per pi, averaged across pooled seeds."""
    keys = sorted([k for k in pooled if k[0] == K_filter], key=lambda k: k[1])
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4), sharey=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, (Kf, pi) in zip(axes, keys):
        # Pool all seeds in this condition into one PR curve
        all_y, all_s = [], []
        for y, s in pooled[(Kf, pi)]:
            all_y.append(y); all_s.append(s)
        y = np.concatenate(all_y); s = np.concatenate(all_s)
        prec, rec, _ = precision_recall_curve(y, s)
        auprc = average_precision_score(y, s)
        ax.plot(rec, prec, color="#2b6aa6", lw=1.6)
        ax.axhline(y.mean(), color="gray", linestyle="--", linewidth=0.7,
                    label=f"prevalence={y.mean():.3f}")
        ax.set_xlabel("recall")
        ax.set_title(f"K_fit={Kf}  pi={pi:.2f}  AUPRC={auprc:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("precision")
    fig.suptitle(f"A2: PR curves at K_fit={K_filter}  (pooled across seeds with seed%10==0)")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def plot_fdr_calibration(pooled: dict, out_path: str, K_filter: int = 10) -> None:
    """Empirical FDR vs nominal 1 - r_beta_threshold. Diagonal = well-calibrated."""
    keys = sorted([k for k in pooled if k[0] == K_filter], key=lambda k: k[1])
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#2b6aa6", "#aa5c2b", "#5b8a3a"]
    for (Kf, pi), c in zip(keys, colors):
        all_y, all_s = [], []
        for y, s in pooled[(Kf, pi)]:
            all_y.append(y); all_s.append(s)
        y = np.concatenate(all_y); s = np.concatenate(all_s)
        thresholds = np.linspace(0.01, 0.99, 50)
        nominal_fdrs, empirical_fdrs = [], []
        for thr in thresholds:
            called = s > thr
            if called.sum() == 0:
                continue
            empirical_fdrs.append(float(((called) & (y == 0)).sum()) / float(called.sum()))
            nominal_fdrs.append(1.0 - thr)
        ax.plot(nominal_fdrs, empirical_fdrs, "-o", ms=3, color=c,
                label=f"pi={pi:.2f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, label="y=x")
    ax.set_xlabel(r"Nominal $1 - r_\beta$ threshold")
    ax.set_ylabel("Empirical FDR")
    ax.set_title(f"A2: FDR calibration at K_fit={K_filter}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def plot_auprc_vs_K(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = df.dropna(subset=["support_auprc_pooled"])
    palette_pi = {0.05: "#3b6db8", 0.10: "#cc7a30", 0.20: "#5e933d"}
    for pi, c in palette_pi.items():
        s = sub[sub["pi_label"] == pi]
        if s.empty:
            continue
        g = s.groupby("K_fit")["support_auprc_pooled"]
        med = g.median()
        q25 = g.quantile(0.25); q75 = g.quantile(0.75)
        xs = med.index.values
        ax.plot(xs, med.values, "-o", color=c, label=f"pi={pi:.2f}")
        ax.fill_between(xs, q25.values, q75.values, alpha=0.18, color=c)
    ax.set_xlabel(r"$K_{\mathrm{fit}}$")
    ax.set_ylabel("AUPRC of r_beta (pooled, median)")
    ax.set_title("A2: support recovery vs K_fit")
    ax.set_ylim(0, 1)
    ax.axhline(0.80, color="black", linestyle=":", linewidth=0.6,
                label="plan target (0.80)")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/A1_factor_recovery")
    ap.add_argument("--config", default="configs/A1_factor_recovery.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/A2_support_recovery")
    ap.add_argument("--figure-dir", default="figures/A2")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df, pooled = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)
    print(f"Wrote {len(df)} rows to long.csv")

    g = df.groupby(["K_fit", "pi_label"]).agg(
        auprc_med=("support_auprc_pooled", "median"),
        auprc_q25=("support_auprc_pooled", lambda s: s.quantile(0.25)),
        auprc_q75=("support_auprc_pooled", lambda s: s.quantile(0.75)),
        fdr_at_0p5_med=("fdr_at_0p5", "median"),
        recall_at_0p5_med=("recall_at_0p5", "median"),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print(g.round(3).to_string(index=False))

    plot_auprc_heatmap(df, str(Path(args.figure_dir) / "auprc_heatmap.png"))
    plot_pr_curves(pooled, str(Path(args.figure_dir) / "pr_curves_K10.png"), K_filter=10)
    plot_fdr_calibration(pooled, str(Path(args.figure_dir) / "fdr_calibration_K10.png"), K_filter=10)
    plot_auprc_vs_K(df, str(Path(args.figure_dir) / "auprc_vs_K.png"))

    print(f"\nFigures: {args.figure_dir}/")


if __name__ == "__main__":
    main()
