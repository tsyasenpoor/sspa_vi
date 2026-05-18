"""
B2 analysis: recovery of the covariate -> program -> phenotype mediation path.

For each (cond, seed):
  1. Re-generate the dataset (deterministic from seed) to recover gt.X, gt.Delta, gt.v.
  2. Load Theta_hat (shape (n, K_fit)) and the Hungarian pi mapping fitted columns
     to true programs.
  3. For each FITTED program k_fit, regress Theta_hat[:, k_fit] on X (OLS) to get
     delta_hat_row in R^q. Assemble Delta_hat in true-program-order via pi.
  4. Pearson r over all (k, ell) entries of (Delta_hat, Delta_true), restricted
     to matched rows.
  5. For the asthma-mediated program (true k=0), report:
       - delta_hat[0, asthma]   (recovered slope)
       - delta_true[0, asthma]  (planted strength; varies per condition)
       - indirect effect estimate IE_hat = v_hat[matched-to-true-0] * delta_hat[0, asthma]
       - IE_true = v_true[0] * delta_true[0, asthma]
       - Bias % = (IE_hat - IE_true) / |IE_true|

Outputs:
  results/aggregated/B2_mediation_delta_sweep/long.csv
  results/aggregated/B2_mediation_delta_sweep/summary.csv
  figures/B2/{delta_scatter, ie_bias, pearson_vs_delta}.png
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
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Covariate index for asthma (column 2 of X by generator convention)
ASTHMA = 2


def _ols_delta_per_column(Theta_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Regress each column of Theta_hat (n, K_fit) on X (n, q) via OLS *without* intercept
    (the intercept is absorbed by the gamma prior mean in the true model and is not
    part of Delta). Returns Delta_hat_fit_order shape (K_fit, q).

    Note: this matches the plan's prescription -- "regress E[theta_{i,k}] on X (OLS)".
    The estimand is biased because Theta ~ Gamma(a + max(X.Delta, 0), b), not
    Theta = X.Delta + noise; but Pearson(Delta_hat, Delta_true) is still informative.
    """
    # Add intercept column to control for non-zero baseline (a_theta/b_theta)
    Xc = np.hstack([np.ones((X.shape[0], 1)), X])                   # (n, 1+q)
    # Solve (Xc' Xc) beta = Xc' Theta_hat for each column
    coef, *_ = np.linalg.lstsq(Xc, Theta_hat, rcond=None)            # (1+q, K_fit)
    # Drop the intercept row; transpose to (K_fit, q)
    return coef[1:].T


def aggregate(raw_dir: str, config_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    cond_meta = {
        i: {"K_fit": c["K_fit"],
            "delta_label": c.get("delta_label", float("nan")),
            "generator_overrides": c.get("generator_overrides", {})}
        for i, c in enumerate(cfg["conditions"])
    }

    from src.generator import generate

    rows = []
    paths = sorted(glob.glob(os.path.join(raw_dir, "cond*_seed*.npz")))
    for path in paths:
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            continue
        rec = {k: d[k] for k in d.files}
        cidx = int(rec["condition_idx"])
        meta = cond_meta.get(cidx, {})

        # Regenerate the dataset to recover ground truth X, Delta, v, rel_idx
        gen_kwargs = dict(cfg["generator_defaults"])
        gen_kwargs.update(meta["generator_overrides"])
        if "delta_entries" in gen_kwargs:
            entries = gen_kwargs.pop("delta_entries")
            gen_kwargs["delta_spec"] = {(int(k), int(ell)): float(v) for k, ell, v in entries}
        if isinstance(gen_kwargs.get("overlap_pair"), list):
            gen_kwargs["overlap_pair"] = tuple(gen_kwargs["overlap_pair"])
        gen_kwargs["seed"] = int(rec["seed"])
        gt = generate(**gen_kwargs)
        Delta_true = gt.Delta        # (K_true, q)
        v_true = gt.v                # (K_true,)
        X_train = gt.X               # (n, q)
        K_true = Delta_true.shape[0]

        # For each method, compute the mediation metrics
        for method in cfg["methods"]:
            if method not in rec:
                continue
            m = rec[method].item() if rec[method].dtype == object else rec[method]
            if not isinstance(m, dict) or m.get("error"):
                continue
            Theta_hat = m.get("Theta_hat")
            pi = m.get("pi")
            v_hat = m.get("v_hat")
            if Theta_hat is None or pi is None or v_hat is None:
                continue
            Theta_hat = np.asarray(Theta_hat)
            pi = np.asarray(pi)
            v_hat = np.asarray(v_hat)

            # OLS Delta_hat in fit order (K_fit, q)
            Delta_hat_fit = _ols_delta_per_column(Theta_hat, X_train)

            # Align to true-program order: Delta_hat_aligned[k_true] = Delta_hat_fit[k_fit]
            Delta_hat_aligned = np.full_like(Delta_true, np.nan)
            v_hat_aligned = np.full(K_true, np.nan)
            for k_fit in range(Delta_hat_fit.shape[0]):
                k_true = int(pi[k_fit])
                if 0 <= k_true < K_true:
                    Delta_hat_aligned[k_true] = Delta_hat_fit[k_fit]
                    if k_fit < v_hat.size:
                        v_hat_aligned[k_true] = v_hat[k_fit]

            # Pearson over all matched (k, ell)
            mask = ~np.isnan(Delta_hat_aligned).any(axis=1)
            if mask.sum() >= 2:
                a = Delta_hat_aligned[mask].ravel()
                b = Delta_true[mask].ravel()
                if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                    r, _ = pearsonr(a, b)
                else:
                    r = float("nan")
            else:
                r = float("nan")

            # Indirect effect for the asthma-mediated program (true k=0)
            delta_hat_0_asthma = float(Delta_hat_aligned[0, ASTHMA]) if not np.isnan(Delta_hat_aligned[0, ASTHMA]) else float("nan")
            delta_true_0_asthma = float(Delta_true[0, ASTHMA])
            v_hat_0  = float(v_hat_aligned[0]) if not np.isnan(v_hat_aligned[0]) else float("nan")
            v_true_0 = float(v_true[0])
            IE_hat  = v_hat_0 * delta_hat_0_asthma
            IE_true = v_true_0 * delta_true_0_asthma
            bias_pct = (IE_hat - IE_true) / abs(IE_true) if abs(IE_true) > 1e-9 else float("nan")

            rows.append({
                "method": method,
                "seed": int(rec["seed"]),
                "condition_idx": cidx,
                "K_fit": meta["K_fit"],
                "delta_label": meta["delta_label"],
                "pearson_delta": r,
                "delta_hat_0_asthma": delta_hat_0_asthma,
                "delta_true_0_asthma": delta_true_0_asthma,
                "v_hat_0": v_hat_0,
                "v_true_0": v_true_0,
                "ie_hat": IE_hat,
                "ie_true": IE_true,
                "ie_bias_pct": bias_pct,
                "cos_mean": float(m.get("cos_mean", float("nan"))),
                "v_spearman": float(m.get("v_spearman", float("nan"))),
                "ood_auroc": float(m.get("ood_auroc", float("nan"))),
            })
    return pd.DataFrame(rows)


def plot_delta_scatter(df: pd.DataFrame, out_path: str) -> None:
    """Scatter of recovered Delta[0, asthma] vs planted, across seeds and conditions."""
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    palette = {"drgp_unmasked": "#2b6aa6", "nmf_lr": "#aa5c2b", "pca_lr": "#888888"}
    methods = sorted(df["method"].unique())
    for method in methods:
        sub = df[df["method"] == method].dropna(subset=["delta_hat_0_asthma", "delta_true_0_asthma"])
        ax.scatter(sub["delta_true_0_asthma"], sub["delta_hat_0_asthma"],
                    s=14, alpha=0.5, color=palette.get(method, "k"), label=method)
    lo = min(df["delta_true_0_asthma"].min(), df["delta_hat_0_asthma"].min()) - 0.2
    hi = max(df["delta_true_0_asthma"].max(), df["delta_hat_0_asthma"].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6, label="y=x")
    ax.set_xlabel(r"$\Delta_{0,\mathrm{asthma}}$ (planted)")
    ax.set_ylabel(r"$\hat{\Delta}_{0,\mathrm{asthma}}$ (OLS on $\hat\Theta$)")
    ax.set_title("Recovery of mediation strength")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pearson_vs_delta(df: pd.DataFrame, out_path: str) -> None:
    """Pearson r(Delta_hat, Delta_true) vs planted Delta[0,asthma], median + IQR."""
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    palette = {"drgp_unmasked": "#2b6aa6", "nmf_lr": "#aa5c2b", "pca_lr": "#888888"}
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].dropna(subset=["pearson_delta"])
        if sub.empty:
            continue
        g = sub.groupby("delta_label")["pearson_delta"]
        med = g.median(); q25 = g.quantile(0.25); q75 = g.quantile(0.75)
        xs = med.index.values
        ax.plot(xs, med.values, "-o", label=method, color=palette.get(method, "k"))
        ax.fill_between(xs, q25.values, q75.values, alpha=0.18,
                        color=palette.get(method, "k"))
    ax.set_xlabel(r"$\Delta_{0,\mathrm{asthma}}$ (planted)")
    ax.set_ylabel(r"Pearson $r(\hat{\Delta}, \Delta_{\mathrm{true}})$")
    ax.set_title("Full mediation matrix recovery")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_ie_bias(df: pd.DataFrame, out_path: str) -> None:
    """Boxplot of indirect-effect bias % by method and Delta condition."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    methods = sorted(df["method"].unique())
    deltas = sorted(df["delta_label"].unique())
    width = 0.8 / max(len(methods), 1)
    palette = {"drgp_unmasked": "#2b6aa6", "nmf_lr": "#aa5c2b", "pca_lr": "#888888"}
    for i, method in enumerate(methods):
        positions = [j + (i - (len(methods)-1)/2) * width for j in range(len(deltas))]
        data = [df[(df["method"] == method) & (df["delta_label"] == d)]["ie_bias_pct"].dropna().values
                for d in deltas]
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                         patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set_facecolor(palette.get(method, "k"))
            box.set_alpha(0.6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([f"{d:.1f}" for d in deltas])
    ax.set_xlabel(r"$\Delta_{0,\mathrm{asthma}}$ (planted)")
    ax.set_ylabel("Indirect-effect bias (fraction of |IE_true|)")
    ax.set_title("Indirect-effect estimation bias")
    ax.grid(axis="y", alpha=0.25)
    # Custom legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=palette.get(m, "k"), alpha=0.6, label=m)
                        for m in methods])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/B2_mediation_delta_sweep")
    ap.add_argument("--config", default="configs/B2_mediation_delta_sweep.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/B2_mediation_delta_sweep")
    ap.add_argument("--figure-dir", default="figures/B2")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)
    print(f"Wrote {len(df)} rows to long.csv")

    g = df.groupby(["method", "delta_label"]).agg(
        pearson_med=("pearson_delta", "median"),
        pearson_q25=("pearson_delta", lambda s: s.quantile(0.25)),
        pearson_q75=("pearson_delta", lambda s: s.quantile(0.75)),
        delta_hat_med=("delta_hat_0_asthma", "median"),
        ie_bias_med=("ie_bias_pct", "median"),
        ie_bias_q25=("ie_bias_pct", lambda s: s.quantile(0.25)),
        ie_bias_q75=("ie_bias_pct", lambda s: s.quantile(0.75)),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print(g.round(3).to_string(index=False))

    plot_delta_scatter(df, str(Path(args.figure_dir) / "delta_scatter.png"))
    plot_pearson_vs_delta(df, str(Path(args.figure_dir) / "pearson_vs_delta.png"))
    plot_ie_bias(df, str(Path(args.figure_dir) / "ie_bias_box.png"))

    print(f"\nFigures: {args.figure_dir}/")


if __name__ == "__main__":
    main()
