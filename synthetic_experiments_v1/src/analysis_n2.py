"""
N2 analysis: joint (DRGP) vs two-stage (scHPF + post-hoc OLS/LR) mediation.

Reuses the B2 aggregator on N2 raw -- both share the same npz schema (saved
Theta_hat + Hungarian pi + v_hat, with delta_entries per condition).

Reports, per method, per Delta:
  - Pearson r(Delta_hat, Delta_true)
  - delta_hat[0, asthma] (recovered)
  - indirect effect bias % vs true IE
Computes:
  - Per-method bias mean / IQR
  - 95% CI coverage : for each (method, delta), fraction of seeds for which
    a 1.96 * SE_bootstrap interval covers IE_true. (SE estimated from seed
    variance within the condition.)

Outputs:
  results/aggregated/N2_joint_vs_twostage/{long.csv, summary.csv, coverage.csv}
  figures/N2/{ie_bias_by_method.png, caterpillar.png, coverage_table.png}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the B2 aggregator -- identical npz schema
from src.analysis_b2 import aggregate                                       # noqa: E402


PALETTE = {"drgp_unmasked": "#2b6aa6", "schpf_lr": "#9c2b6a"}


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each (method, delta), estimate SE(ie_hat) from seed variance and
    compute the fraction of seeds whose +-1.96*SE interval covers ie_true.
    """
    out = []
    for (method, delta), sub in df.groupby(["method", "delta_label"]):
        s = sub.dropna(subset=["ie_hat", "ie_true"])
        if len(s) < 5:
            continue
        # SE estimated as std across seeds within this (method, delta) cell
        se = float(s["ie_hat"].std())
        ie_true = float(s["ie_true"].iloc[0])  # constant per (method, seed)? no, per seed too
        # Use per-seed CI = ie_hat ± 1.96 * se ; count seeds where ie_true is inside
        lo = s["ie_hat"] - 1.96 * se
        hi = s["ie_hat"] + 1.96 * se
        covered = ((s["ie_true"] >= lo) & (s["ie_true"] <= hi)).mean()
        out.append({
            "method": method, "delta_label": delta,
            "se_ie_hat": se,
            "mean_ie_hat": float(s["ie_hat"].mean()),
            "mean_ie_true": float(s["ie_true"].mean()),
            "coverage_95": float(covered),
            "n": len(s),
        })
    return pd.DataFrame(out)


def plot_ie_bias_by_method(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    methods = sorted([m for m in df["method"].unique() if m in PALETTE])
    deltas = sorted(df["delta_label"].unique())
    width = 0.85 / max(len(methods), 1)
    for i, method in enumerate(methods):
        positions = [j + (i - (len(methods)-1)/2) * width for j in range(len(deltas))]
        data = [df[(df["method"] == method) & (df["delta_label"] == d)]["ie_bias_pct"].dropna().values
                for d in deltas]
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", linewidth=1.4))
        for box in bp["boxes"]:
            box.set_facecolor(PALETTE[method])
            box.set_alpha(0.65)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels([f"{d:.1f}" for d in deltas])
    ax.set_xlabel(r"$\Delta_{0,\mathrm{asthma}}$ (planted)")
    ax.set_ylabel("indirect-effect bias (fraction of |IE_true|)")
    ax.set_title("N2: joint (DRGP) vs two-stage (scHPF+LR) indirect-effect bias")
    ax.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PALETTE[m], alpha=0.65, label=m) for m in methods])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180); plt.close()


def plot_caterpillar(df: pd.DataFrame, out_path: str) -> None:
    """Per-seed IE estimates with 1.96*SE bars, two columns (DRGP, scHPF), all Delta."""
    methods = ["drgp_unmasked", "schpf_lr"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method].dropna(subset=["ie_hat", "ie_true"]).sort_values(
            ["delta_label", "seed"])
        if sub.empty:
            ax.set_title(f"{method} (no data)")
            continue
        se = float(sub["ie_hat"].std())  # global SE estimate
        y = np.arange(len(sub))
        ax.errorbar(sub["ie_hat"], y, xerr=1.96 * se, fmt="o", ms=2,
                     ecolor=PALETTE[method], color=PALETTE[method], alpha=0.6,
                     elinewidth=0.6)
        # Mark IE_true for reference
        ax.scatter(sub["ie_true"], y, marker="|", s=30, color="black",
                    alpha=0.5, label="ie_true")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("indirect effect")
        ax.set_title(method)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("seed × Delta (sorted)")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw/N2_joint_vs_twostage")
    ap.add_argument("--config",  default="configs/N2_joint_vs_twostage.yaml")
    ap.add_argument("--out-dir", default="results/aggregated/N2_joint_vs_twostage")
    ap.add_argument("--figure-dir", default="figures/N2")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)

    df = aggregate(args.raw_dir, args.config)
    df.to_csv(Path(args.out_dir) / "long.csv", index=False)
    print(f"Wrote {len(df)} rows to long.csv")

    g = df.groupby(["method", "delta_label"]).agg(
        pearson_med=("pearson_delta", "median"),
        ie_bias_med=("ie_bias_pct", "median"),
        ie_bias_iqr_low=("ie_bias_pct", lambda s: s.quantile(0.25)),
        ie_bias_iqr_high=("ie_bias_pct", lambda s: s.quantile(0.75)),
        ie_bias_abs_med=("ie_bias_pct", lambda s: s.abs().median()),
        n_seeds=("seed", "count"),
    ).reset_index()
    g.to_csv(Path(args.out_dir) / "summary.csv", index=False)
    print("\nPer-method summary:")
    print(g.round(3).to_string(index=False))

    cov = coverage_table(df)
    cov.to_csv(Path(args.out_dir) / "coverage.csv", index=False)
    print("\n95% CI coverage of IE_true:")
    print(cov.round(3).to_string(index=False))

    # Pairwise Wilcoxon |IE bias| at each delta
    print("\nPaired Wilcoxon |ie_bias_pct|: DRGP vs scHPF (per seed)")
    for delta, sub in df.groupby("delta_label"):
        dr = sub[sub["method"] == "drgp_unmasked"][["seed", "ie_bias_pct"]].set_index("seed")
        sh = sub[sub["method"] == "schpf_lr"][["seed", "ie_bias_pct"]].set_index("seed")
        j = dr.join(sh, lsuffix="_drgp", rsuffix="_schpf").dropna()
        if len(j) < 5:
            continue
        try:
            stat, p = wilcoxon(j["ie_bias_pct_drgp"].abs(), j["ie_bias_pct_schpf"].abs())
        except Exception:
            p = float("nan")
        diff = (j["ie_bias_pct_drgp"].abs() - j["ie_bias_pct_schpf"].abs()).median()
        print(f"  delta={delta:.1f}  med(|DRGP| - |scHPF|)={diff:+.3f}  n={len(j)}  p={p:.4g}")

    plot_ie_bias_by_method(df, str(Path(args.figure_dir) / "ie_bias_by_method.png"))
    plot_caterpillar(df, str(Path(args.figure_dir) / "caterpillar.png"))
    print(f"\nFigures: {args.figure_dir}/")


if __name__ == "__main__":
    main()
