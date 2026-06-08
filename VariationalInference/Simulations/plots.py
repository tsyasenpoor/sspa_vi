"""All 8 manuscript figures from metrics.parquet + bottleneck.parquet + stability.parquet."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config

FIG_DIR = config.SIM_ROOT / "figures"


def _load_scalar() -> pd.DataFrame:
    df = pd.read_parquet(config.SIM_ROOT / "metrics.parquet")
    df = df[df["metric_kind"] == "scalar"].copy()
    return df


def auc_vs_r(K: int = 10) -> Path:
    df = _load_scalar()
    df = df[df["K"] == K]
    g = df.groupby(["method", "mode", "r"])["cell_auc_integrated"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for (m, mode), sub in g.groupby(["method", "mode"]):
        ax.errorbar(sub["r"], sub["mean"], yerr=sub["std"], label=f"{m}/{mode}", marker="o")
    ax.set_xlabel("r (subdominance ratio)"); ax.set_ylabel("cell AUC (integrated)")
    ax.set_title(f"AUC vs r  (K={K})"); ax.legend(loc="best", fontsize=8)
    p = FIG_DIR / f"auc_vs_r_K{K}.png"; FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); return p


def auc_vs_h2(r: float = 0.15, K: int = 10) -> Path:
    df = _load_scalar()
    df = df[(df["K"] == K) & (df["r"] == r)]
    g = df.groupby(["method", "mode", "h2"])["cell_auc_integrated"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for (m, mode), sub in g.groupby(["method", "mode"]):
        ax.errorbar(sub["h2"], sub["mean"], yerr=sub["std"], label=f"{m}/{mode}", marker="o")
    ax.set_xlabel("h^2"); ax.set_ylabel("cell AUC (integrated)")
    ax.set_title(f"AUC vs h^2  (r={r}, K={K})"); ax.legend(loc="best", fontsize=8)
    p = FIG_DIR / f"auc_vs_h2_r{r}_K{K}.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); return p


def gap_vs_K(h2: float = 0.3, r: float = 0.15) -> Path:
    df = _load_scalar()
    df = df[(df["h2"] == h2) & (df["r"] == r)]
    drgp_mean = df[df["method"].str.startswith("drgp_")].groupby(["mode", "K"])["cell_auc_integrated"].mean()
    drgp_best = drgp_mean.groupby("K").max()
    unsup_mean = df[df["method"].isin(["nmf_lr", "schpf_lr", "spectra_lr"])].groupby(["method", "K"])["cell_auc_integrated"].mean()
    unsup_best = unsup_mean.groupby("K").max()
    Ks = sorted(set(drgp_best.index) & set(unsup_best.index))
    gap = [drgp_best[k] - unsup_best[k] for k in Ks]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(Ks, gap, marker="o", linewidth=2)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("K"); ax.set_ylabel("AUC gap (best DRGP - best unsup)")
    ax.set_title(f"Gap vs K  (h^2={h2}, r={r})")
    p = FIG_DIR / "gap_vs_K.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); return p


def matched_cosine_vs_r(K: int = 10) -> Path:
    df = pd.read_parquet(config.SIM_ROOT / "metrics.parquet")
    df = df[(df["metric_kind"] == "matched_cosine_per_l") & (df["K"] == K)]
    g = df.groupby(["method", "mode", "r"])["value"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for (m, mode), sub in g.groupby(["method", "mode"]):
        ax.errorbar(sub["r"], sub["mean"], yerr=sub["std"], label=f"{m}/{mode}", marker="o")
    ax.set_xlabel("r"); ax.set_ylabel("matched cosine")
    ax.set_title(f"Recovery vs r  (K={K})"); ax.legend(loc="best", fontsize=8)
    p = FIG_DIR / f"matched_cosine_vs_r_K{K}.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); return p


def splitting_table(out_csv: bool = True) -> Path:
    df = pd.read_parquet(config.SIM_ROOT / "metrics.parquet")
    sub = df[df["metric_kind"].isin(["splitting_concentration", "splitting_coverage"])]
    table = sub.pivot_table(index=["method", "mode"], columns="metric_kind",
                            values="value", aggfunc="mean")
    p = FIG_DIR / "splitting_table.csv"
    table.to_csv(p); return p


def stability_bars() -> Path:
    df = pd.read_parquet(config.SIM_ROOT / "stability.parquet")
    g = df.groupby("method")[["causal_subset_mean_cos", "other_mean_cos"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(g))
    w = 0.4
    ax.bar(x - w/2, g["causal_subset_mean_cos"], w, label="causal subset")
    ax.bar(x + w/2, g["other_mean_cos"], w, label="other")
    ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=30, ha="right")
    ax.set_ylabel("pairwise matched cosine"); ax.set_title("Sec 7.6 stability"); ax.legend()
    p = FIG_DIR / "stability_bars.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); return p


def ranking_table() -> Path:
    df = _load_scalar()
    cols = ["method", "mode", "h2", "r", "K", "ranking_auc", "decoy_rank"]
    out = df[cols].groupby(["method", "mode", "h2", "r", "K"]).mean().reset_index()
    p = FIG_DIR / "ranking_table.csv"
    out.to_csv(p, index=False); return p


def bottleneck_scatter(h2: float = 0.3, r: float = 0.15) -> Path:
    df = pd.read_parquet(config.SIM_ROOT / "bottleneck.parquet")
    sub = df[(df["h2"] == h2) & (df["r"] == r)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for (m, mode), s in sub.groupby(["method", "mode"]):
        ax.scatter(s["hit_rate"], s["cell_auc_integrated"], label=f"{m}/{mode}", alpha=0.6)
    ax.set_xlabel("recovery hit-rate"); ax.set_ylabel("cell AUC (integrated)")
    ax.set_title(f"Bottleneck (h^2={h2}, r={r})"); ax.legend(fontsize=8)
    p = FIG_DIR / f"bottleneck_h{h2}_r{r}.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); return p


def run_all() -> list[Path]:
    outs = [auc_vs_r(), auc_vs_h2(), gap_vs_K(), matched_cosine_vs_r(),
            splitting_table(), stability_bars(), ranking_table(),
            bottleneck_scatter()]
    for o in outs:
        print(f"  wrote {o}")
    return outs
