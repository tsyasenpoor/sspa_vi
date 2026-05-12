#!/usr/bin/env python
"""
Analyze masked (drgp_masked) multi-seed VI results for a single method directory.
=================================================================================

Loads per-seed artifacts from ``{path}/seed_*/{method_subdir}/`` and produces:

  - ``{path}/figures/``  all PNGs (ROC/KDE panel, v_weight dists, A-F plots)
  - ``{path}/tables/``   all aggregated CSVs (vweight summary, shared genes,
                         mixed-sign pathways)

Per-seed gene_programs_filtered.csv.gz is still written next to its source seed
artifacts since it is a per-seed derivative.

Usage:
    python /labs/Aguiar/SSPA_BRAY/BRay/VariationalInference/analysis-utils/analyze_masked_results.py \
        --path /labs/Aguiar/SSPA_BRAY/results/biorepo_vi_masked_reactome/allPBMC_GEX_20260106
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde, kendalltau, spearmanr
from sklearn.metrics import auc, roc_curve

SEED_COLORS = {42: "#1f77b4", 123: "#ff7f0e", 456: "#2ca02c", 789: "#d62728", 1024: "#9467bd"}
SPLIT_STYLES = {
    "Train":      {"ls": "-",  "lw": 1.5},
    "Validation": {"ls": "--", "lw": 1.2},
    "Test":       {"ls": ":",  "lw": 1.2},
}
TARGETS = [
    ("CoVID-19 severity", "Severity"),
    ("Outcome",           "Outcome"),
]
LABELS = ["Outcome", "CoVID-19 severity"]


def detect_labels(pred_df: pd.DataFrame) -> tuple[list[tuple[str, str]], list[str]]:
    targets = []
    for col in pred_df.columns:
        if col.startswith("prob_"):
            raw = col[len("prob_"):]
            targets.append((raw, raw.replace("_", " ").upper()))
    labels = [t[0] for t in targets]
    return targets, labels


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def _resolve_col(df: pd.DataFrame, prefix: str, target_label: str) -> str:
    cols = df.columns.tolist()
    exact = f"{prefix}_{target_label}"
    if exact in df.columns:
        return exact
    lower_map = {c.lower(): c for c in cols}
    if exact.lower() in lower_map:
        return lower_map[exact.lower()]
    pref = _norm(prefix)
    tgt = _norm(target_label)
    for c in cols:
        cn = _norm(c)
        if cn.startswith(pref) and tgt in cn:
            return c
    raise KeyError(f"No column for prefix={prefix!r} target={target_label!r}. Available: {cols}")


def _get_prob(df: pd.DataFrame, target_label: str) -> np.ndarray:
    return df[_resolve_col(df, "prob", target_label)].to_numpy(dtype=np.float64, copy=False)


def discover_seeds(base_dir: Path, method_subdir: str) -> list[tuple[int, Path]]:
    """Return (seed_int, seed_dir) pairs.  Handles both seed_N and seedN layouts."""
    results = []
    for p in sorted(base_dir.glob("seed*")):
        m = re.match(r"seed_?(\d+)$", p.name)
        if not m:
            continue
        seed = int(m.group(1))
        nested = p / method_subdir
        if nested.is_dir():
            results.append((seed, nested))
        elif (p / "vi_theta_train.csv.gz").exists():
            results.append((seed, p))
    return results


def load_seed(seed_dir: Path) -> dict:
    r = {
        "vi_gamma_variance":    pd.read_csv(seed_dir / "vi_gamma_variance.csv.gz", compression="gzip"),
        "vi_gamma_weights":     pd.read_csv(seed_dir / "vi_gamma_weights.csv.gz", compression="gzip"),
        "vi_gene_programs":     pd.read_csv(seed_dir / "vi_gene_programs.csv.gz", compression="gzip"),
        "vi_metrics":           pd.read_csv(seed_dir / "vi_metrics.csv"),
        "vi_test_predictions":  pd.read_csv(seed_dir / "vi_test_predictions.csv.gz", compression="gzip"),
        "vi_theta_train":       pd.read_csv(seed_dir / "vi_theta_train.csv.gz", compression="gzip"),
        "vi_theta_val":         pd.read_csv(seed_dir / "vi_theta_val.csv.gz", compression="gzip"),
        "vi_theta_test":        pd.read_csv(seed_dir / "vi_theta_test.csv.gz", compression="gzip"),
        "vi_train_predictions": pd.read_csv(seed_dir / "vi_train_predictions.csv.gz", compression="gzip"),
        "vi_val_predictions":   pd.read_csv(seed_dir / "vi_val_predictions.csv.gz", compression="gzip"),
        "r_beta":               pd.read_csv(seed_dir / "vi_r_beta.csv.gz", compression="gzip"),
    }
    with gzip.open(seed_dir / "vi_summary.json.gz", "rt", encoding="utf-8") as f:
        r["vi_summary"] = json.load(f)
    return r


def _seed_color(seed: int, idx: int) -> str:
    return SEED_COLORS.get(seed, plt.cm.tab10(idx % 10))


def plot_roc_kde_panel(all_results: dict, seeds: list[int], out_path: Path) -> None:
    fig, axes = plt.subplots(nrows=len(seeds), ncols=2, figsize=(12, 4 * len(seeds)), squeeze=False)
    for row_idx, seed in enumerate(seeds):
        color = _seed_color(seed, row_idx)
        r = all_results[seed]
        splits = [
            ("Train",      r["vi_train_predictions"]),
            ("Validation", r["vi_val_predictions"]),
            ("Test",       r["vi_test_predictions"]),
        ]
        for col_idx, panel in enumerate(["ROC", "KDE"]):
            ax = axes[row_idx, col_idx]
            for target_label, target_name in TARGETS:
                for split_name, pred_df in splits:
                    ls = SPLIT_STYLES[split_name]["ls"]
                    lw = SPLIT_STYLES[split_name]["lw"]
                    true_col = _resolve_col(pred_df, "true", target_label)
                    y_true = pred_df[true_col].to_numpy(dtype=np.int8, copy=False)
                    y_prob = _get_prob(pred_df, target_label)
                    if panel == "ROC":
                        if np.unique(y_true).size >= 2:
                            fpr, tpr, _ = roc_curve(y_true, y_prob)
                            ax.plot(fpr, tpr, color=color, linewidth=lw, linestyle=ls,
                                    label=f"{target_name} | {split_name} (AUC={auc(fpr, tpr):.3f})")
                        else:
                            ax.plot([], [], color=color, linewidth=lw, linestyle=ls,
                                    label=f"{target_name} | {split_name} (AUC=NA)")
                    else:
                        for cls_val, cls_label in [(0, "Class 0"), (1, "Class 1")]:
                            subset = y_prob[y_true == cls_val]
                            if subset.size > 1:
                                x_kde = np.linspace(np.percentile(subset, 0.5),
                                                    np.percentile(subset, 99.5), 200)
                                kde = gaussian_kde(subset)
                                ax.plot(x_kde, kde(x_kde), color=color, linestyle=ls, linewidth=1.2,
                                        alpha=0.7,
                                        label=f"{target_name} | {split_name} | {cls_label}")
            if panel == "ROC":
                ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
                ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.02)
                ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                ax.set_title(f"Seed {seed} | ROC")
            else:
                ax.axvline(x=0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.set_xlabel("Predicted Prob"); ax.set_ylabel("Density")
                ax.set_title(f"Seed {seed} | KDE")
            if row_idx == len(seeds) - 1:
                ax.legend(loc="best", fontsize=7, ncol=2)
            else:
                ax.legend([], [], frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vweight_histograms(all_results: dict, seeds: list[int], out_path: Path) -> None:
    targets = [f"v_weight_{L}" for L in LABELS]
    fig, axes = plt.subplots(nrows=len(targets), ncols=len(seeds),
                             figsize=(4 * len(seeds), 4 * len(targets)), squeeze=False)
    for row_idx, target_col in enumerate(targets):
        for idx, seed in enumerate(seeds):
            ax = axes[row_idx, idx]
            df = all_results[seed]["vi_gene_programs"]
            if target_col not in df.columns:
                raise KeyError(f"{target_col} not found. Available: {list(df.columns)}")
            ax.hist(pd.to_numeric(df[target_col], errors="coerce").dropna(),
                    bins=30, color=_seed_color(seed, idx), alpha=0.8)
            ax.set_title(f"Seed {seed}")
            ax.set_xlabel(target_col); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_gene_programs_filtered(all_results: dict, seeds: list[int], seed_dirs: dict,
                                 beta_threshold: float) -> None:
    v_weight_cols = [f"v_weight_{L}" for L in LABELS]
    for seed in seeds:
        r = all_results[seed]
        gp_df = r["vi_gene_programs"].copy()
        rb_df = r["r_beta"].copy()
        gene_cols = gp_df.columns[gp_df.columns.str.lower().str.startswith("ensg")]
        if "Unnamed: 0" in rb_df.columns:
            rb_df = rb_df.set_index("Unnamed: 0")
        if "Unnamed: 0" in gp_df.columns:
            gp_df = gp_df.set_index("Unnamed: 0")
        missing = [c for c in v_weight_cols if c not in gp_df.columns]
        if missing:
            raise KeyError(f"Missing v_weight columns {missing}. Available: {list(gp_df.columns)}")
        rows = []
        for gp in rb_df.index:
            beta_row = rb_df.loc[gp, gene_cols].astype(float)
            selected_mask = beta_row > beta_threshold
            selected_genes = gene_cols[selected_mask.to_numpy()]
            v_weights = {c: float(gp_df.loc[gp, c]) for c in v_weight_cols}
            n_genes = int(selected_mask.sum())
            for gene in selected_genes:
                rows.append({
                    "gene_program": gp, **v_weights,
                    "n_genes_in_program": n_genes,
                    "ensembl_id": gene,
                    "beta": float(beta_row[gene]),
                    "expression": float(gp_df.loc[gp, gene]),
                })
        out = seed_dirs[seed] / "gene_programs_filtered.csv.gz"
        pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")
        print(f"  seed {seed}: {len(rows)} gene-GP pairs → {out}")


def build_pathway_summary(all_results: dict, seeds: list[int]
                          ) -> tuple[pd.DataFrame, dict]:
    pathway_names = None
    pathway_df = None
    for seed in seeds:
        gp_df = all_results[seed]["vi_gene_programs"].copy()
        if "Unnamed: 0" in gp_df.columns:
            gp_df = gp_df.set_index("Unnamed: 0")
        if pathway_names is None:
            pathway_names = list(gp_df.index)
            pathway_df = pd.DataFrame(index=pathway_names)
            pathway_df.index.name = "pathway"
        elif list(gp_df.index) != pathway_names:
            raise ValueError(f"Pathway list mismatch at seed {seed}")
        for L in LABELS:
            col = f"v_weight_{L}"
            if col not in gp_df.columns:
                raise KeyError(f"Missing {col} at seed {seed}. Available: {list(gp_df.columns)}")
            pathway_df[f"v_weight_{L}_seed{seed}"] = gp_df[col].astype(float).values

    for L in LABELS:
        v_cols = [f"v_weight_{L}_seed{s}" for s in seeds]
        pathway_df[f"v_weight_{L}_mean"]   = pathway_df[v_cols].mean(axis=1)
        pathway_df[f"v_weight_{L}_std"]    = pathway_df[v_cols].std(axis=1)
        pathway_df[f"abs_mean_{L}"]        = pathway_df[f"v_weight_{L}_mean"].abs()
        pathway_df[f"all_positive_{L}"]    = (pathway_df[v_cols] > 0).all(axis=1)
        pathway_df[f"all_negative_{L}"]    = (pathway_df[v_cols] < 0).all(axis=1)
        pathway_df[f"sign_consistent_{L}"] = pathway_df[f"all_positive_{L}"] | pathway_df[f"all_negative_{L}"]
        abs_vals = pathway_df[v_cols].abs()
        for s in seeds:
            pathway_df[f"rank_{L}_seed{s}"] = abs_vals[f"v_weight_{L}_seed{s}"].rank(
                ascending=False, method="average")

    rank_corr = {}
    for L in LABELS:
        v_cols = [f"v_weight_{L}_seed{s}" for s in seeds]
        abs_vals = pathway_df[v_cols].abs()
        rho_mat = np.zeros((len(seeds), len(seeds)))
        tau_mat = np.zeros((len(seeds), len(seeds)))
        for i, _ in enumerate(seeds):
            for j, _ in enumerate(seeds):
                rho, _p = spearmanr(abs_vals.iloc[:, i], abs_vals.iloc[:, j])
                tau, _p = kendalltau(abs_vals.iloc[:, i], abs_vals.iloc[:, j])
                rho_mat[i, j] = rho
                tau_mat[i, j] = tau
        rank_corr[L] = {
            "spearman": pd.DataFrame(rho_mat, index=seeds, columns=seeds),
            "kendall":  pd.DataFrame(tau_mat, index=seeds, columns=seeds),
        }
    return pathway_df, rank_corr


def build_shared_gene_table(all_results: dict, seeds: list[int], pathway_df: pd.DataFrame,
                            top_n_per_label: int, beta_threshold: float) -> pd.DataFrame:
    top_per_label = {
        L: list(pathway_df.sort_values(f"abs_mean_{L}", ascending=False).head(top_n_per_label).index)
        for L in LABELS
    }
    all_top = sorted(set().union(*top_per_label.values()))

    pathway_shared_genes = {}
    for pw in all_top:
        sets = []
        for s in seeds:
            rb = all_results[s]["r_beta"].copy()
            if "Unnamed: 0" in rb.columns:
                rb = rb.set_index("Unnamed: 0")
            gene_cols = rb.columns[rb.columns.str.lower().str.startswith("ensg")]
            row = rb.loc[pw, gene_cols].astype(float)
            sets.append(set(gene_cols[(row > beta_threshold).to_numpy()]))
        pathway_shared_genes[pw] = set.intersection(*sets) if sets else set()

    all_ensg = sorted(set().union(*pathway_shared_genes.values()))
    ensg_to_symbol: dict[str, str] = {}
    if all_ensg:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            query = mg.querymany(all_ensg, scopes="ensembl.gene", fields="symbol",
                                 species="human", verbose=False)
            for hit in query:
                if "symbol" in hit:
                    ensg_to_symbol[hit["query"]] = hit["symbol"]
        except Exception as e:
            print(f"  [warn] mygene lookup skipped: {e}")

    rows = []
    for pw, genes in pathway_shared_genes.items():
        for e in sorted(genes):
            rows.append({"pathway": pw, "ensembl_id": e, "symbol": ensg_to_symbol.get(e, "")})
    return pd.DataFrame(rows)


def plot_panels_abcde(pathway_df: pd.DataFrame, rank_corr: dict, seeds: list[int],
                      figures_dir: Path, top_n: int = 20) -> None:
    for L in LABELS:
        safe_L = L.replace(" ", "_").replace("-", "_")
        v_cols = [f"v_weight_{L}_seed{s}" for s in seeds]
        top = pathway_df.sort_values(f"abs_mean_{L}", ascending=False).head(top_n).copy()

        # A. Rank-correlation heatmap
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(rank_corr[L]["spearman"], annot=True, fmt=".3f", cmap="YlGnBu",
                    vmin=0, vmax=1, square=True, cbar_kws={"label": "Spearman ρ"}, ax=ax)
        ax.set_title(f"{L}: rank correlation of |v_weight| across seeds",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Seed"); ax.set_ylabel("Seed")
        fig.tight_layout()
        fig.savefig(figures_dir / f"A_rank_corr_{safe_L}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # B. Dot plot of top-N v_weights across seeds
        y = np.arange(len(top))
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(top))))
        for idx, seed in enumerate(seeds):
            vals = top[f"v_weight_{L}_seed{seed}"].values
            ax.scatter(vals, y + (idx - 2) * 0.12, s=55, color=_seed_color(seed, idx),
                       zorder=3, edgecolor="white", linewidth=0.5)
        means = top[f"v_weight_{L}_mean"].values
        bar_colors = ["#d62728" if m > 0 else "#1f77b4" for m in means]
        ax.barh(y, means, height=0.6, color=bar_colors, alpha=0.15, zorder=1)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(top.index, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel(f"v_weight_{L}", fontsize=11)
        ax.set_title(f"{L}: top {top_n} pathways by |v_weight_mean|",
                     fontsize=13, fontweight="bold")
        # Vertical legend strip: colored dot + rotated label, to the right of the axes
        n_s = len(seeds)
        for idx, seed in enumerate(seeds):
            y_c = 1.0 - (idx + 0.5) / n_s
            ax.plot(1.015, y_c, "o", color=_seed_color(seed, idx), markersize=7,
                    transform=ax.transAxes, clip_on=False, zorder=5)
            ax.text(1.032, y_c, f"Seed {seed}", transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=8, color="#333333")
        fig.tight_layout()
        fig.savefig(figures_dir / f"B_vweight_dotplot_{safe_L}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # C. Heatmap of v_weights for top-N pathways across seeds
        heat = top[v_cols].copy()
        heat.columns = [f"Seed {s}" for s in seeds]
        vmax = float(max(abs(heat.values.min()), abs(heat.values.max()))) or 1e-6
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(heat))))
        sns.heatmap(heat, cmap="RdBu_r", norm=norm, annot=True, fmt=".3f",
                    linewidths=0.3, linecolor="white", ax=ax,
                    cbar_kws={"label": f"v_weight_{L}"})
        ax.set_title(f"{L}: v_weight across seeds for top {top_n} pathways",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(figures_dir / f"C_vweight_heatmap_{safe_L}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # D. |v_weight| scatter: seed[0] vs each other seed
        ref = seeds[0]
        others = seeds[1:]
        if others:
            fig, axes = plt.subplots(1, len(others), figsize=(3.3 * len(others), 3.4), squeeze=False)
            ref_vals = pathway_df[f"v_weight_{L}_seed{ref}"].abs()
            for i, s in enumerate(others):
                ax = axes[0, i]
                y_vals = pathway_df[f"v_weight_{L}_seed{s}"].abs()
                rho = rank_corr[L]["spearman"].loc[ref, s]
                ax.scatter(ref_vals, y_vals, s=12, alpha=0.6, color=_seed_color(s, i + 1))
                lim = float(max(ref_vals.max(), y_vals.max())) * 1.05
                ax.plot([0, lim], [0, lim], ls="--", color="black", lw=0.8)
                ax.set_xlim(0, lim); ax.set_ylim(0, lim)
                ax.set_xlabel(f"|v_weight| seed {ref}"); ax.set_ylabel(f"|v_weight| seed {s}")
                ax.set_title(f"seed {ref} vs {s}  (ρ={rho:.3f})", fontsize=10)
            fig.suptitle(f"{L}: magnitude agreement across seeds",
                         fontsize=12, fontweight="bold", y=1.02)
            fig.tight_layout()
            fig.savefig(figures_dir / f"D_magnitude_scatter_{safe_L}.png",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)

    # E. Sign-consistency bar
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(LABELS))
    n_pos = [int(pathway_df[f"all_positive_{L}"].sum()) for L in LABELS]
    n_neg = [int(pathway_df[f"all_negative_{L}"].sum()) for L in LABELS]
    n_mixed = [len(pathway_df) - p - n for p, n in zip(n_pos, n_neg)]
    ax.bar(x, n_pos, label="All-positive across seeds", color="#d62728")
    ax.bar(x, n_neg, bottom=n_pos, label="All-negative across seeds", color="#1f77b4")
    ax.bar(x, n_mixed, bottom=[p + n for p, n in zip(n_pos, n_neg)],
           label="Mixed sign", color="#cccccc")
    ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylabel("Number of pathways")
    ax.set_title(f"Sign consistency across {len(seeds)} seeds", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "E_sign_consistency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mixed_sign_diagnostics(pathway_df: pd.DataFrame, seeds: list[int],
                                figures_dir: Path, tables_dir: Path) -> None:
    rows = []
    for L in LABELS:
        safe_L = L.replace(" ", "_").replace("-", "_")
        v_cols = [f"v_weight_{L}_seed{s}" for s in seeds]
        df = pathway_df[v_cols + [f"sign_consistent_{L}", f"v_weight_{L}_mean",
                                  f"v_weight_{L}_std", f"abs_mean_{L}"]].copy()
        df["min_abs_seed"] = df[v_cols].abs().min(axis=1)
        df["max_abs_seed"] = df[v_cols].abs().max(axis=1)
        df["range"]        = df[v_cols].max(axis=1) - df[v_cols].min(axis=1)
        df["snr"]          = df[f"abs_mean_{L}"] / df[f"v_weight_{L}_std"].replace(0, np.nan)
        consistent = df[df[f"sign_consistent_{L}"]]
        mixed      = df[~df[f"sign_consistent_{L}"]]
        global_median_abs = float(np.median(pathway_df[v_cols].abs().to_numpy()))
        near_zero_thresh = global_median_abs

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax = axes[0]
        bins = np.linspace(0, df[f"abs_mean_{L}"].quantile(0.99) or 1.0, 40)
        ax.hist(consistent[f"abs_mean_{L}"], bins=bins, alpha=0.7, color="#2ca02c",
                label=f"Sign-consistent (n={len(consistent)})")
        ax.hist(mixed[f"abs_mean_{L}"], bins=bins, alpha=0.7, color="#d62728",
                label=f"Mixed sign (n={len(mixed)})")
        ax.axvline(near_zero_thresh, color="black", ls="--", lw=1,
                   label=f"median |v| = {near_zero_thresh:.2f}")
        ax.set_xlabel(f"|v_weight_{L}_mean|"); ax.set_ylabel("Count")
        ax.set_title(f"{L}: |mean| distribution"); ax.legend(fontsize=8)

        ax = axes[1]
        ax.scatter(consistent[f"abs_mean_{L}"], consistent[f"v_weight_{L}_std"],
                   s=18, alpha=0.6, color="#2ca02c", label="Sign-consistent")
        ax.scatter(mixed[f"abs_mean_{L}"], mixed[f"v_weight_{L}_std"],
                   s=22, alpha=0.75, color="#d62728", label="Mixed sign")
        lim = float(max(df[f"abs_mean_{L}"].max(), df[f"v_weight_{L}_std"].max())) * 1.05
        ax.plot([0, lim], [0, lim], ls=":", color="black", lw=0.8, label="std = |mean|")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"|v_weight_{L}_mean|"); ax.set_ylabel(f"v_weight_{L}_std")
        ax.set_title(f"{L}: magnitude vs variability"); ax.legend(fontsize=8)

        ax = axes[2]
        top_mixed = mixed.sort_values("max_abs_seed", ascending=False).head(15)
        if len(top_mixed) > 0:
            y = np.arange(len(top_mixed))
            for idx, seed in enumerate(seeds):
                vals = top_mixed[f"v_weight_{L}_seed{seed}"].values
                ax.scatter(vals, y + (idx - 2) * 0.12, s=45,
                           color=_seed_color(seed, idx), label=f"Seed {seed}",
                           zorder=3, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels([p[:35] + "..." if len(p) > 38 else p for p in top_mixed.index],
                               fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel(f"v_weight_{L}")
            ax.set_title(f"{L}: top 15 mixed by max|seed|")
            ax.legend(fontsize=7, ncol=2, loc="lower right")
        else:
            ax.text(0.5, 0.5, "No mixed-sign pathways", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{L}: no mixed pathways")

        fig.tight_layout()
        fig.savefig(figures_dir / f"F_mixed_sign_diagnostics_{safe_L}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        mixed_out = pathway_df[~pathway_df[f"sign_consistent_{L}"]].copy()
        mixed_out["label"] = L
        mixed_out["max_abs_seed"] = mixed_out[v_cols].abs().max(axis=1)
        mixed_out["min_abs_seed"] = mixed_out[v_cols].abs().min(axis=1)
        mixed_out["snr"] = mixed_out[f"abs_mean_{L}"] / mixed_out[f"v_weight_{L}_std"].replace(0, np.nan)
        keep = ["label", f"v_weight_{L}_mean", f"v_weight_{L}_std",
                "max_abs_seed", "min_abs_seed", "snr"] + v_cols
        rows.append(mixed_out[keep].rename(columns={
            f"v_weight_{L}_mean": "mean",
            f"v_weight_{L}_std":  "std",
            **{f"v_weight_{L}_seed{s}": f"seed{s}" for s in seeds},
        }))
    pd.concat(rows).to_csv(tables_dir / "mixed_sign_pathways.csv")


def plot_stratified_heatmaps(
    pathway_df: pd.DataFrame,
    seeds: list[int],
    figures_dir: Path,
    ref_seed: int = 42,
    n: int = 50,
) -> None:
    """Three heatmaps per label (positive / negative / near-zero) using ref_seed for selection."""
    v_cols = {s: {L: f"v_weight_{L}_seed{s}" for L in LABELS} for s in seeds}
    seed_labels = [f"Seed {s}" for s in seeds]

    for L in LABELS:
        safe_L = L.replace(" ", "_").replace("-", "_")
        ref_col = f"v_weight_{L}_seed{ref_seed}"
        if ref_col not in pathway_df.columns:
            print(f"  [skip] {ref_col} not found")
            continue

        ref = pathway_df[ref_col]
        all_cols = [f"v_weight_{L}_seed{s}" for s in seeds]

        groups = {
            "positive": pathway_df.loc[
                ref.sort_values(ascending=False).head(n).index, all_cols
            ],
            "negative": pathway_df.loc[
                ref.sort_values(ascending=True).head(n).index, all_cols
            ],
            "near_zero": pathway_df.loc[
                ref.abs().sort_values(ascending=True).head(n).index, all_cols
            ],
        }

        for grp_name, heat in groups.items():
            heat = heat.copy()
            heat.index = [_clean_pathway_name(p, maxlen=45) for p in heat.index]
            heat.columns = seed_labels

            vmax = float(max(abs(heat.values.min()), abs(heat.values.max()))) or 1e-6
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            fig, ax = plt.subplots(figsize=(8, max(4, 0.36 * n)))
            sns.heatmap(
                heat, cmap="RdBu_r", norm=norm,
                annot=True, fmt=".2f", annot_kws={"size": 5.5},
                linewidths=0.3, linecolor="white", ax=ax,
                cbar_kws={"label": f"v_weight_{L}", "shrink": 0.6},
            )
            ax.set_title(
                f"{L} · {grp_name} (selected by Seed {ref_seed}): "
                f"v_weight across seeds",
                fontsize=11, fontweight="bold",
            )
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            fig.savefig(
                figures_dir / f"H_heatmap_{safe_L}_{grp_name}.png",
                dpi=200, bbox_inches="tight",
            )
            plt.close(fig)


def _clean_pathway_name(name: str, maxlen: int = 35) -> str:
    s = re.sub(r"^REACTOME_", "", str(name))
    s = s.replace("_", " ")
    if len(s) > maxlen:
        s = s[:maxlen - 1] + "…"
    return s


def plot_per_seed_top_pathways(
    pathway_df: pd.DataFrame,
    seeds: list[int],
    figures_dir: Path,
    top_n: int = 50,
) -> None:
    """One figure per label with one panel per seed, showing top_n pathways by |v_weight|."""
    for L in LABELS:
        safe_L = L.replace(" ", "_").replace("-", "_")
        n_seeds = len(seeds)
        fig, axes = plt.subplots(
            1, n_seeds,
            figsize=(6 * n_seeds, max(8, 0.22 * top_n)),
            sharey=False,
        )
        if n_seeds == 1:
            axes = [axes]

        for ax, seed in zip(axes, seeds):
            col = f"v_weight_{L}_seed{seed}"
            order = pathway_df[col].abs().sort_values(ascending=False).head(top_n).index
            vals = pathway_df.loc[order, col].values
            names = [_clean_pathway_name(n) for n in order]
            y = np.arange(len(vals))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
            ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="none")
            ax.axvline(0, color="black", lw=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=5)
            ax.invert_yaxis()
            ax.set_xlabel(f"v_weight_{L}", fontsize=8)
            ax.set_title(f"Seed {seed}", fontsize=10, fontweight="bold")

        fig.suptitle(
            f"{L}: top {top_n} pathways by |v_weight| per seed",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"G_per_seed_top{top_n}_{safe_L}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


def write_per_seed_gene_tables(
    all_results: dict,
    seeds: list[int],
    pathway_df: pd.DataFrame,
    tables_dir: Path,
    beta_threshold: float,
    top_n: int = 50,
) -> None:
    """Per seed per label: CSV of top_n pathways + their non-zero genes (gene symbols via mygene)."""
    all_ensg: set[str] = set()
    for L in LABELS:
        for seed in seeds:
            col = f"v_weight_{L}_seed{seed}"
            top_idx = pathway_df[col].abs().sort_values(ascending=False).head(top_n).index
            rb = all_results[seed]["r_beta"].copy()
            if "Unnamed: 0" in rb.columns:
                rb = rb.set_index("Unnamed: 0")
            gene_cols = rb.columns[rb.columns.str.lower().str.startswith("ensg")]
            for pw in top_idx:
                if pw in rb.index:
                    row = rb.loc[pw, gene_cols].astype(float)
                    all_ensg.update(gene_cols[(row > beta_threshold).to_numpy()])

    ensg_to_symbol: dict[str, str] = {}
    if all_ensg:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            hits = mg.querymany(sorted(all_ensg), scopes="ensembl.gene",
                                fields="symbol", species="human", verbose=False)
            for hit in hits:
                if "symbol" in hit:
                    ensg_to_symbol[hit["query"]] = hit["symbol"]
        except Exception as e:
            print(f"  [warn] mygene lookup skipped: {e}")

    per_seed_dir = tables_dir / "per_seed_top50"
    per_seed_dir.mkdir(exist_ok=True)

    for seed in seeds:
        rb = all_results[seed]["r_beta"].copy()
        if "Unnamed: 0" in rb.columns:
            rb = rb.set_index("Unnamed: 0")
        gene_cols = rb.columns[rb.columns.str.lower().str.startswith("ensg")]

        for L in LABELS:
            safe_L = L.replace(" ", "_").replace("-", "_")
            col = f"v_weight_{L}_seed{seed}"
            top_idx = pathway_df[col].abs().sort_values(ascending=False).head(top_n).index
            rows = []
            for pw in top_idx:
                vw = float(pathway_df.loc[pw, col])
                if pw in rb.index:
                    mask = rb.loc[pw, gene_cols].astype(float) > beta_threshold
                    genes = [ensg_to_symbol.get(g, g) for g in gene_cols[mask.to_numpy()]]
                else:
                    genes = []
                rows.append({
                    "pathway": pw,
                    "pathway_name": _clean_pathway_name(pw, maxlen=9999),
                    f"v_weight_{L}": round(vw, 6),
                    "n_nonzero_genes": len(genes),
                    "genes": ", ".join(sorted(genes)),
                })
            pd.DataFrame(rows).to_csv(
                per_seed_dir / f"seed{seed}_{safe_L}_top{top_n}.csv", index=False
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, type=Path,
                        help="Directory containing seed_*/ subdirectories.")
    parser.add_argument("--method-subdir", default="drgp_masked",
                        help="Subdirectory inside each seed_* dir to load (default: drgp_masked).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit seeds to use. Default: auto-discover.")
    parser.add_argument("--beta-threshold", type=float, default=0.5,
                        help="r_beta threshold for membership (default: 0.5).")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top-N pathways for heatmap/dotplot figures (default: 20).")
    parser.add_argument("--top-n-shared", type=int, default=15,
                        help="Top-N pathways per label for shared-gene table (default: 15).")
    args = parser.parse_args()

    base_dir: Path = args.path.resolve()
    if not base_dir.is_dir():
        raise SystemExit(f"Path does not exist or is not a directory: {base_dir}")

    discovered = discover_seeds(base_dir, args.method_subdir)
    if args.seeds:
        seed_dirs = {s: d for s, d in discovered if s in args.seeds}
        seeds = sorted(seed_dirs)
    else:
        seed_dirs = dict(discovered)
        seeds = sorted(seed_dirs)
    if not seeds:
        raise SystemExit(f"No seed directories found under {base_dir}")

    figures_dir = base_dir / "figures"
    tables_dir  = base_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    print(f"Base dir : {base_dir}")
    print(f"Seeds    : {seeds}")
    print(f"Figures  : {figures_dir}")
    print(f"Tables   : {tables_dir}")

    global TARGETS, LABELS
    all_results = {}
    for seed in seeds:
        all_results[seed] = load_seed(seed_dirs[seed])
        print(f"Loaded seed {seed} from {seed_dirs[seed]}")

    TARGETS, LABELS = detect_labels(all_results[seeds[0]]["vi_test_predictions"])
    print(f"Detected targets: {LABELS}")

    print("\n[1/6] ROC + KDE panel")
    plot_roc_kde_panel(all_results, seeds, figures_dir / "vi_unified_panel.png")

    print("[2/6] v_weight histograms")
    plot_vweight_histograms(all_results, seeds, figures_dir / "vweight_histograms.png")

    print("[3/6] Per-seed gene_programs_filtered.csv.gz")
    write_gene_programs_filtered(all_results, seeds, seed_dirs, args.beta_threshold)

    print("[4/6] Pathway v_weight summary + rank correlations")
    pathway_df, rank_corr = build_pathway_summary(all_results, seeds)
    pathway_df.to_csv(tables_dir / "pathway_vweight_summary.csv")

    print("[5/6] Top-pathway shared-gene table (mygene lookup)")
    shared_df = build_shared_gene_table(all_results, seeds, pathway_df,
                                        args.top_n_shared, args.beta_threshold)
    shared_df.to_csv(tables_dir / "top_pathway_shared_genes.csv", index=False)

    print("[6/7] A-F figures + mixed_sign_pathways.csv")
    plot_panels_abcde(pathway_df, rank_corr, seeds, figures_dir, top_n=args.top_n)
    plot_mixed_sign_diagnostics(pathway_df, seeds, figures_dir, tables_dir)

    print("[7/8] Per-seed top-50 pathway plots + gene tables")
    plot_per_seed_top_pathways(pathway_df, seeds, figures_dir, top_n=50)
    write_per_seed_gene_tables(all_results, seeds, pathway_df, tables_dir,
                               args.beta_threshold, top_n=50)

    ref_seed = seeds[0] if 42 not in seeds else 42
    print(f"[8/8] Stratified heatmaps (pos/neg/near-zero, ref seed {ref_seed})")
    plot_stratified_heatmaps(pathway_df, seeds, figures_dir, ref_seed=ref_seed, n=50)

    print(f"\nDone. Figures → {figures_dir}")
    print(f"      Tables  → {tables_dir}")


if __name__ == "__main__":
    main()
