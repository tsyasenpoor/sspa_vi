#!/usr/bin/env python3
"""
Method comparison pipeline: DRGP (unmasked + masked Reactome), GSVA, MUSTER.

Three axes per (data_subset, outcome):
  A. Predictive AUC, two panels: per-cell pooled  +  per-patient pseudobulk
  B. Pathway-importance overlap vs DRGP-masked Reactome anchor
  C. Stability of the identified signature across replicates

Data subsets:
  allPBMC     (n=34,763 cells, 29 patients)   — all four methods
  monocytes   (n=10,897 cells, 29 patients)   — DRGP both modes; GSVA pending
                                                 (results/gsva_monocytes/ when job 2101624 lands);
                                                 MUSTER pending (held-out re-run, deferred).

Outputs land in /labs/Aguiar/SSPA_BRAY/results/method_comparison/.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

OUTCOMES = ("t2dm", "cvda")
DRGP_SEEDS = (42, 123, 256, 512, 1024)

PATHS = {
    "h5ad_allPBMC":  "/labs/Aguiar/SSPA_BRAY/dataset/biorepository/allPBMC_GEX_20260106.h5ad",
    "h5ad_monocytes": "/labs/Aguiar/SSPA_BRAY/dataset/biorepository/monocytes_GEX_20251201.h5ad",
    "gsva_allPBMC":   "/labs/Aguiar/SSPA_BRAY/results/gsva_pbmc/GSVA_allPBMC_2026-05-04.csv",
    "gsva_monocytes_dir": "/labs/Aguiar/SSPA_BRAY/results/gsva_monocytes",
    "muster_dir":     "/labs/Aguiar/SSPA_BRAY/results/muster_pbmc",
    "drgp_unmasked":  "/labs/Aguiar/SSPA_BRAY/results/biorepo_vi_unmasked",
    "drgp_masked":    "/labs/Aguiar/SSPA_BRAY/results/biorepo_vi_masked_reactome",
    "out_root":       "/labs/Aguiar/SSPA_BRAY/results/method_comparison",
}

DRGP_CT_DIR = {
    "allPBMC":   "allPBMC_GEX_20260106",
    "monocytes": "monocytes_GEX_20251201",
    "Bcell":     "Bcell_GEX_20251201",
    "cd4t":      "cd4t_GEX_20251201",
    "cd8t":      "cd8t_GEX_20251201",
    "NKcell":    "NKcell_GEX_20251201",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_obs(subset: str) -> pd.DataFrame:
    """Load h5ad obs only (cheap, no X)."""
    h5 = PATHS[f"h5ad_{subset}"]
    import anndata as ad
    a = ad.read_h5ad(h5, backed="r")
    obs = a.obs[["pt_id", "Sex", "t2dm", "cvda", "celltype_subclust"]].copy()
    obs["t2dm"] = obs["t2dm"].astype(int)
    obs["cvda"] = obs["cvda"].astype(int)
    return obs


def load_h5ad_X(subset: str, genes: list[str] | None = None,
                 cells: list[str] | None = None) -> pd.DataFrame:
    """Load expression as DataFrame (cells × genes), optionally subset."""
    import anndata as ad
    h5 = PATHS[f"h5ad_{subset}"]
    a = ad.read_h5ad(h5)
    if cells is not None:
        a = a[cells, :]
    if genes is not None:
        present = [g for g in genes if g in a.var_names]
        a = a[:, present]
    X = a.to_df() if hasattr(a, "to_df") else pd.DataFrame(
        np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X),
        index=a.obs_names, columns=a.var_names)
    return X


def load_gsva(subset: str) -> pd.DataFrame:
    """Return (cell × pathway) GSVA enrichment matrix."""
    if subset == "allPBMC":
        p = PATHS["gsva_allPBMC"]
    else:
        # Monocytes file is named with date stamp; find latest
        d = Path(PATHS["gsva_monocytes_dir"])
        cands = sorted(d.glob("GSVA_monocytes_*.csv"))
        if not cands:
            raise FileNotFoundError(f"No GSVA monocytes CSV in {d}")
        p = str(cands[-1])
    print(f"  loading GSVA: {p}")
    df = pd.read_csv(p, index_col=0)
    # Original CSV index column is named 'Name' (cell barcodes)
    df.index.name = "cell_id"
    return df


def load_muster(outcome: str) -> dict:
    """Load MUSTER final + phase1 records for one outcome."""
    d = Path(PATHS["muster_dir"])
    final = sorted(d.glob(f"muster_{outcome}_final_*.json"))[-1]
    phase1 = sorted(d.glob(f"muster_{outcome}_phase1_*.json"))[-1]
    with open(final) as f:
        f_data = json.load(f)
    with open(phase1) as f:
        p1_data = json.load(f)
    return {"final": f_data, "phase1": p1_data,
            "genes": f_data["final_genes"],
            "internal_auc": f_data["final_auc"]}


def load_drgp_all_metrics(mode: str) -> pd.DataFrame:
    """Read DRGP all_metrics.csv (either unmasked or masked Reactome)."""
    p = Path(PATHS[f"drgp_{mode}"]) / "all_metrics.csv"
    return pd.read_csv(p)


def load_drgp_pathway_weights(subset: str) -> pd.DataFrame:
    """Per-pathway × per-seed v_weight summary from DRGP-masked Reactome."""
    ct_dir = DRGP_CT_DIR[subset]
    p = Path(PATHS["drgp_masked"]) / ct_dir / "tables" / "pathway_vweight_summary.csv"
    return pd.read_csv(p)


def load_drgp_theta(subset: str, seed: int, mode: str,
                     split: str = "train") -> pd.DataFrame:
    """DRGP cell-level factor scores for one (subset, seed, mode, split)."""
    ct_dir = DRGP_CT_DIR[subset]
    p = (Path(PATHS[f"drgp_{mode}"]) / ct_dir / f"seed{seed}" /
         f"vi_theta_{split}.csv.gz")
    df = pd.read_csv(p)
    return df


def load_drgp_v_vector(subset: str, seed: int, mode: str) -> np.ndarray:
    """DRGP v_vector_seed{N}.npy. Shape (kappa × K,) flattened."""
    ct_dir = DRGP_CT_DIR[subset]
    p = (Path(PATHS[f"drgp_{mode}"]) / ct_dir / f"seed{seed}" /
         f"v_vector_seed{seed}.npy")
    return np.load(p)


# ─────────────────────────────────────────────────────────────────────────────
# Axis A — predictive AUC
# ─────────────────────────────────────────────────────────────────────────────

def patient_grouped_cv_auc(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                            n_folds: int = 5, random_state: int = 0,
                            n_jobs: int = 4) -> dict:
    """L1-LR with patient-grouped CV. Returns per-fold AUC + mean AUC."""
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        lr = LogisticRegressionCV(
            penalty="l1", solver="saga", Cs=8, cv=3, max_iter=2000,
            class_weight="balanced", random_state=random_state, n_jobs=n_jobs,
        )
        lr.fit(X.iloc[tr].values, y[tr])
        scores = lr.predict_proba(X.iloc[te].values)[:, 1]
        aucs.append(roc_auc_score(y[te], scores))
    return {"fold_aucs": aucs, "mean_auc": float(np.mean(aucs)) if aucs else float("nan")}


def axis_a_drgp_per_cell(subset: str, mode: str) -> pd.DataFrame:
    """Pull DRGP per-cell AUCs from all_metrics.csv. Already CV'd internally."""
    df = load_drgp_all_metrics(mode)
    ct_label = "allPBMC" if subset == "allPBMC" else "monocytes"
    sub = df[(df["cell_type"] == ct_label)].copy()
    sub["method"] = f"drgp_{mode}"
    sub["subset"] = subset
    return sub[["method", "subset", "label", "split", "seed", "auc"]].rename(
        columns={"label": "outcome"})


def axis_a_gsva_per_cell(subset: str, obs: pd.DataFrame) -> list[dict]:
    """L1-LR on (cell × pathway) GSVA scores, patient-grouped CV."""
    gsva = load_gsva(subset)
    # Align: keep cells present in both
    common = gsva.index.intersection(obs.index)
    if len(common) == 0:
        print(f"  WARN: zero barcode overlap between GSVA and h5ad for {subset}")
        return []
    gsva = gsva.loc[common]
    obs_a = obs.loc[common]
    rows = []
    for outcome in OUTCOMES:
        y = obs_a[outcome].values.astype(int)
        groups = obs_a["pt_id"].astype(str).values
        if len(np.unique(y)) < 2:
            continue
        t0 = time.time()
        res = patient_grouped_cv_auc(gsva, y, groups, n_folds=5)
        print(f"  GSVA per-cell {subset}/{outcome}: mean_auc={res['mean_auc']:.4f} "
              f"({len(res['fold_aucs'])} folds, {time.time()-t0:.1f}s)")
        for i, a in enumerate(res["fold_aucs"]):
            rows.append({
                "method": "gsva_lr", "subset": subset, "outcome": outcome,
                "split": "cv", "seed": i, "auc": a,
            })
    return rows


def axis_a_muster_per_cell(subset: str, obs: pd.DataFrame) -> list[dict]:
    """L1-LR on (cell × MUSTER_genes) using h5ad expression."""
    rows = []
    for outcome in OUTCOMES:
        mus = load_muster(outcome)
        genes = mus["genes"]
        # Load h5ad expression for just these genes (cheap)
        try:
            X = load_h5ad_X(subset, genes=genes)
        except Exception as e:
            print(f"  WARN MUSTER X-load failed: {e}")
            continue
        common = X.index.intersection(obs.index)
        X = X.loc[common]
        obs_a = obs.loc[common]
        y = obs_a[outcome].values.astype(int)
        groups = obs_a["pt_id"].astype(str).values
        if len(np.unique(y)) < 2 or X.shape[1] == 0:
            continue
        # log1p-normalize expression for parity with GSVA inputs
        Xn = np.log1p(X.values).astype(np.float32)
        Xn = pd.DataFrame(Xn, index=X.index, columns=X.columns)
        t0 = time.time()
        res = patient_grouped_cv_auc(Xn, y, groups, n_folds=5)
        print(f"  MUSTER per-cell {subset}/{outcome}: mean_auc={res['mean_auc']:.4f} "
              f"({len(genes)} genes, {time.time()-t0:.1f}s)")
        for i, a in enumerate(res["fold_aucs"]):
            rows.append({
                "method": "muster_lr", "subset": subset, "outcome": outcome,
                "split": "cv", "seed": i, "auc": a,
            })
    return rows


def pseudobulk_by_patient(X: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Mean-aggregate rows of X by obs['pt_id']. Returns (patient × feature)."""
    g = X.groupby(obs.loc[X.index, "pt_id"].astype(str).values)
    return g.mean()


def axis_a_per_patient(subset: str, obs: pd.DataFrame) -> list[dict]:
    """Per-patient pseudobulk for all four methods, 5-fold stratified patient CV."""
    rows = []
    # Patient-level outcome labels (one per pt_id)
    pat_obs = obs.drop_duplicates("pt_id").set_index("pt_id")

    # --- GSVA pseudobulk
    try:
        gsva = load_gsva(subset)
        common = gsva.index.intersection(obs.index)
        gsva = gsva.loc[common]
        pb_g = pseudobulk_by_patient(gsva, obs)
        rows += _per_patient_lr(pb_g, pat_obs, method="gsva_lr_pb",
                                  subset=subset)
    except Exception as e:
        print(f"  WARN GSVA pseudobulk skipped: {e}")

    # --- MUSTER (per-outcome gene panel pseudobulk)
    for outcome in OUTCOMES:
        try:
            mus = load_muster(outcome)
            X = load_h5ad_X(subset, genes=mus["genes"])
            common = X.index.intersection(obs.index)
            X = np.log1p(X.loc[common]).astype(np.float32)
            X = pd.DataFrame(X, index=X.index if hasattr(X, 'index') else common,
                              columns=mus["genes"][:X.shape[1]])
            pb_m = pseudobulk_by_patient(X, obs)
            # MUSTER gene panel is outcome-specific; only evaluate same outcome
            rows += _per_patient_lr(pb_m, pat_obs, method=f"muster_lr_pb",
                                      subset=subset, only_outcomes=(outcome,))
            # And report MUSTER's own internal CV AUC as a baseline
            rows.append({
                "method": "muster_internal_cv", "subset": subset,
                "outcome": outcome, "split": "internal", "seed": 0,
                "auc": mus["internal_auc"],
            })
        except Exception as e:
            print(f"  WARN MUSTER pseudobulk {outcome}: {e}")

    # --- DRGP pseudobulk (use vi_theta_* concatenated across splits)
    for mode in ("unmasked", "masked"):
        for seed in DRGP_SEEDS:
            try:
                parts = []
                for split in ("train", "val", "test"):
                    df = load_drgp_theta(subset, seed, mode, split)
                    parts.append(df)
                theta = pd.concat(parts, ignore_index=True)
                # Drop non-factor columns (cell_id, celltype_subclust, Sex)
                meta_cols = [c for c in theta.columns
                             if c in ("cell_id", "celltype_subclust", "Sex",
                                       "Sex_binary", "t2dm", "cvda")]
                feat_cols = [c for c in theta.columns if c not in meta_cols]
                theta_X = theta[feat_cols].copy()
                theta_X.index = theta["cell_id"].astype(str).values
                common = theta_X.index.intersection(obs.index)
                pb_d = pseudobulk_by_patient(theta_X.loc[common], obs)
                pb_d.columns = pb_d.columns.astype(str)
                rows += _per_patient_lr(pb_d, pat_obs,
                                          method=f"drgp_{mode}_pb",
                                          subset=subset, seed_tag=seed)
            except Exception as e:
                print(f"  WARN DRGP {mode} seed{seed} pseudobulk: {e}")
    return rows


def _per_patient_lr(X: pd.DataFrame, pat_obs: pd.DataFrame, method: str,
                     subset: str, only_outcomes=OUTCOMES,
                     seed_tag: int | str = 0) -> list[dict]:
    """5-fold stratified patient CV with L1-LR on a (patient × feature) matrix."""
    common = X.index.intersection(pat_obs.index)
    X = X.loc[common]
    rows = []
    for outcome in only_outcomes:
        y = pat_obs.loc[common, outcome].values.astype(int)
        if len(np.unique(y)) < 2 or X.shape[0] < 10:
            continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr, te in skf.split(X.values, y):
            if len(np.unique(y[te])) < 2:
                continue
            lr = LogisticRegressionCV(
                penalty="l1", solver="saga", Cs=6, cv=3, max_iter=2000,
                class_weight="balanced", random_state=42, n_jobs=2,
            )
            try:
                lr.fit(X.iloc[tr].values, y[tr])
                aucs.append(roc_auc_score(y[te], lr.predict_proba(X.iloc[te].values)[:, 1]))
            except Exception:
                continue
        if aucs:
            print(f"  {method} {subset}/{outcome} seed={seed_tag}: "
                  f"mean_auc={np.mean(aucs):.4f}  (n_folds={len(aucs)})")
        for i, a in enumerate(aucs):
            rows.append({
                "method": method, "subset": subset, "outcome": outcome,
                "split": "cv", "seed": f"{seed_tag}_f{i}", "auc": float(a),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Axis B — pathway-importance overlap
# ─────────────────────────────────────────────────────────────────────────────

def axis_b_pathway_overlap(subset: str, obs: pd.DataFrame,
                            top_k_list=(20, 50, 100)) -> list[dict]:
    """Rank pathways per method × outcome; report Jaccard@K vs DRGP-masked."""
    rows = []
    drgp_pw = load_drgp_pathway_weights(subset)

    for outcome in OUTCOMES:
        # DRGP-masked anchor: mean v_weight magnitude across seeds
        anchor_col = f"abs_mean_{outcome}"
        anchor = (drgp_pw[["pathway", anchor_col]]
                  .sort_values(anchor_col, ascending=False)
                  .reset_index(drop=True))
        anchor_set = anchor["pathway"].tolist()

        # GSVA-derived ranking: |L1-LR coef| per pathway
        try:
            gsva = load_gsva(subset)
            common = gsva.index.intersection(obs.index)
            gsva = gsva.loc[common]
            obs_a = obs.loc[common]
            y = obs_a[outcome].values.astype(int)
            t0 = time.time()
            lr = LogisticRegressionCV(
                penalty="l1", solver="saga", Cs=8, cv=5, max_iter=2000,
                class_weight="balanced", random_state=42, n_jobs=4,
            )
            lr.fit(gsva.values, y)
            coefs = np.abs(lr.coef_.ravel())
            gsva_rank = pd.DataFrame({
                "pathway": gsva.columns, "abs_coef": coefs,
            }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
            print(f"  Axis B GSVA-LR ranking {subset}/{outcome}: "
                  f"{(coefs > 0).sum()} non-zero pathways  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  WARN GSVA-LR ranking failed: {e}")
            gsva_rank = None

        # For each K, compute Jaccard between DRGP-masked top-K and GSVA top-K
        for k in top_k_list:
            anchor_topK = set(anchor_set[:k])
            if gsva_rank is not None:
                gsva_topK = set(gsva_rank["pathway"].head(k).tolist())
                jac = (len(anchor_topK & gsva_topK) /
                       len(anchor_topK | gsva_topK)) if (anchor_topK | gsva_topK) else 0.0
                rows.append({
                    "subset": subset, "outcome": outcome, "K": k,
                    "method_a": "drgp_masked", "method_b": "gsva_lr",
                    "jaccard": jac, "intersect": len(anchor_topK & gsva_topK),
                })

        # MUSTER overlap: hypergeometric enrichment of MUSTER genes vs each pathway,
        # then rank pathways by enrichment p-value (smaller = more enriched).
        try:
            from scipy.stats import hypergeom
            mus = load_muster(outcome)
            mus_genes = set(mus["genes"])
            # Pathway → genes dict from DRGP-masked per-seed top50 files
            # (contains the full Reactome gene lists). Use seed42 as a snapshot.
            per_seed_top50 = (Path(PATHS["drgp_masked"]) / DRGP_CT_DIR[subset] /
                               "tables" / "per_seed_top50" /
                               f"seed42_{outcome}_top50.csv")
            top50 = pd.read_csv(per_seed_top50)
            # Universe size (denominator for hypergeom): rough — use ~20k human genes
            N = 20000
            pw_ranked = []
            for _, r in top50.iterrows():
                pw_genes = set(str(r["genes"]).split(", "))
                K_pw = len(pw_genes)
                k_obs = len(mus_genes & pw_genes)
                n_draw = len(mus_genes)
                if K_pw == 0 or k_obs == 0:
                    pv = 1.0
                else:
                    pv = hypergeom.sf(k_obs - 1, N, K_pw, n_draw)
                pw_ranked.append((r["pathway"], pv, k_obs))
            mus_rank = pd.DataFrame(pw_ranked, columns=["pathway", "pvalue", "n_overlap"]) \
                          .sort_values("pvalue").reset_index(drop=True)
            for k in top_k_list:
                anchor_topK = set(anchor_set[:k])
                mus_topK = set(mus_rank["pathway"].head(k).tolist())
                jac = (len(anchor_topK & mus_topK) /
                       len(anchor_topK | mus_topK)) if (anchor_topK | mus_topK) else 0.0
                rows.append({
                    "subset": subset, "outcome": outcome, "K": k,
                    "method_a": "drgp_masked", "method_b": "muster_enrich",
                    "jaccard": jac, "intersect": len(anchor_topK & mus_topK),
                })
        except Exception as e:
            print(f"  WARN MUSTER enrichment failed: {e}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Axis C — stability of top-K
# ─────────────────────────────────────────────────────────────────────────────

def axis_c_stability(subset: str, K: int = 50) -> list[dict]:
    """Pairwise Jaccard of top-K signatures across replicates."""
    rows = []

    # DRGP-masked: pairwise across 5 seeds, both outcomes
    drgp = load_drgp_pathway_weights(subset)
    for outcome in OUTCOMES:
        seed_topK = {}
        for s in DRGP_SEEDS:
            col = f"v_weight_{outcome}_seed{s}"
            if col not in drgp.columns:
                continue
            order = drgp[["pathway", col]].copy()
            order["abs"] = order[col].abs()
            seed_topK[s] = set(
                order.sort_values("abs", ascending=False).head(K)["pathway"].tolist())
        seeds = list(seed_topK)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a, b = seed_topK[seeds[i]], seed_topK[seeds[j]]
                jac = len(a & b) / len(a | b) if (a | b) else 0.0
                rows.append({
                    "method": "drgp_masked", "subset": subset, "outcome": outcome,
                    "K": K, "pair": f"{seeds[i]}_vs_{seeds[j]}", "jaccard": jac,
                })

    # MUSTER: pairwise across 20 phase-1 batch gene panels (per outcome)
    for outcome in OUTCOMES:
        try:
            mus = load_muster(outcome)
            batches = mus["phase1"]["batches"]
            batch_sets = [set(b["genes"]) for b in batches]
            for i in range(len(batch_sets)):
                for j in range(i + 1, len(batch_sets)):
                    a, b = batch_sets[i], batch_sets[j]
                    jac = len(a & b) / len(a | b) if (a | b) else 0.0
                    rows.append({
                        "method": "muster_phase1", "subset": "allPBMC",
                        "outcome": outcome, "K": None,
                        "pair": f"b{i+1}_vs_b{j+1}", "jaccard": jac,
                    })
        except Exception as e:
            print(f"  WARN MUSTER stability failed: {e}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=("allPBMC", "monocytes"), required=True)
    ap.add_argument("--axes", default="A,B,C",
                    help="Comma-separated axes to run (e.g. A,B  or  B)")
    ap.add_argument("--skip-per-patient", action="store_true",
                    help="Skip the patient-pseudobulk panel of Axis A "
                         "(slow on first run because of h5ad reads)")
    args = ap.parse_args()

    axes = set(a.strip().upper() for a in args.axes.split(","))
    out_dir = Path(PATHS["out_root"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(PATHS["out_root"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading obs for {args.subset} ...")
    obs = load_obs(args.subset)
    print(f"  n_cells={len(obs)}, n_patients={obs['pt_id'].nunique()}")

    # ─── Axis A ──────────────────────────────────────────────────────────────
    if "A" in axes:
        print(f"\n=== Axis A — predictive AUC ({args.subset}) ===")
        rows_pc = []
        for mode in ("unmasked", "masked"):
            df = axis_a_drgp_per_cell(args.subset, mode)
            rows_pc.extend(df.to_dict(orient="records"))
        rows_pc.extend(axis_a_gsva_per_cell(args.subset, obs))
        rows_pc.extend(axis_a_muster_per_cell(args.subset, obs))
        pd.DataFrame(rows_pc).to_csv(
            out_dir / f"axis_a_per_cell_{args.subset}.csv", index=False)
        print(f"  wrote axis_a_per_cell_{args.subset}.csv ({len(rows_pc)} rows)")

        if not args.skip_per_patient:
            print(f"\n--- per-patient pseudobulk panel ({args.subset}) ---")
            rows_pp = axis_a_per_patient(args.subset, obs)
            pd.DataFrame(rows_pp).to_csv(
                out_dir / f"axis_a_per_patient_{args.subset}.csv", index=False)
            print(f"  wrote axis_a_per_patient_{args.subset}.csv ({len(rows_pp)} rows)")

    # ─── Axis B ──────────────────────────────────────────────────────────────
    if "B" in axes:
        print(f"\n=== Axis B — pathway-importance overlap ({args.subset}) ===")
        rows_b = axis_b_pathway_overlap(args.subset, obs)
        pd.DataFrame(rows_b).to_csv(
            out_dir / f"axis_b_pathway_overlap_{args.subset}.csv", index=False)
        print(f"  wrote axis_b_pathway_overlap_{args.subset}.csv ({len(rows_b)} rows)")

    # ─── Axis C ──────────────────────────────────────────────────────────────
    if "C" in axes:
        print(f"\n=== Axis C — stability ({args.subset}) ===")
        rows_c = axis_c_stability(args.subset, K=50)
        pd.DataFrame(rows_c).to_csv(
            out_dir / f"axis_c_stability_{args.subset}.csv", index=False)
        print(f"  wrote axis_c_stability_{args.subset}.csv ({len(rows_c)} rows)")

    print(f"\nDone. Outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
