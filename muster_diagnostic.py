#!/usr/bin/env python3
"""
MUSTER AUC=1.0 diagnostic.

Hypotheses to distinguish:
  H1  GA peeking — MUSTER ran ~13k LR evaluations against the SAME 5 CV folds,
      adversarially overfit those specific folds. Different CV seeds drop AUC.
  H2  Genuine separability — with only 29 patients and 13 genes, perfect
      separation is mathematically plausible regardless of CV draw. Random
      13-gene panels should also reach high AUC.
  H3  Per-fold instability — small folds (~6 patients each, possibly 1-2 of
      minority class) inflate AUC variance, mean across folds happens to hit
      1.0. Per-fold AUCs are extreme (0.5 and 1.0 mixed).

What we compute, for each outcome:
  D1  Reproduce final_auc using saved final_genes + the same make_cv_folds
      with SEED=10086. Should match the recorded 1.0.
  D2  Re-evaluate same panel with 100 different CV seeds. Distribution of
      mean-CV-AUC — tight band near 1.0 means H1 is unlikely; spread or
      drop means H1 is real.
  D3  Per-fold AUC trace for D1 — are some folds 0.5 (undefined → fallback)?
  D4  LOPO-CV: 29-fold AUC with same panel. The most honest unsupervised
      check.
  D5  Random-panel control: 1000 draws of N-random genes from the same
      pseudobulk gene pool. Distribution of mean-CV-AUC vs MUSTER's.
  D6  Class balance per fold (n_pos / n_neg in each fold). Identifies
      folds where AUC is mathematically near-trivial.

Outputs: /labs/Aguiar/SSPA_BRAY/results/muster_diagnostic/{summary.json,
                                                            random_panels_<outcome>.csv,
                                                            cv_seed_trace_<outcome>.csv}
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# Settings (must match muster_pbmc.py exactly)
DATA_PATH = "/labs/Aguiar/SSPA_BRAY/dataset/biorepository/allPBMC_GEX_20260106.h5ad"
MUSTER_DIR = "/labs/Aguiar/SSPA_BRAY/results/muster_pbmc"
OUT_DIR = "/labs/Aguiar/SSPA_BRAY/results/muster_diagnostic"
FIXED_COVS = ['Sex_binary']
SEED = 10086
N_FOLDS = 5
N_CV_SEEDS = 100
N_RANDOM_PANELS = 1000


# Reproduced verbatim from muster_pbmc.py to match its assumptions exactly.
def make_pseudobulk(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    patients = adata.obs['pt_id'].values
    unique_pts = np.unique(patients)
    pb = np.zeros((len(unique_pts), adata.n_vars), dtype=np.float32)
    for i, pt in enumerate(unique_pts):
        mask = patients == pt
        pb[i] = np.asarray(adata.X[mask].mean(axis=0)).ravel()
    return pd.DataFrame(pb, index=unique_pts, columns=adata.var_names)


def normalize_01(df, qlo=0.05, qhi=0.95):
    lo = df.quantile(qlo)
    hi = df.quantile(qhi)
    denom = (hi - lo).replace(0, 1)
    return ((df - lo) / denom).clip(0, 1)


def make_cv_folds(ids, labels, n_folds=5, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    ids, labels = np.array(ids), np.array(labels, dtype=int)
    pos = rng.permutation(ids[labels == 1])
    neg = rng.permutation(ids[labels == 0])
    folds = []
    pos_splits = np.array_split(pos, n_folds)
    neg_splits = np.array_split(neg, n_folds)
    for i in range(n_folds):
        test = np.concatenate([pos_splits[i], neg_splits[i]])
        train = np.concatenate(
            [np.concatenate([s for j, s in enumerate(pos_splits) if j != i]),
             np.concatenate([s for j, s in enumerate(neg_splits) if j != i])]
        )
        folds.append((train, test))
    return folds


def evaluate_panel(feat, cv_folds, outcome, gene_list, return_per_fold=False):
    """Same as muster_pbmc.py's evaluate. Mean CV AUC across folds."""
    cols = list(gene_list) + FIXED_COVS
    aucs, fold_info = [], []
    for fi, (train_ids, test_ids) in enumerate(cv_folds):
        X_tr = feat.loc[train_ids, cols].values
        y_tr = feat.loc[train_ids, outcome].values
        X_te = feat.loc[test_ids, cols].values
        y_te = feat.loc[test_ids, outcome].values
        n_pos_te = int(y_te.sum())
        n_neg_te = int((y_te == 0).sum())
        if len(np.unique(y_te)) < 2:
            fold_info.append({"fold": fi, "auc": float("nan"),
                              "n_test": len(y_te),
                              "n_pos_te": n_pos_te, "n_neg_te": n_neg_te,
                              "skipped": "single_class_test"})
            continue
        try:
            mdl = LogisticRegression(
                class_weight='balanced', max_iter=500,
                solver='lbfgs', C=1.0, random_state=42)
            mdl.fit(X_tr, y_tr)
            a = roc_auc_score(y_te, mdl.predict_proba(X_te)[:, 1])
            aucs.append(a)
            fold_info.append({"fold": fi, "auc": float(a),
                              "n_test": len(y_te),
                              "n_pos_te": n_pos_te, "n_neg_te": n_neg_te,
                              "skipped": ""})
        except Exception as e:
            aucs.append(0.5)
            fold_info.append({"fold": fi, "auc": 0.5,
                              "n_test": len(y_te),
                              "n_pos_te": n_pos_te, "n_neg_te": n_neg_te,
                              "skipped": f"exception: {e}"})
    mean = float(np.mean(aucs)) if aucs else 0.5
    if return_per_fold:
        return mean, fold_info
    return mean


def evaluate_lopo(feat, outcome, gene_list):
    """29-fold leave-one-patient-out AUC for one panel."""
    cols = list(gene_list) + FIXED_COVS
    y = feat[outcome].astype(int).values
    X = feat[cols].values
    n = len(feat)
    scores = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        try:
            mdl = LogisticRegression(
                class_weight='balanced', max_iter=500,
                solver='lbfgs', C=1.0, random_state=42)
            mdl.fit(X[tr], y[tr])
            scores[i] = mdl.predict_proba(X[i:i+1])[0, 1]
        except Exception:
            scores[i] = np.nan
    valid = ~np.isnan(scores)
    if valid.sum() < 5 or len(np.unique(y[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(y[valid], scores[valid]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {"seed": SEED, "n_folds": N_FOLDS,
               "n_cv_seeds": N_CV_SEEDS, "n_random_panels": N_RANDOM_PANELS,
               "outcomes": {}}

    print(f"[{time.strftime('%H:%M:%S')}] Loading {DATA_PATH} ...")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"  Raw: {adata.shape}")

    pb = make_pseudobulk(adata)
    pb = pb.loc[:, (pb > 0).mean() > 0.05]
    pb_norm = normalize_01(pb)
    obs = (adata.obs[['pt_id', 'Sex', 'cvda', 't2dm']]
           .drop_duplicates('pt_id').set_index('pt_id'))
    obs['Sex_binary'] = (obs['Sex'] == 'Male').astype(float)
    common = pb_norm.index.intersection(obs.index)
    pb_norm = pb_norm.loc[common]
    obs = obs.loc[common]
    feat = pb_norm.copy()
    feat['Sex_binary'] = obs['Sex_binary']
    feat['cvda'] = obs['cvda'].astype(float)
    feat['t2dm'] = obs['t2dm'].astype(float)
    gene_pool = pb_norm.columns.tolist()
    print(f"  Patients: {len(common)}  Genes: {len(gene_pool)}")

    for outcome in ('t2dm', 'cvda'):
        print(f"\n{'='*60}\n  outcome: {outcome}\n{'='*60}")
        out = {}

        # Load MUSTER's saved gene panel
        final = sorted(Path(MUSTER_DIR).glob(f"muster_{outcome}_final_*.json"))[-1]
        with open(final) as f:
            mfin = json.load(f)
        gene_list = mfin["final_genes"]
        recorded = mfin["final_auc"]
        out["n_genes"] = len(gene_list)
        out["recorded_final_auc"] = recorded
        print(f"  N_genes={len(gene_list)}  recorded_final_auc={recorded:.4f}")

        valid = obs[outcome].dropna().index
        feat_o = feat.loc[valid].copy()
        labels = feat_o[outcome].astype(int).values
        ids = feat_o.index.tolist()
        print(f"  N_patients={len(ids)}  pos={labels.sum()}  neg={(labels==0).sum()}")
        out["n_patients"] = len(ids)
        out["n_pos"] = int(labels.sum())
        out["n_neg"] = int((labels == 0).sum())

        # ── D1: reproduce final_auc on the original CV folds ────────────────
        cv_orig = make_cv_folds(ids, labels, n_folds=N_FOLDS,
                                 rng=np.random.RandomState(SEED))
        auc_repro, per_fold = evaluate_panel(
            feat_o, cv_orig, outcome, gene_list, return_per_fold=True)
        out["D1_repro_auc"] = auc_repro
        out["D1_per_fold"] = per_fold
        print(f"  D1 reproduced CV AUC = {auc_repro:.4f}   (vs recorded {recorded:.4f})")
        for pf in per_fold:
            print(f"     fold {pf['fold']}: AUC={pf['auc']}  "
                  f"n_test={pf['n_test']} pos/neg={pf['n_pos_te']}/{pf['n_neg_te']}  "
                  f"{pf['skipped']}")

        # ── D6: per-fold class balance (already in D1's per_fold) ───────────

        # ── D2: 100 different CV seeds ──────────────────────────────────────
        t0 = time.time()
        cv_seed_aucs = []
        for cv_s in range(N_CV_SEEDS):
            cv = make_cv_folds(ids, labels, n_folds=N_FOLDS,
                                rng=np.random.RandomState(cv_s))
            cv_seed_aucs.append(evaluate_panel(feat_o, cv, outcome, gene_list))
        cv_seed_aucs = np.array(cv_seed_aucs)
        out["D2_cv_seed_mean"] = float(cv_seed_aucs.mean())
        out["D2_cv_seed_std"] = float(cv_seed_aucs.std())
        out["D2_cv_seed_q05"] = float(np.quantile(cv_seed_aucs, 0.05))
        out["D2_cv_seed_q95"] = float(np.quantile(cv_seed_aucs, 0.95))
        out["D2_cv_seed_min"] = float(cv_seed_aucs.min())
        out["D2_cv_seed_max"] = float(cv_seed_aucs.max())
        pd.Series(cv_seed_aucs, name="auc").to_csv(
            f"{OUT_DIR}/cv_seed_trace_{outcome}.csv")
        print(f"  D2 100 CV seeds: mean={cv_seed_aucs.mean():.4f}  "
              f"std={cv_seed_aucs.std():.4f}  "
              f"5%/95%={np.quantile(cv_seed_aucs, 0.05):.4f}/"
              f"{np.quantile(cv_seed_aucs, 0.95):.4f}  ({time.time()-t0:.1f}s)")

        # ── D4: LOPO-CV ─────────────────────────────────────────────────────
        t0 = time.time()
        lopo = evaluate_lopo(feat_o, outcome, gene_list)
        out["D4_lopo_auc"] = lopo
        print(f"  D4 LOPO-CV AUC = {lopo:.4f}  ({time.time()-t0:.1f}s)")

        # ── D5: random-panel control ────────────────────────────────────────
        t0 = time.time()
        rng = np.random.RandomState(42)
        random_aucs = np.zeros(N_RANDOM_PANELS)
        for i in range(N_RANDOM_PANELS):
            panel = list(rng.choice(gene_pool, len(gene_list), replace=False))
            random_aucs[i] = evaluate_panel(feat_o, cv_orig, outcome, panel)
        out["D5_random_mean"] = float(random_aucs.mean())
        out["D5_random_std"] = float(random_aucs.std())
        out["D5_random_q05"] = float(np.quantile(random_aucs, 0.05))
        out["D5_random_q50"] = float(np.quantile(random_aucs, 0.50))
        out["D5_random_q95"] = float(np.quantile(random_aucs, 0.95))
        out["D5_random_max"] = float(random_aucs.max())
        out["D5_random_p_geq_muster"] = float((random_aucs >= recorded - 1e-9).mean())
        pd.Series(random_aucs, name="auc").to_csv(
            f"{OUT_DIR}/random_panels_{outcome}.csv")
        print(f"  D5 1000 random panels (same K, same folds): "
              f"mean={random_aucs.mean():.4f}  q50={np.quantile(random_aucs, 0.5):.4f}  "
              f"max={random_aucs.max():.4f}  "
              f"P(rand >= MUSTER)={out['D5_random_p_geq_muster']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        summary["outcomes"][outcome] = out

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
