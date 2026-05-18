#!/usr/bin/env python3
"""
MUSTER with held-out patient evaluation.

Difference vs muster_pbmc.py:
  - Before any GA, hold out 5 of 29 patients as TEST (stratified by outcome).
  - Phase I + Phase II run only on the remaining 24-patient CV.
  - Final gene panel is scored on the 5 held-out patients (held_out_auc).
  - The internal CV AUC (subject to GA peeking) is still reported as
    `final_auc_cv`, so the gap held_out_auc vs final_auc_cv quantifies
    CV-overfitting from the GA.

Parameterized via env vars (cleaner than CLI for the SLURM wrapper):
  MUSTER_DATA       : h5ad path (default allPBMC)
  MUSTER_OUTDIR     : results directory
  MUSTER_TAG        : tag added to output filenames (e.g. 'allPBMC' or 'monocytes')
  MUSTER_SEED       : top-level seed (default 10086)
  MUSTER_HOLDOUT_N  : number of held-out patients (default 5)
  MUSTER_PHASE1_CYC / MUSTER_PHASE2_CYC : cycle counts (defaults match original)
"""

import os, json, time, datetime
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── config (env-overridable) ─────────────────────────────────────────────────
DATA_PATH    = os.environ.get(
    "MUSTER_DATA",
    "/labs/Aguiar/SSPA_BRAY/dataset/biorepository/allPBMC_GEX_20260106.h5ad")
OUT_DIR      = os.environ.get(
    "MUSTER_OUTDIR",
    "/labs/Aguiar/SSPA_BRAY/results/muster_pbmc_holdout")
TAG          = os.environ.get("MUSTER_TAG", "allPBMC")
SEED         = int(os.environ.get("MUSTER_SEED", "10086"))
HOLDOUT_N    = int(os.environ.get("MUSTER_HOLDOUT_N", "5"))
PHASE1_CYC   = int(os.environ.get("MUSTER_PHASE1_CYC", "3000"))
PHASE2_CYC   = int(os.environ.get("MUSTER_PHASE2_CYC", "10000"))

OUTCOMES   = ['cvda', 't2dm']
FIXED_COVS = ['Sex_binary']
CAP        = 20
N_BATCHES  = 20
ANCESTOR_CYC = 250
MIN_SIG    = 5
MAX_SIG    = 15


# ── preprocessing (lifted verbatim from muster_pbmc.py) ──────────────────────
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
        test  = np.concatenate([pos_splits[i], neg_splits[i]])
        train = np.concatenate(
            [np.concatenate([s for j, s in enumerate(pos_splits) if j != i]),
             np.concatenate([s for j, s in enumerate(neg_splits) if j != i])])
        folds.append((train, test))
    return folds


def evaluate(feat_df, cv_folds, outcome, gene_list):
    cols = gene_list + FIXED_COVS
    aucs = []
    for train_ids, test_ids in cv_folds:
        X_tr = feat_df.loc[train_ids, cols].values
        y_tr = feat_df.loc[train_ids, outcome].values
        X_te = feat_df.loc[test_ids,  cols].values
        y_te = feat_df.loc[test_ids,  outcome].values
        if len(np.unique(y_te)) < 2:
            continue
        try:
            mdl = LogisticRegression(
                class_weight='balanced', max_iter=500,
                solver='lbfgs', C=1.0, random_state=42)
            mdl.fit(X_tr, y_tr)
            aucs.append(roc_auc_score(y_te, mdl.predict_proba(X_te)[:, 1]))
        except Exception:
            aucs.append(0.5)
    return float(np.mean(aucs)) if aucs else 0.5


def evaluate_holdout(feat_df_train, feat_df_test, outcome, gene_list):
    cols = gene_list + FIXED_COVS
    X_tr = feat_df_train[cols].values
    y_tr = feat_df_train[outcome].values
    X_te = feat_df_test[cols].values
    y_te = feat_df_test[outcome].values
    if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
        return float("nan")
    mdl = LogisticRegression(
        class_weight='balanced', max_iter=500,
        solver='lbfgs', C=1.0, random_state=42)
    mdl.fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, mdl.predict_proba(X_te)[:, 1]))


def mut_probs(san_check, max_mut=10):
    if   san_check <=  500: f = 1.0
    elif san_check <= 1000: f = 0.5
    elif san_check <= 2000: f = 0.2
    else:                   f = 0.1
    ns = np.arange(1, max_mut + 1)
    p  = np.exp(-f * ns)
    return p / p.sum()


def mk_ancestor(feat_df, cv_folds, outcome, gene_pool, cycles, rng):
    best_auc   = 0.5
    best_genes = list(rng.choice(gene_pool, rng.randint(MIN_SIG, MAX_SIG + 1), replace=False))
    for cyc in range(cycles):
        n = rng.randint(MIN_SIG, MAX_SIG + 1)
        candidate = list(rng.choice(gene_pool, n, replace=False))
        auc = evaluate(feat_df, cv_folds, outcome, candidate)
        if auc > best_auc:
            best_auc, best_genes = auc, candidate
    return best_genes, best_auc


def evove(feat_df, cv_folds, outcome, gene_pool, ancestor, anc_auc,
          cycles, rng, tag=''):
    pool_set = set(gene_pool)
    cur, best = ancestor[:], ancestor[:]
    best_auc = anc_auc
    san_check = 0
    auc_trace = np.zeros(cycles)
    for gen in range(cycles):
        probs = mut_probs(san_check)
        mut_size = int(rng.choice(np.arange(1, 11), p=probs))
        mut_size = min(mut_size, len(cur))
        outside = list(pool_set - set(cur))
        mutation = rng.choice(['inse', 'dele', 'repl'])
        if mutation == 'inse' and outside:
            n = min(mut_size, len(outside))
            new = cur + list(rng.choice(outside, n, replace=False))
        elif mutation == 'dele' and len(cur) > 1:
            drop = set(rng.choice(len(cur), mut_size, replace=False))
            new = [g for i, g in enumerate(cur) if i not in drop]
        elif mutation == 'repl' and outside:
            n = min(mut_size, len(outside))
            drop = set(rng.choice(len(cur), n, replace=False))
            new = [g for i, g in enumerate(cur) if i not in drop]
            new += list(rng.choice(outside, n, replace=False))
        else:
            new = cur[:]
        if len(new) > CAP:
            trim = rng.randint(len(new) - CAP, len(new) - CAP + 4)
            drop = set(rng.choice(len(new), trim, replace=False))
            new = [g for i, g in enumerate(new) if i not in drop]
        if not new:
            new = best[:]
        new_auc = evaluate(feat_df, cv_folds, outcome, new)
        if new_auc > best_auc:
            best_auc, best, cur = new_auc, new[:], new[:]
            san_check = 0
        else:
            cur = best[:]
            san_check += 1
        auc_trace[gen] = best_auc
        if (gen + 1) % 1000 == 0:
            print(f"    {tag} gen {gen+1:5d}: AUC={best_auc:.4f}  san={san_check}  n={len(best)}")
    return best, best_auc, auc_trace


def batch_phase1(feat_df, cv_folds, outcome, gene_pool, seeds):
    all_genes = set()
    results = []
    for b, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        print(f"\n  [Batch {b+1}/{len(seeds)}] seed={seed}")
        ancestor, anc_auc = mk_ancestor(
            feat_df, cv_folds, outcome, gene_pool, ANCESTOR_CYC, rng)
        genes, auc, _ = evove(
            feat_df, cv_folds, outcome, gene_pool,
            ancestor, anc_auc, PHASE1_CYC, rng, tag=f'b{b+1}')
        print(f"    batch {b+1} CV AUC={auc:.4f}  genes={len(genes)}")
        all_genes.update(genes)
        results.append({'batch': b+1, 'seed': int(seed),
                         'auc_cv': auc, 'genes': genes})
    return list(all_genes), results


def split_train_test_patients(ids, labels, holdout_n, rng):
    """Stratified holdout: pick floor(holdout_n/2) pos + ceil(holdout_n/2) neg."""
    ids, labels = np.array(ids), np.array(labels, dtype=int)
    pos = rng.permutation(ids[labels == 1])
    neg = rng.permutation(ids[labels == 0])
    n_pos_h = max(1, holdout_n // 2)
    n_neg_h = max(1, holdout_n - n_pos_h)
    test_ids  = np.concatenate([pos[:n_pos_h], neg[:n_neg_h]])
    train_ids = np.concatenate([pos[n_pos_h:], neg[n_neg_h:]])
    return train_ids.tolist(), test_ids.tolist()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    td = datetime.date.today().strftime('%Y-%m-%d')
    rng = np.random.RandomState(SEED)

    print(f"[{datetime.datetime.now()}] Loading {DATA_PATH} ...")
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
    print(f"  Patients: {len(common)}")

    gene_pool_all = pb_norm.columns.tolist()
    feat = pb_norm.copy()
    feat['Sex_binary'] = obs['Sex_binary']
    feat['cvda'] = obs['cvda'].astype(float)
    feat['t2dm'] = obs['t2dm'].astype(float)

    batch_seeds = rng.choice(99999, N_BATCHES, replace=False).tolist()

    for outcome in OUTCOMES:
        print(f"\n{'='*60}")
        print(f"[{datetime.datetime.now()}] MUSTER-holdout — {TAG} / {outcome}")
        valid = obs[outcome].dropna().index
        feat_o = feat.loc[valid].copy()
        labels = feat_o[outcome].astype(int).values
        ids    = feat_o.index.tolist()
        print(f"  N={len(ids)}, pos={labels.sum()}, neg={(labels==0).sum()}")

        # ── held-out split ────────────────────────────────────────────────
        split_rng = np.random.RandomState(SEED + 1)
        train_pts, test_pts = split_train_test_patients(
            ids, labels, HOLDOUT_N, split_rng)
        print(f"  Held out: {test_pts}  (n_train={len(train_pts)}, n_test={len(test_pts)})")

        feat_train = feat_o.loc[train_pts]
        feat_test  = feat_o.loc[test_pts]
        labels_train = feat_train[outcome].astype(int).values
        ids_train    = feat_train.index.tolist()

        cv_folds = make_cv_folds(ids_train, labels_train, n_folds=5,
                                  rng=np.random.RandomState(SEED))

        t0 = time.time()
        print(f"\n--- Phase I ({N_BATCHES} batches × {PHASE1_CYC} cycles) ---")
        enriched, p1_results = batch_phase1(
            feat_train, cv_folds, outcome, gene_pool_all, batch_seeds)
        print(f"\nPhase I enriched pool: {len(enriched)} genes  "
              f"({(time.time()-t0)/60:.1f} min)")

        # Per-batch holdout AUC for stability check
        for r in p1_results:
            r["auc_holdout"] = evaluate_holdout(
                feat_train, feat_test, outcome, r["genes"])

        with open(f"{OUT_DIR}/muster_{TAG}_{outcome}_phase1_{td}.json", 'w') as f:
            json.dump({'outcome': outcome, 'tag': TAG,
                       'enriched_genes': enriched,
                       'batches': p1_results,
                       'train_patients': train_pts, 'test_patients': test_pts}, f, indent=2)

        print(f"\n--- Phase II ({PHASE2_CYC} cycles from enriched pool) ---")
        t1 = time.time()
        rng2 = np.random.RandomState(SEED)
        cv2 = make_cv_folds(ids_train, labels_train, n_folds=5,
                              rng=np.random.RandomState(SEED))
        anc2, anc_auc2 = mk_ancestor(
            feat_train, cv2, outcome, enriched, ANCESTOR_CYC, rng2)
        final_genes, final_auc_cv, trace = evove(
            feat_train, cv2, outcome, enriched, anc2, anc_auc2,
            PHASE2_CYC, rng2, tag='p2')

        # ── score on held-out ────────────────────────────────────────────
        holdout_auc = evaluate_holdout(
            feat_train, feat_test, outcome, final_genes)

        print(f"\nPhase II CV AUC (peeked) = {final_auc_cv:.4f}")
        print(f"Held-out AUC ({HOLDOUT_N} patients) = {holdout_auc:.4f}")
        print(f"Super Species ({len(final_genes)} genes): {final_genes}")
        print(f"Phase II time: {(time.time()-t1)/60:.1f} min  "
              f"(total: {(time.time()-t0)/60:.1f} min)")

        with open(f"{OUT_DIR}/muster_{TAG}_{outcome}_final_{td}.json", 'w') as f:
            json.dump({'outcome': outcome, 'tag': TAG,
                       'final_auc_cv': final_auc_cv,
                       'holdout_auc': holdout_auc,
                       'final_genes': final_genes,
                       'auc_trace': trace.tolist(),
                       'train_patients': train_pts,
                       'test_patients': test_pts}, f, indent=2)

        pd.Series(trace, name='auc').to_csv(
            f"{OUT_DIR}/muster_{TAG}_{outcome}_auc_trace_{td}.csv")

    print(f"\n[{datetime.datetime.now()}] Done.")


if __name__ == '__main__':
    main()
