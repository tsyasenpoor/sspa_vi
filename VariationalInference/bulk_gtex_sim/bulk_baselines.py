#!/usr/bin/env python
"""Baseline methods for the bulk DRGP simulation: NMF, PCA (factor methods -> recovery + prediction)
and gene-L1-LR (prediction only). Uses the SAME DataLoader splits as the DRGP fit, and gives every
method the PRS aux when present (fair [X | X_aux] comparison). scHPF is added separately (needs the
vendored scHPF env); it is the key recovery baseline per the design.

Per factor method writes <out>/<method>/gene_programs.csv.gz (K x p, gene columns) so
check_recovery.py can score recovery identically to DRGP. Held-out AUC -> <out>/<method>/metrics.json.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from VariationalInference.data_loader import DataLoader


def predict_auc(Ftr, Fva, Fte, ytr, yva, yte, l1=False):
    """LR on factor/gene scores; report val+test AUC/AUPRC."""
    clf = LogisticRegression(penalty="l1", solver="saga", max_iter=3000, C=0.5) if l1 \
        else LogisticRegression(max_iter=3000)
    sc = StandardScaler().fit(Ftr)
    clf.fit(sc.transform(Ftr), ytr)
    out = {}
    for name, F, y in [("val", Fva, yva), ("test", Fte, yte)]:
        if len(np.unique(y)) < 2:
            out[f"{name}_auc"] = float("nan"); continue
        prob = clf.predict_proba(sc.transform(F))[:, 1]
        out[f"{name}_auc"] = float(roc_auc_score(y, prob))
        out[f"{name}_auprc"] = float(average_precision_score(y, prob))
    return out


def save_programs(out_dir, beta, genes):
    """beta: p x K -> <out>/gene_programs.csv.gz (K x p, gene columns) for check_recovery."""
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(beta.T, index=[f"comp{k}" for k in range(beta.shape[1])], columns=genes) \
        .to_csv(f"{out_dir}/gene_programs.csv.gz", compression="gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--aux-columns", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dl = DataLoader(args.data, verbose=False)
    data = dl.load_and_preprocess(label_column=["heart_disease"], aux_columns=args.aux_columns,
                                  normalize=False, convert_to_ensembl=False,
                                  filter_protein_coding=False, random_state=args.seed,
                                  return_sparse=False)
    genes = dl.gene_list if dl.gene_list is not None else \
        [f"g{j}" for j in range(data["train"][0].shape[1])]
    (Xtr, Atr, ytr), (Xva, Ava, yva), (Xte, Ate, yte) = \
        (data["train"], data["val"], data["test"])
    Xtr, Xva, Xte = np.asarray(Xtr), np.asarray(Xva), np.asarray(Xte)
    ytr, yva, yte = np.asarray(ytr).ravel(), np.asarray(yva).ravel(), np.asarray(yte).ravel()

    def withaux(F, A):
        A = np.asarray(A)
        return np.hstack([F, A]) if A.size else F

    results = {}
    # --- NMF (counts; nonneg) -> recovery + prediction ---
    nmf = NMF(n_components=args.k, init="nndsvda", max_iter=400, random_state=args.seed)
    Wtr = nmf.fit_transform(Xtr); beta_nmf = nmf.components_.T          # p x K
    save_programs(f"{args.out}/nmf", beta_nmf, genes)
    results["nmf"] = predict_auc(withaux(Wtr, Atr), withaux(nmf.transform(Xva), Ava),
                                 withaux(nmf.transform(Xte), Ate), ytr, yva, yte)

    # --- PCA on log1p (recovery + prediction) ---
    Ltr = np.log1p(Xtr)
    pca = PCA(n_components=args.k, random_state=args.seed).fit(Ltr)
    beta_pca = pca.components_.T                                        # p x K
    save_programs(f"{args.out}/pca", beta_pca, genes)
    results["pca"] = predict_auc(pca.transform(Ltr), pca.transform(np.log1p(Xva)),
                                 pca.transform(np.log1p(Xte)), ytr, yva, yte)
    # attach aux for pca prediction too
    results["pca"] = predict_auc(withaux(pca.transform(Ltr), Atr),
                                 withaux(pca.transform(np.log1p(Xva)), Ava),
                                 withaux(pca.transform(np.log1p(Xte)), Ate), ytr, yva, yte)

    # --- gene-L1-LR (prediction only; gene-level, no programs) ---
    results["gene_l1lr"] = predict_auc(withaux(np.log1p(Xtr), Atr), withaux(np.log1p(Xva), Ava),
                                       withaux(np.log1p(Xte), Ate), ytr, yva, yte, l1=True)

    os.makedirs(args.out, exist_ok=True)
    json.dump(results, open(f"{args.out}/baseline_metrics.json", "w"), indent=2)
    for m, r in results.items():
        print(f"  {m:>10}: val_auc={r.get('val_auc', float('nan')):.3f} "
              f"test_auc={r.get('test_auc', float('nan')):.3f}", flush=True)
    print(f"DONE -> {args.out}/ (nmf/pca gene_programs.csv.gz + baseline_metrics.json)", flush=True)


if __name__ == "__main__":
    main()
