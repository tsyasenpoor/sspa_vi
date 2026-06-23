#!/usr/bin/env python
"""scHPF baseline for the bulk DRGP simulation (the unsupervised Poisson-factorization ancestor).

Fits scHPF on ALL samples (unsupervised, no label leak), extracts gene_score (beta, genes x K) for
recovery and cell_score (theta, samples x K) for a downstream L1-logistic classifier (trained on
the train labels, with the PRS covariate). Writes <out>/schpf/gene_programs.csv.gz (scoreable by
check_recovery) and <out>/schpf_metrics.json (held-out AUC). Run from jax_gpu with
PYTHONPATH including the vendored scHPF.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "/labs/Aguiar/SSPA_BRAY/scHPF")
import schpf  # noqa: E402
from VariationalInference.data_loader import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--aux-columns", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(f"{args.out}/schpf", exist_ok=True)

    dl = DataLoader(args.data, verbose=False)
    data = dl.load_and_preprocess(label_column=["heart_disease"], aux_columns=args.aux_columns,
                                  normalize=False, convert_to_ensembl=False,
                                  filter_protein_coding=False, random_state=args.seed,
                                  return_sparse=False)
    genes = dl.gene_list
    # concatenate splits, tracking which rows are train/val/test
    parts, ys, auxs, tags = [], [], [], []
    for tg in ["train", "val", "test"]:
        X, A, y = data[tg]
        parts.append(np.asarray(X)); ys.append(np.asarray(y).ravel())
        auxs.append(np.asarray(A)); tags += [tg] * X.shape[0]
    X = np.vstack(parts).astype(np.float64); y = np.concatenate(ys)
    aux = np.vstack(auxs) if auxs[0].size else np.zeros((X.shape[0], 0))
    tags = np.array(tags)
    tr, va, te = tags == "train", tags == "val", tags == "test"

    # scHPF: unsupervised fit on ALL samples (counts; cells x genes sparse)
    np.random.seed(args.seed)                  # scHPF seeds via the global RNG
    model = schpf.scHPF(nfactors=args.k)
    model.fit(sp.coo_matrix(X))
    beta = model.gene_score()     # genes x K
    theta = model.cell_score()    # samples x K
    pd.DataFrame(beta.T, index=[f"comp{k}" for k in range(args.k)], columns=genes) \
        .to_csv(f"{args.out}/schpf/gene_programs.csv.gz", compression="gzip")

    # prediction: L1-LR on [theta | PRS], train labels only
    def withaux(F, m):
        return np.hstack([F, aux[m]]) if aux.shape[1] else F
    sc = StandardScaler().fit(theta[tr])
    clf = LogisticRegression(penalty="l1", solver="saga", max_iter=3000, C=0.5)
    clf.fit(withaux(sc.transform(theta[tr]), tr), y[tr])
    res = {}
    for nm, m in [("val", va), ("test", te)]:
        if len(np.unique(y[m])) < 2:
            res[f"{nm}_auc"] = float("nan"); continue
        p = clf.predict_proba(withaux(sc.transform(theta[m]), m))[:, 1]
        res[f"{nm}_auc"] = float(roc_auc_score(y[m], p))
        res[f"{nm}_auprc"] = float(average_precision_score(y[m], p))
    json.dump({"schpf": res}, open(f"{args.out}/schpf_metrics.json", "w"), indent=2)
    print(f"scHPF K={args.k} seed={args.seed}: val_auc={res.get('val_auc', float('nan')):.3f} "
          f"test_auc={res.get('test_auc', float('nan')):.3f}", flush=True)
    print(f"DONE -> {args.out}/schpf/gene_programs.csv.gz + schpf_metrics.json", flush=True)


if __name__ == "__main__":
    main()
