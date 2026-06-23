"""Aggregate COVID experimental BULK repeated-hold-out (methods/bulk_cv/).

5 random split-seeds per (N, method). Report mean +- SD test ROC-AUC across splits, per N.
DRGP severity <- {N}p/drgp_{mode}/split{s}/vi_metrics.csv (split=test, label='CoVID-19 severity')
baseline      <- {N}p/baselines/split{s}/{alg}_results.pkl (key 'test_roc_auc')
Writes methods/bulk_cv/auc_summary_bulkcv.csv (+ long).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import joblib

M = Path("/labs/Aguiar/SSPA_BRAY/results/covid_experimental_v2/methods/bulk_cv")
NS = [50, 100, 148]
SPLITS = [0, 1, 2, 3, 4]
BASE = {"lr": "gene-LR", "lrl": "gene-L1-LR", "lrr": "gene-L2-LR",
        "mflr": "NMF+LR", "mflrl": "NMF+L1-LR", "mflrr": "NMF+L2-LR"}
LABEL = "CoVID-19 severity"


def drgp(N, mode, s):
    p = M / f"{N}p" / f"drgp_{mode}" / f"split{s}" / "vi_metrics.csv"
    if not p.exists():
        return None
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("split") == "test" and r.get("label") == LABEL:
                try:
                    return float(r["auc"])
                except (KeyError, ValueError):
                    return None
    return None


def base(N, alg, s):
    p = M / f"{N}p" / "baselines" / f"split{s}" / f"{alg}_results.pkl"
    if not p.exists():
        return None
    try:
        return float(joblib.load(p).get("test_roc_auc"))
    except Exception:
        return None


def main():
    long_rows, summ = [], []
    for N in NS:
        methods = {}
        for mode in ("unmasked", "masked"):
            methods[f"DRGP-{mode}"] = [drgp(N, mode, s) for s in SPLITS]
        for alg, nm in BASE.items():
            methods[nm] = [base(N, alg, s) for s in SPLITS]
        for m, vals in methods.items():
            v = [x for x in vals if x is not None]
            for s, x in zip(SPLITS, vals):
                if x is not None:
                    long_rows.append((N, m, s, x))
            if v:
                summ.append((N, m, float(np.mean(v)), float(np.std(v)), len(v)))

    M.mkdir(parents=True, exist_ok=True)
    with open(M / "auc_long_bulkcv.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "method", "split_seed", "test_auc"]); w.writerows(long_rows)
    with open(M / "auc_summary_bulkcv.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "method", "mean_test_auc", "sd", "n_splits"])
        for N, m, mu, sd, n in summ:
            w.writerow([N, m, f"{mu:.4f}", f"{sd:.4f}", n])

    print(f"=== COVID bulk repeated hold-out (5 split-seeds): mean test AUC (severity) ===")
    for N in NS:
        rows = sorted([r for r in summ if r[0] == N], key=lambda x: -x[2])
        if not rows:
            print(f"\n[N={N}] (none)"); continue
        print(f"\n[N={N}]  {'method':14s} {'mean AUC':>9s} {'SD':>7s} {'k':>2s}")
        for _, m, mu, sd, n in rows:
            print(f"         {m:14s} {mu:9.3f} {sd:7.3f} {n:2d}")
    print(f"\nwrote {M/'auc_summary_bulkcv.csv'}")


if __name__ == "__main__":
    main()
