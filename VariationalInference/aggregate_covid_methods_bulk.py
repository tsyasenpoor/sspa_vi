"""Aggregate COVID experimental Part-3 (BULK) held-out test AUC across method x seed, per N.

DRGP  <- methods/bulk/{N}p/drgp_{mode}/seed{S}/vi_metrics.csv (split=test, label='CoVID-19 severity')
base  <- methods/bulk/{N}p/baselines/seed{S}/{alg}_results.pkl (key 'test_roc_auc')
Writes results/covid_experimental_v2/methods/bulk/auc_summary_bulk.csv (+ long table).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import joblib

M = Path("/labs/Aguiar/SSPA_BRAY/results/covid_experimental_v2/methods/bulk")
NS = [50, 100, 148]
SEEDS = [42, 123, 456, 789, 1024]
BASE = {"lr": "gene-LR", "lrl": "gene-L1-LR", "lrr": "gene-L2-LR",
        "mflr": "NMF+LR", "mflrl": "NMF+L1-LR", "mflrr": "NMF+L2-LR"}
LABEL = "CoVID-19 severity"


def drgp_auc(N, mode, seed):
    p = M / f"{N}p" / f"drgp_{mode}" / f"seed{seed}" / "vi_metrics.csv"
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


def base_auc(N, alg, seed):
    p = M / f"{N}p" / "baselines" / f"seed{seed}" / f"{alg}_results.pkl"
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
            methods[f"DRGP-{mode}"] = [drgp_auc(N, mode, s) for s in SEEDS]
        for alg, nm in BASE.items():
            methods[nm] = [base_auc(N, alg, s) for s in SEEDS]
        for m, vals in methods.items():
            v = [x for x in vals if x is not None]
            for s, x in zip(SEEDS, vals):
                if x is not None:
                    long_rows.append((N, m, s, x))
            if v:
                summ.append((N, m, float(np.mean(v)), float(np.std(v)), len(v)))

    M.mkdir(parents=True, exist_ok=True)
    with open(M / "auc_long_bulk.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "method", "seed", "test_auc"]); w.writerows(long_rows)
    with open(M / "auc_summary_bulk.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "method", "mean_test_auc", "sd", "n_seeds"])
        for N, m, mu, sd, n in summ:
            w.writerow([N, m, f"{mu:.4f}", f"{sd:.4f}", n])

    print(f"=== COVID bulk Part-3 held-out test AUC (severity) — {len(long_rows)} runs ===")
    for N in NS:
        rows = sorted([r for r in summ if r[0] == N], key=lambda x: -x[2])
        if not rows:
            print(f"\n[N={N}] (no results yet)"); continue
        print(f"\n[N={N}]  {'method':16s} {'mean AUC':>9s} {'sd':>7s} {'n':>3s}")
        for _, m, mu, sd, n in rows:
            print(f"          {m:16s} {mu:9.4f} {sd:7.4f} {n:3d}")
    print(f"\nwrote {M/'auc_summary_bulk.csv'}")


if __name__ == "__main__":
    main()
