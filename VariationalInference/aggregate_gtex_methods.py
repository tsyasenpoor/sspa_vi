"""Aggregate GTEx experimental Part-3 held-out test AUC across methods x seeds.

DRGP test AUC  <- {methods}/drgp_{mode}/seed{S}/vi_metrics.csv  (split=test, level=cell)
baseline AUC   <- {methods}/baselines/seed{S}/{alg}_results.pkl  (key 'test_roc_auc')

Writes a tidy long table + a mean/sd summary to results/gtex_experimental/methods/.
"""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
import numpy as np
import joblib

M = Path("/labs/Aguiar/SSPA_BRAY/results/gtex_experimental/methods")
SEEDS = [42, 123, 456, 789, 1024]
BASE_ALGS = ["lr", "lrl", "lrr", "mflr", "mflrl", "mflrr"]
PRETTY = {"lr": "gene-LR", "lrl": "gene-L1-LR", "lrr": "gene-L2-LR",
          "mflr": "NMF+LR", "mflrl": "NMF+L1-LR", "mflrr": "NMF+L2-LR",
          "pclr": "PCA+LR", "pclrl": "PCA+L1-LR", "pclrr": "PCA+L2-LR"}
PCA_ALGS = ["pclr", "pclrl", "pclrr"]
LABEL = "heart_disease"  # downstream label for scHPF/Spectra summaries


def drgp_auc(mode: str, seed: int):
    p = M / f"drgp_{mode}" / f"seed{seed}" / "vi_metrics.csv"
    if not p.exists():
        return None
    with open(p) as f:
        for row in csv.DictReader(f):
            if row.get("split") == "test" and row.get("level") == "cell":
                return float(row["auc"])
    return None


def base_auc(alg: str, seed: int, subdir: str = "baselines"):
    p = M / subdir / f"seed{seed}" / f"{alg}_results.pkl"
    if not p.exists():
        return None
    try:
        d = joblib.load(p)
    except Exception:
        return None
    v = d.get("test_roc_auc", d.get("test", {}).get("roc_auc") if isinstance(d.get("test"), dict) else None)
    return float(v) if v is not None else None


def _embed_auc(summary_path: Path, prefix: str):
    """Best test AUC across the lr/lrl/lrr downstream heads in a scHPF/Spectra summary."""
    if not summary_path.exists():
        return None
    try:
        d = json.load(open(summary_path))
    except Exception:
        return None
    res = d.get("results", {})
    aucs = [v["test"]["roc_auc"] for k, v in res.items()
            if k.endswith(f"_{LABEL}") and isinstance(v.get("test"), dict)
            and v["test"].get("roc_auc") is not None]
    return float(max(aucs)) if aucs else None


def schpf_auc(seed: int):
    return _embed_auc(M / "schpf" / f"seed{seed}" / "baselines" / "schpf_baselines_summary.json", "schpf")


def spectra_auc(seed: int):
    return _embed_auc(M / "spectra" / f"seed{seed}" / "baselines" / "spectra_baselines_summary.json", "spectra")


def main():
    global M
    if len(sys.argv) > 1:           # optional: aggregate a different methods dir (aux-ablation re-runs)
        M = Path(sys.argv[1])
    rows = []  # (method, seed, auc)
    for mode in ("unmasked", "masked"):
        for s in SEEDS:
            a = drgp_auc(mode, s)
            if a is not None:
                rows.append((f"DRGP-{mode}", s, a))
    for alg in BASE_ALGS:
        for s in SEEDS:
            a = base_auc(alg, s)
            if a is not None:
                rows.append((PRETTY[alg], s, a))
    for alg in PCA_ALGS:
        for s in SEEDS:
            a = base_auc(alg, s, subdir="pca")          # full-aux: dedicated pca/ dir
            if a is None:
                a = base_auc(alg, s, subdir="baselines")  # ablation re-runs: pclr inside baselines/
            if a is not None:
                rows.append((PRETTY[alg], s, a))
    for s in SEEDS:
        a = schpf_auc(s)
        if a is not None:
            rows.append(("scHPF+LR", s, a))
        a = spectra_auc(s)
        if a is not None:
            rows.append(("Spectra+LR", s, a))

    long_p = M / "auc_long.csv"
    with open(long_p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "seed", "test_auc"]); w.writerows(rows)

    # summarize
    methods = dict.fromkeys(m for m, _, _ in rows)
    summ = []
    for m in methods:
        vals = [a for mm, _, a in rows if mm == m]
        summ.append((m, float(np.mean(vals)), float(np.std(vals)), len(vals)))
    summ.sort(key=lambda x: -x[1])
    sum_p = M / "auc_summary.csv"
    with open(sum_p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "mean_test_auc", "sd", "n_seeds"])
        for m, mu, sd, n in summ:
            w.writerow([m, f"{mu:.4f}", f"{sd:.4f}", n])

    print(f"=== GTEx experimental held-out test AUC (heart_disease), {len(rows)} runs ===")
    print(f"{'method':16s} {'mean AUC':>9s} {'sd':>7s} {'n':>3s}")
    for m, mu, sd, n in summ:
        print(f"{m:16s} {mu:9.4f} {sd:7.4f} {n:3d}")
    print(f"\nwrote {sum_p}\n      {long_p}")


if __name__ == "__main__":
    main()
