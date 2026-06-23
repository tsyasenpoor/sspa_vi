#!/usr/bin/env python
"""Aggregate the operating-point method comparison (recovery + prediction) over 5 seeds.

DRGP is read from the gamma*=0 gamma_seeds fits (K=40, rw=0.005); NMF/PCA/gene-L1-LR from the
baselines sweep on the same datasets. Emits a recovery table (disease/nuisance support-AUPRC,
factor methods) and a prediction table (test AUC, all methods)."""
import glob, json, os, re
import numpy as np
import pandas as pd

ROOT = "/labs/Aguiar/SSPA_BRAY/data/Simulations/bulk_gtex_v1"
GS = f"{ROOT}/gamma_seeds"
BL = f"{ROOT}/baselines"
SEEDS = [0, 1, 2, 3, 4]


def recovery_from_txt(path):
    if not os.path.exists(path):
        return (np.nan, np.nan)
    t = open(path).read()
    m = re.search(r"disease=([\d.]+)\s+nuisance=([\d.]+)", t)
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def drgp_test_auc(fit_dir):
    m = os.path.join(fit_dir, "vi_metrics.csv")
    if not os.path.exists(m):
        return np.nan
    df = pd.read_csv(m)
    row = df[df["split"] == "test"]
    return float(row["auc"].iloc[0]) if len(row) else np.nan


def main():
    rec = {"DRGP": [], "scHPF": [], "NMF": [], "PCA": []}     # (disease, nuisance) per seed
    auc = {"DRGP": [], "scHPF+L1-LR": [], "NMF+L1-LR": [], "PCA+L1-LR": [], "gene-L1-LR": []}
    for s in SEEDS:
        cfg = f"{GS}/cfg_e3.0_t0_k40_rw0.005_g0.0_s0_i{s}"
        # DRGP recovery + prediction (existing fit)
        rec["DRGP"].append(recovery_from_txt(f"{cfg}/recovery.txt"))
        auc["DRGP"].append(drgp_test_auc(f"{cfg}/drgp_fit"))
        # baselines
        out = f"{BL}/seed{s}"
        rec["NMF"].append(recovery_from_txt(f"{out}/nmf_recovery.txt"))
        rec["PCA"].append(recovery_from_txt(f"{out}/pca_recovery.txt"))
        rec["scHPF"].append(recovery_from_txt(f"{out}/schpf_recovery.txt"))
        bm = f"{out}/baseline_metrics.json"
        if os.path.exists(bm):
            d = json.load(open(bm))
            auc["NMF+L1-LR"].append(d.get("nmf", {}).get("test_auc", np.nan))
            auc["PCA+L1-LR"].append(d.get("pca", {}).get("test_auc", np.nan))
            auc["gene-L1-LR"].append(d.get("gene_l1lr", {}).get("test_auc", np.nan))
        sm = f"{out}/schpf_metrics.json"
        if os.path.exists(sm):
            auc["scHPF+L1-LR"].append(json.load(open(sm)).get("schpf", {}).get("test_auc", np.nan))

    def ms(xs):
        xs = [x for x in xs if x == x]
        return (np.mean(xs), np.std(xs)) if xs else (np.nan, np.nan)

    print("=== RECOVERY (support-AUPRC, mean +/- sd over seeds; chance ~0.02) ===")
    print(f"{'method':>8} {'disease':>16} {'nuisance':>16}")
    for m, vals in rec.items():
        d = ms([v[0] for v in vals]); n = ms([v[1] for v in vals])
        print(f"{m:>8}  {d[0]:.3f} ({d[1]:.3f})   {n[0]:.3f} ({n[1]:.3f})")
    print("\n=== PREDICTION (test AUC, mean +/- sd over seeds) ===")
    for m, vals in auc.items():
        a = ms(vals)
        print(f"{m:>12}  {a[0]:.3f} ({a[1]:.3f})  n={len([x for x in vals if x==x])}")


if __name__ == "__main__":
    main()
