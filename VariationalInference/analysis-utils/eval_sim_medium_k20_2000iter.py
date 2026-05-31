#!/usr/bin/env python
"""Aggregate the K=20/2000iter DRGP rerun with the existing K=20 grid results.

Pulls from:
  - DRGP K=20/2000iter   <- comparison_k20_2000iter/drgp_unmasked/
  - MFLR K=20            <- comparison_k20/baselines/
  - scHPF K=20           <- comparison_k20/schpf/
  - LR/LRL/LRR (K-indep) <- comparison/baselines/
  - spectra_unsup        <- comparison/spectra_unsup/

Writes:
  - comparison_k20_2000iter/comparison_table.csv
  - comparison_k20_2000iter/comparison_summary.csv
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

warnings.filterwarnings("ignore")
sys.path.insert(0, "/labs/Aguiar/SSPA_BRAY/BRay")
from VariationalInference.gene_convertor import GeneIDConverter  # noqa: E402

DATA_DIR = Path("/labs/Aguiar/SSPA_BRAY/scdesign3_PBMC_10kcells_2kgenes/sim_celltype/sim_medium")
OUT_ROOT = Path("/labs/Aguiar/SSPA_BRAY/results/sim_medium/comparison_k20_2000iter")
K20_DIR  = Path("/labs/Aguiar/SSPA_BRAY/results/sim_medium/comparison_k20")
K10_DIR  = Path("/labs/Aguiar/SSPA_BRAY/results/sim_medium/comparison")

SEEDS = [42, 123, 456, 789, 1024]
MODULE_ORDER = ["myeloid", "tcell", "lymphocyte", "shared"]

gt = json.load(open(DATA_DIR / "ground_truth.json"))
up_syms = set(gt["all_up_genes"]); dn_syms = set(gt["all_down_genes"])
module_markers = {m: set(gt["modules"][m]["up_genes"]) | set(gt["modules"][m]["down_genes"])
                  for m in MODULE_ORDER}

conv = GeneIDConverter(cache_file="/labs/Aguiar/SSPA_BRAY/BRay/gene_id_cache.json")
up_ens_map, _ = conv.symbols_to_ensembl(sorted(up_syms), species="human")
dn_ens_map, _ = conv.symbols_to_ensembl(sorted(dn_syms), species="human")
ens_to_sym = {**{v: k for k, v in up_ens_map.items()},
              **{v: k for k, v in dn_ens_map.items()}}


def auroc_marker(s, pos):
    y = pd.Series(0, index=s.index); c = s.index.intersection(pos)
    if not len(c): return float("nan")
    y.loc[list(c)] = 1
    if y.sum() == 0 or y.sum() == len(y): return float("nan")
    return roc_auc_score(y, s)

def auprc_marker(s, pos):
    y = pd.Series(0, index=s.index); c = s.index.intersection(pos)
    if not len(c): return float("nan")
    y.loc[list(c)] = 1
    if y.sum() == 0: return float("nan")
    p, r, _ = precision_recall_curve(y, s); return float(auc(r, p))

def to_symbol_index(score):
    if not score.index.astype(str).str.startswith("ENSG").any(): return score
    new_idx = [ens_to_sym.get(g, g) for g in score.index]
    out = pd.Series(score.values, index=new_idx)
    return out[~out.index.duplicated(keep="first")]

def score_metrics(score):
    s = to_symbol_index(score).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    out = {"up_auroc":  auroc_marker( s, up_syms),
           "dn_auroc":  auroc_marker(-s, dn_syms),
           "any_auroc": auroc_marker( s.abs(), up_syms | dn_syms),
           "any_auprc": auprc_marker( s.abs(), up_syms | dn_syms)}
    for m in MODULE_ORDER:
        out[f"{m}_auprc"] = auprc_marker(s.abs(), module_markers[m])
    return out


def drgp_gene_score(d):
    f = d / "vi_gene_programs.csv.gz"
    if not f.exists(): return pd.Series(dtype=float)
    gp = pd.read_csv(f, index_col=0)
    v = gp["v_weight_disease"].values
    B = gp.drop(columns="v_weight_disease").T
    col = B.sum(axis=0).replace(0, np.nan)
    return pd.Series((B.div(col, axis=1).fillna(0.0).values * v).sum(axis=1),
                     index=B.index, name="drgp")

def drgp_test_auc(d):
    f = d / "vi_test_predictions.csv.gz"
    if not f.exists(): return float("nan")
    df = pd.read_csv(f)
    truth = next(c for c in df.columns if c.startswith("true_"))
    prob  = next(c for c in df.columns if c.startswith("prob_"))
    return roc_auc_score(df[truth], df[prob])

def baseline_gene_score(d, alg, gene_list_ens):
    p = d / f"{alg}_model.pkl"
    if not p.exists(): return pd.Series(dtype=float)
    m = joblib.load(p)
    coef = m.named_steps["logisticregression"].coef_[0]
    n = len(gene_list_ens)
    if alg.startswith("mf"):
        nmf = joblib.load(d / f"{alg}_nmf.pkl")
        Kc = nmf.components_.shape[0]
        gene_attr = coef[:Kc] @ nmf.components_
    else:
        gene_attr = coef[:n]
    return pd.Series(gene_attr[:n], index=gene_list_ens, name=alg)

def baseline_test_auc(d, alg):
    r = d / f"{alg}_results.pkl"
    return float(joblib.load(r).get("test_roc_auc", float("nan"))) if r.exists() else float("nan")

def _find_lr_model(d, prefix):
    p = d / "baselines" / "disease" / f"{prefix}_lr_model.pkl"
    if p.exists(): return p
    c = list((d / "baselines").rglob(f"{prefix}_lr_model.pkl"))
    return c[0] if c else None

def _summary_test_auc(d, summary_name, key):
    f = d / "baselines" / summary_name
    if not f.exists(): return float("nan")
    s = json.load(open(f))
    e = s.get("results", s).get(key, {})
    return float(e.get("test", {}).get("roc_auc", float("nan")))

def schpf_gene_score(d, gene_list_sym):
    gs = d / "model" / "gene_scores.npy"
    if not gs.exists(): return pd.Series(dtype=float)
    G = np.load(gs)
    lr = _find_lr_model(d, "schpf")
    if lr is None: return pd.Series(dtype=float)
    coef = joblib.load(lr).named_steps["logisticregression"].coef_[0][:G.shape[1]]
    n = min(len(gene_list_sym), G.shape[0])
    return pd.Series((G @ coef)[:n], index=gene_list_sym[:n], name="schpf")

def schpf_test_auc(d):
    return _summary_test_auc(d, "schpf_baselines_summary.json", "schpf_lr_disease")

def spectra_gene_score(d, gene_list_sym):
    fs = d / "model" / "spectra_factors.npy"
    if not fs.exists(): return pd.Series(dtype=float)
    F = np.load(fs)
    lr = _find_lr_model(d, "spectra")
    if lr is None: return pd.Series(dtype=float)
    coef = joblib.load(lr).named_steps["logisticregression"].coef_[0]
    Fmat = F.T if F.shape[0] == len(gene_list_sym) else F
    n = min(len(gene_list_sym), Fmat.shape[1])
    return pd.Series(coef[:Fmat.shape[0]] @ Fmat[:, :n], index=gene_list_sym[:n], name="spectra")

def spectra_test_auc(d):
    return _summary_test_auc(d, "spectra_baselines_summary.json", "spectra_lr_disease")


def _gene_lists():
    import anndata as ad
    a = ad.read_h5ad(DATA_DIR / "counts_perturbed.h5ad")
    gene_list_sym = list(a.var_names)
    from VariationalInference.data_loader import DataLoader
    loader = DataLoader(data_path=str(DATA_DIR / "counts_perturbed.h5ad"),
                        gene_annotation_path=None,
                        cache_dir="/labs/Aguiar/SSPA_BRAY/cache",
                        use_cache=True, verbose=False)
    data = loader.load_and_preprocess(
        label_column="disease", aux_columns=["sex", "comorbidity", "batch"],
        train_ratio=0.7, val_ratio=0.15, stratify_by="disease",
        min_cells_expressing=0.001, layer="raw", convert_to_ensembl=True,
        filter_protein_coding=False, random_state=42, normalize=True,
        normalize_target_sum=0.0, return_sparse=False)
    return gene_list_sym, list(data["gene_list"])


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    gene_sym, gene_ens = _gene_lists()
    print(f"sym genes: {len(gene_sym)}   ens genes: {len(gene_ens)}")

    rows = []
    LR_ALGS = ["lr", "lrl", "lrr"]
    MFLR_ALGS = ["mflr", "mflrl", "mflrr"]

    for seed in SEEDS:
        # DRGP K=20 / 2000iter (the rerun)
        d = OUT_ROOT / "drgp_unmasked" / f"seed{seed}"
        rows.append({"method": "drgp_unmasked", "seed": seed,
                     "test_auc": drgp_test_auc(d),
                     **score_metrics(drgp_gene_score(d))})

        # LR baselines (K-independent) <- comparison/baselines/
        bd_k10 = K10_DIR / "baselines" / f"seed{seed}"
        for alg in LR_ALGS:
            rows.append({"method": f"baseline_{alg}", "seed": seed,
                         "test_auc": baseline_test_auc(bd_k10, alg),
                         **score_metrics(baseline_gene_score(bd_k10, alg, gene_ens))})

        # MFLR baselines K=20 <- comparison_k20/baselines/
        bd_k20 = K20_DIR / "baselines" / f"seed{seed}"
        for alg in MFLR_ALGS:
            rows.append({"method": f"baseline_{alg}", "seed": seed,
                         "test_auc": baseline_test_auc(bd_k20, alg),
                         **score_metrics(baseline_gene_score(bd_k20, alg, gene_ens))})

        # scHPF K=20 <- comparison_k20/schpf/
        d = K20_DIR / "schpf" / f"seed{seed}"
        rows.append({"method": "schpf", "seed": seed,
                     "test_auc": schpf_test_auc(d),
                     **score_metrics(schpf_gene_score(d, gene_sym))})

        # spectra_unsup K=10 <- comparison/spectra_unsup/
        d = K10_DIR / "spectra_unsup" / f"seed{seed}"
        rows.append({"method": "spectra_unsup", "seed": seed,
                     "test_auc": spectra_test_auc(d),
                     **score_metrics(spectra_gene_score(d, gene_sym))})

    df = pd.DataFrame(rows)
    out_csv = OUT_ROOT / "comparison_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df)} rows)")

    metric_cols = ["test_auc", "up_auroc", "dn_auroc", "any_auroc", "any_auprc"] + \
                  [f"{m}_auprc" for m in MODULE_ORDER]
    summary = df.groupby("method")[metric_cols].agg(["mean", "std"]).round(3)
    summary.to_csv(OUT_ROOT / "comparison_summary.csv")
    print(f"Wrote {OUT_ROOT / 'comparison_summary.csv'}")

    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    order = ["drgp_unmasked",
             "baseline_lr", "baseline_lrl", "baseline_lrr",
             "baseline_mflr", "baseline_mflrl", "baseline_mflrr",
             "schpf", "spectra_unsup"]
    print("\n=== Mean ± Std across seeds ===")
    print(summary.reindex(order).to_string())

    # Also report the cross-seed intercept/|v|_max stability for DRGP
    print("\n=== DRGP K=20/2000iter convergence check ===")
    print(f"{'seed':>5s}  {'test_auc':>9s}  {'intercept':>9s}  {'|v|_max':>8s}  {'mean|v|':>8s}")
    for s in SEEDS:
        d = OUT_ROOT / "drgp_unmasked" / f"seed{s}"
        g = pd.read_csv(d / "vi_gamma_weights.csv.gz").iloc[0]
        gp = pd.read_csv(d / "vi_gene_programs.csv.gz", index_col=0)
        v = gp["v_weight_disease"].values
        pred = pd.read_csv(d / "vi_test_predictions.csv.gz")
        a = roc_auc_score(pred["true_disease"], pred["prob_disease"])
        print(f"{s:5d}  {a:9.4f}  {g['intercept']:+9.4f}  {np.abs(v).max():8.3f}  {np.abs(v).mean():8.3f}")


if __name__ == "__main__":
    main()
