#!/usr/bin/env python
"""Aggregate sim_covid easy/medium/hard 4-method x 5-seed comparison.

Usage:
    python eval_sim_covid_comparison.py {easy|medium|hard}

For 'hard', also computes test_auc_patient (mean-pool cell probs per patient_id,
then AUC over patient labels). For easy/medium, only per-cell test AUC.

Outputs comparison_table.csv + comparison_summary.csv under
/labs/Aguiar/SSPA_BRAY/results/sim_covid/sim_{variant}/comparison/.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

warnings.filterwarnings("ignore")
sys.path.insert(0, "/labs/Aguiar/SSPA_BRAY/BRay")
from VariationalInference.gene_convertor import GeneIDConverter  # noqa: E402


SEEDS = [42, 123, 456, 789, 1024]
K = 20
MODULE_ORDER = ["myeloid", "tcell", "lymphocyte", "shared"]
BASE_ALGS = ["lr", "lrl", "lrr", "mflr", "mflrl", "mflrr"]


def auroc_marker(score, pos):
    y = pd.Series(0, index=score.index)
    c = score.index.intersection(pos)
    if not len(c):
        return float("nan")
    y.loc[list(c)] = 1
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return roc_auc_score(y, score)


def auprc_marker(score, pos):
    """Bug-fixed (vs hard aggregator's version): assign y BEFORE checking sum."""
    y = pd.Series(0, index=score.index)
    c = score.index.intersection(pos)
    if not len(c):
        return float("nan")
    y.loc[list(c)] = 1
    if y.sum() == 0:
        return float("nan")
    p, r, _ = precision_recall_curve(y, score)
    return float(auc(r, p))


def to_symbol_index(score, ens_to_sym):
    if not score.index.astype(str).str.startswith("ENSG").any():
        return score
    new_idx = [ens_to_sym.get(g, g) for g in score.index]
    out = pd.Series(score.values, index=new_idx)
    return out[~out.index.duplicated(keep="first")]


def make_score_metrics(up_syms, dn_syms, module_markers, ens_to_sym):
    def score_metrics(score):
        s = (to_symbol_index(score, ens_to_sym)
             .astype(float).replace([np.inf, -np.inf], np.nan).dropna())
        out = {
            "up_auroc":  auroc_marker( s, up_syms),
            "dn_auroc":  auroc_marker(-s, dn_syms),
            "any_auroc": auroc_marker( s.abs(), up_syms | dn_syms),
            "any_auprc": auprc_marker( s.abs(), up_syms | dn_syms),
        }
        for m in MODULE_ORDER:
            out[f"{m}_auprc"] = auprc_marker(s.abs(), module_markers[m])
        return out
    return score_metrics


# ---------- per-method gene scores ----------
def drgp_gene_score(seed_dir):
    f = seed_dir / "vi_gene_programs.csv.gz"
    if not f.exists(): return pd.Series(dtype=float)
    gp = pd.read_csv(f, index_col=0)
    v = gp["v_weight_disease"].values
    B = gp.drop(columns="v_weight_disease").T
    col = B.sum(axis=0).replace(0, np.nan)
    return pd.Series((B.div(col, axis=1).fillna(0.0).values * v).sum(axis=1),
                     index=B.index, name="drgp")


def baseline_gene_score(seed_dir, alg, gene_list_ens):
    p = seed_dir / f"{alg}_model.pkl"
    if not p.exists(): return pd.Series(dtype=float)
    m = joblib.load(p)
    coef = m.named_steps["logisticregression"].coef_[0]
    n = len(gene_list_ens)
    if alg.startswith("mf"):
        nmf_p = seed_dir / f"{alg}_nmf.pkl"
        if not nmf_p.exists(): return pd.Series(dtype=float)
        nmf = joblib.load(nmf_p)
        Kc = nmf.components_.shape[0]
        gene_attr = coef[:Kc] @ nmf.components_
    else:
        gene_attr = coef[:n]
    return pd.Series(gene_attr[:n], index=gene_list_ens, name=alg)


def _find_lr_model(seed_dir, prefix):
    p = seed_dir / "baselines" / "disease" / f"{prefix}_lr_model.pkl"
    if p.exists(): return p
    cand = list((seed_dir / "baselines").rglob(f"{prefix}_lr_model.pkl"))
    return cand[0] if cand else None


def schpf_gene_score(seed_dir, gene_list_sym):
    gs_path = seed_dir / "model" / "gene_scores.npy"
    if not gs_path.exists(): return pd.Series(dtype=float)
    gs = np.load(gs_path)
    lr_pkl = _find_lr_model(seed_dir, "schpf")
    if lr_pkl is None: return pd.Series(dtype=float)
    coef = joblib.load(lr_pkl).named_steps["logisticregression"].coef_[0][:gs.shape[1]]
    gene_attr = gs @ coef
    n = min(len(gene_list_sym), gene_attr.shape[0])
    return pd.Series(gene_attr[:n], index=gene_list_sym[:n], name="schpf")


def spectra_gene_score(seed_dir, gene_list_sym):
    fs = seed_dir / "model" / "spectra_factors.npy"
    if not fs.exists(): return pd.Series(dtype=float)
    factor_scores = np.load(fs)
    lr_pkl = _find_lr_model(seed_dir, "spectra")
    if lr_pkl is None: return pd.Series(dtype=float)
    coef = joblib.load(lr_pkl).named_steps["logisticregression"].coef_[0]
    Fmat = factor_scores.T if factor_scores.shape[0] == len(gene_list_sym) else factor_scores
    gene_attr = coef[:Fmat.shape[0]] @ Fmat
    n = min(len(gene_list_sym), gene_attr.shape[0])
    return pd.Series(gene_attr[:n], index=gene_list_sym[:n], name="spectra")


# ---------- test AUCs ----------
def make_patient_auc(patient_of_cell, patient_label):
    if patient_of_cell is None:
        return None
    def patient_auc(pred_df, prob_col):
        df = pred_df.copy()
        df["patient_id"] = df["cell_id"].map(patient_of_cell)
        df = df.dropna(subset=["patient_id"])
        by_pt = df.groupby("patient_id")[prob_col].mean()
        labels = patient_label.reindex(by_pt.index)
        if labels.nunique() < 2:
            return float("nan")
        return float(roc_auc_score(labels.values, by_pt.values))
    return patient_auc


def drgp_aucs(seed_dir, patient_auc_fn):
    f = seed_dir / "vi_test_predictions.csv.gz"
    if not f.exists(): return float("nan"), float("nan")
    df = pd.read_csv(f)
    truth_col = next((c for c in df.columns if c.startswith("true_")), None)
    prob_col  = next((c for c in df.columns if c.startswith("prob_")), None)
    if truth_col is None or prob_col is None: return float("nan"), float("nan")
    cell_auc = roc_auc_score(df[truth_col], df[prob_col])
    if patient_auc_fn is None or "cell_id" not in df.columns:
        return cell_auc, float("nan")
    return cell_auc, patient_auc_fn(df, prob_col)


def baseline_aucs(seed_dir, alg, patient_auc_fn):
    r = seed_dir / f"{alg}_results.pkl"
    if not r.exists(): return float("nan"), float("nan")
    res = joblib.load(r)
    cell_auc = float(res.get("test_roc_auc", float("nan")))
    if patient_auc_fn is None: return cell_auc, float("nan")
    cids = res.get("cell_ids"); probs = res.get("y_prob")
    if cids is None or probs is None or len(cids) != len(probs):
        return cell_auc, float("nan")
    df = pd.DataFrame({"cell_id": np.asarray(cids), "prob_disease": np.asarray(probs)})
    return cell_auc, patient_auc_fn(df, "prob_disease")


def _summary_test_auc(seed_dir, summary_name, result_key):
    f = seed_dir / "baselines" / summary_name
    if not f.exists(): return float("nan")
    s = json.load(open(f))
    entry = s.get("results", s).get(result_key, {})
    test = entry.get("test", {}) if isinstance(entry, dict) else {}
    return float(test.get("roc_auc", float("nan")))


def _patient_auc_from_preds_npz(seed_dir, prefix, label, patient_auc_fn):
    if patient_auc_fn is None: return float("nan")
    p = seed_dir / "baselines" / label / f"{prefix}_lr_{label}_preds.npz"
    if not p.exists(): return float("nan")
    d = np.load(p, allow_pickle=True)
    if "cell_ids" not in d.files or "y_proba" not in d.files:
        return float("nan")
    proba = d["y_proba"]
    probs_disease = proba[:, 1] if proba.ndim == 2 else proba
    df = pd.DataFrame({"cell_id": d["cell_ids"], "prob_disease": probs_disease})
    return patient_auc_fn(df, "prob_disease")


def schpf_aucs(seed_dir, patient_auc_fn):
    return (_summary_test_auc(seed_dir, "schpf_baselines_summary.json", "schpf_lr_disease"),
            _patient_auc_from_preds_npz(seed_dir, "schpf", "disease", patient_auc_fn))


def spectra_aucs(seed_dir, patient_auc_fn):
    return (_summary_test_auc(seed_dir, "spectra_baselines_summary.json", "spectra_lr_disease"),
            _patient_auc_from_preds_npz(seed_dir, "spectra", "disease", patient_auc_fn))


# ---------- gene lists for ENS↔symbol ----------
def _gene_lists(data_dir):
    a = ad.read_h5ad(data_dir / "counts_perturbed.h5ad")
    gene_list_sym = list(a.var_names)
    from VariationalInference.data_loader import DataLoader
    loader = DataLoader(data_path=str(data_dir / "counts_perturbed.h5ad"),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["easy", "medium", "hard"])
    args = ap.parse_args()
    variant = args.variant

    data_dir    = Path(f"/labs/Aguiar/SSPA_BRAY/scdesign3_covid19_1kcells_2kgenes/sim_celltype/sim_{variant}")
    results_dir = Path(f"/labs/Aguiar/SSPA_BRAY/results/sim_covid/sim_{variant}/comparison")

    print(f"variant={variant}  data={data_dir}  results={results_dir}")

    gt = json.load(open(data_dir / "ground_truth.json"))
    up_syms = set(gt["all_up_genes"]); dn_syms = set(gt["all_down_genes"])
    module_markers = {m: set(gt["modules"][m]["up_genes"]) | set(gt["modules"][m]["down_genes"])
                      for m in MODULE_ORDER}

    conv = GeneIDConverter(cache_file="/labs/Aguiar/SSPA_BRAY/BRay/gene_id_cache.json")
    up_ens_map, _ = conv.symbols_to_ensembl(sorted(up_syms), species="human")
    dn_ens_map, _ = conv.symbols_to_ensembl(sorted(dn_syms), species="human")
    ens_to_sym = {**{v: k for k, v in up_ens_map.items()},
                  **{v: k for k, v in dn_ens_map.items()}}
    score_metrics = make_score_metrics(up_syms, dn_syms, module_markers, ens_to_sym)

    # Patient table only for hard
    if variant == "hard":
        a = ad.read_h5ad(data_dir / "counts_perturbed.h5ad")
        pt = (a.obs[["patient_id", "disease"]]
              .reset_index().rename(columns={"index": "cell_id"}))
        patient_of_cell = pt.set_index("cell_id")["patient_id"].to_dict()
        patient_label = pt.groupby("patient_id")["disease"].first()
        patient_auc_fn = make_patient_auc(patient_of_cell, patient_label)
    else:
        patient_auc_fn = None

    gene_sym, gene_ens = _gene_lists(data_dir)
    print(f"sym genes: {len(gene_sym)}   ens genes: {len(gene_ens)}")

    rows = []
    for seed in SEEDS:
        d = results_dir / "drgp_unmasked" / f"seed{seed}"
        cell_auc, pat_auc = drgp_aucs(d, patient_auc_fn)
        row = {"method": "drgp_unmasked", "seed": seed,
               "test_auc_cell": cell_auc, "test_auc_patient": pat_auc,
               **score_metrics(drgp_gene_score(d))}
        rows.append(row)

        bd = results_dir / "baselines" / f"seed{seed}"
        for alg in BASE_ALGS:
            cell_auc, pat_auc = baseline_aucs(bd, alg, patient_auc_fn)
            rows.append({"method": f"baseline_{alg}", "seed": seed,
                         "test_auc_cell": cell_auc, "test_auc_patient": pat_auc,
                         **score_metrics(baseline_gene_score(bd, alg, gene_ens))})

        d = results_dir / "schpf" / f"seed{seed}"
        cell_auc, pat_auc = schpf_aucs(d, patient_auc_fn)
        rows.append({"method": "schpf", "seed": seed,
                     "test_auc_cell": cell_auc, "test_auc_patient": pat_auc,
                     **score_metrics(schpf_gene_score(d, gene_sym))})

        d = results_dir / "spectra_unsup" / f"seed{seed}"
        cell_auc, pat_auc = spectra_aucs(d, patient_auc_fn)
        rows.append({"method": "spectra_unsup", "seed": seed,
                     "test_auc_cell": cell_auc, "test_auc_patient": pat_auc,
                     **score_metrics(spectra_gene_score(d, gene_sym))})

    df = pd.DataFrame(rows)
    if variant != "hard":
        df = df.drop(columns=["test_auc_patient"])
        df = df.rename(columns={"test_auc_cell": "test_auc"})

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "comparison_table.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")

    metric_cols = (["test_auc_cell", "test_auc_patient"] if variant == "hard"
                   else ["test_auc"]) + \
                  ["up_auroc", "dn_auroc", "any_auroc", "any_auprc"] + \
                  [f"{m}_auprc" for m in MODULE_ORDER]
    summary = df.groupby("method")[metric_cols].agg(["mean", "std"]).round(3)
    summary.to_csv(results_dir / "comparison_summary.csv")
    print(f"Wrote {results_dir / 'comparison_summary.csv'}")

    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    order = ["drgp_unmasked",
             "baseline_lr", "baseline_lrl", "baseline_lrr",
             "baseline_mflr", "baseline_mflrl", "baseline_mflrr",
             "schpf", "spectra_unsup"]
    print("\n=== Mean ± Std across seeds ===")
    print(summary.reindex(order).to_string())


if __name__ == "__main__":
    main()
