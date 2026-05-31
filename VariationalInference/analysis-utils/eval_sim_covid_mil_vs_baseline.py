#!/usr/bin/env python
"""Compare DRGP-MIL-lite vs original per-cell DRGP across the 3 sim_covid variants.

Reads original DRGP from comparison/drgp_unmasked/seed*/ and the MIL-lite
runs from comparison_mil/drgp_unmasked/seed*/. Reports test_auc + gene
recovery metrics for both.
"""
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

warnings.filterwarnings("ignore")
sys.path.insert(0, "/labs/Aguiar/SSPA_BRAY/BRay")
from VariationalInference.gene_convertor import GeneIDConverter  # noqa: E402  pylint: disable=unused-import


SEEDS = [42, 123, 456, 789, 1024]
MODULE_ORDER = ["myeloid", "tcell", "lymphocyte", "shared"]
RESULTS_ROOT = Path("/labs/Aguiar/SSPA_BRAY/results/sim_covid")
DATA_ROOT    = Path("/labs/Aguiar/SSPA_BRAY/scdesign3_covid19_1kcells_2kgenes/sim_celltype")


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
    p, r, _ = precision_recall_curve(y, s)
    return float(auc(r, p))


def drgp_gene_score(seed_dir):
    f = seed_dir / "vi_gene_programs.csv.gz"
    if not f.exists(): return pd.Series(dtype=float)
    gp = pd.read_csv(f, index_col=0)
    v = gp["v_weight_disease"].values
    B = gp.drop(columns="v_weight_disease").T
    col = B.sum(axis=0).replace(0, np.nan)
    return pd.Series((B.div(col, axis=1).fillna(0.0).values * v).sum(axis=1),
                     index=B.index, name="drgp")


def drgp_aucs(seed_dir, patient_of_cell=None, patient_label=None):
    f = seed_dir / "vi_test_predictions.csv.gz"
    if not f.exists(): return float("nan"), float("nan")
    df = pd.read_csv(f)
    truth_col = next((c for c in df.columns if c.startswith("true_")), None)
    prob_col  = next((c for c in df.columns if c.startswith("prob_")), None)
    if truth_col is None or prob_col is None: return float("nan"), float("nan")
    cell_auc = roc_auc_score(df[truth_col], df[prob_col])
    pat_auc = float("nan")
    if patient_of_cell is not None and "cell_id" in df.columns:
        x = df.copy()
        x["patient_id"] = x["cell_id"].map(patient_of_cell)
        x = x.dropna(subset=["patient_id"])
        by_pt = x.groupby("patient_id")[prob_col].mean()
        labels = patient_label.reindex(by_pt.index)
        if labels.nunique() >= 2:
            pat_auc = float(roc_auc_score(labels.values, by_pt.values))
    return cell_auc, pat_auc


def score_metrics(score, up_syms, dn_syms, module_markers, ens_to_sym):
    if score.index.astype(str).str.startswith("ENSG").any():
        new_idx = [ens_to_sym.get(g, g) for g in score.index]
        score = pd.Series(score.values, index=new_idx)
        score = score[~score.index.duplicated(keep="first")]
    s = score.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "up_auroc":  auroc_marker( s, up_syms),
        "dn_auroc":  auroc_marker(-s, dn_syms),
        "any_auroc": auroc_marker( s.abs(), up_syms | dn_syms),
        "any_auprc": auprc_marker( s.abs(), up_syms | dn_syms),
    }
    for m in MODULE_ORDER:
        out[f"{m}_auprc"] = auprc_marker(s.abs(), module_markers[m])
    return out


def patient_table(h5ad_path):
    a = ad.read_h5ad(h5ad_path)
    pt = (a.obs[["patient_id", "disease"]]
          .reset_index().rename(columns={"index": "cell_id"}))
    return (pt.set_index("cell_id")["patient_id"].to_dict(),
            pt.groupby("patient_id")["disease"].first())


def main():
    rows = []
    for variant in ["easy", "medium", "hard"]:
        data_dir = DATA_ROOT / f"sim_{variant}"
        gt = json.load(open(data_dir / "ground_truth.json"))
        up_syms = set(gt["all_up_genes"]); dn_syms = set(gt["all_down_genes"])
        module_markers = {m: set(gt["modules"][m]["up_genes"]) | set(gt["modules"][m]["down_genes"])
                          for m in MODULE_ORDER}
        conv = GeneIDConverter(cache_file="/labs/Aguiar/SSPA_BRAY/BRay/gene_id_cache.json")
        up_ens_map, _ = conv.symbols_to_ensembl(sorted(up_syms), species="human")
        dn_ens_map, _ = conv.symbols_to_ensembl(sorted(dn_syms), species="human")
        ens_to_sym = {**{v: k for k, v in up_ens_map.items()},
                      **{v: k for k, v in dn_ens_map.items()}}

        # Use the MIL h5ad (which has synthetic patient_id for easy/medium)
        mil_h5ad = data_dir / "counts_perturbed_mil.h5ad"
        pcell, plab = patient_table(mil_h5ad) if mil_h5ad.exists() else (None, None)

        for label, sub in [("orig", "comparison/drgp_unmasked"),
                           ("mil",  "comparison_mil/drgp_unmasked")]:
            for seed in SEEDS:
                d = RESULTS_ROOT / f"sim_{variant}" / sub / f"seed{seed}"
                # MIL uses patient-pool always; orig only meaningful for hard
                pof, plb = (pcell, plab) if (label == "mil" or variant == "hard") else (None, None)
                cell_auc, pat_auc = drgp_aucs(d, pof, plb)
                gs = drgp_gene_score(d)
                metrics = score_metrics(gs, up_syms, dn_syms, module_markers, ens_to_sym) if len(gs) else {}
                rows.append({"variant": variant, "method": f"drgp_{label}", "seed": seed,
                             "test_auc_cell": cell_auc, "test_auc_patient": pat_auc,
                             **metrics})

    df = pd.DataFrame(rows)
    out_csv = RESULTS_ROOT / "mil_vs_orig_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df)} rows)")

    cols = ["test_auc_cell", "test_auc_patient", "any_auprc", "shared_auprc", "tcell_auprc", "myeloid_auprc"]
    summ = df.groupby(["variant", "method"])[cols].agg(["mean", "std"]).round(3)
    summ.to_csv(RESULTS_ROOT / "mil_vs_orig_summary.csv")
    print(f"Wrote {RESULTS_ROOT / 'mil_vs_orig_summary.csv'}")

    print("\n=== DRGP per-cell (orig) vs DRGP MIL-lite (mil) — mean±std across 5 seeds ===")
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    print(summ.to_string())


if __name__ == "__main__":
    main()
