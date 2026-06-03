#!/usr/bin/env python
"""Aggregate sim_sex_only_v1 method comparison.

For each (method, seed) compute:
  - test_auc           : disease-label (y) prediction AUC on held-out cells
  - prog{l}_auroc      : per-gene method score vs membership in program l, l in {0,1,2}
  - any_auroc          : per-gene score vs union of all program genes (90 genes, all up)
  - any_auprc          : same, AUPRC

All program genes are up-regulated in v1, so no up/down split.

Methods:
  - drgp_unmasked            : gene_v_score from vi_gene_programs.csv.gz
  - baseline_{lr,lrl,lrr}    : LR/L1/L2 coefs (gene_v_score from a regularized LR)
  - baseline_{mflr,mflrl,mflrr}: NMF.components_ @ downstream LR coef
  - schpf                    : gene_scores @ downstream LR coef
  - spectra_unsup / sup_sim  : factor_scores @ downstream LR coef
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

DATA_DIR    = Path("/labs/Aguiar/SSPA_BRAY/scdesign3_covid19_8kcells_10kgenes/perturbed_sex_only_v1")
RESULTS_DIR = Path("/labs/Aguiar/SSPA_BRAY/results/sim_sex_only_v1/comparison")
SEEDS = [42, 123, 456, 789, 1024]
K = 10
LABEL = "y"

# ---------- ground truth ----------
gt = np.load(DATA_DIR / "ground_truth.npz", allow_pickle=True)
gene_arr = gt["gene_names"]
sizes  = gt["program_gene_idx_sizes"]
splits = np.split(gt["program_gene_idx_concat"], np.cumsum(sizes)[:-1])
program_syms = [{str(gene_arr[i]) for i in idx} for idx in splits]
all_program_syms = set().union(*program_syms)
N_PROG = len(program_syms)
print(f"Loaded ground truth: {N_PROG} programs, "
      f"sizes {[len(s) for s in program_syms]}, union={len(all_program_syms)} genes")

# Map symbols -> ENSG for baselines indexed in ENSG space.
conv = GeneIDConverter(cache_file="/labs/Aguiar/SSPA_BRAY/BRay/gene_id_cache.json")
sym_to_ens, _ = conv.symbols_to_ensembl(list(all_program_syms), species="human")
ens_to_sym = {v: k for k, v in sym_to_ens.items()}


def auroc_marker(score: pd.Series, pos: set) -> float:
    y = pd.Series(0, index=score.index)
    common = score.index.intersection(pos)
    if not len(common):
        return float("nan")
    y.loc[common] = 1
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return roc_auc_score(y, score)


def auprc_marker(score: pd.Series, pos: set) -> float:
    y = pd.Series(0, index=score.index)
    common = score.index.intersection(pos)
    if not len(common):
        return float("nan")
    y.loc[common] = 1
    if y.sum() == 0:
        return float("nan")
    p, r, _ = precision_recall_curve(y, score)
    return float(auc(r, p))


def to_symbol_index(score: pd.Series) -> pd.Series:
    """If the index looks like ENSG, swap marker ENSG entries to symbols (others kept as-is)."""
    if not score.index.astype(str).str.startswith("ENSG").any():
        return score
    new_idx = [ens_to_sym.get(g, g) for g in score.index]
    out = pd.Series(score.values, index=new_idx)
    return out[~out.index.duplicated(keep="first")]


def score_metrics(score: pd.Series) -> dict:
    s = to_symbol_index(score).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    out = {"any_auroc": auroc_marker(s, all_program_syms),
           "any_auprc": auprc_marker(s, all_program_syms)}
    for l in range(N_PROG):
        out[f"prog{l}_auroc"] = auroc_marker(s, program_syms[l])
    return out


# ---------- per-method gene-score extractors ----------
def drgp_gene_score(seed_dir: Path) -> pd.Series:
    f = seed_dir / "vi_gene_programs.csv.gz"
    if not f.exists():
        return pd.Series(dtype=float)
    gp = pd.read_csv(f, index_col=0)
    v_col = next((c for c in gp.columns if c.startswith("v_weight_")), None)
    if v_col is None:
        return pd.Series(dtype=float)
    v = gp[v_col].values
    B = gp.drop(columns=v_col).T  # genes x K
    col = B.sum(axis=0).replace(0, np.nan)
    TF = B.div(col, axis=1).fillna(0.0)
    score = (TF.values * v).sum(axis=1)
    return pd.Series(score, index=B.index, name="drgp")


def drgp_test_auc(seed_dir: Path) -> float:
    f = seed_dir / "vi_test_predictions.csv.gz"
    if not f.exists():
        return float("nan")
    df = pd.read_csv(f)
    truth_col = next(c for c in df.columns if c.startswith("true_") or c == "label")
    prob_col  = next(c for c in df.columns if c.startswith("prob_"))
    return roc_auc_score(df[truth_col], df[prob_col])


def baseline_gene_score(seed_dir: Path, alg: str, gene_list_ens: list) -> pd.Series:
    model_pkl = seed_dir / f"{alg}_model.pkl"
    if not model_pkl.exists():
        return pd.Series(dtype=float)
    model = joblib.load(model_pkl)
    lr = model.named_steps["logisticregression"]
    coef = lr.coef_[0]
    n = len(gene_list_ens)
    if alg.startswith("mf"):
        nmf_pkl = seed_dir / f"{alg}_nmf.pkl"
        if not nmf_pkl.exists():
            return pd.Series(dtype=float)
        nmf = joblib.load(nmf_pkl)
        Kc = nmf.components_.shape[0]
        gene_attr = coef[:Kc] @ nmf.components_
    else:
        gene_attr = coef[:n]
    return pd.Series(gene_attr[:n], index=gene_list_ens, name=alg)


def baseline_test_auc(seed_dir: Path, alg: str) -> float:
    r = seed_dir / f"{alg}_results.pkl"
    if not r.exists():
        return float("nan")
    return float(joblib.load(r).get("test_roc_auc", float("nan")))


def _find_lr_model(seed_dir: Path, prefix: str) -> Path | None:
    p = seed_dir / "baselines" / "disease" / f"{prefix}_lr_model.pkl"
    if p.exists():
        return p
    cand = list((seed_dir / "baselines").rglob(f"{prefix}_lr_model.pkl"))
    return cand[0] if cand else None


def _summary_test_auc(seed_dir: Path, summary_name: str, result_key: str) -> float:
    f = seed_dir / "baselines" / summary_name
    if not f.exists():
        return float("nan")
    s = json.load(open(f))
    results = s.get("results", s)
    entry = results.get(result_key)
    if not isinstance(entry, dict):
        return float("nan")
    test = entry.get("test", {})
    return float(test.get("roc_auc", entry.get("test_roc_auc", float("nan"))))


def schpf_gene_score(seed_dir: Path, gene_list_sym: list) -> pd.Series:
    gs_path = seed_dir / "model" / "gene_scores.npy"
    if not gs_path.exists():
        return pd.Series(dtype=float)
    gs = np.load(gs_path)
    lr_pkl = _find_lr_model(seed_dir, "schpf")
    if lr_pkl is None:
        return pd.Series(dtype=float)
    coef = joblib.load(lr_pkl).named_steps["logisticregression"].coef_[0][:gs.shape[1]]
    gene_attr = gs @ coef
    n = min(len(gene_list_sym), gene_attr.shape[0])
    return pd.Series(gene_attr[:n], index=gene_list_sym[:n], name="schpf")


def schpf_test_auc(seed_dir: Path) -> float:
    return _summary_test_auc(seed_dir, "schpf_baselines_summary.json", f"schpf_lr_{LABEL}")


def spectra_gene_score(seed_dir: Path, gene_list_sym: list) -> pd.Series:
    fs = seed_dir / "model" / "spectra_factors.npy"
    if not fs.exists():
        return pd.Series(dtype=float)
    factor_scores = np.load(fs)
    lr_pkl = _find_lr_model(seed_dir, "spectra")
    if lr_pkl is None:
        return pd.Series(dtype=float)
    coef = joblib.load(lr_pkl).named_steps["logisticregression"].coef_[0]
    Fmat = factor_scores.T if factor_scores.shape[0] == len(gene_list_sym) else factor_scores
    gene_attr = coef[:Fmat.shape[0]] @ Fmat
    n = min(len(gene_list_sym), gene_attr.shape[0])
    return pd.Series(gene_attr[:n], index=gene_list_sym[:n], name="spectra")


def spectra_test_auc(seed_dir: Path) -> float:
    return _summary_test_auc(seed_dir, "spectra_baselines_summary.json", f"spectra_lr_{LABEL}")


# ---------- main loop ----------
def _gene_lists():
    """Symbol order from h5ad var_names; ENSG order from DataLoader's preprocessing."""
    import anndata as ad
    a = ad.read_h5ad(DATA_DIR / "perturbed.h5ad")
    gene_list_sym = list(a.var_names)
    from VariationalInference.data_loader import DataLoader
    loader = DataLoader(
        data_path=str(DATA_DIR / "perturbed.h5ad"),
        gene_annotation_path=None,
        cache_dir="/labs/Aguiar/SSPA_BRAY/cache",
        use_cache=True,
        verbose=False,
    )
    data = loader.load_and_preprocess(
        label_column="y",
        aux_columns=[],
        train_ratio=0.7, val_ratio=0.15,
        stratify_by="y",
        min_cells_expressing=0.001,
        layer="raw",
        convert_to_ensembl=True,
        filter_protein_coding=False,
        random_state=42,
        normalize=True,
        normalize_target_sum=0.0,
        return_sparse=False,
    )
    return gene_list_sym, list(data["gene_list"])


def main():
    gene_sym, gene_ens = _gene_lists()
    print(f"sym genes: {len(gene_sym)}   ens genes: {len(gene_ens)}")

    rows = []
    BASE_ALGS = ["lr", "lrl", "lrr", "mflr", "mflrl", "mflrr"]

    for seed in SEEDS:
        d = RESULTS_DIR / "drgp_unmasked" / f"seed{seed}"
        rows.append({"method": "drgp_unmasked", "seed": seed,
                     "test_auc": drgp_test_auc(d),
                     **score_metrics(drgp_gene_score(d))})

        bd = RESULTS_DIR / "baselines" / f"seed{seed}"
        for alg in BASE_ALGS:
            rows.append({"method": f"baseline_{alg}", "seed": seed,
                         "test_auc": baseline_test_auc(bd, alg),
                         **score_metrics(baseline_gene_score(bd, alg, gene_ens))})

        d = RESULTS_DIR / "schpf" / f"seed{seed}"
        rows.append({"method": "schpf", "seed": seed,
                     "test_auc": schpf_test_auc(d),
                     **score_metrics(schpf_gene_score(d, gene_sym))})

        for name in ["spectra_unsup", "spectra_sup_sim"]:
            d = RESULTS_DIR / name / f"seed{seed}"
            rows.append({"method": name, "seed": seed,
                         "test_auc": spectra_test_auc(d),
                         **score_metrics(spectra_gene_score(d, gene_sym))})

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "comparison_table.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows)")
    print("\n=== Mean +/- Std across seeds ===")
    summary = df.groupby("method").agg(["mean", "std"]).round(3)
    print(summary)
    summary.to_csv(RESULTS_DIR / "comparison_summary.csv")
    print(f"Wrote {RESULTS_DIR / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
