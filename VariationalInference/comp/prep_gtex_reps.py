#!/usr/bin/env python
"""Emit scHPF/Spectra downstream reps for the GTEx WB experimental h5ad.

Produces, with IDENTICAL row (cell) order matching the source h5ad:
  - gtex_wb_counts.mtx      cells x genes integer COO (scHPF `train -i` reads via mmread)
  - gtex_wb_genes.txt       2-col tab (ENSEMBL_ID, GENE_NAME) in mtx column order
  - gtex_wb_meta.csv.gz     index = obs_names; columns = ENSG gene counts + metadata
                            (heart_disease, subject_id + 10 aux). Used by
                            run_schpf_baselines.py / run_spectra_baselines.py
                            (asserts CSV cell count == model cell count) and by
                            run_spectra.py --data.

scHPF `mmread('filtered.mtx')` yields a cells x genes matrix; we write exactly
that orientation so model.cell_score() rows align 1:1 with the CSV rows.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import anndata as ad

META_KEEP = ["heart_disease", "subject_id",
             "sex_female", "race_indian", "race_asian", "race_black",
             "race_missing", "age", "BMI", "smoking", "MHHTN", "MHT2D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="gtex_wb")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[prep] loading {args.h5ad}")
    a = ad.read_h5ad(args.h5ad)
    a.var_names_make_unique()
    X = a.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    # integer counts; round defensively, cast to int
    X = X.tocsr()
    X.data = np.rint(X.data)
    X = X.astype(np.int64)
    n_cells, n_genes = X.shape
    print(f"[prep] {n_cells} cells x {n_genes} genes; nnz={X.nnz}")

    genes = a.var_names.astype(str).to_numpy()

    # 1. mtx (cells x genes, integer) — scHPF train reads via mmread
    mtx_path = out / f"{args.prefix}_counts.mtx"
    print(f"[prep] writing {mtx_path}")
    sio.mmwrite(str(mtx_path), X.tocoo(), field="integer")

    # 2. genes.txt (ENSEMBL_ID \t GENE_NAME) in mtx column order
    genes_path = out / f"{args.prefix}_genes.txt"
    pd.DataFrame({"ensembl": genes, "name": genes}).to_csv(
        genes_path, sep="\t", header=False, index=False)
    print(f"[prep] writing {genes_path}")

    # 3. CSV: gene columns (ENSG) + metadata, index = obs_names, SAME row order
    meta_cols = [c for c in META_KEEP if c in a.obs.columns]
    missing = [c for c in META_KEEP if c not in a.obs.columns]
    if missing:
        print(f"[prep] WARNING obs missing: {missing}")
    dense = X.toarray()
    df = pd.DataFrame(dense, index=a.obs_names.astype(str), columns=genes)
    for c in meta_cols:
        df[c] = a.obs[c].to_numpy()
    csv_path = out / f"{args.prefix}_meta.csv.gz"
    print(f"[prep] writing {csv_path} (shape {df.shape})")
    df.to_csv(csv_path, compression="gzip")

    # sanity
    assert len(df) == n_cells, "CSV row count must equal cell count"
    print(f"[prep] DONE. cells={n_cells} genes={n_genes} meta={meta_cols}")
    print(f"[prep]   mtx:   {mtx_path}")
    print(f"[prep]   genes: {genes_path}")
    print(f"[prep]   csv:   {csv_path}")


if __name__ == "__main__":
    main()
