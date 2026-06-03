#!/usr/bin/env python3
"""
GSVA (Gene Set Variation Analysis) on monocytes_GEX_20251201.h5ad
using gseapy + REACTOME pathways from sspa.

Identical to gsva_pbmc.py except DATA_PATH / OUT_DIR. The PBMC-wide cell-type
exclusion (Platelets, RBCs) is irrelevant here since the input is already
restricted to monocytes; keeping it is a no-op.
"""

import time
import datetime
import scanpy as sc
import numpy as np
import pandas as pd
import sspa
import gseapy

if not hasattr(pd.DataFrame, 'iteritems'):
    pd.DataFrame.iteritems = pd.DataFrame.items

DATA_PATH = "/labs/Aguiar/SSPA_BRAY/dataset/biorepository/monocytes_GEX_20251201.h5ad"
GMT_PATH  = "/archive/projects/SSPA_BRAY/sspa/c2.cp.v2024.1.Hs.symbols.gmt"
OUT_DIR   = "/labs/Aguiar/SSPA_BRAY/results/gsva_monocytes"
N_THREADS = 31

print(f"[{datetime.datetime.now()}] Loading data ...")
adata = sc.read_h5ad(DATA_PATH)
print(f"  Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")

excl = ['Platelets', 'RBCs']
if 'celltype_subclust' in adata.obs.columns:
    adata = adata[~adata.obs['celltype_subclust'].isin(excl), :]
    print(f"  After cell-type exclusion: {adata.shape[0]} cells")

n_filt = round(adata.shape[0] / 100)
sc.pp.filter_genes(adata, min_cells=n_filt, inplace=True)
print(f"  Gene filter (min_cells={n_filt}): {adata.shape[1]} genes retained")

mat = adata.to_df()

print(f"[{datetime.datetime.now()}] Loading REACTOME pathways ...")
paths = sspa.process_gmt(GMT_PATH)
paths = paths.filter(regex='REACTOME', axis=0)
print(f"  {len(paths)} REACTOME pathways loaded")

pathways = sspa.utils.pathwaydf_to_dict(paths)
genes_present = set(mat.columns.tolist())
pathways = {
    k: v for k, v in pathways.items()
    if sum(1 for g in v if g in genes_present) >= 2
}
print(f"  {len(pathways)} pathways with ≥2 present genes")

print(f"[{datetime.datetime.now()}] Running GSVA on {mat.shape[0]} cells ...")
t0 = time.time()

gsva_res = gseapy.gsva(
    mat.T,
    gene_sets=pathways,
    min_size=2,
    max_size=2000,
    threads=N_THREADS,
    outdir=None,
    verbose=True,
)

elapsed = (time.time() - t0) / 60
print(f"[{datetime.datetime.now()}] GSVA done in {elapsed:.1f} min")

td = datetime.date.today().strftime('%Y-%m-%d')
import os
os.makedirs(OUT_DIR, exist_ok=True)
out_path = f"{OUT_DIR}/GSVA_monocytes_{td}.csv"

gsva_scores = gsva_res.res2d.pivot(index='Term', columns='Name', values='ES').T
gsva_scores.to_csv(out_path)
print(f"Saved: {out_path}  (shape {gsva_scores.shape})")
