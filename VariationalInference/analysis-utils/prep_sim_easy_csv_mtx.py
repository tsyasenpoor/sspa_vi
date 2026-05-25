#!/usr/bin/env python
"""One-time prep for sim_easy method comparison.

Generates from counts_perturbed.h5ad:
  - counts_perturbed.csv.gz  (metadata cols + gene cols, one row per cell)
  - schpf_input/filtered.mtx (cells x genes, matrix-market integer)
  - schpf_input/genes.txt
  - schpf_input/metadata_schpf.csv

Layout mirrors /scdesign3_PBMC_10kcells_2kgenes/synthetic_programs_pathway/exp0_easy/.
"""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path

ROOT = Path("/labs/Aguiar/SSPA_BRAY/scdesign3_IBD_10kcells_2kgenes/perturbed_disease_only_no_aux")
H5AD = ROOT / "counts_perturbed.h5ad"
CSV  = ROOT / "counts_perturbed.csv.gz"
MTX_DIR = ROOT / "schpf_input"
MTX_DIR.mkdir(exist_ok=True, parents=True)

a = ad.read_h5ad(H5AD)
print(f"Loaded {a.shape} from {H5AD}")
X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
X = X.astype(np.int32)

genes = list(a.var_names)
meta_cols = list(a.obs.columns)

# --- CSV: cell index, genes..., metadata...  (mirrors exp0_easy layout) ---
df = pd.DataFrame(X, index=a.obs_names, columns=genes)
for c in meta_cols:
    df[c] = a.obs[c].values
df.index.name = None
print(f"Writing {CSV}  ({df.shape}, last cols = {meta_cols})")
df.to_csv(CSV, compression="gzip")

# --- scHPF input ---
mtx_path = MTX_DIR / "filtered.mtx"
print(f"Writing {mtx_path}  (cells x genes = {X.shape})")
sio.mmwrite(str(mtx_path), sp.csr_matrix(X), field="integer")

genes_path = MTX_DIR / "genes.txt"
with open(genes_path, "w") as fh:
    for g in genes:
        fh.write(f"{g}\t{g}\n")
print(f"Wrote {genes_path}  ({len(genes)} genes)")

meta_path = MTX_DIR / "metadata_schpf.csv"
a.obs.to_csv(meta_path)
print(f"Wrote {meta_path}  ({a.obs.shape})")

print("Done.")
