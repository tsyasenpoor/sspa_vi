#!/usr/bin/env python
"""One-time prep for sim_sex_only_v1 method comparison.

Generates from perturbed.h5ad + ground_truth.npz:
  - perturbed.csv.gz                (metadata cols + gene cols, one row per cell)
  - schpf_input/filtered.mtx        (cells x genes, MTX integer)
  - schpf_input/genes.txt           (symbol\\tsymbol)
  - schpf_input/metadata_schpf.csv
  - sim_sex_only_v1_markers.gmt     (one row per program, gene symbols)

Layout mirrors /scdesign3_IBD_10kcells_2kgenes/perturbed_disease_only_no_aux/.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path("/labs/Aguiar/SSPA_BRAY/scdesign3_covid19_8kcells_10kgenes/perturbed_sex_only_v1")
H5AD = ROOT / "perturbed.h5ad"
CSV  = ROOT / "perturbed.csv.gz"
GT   = ROOT / "ground_truth.npz"
GMT  = ROOT / "sim_sex_only_v1_markers.gmt"
MTX_DIR = ROOT / "schpf_input"
MTX_DIR.mkdir(exist_ok=True, parents=True)

a = ad.read_h5ad(H5AD)
print(f"Loaded {a.shape} from {H5AD}")
X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
X = X.astype(np.int32)

genes = list(a.var_names)
meta_cols = list(a.obs.columns)

# --- CSV: cell index, gene cols..., metadata cols... ---
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

with open(MTX_DIR / "genes.txt", "w") as fh:
    for g in genes:
        fh.write(f"{g}\t{g}\n")
print(f"Wrote {MTX_DIR / 'genes.txt'}  ({len(genes)} genes)")

a.obs.to_csv(MTX_DIR / "metadata_schpf.csv")
print(f"Wrote {MTX_DIR / 'metadata_schpf.csv'}  ({a.obs.shape})")

# --- GMT: one gene set per disease program, prefix SIM_SEX_V1 for Spectra-sup ---
gt = np.load(GT, allow_pickle=True)
sizes  = gt["program_gene_idx_sizes"]
concat = gt["program_gene_idx_concat"]
gene_arr = gt["gene_names"]
splits = np.split(concat, np.cumsum(sizes)[:-1])
with open(GMT, "w") as fh:
    for l, idx in enumerate(splits):
        gset = [str(gene_arr[i]) for i in idx]
        fh.write(f"SIM_SEX_V1_program_{l}\tSIM_SEX_V1\t" + "\t".join(gset) + "\n")
print(f"Wrote {GMT}  ({len(splits)} programs, "
      f"{[len(s) for s in splits]} genes each)")

print("Done.")
