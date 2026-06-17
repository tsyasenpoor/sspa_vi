"""Build per-patient pseudo-bulk h5ads from the single-cell COVID atlas.

The source X is log-normalized (not raw counts), so pseudo-bulk = per-patient MEAN of the
log-normalized profile -- the same value scale the cell-level CAVI already consumes, keeping
single vs bulk consistent. Patient subsets match the cell-level runs exactly via
subsample_adata(n_patients=N, subsample_seed=0). One row per patient; labels/aux are the
patient-level values (constant within patient under inherited labels).

Output: data/covid_pseudobulk/bulk_covid_{N}p_seed0.h5ad for N in {50, 100, 148}.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, "/labs/Aguiar/SSPA_BRAY/BRay")
from VariationalInference.create_subsamples import subsample_adata

SRC = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19/covid19_filtered_fullgenes_clean.h5ad"
OUT = Path("/labs/Aguiar/SSPA_BRAY/data/covid_pseudobulk")
OUT.mkdir(parents=True, exist_ok=True)
N_VALUES = [50, 100, 148]
PATIENT_COL = "sampleID"
PATIENT_OBS = ["CoVID-19 severity", "Outcome", "Sex", "Age",
               "cm_diabetes", "cm_cardio", "cm_asthma_copd", "majorType"]

print("loading full single-cell h5ad ...", flush=True)
full = ad.read_h5ad(SRC)
full.var_names_make_unique()
d = full.X[:5000]
d = d.data if sp.issparse(d) else np.asarray(d).ravel()
print(f"source X: dtype={d.dtype} max={d.max():.3f} all_integer={bool(np.allclose(d, np.round(d)))} "
      f"-> {'RAW COUNTS' if np.allclose(d, np.round(d)) else 'NORMALIZED (mean-aggregating)'}", flush=True)

for N in N_VALUES:
    sub = subsample_adata(full, n_patients=N, subsample_seed=0, verbose=False)
    sids = sub.obs[PATIENT_COL].astype(str).to_numpy()
    uniq = pd.unique(sids)                                   # first-appearance order
    P, ncell = len(uniq), sub.n_obs
    ridx = pd.Series(np.arange(P), index=uniq).loc[sids].to_numpy()
    D = sp.csr_matrix((np.ones(ncell, np.float32), (ridx, np.arange(ncell))), shape=(P, ncell))
    cells_per = np.asarray(D.sum(1)).ravel()
    Dn = sp.diags(1.0 / cells_per) @ D                       # row-normalized -> mean
    X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    bulk = (Dn @ X).toarray().astype(np.float32)             # (P, genes) dense mean profile

    first = sub.obs.groupby(PATIENT_COL, sort=False).first().reindex(uniq)
    obs = pd.DataFrame({c: first[c].to_numpy() for c in PATIENT_OBS if c in first.columns})
    obs[PATIENT_COL] = uniq
    obs["n_cells"] = cells_per.astype(int)
    obs.index = pd.Index(uniq, name=PATIENT_COL)

    B = ad.AnnData(X=sp.csr_matrix(bulk), obs=obs, var=sub.var.copy())
    out = OUT / f"bulk_covid_{N}p_seed0.h5ad"
    B.write_h5ad(out, compression="gzip")
    sev = obs["CoVID-19 severity"].value_counts().to_dict()
    print(f"N={N}: {P} patients x {B.n_vars} genes -> {out.name}  "
          f"cells/patient {cells_per.min():.0f}-{cells_per.max():.0f}  severity={sev}", flush=True)

print("DONE", flush=True)
