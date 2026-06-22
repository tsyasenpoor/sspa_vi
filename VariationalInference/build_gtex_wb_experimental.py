"""Build the experimental GTEx Whole-Blood real-count dataset for the DRGP bulk arm.

Real-data (NOT simulation) arm: run DRGP on real GTEx WB read counts with the real
heart_disease label + clinical aux, analogous to the COVID experimental arm. Uses true
v8 read counts (Poisson-valid), NOT the on-disk rounded-TPM matrix.

Pipeline:
  1. Whole-Blood RNASEQ samples from the sample-attributes file (SAMPID col2, SMTSD col15,
     SMAFRZE col29) -> 755 WB samples (1 per subject; WB is a single tissue).
  2. Keep WB samples that have a heart_disease label + aux in the preprocessed responses/aux
     (subject-level phenotypes) -> 708 samples.
  3. Extract those WB columns from gene_reads.gct, restricted to the prior 28,587-gene
     protein-coding set (exact versioned-Ensembl subset of the gct's 56,200) for consistency
     with earlier GTEx preprocessing.
  4. QC: drop genes expressed (count>0) in < MIN_FRAC of samples.
  5. Write an h5ad with int32 counts, obs = heart_disease + subject_id + clinical aux.

Output: data/gtex_wb_experimental/gtex_wb_counts.h5ad
"""
from __future__ import annotations
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

ROOT = Path("/labs/Aguiar/SSPA_BRAY/dataset/GTEX")
SA = ROOT / "files/metadata/phs000424.v8.pht002743.v8.p2.c1.GTEx_Sample_Attributes.GRU.txt.gz"
GCT = ROOT / "preprocessed/real_counts/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz"
RESP = ROOT / "preprocessed/responses.csv.gz"
AUX = ROOT / "preprocessed/aux_data.csv.gz"
PP_EXPR = ROOT / "preprocessed/gene_expression.csv.gz"   # prior gene set (header only is read)
OUT = Path("/labs/Aguiar/SSPA_BRAY/data/gtex_wb_experimental")
OUT.mkdir(parents=True, exist_ok=True)

LABEL = "heart_disease"
AUX_COLS = ["sex_female", "race_indian", "race_asian", "race_black", "race_missing",
            "age", "BMI", "smoking", "MHHTN", "MHT2D"]
MIN_FRAC = 0.10   # drop genes expressed in < 10% of WB samples
SUBJECT = lambda s: "-".join(s.split("-")[:2])


def wb_sampids() -> list[str]:
    """SAMPID (col2) where SMTSD (col15)=='Whole Blood' and SMAFRZE (col29)=='RNASEQ'."""
    out = []
    with gzip.open(SA, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > 28 and p[1].startswith("GTEX") and p[14] == "Whole Blood" and p[28] == "RNASEQ":
                out.append(p[1])
    return sorted(set(out))


def main():
    print("[1] Whole-Blood RNASEQ samples ...", flush=True)
    wb = wb_sampids()
    print(f"    WB RNASEQ samples: {len(wb)}", flush=True)

    print("[2] labels + aux (subject-level phenotypes, keyed by Sample_ID) ...", flush=True)
    resp = pd.read_csv(RESP, index_col=0)[LABEL]
    aux = pd.read_csv(AUX, index_col=0)[AUX_COLS]
    keep = [s for s in wb if s in resp.index and s in aux.index]
    print(f"    WB samples with label+aux: {len(keep)}", flush=True)
    assert len(keep) > 0

    print("[3] prior gene set ...", flush=True)
    prior_genes = pd.read_csv(PP_EXPR, index_col=0, nrows=0).columns.tolist()
    prior_set = set(prior_genes)
    print(f"    prior genes: {len(prior_genes)}", flush=True)

    print("[3b] reading gct header to locate WB columns ...", flush=True)
    with gzip.open(GCT, "rt") as f:
        f.readline(); f.readline()                 # '#1.2' ; '<ngenes>\t<nsamples>'
        header = f.readline().rstrip("\n").split("\t")
    pos = {name: i for i, name in enumerate(header)}
    keep = [s for s in keep if s in pos]           # must be a gct column
    print(f"    WB samples present in gct: {len(keep)}", flush=True)
    usecols = [0] + [pos[s] for s in keep]         # col0 = 'Name'

    print("[3c] reading gct counts (usecols) — this takes a few minutes ...", flush=True)
    df = pd.read_csv(GCT, sep="\t", skiprows=2, usecols=usecols)
    df = df.set_index("Name")
    df = df.loc[df.index.isin(prior_set)]          # restrict to prior protein-coding set
    df = df[keep]                                  # order columns = sample order
    print(f"    counts: {df.shape[0]} genes x {df.shape[1]} samples", flush=True)

    print("[4] gene QC (min expressed fraction) ...", flush=True)
    counts = df.to_numpy(dtype=np.int32).T          # samples x genes
    expr_frac = (counts > 0).mean(axis=0)
    gmask = expr_frac >= MIN_FRAC
    counts = counts[:, gmask]
    genes = df.index.to_numpy()[gmask]
    print(f"    kept {gmask.sum()} / {gmask.size} genes (>= {MIN_FRAC:.0%} expressed)", flush=True)

    # Strip Ensembl version suffixes so masked-mode pathway matching works: load_pathways
    # converts GMT symbols -> UNVERSIONED Ensembl, which won't match versioned var (0 overlap).
    base = pd.Index([g.split(".")[0] for g in genes])
    if base.duplicated().any():
        keep = ~base.duplicated().to_numpy()      # drop rare PAR_Y collisions
        counts, genes = counts[:, keep], genes[keep]
        base = pd.Index([g.split(".")[0] for g in genes])
    genes = base.to_numpy()

    print("[5] assemble h5ad ...", flush=True)
    obs = pd.DataFrame(index=pd.Index(keep, name="Sample_ID"))
    obs[LABEL] = resp.loc[keep].astype(int).to_numpy()
    obs["subject_id"] = [SUBJECT(s) for s in keep]
    for c in AUX_COLS:
        obs[c] = aux.loc[keep, c].to_numpy()
    # GTEx comorbidity flags carry uncleaned sentinels (98=not reported, 99=unknown).
    # Recode to 0 ("no comorbidity reported"); affects very few samples. Without this the
    # logistic head would see x_aux=99 and the gamma term would blow up.
    for c in ("MHHTN", "MHT2D"):
        n_sent = int(obs[c].isin([98, 99]).sum())
        if n_sent:
            print(f"    recoding {n_sent} sentinel (98/99) values in {c} -> 0", flush=True)
            obs.loc[obs[c].isin([98, 99]), c] = 0.0
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    A = ad.AnnData(X=sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var)

    out = OUT / "gtex_wb_counts.h5ad"
    A.write_h5ad(out, compression="gzip")
    bal = obs[LABEL].value_counts().to_dict()
    print(f"DONE -> {out}", flush=True)
    print(f"    shape: {A.n_obs} samples x {A.n_vars} genes ; {LABEL} balance: {bal}", flush=True)
    print(f"    unique subjects: {obs['subject_id'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
