#!/usr/bin/env python
"""Build the GTEx Whole Blood real-counts reference matrix for the bulk DRGP simulation.

Parses the GTEx v8 gene_reads GCT (true read counts, NOT the on-disk rounded-TPM), subsets to
Whole Blood RNA-seq samples (one per subject), filters to protein-coding genes + a min-expression
QC, and saves a canonical reference (genes x samples) for SPsimSeq fitting + downstream injection.

Run (no args needed):
    conda activate jax_gpu
    PYTHONPATH=/labs/Aguiar/SSPA_BRAY/BRay \
    python -u BRay/VariationalInference/bulk_gtex_sim/build_wb_reference.py --verbose
"""
import argparse, gzip, io, os
import numpy as np
import pandas as pd

DATA = "/labs/Aguiar/SSPA_BRAY/dataset/GTEX"
GCT = f"{DATA}/preprocessed/real_counts/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz"
SAMPLE_ATTR = f"{DATA}/files/metadata/phs000424.v8.pht002743.v8.p2.c1.GTEx_Sample_Attributes.GRU.txt.gz"
WGS_SAMPLES = "/tmp/gtex_wgs_samples.txt"  # bcftools query -l of the WGS VCF (866 subjects)
OUT_DIR = "/labs/Aguiar/SSPA_BRAY/data/Simulations/bulk_gtex_v1"
PC_GENES = f"{OUT_DIR}/protein_coding_genes_v27.txt"


def log(msg, on=True):
    if on:
        print(msg, flush=True)


def wb_rnaseq_sampids(verbose=True):
    """Whole Blood RNA-seq SAMPIDs from the GTEx sample-attributes table."""
    with gzip.open(SAMPLE_ATTR, "rt") as f:
        lines = f.readlines()
    hdr = next(i for i, l in enumerate(lines)
               if l.startswith("dbGaP_Sample_ID") or l.split("\t")[:2] == ["dbGaP_Sample_ID", "SAMPID"])
    attr = pd.read_csv(io.StringIO("".join(lines[hdr:])), sep="\t", dtype=str)
    wb = attr[(attr["SMTSD"] == "Whole Blood") & (attr["SMAFRZE"] == "RNASEQ")]
    sampids = sorted(wb["SAMPID"].unique())
    log(f"  Whole Blood RNA-seq SAMPIDs: {len(sampids)}", verbose)
    return sampids


def subject_of(sampid):
    """GTEX-1117F-0226-SM-5GZZ7 -> GTEX-1117F."""
    p = sampid.split("-")
    return f"{p[0]}-{p[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-frac", type=float, default=0.10,
                    help="keep genes expressed (count>0) in >= this fraction of samples")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    v = args.verbose
    os.makedirs(args.out_dir, exist_ok=True)

    # --- samples ---
    log("[1/5] resolving Whole Blood RNA-seq samples ...", v)
    wb_ids = wb_rnaseq_sampids(v)
    wgs_subjects = set(l.strip() for l in open(WGS_SAMPLES)) if os.path.exists(WGS_SAMPLES) else set()
    log(f"  WGS subjects available: {len(wgs_subjects)}", v)

    # --- gct header -> which columns to read ---
    log("[2/5] reading GCT header ...", v)
    header = pd.read_csv(GCT, sep="\t", skiprows=2, nrows=0)  # skips '#1.2' + dims line
    gct_cols = list(header.columns)  # Name, Description, <SAMPID...>
    wb_in_gct = [s for s in wb_ids if s in set(gct_cols)]
    log(f"  WB SAMPIDs present in GCT: {len(wb_in_gct)} / {len(wb_ids)}", v)
    usecols = ["Name", "Description"] + wb_in_gct

    # --- read counts (WB columns only) ---
    log(f"[3/5] reading GCT counts for {len(wb_in_gct)} WB columns (slow, ~minutes) ...", v)
    df = pd.read_csv(GCT, sep="\t", skiprows=2, usecols=usecols)
    df = df[usecols]  # enforce column order
    gene_ver = df["Name"].to_numpy()             # versioned ENSG (ENSG....N)
    gene_sym = df["Description"].to_numpy()
    counts = df[wb_in_gct].to_numpy(dtype=np.int32)  # genes x samples
    log(f"  raw counts matrix: {counts.shape[0]} genes x {counts.shape[1]} samples", v)

    # --- protein-coding filter ---
    log("[4/5] filtering protein-coding + min-expression QC ...", v)
    gene_base = np.array([g.split(".")[0] for g in gene_ver])
    pc = set(l.strip() for l in open(PC_GENES))
    pc_mask = np.array([g in pc for g in gene_base])
    log(f"  protein-coding genes: {pc_mask.sum()} / {len(gene_base)}", v)
    # min-expression QC (computed on protein-coding subset)
    n_samp = counts.shape[1]
    expr_frac = (counts > 0).mean(axis=1)
    qc_mask = pc_mask & (expr_frac >= args.min_frac)
    log(f"  after min-frac>={args.min_frac} expressed: {qc_mask.sum()} genes", v)

    counts = counts[qc_mask]
    gene_ver = gene_ver[qc_mask]
    gene_base = gene_base[qc_mask]
    gene_sym = gene_sym[qc_mask]

    # --- sample metadata ---
    subjects = np.array([subject_of(s) for s in wb_in_gct])
    has_wgs = np.array([sub in wgs_subjects for sub in subjects])
    log(f"  samples with WGS: {has_wgs.sum()} / {len(has_wgs)}", v)

    # --- save ---
    log("[5/5] saving reference ...", v)
    np.savez_compressed(
        f"{args.out_dir}/wb_reference_counts.npz",
        counts=counts, gene_ver=gene_ver, gene_base=gene_base, gene_sym=gene_sym,
        sample_ids=np.array(wb_in_gct), subjects=subjects, has_wgs=has_wgs,
    )
    pd.DataFrame({"sample_id": wb_in_gct, "subject": subjects, "has_wgs": has_wgs}) \
        .to_csv(f"{args.out_dir}/wb_reference_samples.csv", index=False)
    # genes x samples TSV.gz for SPsimSeq (R reads via data.table::fread)
    out_tsv = f"{args.out_dir}/wb_reference_counts.tsv.gz"
    pd.DataFrame(counts, index=gene_ver, columns=wb_in_gct) \
        .to_csv(out_tsv, sep="\t", compression="gzip")
    log(f"\nDONE. {counts.shape[0]} genes x {counts.shape[1]} samples", v)
    log(f"  {args.out_dir}/wb_reference_counts.npz", v)
    log(f"  {args.out_dir}/wb_reference_counts.tsv.gz  (SPsimSeq input)", v)
    log(f"  {args.out_dir}/wb_reference_samples.csv", v)


if __name__ == "__main__":
    main()
