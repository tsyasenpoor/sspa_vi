#!/usr/bin/env python
"""Compute a published PGS-Catalog CAD score per GTEx subject from the WGS VCF (LABEL-ONLY aux).

Default score: PGS000018 (Inouye 2018 metaGRS_CAD, 1.74M variants). The harmonized GRCh38 file
gives GRCh38 coords in hm_chr/hm_pos. We match to the GTEx VCF by chr:pos (plink2
--set-all-var-ids '@:#') and let plink2 align the effect allele. Per-subject score -> z-scored
(CLT => ~N(0,1)); the simulation samples n_sim values from this empirical distribution.

Run (heavy: streams the 5.4GB VCF once):
  python compute_prs.py --pgs-file .../PGS000018_hmPOS_GRCh38.txt.gz --out-dir .../prs
"""
import argparse, os, subprocess
import numpy as np
import pandas as pd

VCF = "/labs/Aguiar/SSPA_BRAY/dataset/GTEX/files/genetic_data/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_866Indiv.vcf.gz"
PLINK2 = "/home/FCAM/tyasenpoor/miniconda3/envs/gtex_geno/bin/plink2"
BCFTOOLS = "/home/FCAM/tyasenpoor/miniconda3/envs/gtex_geno/bin/bcftools"


def build_score_file(pgs_file, out_prefix):
    """PGS harmonized -> (plink2 score file ID=chr{hm_chr}:{hm_pos} A1 W) + (bcftools targets TSV).

    plink2 cannot stream the GTEx VCF directly (malformed-BGZF; needs two passes / seekable), so we
    first subset to PGS positions with bcftools -T (streams, no index needed)."""
    df = pd.read_csv(pgs_file, sep="\t", comment="#", dtype=str)
    need = {"hm_chr", "hm_pos", "effect_allele", "effect_weight"}
    assert need <= set(df.columns), f"missing {need - set(df.columns)}"
    df = df.dropna(subset=["hm_chr", "hm_pos", "effect_allele", "effect_weight"])
    df = df[df["hm_pos"].str.match(r"^\d+$", na=False)]
    chrom = "chr" + df["hm_chr"].astype(str)
    # plink2 --set-all-var-ids '@:#' normalizes CHROM (chr1 -> 1), so weights use bare 'hm_chr:pos'
    score = pd.DataFrame({"ID": df["hm_chr"].astype(str) + ":" + df["hm_pos"].astype(str),
                          "A1": df["effect_allele"].str.upper(),
                          "W": df["effect_weight"].astype(float)}).drop_duplicates("ID")
    score_file = out_prefix + ".weights.tsv"
    score.to_csv(score_file, sep="\t", index=False)
    targets_file = out_prefix + ".targets.tsv"    # bcftools -T : CHROM<TAB>POS (1-based)
    pd.DataFrame({"chrom": chrom, "pos": df["hm_pos"]}).drop_duplicates() \
        .to_csv(targets_file, sep="\t", index=False, header=False)
    return score_file, targets_file, len(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgs-file", required=True, help="PGS-Catalog hmPOS_GRCh38 scoring file")
    ap.add_argument("--vcf", default=VCF)
    ap.add_argument("--keep-subjects", default=None, help="optional subject-ID file (e.g. 670 WB-geno)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, "cad_prs")

    score_file, targets_file, nvar = build_score_file(args.pgs_file, prefix)
    print(f"score file: {nvar} unique chr:pos PGS variants", flush=True)

    # 1. subset the VCF to PGS positions. The on-disk GTEx VCF is TRUNCATED (no BGZF EOF marker),
    #    so bcftools/plink2 error at the final block. zcat tolerates the trailing truncation
    #    (loses only the last block ~ end of the last chromosome); pipe its clean text to bcftools.
    subset = prefix + ".subset.vcf.gz"
    if not os.path.exists(subset) or os.path.getsize(subset) < 1000:
        # awk drops malformed records (the VCF has a truncated line mid-chr1 with 441!=866 sample
        # cols, plus the trailing truncated block) so bcftools doesn't abort; 875 = 9 fixed + 866.
        pipe = (f"zcat {args.vcf} 2>/dev/null | "
                f"awk -F'\\t' 'NF==875 || /^#/' | "
                f"{BCFTOOLS} view -T {targets_file} -Oz -o {subset} /dev/stdin")
        print("+", pipe, flush=True)
        # zcat exits nonzero at the truncation; that's expected — plink2 below reports the count
        subprocess.run(pipe, shell=True, check=False)
    assert os.path.exists(subset) and os.path.getsize(subset) > 1000, "subset VCF not produced"
    print(f"subset VCF: {subset} ({os.path.getsize(subset)//1_000_000} MB)", flush=True)

    # 2. plink2 --score on the small clean subset
    cmd = [PLINK2, "--vcf", subset, "--set-all-var-ids", "@:#",
           "--rm-dup", "exclude-all",
           "--score", score_file, "1", "2", "3", "header", "cols=+scoresums",
           "--out", prefix]
    if args.keep_subjects:
        keep = pd.read_csv(args.keep_subjects, header=None)[0]
        kf = prefix + ".keep"
        pd.DataFrame({"FID": keep, "IID": keep}).to_csv(kf, sep="\t", index=False, header=False)
        cmd += ["--keep", kf]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    ss = pd.read_csv(prefix + ".sscore", sep="\t")
    raw = ss["SCORE1_SUM"].to_numpy()                  # weighted-allele sum (the PRS)
    z = (raw - raw.mean()) / raw.std()
    id_col = "#IID" if "#IID" in ss.columns else "IID"  # plink2 writes '#IID'
    ids_out = ss[id_col].astype(str).to_numpy()
    np.save(f"{args.out_dir}/prs_z.npy", z)
    pd.DataFrame({"subject": ids_out, "prs_z": z}).to_csv(
        f"{args.out_dir}/prs_z.csv", index=False)
    print(f"PRS: {len(z)} subjects scored | matched variants reported in {prefix}.log", flush=True)
    print(f"DONE -> {args.out_dir}/prs_z.npy (+ prs_z.csv)", flush=True)


if __name__ == "__main__":
    main()
