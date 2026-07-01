#!/usr/bin/env python
"""Draw ground-truth gene programs for the bulk DRGP simulation, aligned to a SPsimSeq X0.

Produces theta* (sample x K carrier-activity design) and beta* (gene x K log2-fold-change
loadings) -- the `design_fixed`/`coef_fixed` matrices seqgendiff thin_diff consumes -- plus the
regression weights upsilon* (nonzero only for disease-relevant programs).

v2 composition (2026-06-24): programs are split into DE NOVO (hidden) and PATHWAY (annotated,
exposed to masked/combined via a binary mask M + an Ensembl GMT). This lets the bulk arm test
masked/combined modes, which the all-de-novo v1 design could not. Disease-relevant programs carry
BALANCED +/- label weights (protective vs risk). Default = the locked 8-program design:
  4 de novo disease (2 protective upsilon<0, 2 risk upsilon>0)
  2 pathway disease (1 protective, 1 risk)        <- annotated
  1 de novo nuisance (upsilon=0)
  1 pathway nuisance (upsilon=0)                   <- annotated
=> K*=8 ; 6 disease-relevant + 2 nuisance ; 3 annotated (mask M gene x3) + 5 de novo.

Carrier gene sets are DISJOINT across programs (drawn from one permutation) so annotated and de
novo programs never share genes -- keeps the mask and per-program recovery unambiguous.

Run:
    python make_truth.py --x0 .../X0_null.tsv.gz --out-dir .../truth0 \
        --prog-size 100 --effect 3.0 --carrier-prob 0.5 --upsilon 1.5 --truth-seed 0
"""
import argparse, hashlib, os
import numpy as np
import pandas as pd


def rng_for(seed_str):
    h = hashlib.blake2b(seed_str.encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x0", required=True, help="SPsimSeq null background tsv.gz (gene x samples)")
    ap.add_argument("--out-dir", required=True)
    # composition (defaults = locked 8-program v2 design)
    ap.add_argument("--n-denovo-disease", type=int, default=4, help="de novo disease-relevant programs")
    ap.add_argument("--n-pathway-disease", type=int, default=2, help="annotated disease-relevant programs")
    ap.add_argument("--n-denovo-nuisance", type=int, default=1, help="de novo nuisance programs")
    ap.add_argument("--n-pathway-nuisance", type=int, default=1, help="annotated nuisance programs")
    ap.add_argument("--prog-size", type=int, default=100, help="genes per program (disjoint carriers)")
    ap.add_argument("--effect", type=float, default=3.0, help="mean per-gene log2 fold-change")
    ap.add_argument("--carrier-prob", type=float, default=0.5, help="P(sample carries a program)")
    ap.add_argument("--upsilon", type=float, default=1.5, help="|label weight| for disease programs")
    ap.add_argument("--truth-seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = rng_for(f"truth{args.truth_seed}_v2_dd{args.n_denovo_disease}_pd{args.n_pathway_disease}"
                  f"_dn{args.n_denovo_nuisance}_pn{args.n_pathway_nuisance}")

    # gene + sample axes from X0
    head = pd.read_csv(args.x0, sep="\t", nrows=0)
    sample_ids = list(head.columns[1:])
    gene_ids = pd.read_csv(args.x0, sep="\t", usecols=[0]).iloc[:, 0].to_numpy()
    G, N = len(gene_ids), len(sample_ids)

    # program layout: [denovo_disease, pathway_disease, denovo_nuisance, pathway_nuisance]
    n_dd, n_pd, n_dn, n_pn = (args.n_denovo_disease, args.n_pathway_disease,
                              args.n_denovo_nuisance, args.n_pathway_nuisance)
    K = n_dd + n_pd + n_dn + n_pn
    ptype = (["denovo_disease"] * n_dd + ["pathway_disease"] * n_pd
             + ["denovo_nuisance"] * n_dn + ["pathway_nuisance"] * n_pn)
    annotated = np.array([t.startswith("pathway") for t in ptype], dtype=bool)
    disease_relevant = np.array([t.endswith("disease") for t in ptype], dtype=bool)
    if K * args.prog_size > G:
        raise ValueError(f"K*prog_size={K*args.prog_size} > G={G}; carriers cannot be disjoint")
    print(f"X0: {G} genes x {N} samples; K={K} programs "
          f"({n_dd} denovo-disease + {n_pd} pathway-disease + {n_dn} denovo-nuisance "
          f"+ {n_pn} pathway-nuisance); {annotated.sum()} annotated", flush=True)

    # disjoint carrier sets from a single permutation
    pool = rng.permutation(G)[:K * args.prog_size].reshape(K, args.prog_size)
    members = [np.sort(pool[k]) for k in range(K)]

    beta = np.zeros((G, K), dtype=np.float64)   # gene x K  log2FC loadings (coef_fixed)
    theta = np.zeros((N, K), dtype=np.float64)  # sample x K carrier activity (design_fixed)
    for k in range(K):
        beta[members[k], k] = args.effect * rng.uniform(0.5, 1.5, size=args.prog_size)
        theta[:, k] = (rng.random(N) < args.carrier_prob).astype(float)

    # BALANCED +/- upsilon within each disease group (protective vs risk); nuisance = 0.
    upsilon = np.zeros(K)

    def balanced_signs(n):
        s = np.array([-1.0] * (n // 2) + [1.0] * (n - n // 2))
        return s  # deterministic: first half protective, second half risk

    dd_idx = np.arange(0, n_dd)
    pd_idx = np.arange(n_dd, n_dd + n_pd)
    upsilon[dd_idx] = balanced_signs(n_dd) * args.upsilon
    upsilon[pd_idx] = balanced_signs(n_pd) * args.upsilon

    # binary mask over annotated programs (gene x n_annotated), column order = annotated program order
    ann_idx = np.where(annotated)[0]
    mask_M = np.zeros((G, len(ann_idx)), dtype=np.uint8)
    for j, k in enumerate(ann_idx):
        mask_M[members[k], j] = 1

    np.savez_compressed(
        f"{args.out_dir}/truth.npz",
        beta=beta, theta=theta, upsilon=upsilon, disease_relevant=disease_relevant,
        annotated=annotated, annotated_idx=ann_idx, mask_M=mask_M,
        program_type=np.array(ptype), members=np.array(members, dtype=object),
        gene_ids=gene_ids, sample_ids=np.array(sample_ids),
    )
    # design/coef TSVs for the R injection step (gene/sample labels matching X0 order)
    cols = [f"prog{k}" for k in range(K)]
    pd.DataFrame(beta, index=gene_ids, columns=cols).to_csv(
        f"{args.out_dir}/beta_coef.tsv.gz", sep="\t", compression="gzip")
    pd.DataFrame(theta, index=sample_ids, columns=cols).to_csv(
        f"{args.out_dir}/theta_design.tsv.gz", sep="\t", compression="gzip")

    # Ensembl GMT for the annotated programs -> masked/combined via quick_reference
    # --pathway-file (pass --pathway-genes-ensembl so IDs are used directly, no symbol conversion).
    with open(f"{args.out_dir}/pathways.gmt", "w") as f:
        for k in ann_idx:
            genes = "\t".join(str(g) for g in gene_ids[members[k]])
            f.write(f"prog{k}\t{ptype[k]}\t{genes}\n")

    print(f"program_type: {ptype}", flush=True)
    print(f"upsilon: {np.round(upsilon, 2)}", flush=True)
    print(f"annotated programs (mask cols): {list(ann_idx)}  -> pathways.gmt ({len(ann_idx)} sets)", flush=True)
    print(f"DONE -> {args.out_dir}/{{truth.npz, beta_coef.tsv.gz, theta_design.tsv.gz, pathways.gmt}}", flush=True)


if __name__ == "__main__":
    main()
