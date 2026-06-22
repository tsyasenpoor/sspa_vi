#!/usr/bin/env python
"""Draw ground-truth gene programs for the bulk DRGP simulation, aligned to a SPsimSeq X0.

Produces theta* (sample x K carrier-activity design) and beta* (gene x K log2-fold-change
loadings) — the `design_fixed`/`coef_fixed` matrices that seqgendiff thin_diff consumes — plus
the regression weights upsilon* (nonzero only for disease-relevant programs) used by the label
model. K* programs = (disease-relevant) + (nuisance). Deterministic per --truth-seed.

Run:
    python make_truth.py --x0 .../X0_null.tsv.gz --out-dir .../truth0 \
        --k 10 --n-disease 3 --prog-size 100 --effect 1.0 --carrier-prob 0.5 --truth-seed 0
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
    ap.add_argument("--k", type=int, default=10, help="total programs K*")
    ap.add_argument("--n-disease", type=int, default=3, help="# disease-relevant programs")
    ap.add_argument("--prog-size", type=int, default=100, help="genes per program")
    ap.add_argument("--effect", type=float, default=1.0, help="mean per-gene log2 fold-change")
    ap.add_argument("--carrier-prob", type=float, default=0.5, help="P(sample carries a program)")
    ap.add_argument("--upsilon", type=float, default=1.5, help="|label weight| for disease programs")
    ap.add_argument("--truth-seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = rng_for(f"truth{args.truth_seed}_k{args.k}_d{args.n_disease}")

    # gene + sample axes from X0
    head = pd.read_csv(args.x0, sep="\t", nrows=0)
    sample_ids = list(head.columns[1:])
    genes = pd.read_csv(args.x0, sep="\t", usecols=[0])
    gene_ids = genes.iloc[:, 0].to_numpy()
    G, N, K = len(gene_ids), len(sample_ids), args.k
    print(f"X0: {G} genes x {N} samples; drawing K={K} programs "
          f"({args.n_disease} disease-relevant + {K - args.n_disease} nuisance)", flush=True)

    beta = np.zeros((G, K), dtype=np.float64)   # gene x K  log2FC loadings (coef_fixed)
    theta = np.zeros((N, K), dtype=np.float64)  # sample x K carrier activity (design_fixed)
    members = []
    for k in range(K):
        S = rng.choice(G, size=args.prog_size, replace=False)
        # per-gene log2FC with heterogeneity (cf. sim_flat_v1 u ~ U[0.5,1.5])
        beta[S, k] = args.effect * rng.uniform(0.5, 1.5, size=args.prog_size)
        # binary carriers (activity 1) — clean program signal
        theta[:, k] = (rng.random(N) < args.carrier_prob).astype(float)
        members.append(np.sort(S))

    # disease-relevant programs get balanced +/- label weights; nuisance = 0
    upsilon = np.zeros(K)
    dz = rng.choice(K, size=args.n_disease, replace=False)
    signs = rng.choice([-1.0, 1.0], size=args.n_disease)
    upsilon[dz] = signs * args.upsilon
    disease_relevant = np.zeros(K, dtype=bool); disease_relevant[dz] = True

    np.savez_compressed(
        f"{args.out_dir}/truth.npz",
        beta=beta, theta=theta, upsilon=upsilon, disease_relevant=disease_relevant,
        members=np.array(members, dtype=object), gene_ids=gene_ids,
        sample_ids=np.array(sample_ids),
    )
    # TSV for the R injection step (gene/sample labels in col/row order matching X0)
    pd.DataFrame(beta, index=gene_ids, columns=[f"prog{k}" for k in range(K)]) \
        .to_csv(f"{args.out_dir}/beta_coef.tsv.gz", sep="\t", compression="gzip")
    pd.DataFrame(theta, index=sample_ids, columns=[f"prog{k}" for k in range(K)]) \
        .to_csv(f"{args.out_dir}/theta_design.tsv.gz", sep="\t", compression="gzip")
    print(f"disease-relevant programs: {list(dz)}  upsilon={upsilon[dz]}", flush=True)
    print(f"DONE -> {args.out_dir}/{{truth.npz, beta_coef.tsv.gz, theta_design.tsv.gz}}", flush=True)


if __name__ == "__main__":
    main()
