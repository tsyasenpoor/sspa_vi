#!/usr/bin/env python
"""Score program recovery of a DRGP (or baseline) fit against the planted bulk-sim ground truth.

Reuses Simulations/evaluate.py recovery metrics (Hungarian alignment + support-AUPRC) on the
bulk-sim truth.npz (instead of the single-cell h5ad). Reports per-program cosine + support-AUPRC
split into disease-relevant vs nuisance programs, plus whether |upsilon_hat| (mu_v) ranks the
disease-relevant matched factors above nuisance.
"""
import argparse, glob, gzip, os, pickle
import numpy as np
import pandas as pd
from VariationalInference.Simulations.evaluate import (
    hungarian_match, recovery_cosine, recovery_support_auprc)


def load_fit(fit_dir):
    """Return (beta_hat p x K, gene_list, mu_v len-K or None) from a DRGP output dir."""
    gp = next((p for p in (os.path.join(fit_dir, "vi_gene_programs.csv.gz"),
                           os.path.join(fit_dir, "gene_programs.csv.gz")) if os.path.exists(p)), None)
    if gp:                                        # K x p with optional v_weight_* cols + gene cols
        df = pd.read_csv(gp, index_col=0)
        vcols = [c for c in df.columns if c.startswith("v_weight_")]
        mu_v = df[vcols[0]].to_numpy() if vcols else None
        genes = [c for c in df.columns if c not in vcols]
        beta_hat = df[genes].to_numpy().T        # p x K
        return beta_hat, list(genes), mu_v
    # fallback: vi_model_params.npz (+ need a gene list from the dataset)
    npz = glob.glob(os.path.join(fit_dir, "*model_params.npz")) + \
          glob.glob(os.path.join(fit_dir, "*.npz"))
    if npz:
        d = np.load(npz[0], allow_pickle=True)
        beta_hat = np.asarray(d["E_beta"])       # p x K
        mu_v = np.asarray(d["mu_v"]).ravel() if "mu_v" in d else None
        return beta_hat, None, mu_v
    raise FileNotFoundError(f"no recognizable fit output in {fit_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--truth", required=True, help="truth.npz from make_truth")
    args = ap.parse_args()

    beta_hat, gene_list, mu_v = load_fit(args.fit_dir)
    t = np.load(args.truth, allow_pickle=True)
    beta_star = t["beta"]                          # p x K_truth (truth gene order)
    members = [np.asarray(m, dtype=int) for m in t["members"]]  # gene-index arrays (truth order)
    dz = t["disease_relevant"]                     # K_truth bool
    truth_genes = list(t["gene_ids"])

    # align truth genes -> fit gene order (if fit exposes its gene list)
    if gene_list is not None and beta_hat.shape[0] == len(gene_list):
        pos = {g: i for i, g in enumerate(gene_list)}
        idx = np.array([pos[g] for g in truth_genes])  # fit-row index for each truth gene
        beta_hat = beta_hat[idx]                        # reorder fit rows to truth gene order
    assert beta_hat.shape[0] == beta_star.shape[0], \
        f"gene mismatch: fit {beta_hat.shape[0]} vs truth {beta_star.shape[0]}"

    assign, _ = hungarian_match(beta_hat, beta_star)
    cos = recovery_cosine(beta_hat, beta_star, assign)
    auprc = np.array(recovery_support_auprc(beta_hat, members, assign))
    p = beta_hat.shape[0]
    chance = np.array([len(s) / p for s in members])

    print(f"K_fit={beta_hat.shape[1]}  K_truth={beta_star.shape[1]}  genes={p}")
    print(f"{'prog':>4} {'type':>8} {'cosine':>7} {'AUPRC':>7} {'chance':>7}")
    for l in range(beta_star.shape[1]):
        print(f"{l:>4} {'DISEASE' if dz[l] else 'nuisance':>8} "
              f"{cos[l]:>7.3f} {auprc[l]:>7.3f} {chance[l]:>7.3f}")
    d, n = dz, ~dz
    print(f"\nmean support-AUPRC  disease={auprc[d].mean():.3f}  nuisance={auprc[n].mean():.3f}  "
          f"chance≈{chance.mean():.3f}")
    print(f"mean cosine         disease={cos[d].mean():.3f}  nuisance={cos[n].mean():.3f}")
    if mu_v is not None:
        imp = np.abs(mu_v[assign])                 # |v| on each truth-matched factor
        print(f"|mu_v| on matched factors: disease={imp[d].mean():.3f}  nuisance={imp[n].mean():.3f}"
              f"  (disease should rank higher)")


if __name__ == "__main__":
    main()
