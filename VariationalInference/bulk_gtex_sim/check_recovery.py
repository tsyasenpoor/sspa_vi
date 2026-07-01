#!/usr/bin/env python
"""Score program recovery of a DRGP (or baseline) fit against the planted bulk-sim ground truth.

Reuses Simulations/evaluate.py recovery metrics (Hungarian alignment + support-AUPRC) on the
bulk-sim truth.npz. Reports per-program cosine + support-AUPRC, broken down by the v2 program
taxonomy (annotated/de novo x disease/nuisance), plus attribution: whether |upsilon_hat| (mu_v)
ranks disease-relevant matched factors above nuisance, and whether the RECOVERED v sign matches
the planted protective(-)/risk(+) direction on disease programs.

Masked fits restrict X to the pathway genes; the fit loadings are padded back onto the full truth
gene axis (zeros elsewhere) so masked recovers only the annotated programs (de novo fall to 0, as
they must -- their genes are outside the masked space).
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from VariationalInference.Simulations.evaluate import (
    hungarian_match, recovery_cosine, recovery_support_auprc)


def load_fit(fit_dir, eta_norm=True):
    """Return (beta_hat p_fit x K, gene_list len-p_fit or None, mu_v len-K or None).

    For DRGP fits (vi_gene_programs + vi_r_beta both present), beta_hat is reported as the
    HIERARCHICALLY NORMALIZED gene score -- the like-for-like analogue of scHPF's gene_score
    (= E[beta] * E[eta], scHPF_.py _score). DRGP saves raw E[beta] = r_beta * slab; the per-gene
    capacity eta_j = a_eta/(dp + sum_k slab_jk) ~ 1/sum_k slab_jk divides out the per-gene baseline
    (severe on dense bulk: high-expression genes inflate raw loadings). We score slab = E[beta]/r_beta
    (the spike's continuous loading, the conjugate partner of eta) row-normalized to constant total,
    matching scHPF whose gene_score rows already sum to ~a_eta. The a_eta constant and dp cancel in
    cosine/AUPRC, so no hyperparameters or saved eta are needed. eta_norm=False -> legacy raw E[beta].
    """
    vi_gp = os.path.join(fit_dir, "vi_gene_programs.csv.gz")
    gp = next((p for p in (vi_gp, os.path.join(fit_dir, "gene_programs.csv.gz")) if os.path.exists(p)), None)
    if gp:
        df = pd.read_csv(gp, index_col=0)
        vcols = [c for c in df.columns if c.startswith("v_weight_")]
        mu_v = df[vcols[0]].to_numpy() if vcols else None
        genes = [c for c in df.columns if c not in vcols]
        beta = df[genes].to_numpy().T                              # genes x K
        rb_path = os.path.join(fit_dir, "vi_r_beta.csv.gz")
        if eta_norm and gp == vi_gp and os.path.exists(rb_path):
            rb = pd.read_csv(rb_path, index_col=0).reindex(columns=genes).to_numpy().T
            slab = np.where(rb > 1e-9, beta / np.clip(rb, 1e-9, None), 0.0)  # E[beta]/r_beta = a_beta/b_beta
            row = slab.sum(axis=1, keepdims=True)                  # eta_j ~ 1/sum_k slab (a_eta, dp drop out)
            beta = np.where(row > 0, slab / (row + 1e-12), 0.0)    # hierarchically normalized gene score
        return beta, list(genes), mu_v
    npz = glob.glob(os.path.join(fit_dir, "*model_params.npz")) + glob.glob(os.path.join(fit_dir, "*.npz"))
    if npz:
        d = np.load(npz[0], allow_pickle=True)
        mu_v = np.asarray(d["mu_v"]).ravel() if "mu_v" in d else None
        return np.asarray(d["E_beta"]), None, mu_v
    raise FileNotFoundError(f"no recognizable fit output in {fit_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--truth", required=True, help="truth.npz from make_truth")
    ap.add_argument("--raw-loading", action="store_true",
                    help="score DRGP on raw E[beta] (legacy) instead of the eta-normalized "
                         "gene score that matches scHPF's gene_score")
    args = ap.parse_args()

    beta_hat, gene_list, mu_v = load_fit(args.fit_dir, eta_norm=not args.raw_loading)
    t = np.load(args.truth, allow_pickle=True)
    beta_star = t["beta"]                                       # G x K_truth (truth gene order)
    members = [np.asarray(m, dtype=int) for m in t["members"]]
    dz = np.asarray(t["disease_relevant"], dtype=bool)
    truth_genes = list(t["gene_ids"]); G = len(truth_genes)
    ann = np.asarray(t["annotated"], dtype=bool) if "annotated" in t.files else np.zeros(len(dz), bool)
    ptype = list(t["program_type"]) if "program_type" in t.files else \
        ["disease" if d else "nuisance" for d in dz]
    ups = np.asarray(t["upsilon"]) if "upsilon" in t.files else np.zeros(len(dz))

    # Pad/reorder fit loadings onto the full truth gene axis (handles masked's gene subset).
    if gene_list is not None:
        pos = {g: i for i, g in enumerate(truth_genes)}
        full = np.zeros((G, beta_hat.shape[1]), dtype=beta_hat.dtype)
        n_hit = 0
        for i, g in enumerate(gene_list):
            j = pos.get(g)
            if j is not None:
                full[j] = beta_hat[i]; n_hit += 1
        beta_hat = full
        if n_hit < len(gene_list):
            print(f"  note: {len(gene_list)-n_hit}/{len(gene_list)} fit genes not in truth (dropped)")
    assert beta_hat.shape[0] == beta_star.shape[0], \
        f"gene mismatch: fit {beta_hat.shape[0]} vs truth {beta_star.shape[0]}"

    assign, _ = hungarian_match(beta_hat, beta_star)
    cos = recovery_cosine(beta_hat, beta_star, assign)
    auprc = np.array(recovery_support_auprc(beta_hat, members, assign))
    p = beta_hat.shape[0]
    chance = np.array([len(s) / p for s in members])

    print(f"K_fit={beta_hat.shape[1]}  K_truth={beta_star.shape[1]}  genes={p}")
    print(f"{'prog':>4} {'type':>17} {'ups':>5} {'cosine':>7} {'AUPRC':>7} {'chance':>7}")
    for l in range(beta_star.shape[1]):
        print(f"{l:>4} {ptype[l]:>17} {ups[l]:>5.1f} {cos[l]:>7.3f} {auprc[l]:>7.3f} {chance[l]:>7.3f}")

    def grp(mask, name):
        if mask.any():
            print(f"  {name:<22} support-AUPRC={auprc[mask].mean():.3f}  cosine={cos[mask].mean():.3f}  (n={int(mask.sum())})")
    print(f"\nrecovery by group (chance≈{chance.mean():.3f}):")
    grp(ann & dz,  "annotated-disease")
    grp(ann & ~dz, "annotated-nuisance")
    grp(~ann & dz, "denovo-disease")
    grp(~ann & ~dz,"denovo-nuisance")
    grp(dz, "ALL disease"); grp(~dz, "ALL nuisance")

    if mu_v is not None:
        v_match = mu_v[assign]                                  # recovered v per truth program
        imp = np.abs(v_match)
        print(f"\nattribution |mu_v|: disease={imp[dz].mean():.3f}  nuisance={imp[~dz].mean():.3f}"
              f"  (disease should rank higher)")
        # protective(-)/risk(+) sign recovery on disease programs
        sign_ok = np.sign(v_match[dz]) == np.sign(ups[dz])
        print(f"protective/risk sign match (disease progs): {int(sign_ok.sum())}/{int(dz.sum())}"
              f"  [planted ups={ups[dz].astype(int).tolist()}  recovered sign={np.sign(v_match[dz]).astype(int).tolist()}]")


if __name__ == "__main__":
    main()
