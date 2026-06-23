#!/usr/bin/env python
"""Score genetic-attribution for a DRGP fit with a PRS auxiliary covariate.

The benchmark's genetic test: the model must route genetic risk through the auxiliary weight
gamma (PRS -> gamma), NOT into the gene programs. This checks:
  (1) gamma_hat (mu_gamma) vs the planted gamma* (recovery of the genetic effect);
  (2) gamma*=0 negative control -> gamma_hat ~ 0 (model must not invent a genetic effect);
  (3) corr(theta_hat_k, PRS) ~ 0 for all programs (no PRS leakage into program activity).
"""
import argparse, glob, os
import numpy as np
import pandas as pd


def load_gamma_prs(fit_dir):
    """Return the PRS (aux) coefficient, NOT the intercept. The saved gamma vector is
    [intercept, aux_0, aux_1, ...]; with a single PRS covariate the PRS weight is 'aux_0'."""
    gw = os.path.join(fit_dir, "vi_gamma_weights.csv.gz")
    if os.path.exists(gw):                          # labeled columns: 'intercept', 'aux_0', ...
        df = pd.read_csv(gw, index_col=0)
        aux = [c for c in df.columns if c != "intercept"]
        return float(df[aux[0]].iloc[0])            # first (only) aux = PRS
    npz = glob.glob(os.path.join(fit_dir, "*model_params.npz"))
    if npz:
        d = np.load(npz[0], allow_pickle=True)
        if "mu_gamma" in d:
            mg = np.asarray(d["mu_gamma"]).ravel()  # [intercept, PRS]; take the PRS (last/aux) col
            return float(mg[-1]) if mg.size >= 2 else float(mg[0])
    raise FileNotFoundError(f"no gamma weights in {fit_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--labels", required=True, help="labels.npz (has gamma_prs, prs)")
    args = ap.parse_args()

    gamma_hat = load_gamma_prs(args.fit_dir)
    lab = np.load(args.labels, allow_pickle=True)
    gamma_star = float(lab["gamma_prs"])
    prs = lab["prs"]

    # theta_hat on train split -> corr with PRS (need the train sample ids to align PRS)
    leak = np.nan
    tt = os.path.join(args.fit_dir, "vi_theta_train.csv.gz")
    if os.path.exists(tt):
        th = pd.read_csv(tt, index_col=0)
        # the saved theta frame appends aux/label columns; keep only gene-program columns (GP*)
        gp_cols = [c for c in th.columns if str(c).startswith("GP")]
        th = th[gp_cols]
        # train rows are 'sim_<i>' (1-based) or Patient_*; map to PRS by integer suffix if possible
        try:
            idx = [int(s.split("_")[-1]) - 1 for s in th.index.astype(str)]
            prs_tr = prs[idx]
            cors = [abs(np.corrcoef(prs_tr, th.iloc[:, k])[0, 1]) for k in range(th.shape[1])]
            leak = float(np.nanmax(cors))
        except Exception:
            pass


    print(f"gamma*  (planted)     = {gamma_star:+.4f}")
    print(f"gamma_hat (recovered) = {gamma_hat:+.4f}")
    if gamma_star == 0.0:
        print(f"  [neg control] |gamma_hat| = {abs(gamma_hat):.4f}  (want ~0: no invented effect)")
    else:
        print(f"  sign match: {np.sign(gamma_hat)==np.sign(gamma_star)}  "
              f"ratio gamma_hat/gamma* = {gamma_hat/gamma_star:+.3f}")
    if not np.isnan(leak):
        print(f"max |corr(theta_hat_k, PRS)| = {leak:.4f}  (want ~0: no PRS leakage into programs)")


if __name__ == "__main__":
    main()
