#!/usr/bin/env python
"""Generate the binary disease label from program activity + (optional) genetic PRS.

y_i ~ Bernoulli(g(alpha + theta*_i . upsilon* + PRS_i * gamma_prs + xaux_i . gamma_aux))

- theta*, upsilon* come from make_truth (only disease-relevant programs have nonzero upsilon*).
- PRS enters ONLY here (label), never expression. Sampled from an empirical/N(0,1) distribution
  INDEPENDENT of the samples (per the design: the per-subject PRS link is biologically vacuous in
  synthetic data; what we retain is the PRS marginal).
- alpha is calibrated so the marginal prevalence matches --prevalence (default 0.26 ~ real WB
  heart_disease).
"""
import argparse, os
import numpy as np
from scipy.optimize import brentq


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="truth.npz from make_truth")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prevalence", type=float, default=0.26)
    ap.add_argument("--gamma-prs", type=float, default=0.0,
                    help="PRS label coefficient on the z-scored PRS (log-odds per SD). 0 = no genetic effect")
    ap.add_argument("--prs", default=None, help="optional .npy of z-scored PRS values (len n_sim); "
                    "if omitted and gamma-prs!=0, PRS ~ N(0,1) is drawn")
    ap.add_argument("--label-seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.label_seed)

    t = np.load(args.truth, allow_pickle=True)
    theta = t["theta"]            # n_sim x K  carrier activity
    upsilon = t["upsilon"]        # K,
    n = theta.shape[0]

    # program contribution to the logit
    z_prog = theta @ upsilon      # n_sim,

    # genetic contribution (label-only)
    if args.gamma_prs != 0.0:
        if args.prs:
            prs = np.load(args.prs).astype(float)
            assert len(prs) == n, f"PRS length {len(prs)} != n_sim {n}"
        else:
            prs = rng.standard_normal(n)          # empirical/N(0,1) marginal, label-independent draw
        z_gen = args.gamma_prs * prs
    else:
        prs = np.zeros(n)
        z_gen = np.zeros(n)

    z_core = z_prog + z_gen

    # calibrate intercept alpha so E[sigmoid(alpha + z_core)] == prevalence
    f = lambda a: sigmoid(a + z_core).mean() - args.prevalence
    alpha = brentq(f, -50, 50)
    p = sigmoid(alpha + z_core)
    y = (rng.random(n) < p).astype(int)

    np.savez_compressed(
        f"{args.out_dir}/labels.npz",
        y=y, p=p, prs=prs, alpha=alpha, z_prog=z_prog, z_gen=z_gen,
        gamma_prs=args.gamma_prs, upsilon=upsilon,
    )
    print(f"labels: n={n} prevalence={y.mean():.3f} (target {args.prevalence}) "
          f"alpha={alpha:.3f} gamma_prs={args.gamma_prs}", flush=True)
    print(f"DONE -> {args.out_dir}/labels.npz", flush=True)


if __name__ == "__main__":
    main()
