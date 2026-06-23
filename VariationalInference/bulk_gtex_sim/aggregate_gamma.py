#!/usr/bin/env python
"""Aggregate the multi-seed gamma* sweep: parse attribution.txt per config, report gamma_hat
mean +/- sd vs planted gamma*, plus leakage corr(theta,PRS), and plot the calibration line."""
import argparse, glob, os, re
import numpy as np
import pandas as pd


def parse(path):
    txt = open(path).read()
    gs = re.search(r"gamma\*\s*\(planted\)\s*=\s*([+\-\d.]+)", txt)
    gh = re.search(r"gamma_hat \(recovered\)\s*=\s*([+\-\d.]+)", txt)
    lk = re.search(r"max \|corr\(theta_hat_k, PRS\)\|\s*=\s*([\d.]+)", txt)
    return (float(gs.group(1)) if gs else np.nan,
            float(gh.group(1)) if gh else np.nan,
            float(lk.group(1)) if lk else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(f"{args.root}/cfg_*/attribution.txt")):
        cfg = os.path.basename(os.path.dirname(f))
        gm = re.search(r"_g([\d.]+)_s\d+_i(\d+)", cfg)
        gstar, ghat, leak = parse(f)
        rows.append(dict(gamma_star=gstar, gamma_hat=ghat, leak=leak,
                         seed=int(gm.group(2)) if gm else -1))
    if not rows:
        print(f"no attribution.txt under {args.root}/cfg_*/"); return
    df = pd.DataFrame(rows)
    g = df.groupby("gamma_star").agg(
        ghat_mean=("gamma_hat", "mean"), ghat_sd=("gamma_hat", "std"),
        leak_mean=("leak", "mean"), n=("gamma_hat", "size")).reset_index()
    g.to_csv(f"{args.root}/gamma_calibration.csv", index=False)
    print(g.to_string(index=False))
    print(f"\nleakage corr(theta,PRS): mean={df.leak.mean():.3f} max={df.leak.max():.3f} "
          f"(want ~0; chance ~1/sqrt(n))")

    if not args.no_plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        lim = [min(g.gamma_star.min(), g.ghat_mean.min()) - 0.05, g.gamma_star.max() + 0.1]
        ax.plot(lim, lim, ls="--", color="gray", label="identity ($\\hat\\gamma=\\gamma^\\star$)")
        ax.errorbar(g.gamma_star, g.ghat_mean, yerr=g.ghat_sd, fmt="o-", color="#8e44ad",
                    capsize=3, lw=2, ms=6, label="DRGP")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel("planted genetic effect $\\gamma^\\star$ (log-odds/SD)")
        ax.set_ylabel("recovered $\\hat\\gamma$")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        out = f"{args.root}/gamma_calibration.pdf"
        fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
        print(f"plot -> {out}")


if __name__ == "__main__":
    main()
