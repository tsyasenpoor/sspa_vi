#!/usr/bin/env python
"""Assemble the loader-compatible simulated dataset CSV from injected counts + labels + aux.

DataLoader (simulated-CSV path) expects: rows = samples, gene columns detected by ENSG prefix,
remaining columns = metadata (label + aux). Output: <name>.csv.gz under a 'simulated' path so
DataLoader.is_simulated triggers.
"""
import argparse, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injected", required=True, help="X_injected.tsv.gz (gene x sample)")
    ap.add_argument("--labels", required=True, help="labels.npz from make_labels")
    ap.add_argument("--out", required=True, help="output .csv.gz (path should contain 'simulated' or 'sim_')")
    ap.add_argument("--label-name", default="heart_disease")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # genes x samples -> samples x genes
    Xg = pd.read_csv(args.injected, sep="\t", index_col=0)
    X = Xg.T                                   # samples x genes (ENSG columns)
    X.index.name = "sample_id"

    lab = np.load(args.labels, allow_pickle=True)
    y, prs = lab["y"], lab["prs"]
    prs_present = bool(lab["prs_present"]) if "prs_present" in lab else float(lab["gamma_prs"]) != 0.0
    assert X.shape[0] == len(y), f"{X.shape[0]} samples != {len(y)} labels"

    X[args.label_name] = y.astype(int)
    if prs_present:                            # PRS is a covariate (incl. gamma*=0 negative control)
        X["PRS"] = prs.astype(float)

    X.to_csv(args.out, compression="gzip")
    n_gene = Xg.shape[0]
    print(f"dataset: {X.shape[0]} samples x {n_gene} genes | label '{args.label_name}' "
          f"prevalence={y.mean():.3f} | PRS aux={'yes' if prs_present else 'no'}", flush=True)
    print(f"DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
