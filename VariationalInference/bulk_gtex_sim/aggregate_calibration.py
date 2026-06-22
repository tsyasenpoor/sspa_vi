#!/usr/bin/env python
"""Aggregate the effect-size calibration sweep: parse each config's recovery.txt, build the
disease-AUPRC-vs-effect curve, and recommend the effect where disease programs clear the copula
modes (disease support-AUPRC comfortably above nuisance and chance)."""
import argparse, glob, os, re
import numpy as np
import pandas as pd


def parse_recovery(path):
    """Pull mean disease/nuisance support-AUPRC + chance from a recovery.txt."""
    txt = open(path).read()
    m = re.search(r"mean support-AUPRC\s+disease=([\d.]+)\s+nuisance=([\d.]+)\s+chance.([\d.]+)", txt)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (np.nan,)*3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    rows = []
    for rec in sorted(glob.glob(f"{args.root}/cfg_*/recovery.txt")):
        cfg = os.path.basename(os.path.dirname(rec))
        m = re.search(r"cfg_e([\d.]+)_", cfg)
        eff = float(m.group(1)) if m else np.nan
        d, n, c = parse_recovery(rec)
        rows.append(dict(effect=eff, disease_auprc=d, nuisance_auprc=n, chance=c,
                         margin=d - n, lift=d - c))
    if not rows:
        print(f"no recovery.txt under {args.root}/cfg_*/"); return
    df = pd.DataFrame(rows).sort_values("effect").reset_index(drop=True)
    df.to_csv(f"{args.root}/calibration_curve.csv", index=False)
    print(df.to_string(index=False))
    # recommend smallest effect where disease clears nuisance by >=0.10 and chance by >=0.15
    ok = df[(df.margin >= 0.10) & (df.lift >= 0.15)]
    rec = ok.effect.min() if len(ok) else float("nan")
    print(f"\nRecommended effect (disease-nuisance>=0.10 & disease-chance>=0.15): {rec}")
    print(f"curve -> {args.root}/calibration_curve.csv")


if __name__ == "__main__":
    main()
