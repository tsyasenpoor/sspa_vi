"""GTEx experimental: aux-only-LR baseline + bootstrap 95% CIs on the held-out test fold.

Motivation: gene-LR (with aux) lands near chance while factor methods reach ~0.72-0.76. Need to
know (a) how much AUC is the clinical aux alone, and (b) whether differences are real given the
single 107-subject test fold. All methods share the SAME fixed split (split-seed 0), so we read
their saved test predictions and bootstrap on common cells; aux-only is fit on the replicated split.

Outputs results/gtex_experimental/methods/auc_bootstrap_ci.csv (per-method AUC + CI) and prints
paired bootstrap deltas for the key comparisons.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/labs/Aguiar/SSPA_BRAY/BRay")
M = Path("/labs/Aguiar/SSPA_BRAY/results/gtex_experimental/methods")
DATA = "/labs/Aguiar/SSPA_BRAY/data/gtex_wb_experimental/gtex_wb_counts.h5ad"
LABEL = "heart_disease"
AUX = ["sex_female", "race_indian", "race_asian", "race_black", "race_missing",
       "age", "BMI", "smoking", "MHHTN", "MHT2D"]
SEED = 42          # any seed: the split is fixed (split-seed 0); preds differ only by init
B = 5000


def boot_ci(y, p, rng, B=B):
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    idx = rng.integers(0, n, size=(B, n))
    aucs = []
    for b in idx:
        yb = y[b]
        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc_score(yb, p[b]))
    a = np.array(aucs)
    return roc_auc_score(y, p), np.percentile(a, 2.5), np.percentile(a, 97.5)


def paired_delta(y, pa, pb, rng, B=B):
    """Bootstrap CI for AUC(a) - AUC(b) on the same resamples (paired)."""
    y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb); n = len(y)
    idx = rng.integers(0, n, size=(B, n)); d = []
    for bi in idx:
        yb = y[bi]
        if yb.min() == yb.max():
            continue
        d.append(roc_auc_score(yb, pa[bi]) - roc_auc_score(yb, pb[bi]))
    d = np.array(d)
    return roc_auc_score(y, pa) - roc_auc_score(y, pb), np.percentile(d, 2.5), np.percentile(d, 97.5)


def drgp_preds(mode):
    p = M / f"drgp_{mode}" / f"seed{SEED}" / "vi_test_predictions.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.set_index("cell_id")[f"prob_{LABEL}"].rename(f"DRGP-{mode}"), \
           df.set_index("cell_id")[f"true_{LABEL}"]


def base_preds(alg, name):
    p = M / "baselines" / f"seed{SEED}" / f"{alg}_results.pkl"
    if not p.exists():
        return None
    d = joblib.load(p)
    return pd.Series(np.asarray(d["y_prob"]).ravel(), index=list(d["cell_ids"]), name=name), \
           pd.Series(np.asarray(d["y_true"]).ravel(), index=list(d["cell_ids"]))


def aux_only():
    from VariationalInference.data_loader import DataLoader
    loader = DataLoader(DATA, verbose=False)
    data = loader.load_and_preprocess(label_column=LABEL, aux_columns=AUX, train_ratio=0.7,
                                      val_ratio=0.15, stratify_by=LABEL, min_cells_expressing=0.001,
                                      random_state=0, patient_column="subject_id")
    _, Xa_tr, ytr = data["train"]; _, Xa_te, yte = data["test"]
    ytr = np.asarray(ytr).ravel(); yte = np.asarray(yte).ravel()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(np.asarray(Xa_tr), ytr)
    prob = clf.predict_proba(np.asarray(Xa_te))[:, 1]
    # test cell ids (match baseline order via splits if available)
    ids = None
    sp = data.get("splits")
    if isinstance(sp, dict) and "test" in sp:
        ids = list(sp["test"])
    return prob, yte, ids


def main():
    rng = np.random.default_rng(0)
    series = {}; truth = None
    for mode in ("unmasked", "masked"):
        r = drgp_preds(mode)
        if r: series[f"DRGP-{mode}"], truth = r[0], r[1] if truth is None else truth
    for alg, nm in [("lr", "gene-LR"), ("lrl", "gene-L1-LR"), ("mflr", "NMF+LR")]:
        r = base_preds(alg, nm)
        if r: series[nm] = r[0]
    # aux-only (align to baseline test cell ids)
    ap, ay, aids = aux_only()
    if aids is not None and len(aids) == len(ap):
        series["aux-only-LR"] = pd.Series(ap, index=aids, name="aux-only-LR")
    else:
        # fall back: align by the baseline cell_ids (same split) assuming same order
        bser, bt = base_preds("lr", "gene-LR")
        series["aux-only-LR"] = pd.Series(ap, index=list(bt.index)[:len(ap)], name="aux-only-LR")

    # common cells across all methods + truth
    df = pd.concat(list(series.values()) + [truth.rename("y")], axis=1, join="inner").dropna()
    y = df["y"].to_numpy().astype(int)
    print(f"=== GTEx test fold: {len(df)} subjects, {int(y.sum())} positive ===\n")
    rows = []
    print(f"{'method':16s} {'AUC':>6s}  95% CI")
    for m in series:
        auc, lo, hi = boot_ci(y, df[m].to_numpy(), rng)
        rows.append((m, auc, lo, hi)); print(f"{m:16s} {auc:6.3f}  [{lo:.3f}, {hi:.3f}]")
    with open(M / "auc_bootstrap_ci.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "auc", "ci_lo", "ci_hi"]); w.writerows(rows)

    print("\n=== paired bootstrap deltas (same resamples) ===")
    pairs = [("NMF+LR", "DRGP-unmasked"), ("DRGP-unmasked", "aux-only-LR"),
             ("NMF+LR", "aux-only-LR"), ("DRGP-unmasked", "gene-LR")]
    for a, b in pairs:
        if a in series and b in series:
            d, lo, hi = paired_delta(y, df[a].to_numpy(), df[b].to_numpy(), rng)
            sig = "" if (lo <= 0 <= hi) else "  *significant"
            print(f"{a:14s} - {b:14s} = {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
    print(f"\nwrote {M/'auc_bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()
