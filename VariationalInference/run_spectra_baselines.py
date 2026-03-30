#!/usr/bin/env python
"""
Spectra + Downstream Classifier Pipeline
==========================================

Loads pre-computed Spectra cell_scores (.npy) and original CSV metadata,
then trains LR / Lasso-LR / Ridge-LR for severity and outcome prediction.

Usage:
    python run_spectra_baselines.py \
        --cell-scores /path/to/spectra_cell_scores.npy \
        --data /path/to/exp/exp.csv.gz \
        --output-dir ./results/spectra_baselines/exp
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
import json
from datetime import datetime

META_COLS = ["sex", "comorbidity", "severity", "outcome", "cell_type", "sampleID"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Spectra embeddings + downstream classifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cell-scores", required=True,
                   help="Path to spectra_cell_scores.npy")
    p.add_argument("--data", required=True,
                   help="Path to experiment CSV (.csv.gz) with metadata")
    p.add_argument("--labels", nargs="+", default=["severity", "outcome"],
                   help="Label columns to predict")
    p.add_argument("--output-dir", "-o", default="./results/spectra_baselines")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    return p.parse_args()


_CLASSIFIERS = {
    "spectra_lr":  lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "spectra_lrl": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="saga", max_iter=2000)),
    "spectra_lrr": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", max_iter=2000)),
}


def evaluate(model, X, y_true, label_name, alg_name, save_path):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    n_classes = y_proba.shape[1]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

    print(f"  [{alg_name} | {label_name}]  acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")
    print(f"    confusion matrix:\n{cm}")

    os.makedirs(save_path, exist_ok=True)
    np.savez(os.path.join(save_path, f"{alg_name}_{label_name}_preds.npz"),
             y_true=y_true, y_pred=y_pred, y_proba=y_proba)

    return {
        "accuracy": float(acc), "precision": float(prec),
        "recall": float(rec), "f1": float(f1), "roc_auc": float(auc),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 80)
    print("SPECTRA + DOWNSTREAM CLASSIFIERS")
    print("=" * 80)

    # Load cell scores
    cell_scores = np.load(args.cell_scores)
    print(f"  Cell scores: {cell_scores.shape}")

    # Load metadata
    df = pd.read_csv(args.data, index_col=0)
    meta = df[[c for c in META_COLS if c in df.columns]]
    assert len(df) == cell_scores.shape[0], "Cell count mismatch"

    # Split — patient-grouped if sampleID available, else cell-level
    n = cell_scores.shape[0]
    indices = np.arange(n)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    if "sampleID" in df.columns:
        # Patient-grouped split: no donor leakage
        patient_ids = df["sampleID"].values.astype(str)
        unique_patients = np.unique(patient_ids)
        train_pat, temp_pat = train_test_split(
            unique_patients, test_size=(args.val_ratio + test_ratio),
            random_state=args.seed
        )
        val_proportion = args.val_ratio / (args.val_ratio + test_ratio)
        val_pat, test_pat = train_test_split(
            temp_pat, test_size=(1 - val_proportion), random_state=args.seed
        )
        train_set, val_set, test_set = set(train_pat), set(val_pat), set(test_pat)
        train_idx = np.where([p in train_set for p in patient_ids])[0]
        val_idx = np.where([p in val_set for p in patient_ids])[0]
        test_idx = np.where([p in test_set for p in patient_ids])[0]
        print(f"  Patient-grouped split: {len(train_pat)} / {len(val_pat)} / {len(test_pat)} patients")
    else:
        train_idx, temp_idx = train_test_split(
            indices, test_size=(args.val_ratio + test_ratio), random_state=args.seed
        )
        val_proportion = args.val_ratio / (args.val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_proportion), random_state=args.seed
        )

    X_train = cell_scores[train_idx]
    X_val = cell_scores[val_idx]
    X_test = cell_scores[test_idx]
    print(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for label_col in args.labels:
        if label_col not in meta.columns:
            print(f"  WARNING: '{label_col}' not found, skipping.")
            continue

        y_all = meta[label_col].values.astype(int)
        y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]

        print(f"\n--- Label: {label_col} ---")
        print(f"    train dist: {np.bincount(y_train)}")

        for alg_name, clf_fn in _CLASSIFIERS.items():
            print(f"\n  Training {alg_name} for {label_col}...")
            clf = clf_fn()
            clf.fit(X_train, y_train)
            print(f"    train acc: {clf.score(X_train, y_train):.4f}")

            alg_dir = str(output_dir / label_col)
            os.makedirs(alg_dir, exist_ok=True)
            joblib.dump(clf, os.path.join(alg_dir, f"{alg_name}_model.pkl"))

            print("  [validation]")
            val_res = evaluate(clf, X_val, y_val, label_col, alg_name, alg_dir)
            print("  [test]")
            test_res = evaluate(clf, X_test, y_test, label_col, alg_name, alg_dir)

            all_results[f"{alg_name}_{label_col}"] = {
                "train_accuracy": float(clf.score(X_train, y_train)),
                "val": val_res, "test": test_res,
            }

    # Summary
    summary = {"timestamp": datetime.now().isoformat(), "args": vars(args), "results": all_results}
    with open(output_dir / "spectra_baselines_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"{'Algorithm':<16} {'Label':<12} {'Val Acc':>8} {'Val F1':>8} {'Val AUC':>8} {'Test Acc':>9} {'Test F1':>8} {'Test AUC':>9}")
    print("-" * 80)
    for key, res in all_results.items():
        parts = key.rsplit("_", 1)
        alg, lab = parts[0], parts[1]
        v, t = res["val"], res["test"]
        print(f"{alg:<16} {lab:<12} {v['accuracy']:>8.4f} {v['f1']:>8.4f} {v['roc_auc']:>8.4f} "
              f"{t['accuracy']:>9.4f} {t['f1']:>8.4f} {t['roc_auc']:>9.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
