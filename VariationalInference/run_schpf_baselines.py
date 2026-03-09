#!/usr/bin/env python
"""
scHPF + Downstream Classifier Pipeline
=======================================

Loads a trained scHPF model, extracts cell_score (θ·ξ) embeddings,
then trains LR / Lasso-LR / Ridge-LR classifiers for severity and outcome.

Usage:
    python run_schpf_baselines.py \
        --model /path/to/model.joblib \
        --data  /path/to/exp/exp.csv.gz \
        --output-dir ./results/schpf_baselines/exp
"""
import os
import sys
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

# scHPF needs to be importable
SCHPF_ROOT = "/labs/Aguiar/SSPA_BRAY/scHPF"
if SCHPF_ROOT not in sys.path:
    sys.path.insert(0, SCHPF_ROOT)


def parse_args():
    p = argparse.ArgumentParser(
        description="scHPF embeddings + downstream classifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Path to trained scHPF .joblib model")
    p.add_argument("--data", required=True,
                   help="Path to experiment CSV (.csv.gz) with metadata columns")
    p.add_argument("--labels", nargs="+", default=["severity", "outcome"],
                   help="Label columns to predict")
    p.add_argument("--meta-cols", nargs="+",
                   default=["sex", "comorbidity", "severity", "outcome", "cell_type"],
                   help="Columns that are metadata (not genes)")
    p.add_argument("--output-dir", "-o", default="./results/schpf_baselines",
                   help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


_CLASSIFIERS = {
    "schpf_lr":  lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "schpf_lrl": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="saga", max_iter=2000)),
    "schpf_lrr": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", max_iter=2000)),
}


def evaluate(model, X, y_true, label_name, alg_name, save_path):
    """Evaluate classifier and return metrics dict."""
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

    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
    }

    # Save predictions
    os.makedirs(save_path, exist_ok=True)
    np.savez(os.path.join(save_path, f"{alg_name}_{label_name}_preds.npz"),
             y_true=y_true, y_pred=y_pred, y_proba=y_proba)

    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 80)
    print("scHPF + DOWNSTREAM CLASSIFIERS")
    print("=" * 80)
    print(f"  Model:      {args.model}")
    print(f"  Data:       {args.data}")
    print(f"  Labels:     {args.labels}")
    print(f"  Seed:       {args.seed}")

    # ------------------------------------------------------------------
    # 1. Load scHPF model and extract cell embeddings
    # ------------------------------------------------------------------
    print("\n[1] Loading scHPF model...")
    model = joblib.load(args.model)
    cell_scores = model.cell_score()  # (n_cells, K)
    print(f"    cell_score shape: {cell_scores.shape}")

    # ------------------------------------------------------------------
    # 2. Load CSV metadata for labels
    # ------------------------------------------------------------------
    print("[2] Loading metadata from CSV...")
    df = pd.read_csv(args.data, index_col=0)
    meta = df[[c for c in args.meta_cols if c in df.columns]].copy()
    n_cells_csv = len(df)
    n_cells_model = cell_scores.shape[0]
    print(f"    CSV cells: {n_cells_csv}, model cells: {n_cells_model}")

    if n_cells_csv != n_cells_model:
        raise ValueError(
            f"Cell count mismatch: CSV has {n_cells_csv}, model has {n_cells_model}. "
            "Ensure the same data was used for scHPF training."
        )

    # ------------------------------------------------------------------
    # 3. Split indices (same seed → reproducible across experiments)
    # ------------------------------------------------------------------
    print("[3] Splitting data...")
    indices = np.arange(n_cells_model)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    train_idx, temp_idx = train_test_split(
        indices, test_size=(args.val_ratio + test_ratio),
        random_state=args.seed
    )
    val_proportion = args.val_ratio / (args.val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_proportion),
        random_state=args.seed
    )

    X_train = cell_scores[train_idx]
    X_val = cell_scores[val_idx]
    X_test = cell_scores[test_idx]

    print(f"    train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ------------------------------------------------------------------
    # 4. Train + evaluate for each label × classifier
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for label_col in args.labels:
        if label_col not in meta.columns:
            print(f"  WARNING: '{label_col}' not in metadata, skipping.")
            continue

        y_all = meta[label_col].values.astype(int)
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        y_test = y_all[test_idx]

        print(f"\n--- Label: {label_col} ---")
        print(f"    train dist: {np.bincount(y_train)}")
        print(f"    val   dist: {np.bincount(y_val)}")
        print(f"    test  dist: {np.bincount(y_test)}")

        for alg_name, clf_fn in _CLASSIFIERS.items():
            print(f"\n  Training {alg_name} for {label_col}...")
            clf = clf_fn()
            clf.fit(X_train, y_train)

            train_acc = clf.score(X_train, y_train)
            print(f"    train accuracy: {train_acc:.4f}")

            # Save model
            alg_dir = str(output_dir / label_col)
            os.makedirs(alg_dir, exist_ok=True)
            joblib.dump(clf, os.path.join(alg_dir, f"{alg_name}_model.pkl"))

            # Evaluate val
            print("  [validation]")
            val_res = evaluate(clf, X_val, y_val, label_col, alg_name, alg_dir)

            # Evaluate test
            print("  [test]")
            test_res = evaluate(clf, X_test, y_test, label_col, alg_name, alg_dir)

            all_results[f"{alg_name}_{label_col}"] = {
                "train_accuracy": float(train_acc),
                "val": val_res,
                "test": test_res,
            }

    # ------------------------------------------------------------------
    # 5. Save summary
    # ------------------------------------------------------------------
    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": all_results,
    }
    summary_path = output_dir / "schpf_baselines_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    # Print table
    print("\n" + "=" * 80)
    print(f"{'Algorithm':<15} {'Label':<12} {'Val Acc':>8} {'Val F1':>8} {'Val AUC':>8} {'Test Acc':>9} {'Test F1':>8} {'Test AUC':>9}")
    print("-" * 80)
    for key, res in all_results.items():
        parts = key.rsplit("_", 1)
        alg = parts[0]
        lab = parts[1] if len(parts) > 1 else key
        v, t = res["val"], res["test"]
        print(f"{alg:<15} {lab:<12} {v['accuracy']:>8.4f} {v['f1']:>8.4f} {v['roc_auc']:>8.4f} "
              f"{t['accuracy']:>9.4f} {t['f1']:>8.4f} {t['roc_auc']:>9.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
