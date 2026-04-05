#!/usr/bin/env python
"""
Patient-level evaluation of saved baseline models.
===================================================

Re-loads saved model pickles from a prior run_baselines.py run, re-loads the
same data (deterministic with the same seed/subsample), predicts cell-level
probabilities, aggregates to patient level (mean probability), and computes
patient-level classification metrics.

No retraining — only prediction + aggregation.

Usage:
    python run_baselines_patient_eval.py \
        --data /path/to/covid19.h5ad \
        --model-dir /path/to/baselines/severity \
        --label-column "CoVID-19 severity" \
        --aux-columns Sex cm_asthma_copd cm_cardio cm_diabetes \
        --patient-column sampleID \
        --seed 42 \
        --output-dir /path/to/baselines_patient/severity
"""
import os
import sys
from pathlib import Path

# Add BRay/ to path for VariationalInference imports (comp/ → VI/ → BRay/)
script_dir = Path(__file__).resolve().parent
bray_dir = script_dir.parent.parent
if str(bray_dir) not in sys.path:
    sys.path.insert(0, str(bray_dir))

import argparse
import json
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ALGORITHMS = ["svm", "lr", "lrl", "lrr", "mflr", "mflrl", "mflrr"]


def _to_dense(x):
    import scipy.sparse as sp
    if hasattr(x, "numpy"):
        x = x.numpy()
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def _aggregate_to_patient(patient_ids, y_true, y_proba):
    """Aggregate cell-level predictions to patient level via mean probability.

    Returns patient_ids_unique, y_patient_true, y_patient_proba, y_patient_pred.
    """
    df = pd.DataFrame({
        "patient": patient_ids,
        "y_true": y_true,
        "y_proba": y_proba,
    })
    grouped = df.groupby("patient").agg(
        y_true=("y_true", "first"),  # same for all cells of a patient
        y_proba=("y_proba", "mean"),
    )
    y_patient_true = grouped["y_true"].values.astype(int)
    y_patient_proba = grouped["y_proba"].values
    y_patient_pred = (y_patient_proba >= 0.5).astype(int)
    return grouped.index.values, y_patient_true, y_patient_proba, y_patient_pred


def _compute_metrics(y_true, y_pred, y_proba):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_patients": len(y_true),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Patient-level evaluation of saved baseline models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", "-d", type=str, required=True,
                   help="Path to h5ad file (same as original run)")
    p.add_argument("--model-dir", type=str, required=True,
                   help="Directory containing saved *_model.pkl files from run_baselines.py")
    p.add_argument("--label-column", type=str, required=True,
                   help="Label column (must match original run)")
    p.add_argument("--aux-columns", type=str, nargs="*", default=[],
                   help="Auxiliary columns (must match original run)")
    p.add_argument("--patient-column", type=str, default="sampleID",
                   help="Column identifying patients/donors")
    p.add_argument("--output-dir", "-o", type=str, required=True,
                   help="Directory to save patient-level results")
    p.add_argument("--latent-dim", type=int, default=50,
                   help="NMF latent dim (must match original run)")
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed (must match original run for identical splits)")
    p.add_argument("--cache-dir", type=str, default="/labs/Aguiar/SSPA_BRAY/cache")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--normalize-target-sum", type=float, default=1e4)
    p.add_argument("--normalize-method", type=str, default="library_size")
    p.add_argument("--subsample-ratio", type=float, default=None)
    p.add_argument("--subsample-n-patients", type=int, default=None)
    p.add_argument("--subsample-seed", type=int, default=0)
    p.add_argument("--algorithms", type=str, nargs="+", default=ALGORITHMS)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PATIENT-LEVEL EVALUATION OF SAVED BASELINE MODELS")
    print("=" * 70)
    print(f"  Model dir:  {model_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Seed:       {args.seed}")

    # ------------------------------------------------------------------
    # 1. Re-load data with identical config to recover cell→patient map
    # ------------------------------------------------------------------
    print("\nLoading data (same config as original run)...")

    from VariationalInference.data_loader import DataLoader

    _preloaded_adata = None
    if args.subsample_ratio is not None or args.subsample_n_patients is not None:
        import anndata as ad
        from VariationalInference.create_subsamples import subsample_adata
        _full = ad.read_h5ad(args.data)
        _full.var_names_make_unique()
        _preloaded_adata = subsample_adata(
            _full,
            ratio=args.subsample_ratio,
            n_patients=args.subsample_n_patients,
            subsample_seed=args.subsample_seed,
            verbose=True,
        )
        del _full

    loader = DataLoader(
        data_path=args.data,
        cache_dir=args.cache_dir,
        use_cache=True,
        verbose=args.verbose,
        adata=_preloaded_adata,
    )

    data = loader.load_and_preprocess(
        label_column=args.label_column,
        aux_columns=args.aux_columns,
        train_ratio=0.7,
        val_ratio=0.15,
        stratify_by=args.label_column,
        min_cells_expressing=0.001,
        layer="raw",
        convert_to_ensembl=True,
        filter_protein_coding=False,
        random_state=args.seed,
        normalize=args.normalize,
        normalize_target_sum=args.normalize_target_sum,
        normalize_method=args.normalize_method,
        return_sparse=False,
        patient_column=args.patient_column,
    )

    X_val, X_aux_val, y_val = data["val"]
    X_test, X_aux_test, y_test = data["test"]
    splits = data["splits"]
    cell_metadata = data["cell_metadata"]

    # Build cell→patient mapping for val and test
    val_patients = cell_metadata.loc[splits["val"], args.patient_column].values
    test_patients = cell_metadata.loc[splits["test"], args.patient_column].values

    n_val_patients = len(np.unique(val_patients))
    n_test_patients = len(np.unique(test_patients))
    print(f"  Val:  {len(y_val)} cells, {n_val_patients} patients")
    print(f"  Test: {len(y_test)} cells, {n_test_patients} patients")

    # ------------------------------------------------------------------
    # 2. Load saved models and evaluate at patient level
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Patient-Level Evaluation")
    print("=" * 70)

    all_results = {}

    for alg in args.algorithms:
        model_path = model_dir / f"{alg}_model.pkl"
        if not model_path.exists():
            print(f"\n  [{alg}] SKIPPED — {model_path} not found")
            continue

        print(f"\n  [{alg}] Loading model...")
        model = joblib.load(model_path)

        nmf_obj = None
        if alg.startswith("mf"):
            nmf_path = model_dir / f"{alg}_nmf.pkl"
            if nmf_path.exists():
                nmf_obj = joblib.load(nmf_path)
            else:
                print(f"    WARNING: {nmf_path} not found, skipping {alg}")
                continue

        # Prepare features
        def _prepare(X_gex, X_aux):
            X_gex = _to_dense(X_gex)
            X_aux = _to_dense(X_aux)
            if alg.startswith("mf"):
                X_latent = nmf_obj.transform(X_gex)
                return np.concatenate((X_latent, X_aux), axis=1)
            return np.concatenate((X_gex, X_aux), axis=1)

        for split_name, X_gex, X_aux, y_true, patient_ids in [
            ("val", X_val, X_aux_val, y_val, val_patients),
            ("test", X_test, X_aux_test, y_test, test_patients),
        ]:
            X_feat = _prepare(X_gex, X_aux)
            y_proba_full = model.predict_proba(X_feat)
            # Binary: use positive-class probability
            y_proba = y_proba_full[:, 1] if y_proba_full.shape[1] == 2 else y_proba_full[:, 1]

            pat_ids, y_pat_true, y_pat_proba, y_pat_pred = _aggregate_to_patient(
                patient_ids, y_true, y_proba
            )
            metrics = _compute_metrics(y_pat_true, y_pat_pred, y_pat_proba)

            key = f"{split_name}"
            all_results.setdefault(alg, {})[key] = metrics

            print(f"    {split_name}: acc={metrics['accuracy']:.4f}  "
                  f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}  "
                  f"({metrics['n_patients']} patients)")

            # Save per-patient predictions
            pat_df = pd.DataFrame({
                "patient": pat_ids,
                "y_true": y_pat_true,
                "y_proba_mean": y_pat_proba,
                "y_pred": y_pat_pred,
            })
            alg_out = output_dir / alg
            alg_out.mkdir(parents=True, exist_ok=True)
            pat_df.to_csv(alg_out / f"{split_name}_patient_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Patient-Level Summary (test set)")
    print("-" * 70)
    print(f"{'Algorithm':<10} {'Acc':<10} {'F1':<10} {'AUC':<10} {'N_pat':<6}")
    print("-" * 70)
    for alg, res in all_results.items():
        t = res.get("test", {})
        print(f"{alg:<10} {t.get('accuracy', 0):.4f}     "
              f"{t.get('f1', 0):.4f}     {t.get('roc_auc', 0):.4f}     "
              f"{t.get('n_patients', 0)}")
    print("-" * 70)

    # Save full results
    with open(output_dir / "patient_level_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    config = {
        "model_dir": str(model_dir),
        "data": args.data,
        "label_column": args.label_column,
        "patient_column": args.patient_column,
        "seed": args.seed,
        "subsample_n_patients": args.subsample_n_patients,
        "subsample_seed": args.subsample_seed,
        "algorithms": args.algorithms,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
