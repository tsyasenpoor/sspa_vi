#!/usr/bin/env python
"""
Quick Reference: Variational Inference for Single-Cell Analysis
================================================================

This script demonstrates the complete VI/SVI workflow for gene program discovery
and phenotype classification from single-cell RNA-seq data.

USAGE:
------
    # Basic usage with batch VI (default)
    python quick_reference.py --data /path/to/data.h5ad --n-factors 50

    # Use Stochastic VI (SVI) for large datasets
    python quick_reference.py \
        --data /path/to/data.h5ad \
        --n-factors 50 \
        --method svi \
        --batch-size 128 \
        --max-epochs 100

    # Full example with VI
    python quick_reference.py \
        --data /path/to/data.h5ad \
        --n-factors 50 \
        --method vi \
        --max-iter 200 \
        --label-column t2dm \
        --aux-columns Sex \
        --output-dir ./results \
        --verbose

WORKFLOW:
---------
1. Load and preprocess h5ad data (gene expression)
2. Create random train/val/test splits
3. Train VI or SVI model with specified parameters
4. Evaluate on validation and test sets
5. Save results (model, gene programs, predictions)

For more information:
    python quick_reference.py --help

For CLI usage:
    python -m VariationalInference.cli train --help
"""

import sys
from pathlib import Path

# Add parent directory to path to allow VariationalInference imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import argparse
import numpy as np
import pandas as pd
from typing import Optional, List


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Variational Inference Quick Reference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to h5ad file with gene expression data'
    )
    parser.add_argument(
        '--n-factors', '-k',
        type=int,
        required=True,
        help='Number of latent gene programs to discover'
    )

    # Method selection
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['vi', 'svi'],
        default='vi',
        help='Inference method: vi (batch) or svi (stochastic). Use svi for large datasets.'
    )

    # Data options
    parser.add_argument(
        '--label-column',
        type=str,
        default='t2dm',
        help='Column name in adata.obs for classification labels'
    )
    parser.add_argument(
        '--aux-columns',
        type=str,
        nargs='+',
        default=None,
        help='Column names for auxiliary features (e.g., Sex, batch)'
    )
    parser.add_argument(
        '--layer',
        type=str,
        default='raw',
        help='Which layer to use from h5ad file (default: raw)'
    )
    parser.add_argument(
        '--gene-annotation',
        type=str,
        default=None,
        help='Path to gene annotation CSV for protein-coding filter'
    )

    # VI Training options
    parser.add_argument(
        '--max-iter',
        type=int,
        default=200,
        help='Maximum training iterations (VI only)'
    )
    parser.add_argument(
        '--min-iter',
        type=int,
        default=50,
        help='Minimum iterations before checking convergence (VI only)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience'
    )

    # SVI-specific options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Mini-batch size for SVI (only used with --method svi)'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=100,
        help='Maximum epochs for SVI (only used with --method svi)'
    )
    parser.add_argument(
        '--min-epochs',
        type=int,
        default=10,
        help='Minimum epochs before convergence check for SVI'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Initial learning rate for SVI'
    )
    parser.add_argument(
        '--learning-rate-decay',
        type=float,
        default=0.75,
        help='Learning rate decay exponent (kappa) for SVI'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./vi_results',
        help='Directory to save results'
    )

    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None = true random)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress during training'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print detailed debug information'
    )

    return parser.parse_args()


def main():
    """Main function demonstrating the VI/SVI workflow."""
    # =========================================================================
    # STEP 1: Parse Arguments
    # =========================================================================
    args = parse_args()

    use_svi = args.method.lower() == 'svi'
    method_name = "STOCHASTIC VARIATIONAL INFERENCE" if use_svi else "VARIATIONAL INFERENCE"

    print("=" * 80)
    print(f"{method_name} - QUICK REFERENCE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Method:       {'SVI (Stochastic)' if use_svi else 'VI (Batch)'}")
    print(f"  Data:         {args.data}")
    print(f"  n_factors:    {args.n_factors}")
    if use_svi:
        print(f"  batch_size:   {args.batch_size}")
        print(f"  max_epochs:   {args.max_epochs}")
        print(f"  learning_rate:{args.learning_rate}")
    else:
        print(f"  max_iter:     {args.max_iter}")
    print(f"  label_column: {args.label_column}")
    print(f"  aux_columns:  {args.aux_columns}")
    print(f"  random_seed:  {args.seed if args.seed else 'None (random)'}")
    print(f"  output_dir:   {args.output_dir}")

    # =========================================================================
    # STEP 2: Import Modules
    # =========================================================================
    from VariationalInference.vi import VI
    from VariationalInference.svi import SVI
    from VariationalInference.data_loader import DataLoader
    from VariationalInference.utils import (
        compute_metrics, save_results, print_model_summary
    )

    # =========================================================================
    # STEP 3: Load and Preprocess Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Loading and Preprocessing Data")
    print("=" * 80)

    loader = DataLoader(
        data_path=args.data,
        gene_annotation_path=args.gene_annotation,
        verbose=args.verbose
    )

    data = loader.load_and_preprocess(
        label_column=args.label_column,
        aux_columns=args.aux_columns,
        train_ratio=0.7,
        val_ratio=0.15,
        stratify_by=args.label_column,
        min_cells_expressing=0.02,
        layer=args.layer,
        convert_to_ensembl=True,
        filter_protein_coding=args.gene_annotation is not None,
        random_state=args.seed
    )

    # Unpack data
    X_train, X_aux_train, y_train = data['train']
    X_val, X_aux_val, y_val = data['val']
    X_test, X_aux_test, y_test = data['test']
    gene_list = data['gene_list']
    splits = data['splits']

    print(f"\nData Summary:")
    print(f"  Genes:          {len(gene_list)}")
    print(f"  Training cells: {len(splits['train'])}")
    print(f"  Validation:     {len(splits['val'])}")
    print(f"  Test:           {len(splits['test'])}")
    print(f"  Aux features:   {X_aux_train.shape[1]}")
    print(f"  Label dist:     {np.bincount(y_train)}")

    # =========================================================================
    # STEP 4: Train Model (VI or SVI)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"Training {'SVI' if use_svi else 'VI'} Model")
    print("=" * 80)

    if use_svi:
        # Create SVI model for stochastic training
        model = SVI(
            n_factors=args.n_factors,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay,
            alpha_theta=0.5,   # Loose prior on theta
            alpha_beta=2.0,    # Tight prior on beta
            alpha_xi=2.0,
            lambda_xi=2.0,
            sigma_v=2.0,       # Regularization for classification weights
            sigma_gamma=1.0,   # Regularization for auxiliary effects
            random_state=args.seed
        )

        # Train with SVI
        model.fit(
            X=X_train,
            y=y_train,
            X_aux=X_aux_train,
            max_epochs=args.max_epochs,
            tol=10.0,
            rel_tol=2e-4,
            elbo_freq=10,
            min_epochs=args.min_epochs,
            patience=args.patience,
            verbose=True,
            debug=args.debug
        )
    else:
        # Create batch VI model
        model = VI(
            n_factors=args.n_factors,
            alpha_theta=0.5,   # Loose prior on theta
            alpha_beta=2.0,    # Tight prior on beta
            alpha_xi=2.0,
            lambda_xi=2.0,
            sigma_v=2.0,       # Regularization for classification weights
            sigma_gamma=1.0,   # Regularization for auxiliary effects
            random_state=args.seed  # None = random, or pass seed for reproducibility
        )

        # Train with adaptive damping
        model.fit(
            X=X_train,
            y=y_train,
            X_aux=X_aux_train,
            max_iter=args.max_iter,
            tol=10.0,
            rel_tol=2e-4,
            elbo_freq=10,
            min_iter=args.min_iter,
            patience=args.patience,
            verbose=True,
            theta_damping=0.8,
            beta_damping=0.8,
            v_damping=0.7,
            gamma_damping=0.7,
            xi_damping=0.9,
            eta_damping=0.9,
            debug=args.debug
        )

    print("\nTraining complete!")
    print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}")
    print(f"  Iterations: {model.elbo_history_[-1][0] + 1}")
    print(f"  Time:       {model.training_time_:.2f}s")

    # =========================================================================
    # STEP 5: Evaluate on Validation Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Validation Set Evaluation")
    print("=" * 80)

    y_val_proba = model.predict_proba(X_val, X_aux_val, max_iter=100, verbose=args.verbose)
    y_val_pred = (y_val_proba.ravel() > 0.5).astype(int)

    val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba.ravel())
    print(f"\nValidation Results:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    if 'auc' in val_metrics:
        print(f"  AUC:       {val_metrics['auc']:.4f}")
    print(f"  F1:        {val_metrics['f1']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")

    # Infer theta for validation set (for downstream analysis)
    E_theta_val, _, _ = model.infer_theta(
        X_val, X_aux_val, max_iter=100, tol=1e-4, verbose=args.verbose
    )

    # =========================================================================
    # STEP 6: Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)

    y_test_proba = model.predict_proba(X_test, X_aux_test, max_iter=100, verbose=args.verbose)
    y_test_pred = (y_test_proba.ravel() > 0.5).astype(int)

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba.ravel())
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    if 'auc' in test_metrics:
        print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")

    # =========================================================================
    # STEP 7: Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    prefix = 'svi' if use_svi else 'vi'
    saved_files = save_results(
        model=model,
        output_dir=output_dir,
        gene_list=gene_list,
        splits=splits,
        prefix=prefix,
        compress=True
    )

    # Save additional files

    # Validation theta
    theta_val_df = pd.DataFrame(
        E_theta_val,
        index=splits['val'],
        columns=[f"GP{k+1}" for k in range(model.d)]
    )
    theta_val_df.to_csv(output_dir / f'{prefix}_theta_val.csv.gz', compression='gzip')
    print(f"Saved validation theta to {output_dir / f'{prefix}_theta_val.csv.gz'}")

    # Test theta
    E_theta_test, _, _ = model.infer_theta(X_test, X_aux_test, max_iter=100, tol=1e-4)
    theta_test_df = pd.DataFrame(
        E_theta_test,
        index=splits['test'],
        columns=[f"GP{k+1}" for k in range(model.d)]
    )
    theta_test_df.to_csv(output_dir / f'{prefix}_theta_test.csv.gz', compression='gzip')
    print(f"Saved test theta to {output_dir / f'{prefix}_theta_test.csv.gz'}")

    # Predictions
    val_pred_df = pd.DataFrame({
        'cell_id': splits['val'],
        'true_label': y_val,
        'pred_prob': y_val_proba.ravel(),
        'pred_label': y_val_pred
    })
    val_pred_df.to_csv(output_dir / f'{prefix}_val_predictions.csv.gz', compression='gzip', index=False)

    test_pred_df = pd.DataFrame({
        'cell_id': splits['test'],
        'true_label': y_test,
        'pred_prob': y_test_proba.ravel(),
        'pred_label': y_test_pred
    })
    test_pred_df.to_csv(output_dir / f'{prefix}_test_predictions.csv.gz', compression='gzip', index=False)
    print(f"Saved predictions to {output_dir}")

    # =========================================================================
    # STEP 8: Model Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Model Summary")
    print("=" * 80)

    print_model_summary(model, gene_list)

    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)

    print(f"\nSaved files:")
    for name, path in saved_files.items():
        print(f"  - {path}")
    print(f"  - {output_dir / f'{prefix}_theta_val.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_theta_test.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_val_predictions.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_test_predictions.csv.gz'}")

    print(f"\nNext steps:")
    print(f"  1. Load gene programs: pd.read_csv('{prefix}_gene_programs.csv.gz')")
    print("  2. Identify top genes per program and run pathway enrichment")
    print("  3. Load theta matrices to analyze sample-level program activity")
    print("  4. Correlate theta values with phenotypes/outcomes")


if __name__ == '__main__':
    main()
