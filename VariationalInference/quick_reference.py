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

    # With profiling enabled (for performance analysis)
    python quick_reference.py \
        --data /path/to/data.h5ad \
        --n-factors 50 \
        --profile \
        --profile-sort tottime \
        --profile-lines 30

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
from scipy.special import expit


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
        help='Path to h5ad or CSV file (auto-detects simulated CSV if "simulated" in filename)'
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
        default=10,
        help='Early stopping patience (increased for small datasets)'
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
    parser.add_argument(
        '--learning-rate-min',
        type=float,
        default=1e-4,
        help='Minimum learning rate for SVI to prevent stagnation'
    )
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=5,
        help='Number of epochs for learning rate warmup in SVI'
    )
    parser.add_argument(
        '--regression-weight',
        type=float,
        default=1.0,
        help='Weight for classification objective in SVI (higher=more focus on classification)'
    )

    # Hyperparameter options (Priors & Regularization)
    parser.add_argument(
        '--alpha-theta',
        type=float,
        default=None,
        help='Gamma prior shape for theta (gene programs). None=auto (1.0 for N<200, 0.5 else)'
    )
    parser.add_argument(
        '--alpha-beta',
        type=float,
        default=2.0,
        help='Gamma prior shape for beta (gene-program loadings)'
    )
    parser.add_argument(
        '--alpha-xi',
        type=float,
        default=2.0,
        help='Gamma prior shape for xi (auxiliary precision)'
    )
    parser.add_argument(
        '--alpha-eta',
        type=float,
        default=2.0,
        help='Gamma prior shape for eta (gene-specific precision)'
    )
    parser.add_argument(
        '--lambda-xi',
        type=float,
        default=2.0,
        help='Gamma prior rate for xi'
    )
    parser.add_argument(
        '--lambda-eta',
        type=float,
        default=2.0,
        help='Gamma prior rate for eta'
    )
    parser.add_argument(
        '--sigma-v',
        type=float,
        default=None,
        help='Gaussian prior std for v (classification weights). None=auto (1.0 for N<200, 2.0 else)'
    )
    parser.add_argument(
        '--sigma-gamma',
        type=float,
        default=1.0,
        help='Gaussian prior std for gamma (auxiliary effects)'
    )
    parser.add_argument(
        '--pi-v',
        type=float,
        default=0.5,
        help='Spike-and-slab prior probability for v (0=all spike, 1=all slab)'
    )
    parser.add_argument(
        '--pi-beta',
        type=float,
        default=0.5,
        help='Spike-and-slab prior probability for beta'
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
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable cProfile profiling to measure function execution times and find bottlenecks'
    )
    parser.add_argument(
        '--profile-sort',
        type=str,
        default='cumulative',
        choices=['cumulative', 'tottime', 'calls', 'ncalls', 'filename'],
        help='Sort order for profile output (cumulative=time including subcalls, tottime=time excluding subcalls)'
    )
    parser.add_argument(
        '--profile-lines',
        type=int,
        default=50,
        help='Number of lines to show in profile summary'
    )

    return parser.parse_args()


def main():
    """Main function demonstrating the VI/SVI workflow."""
    import cProfile
    import pstats
    import io

    # =========================================================================
    # STEP 1: Parse Arguments
    # =========================================================================
    args = parse_args()

    # Initialize profiler if requested
    profiler = None
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        print("\n[PROFILER] Profiling enabled - measuring function execution times...")

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
    # STEP 3.5: Set Hyperparameters
    # =========================================================================
    n_train = X_train.shape[0]
    
    # Auto-configure hyperparameters if not provided
    if args.alpha_theta is None:
        alpha_theta = 1.0 if n_train < 200 else 0.5
    else:
        alpha_theta = args.alpha_theta
    
    if args.sigma_v is None:
        sigma_v = 1.0 if n_train < 200 else 2.0
    else:
        sigma_v = args.sigma_v
    
    print(f"\nModel Hyperparameters:")
    print(f"  alpha_theta:  {alpha_theta:.4f}")
    print(f"  alpha_beta:   {args.alpha_beta:.4f}")
    print(f"  alpha_xi:     {args.alpha_xi:.4f}")
    print(f"  alpha_eta:    {args.alpha_eta:.4f}")
    print(f"  lambda_xi:    {args.lambda_xi:.4f}")
    print(f"  lambda_eta:   {args.lambda_eta:.4f}")
    print(f"  sigma_v:      {sigma_v:.4f}")
    print(f"  sigma_gamma:  {args.sigma_gamma:.4f}")
    print(f"  pi_v:         {args.pi_v:.4f}")
    print(f"  pi_beta:      {args.pi_beta:.4f}")

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
            learning_rate_min=args.learning_rate_min,
            warmup_epochs=args.warmup_epochs,
            regression_weight=args.regression_weight,
            alpha_theta=alpha_theta,
            alpha_beta=args.alpha_beta,
            alpha_xi=args.alpha_xi,
            lambda_xi=args.lambda_xi,
            alpha_eta=args.alpha_eta,
            lambda_eta=args.lambda_eta,
            sigma_v=sigma_v,
            sigma_gamma=args.sigma_gamma,
            pi_v=args.pi_v,
            pi_beta=args.pi_beta,
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
            alpha_theta=alpha_theta,
            alpha_beta=args.alpha_beta,
            alpha_xi=args.alpha_xi,
            lambda_xi=args.lambda_xi,
            alpha_eta=args.alpha_eta,
            lambda_eta=args.lambda_eta,
            sigma_v=sigma_v,
            sigma_gamma=args.sigma_gamma,
            pi_v=args.pi_v,
            pi_beta=args.pi_beta,
            random_state=args.seed
        )

        # Train with adaptive damping
        # Use higher damping for small datasets to prevent oscillation
        n_train = X_train.shape[0]
        if n_train < 200:
            # Small dataset: conservative damping
            theta_damp, beta_damp, v_damp = 0.95, 0.95, 0.90
            gamma_damp, xi_damp, eta_damp = 0.90, 0.95, 0.95
        else:
            # Large dataset: standard damping
            theta_damp, beta_damp, v_damp = 0.80, 0.80, 0.70
            gamma_damp, xi_damp, eta_damp = 0.70, 0.90, 0.90
        
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
            theta_damping=theta_damp,
            beta_damping=beta_damp,
            v_damping=v_damp,
            gamma_damping=gamma_damp,
            xi_damping=xi_damp,
            eta_damping=eta_damp,
            debug=args.debug
        )

    print("\nTraining complete!")
    print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}")
    print(f"  Iterations: {model.elbo_history_[-1][0] + 1}")
    print(f"  Time:       {model.training_time_:.2f}s")

    # =========================================================================
    # STEP 4.5: Evaluate on Training Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training Set Evaluation")
    print("=" * 80)

    # For training set, we can use the E_theta computed during training
    E_theta_train = model.E_theta
    linear_pred_train = E_theta_train @ model.E_v.T + X_aux_train @ model.E_gamma.T
    y_train_proba = expit(linear_pred_train)
    if args.verbose:
        print(f"Predicted probabilities: min={y_train_proba.min():.4f}, max={y_train_proba.max():.4f}")
    y_train_pred = (y_train_proba.ravel() > 0.5).astype(int)

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba.ravel())
    print(f"\nTraining Results:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    if 'auc' in train_metrics:
        print(f"  AUC:       {train_metrics['auc']:.4f}")
    print(f"  F1:        {train_metrics['f1']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")

    # =========================================================================
    # STEP 5: Evaluate on Validation Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Validation Set Evaluation")
    print("=" * 80)

    # Infer theta once and compute probabilities from it (avoid duplicate theta calculation)
    E_theta_val, _, _ = model.infer_theta(
        X_val, X_aux_val, max_iter=100, tol=1e-4, verbose=args.verbose
    )
    # Compute probabilities using inferred theta
    linear_pred_val = E_theta_val @ model.E_v.T + X_aux_val @ model.E_gamma.T
    y_val_proba = expit(linear_pred_val)
    if args.verbose:
        print(f"Predicted probabilities: min={y_val_proba.min():.4f}, max={y_val_proba.max():.4f}")
    y_val_pred = (y_val_proba.ravel() > 0.5).astype(int)

    val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba.ravel())
    print(f"\nValidation Results (before calibration):")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    if 'auc' in val_metrics:
        print(f"  AUC:       {val_metrics['auc']:.4f}")
    print(f"  F1:        {val_metrics['f1']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")

    # =========================================================================
    # STEP 5.5: Fit Calibration
    # =========================================================================
    print("\n" + "=" * 80)
    print("Fitting Calibration")
    print("=" * 80)

    model.fit_calibration(
        X_val, y_val, X_aux_val,
        method='platt',
        optimize_threshold=True,
        threshold_metric='f1',
        verbose=True
    )

    # Re-evaluate validation with calibration
    y_val_proba_cal = model._apply_calibration(y_val_proba)
    y_val_pred_cal = (y_val_proba_cal.ravel() >= model.optimal_threshold_).astype(int)

    val_metrics_cal = compute_metrics(y_val, y_val_pred_cal, y_val_proba_cal.ravel())
    print(f"\nValidation Results (after calibration):")
    print(f"  Accuracy:  {val_metrics_cal['accuracy']:.4f}")
    if 'auc' in val_metrics_cal:
        print(f"  AUC:       {val_metrics_cal['auc']:.4f}")
    print(f"  F1:        {val_metrics_cal['f1']:.4f}")
    print(f"  Precision: {val_metrics_cal['precision']:.4f}")
    print(f"  Recall:    {val_metrics_cal['recall']:.4f}")

    # Update validation predictions for saving
    y_val_proba = y_val_proba_cal
    y_val_pred = y_val_pred_cal

    # Re-evaluate training with calibration
    y_train_proba_cal = model._apply_calibration(expit(linear_pred_train))
    y_train_pred_cal = (y_train_proba_cal.ravel() >= model.optimal_threshold_).astype(int)

    train_metrics_cal = compute_metrics(y_train, y_train_pred_cal, y_train_proba_cal.ravel())
    print(f"\nTraining Results (after calibration):")
    print(f"  Accuracy:  {train_metrics_cal['accuracy']:.4f}")
    if 'auc' in train_metrics_cal:
        print(f"  AUC:       {train_metrics_cal['auc']:.4f}")
    print(f"  F1:        {train_metrics_cal['f1']:.4f}")
    print(f"  Precision: {train_metrics_cal['precision']:.4f}")
    print(f"  Recall:    {train_metrics_cal['recall']:.4f}")

    # Update training predictions for saving
    y_train_proba = y_train_proba_cal
    y_train_pred = y_train_pred_cal

    # =========================================================================
    # STEP 6: Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)

    # Infer theta once and compute probabilities from it (avoid duplicate theta calculation)
    E_theta_test, _, _ = model.infer_theta(
        X_test, X_aux_test, max_iter=100, tol=1e-4, verbose=args.verbose
    )
    # Compute raw probabilities using inferred theta
    linear_pred_test = E_theta_test @ model.E_v.T + X_aux_test @ model.E_gamma.T
    y_test_proba_raw = expit(linear_pred_test)
    if args.verbose:
        print(f"Raw predicted probabilities: min={y_test_proba_raw.min():.4f}, max={y_test_proba_raw.max():.4f}")

    # Apply calibration
    y_test_proba = model._apply_calibration(y_test_proba_raw)
    y_test_pred = (y_test_proba.ravel() >= model.optimal_threshold_).astype(int)

    if args.verbose:
        print(f"Calibrated probabilities: min={y_test_proba.min():.4f}, max={y_test_proba.max():.4f}")
        print(f"Using threshold: {model.optimal_threshold_:.4f}")

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba.ravel())
    print(f"\nTest Results (with calibration):")
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

    # Test theta (already computed above during evaluation)
    theta_test_df = pd.DataFrame(
        E_theta_test,
        index=splits['test'],
        columns=[f"GP{k+1}" for k in range(model.d)]
    )
    theta_test_df.to_csv(output_dir / f'{prefix}_theta_test.csv.gz', compression='gzip')
    print(f"Saved test theta to {output_dir / f'{prefix}_theta_test.csv.gz'}")

    # Predictions
    train_pred_df = pd.DataFrame({
        'cell_id': splits['train'],
        'true_label': y_train,
        'pred_prob': y_train_proba.ravel(),
        'pred_label': y_train_pred
    })
    train_pred_df.to_csv(output_dir / f'{prefix}_train_predictions.csv.gz', compression='gzip', index=False)

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
    # Profiling Output (if enabled)
    # =========================================================================
    if profiler is not None:
        profiler.disable()

        # Save raw profile data for external analysis (e.g., snakeviz, pstats)
        profile_path = output_dir / f'{prefix}_profile.prof'
        profiler.dump_stats(str(profile_path))
        print(f"\n[PROFILER] Raw profile data saved to: {profile_path}")
        print(f"[PROFILER] Analyze with: python -m pstats {profile_path}")
        print(f"[PROFILER] Or visualize with: snakeviz {profile_path}")

        # Print profile summary to console
        print("\n" + "=" * 80)
        print(f"PROFILE SUMMARY (sorted by {args.profile_sort}, top {args.profile_lines} functions)")
        print("=" * 80)

        # Create stats object and print to string buffer
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats(args.profile_sort)
        stats.print_stats(args.profile_lines)
        print(stream.getvalue())

        # Also save text summary to file
        profile_txt_path = output_dir / f'{prefix}_profile_summary.txt'
        stream_file = io.StringIO()
        stats_file = pstats.Stats(profiler, stream=stream_file)
        stats_file.strip_dirs()
        stats_file.sort_stats(args.profile_sort)
        stats_file.print_stats()  # Full output to file
        with open(profile_txt_path, 'w') as f:
            f.write(f"Profile Summary (sorted by {args.profile_sort})\n")
            f.write("=" * 80 + "\n")
            f.write(stream_file.getvalue())
        print(f"[PROFILER] Full profile summary saved to: {profile_txt_path}")

        # Print key bottleneck analysis
        print("\n" + "-" * 80)
        print("TOP BOTTLENECKS BY CUMULATIVE TIME (time including subcalls):")
        print("-" * 80)
        stream_cum = io.StringIO()
        stats_cum = pstats.Stats(profiler, stream=stream_cum)
        stats_cum.strip_dirs()
        stats_cum.sort_stats('cumulative')
        stats_cum.print_stats(15)
        print(stream_cum.getvalue())

        print("-" * 80)
        print("TOP BOTTLENECKS BY TOTAL TIME (time excluding subcalls):")
        print("-" * 80)
        stream_tot = io.StringIO()
        stats_tot = pstats.Stats(profiler, stream=stream_tot)
        stats_tot.strip_dirs()
        stats_tot.sort_stats('tottime')
        stats_tot.print_stats(15)
        print(stream_tot.getvalue())

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
    print(f"  - {output_dir / f'{prefix}_train_predictions.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_val_predictions.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_test_predictions.csv.gz'}")

    print(f"\nNext steps:")
    print(f"  1. Load gene programs: pd.read_csv('{prefix}_gene_programs.csv.gz')")
    print("  2. Identify top genes per program and run pathway enrichment")
    print("  3. Load theta matrices to analyze sample-level program activity")
    print("  4. Correlate theta values with phenotypes/outcomes")


if __name__ == '__main__':
    main()
