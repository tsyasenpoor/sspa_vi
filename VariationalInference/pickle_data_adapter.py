#!/usr/bin/env python
"""
Pickle Data Adapter for Quick Reference
========================================

This helper script allows you to use quick_reference.py with pre-processed
pickle files (df.pkl and features.pkl) instead of h5ad files.

It creates a temporary h5ad-like structure that the DataLoader can use,
bypassing the normal data loading pipeline.

USAGE:
------
    python pickle_data_adapter.py \
        --df /path/to/df.pkl \
        --features /path/to/features.pkl \
        --label-column t2dm \
        --n-factors 10 \
        --method svi \
        --batch-size 128 \
        --max-epochs 128 \
        --learning-rate 0.5 \
        --output-dir ./results/bcell-svi \
        --verbose

This script will:
1. Load df.pkl (gene expression matrix)
2. Load features.pkl (labels)
3. Create train/val/test splits
4. Run the VI/SVI model directly
5. Save results

The main quick_reference.py remains unchanged and can still be used
for h5ad files with full preprocessing.
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from scipy.special import expit


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='VI/SVI with Pickle Data Files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--df',
        type=str,
        required=True,
        help='Path to df.pkl file with gene expression matrix'
    )
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to features.pkl file with labels'
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
        help='Inference method: vi (batch) or svi (stochastic)'
    )

    # Data options
    parser.add_argument(
        '--label-column',
        type=str,
        default='t2dm',
        help='Column name in features for classification labels'
    )
    parser.add_argument(
        '--aux-columns',
        type=str,
        nargs='+',
        default=None,
        help='Column names for auxiliary features (leave empty if none)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Proportion of data for training'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Proportion of data for validation'
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
        help='Mini-batch size for SVI'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=100,
        help='Maximum epochs for SVI'
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
        help='Learning rate decay exponent for SVI'
    )
    parser.add_argument(
        '--use-spike-slab',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help='Use spike-and-slab priors (default: True). Set to False for simpler Normal/Gamma priors.'
    )
    parser.add_argument(
        '--rho-v-delay-epochs',
        type=int,
        default=0,
        help='Number of epochs to delay rho_v updates (default: 0)'
    )
    parser.add_argument(
        '--reset-lr-on-restore',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help='Reset LR multiplier when restoring parameters (default: True)'
    )
    parser.add_argument(
        '--regression-lr-multiplier',
        type=float,
        default=10.0,
        help='Learning rate multiplier for regression parameters v, gamma (default: 10.0)'
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
        help='Random seed for reproducibility'
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
        help='Enable cProfile profiling'
    )
    parser.add_argument(
        '--profile-sort',
        type=str,
        default='cumulative',
        choices=['cumulative', 'tottime', 'calls', 'ncalls', 'filename'],
        help='Sort order for profile output'
    )
    parser.add_argument(
        '--profile-lines',
        type=int,
        default=50,
        help='Number of lines to show in profile summary'
    )

    return parser.parse_args()


def load_pickle_data(
    df_path: str,
    features_path: str,
    label_column: str,
    aux_columns: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Load pickle files and prepare data splits.
    
    Parameters
    ----------
    df_path : str
        Path to df.pkl with gene expression matrix
    features_path : str
        Path to features.pkl with labels
    label_column : str
        Column name for labels
    aux_columns : list, optional
        Column names for auxiliary features
    train_ratio : float
        Proportion for training set
    val_ratio : float
        Proportion for validation set
    random_state : int, optional
        Random seed
        
    Returns
    -------
    dict
        Dictionary with train/val/test splits and metadata
    """
    print(f"Loading gene expression from: {df_path}")
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    
    print(f"Loading features from: {features_path}")
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    
    # Ensure df is a DataFrame
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    
    # Ensure features is a DataFrame
    if isinstance(features, np.ndarray):
        # If features is just an array, assume it's the labels
        features = pd.DataFrame({label_column: features})
    elif isinstance(features, pd.Series):
        features = features.to_frame()
    
    print(f"\nData shapes:")
    print(f"  Gene expression: {df.shape}")
    print(f"  Features: {features.shape}")
    
    # Check if indices match
    if len(df) != len(features):
        raise ValueError(f"Mismatch: df has {len(df)} rows, features has {len(features)} rows")
    
    # Get gene expression matrix
    X = df.values.astype(np.float64)
    n_cells, n_genes = X.shape
    
    # Get labels
    if label_column not in features.columns:
        raise ValueError(f"Label column '{label_column}' not found in features. "
                        f"Available columns: {list(features.columns)}")
    y = features[label_column].values.astype(int)
    
    # Get auxiliary features
    if aux_columns is None or len(aux_columns) == 0:
        # No auxiliary features - just intercept
        X_aux = np.zeros((n_cells, 0), dtype=np.float64)
        print(f"  No auxiliary features specified")
    else:
        missing_cols = [col for col in aux_columns if col not in features.columns]
        if missing_cols:
            raise ValueError(f"Auxiliary columns not found: {missing_cols}. "
                           f"Available: {list(features.columns)}")
        X_aux = features[aux_columns].values.astype(np.float64)
        print(f"  Auxiliary features: {X_aux.shape[1]} columns")

    # No intercept - gamma only models auxiliary variable effects
    # The model prediction is: theta @ v + X_aux @ gamma
    print(f"  Total aux features: {X_aux.shape[1]} columns")
    
    # Get gene names
    if isinstance(df.columns, pd.Index):
        gene_list = df.columns.tolist()
    else:
        gene_list = [f"Gene_{i}" for i in range(n_genes)]
    
    # Get cell IDs
    if isinstance(df.index, pd.Index):
        cell_ids = df.index.tolist()
    else:
        cell_ids = [f"Cell_{i}" for i in range(n_cells)]
    
    # Create train/val/test splits
    if random_state is not None:
        np.random.seed(random_state)
    
    # Stratified split by label
    indices_0 = np.where(y == 0)[0]
    indices_1 = np.where(y == 1)[0]
    
    np.random.shuffle(indices_0)
    np.random.shuffle(indices_1)
    
    # Calculate split sizes
    n_train_0 = int(len(indices_0) * train_ratio)
    n_val_0 = int(len(indices_0) * val_ratio)
    
    n_train_1 = int(len(indices_1) * train_ratio)
    n_val_1 = int(len(indices_1) * val_ratio)
    
    # Split
    train_idx_0 = indices_0[:n_train_0]
    val_idx_0 = indices_0[n_train_0:n_train_0+n_val_0]
    test_idx_0 = indices_0[n_train_0+n_val_0:]
    
    train_idx_1 = indices_1[:n_train_1]
    val_idx_1 = indices_1[n_train_1:n_train_1+n_val_1]
    test_idx_1 = indices_1[n_train_1+n_val_1:]
    
    # Combine
    train_idx = np.concatenate([train_idx_0, train_idx_1])
    val_idx = np.concatenate([val_idx_0, val_idx_1])
    test_idx = np.concatenate([test_idx_0, test_idx_1])
    
    # Shuffle
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    # Create splits
    splits = {
        'train': [cell_ids[i] for i in train_idx],
        'val': [cell_ids[i] for i in val_idx],
        'test': [cell_ids[i] for i in test_idx]
    }
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)} ({np.bincount(y[train_idx])})")
    print(f"  Val:   {len(val_idx)} ({np.bincount(y[val_idx])})")
    print(f"  Test:  {len(test_idx)} ({np.bincount(y[test_idx])})")
    
    return {
        'train': (X[train_idx], X_aux[train_idx], y[train_idx]),
        'val': (X[val_idx], X_aux[val_idx], y[val_idx]),
        'test': (X[test_idx], X_aux[test_idx], y[test_idx]),
        'gene_list': gene_list,
        'splits': splits
    }


def main():
    """Main function for pickle data adapter."""
    import cProfile
    import pstats
    import io
    
    # Parse arguments
    args = parse_args()
    
    # Initialize profiler if requested
    profiler = None
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        print("\n[PROFILER] Profiling enabled...")
    
    use_svi = args.method.lower() == 'svi'
    method_name = "STOCHASTIC VARIATIONAL INFERENCE" if use_svi else "VARIATIONAL INFERENCE"
    
    print("=" * 80)
    print(f"{method_name} - PICKLE DATA ADAPTER")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Method:       {'SVI (Stochastic)' if use_svi else 'VI (Batch)'}")
    print(f"  Data (df):    {args.df}")
    print(f"  Features:     {args.features}")
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
    
    # Import modules
    from VariationalInference.svi_corrected import SVI
    from VariationalInference.utils import (
        compute_metrics, save_results, print_model_summary
    )
    
    # Load data
    print("\n" + "=" * 80)
    print("Loading Pickle Data")
    print("=" * 80)
    
    data = load_pickle_data(
        df_path=args.df,
        features_path=args.features,
        label_column=args.label_column,
        aux_columns=args.aux_columns,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed
    )
    
    # Unpack data
    X_train, X_aux_train, y_train = data['train']
    X_val, X_aux_val, y_val = data['val']
    X_test, X_aux_test, y_test = data['test']
    gene_list = data['gene_list']
    splits = data['splits']
    
    print(f"\nReady for training:")
    print(f"  Genes:          {len(gene_list)}")
    print(f"  Training cells: {len(splits['train'])}")
    print(f"  Aux features:   {X_aux_train.shape[1]}")
    
    # Train model
    print("\n" + "=" * 80)
    print(f"Training {'SVI' if use_svi else 'VI'} Model")
    print("=" * 80)
    
    if use_svi:
        model = SVI(
            n_factors=args.n_factors,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            learning_rate_decay=args.learning_rate_decay,
            alpha_theta=0.5,
            alpha_beta=2.0,
            alpha_xi=2.0,
            lambda_xi=2.0,
            sigma_v=2.0,
            sigma_gamma=1.0,
            random_state=args.seed,
            use_spike_slab=args.use_spike_slab,
            rho_v_delay_epochs=args.rho_v_delay_epochs,
            reset_lr_on_restore=args.reset_lr_on_restore,
            regression_lr_multiplier=args.regression_lr_multiplier
        )
        
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
        model = VI(
            n_factors=args.n_factors,
            alpha_theta=0.5,
            alpha_beta=2.0,
            alpha_xi=2.0,
            lambda_xi=2.0,
            sigma_v=2.0,
            sigma_gamma=1.0,
            random_state=args.seed
        )
        
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

    # Training evaluation
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

    # Validation evaluation
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

    # Fit calibration on validation set
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

    # Test evaluation
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
    
    # Save results
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
    
    # Save validation theta
    theta_val_df = pd.DataFrame(
        E_theta_val,
        index=splits['val'],
        columns=[f"GP{k+1}" for k in range(model.d)]
    )
    theta_val_df.to_csv(output_dir / f'{prefix}_theta_val.csv.gz', compression='gzip')
    print(f"Saved validation theta to {output_dir / f'{prefix}_theta_val.csv.gz'}")
    
    # Save test theta (already computed above during evaluation)
    theta_test_df = pd.DataFrame(
        E_theta_test,
        index=splits['test'],
        columns=[f"GP{k+1}" for k in range(model.d)]
    )
    theta_test_df.to_csv(output_dir / f'{prefix}_theta_test.csv.gz', compression='gzip')
    print(f"Saved test theta to {output_dir / f'{prefix}_theta_test.csv.gz'}")
    
    # Save predictions
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
    
    # Model summary
    print("\n" + "=" * 80)
    print("Model Summary")
    print("=" * 80)
    print_model_summary(model, gene_list)
    
    # Profiling output
    if profiler is not None:
        profiler.disable()
        
        profile_path = output_dir / f'{prefix}_profile.prof'
        profiler.dump_stats(str(profile_path))
        print(f"\n[PROFILER] Raw profile data saved to: {profile_path}")
        
        print("\n" + "=" * 80)
        print(f"PROFILE SUMMARY (sorted by {args.profile_sort}, top {args.profile_lines})")
        print("=" * 80)
        
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats(args.profile_sort)
        stats.print_stats(args.profile_lines)
        print(stream.getvalue())
        
        profile_txt_path = output_dir / f'{prefix}_profile_summary.txt'
        stream_file = io.StringIO()
        stats_file = pstats.Stats(profiler, stream=stream_file)
        stats_file.strip_dirs()
        stats_file.sort_stats(args.profile_sort)
        stats_file.print_stats()
        with open(profile_txt_path, 'w') as f:
            f.write(f"Profile Summary (sorted by {args.profile_sort})\n")
            f.write("=" * 80 + "\n")
            f.write(stream_file.getvalue())
        print(f"[PROFILER] Full profile summary saved to: {profile_txt_path}")
    
    # Done
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


if __name__ == '__main__':
    main()
