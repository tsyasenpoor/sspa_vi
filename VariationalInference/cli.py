#!/usr/bin/env python
"""
Command-Line Interface for Variational Inference
=================================================

This module provides a command-line interface for running VI experiments
on single-cell RNA-seq data.

Usage:
    # Basic usage
    python -m VariationalInference.cli train --data /path/to/data.h5ad --n-factors 50

    # With all options
    python -m VariationalInference.cli train \
        --data /path/to/data.h5ad \
        --n-factors 50 \
        --max-iter 200 \
        --label-column t2dm \
        --aux-columns Sex \
        --output-dir ./results \
        --verbose

    # Using a config file
    python -m VariationalInference.cli train \
        --data /path/to/data.h5ad \
        --config config.json

    # Predict with trained model
    python -m VariationalInference.cli predict \
        --model ./results/vi_model.pkl \
        --data /path/to/new_data.h5ad \
        --output predictions.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import numpy as np


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog='VariationalInference',
        description='Variational Inference for Single-Cell Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # =========================================================================
    # Train command
    # =========================================================================
    train_parser = subparsers.add_parser(
        'train',
        help='Train a VI model on single-cell data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    train_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to h5ad file'
    )
    train_parser.add_argument(
        '--n-factors', '-k',
        type=int,
        required=True,
        help='Number of latent gene programs'
    )

    # Data options
    train_parser.add_argument(
        '--label-column',
        type=str,
        default='t2dm',
        help='Column name in adata.obs for labels'
    )
    train_parser.add_argument(
        '--aux-columns',
        type=str,
        nargs='+',
        default=None,
        help='Column names for auxiliary features (e.g., Sex batch)'
    )
    train_parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Proportion of data for training'
    )
    train_parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Proportion of data for validation'
    )
    train_parser.add_argument(
        '--min-cells',
        type=float,
        default=0.02,
        help='Minimum fraction of cells expressing each gene'
    )
    train_parser.add_argument(
        '--layer',
        type=str,
        default='raw',
        help='Which layer to use from h5ad file'
    )
    train_parser.add_argument(
        '--no-ensembl',
        action='store_true',
        help='Skip gene symbol to Ensembl conversion'
    )
    train_parser.add_argument(
        '--no-protein-coding',
        action='store_true',
        help='Skip protein-coding gene filter'
    )
    train_parser.add_argument(
        '--gene-annotation',
        type=str,
        default=None,
        help='Path to gene annotation CSV for protein-coding filter'
    )

    # Model hyperparameters
    train_parser.add_argument(
        '--alpha-theta',
        type=float,
        default=2.0,
        help='Prior shape for theta'
    )
    train_parser.add_argument(
        '--alpha-beta',
        type=float,
        default=2.0,
        help='Prior shape for beta'
    )
    train_parser.add_argument(
        '--sigma-v',
        type=float,
        default=0.2,
        help='Prior std for classification weights'
    )
    train_parser.add_argument(
        '--pi-v',
        type=float,
        default=0.2,
        help='Prior probability of v being active'
    )
    train_parser.add_argument(
        '--pi-beta',
        type=float,
        default=0.05,
        help='Prior probability of beta being active'
    )

    # Training parameters
    train_parser.add_argument(
        '--max-iter',
        type=int,
        default=200,
        help='Maximum training iterations'
    )
    train_parser.add_argument(
        '--tol',
        type=float,
        default=10.0,
        help='Absolute ELBO tolerance for convergence'
    )
    train_parser.add_argument(
        '--rel-tol',
        type=float,
        default=2e-4,
        help='Relative ELBO tolerance for convergence'
    )
    train_parser.add_argument(
        '--min-iter',
        type=int,
        default=50,
        help='Minimum iterations before checking convergence'
    )
    train_parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience'
    )

    # Output options
    train_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./vi_results',
        help='Directory to save results'
    )
    train_parser.add_argument(
        '--prefix',
        type=str,
        default='vi',
        help='Prefix for output files'
    )
    train_parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Do not compress output files'
    )

    # Other options
    train_parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file (overrides other options)'
    )
    train_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (default: None = true random)'
    )
    train_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress during training'
    )
    train_parser.add_argument(
        '--debug',
        action='store_true',
        help='Print detailed debug information'
    )
    train_parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Directory to cache preprocessed data'
    )

    # =========================================================================
    # Predict command
    # =========================================================================
    predict_parser = subparsers.add_parser(
        'predict',
        help='Make predictions with a trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    predict_parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model pickle file'
    )
    predict_parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to h5ad file with new data'
    )
    predict_parser.add_argument(
        '--output', '-o',
        type=str,
        default='predictions.csv',
        help='Output file for predictions'
    )
    predict_parser.add_argument(
        '--aux-columns',
        type=str,
        nargs='+',
        default=None,
        help='Auxiliary feature columns (must match training)'
    )
    predict_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress'
    )

    # =========================================================================
    # Info command
    # =========================================================================
    info_parser = subparsers.add_parser(
        'info',
        help='Show information about a trained model'
    )
    info_parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model pickle file'
    )

    return parser


def cmd_train(args: argparse.Namespace) -> int:
    """Execute the train command."""
    from .vi import VI
    from .data_loader import DataLoader
    from .config import VIConfig
    from .utils import save_results, compute_metrics, print_model_summary

    print("=" * 60)
    print("VARIATIONAL INFERENCE TRAINING")
    print("=" * 60)

    # Load config if provided
    if args.config:
        print(f"\nLoading config from: {args.config}")
        config = VIConfig.from_json(args.config)
    else:
        config = VIConfig(
            n_factors=args.n_factors,
            alpha_theta=args.alpha_theta,
            alpha_beta=args.alpha_beta,
            sigma_v=args.sigma_v,
            pi_v=args.pi_v,
            pi_beta=args.pi_beta,
            max_iter=args.max_iter,
            tol=args.tol,
            rel_tol=args.rel_tol,
            min_iter=args.min_iter,
            patience=args.patience,
            label_column=args.label_column,
            aux_columns=args.aux_columns,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            min_cells_expressing=args.min_cells,
            verbose=args.verbose,
            debug=args.debug,
            random_state=args.seed,
        )

    # Print configuration
    print(f"\nData: {args.data}")
    print(f"n_factors: {config.n_factors}")
    print(f"max_iter: {config.max_iter}")
    print(f"label_column: {config.label_column}")
    print(f"aux_columns: {config.aux_columns}")
    print(f"random_state: {config.random_state} {'(random)' if config.random_state is None else ''}")

    # Load and preprocess data
    print("\n" + "-" * 40)
    print("Loading and preprocessing data...")
    print("-" * 40)

    loader = DataLoader(
        data_path=args.data,
        gene_annotation_path=args.gene_annotation,
        cache_dir=args.cache_dir,
        verbose=args.verbose
    )

    data = loader.load_and_preprocess(
        label_column=config.label_column,
        aux_columns=config.aux_columns,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        stratify_by=config.label_column,
        min_cells_expressing=config.min_cells_expressing,
        layer=args.layer,
        convert_to_ensembl=not args.no_ensembl,
        filter_protein_coding=not args.no_protein_coding,
        random_state=config.random_state
    )

    X_train, X_aux_train, y_train = data['train']
    X_val, X_aux_val, y_val = data['val']
    X_test, X_aux_test, y_test = data['test']

    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} genes, {X_aux_train.shape[1]} aux features")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Label distribution (train): {np.bincount(y_train)}")

    # Create and train model
    print("\n" + "-" * 40)
    print("Training model...")
    print("-" * 40)

    model = VI(**config.model_params())
    model.fit(
        X=X_train,
        y=y_train,
        X_aux=X_aux_train,
        **config.training_params()
    )

    # Evaluate on validation set
    print("\n" + "-" * 40)
    print("Evaluating on validation set...")
    print("-" * 40)

    y_val_proba = model.predict_proba(X_val, X_aux_val, verbose=args.verbose)
    y_val_pred = (y_val_proba.ravel() > 0.5).astype(int)

    val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba.ravel())
    print(f"\nValidation Metrics:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  AUC:      {val_metrics.get('auc', 'N/A'):.4f}" if 'auc' in val_metrics else "")
    print(f"  F1:       {val_metrics['f1']:.4f}")

    # Evaluate on test set
    print("\n" + "-" * 40)
    print("Evaluating on test set...")
    print("-" * 40)

    y_test_proba = model.predict_proba(X_test, X_aux_test, verbose=args.verbose)
    y_test_pred = (y_test_proba.ravel() > 0.5).astype(int)

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba.ravel())
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  AUC:      {test_metrics.get('auc', 'N/A'):.4f}" if 'auc' in test_metrics else "")
    print(f"  F1:       {test_metrics['f1']:.4f}")

    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)

    output_dir = Path(args.output_dir)
    saved_files = save_results(
        model=model,
        output_dir=output_dir,
        gene_list=data['gene_list'],
        splits=data['splits'],
        prefix=args.prefix,
        compress=not args.no_compress
    )

    # Save config
    config_path = output_dir / f'{args.prefix}_config.json'
    config.to_json(str(config_path))
    print(f"Saved config to {config_path}")

    # Print model summary
    if args.verbose:
        print_model_summary(model, data['gene_list'])

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Execute the predict command."""
    from .utils import load_model
    from .data_loader import DataLoader
    import pandas as pd

    print("Loading model...")
    model = load_model(args.model)

    print("Loading data...")
    loader = DataLoader(data_path=args.data, verbose=args.verbose)
    loader.preprocess()

    # Get expression matrix
    X = loader.raw_df.values
    cell_ids = loader.cell_ids

    # Get auxiliary features if specified
    if args.aux_columns:
        X_aux = loader.get_auxiliary_features(args.aux_columns)
    else:
        X_aux = np.zeros((X.shape[0], 0))

    print("Making predictions...")
    proba = model.predict_proba(X, X_aux, verbose=args.verbose)

    # Save predictions
    results_df = pd.DataFrame({
        'cell_id': cell_ids,
        'pred_prob': proba.ravel(),
        'pred_label': (proba.ravel() > 0.5).astype(int)
    })
    results_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    from .utils import load_model, print_model_summary

    model = load_model(args.model)
    print_model_summary(model)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == 'train':
        return cmd_train(args)
    elif args.command == 'predict':
        return cmd_predict(args)
    elif args.command == 'info':
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
