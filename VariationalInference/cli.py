#!/usr/bin/env python
"""
Command-Line Interface for Variational Inference (CAVI)
========================================================

This module provides a command-line interface for running VI experiments
on single-cell RNA-seq data.

Usage:
    # Basic usage
    python -m VariationalInference.cli train --data /path/to/data.h5ad --n-factors 50

    # Full example
    python -m VariationalInference.cli train \
        --data /path/to/data.h5ad \
        --n-factors 50 \
        --max-iter 200 \
        --label-column t2dm \
        --aux-columns Sex \
        --output-dir ./results \
        --verbose

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
        description='Coordinate Ascent Variational Inference for Single-Cell Analysis',
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
        '--a',
        type=float,
        default=0.3,
        help='Gamma shape prior for theta (cell loadings). scHPF default 0.3.'
    )
    train_parser.add_argument(
        '--c',
        type=float,
        default=0.3,
        help='Gamma shape prior for beta (gene loadings). scHPF default 0.3.'
    )
    train_parser.add_argument(
        '--sigma-v',
        type=float,
        default=1.0,
        help='Gaussian prior std for v (classification weights). Used when --v-prior=normal.'
    )
    train_parser.add_argument(
        '--b-v',
        type=float,
        default=1.0,
        help='Laplace prior scale for v (classification weights). Used when --v-prior=laplace. '
             'Smaller b_v = stronger sparsity.'
    )
    train_parser.add_argument(
        '--v-prior',
        type=str,
        default='normal',
        choices=['normal', 'laplace'],
        help="Prior distribution for v: 'normal' (Gaussian) or 'laplace' (Bayesian Lasso)."
    )
    train_parser.add_argument(
        '--sigma-gamma',
        type=float,
        default=1.0,
        help='Gaussian prior std for gamma (auxiliary effects)'
    )

    # Training parameters
    train_parser.add_argument(
        '--max-iter',
        type=int,
        default=600,
        help='Maximum CAVI iterations'
    )
    train_parser.add_argument(
        '--tol',
        type=float,
        default=0.001,
        help='Convergence tolerance (percent change in loss)'
    )
    train_parser.add_argument(
        '--v-warmup',
        type=int,
        default=50,
        help='Iterations before regression (v/gamma) updates begin'
    )
    train_parser.add_argument(
        '--check-freq',
        type=int,
        default=5,
        help='Check convergence / compute held-out LL every N iterations'
    )

    train_parser.add_argument(
        '--regression-weight',
        type=float,
        default=1.0,
        help='Weight for classification objective (higher=more focus on classification).'
    )

    train_parser.add_argument(
        '--early-stopping',
        type=str,
        choices=['heldout_ll', 'elbo', 'none'],
        default='heldout_ll',
        help='Early stopping criterion: heldout_ll (default, stop on held-out LL), '
             'elbo (stop on ELBO convergence), none (disable early stopping)'
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
        '--profile',
        action='store_true',
        help='Enable cProfile profiling to measure function execution times and find bottlenecks'
    )
    train_parser.add_argument(
        '--profile-sort',
        type=str,
        default='cumulative',
        choices=['cumulative', 'tottime', 'calls', 'ncalls', 'filename'],
        help='Sort order for profile output (cumulative=time including subcalls, tottime=time excluding subcalls)'
    )
    train_parser.add_argument(
        '--profile-lines',
        type=int,
        default=50,
        help='Number of lines to show in profile summary'
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
    import cProfile
    import pstats
    import io
    from .data_loader import DataLoader
    from .utils import save_results, compute_metrics, print_model_summary
    from .vi_cavi import CAVI as ModelClass

    # Initialize profiler if requested
    profiler = None
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        print("\n[PROFILER] Profiling enabled - measuring function execution times...")

    print("=" * 60)
    print(f"COORDINATE ASCENT VI TRAINING")
    print("=" * 60)

    # Print configuration
    print(f"\nData: {args.data}")
    print(f"n_factors: {args.n_factors}")
    print(f"max_iter: {args.max_iter}")
    print(f"label_column: {args.label_column}")
    print(f"aux_columns: {args.aux_columns}")
    print(f"random_state: {args.seed if args.seed else 'None (random)'}")

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
        label_column=args.label_column,
        aux_columns=args.aux_columns,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stratify_by=args.label_column,
        min_cells_expressing=args.min_cells,
        layer=args.layer,
        convert_to_ensembl=not args.no_ensembl,
        filter_protein_coding=not args.no_protein_coding,
        random_state=args.seed
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
    print("Training VI (CAVI) model...")
    print("-" * 40)

    model_kwargs = dict(
        n_factors=args.n_factors,
        a=args.a,
        ap=1.0,
        c=args.c,
        cp=1.0,
        sigma_v=args.sigma_v,
        b_v=args.b_v,
        v_prior=args.v_prior,
        sigma_gamma=args.sigma_gamma,
        regression_weight=args.regression_weight,
        random_state=args.seed,
    )

    model = ModelClass(**model_kwargs)

    fit_kwargs = dict(
        X_train=X_train,
        y_train=y_train,
        X_aux_train=X_aux_train,
        X_val=X_val,
        y_val=y_val,
        X_aux_val=X_aux_val,
        max_iter=args.max_iter,
        check_freq=args.check_freq,
        tol=args.tol,
        v_warmup=args.v_warmup,
        verbose=args.verbose,
        early_stopping=args.early_stopping,
    )

    model.fit(**fit_kwargs)

    # Evaluate on validation set
    print("\n" + "-" * 40)
    print("Evaluating on validation set...")
    print("-" * 40)

    y_val_proba = model.predict_proba(X_val, X_aux_val, n_iter=20)
    y_val_pred = (y_val_proba.ravel() > 0.5).astype(int)

    val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba.ravel())
    print(f"\nValidation Metrics:")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    if 'auc' in val_metrics:
        print(f"  AUC:      {val_metrics['auc']:.4f}")
    print(f"  F1:       {val_metrics['f1']:.4f}")

    # Evaluate on test set
    print("\n" + "-" * 40)
    print("Evaluating on test set...")
    print("-" * 40)

    y_test_proba = model.predict_proba(X_test, X_aux_test, n_iter=20)
    y_test_pred = (y_test_proba.ravel() > 0.5).astype(int)

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba.ravel())
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    if 'auc' in test_metrics:
        print(f"  AUC:      {test_metrics['auc']:.4f}")
    print(f"  F1:       {test_metrics['f1']:.4f}")

    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)

    output_dir = Path(args.output_dir)
    label_cols = args.label_column if isinstance(args.label_column, list) else [args.label_column]
    saved_files = save_results(
        model=model,
        output_dir=output_dir,
        gene_list=data['gene_list'],
        splits=data['splits'],
        prefix=args.prefix,
        compress=not args.no_compress,
        label_columns=label_cols,
        aux_columns=getattr(args, 'aux_columns', None),
        val_test_data={
            'X_val': X_val, 'X_aux_val': X_aux_val,
            'X_test': X_test, 'X_aux_test': X_aux_test,
        },
        cell_metadata=data.get('cell_metadata'),
    )

    # Print model summary
    if args.verbose:
        print_model_summary(model, data['gene_list'])

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    # Finalize profiling if enabled
    if profiler is not None:
        profiler.disable()

        # Save raw profile data for external analysis (e.g., snakeviz, pstats)
        profile_path = output_dir / f'{args.prefix}_profile.prof'
        profiler.dump_stats(str(profile_path))
        print(f"\n[PROFILER] Raw profile data saved to: {profile_path}")
        print(f"[PROFILER] Analyze with: python -m pstats {profile_path}")
        print(f"[PROFILER] Or visualize with: snakeviz {profile_path}")

        # Print profile summary to console
        print("\n" + "=" * 60)
        print(f"PROFILE SUMMARY (sorted by {args.profile_sort}, top {args.profile_lines} functions)")
        print("=" * 60)

        # Create stats object and print to string buffer
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats(args.profile_sort)
        stats.print_stats(args.profile_lines)
        print(stream.getvalue())

        # Also save text summary to file
        profile_txt_path = output_dir / f'{args.prefix}_profile_summary.txt'
        stream_file = io.StringIO()
        stats_file = pstats.Stats(profiler, stream=stream_file)
        stats_file.strip_dirs()
        stats_file.sort_stats(args.profile_sort)
        stats_file.print_stats()  # Full output to file
        with open(profile_txt_path, 'w') as f:
            f.write(f"Profile Summary (sorted by {args.profile_sort})\n")
            f.write("=" * 60 + "\n")
            f.write(stream_file.getvalue())
        print(f"[PROFILER] Full profile summary saved to: {profile_txt_path}")

        # Print key bottleneck analysis
        print("\n" + "-" * 60)
        print("TOP BOTTLENECKS BY CUMULATIVE TIME (time including subcalls):")
        print("-" * 60)
        stream_cum = io.StringIO()
        stats_cum = pstats.Stats(profiler, stream=stream_cum)
        stats_cum.strip_dirs()
        stats_cum.sort_stats('cumulative')
        stats_cum.print_stats(15)
        print(stream_cum.getvalue())

        print("-" * 60)
        print("TOP BOTTLENECKS BY TOTAL TIME (time excluding subcalls):")
        print("-" * 60)
        stream_tot = io.StringIO()
        stats_tot = pstats.Stats(profiler, stream=stream_tot)
        stats_tot.strip_dirs()
        stats_tot.sort_stats('tottime')
        stats_tot.print_stats(15)
        print(stream_tot.getvalue())

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
    proba = model.predict_proba(X, X_aux, n_iter=20)

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
