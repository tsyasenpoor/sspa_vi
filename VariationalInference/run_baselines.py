#!/usr/bin/env python
"""
Run Baseline Classification Methods on EMTAB Data
=================================================

This script runs baseline classification methods (SVM, LR, Lasso LR, Ridge LR, 
NMF+LR variants) on the same preprocessed EMTAB data used for VI experiments.

Usage:
    python run_baselines.py \
        --data /path/to/EMTAB11349/preprocessed \
        --gene-annotation /path/to/gene_annotation.csv \
        --label-column IBD \
        --aux-columns sex_female \
        --output-dir ./results/baselines \
        --latent-dim 100 \
        --seed 42
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import argparse
import numpy as np
import random
import json
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run baseline classification methods on EMTAB data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to EMTAB preprocessed directory or h5ad file')
    parser.add_argument('--gene-annotation', type=str, default=None,
                        help='Path to gene annotation CSV for protein-coding filter')
    parser.add_argument('--label-column', type=str, default='IBD',
                        help='Column name for classification labels')
    parser.add_argument('--aux-columns', type=str, nargs='*', default=[],
                        help='Column names for auxiliary features (omit for pure expression baselines)')
    parser.add_argument('--output-dir', '-o', type=str, default='./results/baselines',
                        help='Directory to save results')
    parser.add_argument('--latent-dim', type=int, default=100,
                        help='Number of latent dimensions for NMF-based methods')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--cache-dir', type=str, default='/labs/Aguiar/SSPA_BRAY/cache',
                        help='Directory for caching preprocessed data')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize counts (library size normalization)')
    parser.add_argument('--normalize-target-sum', type=float, default=1e4,
                        help='Target library size for normalization')
    parser.add_argument('--normalize-method', type=str, default='library_size',
                        choices=['library_size', 'median_ratio'],
                        help='Normalization method')
    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=['svm', 'lr', 'lrl', 'lrr', 'mflr', 'mflrl', 'mflrr'],
                        help='Algorithms to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("BASELINE CLASSIFICATION METHODS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data:           {args.data}")
    print(f"  Label column:   {args.label_column}")
    print(f"  Aux columns:    {args.aux_columns}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Latent dim:     {args.latent_dim}")
    print(f"  Seed:           {args.seed}")
    print(f"  Algorithms:     {args.algorithms}")
    print(f"  Normalize:      {args.normalize}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load and Preprocess Data (same as VI)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Loading and Preprocessing Data")
    print("=" * 80)
    
    from VariationalInference.data_loader import DataLoader
    
    loader = DataLoader(
        data_path=args.data,
        gene_annotation_path=args.gene_annotation,
        cache_dir=args.cache_dir,
        use_cache=True,
        verbose=args.verbose
    )
    
    data = loader.load_and_preprocess(
        label_column=args.label_column,
        aux_columns=args.aux_columns,
        train_ratio=0.7,
        val_ratio=0.15,
        stratify_by=args.label_column,
        min_cells_expressing=0.02,
        layer='raw',
        convert_to_ensembl=True,
        filter_protein_coding=args.gene_annotation is not None,
        random_state=args.seed,
        normalize=args.normalize,
        normalize_target_sum=args.normalize_target_sum,
        normalize_method=args.normalize_method,
        return_sparse=False  # comp_methods expects dense arrays
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
    print(f"  Label dist (train): {np.bincount(y_train)}")
    
    # =========================================================================
    # STEP 2: Train and Evaluate Baseline Methods
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training and Evaluating Baseline Methods")
    print("=" * 80)
    
    from VariationalInference.comp_methods import train_alg, eval_alg
    
    all_results = {}
    
    for alg in args.algorithms:
        print(f"\n{'='*40}")
        print(f"Algorithm: {alg.upper()}")
        print(f"{'='*40}")
        
        try:
            # Train
            print(f"\nTraining {alg}...")
            model, nmf_obj = train_alg(
                algorithm=alg,
                x_data_train=X_train,
                x_aux_data_train=X_aux_train,
                y_data_train=y_train,
                save_path=str(output_dir),
                latent_dim=args.latent_dim
            )
            
            # Evaluate on test set
            print(f"\nEvaluating {alg} on test set...")
            eval_alg(
                model=model,
                algorithm=alg,
                x_data_test=X_test,
                x_aux_data_test=X_aux_test,
                y_data_test=y_test,
                save_path=str(output_dir),
                latent_dim=args.latent_dim,
                nmf=nmf_obj
            )
            
            # Also evaluate on validation set
            print(f"\nEvaluating {alg} on validation set...")
            val_results = eval_alg(
                model=model,
                algorithm=alg,
                x_data_test=X_val,
                x_aux_data_test=X_aux_val,
                y_data_test=y_val,
                save_path=str(output_dir / f"{alg}_val"),
                latent_dim=args.latent_dim,
                nmf=nmf_obj
            )
            
            all_results[alg] = {
                'val_accuracy': val_results['test_accuracy'],
                'val_f1': val_results['test_f1'],
                'val_auc': val_results['test_roc_auc'],
                'status': 'success'
            }
            
        except Exception as e:
            print(f"ERROR running {alg}: {e}")
            all_results[alg] = {'status': 'failed', 'error': str(e)}
    
    # =========================================================================
    # STEP 3: Save Summary Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    # Save config
    config = {
        'data': args.data,
        'gene_annotation': args.gene_annotation,
        'label_column': args.label_column,
        'aux_columns': args.aux_columns,
        'latent_dim': args.latent_dim,
        'seed': args.seed,
        'normalize': args.normalize,
        'normalize_target_sum': args.normalize_target_sum,
        'n_genes': len(gene_list),
        'n_train': len(splits['train']),
        'n_val': len(splits['val']),
        'n_test': len(splits['test']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Print summary table
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Algorithm':<10} {'Val Acc':<10} {'Val F1':<10} {'Val AUC':<10}")
    print("-" * 50)
    
    for alg, res in all_results.items():
        if res['status'] == 'success':
            print(f"{alg:<10} {res['val_accuracy']:.4f}     {res['val_f1']:.4f}     {res['val_auc']:.4f}")
        else:
            print(f"{alg:<10} FAILED: {res.get('error', 'unknown')[:30]}")
    
    print("-" * 50)
    
    # Save summary
    summary = {
        'config': config,
        'results': all_results
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")
    print("Job completed at:", datetime.now().isoformat())


if __name__ == '__main__':
    main()
