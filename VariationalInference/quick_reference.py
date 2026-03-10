#!/usr/bin/env python
"""
Quick Reference: Stochastic Variational Inference for Single-Cell Analysis
===========================================================================

This script demonstrates the complete SVI workflow for gene program discovery
and phenotype classification from single-cell RNA-seq data.

USAGE:
------
    # Basic usage
    python quick_reference.py --data /path/to/data.h5ad --n-factors 50

    # Full example
    python quick_reference.py \
        --data /path/to/data.h5ad \
        --n-factors 50 \
        --batch-size 128 \
        --max-epochs 100 \
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
3. Train SVI model with specified parameters
4. Evaluate on validation and test sets
5. Save results (model, gene programs, predictions)

For more information:
    python quick_reference.py --help

For CLI usage:
    python -m VariationalInference.cli train --help
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to allow VariationalInference imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import argparse
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.metrics import precision_recall_curve


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Stochastic Variational Inference Quick Reference',
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

    # Data options
    parser.add_argument(
        '--label-column',
        type=str,
        nargs='+',
        default=['t2dm'],
        help='Column name(s) in adata.obs for classification labels. '
             'Multiple columns enable multi-outcome inference (e.g., --label-column severity outcome).'
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
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='/labs/Aguiar/SSPA_BRAY/cache',
        help='Directory for caching preprocessed data (default: /labs/Aguiar/SSPA_BRAY/cache)'
    )

    # Inference method selection
    parser.add_argument(
        '--method',
        type=str,
        default='svi',
        choices=['vi', 'svi'],
        help='Inference method: svi (stochastic variational inference) or vi (coordinate ascent VI)'
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        default='unmasked',
        choices=['unmasked', 'masked', 'pathway_init', 'combined'],
        help='Model mode: unmasked (standard), masked (β fixed to pathway structure), '
             'pathway_init (β initialized from pathways but free to deviate), '
             'combined (pathway-constrained + unconstrained DRGPs)'
    )
    parser.add_argument(
        '--pathway-file',
        type=str,
        default='/archive/projects/SSPA_BRAY/sspa/c2.cp.v2024.1.Hs.symbols.gmt',
        help='GMT file for pathway definitions (used in masked/pathway_init/combined modes)'
    )
    parser.add_argument(
        '--pathway-min-genes',
        type=int,
        default=1,
        help='Minimum genes per pathway (for filtering)'
    )
    parser.add_argument(
        '--pathway-max-genes',
        type=int,
        default=2000,
        help='Maximum genes per pathway (for filtering)'
    )
    parser.add_argument(
        '--n-drgps',
        type=int,
        default=50,
        help='Number of unconstrained data-driven gene programs (DRGPs) for combined mode'
    )

    # SVI Training options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Mini-batch size for SVI'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=500,
        help='Maximum epochs for SVI'
    )
    parser.add_argument(
        '--learning-rate-decay',
        type=float,
        default=0.75,
        help='Learning rate decay exponent (kappa) for SVI Robbins-Monro schedule'
    )
    parser.add_argument(
        '--learning-rate-delay',
        type=float,
        default=1.0,
        help='Learning rate delay (tau) for SVI Robbins-Monro schedule'
    )
    parser.add_argument(
        '--local-iterations',
        type=int,
        default=10,
        help='Number of local parameter iterations per batch'
    )
    parser.add_argument(
        '--regression-weight',
        type=float,
        default=1.0,
        help='Weight for classification objective (higher=more focus on classification)'
    )
    parser.add_argument(
        '--check-freq',
        type=int,
        default=5,
        help='Check convergence / compute held-out LL every N iterations/epochs'
    )
    
    # Hyperparameter options (scHPF priors — defaults work well, rarely need tuning)
    parser.add_argument(
        '--a',
        type=float,
        default=0.3,
        help='Gamma shape prior for theta (cell loadings). scHPF default 0.3.'
    )
    parser.add_argument(
        '--c',
        type=float,
        default=0.3,
        help='Gamma shape prior for beta (gene loadings). scHPF default 0.3.'
    )
    parser.add_argument(
        '--sigma-v',
        type=float,
        default=1.0,
        help='Gaussian prior std for v (classification weights).'
    )
    parser.add_argument(
        '--sigma-gamma',
        type=float,
        default=1.0,
        help='Gaussian prior std for gamma (auxiliary effects)'
    )

    # Bayesian optimization parameter names (override --a, --c when provided)
    parser.add_argument(
        '--alpha-theta',
        type=float,
        default=None,
        help='Gamma shape prior for theta (=a in scHPF). Overrides --a if set.'
    )
    parser.add_argument(
        '--alpha-beta',
        type=float,
        default=None,
        help='Gamma shape prior for beta (=c in scHPF). Overrides --c if set.'
    )
    parser.add_argument(
        '--alpha-xi',
        type=float,
        default=None,
        help='Gamma shape prior for xi (=ap in scHPF). Default 1.0.'
    )
    parser.add_argument(
        '--alpha-eta',
        type=float,
        default=None,
        help='Gamma shape prior for eta (=cp in scHPF). Default 1.0.'
    )
    parser.add_argument(
        '--lambda-xi',
        type=float,
        default=None,
        help='Rate multiplier for xi prior. Stored but not yet used by CAVI/SVI.'
    )
    parser.add_argument(
        '--lambda-eta',
        type=float,
        default=None,
        help='Rate multiplier for eta prior. Stored but not yet used by CAVI/SVI.'
    )
    parser.add_argument(
        '--pi-v',
        type=float,
        default=None,
        help='Spike-and-slab mixture weight for v. Stored but not yet used by CAVI/SVI.'
    )
    parser.add_argument(
        '--pi-beta',
        type=float,
        default=None,
        help='Spike-and-slab mixture weight for beta. Stored but not yet used by CAVI/SVI.'
    )

    # Damping parameters (VI coordinate ascent)
    parser.add_argument(
        '--theta-damping',
        type=float,
        default=None,
        help='Damping for theta updates. Stored but not yet used by CAVI/SVI.'
    )
    parser.add_argument(
        '--beta-damping',
        type=float,
        default=None,
        help='Damping for beta updates. Stored but not yet used by CAVI/SVI.'
    )
    parser.add_argument(
        '--v-damping',
        type=float,
        default=None,
        help='Damping for v updates. Stored but not yet used by CAVI/SVI.'
    )
    parser.add_argument(
        '--gamma-damping',
        type=float,
        default=None,
        help='Damping for gamma updates. Stored but not yet used by CAVI/SVI.'
    )
    parser.add_argument(
        '--xi-damping',
        type=float,
        default=None,
        help='Damping for xi updates. Stored but not yet used by CAVI/SVI.'
    )
    parser.add_argument(
        '--eta-damping',
        type=float,
        default=None,
        help='Damping for eta updates. Stored but not yet used by CAVI/SVI.'
    )

    parser.add_argument(
        '--heldout-patience',
        type=int,
        default=50,
        help='[SVI] Epochs without HO-LL improvement before stopping'
    )

    # VI (CAVI) specific training options
    parser.add_argument(
        '--max-iter',
        type=int,
        default=600,
        help='[VI only] Maximum iterations for coordinate ascent VI'
    )
    parser.add_argument(
        '--tol',
        type=float,
        default=0.001,
        help='[VI only] Convergence tolerance (percent change in loss)'
    )
    parser.add_argument(
        '--v-warmup',
        type=int,
        default=50,
        help='Iterations/epochs before regression (v/gamma) updates begin'
    )

    # Normalization options
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize counts (library size normalization + integer rounding). Recommended for overdispersed data.'
    )
    parser.add_argument(
        '--normalize-target-sum',
        type=float,
        default=1e4,
        help='Target library size for normalization'
    )
    parser.add_argument(
        '--normalize-method',
        type=str,
        default='library_size',
        choices=['library_size', 'median_ratio'],
        help='Normalization method'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./svi_results',
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
    """Main function demonstrating the SVI workflow."""
    import cProfile
    import pstats
    import io
    import random

    # =========================================================================
    # STEP 1: Parse Arguments
    # =========================================================================
    args = parse_args()

    # =========================================================================
    # STEP 1.5: Set Random Seeds for Full Reproducibility
    # =========================================================================
    # CRITICAL: Set seeds BEFORE any random operations (imports, data loading, etc.)
    # This ensures:
    #   1. NumPy random (used for train/test split, epoch shuffling, etc.)
    #   2. Python random (used by some libraries internally)
    #   3. JAX random (handled by SVI class via random_state argument)
    # are all deterministic given the same seed.
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        # JAX random is handled separately in SVI class via random_state
        print(f"[SEED] Set Python/NumPy random seed to {args.seed}")
    else:
        # Generate a random seed from system entropy and log it for reproducibility debugging
        import time
        auto_seed = int(time.time() * 1000) % (2**32)
        random.seed(auto_seed)
        np.random.seed(auto_seed)
        # Store for logging
        args._auto_seed = auto_seed
        print(f"[SEED] No seed provided, using auto-generated seed: {auto_seed}")

    # Initialize profiler if requested
    profiler = None
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        print("\n[PROFILER] Profiling enabled - measuring function execution times...")

    print("=" * 80)
    print("STOCHASTIC VARIATIONAL INFERENCE - QUICK REFERENCE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data:         {args.data}")
    print(f"  Method:       {args.method}")
    print(f"  Mode:         {args.mode}")
    print(f"  n_factors:    {args.n_factors}" + (" (may be overridden by pathway count)" if args.mode != 'unmasked' else ""))
    if args.method == 'svi':
        print(f"  batch_size:   {args.batch_size}")
        print(f"  max_epochs:   {args.max_epochs}")
    else:
        print(f"  max_iter:     {args.max_iter}")
    # Normalise label_column to list internally
    label_columns = args.label_column if isinstance(args.label_column, list) else [args.label_column]
    n_outcomes = len(label_columns)

    print(f"  label_column: {label_columns}")
    print(f"  aux_columns:  {args.aux_columns}")
    seed_display = args.seed if args.seed else f"auto ({getattr(args, '_auto_seed', 'unknown')})"
    print(f"  random_seed:  {seed_display}")
    print(f"  output_dir:   {args.output_dir}")
    if args.mode in ['masked', 'pathway_init', 'combined']:
        print(f"  pathway_file: {args.pathway_file}")
        print(f"  pathway_size: [{args.pathway_min_genes}, {args.pathway_max_genes}]")
    if args.normalize:
        print(f"  normalize:    True (target_sum={args.normalize_target_sum:.0f}, method={args.normalize_method})")

    # =========================================================================
    # STEP 2: Import Modules
    # =========================================================================
    if args.method == 'vi':
        from VariationalInference.vi_cavi import CAVI as ModelClass
        print("[METHOD] Using CAVI (coordinate ascent, full-batch)")
    else:
        from VariationalInference.svi_corrected import SVI as ModelClass
        print("[METHOD] Using SVI (stochastic, mini-batch)")
    from VariationalInference.data_loader import DataLoader
    from VariationalInference.utils import (
        compute_metrics, save_results, print_model_summary, load_pathways
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
        cache_dir=args.cache_dir,
        use_cache=True,
        verbose=args.verbose
    )

    data = loader.load_and_preprocess(
        label_column=label_columns,
        aux_columns=args.aux_columns,
        train_ratio=0.7,
        val_ratio=0.15,
        stratify_by=label_columns[0],
        min_cells_expressing=0.02,
        layer=args.layer,
        convert_to_ensembl=True,
        filter_protein_coding=args.gene_annotation is not None,
        random_state=args.seed,
        normalize=args.normalize,
        normalize_target_sum=args.normalize_target_sum,
        normalize_method=args.normalize_method
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
    if y_train.ndim == 1:
        print(f"  Label ({label_columns[0]}) distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    else:
        for k, lc in enumerate(label_columns):
            print(f"  Label ({lc}) distribution: {dict(zip(*np.unique(y_train[:, k], return_counts=True)))}")


    # =========================================================================
    # STEP 3.25: Load Pathways (for masked/pathway_init modes)
    # =========================================================================
    pathway_mask = None
    pathway_names = None
    n_factors = args.n_factors  # Default from CLI
    
    if args.mode in ['masked', 'pathway_init', 'combined']:
        print("\n" + "=" * 80)
        print(f"Loading Pathways for {args.mode.upper()} Mode")
        print("=" * 80)
        
        # Auto-detect gene ID format: if most genes start with 'ENSG', expression
        # data uses Ensembl IDs so GMT symbols need conversion; otherwise the
        # expression data already uses symbols and no conversion is needed.
        n_ensg = sum(1 for g in gene_list if g.startswith('ENSG'))
        genes_are_ensembl = n_ensg > len(gene_list) * 0.5
        convert_flag = genes_are_ensembl
        print(f"  Gene ID format: {'Ensembl' if genes_are_ensembl else 'Symbol'} "
              f"({n_ensg}/{len(gene_list)} ENSG prefix)")
        print(f"  convert_to_ensembl = {convert_flag}")

        # Load pathways from GMT file, filtering to genes in our expression data
        # Uses caching to avoid repeated Ensembl conversion (cache_dir from args)
        pathway_mat, pathway_names_raw, pathway_genes = load_pathways(
            gmt_path=args.pathway_file,
            convert_to_ensembl=convert_flag,
            species='human',
            gene_filter=gene_list,  # Only include genes in our expression data
            min_genes=args.pathway_min_genes,
            max_genes=args.pathway_max_genes,
            cache_dir=args.cache_dir,  # Use same cache dir as data preprocessing
            use_cache=True
        )
        
        # Align pathway matrix columns to match gene_list order
        # pathway_genes is the column order from load_pathways
        # gene_list is the order of columns in X_train
        print(f"\nAligning pathway matrix to expression data gene order...")
        
        # Create mapping from pathway gene order to expression data gene order
        gene_to_expr_idx = {g: i for i, g in enumerate(gene_list)}
        gene_to_pathway_idx = {g: i for i, g in enumerate(pathway_genes)}
        
        # Find common genes
        common_genes = set(gene_list) & set(pathway_genes)
        print(f"  Common genes: {len(common_genes)} / {len(gene_list)} expression genes")
        
        if len(common_genes) < 100:
            raise ValueError(
                f"Too few common genes ({len(common_genes)}) between expression data and pathways. "
                f"Check gene ID format (should be Ensembl) or pathway file."
            )
        
        # Build aligned pathway mask: (n_pathways, n_genes_in_expression)
        # For genes not in any pathway, columns will be all zeros
        n_pathways = pathway_mat.shape[0]
        n_genes_expr = len(gene_list)
        pathway_mask = np.zeros((n_pathways, n_genes_expr), dtype=np.float32)
        
        for gene in common_genes:
            expr_idx = gene_to_expr_idx[gene]
            pathway_idx = gene_to_pathway_idx[gene]
            pathway_mask[:, expr_idx] = pathway_mat[:, pathway_idx]
        
        # Summary
        genes_per_pathway = pathway_mask.sum(axis=1)
        pathways_per_gene = pathway_mask.sum(axis=0)
        genes_in_pathways = (pathways_per_gene > 0).sum()
        
        print(f"\nAligned pathway matrix: {n_pathways} pathways x {n_genes_expr} genes")
        print(f"  Genes covered by pathways: {genes_in_pathways} / {n_genes_expr}")
        print(f"  Genes/pathway: min={genes_per_pathway.min():.0f}, max={genes_per_pathway.max():.0f}, "
              f"mean={genes_per_pathway.mean():.1f}")
        print(f"  Matrix density: {pathway_mask.mean()*100:.2f}%")
        
        # Use pathway names
        pathway_names = pathway_names_raw
        
        # Determine n_factors based on mode
        if args.mode == 'combined':
            # Combined mode: n_factors = n_pathways + n_drgps
            n_drgps = args.n_drgps
            n_factors = n_pathways + n_drgps
            
            # Extend pathway_mask to include unconstrained DRGP columns
            # Pathway columns (0..n_pathways-1): constrained by pathway_mask
            # DRGP columns (n_pathways..n_factors-1): all ones (unconstrained)
            drgp_mask = np.ones((n_drgps, n_genes_expr), dtype=np.float32)
            pathway_mask_combined = np.vstack([pathway_mask, drgp_mask])
            
            # Extend pathway names with DRGP labels
            drgp_names = [f"DRGP_{i+1}" for i in range(n_drgps)]
            pathway_names = pathway_names_raw + drgp_names
            
            # Replace pathway_mask with the extended version
            pathway_mask = pathway_mask_combined
            
            print(f"\n  [COMBINED MODE] n_factors = {n_pathways} pathways + {n_drgps} DRGPs = {n_factors}")
            print(f"    Pathway factors [0:{n_pathways}]: β constrained by pathway membership")
            print(f"    DRGP factors [{n_pathways}:{n_factors}]: β unconstrained (all genes)")
        else:
            # masked/pathway_init modes: n_factors = n_pathways
            n_factors = n_pathways
            print(f"\n  Setting n_factors = {n_factors} (number of pathways)")
            
            if args.n_factors != n_pathways:
                print(f"  NOTE: --n-factors={args.n_factors} overridden by pathway count ({n_pathways})")
    
    else:
        print(f"\n[UNMASKED MODE] Using {n_factors} latent factors (no pathway constraint)")

    # =========================================================================
    # STEP 3.5: Hyperparameters (scHPF defaults — rarely need tuning)
    # =========================================================================
    sigma_v = args.sigma_v

    print(f"\nModel Hyperparameters (scHPF):")
    print(f"  a (theta shape): {args.a:.4f}")
    print(f"  c (beta shape):  {args.c:.4f}")
    print(f"  ap, cp:          1.0, 1.0 (fixed)")
    print(f"  bp, dp:          empirical (computed from data)")
    print(f"  sigma_v:         {sigma_v:.4f}")
    print(f"  sigma_gamma:     {args.sigma_gamma:.4f}")
    print(f"  regression_wt:   {args.regression_weight:.4f}")
    if args.method == 'svi':
        print(f"  lr_decay (kappa):{args.learning_rate_decay:.4f}")
        print(f"  lr_delay (tau):  {args.learning_rate_delay:.4f}")
    else:
        print(f"  max_iter:        {args.max_iter}")
        print(f"  tol:             {args.tol}")

    # =========================================================================
    # STEP 4: Train Model
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"Training {args.method.upper()} Model")
    print("=" * 80)
    
    print(f"\nModel Configuration:")
    print(f"  Mode:         {args.mode}")
    print(f"  n_factors:    {n_factors}")
    if pathway_names is not None:
        print(f"  Pathways:     {len(pathway_names)}")
    
    # For combined mode, track the number of pathway factors
    n_pathway_factors = None
    if args.mode == 'combined':
        # n_pathways was computed during pathway loading
        n_pathway_factors = n_factors - args.n_drgps
        print(f"  n_pathway_factors: {n_pathway_factors}")
        print(f"  n_drgps:          {args.n_drgps}")

    # Common model kwargs (shared by CAVI and SVI)
    # Bayes opt names override scHPF names when provided:
    #   alpha_theta -> a,  alpha_beta -> c,  alpha_xi -> ap,  alpha_eta -> cp
    a_val = args.alpha_theta if args.alpha_theta is not None else args.a
    c_val = args.alpha_beta if args.alpha_beta is not None else args.c
    ap_val = args.alpha_xi if args.alpha_xi is not None else 1.0
    cp_val = args.alpha_eta if args.alpha_eta is not None else 1.0

    common_kwargs = dict(
        n_factors=n_factors,
        a=a_val,
        ap=ap_val,
        c=c_val,
        cp=cp_val,
        sigma_v=sigma_v,
        sigma_gamma=args.sigma_gamma,
        regression_weight=args.regression_weight,
        random_state=args.seed,
        mode=args.mode,
        pathway_mask=pathway_mask,
        pathway_names=pathway_names,
        n_pathway_factors=n_pathway_factors,
    )

    if args.method == 'svi':
        common_kwargs.update(
            batch_size=args.batch_size,
            learning_rate_delay=args.learning_rate_delay,
            learning_rate_decay=args.learning_rate_decay,
            local_iterations=args.local_iterations,
        )

    # Create model
    model = ModelClass(**common_kwargs)

    # Build fit kwargs
    if args.method == 'svi':
        fit_kwargs = dict(
            X_train=X_train,
            y_train=y_train,
            X_aux_train=X_aux_train,
            X_val=X_val,
            y_val=y_val,
            X_aux_val=X_aux_val,
            max_epochs=args.max_epochs,
            check_freq=args.check_freq,
            v_warmup=args.v_warmup,
            verbose=True,
            heldout_patience=args.heldout_patience,
        )
    else:
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
            verbose=True,
        )

    # Train model
    model.fit(**fit_kwargs)

    print("\nTraining complete!")
    if hasattr(model, 'elbo_history_') and model.elbo_history_:
        print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}")
        print(f"  Iterations: {model.elbo_history_[-1][0] + 1}")
    if hasattr(model, 'holl_history_') and model.holl_history_:
        print(f"  Best HO-LL: {model.holl_history_[-1][1]:.4f}")

    # =========================================================================
    # DEBUG: v Parameter Diagnostics
    # =========================================================================
    # Setup output directory early for incremental saves
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("DEBUG: Learned Parameter Diagnostics")
    print("=" * 80)
    print(f"mu_v shape: {model.mu_v.shape}")
    print(f"mu_v range: [{model.mu_v.min():.6f}, {model.mu_v.max():.6f}]")
    print(f"mu_v mean:  {model.mu_v.mean():.6f}")
    print(f"mu_v std:   {model.mu_v.std():.6f}")
    print(f"mu_v sum:   {model.mu_v.sum():.6f}")
    
    # STABILITY DIAGNOSTIC: Save v vector for cross-run comparison
    # Run multiple seeds and compute: np.corrcoef(v_seed1, v_seed2) to check stability
    seed_label = args.seed if args.seed else getattr(args, '_auto_seed', 'unknown')
    v_stability_path = output_dir / f'v_vector_seed{seed_label}.npy'
    np.save(v_stability_path, model.mu_v.flatten())
    print(f"\n[STABILITY] Saved v vector to {v_stability_path}")
    print(f"[STABILITY] To check stability across seeds, run:")
    print(f"    v1 = np.load('v_vector_seed<A>.npy')")
    print(f"    v2 = np.load('v_vector_seed<B>.npy')")
    print(f"    print(f'Spearman r = {{spearmanr(v1, v2).correlation:.4f}}')")

    # =========================================================================
    # CHECKPOINT: Save model parameters immediately after training
    # This prevents data loss if downstream inference (transform) OOMs
    # =========================================================================
    print("\n[CHECKPOINT] Saving model parameters immediately after training...")
    checkpoint_params = {
        'n_factors': model.K,
        'a': float(model.a),
        'c': float(model.c),
        'sigma_v': float(model.sigma_v),
        'E_beta': np.array(model.E_beta),
        'E_log_beta': np.array(model.E_log_beta),
        'mu_v': np.array(model.mu_v),
        'sigma_v_diag': np.array(model.sigma_v_diag),
        'mu_gamma': np.array(model.mu_gamma),
        'n': model.n,
        'p': model.p,
        'p_aux': model.p_aux,
    }
    if hasattr(model, 'elbo_history_'):
        checkpoint_params['elbo_history'] = model.elbo_history_
    if hasattr(model, 'holl_history_'):
        checkpoint_params['holl_history'] = model.holl_history_

    checkpoint_path = output_dir / 'model_checkpoint.npz'
    np.savez_compressed(checkpoint_path, **checkpoint_params)
    print(f"[CHECKPOINT] Saved to {checkpoint_path}")
    print(f"[CHECKPOINT] If downstream OOMs, model can be reconstructed from this file")

    # =========================================================================
    # STEP 5: Evaluate on Training Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training Set Evaluation")
    print("=" * 80)

    # Predict on training set
    y_train_proba = model.predict_proba(X_train, X_aux_train, n_iter=20)
    E_theta_train = model.transform(X_train)['E_theta']

    # Ensure y and proba are 2D for uniform κ-outcome handling
    _y_train_2d = y_train if y_train.ndim == 2 else y_train[:, np.newaxis]
    _proba_train_2d = y_train_proba if y_train_proba.ndim == 2 else y_train_proba[:, np.newaxis]

    # DEBUG: Compute logits and correlation with labels
    train_logits = E_theta_train @ np.array(model.mu_v).T
    if model.p_aux > 0 and model.mu_gamma is not None:
        train_logits = train_logits + X_aux_train @ model.mu_gamma.T
    if train_logits.ndim == 1:
        train_logits = train_logits[:, np.newaxis]

    print(f"\nDEBUG: Training Logit Analysis")
    for k in range(n_outcomes):
        col_logits = train_logits[:, k]
        col_y = _y_train_2d[:, k]
        lname = label_columns[k]
        print(f"  [{lname}] Logits range: [{col_logits.min():.4f}, {col_logits.max():.4f}]")
        print(f"  [{lname}] Logits std:   {col_logits.std():.4f}")
        logit_label_corr = np.corrcoef(col_logits, col_y)[0, 1]
        print(f"  [{lname}] Logit-label correlation: {logit_label_corr:.4f}")
        if logit_label_corr < 0:
            print(f"  [{lname}] WARNING: Negative correlation! Model predictions are INVERTED!")
        if np.abs(logit_label_corr) < 0.1:
            print(f"  [{lname}] WARNING: Weak correlation! Model is not discriminating!")
        logits_c0 = col_logits[col_y == 0]
        logits_c1 = col_logits[col_y == 1]
        print(f"  [{lname}] Class 0 logits: mean={logits_c0.mean():.4f}, std={logits_c0.std():.4f}")
        print(f"  [{lname}] Class 1 logits: mean={logits_c1.mean():.4f}, std={logits_c1.std():.4f}")
        if logits_c1.mean() < logits_c0.mean():
            print(f"  [{lname}] WARNING: Class 1 has LOWER logits than Class 0!")

    # Per-outcome metrics
    train_metrics_all = {}
    for k in range(n_outcomes):
        lname = label_columns[k]
        y_pred_k = (_proba_train_2d[:, k] > 0.5).astype(int)
        metrics_k = compute_metrics(_y_train_2d[:, k], y_pred_k, _proba_train_2d[:, k])
        train_metrics_all[lname] = metrics_k
        print(f"\nTraining Results [{lname}]:")
        print(f"  Accuracy:  {metrics_k['accuracy']:.4f}")
        if 'auc' in metrics_k:
            print(f"  AUC:       {metrics_k['auc']:.4f}")
        print(f"  F1:        {metrics_k['f1']:.4f}")
        print(f"  Precision: {metrics_k['precision']:.4f}")
        print(f"  Recall:    {metrics_k['recall']:.4f}")

    # Backward-compatible single-outcome aliases
    train_metrics = train_metrics_all[label_columns[0]]
    y_train_pred = (_proba_train_2d[:, 0] > 0.5).astype(int)
    
    # Save training results immediately
    train_results_path = output_dir / 'training_results.pkl'
    with open(train_results_path, 'wb') as f:
        pickle.dump({
            'label_columns': label_columns,
            'metrics': train_metrics_all,
            'predictions': {lc: (_proba_train_2d[:, k] > 0.5).astype(int) for k, lc in enumerate(label_columns)},
            'probabilities': {lc: _proba_train_2d[:, k] for k, lc in enumerate(label_columns)},
            'logits': train_logits,
            'E_theta': E_theta_train
        }, f)
    print(f"Training results saved to {train_results_path}")

    # =========================================================================
    # STEP 6: Evaluate on Validation Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Validation Set Evaluation")
    print("=" * 80)

    # Predict on validation set
    y_val_proba = model.predict_proba(X_val, X_aux_val, n_iter=20)
    E_theta_val = model.transform(X_val)['E_theta']
    
    # Ensure 2D for uniform κ-outcome handling
    _y_val_2d = y_val if y_val.ndim == 2 else y_val[:, np.newaxis]
    _proba_val_2d = y_val_proba if y_val_proba.ndim == 2 else y_val_proba[:, np.newaxis]

    # Find optimal threshold on validation set (per outcome)
    optimal_thresholds = {}
    val_metrics_all = {}
    for k in range(n_outcomes):
        lname = label_columns[k]
        prec_k, rec_k, thr_k = precision_recall_curve(_y_val_2d[:, k], _proba_val_2d[:, k])
        f1_k = 2 * prec_k * rec_k / (prec_k + rec_k + 1e-8)
        opt_idx = np.argmax(f1_k[:-1])
        opt_thr = thr_k[opt_idx] if len(thr_k) > 0 else 0.5
        optimal_thresholds[lname] = opt_thr
        print(f"Optimal threshold [{lname}] (validation F1): {opt_thr:.4f}")

        y_val_pred_k = (_proba_val_2d[:, k] > opt_thr).astype(int)
        metrics_k = compute_metrics(_y_val_2d[:, k], y_val_pred_k, _proba_val_2d[:, k])
        val_metrics_all[lname] = metrics_k
        print(f"\nValidation Results [{lname}]:")
        print(f"  Accuracy:  {metrics_k['accuracy']:.4f}")
        if 'auc' in metrics_k:
            print(f"  AUC:       {metrics_k['auc']:.4f}")
        print(f"  F1:        {metrics_k['f1']:.4f}")
        print(f"  Precision: {metrics_k['precision']:.4f}")
        print(f"  Recall:    {metrics_k['recall']:.4f}")

    # Backward-compatible aliases (first outcome)
    optimal_threshold = optimal_thresholds[label_columns[0]]
    val_metrics = val_metrics_all[label_columns[0]]
    y_val_pred = (_proba_val_2d[:, 0] > optimal_threshold).astype(int)
    
    # Save validation results immediately
    val_results_path = output_dir / 'validation_results.pkl'
    with open(val_results_path, 'wb') as f:
        pickle.dump({
            'label_columns': label_columns,
            'metrics': val_metrics_all,
            'predictions': {lc: (_proba_val_2d[:, k] > optimal_thresholds[lc]).astype(int) for k, lc in enumerate(label_columns)},
            'probabilities': {lc: _proba_val_2d[:, k] for k, lc in enumerate(label_columns)},
            'optimal_thresholds': optimal_thresholds,
            'E_theta': E_theta_val
        }, f)
    print(f"Validation results saved to {val_results_path}")

    # =========================================================================
    # STEP 7: Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)

    # Predict on test set
    y_test_proba = model.predict_proba(X_test, X_aux_test, n_iter=20)
    E_theta_test = model.transform(X_test)['E_theta']

    # Ensure 2D for uniform κ-outcome handling
    _y_test_2d = y_test if y_test.ndim == 2 else y_test[:, np.newaxis]
    _proba_test_2d = y_test_proba if y_test_proba.ndim == 2 else y_test_proba[:, np.newaxis]

    # Per-outcome test metrics using per-outcome optimal thresholds from validation
    test_metrics_all = {}
    for k in range(n_outcomes):
        lname = label_columns[k]
        thr_k = optimal_thresholds[lname]
        print(f"Using optimal threshold [{lname}] from validation: {thr_k:.4f}")
        y_test_pred_k = (_proba_test_2d[:, k] > thr_k).astype(int)
        metrics_k = compute_metrics(_y_test_2d[:, k], y_test_pred_k, _proba_test_2d[:, k])
        test_metrics_all[lname] = metrics_k
        print(f"\nTest Results [{lname}]:")
        print(f"  Accuracy:  {metrics_k['accuracy']:.4f}")
        if 'auc' in metrics_k:
            print(f"  AUC:       {metrics_k['auc']:.4f}")
        print(f"  F1:        {metrics_k['f1']:.4f}")
        print(f"  Precision: {metrics_k['precision']:.4f}")
        print(f"  Recall:    {metrics_k['recall']:.4f}")

    # Backward-compatible aliases
    test_metrics = test_metrics_all[label_columns[0]]
    y_test_pred = (_proba_test_2d[:, 0] > optimal_threshold).astype(int)

    # Save test results immediately
    test_results_path = output_dir / 'test_results.pkl'
    with open(test_results_path, 'wb') as f:
        pickle.dump({
            'label_columns': label_columns,
            'metrics': test_metrics_all,
            'predictions': {lc: (_proba_test_2d[:, k] > optimal_thresholds[lc]).astype(int) for k, lc in enumerate(label_columns)},
            'probabilities': {lc: _proba_test_2d[:, k] for k, lc in enumerate(label_columns)},
            'optimal_thresholds': optimal_thresholds,
            'E_theta': E_theta_test
        }, f)
    print(f"Test results saved to {test_results_path}")

    # =========================================================================
    # STEP 8: Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    # output_dir already defined earlier for incremental saves
    prefix = 'svi'
    saved_files = save_results(
        model=model,
        output_dir=output_dir,
        gene_list=gene_list,
        splits=splits,
        prefix=prefix,
        compress=True,
        optimal_threshold=optimal_thresholds,
        program_names=pathway_names,  # Use pathway names if in masked/pathway_init mode
        mode=args.mode
    )

    # # Save additional files

    # # Training theta
    # theta_train_df = pd.DataFrame(
    #     E_theta_train,
    #     index=splits['train'],
    #     columns=[f"GP{k+1}" for k in range(model.d)]
    # )
    # theta_train_df.to_csv(output_dir / f'{prefix}_theta_train.csv.gz', compression='gzip')
    # print(f"Saved training theta to {output_dir / f'{prefix}_theta_train.csv.gz'}")

    # # Validation theta
    # theta_val_df = pd.DataFrame(
    #     E_theta_val,
    #     index=splits['val'],
    #     columns=[f"GP{k+1}" for k in range(model.d)]
    # )
    # theta_val_df.to_csv(output_dir / f'{prefix}_theta_val.csv.gz', compression='gzip')
    # print(f"Saved validation theta to {output_dir / f'{prefix}_theta_val.csv.gz'}")

    # # Test theta
    # theta_test_df = pd.DataFrame(
    #     E_theta_test,
    #     index=splits['test'],
    #     columns=[f"GP{k+1}" for k in range(model.d)]
    # )
    # theta_test_df.to_csv(output_dir / f'{prefix}_theta_test.csv.gz', compression='gzip')
    # print(f"Saved test theta to {output_dir / f'{prefix}_theta_test.csv.gz'}")

    # Predictions (per-outcome columns)
    def _build_pred_df(cell_ids, y_2d, proba_2d, thresholds_dict):
        d = {'cell_id': cell_ids}
        for k, lc in enumerate(label_columns):
            thr = thresholds_dict.get(lc, 0.5)
            d[f'true_{lc}'] = y_2d[:, k]
            d[f'prob_{lc}'] = proba_2d[:, k]
            d[f'pred_{lc}'] = (proba_2d[:, k] > thr).astype(int)
        return pd.DataFrame(d)

    # Training: threshold = 0.5 (no validation-tuned threshold for training)
    train_thresholds = {lc: 0.5 for lc in label_columns}
    train_pred_df = _build_pred_df(splits['train'], _y_train_2d, _proba_train_2d, train_thresholds)
    train_pred_df.to_csv(output_dir / f'{prefix}_train_predictions.csv.gz', compression='gzip', index=False)

    val_pred_df = _build_pred_df(splits['val'], _y_val_2d, _proba_val_2d, optimal_thresholds)
    val_pred_df.to_csv(output_dir / f'{prefix}_val_predictions.csv.gz', compression='gzip', index=False)

    test_pred_df = _build_pred_df(splits['test'], _y_test_2d, _proba_test_2d, optimal_thresholds)
    test_pred_df.to_csv(output_dir / f'{prefix}_test_predictions.csv.gz', compression='gzip', index=False)
    print(f"Saved predictions to {output_dir}")

    # =========================================================================
    # STEP 8: Model Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Model Summary")
    print("=" * 80)

    print_model_summary(model, gene_list)

    # # =========================================================================
    # # DEBUG: Automatic Recommendations
    # # =========================================================================
    # issues_found = []
    # recommendations = []

    # # Check 1: Probability range too narrow
    # prob_range = y_train_proba.max() - y_train_proba.min()
    # if prob_range < 0.2:
    #     issues_found.append(f"Probability range ({prob_range:.4f}) is very narrow - model not discriminating")
    #     recommendations.append("Increase --regression-weight (try 100-200)")
    #     recommendations.append("Increase --max-epochs (try 200-500)")
    #     recommendations.append("Increase --learning-rate (try 0.5)")

    # # Check 2: AUC below 0.5
    # if 'auc' in train_metrics and train_metrics['auc'] < 0.5:
    #     issues_found.append(f"Training AUC ({train_metrics['auc']:.4f}) < 0.5 - predictions inversely correlated")
    #     recommendations.append("The model learned the wrong direction - try increasing --regression-weight")

    # # Check 3: v not learned
    # v_range = model.mu_v.max() - model.mu_v.min()
    # if v_range < 0.5:
    #     issues_found.append(f"v parameter range ({v_range:.4f}) is very small - discrimination weights not learned")
    #     recommendations.append("Learning rate may be dying too fast - try --learning-rate-decay 0.6")
    #     recommendations.append("Try --learning-rate-min 0.01 (higher floor)")

    # # Check 4: Learning rate too aggressive decay
    # final_lr = args.learning_rate * (args.learning_rate_delay + model.elbo_history_[-1][0]) ** (-args.learning_rate_decay)
    # if final_lr < 0.001:
    #     issues_found.append(f"Final learning rate ({final_lr:.6f}) is very small - may have stopped learning early")
    #     recommendations.append("Try --learning-rate-decay 0.6 instead of 0.75")
    #     recommendations.append("Try --learning-rate-min 0.01")

    # if issues_found:
    #     print("\n" + "=" * 80)
    #     print("DEBUG: Issues Detected & Recommendations")
    #     print("=" * 80)
    #     print("\nIssues found:")
    #     for i, issue in enumerate(issues_found, 1):
    #         print(f"  {i}. {issue}")
    #     print("\nRecommendations:")
    #     seen = set()
    #     for rec in recommendations:
    #         if rec not in seen:
    #             print(f"  - {rec}")
    #             seen.add(rec)
    #     print("\nSuggested command:")
    #     print(f"  python quick_reference.py --data {args.data} --n-factors {args.n_factors} \\")
    #     print(f"    --regression-weight 100 --learning-rate 0.5 --learning-rate-decay 0.6 \\")
    #     print(f"    --learning-rate-min 0.01 --max-epochs 200")

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
    # print(f"  - {output_dir / f'{prefix}_theta_val.csv.gz'}")
    # print(f"  - {output_dir / f'{prefix}_theta_test.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_val_predictions.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_test_predictions.csv.gz'}")

    print(f"\nNext steps:")
    print(f"  1. Load gene programs: pd.read_csv('{prefix}_gene_programs.csv.gz')")
    print("  2. Identify top genes per program and run pathway enrichment")
    print("  3. Load theta matrices to analyze sample-level program activity")
    print("  4. Correlate theta values with phenotypes/outcomes")


if __name__ == '__main__':
    main()
