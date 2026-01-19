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
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

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
from scipy.special import expit
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
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='/labs/Aguiar/SSPA_BRAY/cache',
        help='Directory for caching preprocessed data (default: /labs/Aguiar/SSPA_BRAY/cache)'
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
        default=30,
        help='Minimum genes per pathway (for filtering)'
    )
    parser.add_argument(
        '--pathway-max-genes',
        type=int,
        default=200,
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
        default=100,
        help='Maximum epochs for SVI'
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
        '--learning-rate-delay',
        type=float,
        default=1.0,
        help='Learning rate delay (tau) for SVI'
    )
    parser.add_argument(
        '--learning-rate-min',
        type=float,
        default=1e-4,
        help='Minimum learning rate for SVI to prevent stagnation'
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
        '--elbo-freq',
        type=int,
        default=10,
        help='Compute ELBO every N iterations'
    )
    
    # Convergence options
    parser.add_argument(
        '--ema-decay',
        type=float,
        default=0.95,
        help='EMA smoothing factor for ELBO tracking (lower=faster response, default: 0.95)'
    )
    parser.add_argument(
        '--convergence-tol',
        type=float,
        default=1e-4,
        help='Relative change threshold for convergence (default: 1e-4)'
    )
    parser.add_argument(
        '--convergence-window',
        type=int,
        default=10,
        help='Number of consecutive epochs below tol to trigger convergence (default: 10)'
    )
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        default=False,
        help='Enable early stopping when ELBO converges (recommended for stability)'
    )
    parser.add_argument(
        '--early-stopping-metric',
        type=str,
        default='elbo',
        choices=['elbo', 'heldout_ll'],
        help='Metric for early stopping: elbo (training) or heldout_ll (validation)'
    )
    parser.add_argument(
        '--heldout-ll-patience',
        type=int,
        default=10,
        help='Epochs without HO-LL improvement before stopping (only for heldout_ll metric)'
    )
    parser.add_argument(
        '--heldout-ll-ema-decay',
        type=float,
        default=0.9,
        help='EMA smoothing for HO-LL tracking (lower=faster response)'
    )
    parser.add_argument(
        '--restore-best-heldout',
        action='store_true',
        default=True,
        help='Restore parameters to best HO-LL epoch on early stop'
    )
    parser.add_argument(
        '--min-epochs-before-stopping',
        type=int,
        default=20,
        help='Minimum epochs before early stopping can trigger'
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
    parser.add_argument(
        '--use-spike-slab',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help='Use spike-and-slab priors (default: False). Set to True for sparse priors.'
    )
    parser.add_argument(
        '--count-scale',
        type=float,
        default=None,
        help='Count scaling factor for Poisson likelihood. None=auto (median library size)'
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
    print(f"  Mode:         {args.mode}")
    print(f"  n_factors:    {args.n_factors}" + (" (may be overridden by pathway count)" if args.mode != 'unmasked' else ""))
    print(f"  batch_size:   {args.batch_size}")
    print(f"  max_epochs:   {args.max_epochs}")
    print(f"  learning_rate:{args.learning_rate}")
    print(f"  label_column: {args.label_column}")
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
    from VariationalInference.svi_corrected import SVI
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
        label_column=args.label_column,
        aux_columns=args.aux_columns,
        train_ratio=0.7,
        val_ratio=0.15,
        stratify_by=args.label_column,
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
    print(f"  Label dist:     {np.bincount(y_train)}")

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
        
        # Load pathways from GMT file, filtering to genes in our expression data
        # Uses caching to avoid repeated Ensembl conversion (cache_dir from args)
        pathway_mat, pathway_names_raw, pathway_genes = load_pathways(
            gmt_path=args.pathway_file,
            convert_to_ensembl=True,
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
    print(f"  count_scale:  {args.count_scale if args.count_scale else 'auto'}")

    # =========================================================================
    # STEP 4: Train SVI Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training SVI Model")
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

    # Create SVI model
    model = SVI(
        n_factors=n_factors,  # Use n_factors (may be overridden by pathway count)
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        learning_rate_delay=args.learning_rate_delay,
        learning_rate_min=args.learning_rate_min,
        local_iterations=args.local_iterations,
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
        count_scale=args.count_scale,
        random_state=args.seed,
        use_spike_slab=args.use_spike_slab,
        ema_decay=args.ema_decay,
        convergence_tol=args.convergence_tol,
        convergence_window=args.convergence_window,
        # Early stopping settings
        early_stopping_metric=args.early_stopping_metric,
        heldout_ll_patience=args.heldout_ll_patience,
        heldout_ll_ema_decay=args.heldout_ll_ema_decay,
        restore_best_heldout=args.restore_best_heldout,
        min_epochs_before_stopping=args.min_epochs_before_stopping,
        # Pathway mode settings
        mode=args.mode,
        pathway_mask=pathway_mask,
        pathway_names=pathway_names,
        n_pathway_factors=n_pathway_factors  # For combined mode boundary
    )

    # Train model with held-out validation data for convergence tracking
    model.fit(
        X=X_train,
        y=y_train,
        X_aux=X_aux_train,
        max_epochs=args.max_epochs,
        elbo_freq=args.elbo_freq,
        verbose=True,
        early_stopping=args.early_stopping,
        # Held-out data for Blei-style convergence tracking
        X_heldout=X_val,
        y_heldout=y_val,
        X_aux_heldout=X_aux_val,
        heldout_freq=5  # Compute held-out LL every 5 epochs
    )

    print("\nTraining complete!")
    print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}")
    print(f"  Iterations: {model.elbo_history_[-1][0] + 1}")
    print(f"  Time:       {model.training_time_:.2f}s")

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
    # STEP 5: Evaluate on Training Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training Set Evaluation")
    print("=" * 80)

    # Use the stored training parameters from the final epoch
    # These are the actual θ values that were used during training
    # E_theta_train = model.train_a_theta_ / model.train_b_theta_
    print("Using stored training set θ from final epoch")

    # Determine appropriate batch size for memory efficiency
    # Target ~500MB per batch for the (n_batch, p, d) tensor
    auto_batch_size = model._compute_memory_efficient_batch_size(target_gb=0.5)
    print(f"Using batched processing with batch_size={auto_batch_size} for memory efficiency")

    # Compute probabilities with BATCHED method to avoid OOM
    y_train_proba = model.predict_proba_batched(X_train, X_aux_train, n_iter=50,
                                                 batch_size=auto_batch_size, verbose=args.verbose)
    if args.verbose:
        print(f"Predicted probabilities: min={y_train_proba.min():.4f}, max={y_train_proba.max():.4f}")

    # DEBUG: Compute logits and correlation with labels (batched)
    train_result = model.transform_batched(X_train, y_new=None, X_aux_new=X_aux_train,
                                           n_iter=50, batch_size=auto_batch_size,
                                           verbose=args.verbose)
    E_theta_train = train_result['E_theta']
    train_logits = E_theta_train @ model.mu_v.T
    if model.p_aux > 0 and model.mu_gamma is not None:
        train_logits = train_logits + X_aux_train @ model.mu_gamma.T
    train_logits = train_logits.flatten()

    print(f"\nDEBUG: Training Logit Analysis")
    print(f"  Logits range: [{train_logits.min():.4f}, {train_logits.max():.4f}]")
    print(f"  Logits std:   {train_logits.std():.4f}")
    logit_label_corr = np.corrcoef(train_logits, y_train.flatten())[0, 1]
    print(f"  Logit-label correlation: {logit_label_corr:.4f}")
    if logit_label_corr < 0:
        print(f"  WARNING: Negative correlation! Model predictions are INVERTED!")
    if np.abs(logit_label_corr) < 0.1:
        print(f"  WARNING: Weak correlation! Model is not discriminating!")

    # Show logit distribution by class
    y_flat = y_train.flatten()
    logits_class0 = train_logits[y_flat == 0]
    logits_class1 = train_logits[y_flat == 1]
    print(f"  Class 0 logits: mean={logits_class0.mean():.4f}, std={logits_class0.std():.4f}")
    print(f"  Class 1 logits: mean={logits_class1.mean():.4f}, std={logits_class1.std():.4f}")
    if logits_class1.mean() < logits_class0.mean():
        print(f"  WARNING: Class 1 has LOWER logits than Class 0 - model learned wrong direction!")

    y_train_pred = (y_train_proba.ravel() > 0.5).astype(int)

    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba.ravel())
    print(f"\nTraining Results:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    if 'auc' in train_metrics:
        print(f"  AUC:       {train_metrics['auc']:.4f}")
    print(f"  F1:        {train_metrics['f1']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    
    # Save training results immediately
    train_results_path = output_dir / 'training_results.pkl'
    with open(train_results_path, 'wb') as f:
        pickle.dump({
            'metrics': train_metrics,
            'predictions': y_train_pred,
            'probabilities': y_train_proba,
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

    # Infer theta for validation set (BATCHED to avoid OOM)
    val_result = model.transform_batched(X_val, y_new=None, X_aux_new=X_aux_val,
                                         n_iter=50, batch_size=auto_batch_size,
                                         verbose=args.verbose)
    # E_theta_val = val_result['E_theta']

    # Compute probabilities (BATCHED to avoid OOM)
    y_val_proba = model.predict_proba_batched(X_val, X_aux_val, n_iter=50,
                                               batch_size=auto_batch_size,
                                               verbose=args.verbose)
    if args.verbose:
        print(f"Predicted probabilities: min={y_val_proba.min():.4f}, max={y_val_proba.max():.4f}")
    
    # Find optimal threshold on validation set
    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_val, y_val_proba.ravel())
    f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)
    optimal_idx = np.argmax(f1_curve[:-1])  # Last element is threshold=1.0
    optimal_threshold = thresholds_curve[optimal_idx] if len(thresholds_curve) > 0 else 0.5
    print(f"Optimal threshold (validation F1): {optimal_threshold:.4f}")
    
    y_val_pred = (y_val_proba.ravel() > optimal_threshold).astype(int)

    val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba.ravel())
    print(f"\nValidation Results:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    if 'auc' in val_metrics:
        print(f"  AUC:       {val_metrics['auc']:.4f}")
    print(f"  F1:        {val_metrics['f1']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    
    # Save validation results immediately
    val_results_path = output_dir / 'validation_results.pkl'
    with open(val_results_path, 'wb') as f:
        pickle.dump({
            'metrics': val_metrics,
            'predictions': y_val_pred,
            'probabilities': y_val_proba,
            'optimal_threshold': optimal_threshold,
            'E_theta': val_result['E_theta']
        }, f)
    print(f"Validation results saved to {val_results_path}")

    # =========================================================================
    # STEP 7: Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)

    # Infer theta for test set (uses stability averaging automatically now)
    test_result = model.transform(X_test, y_new=None, X_aux_new=X_aux_test, n_iter=50)

    # Compute raw probabilities (no hard PIP threshold - use all factors)
    y_test_proba = model.predict_proba(X_test, X_aux_test, n_iter=50)
    
    if args.verbose:
        print(f"Predicted probabilities: min={y_test_proba.min():.4f}, max={y_test_proba.max():.4f}")
    
    print(f"Using optimal threshold from validation: {optimal_threshold:.4f}")
    y_test_pred = (y_test_proba.ravel() > optimal_threshold).astype(int)

    # Report factor summary (soft sparsity via rho_v, no hard threshold)
    if hasattr(model, 'get_sparse_factors') and model.use_spike_slab:
        # Use median PIP as adaptive threshold for reporting
        rho_v_median = float(np.median(np.array(model.rho_v)))
        sparse_info = model.get_sparse_factors(pip_threshold=rho_v_median)
        n_active = len(sparse_info['active_factors'])
        print(f"\nActive DRGPs (PIP > {rho_v_median:.2f} median): {n_active} / {model.d}")
        if n_active < 10:
            for i in range(n_active):
                f_idx = sparse_info['active_factors'][i][1]
                v_val = sparse_info['v_values'][i]
                direction = sparse_info['direction'][i]
                print(f"  Factor {f_idx}: v={v_val:.4f} ({direction})")

    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba.ravel())
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    if 'auc' in test_metrics:
        print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    
    # Save test results immediately
    test_results_path = output_dir / 'test_results.pkl'
    with open(test_results_path, 'wb') as f:
        pickle.dump({
            'metrics': test_metrics,
            'predictions': y_test_pred,
            'probabilities': y_test_proba,
            'E_theta': test_result['E_theta']
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
        optimal_threshold=optimal_threshold,
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
