#!/usr/bin/env python
"""
Quick Reference: Variational Inference (CAVI) for Single-Cell Analysis
=======================================================================

This script demonstrates the complete VI (CAVI) workflow for gene program
discovery and phenotype classification from single-cell RNA-seq data.

USAGE:
------
    # Basic usage
    python quick_reference.py --data /path/to/data.h5ad --n-factors 50

    # Full example
    python quick_reference.py \
        --data /path/to/data.h5ad \
        --n-factors 50 \
        --max-iter 600 \
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
3. Train CAVI model with specified parameters
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
        description='Variational Inference (CAVI) Quick Reference',
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

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        default='unmasked',
        choices=['unmasked', 'masked', 'pathway_init', 'combined'],
        help='Model mode: unmasked (standard), masked (beta fixed to pathway structure), '
             'pathway_init (beta initialized from pathways but free to deviate), '
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
        default=0,
        help='Minimum genes per pathway (for filtering)'
    )
    parser.add_argument(
        '--pathway-max-genes',
        type=int,
        default=200000,
        help='Maximum genes per pathway (for filtering)'
    )
    parser.add_argument(
        '--n-drgps',
        type=int,
        default=50,
        help='Number of unconstrained data-driven gene programs (DRGPs) for combined mode'
    )

    # Training options
    parser.add_argument(
        '--check-freq',
        type=int,
        default=5,
        help='Check convergence / compute held-out LL every N iterations'
    )

    # Hyperparameter options (scHPF priors)
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
        '--b-v',
        type=float,
        default=1.0,
        help='Laplace prior scale for v (classification weights). '
             'Smaller b_v = stronger sparsity.'
    )
    parser.add_argument(
        '--sigma-gamma',
        type=float,
        default=1.0,
        help='Gaussian prior std for gamma (auxiliary effects)'
    )
    parser.add_argument(
        '--no-intercept',
        action='store_true',
        default=False,
        help='Disable implicit intercept (constant column prepended to X_aux)'
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
        '--pi-v',
        type=float,
        default=None,
        help='Spike-and-slab mixture weight for v.'
    )
    parser.add_argument(
        '--pi-beta',
        type=float,
        default=None,
        help='Spike-and-slab mixture weight for beta.'
    )
    parser.add_argument(
        '--alpha-pi',
        type=float,
        default=1.0,
        help='Beta prior shape alpha for gene inclusion probability (default=1.0).'
    )
    parser.add_argument(
        '--beta-pi-scale',
        type=float,
        default=None,
        help='Beta prior shape beta for gene inclusion probability (default=K).'
    )

    parser.add_argument(
        '--early-stopping',
        type=str,
        choices=['heldout_ll', 'elbo', 'none'],
        default='heldout_ll',
        help='Early stopping criterion: heldout_ll (default, stop on held-out LL), '
             'elbo (stop on ELBO convergence), none (disable early stopping)'
    )

    # VI (CAVI) training options
    parser.add_argument(
        '--max-iter',
        type=int,
        default=600,
        help='Maximum iterations for coordinate ascent VI'
    )
    parser.add_argument(
        '--tol',
        type=float,
        default=0.001,
        help='Convergence tolerance (percent change in loss)'
    )
    parser.add_argument(
        '--v-warmup',
        type=int,
        default=50,
        help='Poisson-only warmup iterations before regression (v/gamma/zeta) updates begin. '
             'Higher values (100-200) recommended for masked mode with many pathways.'
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

    # On-the-fly subsampling (for scalability benchmarking without saving to disk)
    parser.add_argument(
        '--subsample-ratio',
        type=float,
        default=None,
        help='Subsample dataset to this fraction of patients before processing.'
    )
    parser.add_argument(
        '--subsample-n-patients',
        type=int,
        default=None,
        help='Subsample dataset to exactly this many patients before processing.'
    )
    parser.add_argument(
        '--subsample-seed',
        type=int,
        default=0,
        help='Seed for patient-level subsampling (deterministic).'
    )

    # Patient-level splitting (prevents donor leakage in single-cell data)
    parser.add_argument(
        '--patient-column',
        type=str,
        default=None,
        help='Column in adata.obs identifying patients/donors (e.g., sampleID). '
             'When set, train/val/test splits are performed at the patient level '
             'so no patient appears in more than one split. Also enables '
             'patient-level evaluation metrics via mean-pooled cell predictions.'
    )

    return parser.parse_args()


def main():
    """Main function demonstrating the CAVI workflow."""
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
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[SEED] Set Python/NumPy random seed to {args.seed}")
    else:
        import time
        auto_seed = int(time.time() * 1000) % (2**32)
        random.seed(auto_seed)
        np.random.seed(auto_seed)
        args._auto_seed = auto_seed
        print(f"[SEED] No seed provided, using auto-generated seed: {auto_seed}")

    # Initialize profiler if requested
    profiler = None
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        print("\n[PROFILER] Profiling enabled - measuring function execution times...")

    print("=" * 80)
    print("COORDINATE ASCENT VARIATIONAL INFERENCE - QUICK REFERENCE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data:         {args.data}")
    print(f"  Mode:         {args.mode}")
    print(f"  n_factors:    {args.n_factors}" + (" (may be overridden by pathway count)" if args.mode != 'unmasked' else ""))
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
    from VariationalInference.vi_cavi import CAVI as ModelClass
    print("[METHOD] Using CAVI (coordinate ascent, full-batch)")
    from VariationalInference.data_loader import DataLoader
    from VariationalInference.utils import (
        compute_metrics, save_results, print_model_summary, load_pathways,
        plot_diagnostics,
    )

    # =========================================================================
    # STEP 3: Load and Preprocess Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Loading and Preprocessing Data")
    print("=" * 80)

    # On-the-fly subsampling (avoids saving large subsample h5ad files to disk)
    _preloaded_adata = None
    if args.subsample_ratio is not None or args.subsample_n_patients is not None:
        import anndata as ad
        from VariationalInference.create_subsamples import subsample_adata
        print(f"[SUBSAMPLE] Loading full h5ad for on-the-fly subsampling "
              f"(ratio={args.subsample_ratio}, n_patients={args.subsample_n_patients}, "
              f"seed={args.subsample_seed})")
        _full_adata = ad.read_h5ad(args.data)
        _full_adata.var_names_make_unique()
        _preloaded_adata = subsample_adata(
            _full_adata,
            ratio=args.subsample_ratio,
            n_patients=args.subsample_n_patients,
            subsample_seed=args.subsample_seed,
            verbose=True,
        )
        del _full_adata  # free memory

    loader = DataLoader(
        data_path=args.data,
        gene_annotation_path=args.gene_annotation,
        cache_dir=args.cache_dir,
        use_cache=True,
        verbose=args.verbose,
        adata=_preloaded_adata,
    )

    data = loader.load_and_preprocess(
        label_column=label_columns,
        aux_columns=args.aux_columns,
        train_ratio=0.7,
        val_ratio=0.15,
        stratify_by=label_columns[0],
        min_cells_expressing=0.001,
        layer=args.layer,
        convert_to_ensembl=True,
        filter_protein_coding=args.gene_annotation is not None,
        random_state=args.seed,
        normalize=args.normalize,
        normalize_target_sum=args.normalize_target_sum,
        normalize_method=args.normalize_method,
        patient_column=args.patient_column,
    )

    # Unpack data
    X_train, X_aux_train, y_train = data['train']
    X_val, X_aux_val, y_val = data['val']
    X_test, X_aux_test, y_test = data['test']
    gene_list = data['gene_list']
    splits = data['splits']
    patient_split = splits.get('patient_split', None)
    aux_column_names = data.get('aux_column_names', None)

    # Build per-cell patient ID array for patient-level class weights
    train_patient_ids = None
    if patient_split is not None and args.patient_column:
        cell_meta = data['cell_metadata']
        train_cells = splits['train']
        train_patient_ids = cell_meta.loc[train_cells, args.patient_column].values

    print(f"\nData Summary:")
    print(f"  Genes:          {len(gene_list)}")
    print(f"  Training cells: {len(splits['train'])}")
    print(f"  Validation:     {len(splits['val'])}")
    print(f"  Test:           {len(splits['test'])}")
    n_train_patients = None
    if patient_split is not None:
        from collections import Counter
        _ps_counts = Counter(patient_split.values())
        n_train_patients = _ps_counts.get('train', 0)
        print(f"  Split mode:     PATIENT-GROUPED (no donor leakage)")
        print(f"  Train patients: {n_train_patients}")
        print(f"  Val patients:   {_ps_counts.get('val', 0)}")
        print(f"  Test patients:  {_ps_counts.get('test', 0)}")
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

        valid_gene_list = [g for g in gene_list if isinstance(g, str) and g]
        n_invalid_genes = len(gene_list) - len(valid_gene_list)
        if n_invalid_genes > 0:
            print(f"  Warning: {n_invalid_genes} invalid gene IDs detected (None/non-string/empty); excluding from pathway matching.")

        n_ensg = sum(1 for g in valid_gene_list if g.startswith('ENSG'))
        genes_are_ensembl = n_ensg > len(valid_gene_list) * 0.5 if len(valid_gene_list) > 0 else False
        convert_flag = genes_are_ensembl
        print(f"  Gene ID format: {'Ensembl' if genes_are_ensembl else 'Symbol'} "
              f"({n_ensg}/{len(valid_gene_list)} ENSG prefix among valid IDs)")
        print(f"  convert_to_ensembl = {convert_flag}")

        pathway_mat, pathway_names_raw, pathway_genes = load_pathways(
            gmt_path=args.pathway_file,
            convert_to_ensembl=convert_flag,
            species='human',
            gene_filter=valid_gene_list,
            min_genes=args.pathway_min_genes,
            max_genes=args.pathway_max_genes,
            cache_dir=args.cache_dir,
            use_cache=True
        )

        # Align pathway matrix columns to match gene_list order
        print(f"\nAligning pathway matrix to expression data gene order...")

        gene_to_expr_idx = {g: i for i, g in enumerate(gene_list)}
        gene_to_pathway_idx = {g: i for i, g in enumerate(pathway_genes)}

        common_genes = set(valid_gene_list) & set(pathway_genes)
        print(f"  Common genes: {len(common_genes)} / {len(valid_gene_list)} valid expression genes")

        if len(common_genes) < 100:
            raise ValueError(
                f"Too few common genes ({len(common_genes)}) between expression data and pathways. "
                f"Check gene ID format (should be Ensembl) or pathway file."
            )

        n_pathways = pathway_mat.shape[0]
        n_genes_expr = len(gene_list)
        pathway_mask = np.zeros((n_pathways, n_genes_expr), dtype=np.float32)

        for gene in common_genes:
            expr_idx = gene_to_expr_idx[gene]
            pathway_idx = gene_to_pathway_idx[gene]
            pathway_mask[:, expr_idx] = pathway_mat[:, pathway_idx]

        genes_per_pathway = pathway_mask.sum(axis=1)
        pathways_per_gene = pathway_mask.sum(axis=0)
        genes_in_pathways = (pathways_per_gene > 0).sum()

        print(f"\nAligned pathway matrix: {n_pathways} pathways x {n_genes_expr} genes")
        print(f"  Genes covered by pathways: {genes_in_pathways} / {n_genes_expr}")
        print(f"  Genes/pathway: min={genes_per_pathway.min():.0f}, max={genes_per_pathway.max():.0f}, "
              f"mean={genes_per_pathway.mean():.1f}")
        print(f"  Matrix density: {pathway_mask.mean()*100:.2f}%")

        # -----------------------------------------------------------------
        # Masked-mode gene restriction: keep only pathway genes
        # -----------------------------------------------------------------
        # In masked mode, genes outside any pathway have beta forced to ~0,
        # so they contribute noise to the Poisson likelihood without adding
        # signal.  Restricting X to pathway genes removes this dilution.
        # This does NOT apply to combined (free DRGPs need all genes),
        # pathway_init (genes evolve freely after init), or unmasked.
        if args.mode == 'masked':
            pathway_gene_idx = np.where(pathways_per_gene > 0)[0]
            n_kept = len(pathway_gene_idx)
            n_dropped = n_genes_expr - n_kept
            print(f"\n  [MASKED] Restricting to {n_kept} pathway genes "
                  f"(dropping {n_dropped} non-pathway genes)")

            # Subset sparse X matrices (column selection)
            X_train = X_train[:, pathway_gene_idx]
            X_val = X_val[:, pathway_gene_idx]
            X_test = X_test[:, pathway_gene_idx]

            # Subset pathway_mask and gene_list
            pathway_mask = pathway_mask[:, pathway_gene_idx]
            gene_list = [gene_list[i] for i in pathway_gene_idx]

            print(f"  New X shape: {X_train.shape[0]} cells x {X_train.shape[1]} genes")
            print(f"  New pathway_mask: {pathway_mask.shape[0]} x {pathway_mask.shape[1]}")
            print(f"  Matrix density: {pathway_mask.mean()*100:.2f}%")

        pathway_names = pathway_names_raw

        if args.mode == 'combined':
            n_drgps = args.n_drgps
            n_factors = n_pathways + n_drgps

            drgp_mask = np.ones((n_drgps, n_genes_expr), dtype=np.float32)
            pathway_mask_combined = np.vstack([pathway_mask, drgp_mask])

            drgp_names = [f"DRGP_{i+1}" for i in range(n_drgps)]
            pathway_names = pathway_names_raw + drgp_names

            pathway_mask = pathway_mask_combined

            print(f"\n  [COMBINED MODE] n_factors = {n_pathways} pathways + {n_drgps} DRGPs = {n_factors}")
            print(f"    Pathway factors [0:{n_pathways}]: beta constrained by pathway membership")
            print(f"    DRGP factors [{n_pathways}:{n_factors}]: beta unconstrained (all genes)")
        else:
            n_factors = n_pathways
            print(f"\n  Setting n_factors = {n_factors} (number of pathways)")

            if args.n_factors != n_pathways:
                print(f"  NOTE: --n-factors={args.n_factors} overridden by pathway count ({n_pathways})")

    else:
        print(f"\n[UNMASKED MODE] Using {n_factors} latent factors (no pathway constraint)")

    # =========================================================================
    # STEP 3.5: Hyperparameters
    # =========================================================================
    print(f"\nModel Hyperparameters (scHPF):")
    print(f"  a (theta shape): {args.a:.4f}")
    print(f"  c (beta shape):  {args.c:.4f}")
    print(f"  ap, cp:          1.0, 1.0 (fixed)")
    print(f"  bp, dp:          empirical (computed from data)")
    print(f"  b_v:             {args.b_v:.4f}")
    print(f"  sigma_gamma:     {args.sigma_gamma:.4f}")
    print(f"  max_iter:        {args.max_iter}")
    print(f"  tol:             {args.tol}")

    # =========================================================================
    # STEP 4: Train Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Training CAVI Model")
    print("=" * 80)

    print(f"\nModel Configuration:")
    print(f"  Mode:         {args.mode}")
    print(f"  n_factors:    {n_factors}")
    if pathway_names is not None:
        print(f"  Pathways:     {len(pathway_names)}")

    n_pathway_factors = None
    if args.mode == 'combined':
        n_pathway_factors = n_factors - args.n_drgps
        print(f"  n_pathway_factors: {n_pathway_factors}")
        print(f"  n_drgps:          {args.n_drgps}")

    # Build model kwargs
    a_val = args.alpha_theta if args.alpha_theta is not None else args.a
    c_val = args.alpha_beta if args.alpha_beta is not None else args.c
    ap_val = args.alpha_xi if args.alpha_xi is not None else 1.0
    cp_val = args.alpha_eta if args.alpha_eta is not None else 1.0

    model_kwargs = dict(
        n_factors=n_factors,
        a=a_val,
        ap=ap_val,
        c=c_val,
        cp=cp_val,
        b_v=args.b_v,
        sigma_gamma=args.sigma_gamma,
        use_intercept=not args.no_intercept,
        random_state=args.seed,
        mode=args.mode,
        pathway_mask=pathway_mask,
        pathway_names=pathway_names,
        n_pathway_factors=n_pathway_factors,
        alpha_pi=args.alpha_pi,
        beta_pi_scale=args.beta_pi_scale,
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
        verbose=True,
        early_stopping=args.early_stopping,
        n_patients=n_train_patients,
        patient_ids=train_patient_ids,
    )

    model.fit(**fit_kwargs)

    print("\nTraining complete!")
    if hasattr(model, 'elbo_history_') and model.elbo_history_:
        print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}")
        print(f"  Iterations: {model.elbo_history_[-1][0] + 1}")
    if hasattr(model, 'holl_history_') and model.holl_history_:
        print(f"  Best HO-LL: {max(entry[1] for entry in model.holl_history_):.4f}")

    # =========================================================================
    # DEBUG: v Parameter Diagnostics
    # =========================================================================
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
    # =========================================================================
    print("\n[CHECKPOINT] Saving model parameters immediately after training...")
    checkpoint_params = {
        'n_factors': model.K,
        'a': float(model.a),
        'c': float(model.c),
        'b_v': float(model.b_v),
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

    y_train_proba = model.predict_proba(X_train, X_aux_train, n_iter=20)
    E_theta_train = model.transform(X_train, X_aux_new=X_aux_train)['E_theta']

    _y_train_2d = y_train if y_train.ndim == 2 else y_train[:, np.newaxis]
    _proba_train_2d = y_train_proba if y_train_proba.ndim == 2 else y_train_proba[:, np.newaxis]

    # DEBUG: Compute logits and correlation with labels (use normalized theta)
    _theta_sums = E_theta_train.sum(axis=1, keepdims=True)
    _theta_norm_train = E_theta_train / np.maximum(_theta_sums, 1e-8)
    train_logits = _theta_norm_train @ np.array(model.mu_v).T
    if model.p_aux > 0 and model.mu_gamma is not None:
        _X_aux_debug = model._prepend_intercept(X_aux_train, n=X_aux_train.shape[0])
        train_logits = train_logits + _X_aux_debug @ model.mu_gamma.T
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

    train_metrics = train_metrics_all[label_columns[0]]
    y_train_pred = (_proba_train_2d[:, 0] > 0.5).astype(int)

    # Training metrics printed above; predictions saved in CSV below.

    # =========================================================================
    # STEP 6: Evaluate on Validation Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Validation Set Evaluation")
    print("=" * 80)

    y_val_proba = model.predict_proba(X_val, X_aux_val, n_iter=20)
    E_theta_val = model.transform(X_val, X_aux_new=X_aux_val)['E_theta']

    _y_val_2d = y_val if y_val.ndim == 2 else y_val[:, np.newaxis]
    _proba_val_2d = y_val_proba if y_val_proba.ndim == 2 else y_val_proba[:, np.newaxis]

    # Find optimal threshold on validation set (per outcome)
    optimal_thresholds = {}
    val_metrics_all = {}
    for k in range(n_outcomes):
        lname = label_columns[k]
        n_pos = int(_y_val_2d[:, k].sum())
        n_neg = len(_y_val_2d[:, k]) - n_pos
        if n_pos == 0 or n_neg == 0 or len(np.unique(_y_val_2d[:, k])) < 2:
            # Degenerate validation set (single class) — F1-based threshold
            # optimization is undefined.  Fall back to 0.5.
            opt_thr = 0.5
            print(f"  WARNING: validation set for [{lname}] has only one class "
                  f"(pos={n_pos}, neg={n_neg}). Using default threshold 0.5.")
        else:
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

    optimal_threshold = optimal_thresholds[label_columns[0]]
    val_metrics = val_metrics_all[label_columns[0]]
    y_val_pred = (_proba_val_2d[:, 0] > optimal_threshold).astype(int)

    # Validation metrics printed above; predictions saved in CSV below.

    # =========================================================================
    # STEP 6b: Fit Platt calibrators on validation set
    # =========================================================================
    calibrators = {}
    # Compute validation probabilities once, then slice per outcome
    from sklearn.linear_model import LogisticRegression as _PlattLR
    _probs_val_all = model.predict_proba(X_val, X_aux_val)
    for k in range(n_outcomes):
        lname = label_columns[k]
        if _probs_val_all.ndim > 1:
            _probs_k = _probs_val_all[:, k]
        else:
            _probs_k = _probs_val_all
        _probs_k = np.clip(_probs_k, 1e-7, 1 - 1e-7)
        _logits_k = np.log(_probs_k / (1 - _probs_k))
        _lr = _PlattLR(C=1e10, solver='lbfgs', max_iter=1000)
        _lr.fit(_logits_k.reshape(-1, 1), _y_val_2d[:, k])
        cal = {'method': 'platt', 'a': float(_lr.coef_[0, 0]), 'b': float(_lr.intercept_[0])}
        calibrators[lname] = cal
        print(f"Platt calibrator [{lname}]: a={cal['a']:.4f}, b={cal['b']:.4f}")

    # =========================================================================
    # STEP 7: Evaluate on Test Set
    # =========================================================================
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)

    # Get raw test probabilities, then apply per-outcome Platt calibration
    y_test_proba_raw = model.predict_proba(X_test, X_aux_test, n_iter=20)
    E_theta_test = model.transform(X_test, X_aux_new=X_aux_test)['E_theta']

    _y_test_2d = y_test if y_test.ndim == 2 else y_test[:, np.newaxis]
    _proba_raw_2d = y_test_proba_raw if y_test_proba_raw.ndim == 2 else y_test_proba_raw[:, np.newaxis]

    # Apply Platt calibration per outcome
    from scipy.special import expit as _expit_np
    _proba_test_2d = np.empty_like(_proba_raw_2d)
    for k in range(n_outcomes):
        lname = label_columns[k]
        cal = calibrators[lname]
        raw_k = np.clip(_proba_raw_2d[:, k], 1e-7, 1 - 1e-7)
        logits_k = np.log(raw_k / (1 - raw_k))
        _proba_test_2d[:, k] = _expit_np(cal['a'] * logits_k + cal['b'])
    y_test_proba = _proba_test_2d.squeeze()

    test_metrics_all = {}
    for k in range(n_outcomes):
        lname = label_columns[k]
        thr_k = optimal_thresholds[lname]
        print(f"Using optimal threshold [{lname}] from validation: {thr_k:.4f}")
        y_test_pred_k = (_proba_test_2d[:, k] > thr_k).astype(int)
        metrics_k = compute_metrics(_y_test_2d[:, k], y_test_pred_k, _proba_test_2d[:, k])
        test_metrics_all[lname] = metrics_k
        print(f"\nTest Results [{lname}] (cell-level, {len(_y_test_2d)} cells):")
        print(f"  Accuracy:  {metrics_k['accuracy']:.4f}")
        if 'auc' in metrics_k:
            print(f"  AUC:       {metrics_k['auc']:.4f}")
        print(f"  F1:        {metrics_k['f1']:.4f}")
        print(f"  Precision: {metrics_k['precision']:.4f}")
        print(f"  Recall:    {metrics_k['recall']:.4f}")

    test_metrics = test_metrics_all[label_columns[0]]
    y_test_pred = (_proba_test_2d[:, 0] > optimal_threshold).astype(int)

    # Test metrics printed above; predictions saved in CSV below.

    # =========================================================================
    # STEP 8: Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    prefix = 'vi'
    saved_files = save_results(
        model=model,
        output_dir=output_dir,
        gene_list=gene_list,
        splits=splits,
        prefix=prefix,
        compress=True,
        optimal_threshold=optimal_thresholds,
        program_names=pathway_names,
        mode=args.mode,
        label_columns=label_columns,
        aux_columns=aux_column_names,
        val_test_data={
            'X_val': X_val, 'X_aux_val': X_aux_val,
            'X_test': X_test, 'X_aux_test': X_aux_test,
        },
        cell_metadata=data.get('cell_metadata'),
    )

    # Predictions (per-outcome columns)
    def _build_pred_df(cell_ids, y_2d, proba_2d, thresholds_dict):
        d = {'cell_id': cell_ids}
        for k, lc in enumerate(label_columns):
            thr = thresholds_dict.get(lc, 0.5)
            d[f'true_{lc}'] = y_2d[:, k]
            d[f'prob_{lc}'] = proba_2d[:, k]
            d[f'pred_{lc}'] = (proba_2d[:, k] > thr).astype(int)
        return pd.DataFrame(d)

    train_thresholds = {lc: 0.5 for lc in label_columns}
    train_pred_df = _build_pred_df(splits['train'], _y_train_2d, _proba_train_2d, train_thresholds)
    train_pred_df.to_csv(output_dir / f'{prefix}_train_predictions.csv.gz', compression='gzip', index=False)

    val_pred_df = _build_pred_df(splits['val'], _y_val_2d, _proba_val_2d, optimal_thresholds)
    val_pred_df.to_csv(output_dir / f'{prefix}_val_predictions.csv.gz', compression='gzip', index=False)

    test_pred_df = _build_pred_df(splits['test'], _y_test_2d, _proba_test_2d, optimal_thresholds)
    test_pred_df.to_csv(output_dir / f'{prefix}_test_predictions.csv.gz', compression='gzip', index=False)
    print(f"Saved predictions to {output_dir}")

    # Save Platt calibrators
    import json as _json
    cal_path = output_dir / f'{prefix}_calibrators.json'
    with open(cal_path, 'w') as f:
        _json.dump(calibrators, f, indent=2)
    print(f"Saved calibrators to {cal_path}")

    # =========================================================================
    # STEP 7b: Patient-Level Evaluation (when --patient-column is set)
    # =========================================================================
    patient_metrics_all = {}  # {split_name: {label: metrics_dict}}
    if patient_split is not None and args.patient_column is not None:
        print("\n" + "=" * 80)
        print("Patient-Level Evaluation (mean-pooled cell predictions)")
        print("=" * 80)

        cell_metadata = data.get('cell_metadata')
        pcol = args.patient_column

        def _patient_level_metrics(cell_ids, y_2d, proba_2d, thresholds_dict, split_name):
            """Aggregate cell predictions to patient level and compute metrics."""
            # Map cell_id -> patient_id
            pid_series = cell_metadata.loc[cell_ids, pcol].astype(str)
            results = {}
            patient_pred_rows = []

            for k, lc in enumerate(label_columns):
                thr = thresholds_dict.get(lc, 0.5)
                df = pd.DataFrame({
                    'cell_id': cell_ids,
                    'patient_id': pid_series.values,
                    'y_true': y_2d[:, k],
                    'prob': proba_2d[:, k],
                })
                # Pool: mean probability per patient
                patient_df = df.groupby('patient_id').agg(
                    y_true=('y_true', 'first'),  # same for all cells
                    prob_mean=('prob', 'mean'),
                    n_cells=('cell_id', 'count'),
                ).reset_index()

                y_pat = patient_df['y_true'].values
                prob_pat = patient_df['prob_mean'].values
                pred_pat = (prob_pat > thr).astype(int)

                m = compute_metrics(y_pat, pred_pat, prob_pat)
                results[lc] = m

                n_pat = len(patient_df)
                print(f"\n  [{split_name}] Patient-level [{lc}] ({n_pat} patients):")
                print(f"    Accuracy:  {m['accuracy']:.4f}")
                if 'auc' in m:
                    print(f"    AUC:       {m['auc']:.4f}")
                print(f"    F1:        {m['f1']:.4f}")
                print(f"    Precision: {m['precision']:.4f}")
                print(f"    Recall:    {m['recall']:.4f}")
                print(f"    Cells/patient: mean={patient_df['n_cells'].mean():.1f}, "
                      f"min={patient_df['n_cells'].min()}, max={patient_df['n_cells'].max()}")

                patient_df['label'] = lc
                patient_df['split'] = split_name
                patient_pred_rows.append(patient_df)

            return results, patient_pred_rows

        all_patient_pred_rows = []
        for split_name, cell_ids, y_2d, proba_2d, thresholds in [
            ('train', splits['train'], _y_train_2d, _proba_train_2d, {lc: 0.5 for lc in label_columns}),
            ('val', splits['val'], _y_val_2d, _proba_val_2d, optimal_thresholds),
            ('test', splits['test'], _y_test_2d, _proba_test_2d, optimal_thresholds),
        ]:
            pm, pred_rows = _patient_level_metrics(cell_ids, y_2d, proba_2d, thresholds, split_name)
            patient_metrics_all[split_name] = pm
            all_patient_pred_rows.extend(pred_rows)

        # Save patient-level predictions
        patient_pred_df = pd.concat(all_patient_pred_rows, ignore_index=True)
        patient_pred_path = output_dir / f'{prefix}_patient_predictions.csv.gz'
        patient_pred_df.to_csv(patient_pred_path, compression='gzip', index=False)
        print(f"\nSaved patient-level predictions to {patient_pred_path}")

        # --- Summary: Cell-level vs Patient-level AUC ---
        print("\n" + "-" * 60)
        print("AUC Summary: Cell-level vs Patient-level")
        print("-" * 60)
        print(f"  {'Label':<25} {'Split':<7} {'Cell AUC':>10} {'Patient AUC':>12} {'N_cells':>8} {'N_patients':>11}")
        for split_name in ['val', 'test']:
            cell_m = (val_metrics_all if split_name == 'val' else test_metrics_all)
            pat_m = patient_metrics_all.get(split_name, {})
            _split_cells = len(splits[split_name])
            for lc in label_columns:
                c_auc = cell_m.get(lc, {}).get('auc', float('nan'))
                p_auc = pat_m.get(lc, {}).get('auc', float('nan'))
                n_pat = 'N/A'
                if split_name in patient_metrics_all:
                    # Count patients from the patient predictions
                    _pat_rows = [r for r in all_patient_pred_rows
                                 if r['split'].iloc[0] == split_name and r['label'].iloc[0] == lc]
                    n_pat = str(len(_pat_rows[0])) if _pat_rows else 'N/A'
                print(f"  {lc:<25} {split_name:<7} {c_auc:>10.4f} {p_auc:>12.4f} {_split_cells:>8} {n_pat:>11}")
        print("-" * 60)

    # =========================================================================
    # STEP 7c: Decomposed Diagnostics — cov-only vs factor vs full
    # =========================================================================
    if patient_split is not None and args.patient_column is not None:
        from scipy.special import expit as _sigmoid
        cell_metadata = data.get('cell_metadata')
        pcol = args.patient_column

        print("\n" + "=" * 80)
        print("DIAGNOSTIC: Logit Decomposition & Patient-Level Ranking Analysis")
        print("=" * 80)

        for split_name, cell_ids, X_gex, X_aux, y_2d, proba_2d in [
            ('train', splits['train'], X_train, X_aux_train, _y_train_2d, _proba_train_2d),
            ('val', splits['val'], X_val, X_aux_val, _y_val_2d, _proba_val_2d),
            ('test', splits['test'], X_test, X_aux_test, _y_test_2d, _proba_test_2d),
        ]:
            print(f"\n--- {split_name.upper()} ({len(cell_ids)} cells) ---")

            # Compute E[theta] for this split
            E_theta_split = model.transform(X_gex, X_aux_new=X_aux)['E_theta']
            X_aux_prep = model._prepend_intercept(
                np.asarray(X_aux, dtype=np.float32), n=len(cell_ids))

            # Logit decomposition (use normalized theta)
            logit_cov = np.array(X_aux_prep @ model.mu_gamma.T)      # (n, kappa)
            _ts = E_theta_split.sum(axis=1, keepdims=True)
            _tn = E_theta_split / np.maximum(_ts, 1e-8)
            logit_theta = np.array(_tn @ model.mu_v.T)               # (n, kappa)
            logit_full = logit_cov + logit_theta

            for k in range(n_outcomes):
                lname = label_columns[k]
                lc = logit_cov[:, k] if logit_cov.ndim > 1 else logit_cov
                lt = logit_theta[:, k] if logit_theta.ndim > 1 else logit_theta
                lf = logit_full[:, k] if logit_full.ndim > 1 else logit_full
                y_k = y_2d[:, k] if y_2d.ndim > 1 else y_2d
                p_full = proba_2d[:, k] if proba_2d.ndim > 1 else proba_2d

                # --- Logit magnitude comparison ---
                print(f"\n  [{lname}] Logit Decomposition:")
                print(f"    logit_cov:   mean={lc.mean():.4f}  SD={lc.std():.4f}  "
                      f"range=[{lc.min():.4f}, {lc.max():.4f}]  "
                      f"p5/p50/p95=[{np.percentile(lc,5):.4f}, {np.percentile(lc,50):.4f}, {np.percentile(lc,95):.4f}]")
                print(f"    logit_theta: mean={lt.mean():.4f}  SD={lt.std():.4f}  "
                      f"range=[{lt.min():.4f}, {lt.max():.4f}]  "
                      f"p5/p50/p95=[{np.percentile(lt,5):.4f}, {np.percentile(lt,50):.4f}, {np.percentile(lt,95):.4f}]")
                print(f"    logit_full:  mean={lf.mean():.4f}  SD={lf.std():.4f}  "
                      f"range=[{lf.min():.4f}, {lf.max():.4f}]")
                sd_ratio = lt.std() / max(lc.std(), 1e-8)
                print(f"    SD ratio (theta/cov): {sd_ratio:.4f}"
                      + ("  *** FACTOR BLOCK DOMINATES ***" if sd_ratio > 1.0 else ""))

                # --- AUC inversion test ---
                n_unique = len(np.unique(y_k))
                if n_unique > 1:
                    from sklearn.metrics import roc_auc_score
                    p_cov = _sigmoid(lc)
                    p_theta = _sigmoid(lt)
                    p_f = _sigmoid(lf)
                    auc_full = roc_auc_score(y_k, p_full)
                    auc_inv = roc_auc_score(y_k, 1.0 - p_full)
                    auc_cov = roc_auc_score(y_k, p_cov)
                    auc_theta = roc_auc_score(y_k, p_theta)
                    print(f"    Cell AUC(p):   {auc_full:.4f}   AUC(1-p): {auc_inv:.4f}"
                          + ("  *** INVERSION DETECTED ***" if auc_inv > 0.9 and auc_full < 0.1 else ""))
                    print(f"    Cell AUC cov-only: {auc_cov:.4f}   theta-only: {auc_theta:.4f}")

                # --- Patient-level decomposed table ---
                pid_series = cell_metadata.loc[cell_ids, pcol].astype(str)
                pdf = pd.DataFrame({
                    'patient': pid_series.values,
                    'y_true': y_k,
                    'logit_cov': lc,
                    'logit_theta': lt,
                    'logit_full': lf,
                    'prob_full': p_full,
                })
                pat = pdf.groupby('patient').agg(
                    y_true=('y_true', 'first'),
                    n_cells=('patient', 'count'),
                    logit_cov=('logit_cov', 'mean'),
                    logit_theta=('logit_theta', 'mean'),
                    logit_full=('logit_full', 'mean'),
                    prob_full=('prob_full', 'mean'),
                ).reset_index()
                pat['prob_cov'] = _sigmoid(pat['logit_cov'].values)
                pat['delta_logit'] = pat['logit_theta']
                pat['rank_cov'] = pat['prob_cov'].rank(ascending=False).astype(int)
                pat['rank_full'] = pat['prob_full'].rank(ascending=False).astype(int)

                n_pat = len(pat)
                print(f"\n    Patient-level table ({n_pat} patients):")
                print(f"    {'patient':<15} {'y':>2} {'cells':>6} {'logit_cov':>10} {'logit_θ':>10} "
                      f"{'logit_full':>10} {'p(cov)':>7} {'p(full)':>7} {'Δlogit':>8} "
                      f"{'rk_cov':>6} {'rk_full':>7}")
                for _, r in pat.sort_values('rank_full').iterrows():
                    flipped = (r['rank_cov'] != r['rank_full'])
                    marker = " ←FLIP" if flipped else ""
                    print(f"    {r['patient']:<15} {int(r['y_true']):>2} {int(r['n_cells']):>6} "
                          f"{r['logit_cov']:>10.4f} {r['logit_theta']:>10.4f} "
                          f"{r['logit_full']:>10.4f} {r['prob_cov']:>7.4f} {r['prob_full']:>7.4f} "
                          f"{r['delta_logit']:>8.4f} {int(r['rank_cov']):>6} {int(r['rank_full']):>7}{marker}")

                # Patient-level AUC: cov-only vs full
                if n_pat > 1 and len(np.unique(pat['y_true'])) > 1:
                    pat_auc_cov = roc_auc_score(pat['y_true'], pat['prob_cov'])
                    pat_auc_full = roc_auc_score(pat['y_true'], pat['prob_full'])
                    pat_auc_inv = roc_auc_score(pat['y_true'], 1.0 - pat['prob_full'])
                    print(f"\n    Patient AUC  cov-only: {pat_auc_cov:.4f}  "
                          f"full: {pat_auc_full:.4f}  full(1-p): {pat_auc_inv:.4f}")
                    if pat_auc_full < pat_auc_cov - 0.05:
                        print(f"    *** FACTOR BLOCK HURTS patient AUC by "
                              f"{pat_auc_cov - pat_auc_full:.4f} ***")

    # Save consolidated metrics CSV (replaces separate pkl files)
    metrics_rows = []
    for split_name, metrics_dict, thresholds in [
        ('train', train_metrics_all, {lc: 0.5 for lc in label_columns}),
        ('val', val_metrics_all, optimal_thresholds),
        ('test', test_metrics_all, optimal_thresholds),
    ]:
        for lc, m in metrics_dict.items():
            row = {'split': split_name, 'label': lc, 'threshold': thresholds.get(lc, 0.5),
                   'level': 'cell'}
            row.update(m)
            metrics_rows.append(row)

    # Add patient-level metrics rows
    for split_name, thresholds in [
        ('train', {lc: 0.5 for lc in label_columns}),
        ('val', optimal_thresholds),
        ('test', optimal_thresholds),
    ]:
        if split_name in patient_metrics_all:
            for lc, m in patient_metrics_all[split_name].items():
                row = {'split': split_name, 'label': lc, 'threshold': thresholds.get(lc, 0.5),
                       'level': 'patient'}
                row.update(m)
                metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / f'{prefix}_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    # =========================================================================
    # STEP 8: Model Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Model Summary")
    print("=" * 80)

    print_model_summary(model, gene_list)

    # Save diagnostic plots
    if hasattr(model, 'diagnostics_') and model.diagnostics_ is not None:
        plot_diagnostics(model.diagnostics_, save_dir=output_dir)

    # =========================================================================
    # Profiling Output (if enabled)
    # =========================================================================
    if profiler is not None:
        profiler.disable()

        profile_path = output_dir / f'{prefix}_profile.prof'
        profiler.dump_stats(str(profile_path))
        print(f"\n[PROFILER] Raw profile data saved to: {profile_path}")
        print(f"[PROFILER] Analyze with: python -m pstats {profile_path}")
        print(f"[PROFILER] Or visualize with: snakeviz {profile_path}")

        print("\n" + "=" * 80)
        print(f"PROFILE SUMMARY (sorted by {args.profile_sort}, top {args.profile_lines} functions)")
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
    print(f"  - {output_dir / f'{prefix}_train_predictions.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_val_predictions.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_test_predictions.csv.gz'}")
    print(f"  - {output_dir / f'{prefix}_metrics.csv'}")

    print(f"\nNext steps:")
    print(f"  1. Load gene programs: pd.read_csv('{prefix}_gene_programs.csv.gz')")
    print("  2. Identify top genes per program and run pathway enrichment")
    print("  3. Load theta matrices to analyze sample-level program activity")
    print("  4. Correlate theta values with phenotypes/outcomes")


if __name__ == '__main__':
    main()
