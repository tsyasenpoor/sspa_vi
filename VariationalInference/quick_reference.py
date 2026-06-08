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
    parser.add_argument(
        '--require-prefix',
        type=str,
        default=None,
        help='If set, only pathways whose names start with this prefix are kept '
             '(e.g. REACTOME). Applied inside load_pathways before size filters.'
    )
    parser.add_argument(
        '--no-adaptive-filter',
        action='store_true',
        help='Disable the adaptive overlap-size filter inside load_pathways '
             '(keep every pathway regardless of how many genes intersect the data).'
    )
    parser.add_argument(
        '--excluded-keywords',
        nargs='*',
        default=[],
        metavar='KW',
        help='Pathway name substrings to exclude (case-insensitive). '
             'Defaults to empty — no keyword filtering. '
             'Example: --excluded-keywords ADME DRUG'
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
        '--regression-weight',
        type=float,
        default=1.0,
        help='Base regression_weight multiplier. Auto-scaled internally by '
             'nnz/n (see vi_cavi.py:_initialize), so effective rw = base * nnz/n. '
             'Default 1.0 gives effective rw ≈ nnz/n which on typical scRNA is '
             '~600-1000, well below p — supervision under-engaged. Calibration '
             'target: effective rw ≈ p (e.g. base=15 → ~9000 on 8000-cell × '
             '8674-gene data). For the hierarchical (sc) model, supervision is '
             'summed over G ≪ n so larger base values are required; sweep '
             'independently from flat.'
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
        help='Poisson-only warmup iterations before regression (v/gamma/omega) updates begin. '
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
        default=0.0,
        help='Target library size for normalization. Default 0 = auto = median '
             'library size (counts-per-median, scHPF-style; data-adaptive).'
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
        help='Seed for MODEL initialization only (default: None = auto from time). '
             'Does NOT affect the train/val/test split — that is controlled by '
             '--split-seed. Passed to CAVI(random_state=...), which reseeds '
             'np.random locally at model construction.'
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
        '--diag-every-iter',
        action='store_true',
        help='Diagnostic: print mu_v[:, :3] at EVERY iteration (not just every '
             '--check-freq) so the CAVI v-update oscillation period can be '
             'measured. Small print overhead per iteration.'
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
    parser.add_argument(
        '--split-seed',
        type=int,
        default=0,
        help='Seed for train/val/test splitting (deterministic). Decoupled from '
             '--seed so the same split can be reused across different VI init seeds.'
    )

    # Patient-level splitting (prevents donor leakage in single-cell data)
    parser.add_argument(
        '--patient-column',
        type=str,
        default=None,
        help='Column in adata.obs identifying patients/donors (e.g., sampleID). '
             'When set, train/val/test splits are performed at the patient level '
             'so no patient appears in more than one split. Also enables '
             'patient-level evaluation metrics via mean-pooled cell predictions. '
             'Required for --model-type=sc.'
    )

    # Hierarchical (single-cell) model — Chapter 7
    parser.add_argument(
        '--model-type',
        type=str,
        default='flat',
        choices=['flat', 'sc'],
        help='flat = Chapter 3 (cell-level supervision via θ_i; bulk + naive '
             'single-cell). sc = Chapter 7 hierarchical (patient-level '
             'supervision via Θ_g; cell-type-aware via ζ). The two models '
             'are sibling classes sharing the gene-block + PG-CAVI kernel.'
    )
    parser.add_argument(
        '--cell-type-column',
        type=str,
        default=None,
        help='Column in adata.obs holding cell-type labels. Required for '
             '--model-type=sc. Labels are mapped to dense int indices in '
             '[0, T) by first-seen order.'
    )
    parser.add_argument(
        '--unseen-celltype',
        type=str,
        default='error',
        choices=['error', 'prior'],
        help='Behavior on test-time cells with a cell type not present in '
             'training: error (default — tripwire for label-naming bugs) '
             'or prior (back off to the cohort prior mean E[ζ] = α_ζ/λ_ζ).'
    )
    parser.add_argument(
        '--alpha-Theta',
        type=float,
        default=1.0,
        help='[sc] Gamma shape prior on Θ_{gℓ} (patient-level program activity).'
    )
    parser.add_argument(
        '--lambda-Theta',
        type=float,
        default=1.0,
        help='[sc] Gamma rate prior on Θ_{gℓ}.'
    )
    parser.add_argument(
        '--alpha-zeta',
        type=float,
        default=1.0,
        help='[sc] Gamma shape prior on ζ_{tℓ} (cell-type × program affinity).'
    )
    parser.add_argument(
        '--lambda-zeta',
        type=float,
        default=1.0,
        help='[sc] Gamma rate prior on ζ_{tℓ}.'
    )

    return parser.parse_args()


def _run_sc_pipeline(args, label_columns):
    """Minimal hierarchical (Chapter 7) fit-and-save pipeline.

    Loads the dataset, builds patient_ids + cell_type_ids from adata.obs,
    instantiates CAVIHierarchical, runs Algorithm 2, and saves the new
    hier outputs (Θ per patient, ζ per cell type) alongside the standard
    DRGP artifacts (gene programs, υ, γ). Cell-level evaluation/prediction
    is intentionally absent — for sc, prediction lives at the patient
    level via Θ and is handled by a separate downstream eval script
    (using transform_hier for inductive fold-in on held-out patients).
    """
    import json
    import gzip
    from pathlib import Path
    import anndata as ad
    import pandas as pd

    from VariationalInference.vi_cavi_sc import CAVIHierarchical
    from VariationalInference.data_loader import DataLoader
    from VariationalInference.utils import load_pathways

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + preprocess (patient-grouped split via DataLoader).
    print("\n[sc] Loading data via DataLoader (patient-grouped split).")
    loader = DataLoader(
        data_path=args.data,
        gene_annotation_path=args.gene_annotation,
        cache_dir=args.cache_dir,
        use_cache=True,
        verbose=True,
    )
    # NOTE: stratify_by intentionally None for patient-grouped splits — the
    # DataLoader's patient-split path has an indexer bug when stratify_by
    # is set with int-valued patient_id (TypeError on .loc[str_keys]).
    # Patient-level class balance is handled by use_class_weights in CAVIHierarchical.
    data = loader.load_and_preprocess(
        label_column=label_columns,
        aux_columns=args.aux_columns,
        train_ratio=0.7,
        val_ratio=0.15,
        stratify_by=None,
        min_cells_expressing=0.001,
        layer=args.layer,
        convert_to_ensembl=True,
        filter_protein_coding=args.gene_annotation is not None,
        random_state=args.split_seed,
        normalize=args.normalize,
        normalize_target_sum=args.normalize_target_sum,
        normalize_method=args.normalize_method,
        patient_column=args.patient_column,
    )
    X_train, X_aux_train, y_train = data['train']
    X_val, X_aux_val, y_val = data['val']
    splits = data['splits']
    gene_list = data.get('gene_list')
    if gene_list is None:
        gene_list = [f'gene_{j}' for j in range(X_train.shape[1])]
    # DataLoader returns 'cell_metadata' (obs DataFrame indexed by cell name)
    # for all union cells, and 'splits' contains cell-id LISTS per split.
    cell_meta_all = data.get('cell_metadata')
    if cell_meta_all is None:
        raise SystemExit("DataLoader did not return cell_metadata — required for sc pipeline.")
    train_cell_ids = splits.get('train')
    val_cell_ids = splits.get('val')
    if train_cell_ids is None:
        raise SystemExit("splits['train'] missing from DataLoader output.")
    cell_meta_train = cell_meta_all.loc[train_cell_ids]
    cell_meta_val = cell_meta_all.loc[val_cell_ids] if val_cell_ids is not None else None

    # 2. Build patient_ids and cell_type_ids for train cells.
    if args.cell_type_column not in cell_meta_train.columns:
        raise SystemExit(
            f"--cell-type-column '{args.cell_type_column}' not found in adata.obs. "
            f"Available: {list(cell_meta_train.columns)}"
        )
    pat_series = cell_meta_train[args.patient_column].astype(str)
    ct_series = cell_meta_train[args.cell_type_column].astype(str)
    pat_uniq = pat_series.unique()
    ct_uniq = ct_series.unique()
    pat_to_idx = {p: i for i, p in enumerate(pat_uniq)}
    ct_to_idx = {c: i for i, c in enumerate(ct_uniq)}
    patient_ids = pat_series.map(pat_to_idx).values.astype(np.int32)
    cell_type_ids = ct_series.map(ct_to_idx).values.astype(np.int32)
    print(f"[sc] {len(pat_uniq)} patients, {len(ct_uniq)} cell types, "
          f"{len(patient_ids)} train cells, {X_train.shape[1]} genes.")
    print(f"[sc] cell types: {list(ct_uniq)}")

    # Val patient/cell-type ids (for HO-LL fold-in early stopping).
    patient_ids_val = None
    cell_type_ids_val = None
    if cell_meta_val is not None and len(cell_meta_val) > 0:
        pat_val_series = cell_meta_val[args.patient_column].astype(str)
        ct_val_series = cell_meta_val[args.cell_type_column].astype(str)
        # Val patients are disjoint from train (group-split); assign fresh ids
        # in [0, G_val) — transform_hier re-densifies them anyway.
        val_pat_uniq = pat_val_series.unique()
        val_pat_to_idx = {p: i for i, p in enumerate(val_pat_uniq)}
        patient_ids_val = pat_val_series.map(val_pat_to_idx).values.astype(np.int32)
        # Cell types share the train mapping; unseen types back off via
        # transform_hier(unseen_celltype=args.unseen_celltype).
        unseen_ct = sorted(set(ct_val_series.unique()) - set(ct_to_idx.keys()))
        if unseen_ct:
            print(f"[sc] val cells include unseen cell types: {unseen_ct} "
                  f"(handled via unseen_celltype={args.unseen_celltype!r}).")
            next_id = len(ct_to_idx)
            for c in unseen_ct:
                ct_to_idx[c] = next_id
                next_id += 1
        cell_type_ids_val = ct_val_series.map(ct_to_idx).values.astype(np.int32)
        print(f"[sc] val: {len(val_pat_uniq)} patients, {len(patient_ids_val)} cells.")

    # 2b. Pathway loading (masked / pathway_init / combined modes).
    pathway_mask = None
    pathway_names = None
    n_factors = args.n_factors
    n_pathway_factors = None

    if args.mode in ('masked', 'pathway_init', 'combined'):
        print(f"\n[sc] Loading pathways for {args.mode.upper()} mode.")
        valid_gene_list = [g for g in gene_list if isinstance(g, str) and g]
        n_ensg = sum(1 for g in valid_gene_list if g.startswith('ENSG'))
        genes_are_ensembl = (n_ensg > len(valid_gene_list) * 0.5) if valid_gene_list else False
        pathway_mat, pathway_names_raw, pathway_genes = load_pathways(
            gmt_path=args.pathway_file,
            convert_to_ensembl=genes_are_ensembl,
            species='human',
            gene_filter=valid_gene_list,
            min_genes=args.pathway_min_genes,
            max_genes=args.pathway_max_genes,
            cache_dir=args.cache_dir,
            use_cache=True,
            excluded_keywords=args.excluded_keywords,
            require_prefix=args.require_prefix,
            disable_adaptive_filter=args.no_adaptive_filter,
        )
        gene_to_expr_idx = {g: i for i, g in enumerate(gene_list)}
        gene_to_pathway_idx = {g: i for i, g in enumerate(pathway_genes)}
        common_genes = set(valid_gene_list) & set(pathway_genes)
        if len(common_genes) < 10:
            raise ValueError(
                f"Too few common genes ({len(common_genes)}) between expression and pathways."
            )
        n_pathways = pathway_mat.shape[0]
        n_genes_expr = len(gene_list)
        pathway_mask = np.zeros((n_pathways, n_genes_expr), dtype=np.float32)
        for gene in common_genes:
            pathway_mask[:, gene_to_expr_idx[gene]] = pathway_mat[:, gene_to_pathway_idx[gene]]
        pathways_per_gene = pathway_mask.sum(axis=0)
        print(f"[sc] pathway matrix: {n_pathways} × {n_genes_expr}, "
              f"density={pathway_mask.mean()*100:.2f}%, "
              f"genes covered={int((pathways_per_gene > 0).sum())}")

        # Masked mode: restrict X to pathway genes (matches flat pipeline).
        if args.mode == 'masked':
            pathway_gene_idx = np.where(pathways_per_gene > 0)[0]
            X_train = X_train[:, pathway_gene_idx]
            if X_val is not None:
                X_val = X_val[:, pathway_gene_idx]
            pathway_mask = pathway_mask[:, pathway_gene_idx]
            gene_list = [gene_list[i] for i in pathway_gene_idx]
            print(f"[sc] [MASKED] restricted to {len(pathway_gene_idx)} pathway genes; "
                  f"X_train now {X_train.shape}")

        pathway_names = pathway_names_raw

        if args.mode == 'combined':
            n_drgps = args.n_drgps
            n_factors = n_pathways + n_drgps
            drgp_mask = np.ones((n_drgps, pathway_mask.shape[1]), dtype=np.float32)
            pathway_mask = np.vstack([pathway_mask, drgp_mask])
            pathway_names = pathway_names_raw + [f"DRGP_{i+1}" for i in range(n_drgps)]
            n_pathway_factors = n_pathways
            print(f"[sc] [COMBINED] n_factors = {n_pathways} pathways + {n_drgps} free = {n_factors}")
        else:
            n_factors = n_pathways
            if args.n_factors != n_pathways:
                print(f"[sc]   NOTE: --n-factors={args.n_factors} overridden by pathway count ({n_pathways})")

    # 3. Build model.
    a_val = args.alpha_theta if args.alpha_theta is not None else args.a
    c_val = args.alpha_beta if args.alpha_beta is not None else args.c
    ap_val = args.alpha_xi if args.alpha_xi is not None else 1.0
    cp_val = args.alpha_eta if args.alpha_eta is not None else 1.0
    model = CAVIHierarchical(
        n_factors=n_factors,
        a=a_val, ap=ap_val, c=c_val, cp=cp_val,
        b_v=args.b_v,
        sigma_gamma=args.sigma_gamma,
        regression_weight=args.regression_weight,
        use_intercept=not args.no_intercept,
        random_state=args.seed,
        mode=args.mode,
        pathway_mask=pathway_mask,
        pathway_names=pathway_names,
        n_pathway_factors=n_pathway_factors,
        alpha_pi=args.alpha_pi,
        beta_pi_scale=args.beta_pi_scale,
        alpha_Theta=args.alpha_Theta,
        lambda_Theta=args.lambda_Theta,
        alpha_zeta=args.alpha_zeta,
        lambda_zeta=args.lambda_zeta,
    )

    # 4. Fit.
    print(f"\n[sc] Fit: max_iter={args.max_iter}, v_warmup={args.v_warmup}, "
          f"regression_weight base={args.regression_weight}")
    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_aux_train=X_aux_train,
        X_val=X_val,
        y_val=y_val,
        X_aux_val=X_aux_val,
        patient_ids=patient_ids,
        cell_type_ids=cell_type_ids,
        patient_ids_val=patient_ids_val,
        cell_type_ids_val=cell_type_ids_val,
        max_iter=args.max_iter,
        check_freq=args.check_freq,
        tol=args.tol,
        v_warmup=args.v_warmup,
        verbose=args.verbose,
        early_stopping=args.early_stopping,
    )

    # 5. Save sc-specific artifacts + standard gene programs.
    print(f"\n[sc] Saving artifacts to {out_dir}/")
    np.savez_compressed(
        out_dir / 'vi_model_params.npz',
        E_beta=np.asarray(model.E_beta),
        r_beta=np.asarray(model.r_beta),
        mu_v=np.asarray(model.mu_v),
        sigma_v_diag=np.asarray(model.sigma_v_diag),
        mu_gamma=np.asarray(model.mu_gamma),
        Sigma_gamma=np.asarray(model.Sigma_gamma),
        E_Theta=np.asarray(model.E_Theta),
        a_Theta=np.asarray(model.a_Theta),
        b_Theta=np.asarray(model.b_Theta),
        E_zeta=np.asarray(model.E_zeta),
        a_zeta=np.asarray(model.a_zeta),
        b_zeta=np.asarray(model.b_zeta),
        c_pg=np.asarray(model.c_pg),
        wbar=np.asarray(model.wbar),
        n_factors=n_factors,
        G=model.G, T=model.T, n=model.n, p=model.p, p_aux=model.p_aux,
        elbo_history=np.array(model.elbo_history_, dtype=object),
    )

    # Theta (G × K) — patient-level program activity.
    Theta_df = pd.DataFrame(
        np.asarray(model.E_Theta),
        index=[f'Patient_{p}' for p in pat_uniq],
        columns=[f'GP{k+1}' for k in range(n_factors)],
    )
    Theta_df.to_csv(out_dir / 'vi_patient_activities.csv.gz', compression='gzip')

    # zeta (T × K) — cell-type × program affinity.
    zeta_df = pd.DataFrame(
        np.asarray(model.E_zeta),
        index=list(ct_uniq),
        columns=[f'GP{k+1}' for k in range(n_factors)],
    )
    zeta_df.to_csv(out_dir / 'vi_celltype_affinities.csv.gz', compression='gzip')

    # Gene programs (E[β] × factor, with υ alongside) — flat-compatible format.
    # gene_list reflects any masked-mode gene restriction applied above.
    gp_df = pd.DataFrame(
        np.asarray(model.E_beta).T,    # (K, p)
        index=[f'GP{k+1}' for k in range(n_factors)],
        columns=gene_list,
    )
    for k_idx, k_name in enumerate(label_columns):
        gp_df.insert(0, f'v_weight_{k_name}', np.asarray(model.mu_v)[k_idx])
    gp_df.to_csv(out_dir / 'vi_gene_programs.csv.gz', compression='gzip')

    # Run summary.
    summary = {
        'model_type': 'sc (hierarchical, Chapter 7)',
        'hyperparameters': {
            'n_factors': n_factors,
            'a': a_val, 'c': c_val,
            'b_v': args.b_v,
            'sigma_gamma': args.sigma_gamma,
            'regression_weight_base': args.regression_weight,
            'regression_weight_effective': float(model.regression_weight),
            'alpha_Theta': args.alpha_Theta, 'lambda_Theta': args.lambda_Theta,
            'alpha_zeta': args.alpha_zeta, 'lambda_zeta': args.lambda_zeta,
        },
        'data_shapes': {
            'n_cells': model.n, 'n_genes': model.p,
            'n_patients': model.G, 'n_cell_types': model.T,
            'kappa': model.kappa, 'p_aux': model.p_aux,
        },
        'training': {
            'final_elbo': model.elbo_history_[-1][1] if model.elbo_history_ else None,
            'n_iterations': model.elbo_history_[-1][0] if model.elbo_history_ else 0,
        },
        'label_columns': label_columns,
        'aux_columns': args.aux_columns,
        'patient_column': args.patient_column,
        'cell_type_column': args.cell_type_column,
    }
    with gzip.open(out_dir / 'vi_summary.json.gz', 'wt') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[sc] Done. Final ELBO = {summary['training']['final_elbo']}")


def main():
    """Main function demonstrating the CAVI workflow."""
    import cProfile
    import pstats
    import io

    # =========================================================================
    # STEP 1: Parse Arguments
    # =========================================================================
    args = parse_args()

    # =========================================================================
    # STEP 1.5: Resolve seeds (model init decoupled from data split)
    # =========================================================================
    # --seed controls MODEL initialization only. CAVI(random_state=args.seed)
    # reseeds np.random locally at model construction (vi_cavi.py:186-188).
    # --split-seed controls the train/val/test split (default 0), passed
    # explicitly to DataLoader. No global RNG is set here so pre-model code
    # paths cannot accidentally leak --seed into data randomization.
    if args.seed is None:
        import time
        args.seed = int(time.time() * 1000) % (2**32)
    print(f"[SEED] Model init seed = {args.seed}  "
          f"(data split seed = {args.split_seed}, decoupled)")

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
    print(f"  random_seed:  {args.seed}")
    print(f"  output_dir:   {args.output_dir}")
    if args.mode in ['masked', 'pathway_init', 'combined']:
        print(f"  pathway_file: {args.pathway_file}")
        print(f"  pathway_size: [{args.pathway_min_genes}, {args.pathway_max_genes}]")
        kw_str = ', '.join(args.excluded_keywords) if args.excluded_keywords else "(none)"
        print(f"  excl_keywords:{kw_str}")
        print(f"  adaptive_filt:{'disabled' if args.no_adaptive_filter else 'enabled'}")
    if args.normalize:
        print(f"  normalize:    True (target_sum={args.normalize_target_sum:.0f}, method={args.normalize_method})")

    # =========================================================================
    # STEP 2: Import Modules
    # =========================================================================
    # Factory dispatch — flat (Ch.3) vs hierarchical (Ch.7). The hier path
    # short-circuits the flat pipeline and runs its own minimal fit+save below,
    # because the cell-level evaluation downstream (per-cell predictions,
    # decomposed AUC, etc.) doesn't apply when supervision is at the patient
    # level (predictions live on Θ, not θ).
    if args.model_type == 'sc':
        if not args.patient_column:
            raise SystemExit("--model-type=sc requires --patient-column.")
        if not args.cell_type_column:
            raise SystemExit("--model-type=sc requires --cell-type-column.")
        print("[METHOD] Using CAVIHierarchical (Chapter 7, patient-level supervision via Θ).")
        _run_sc_pipeline(args, label_columns)
        return
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
        random_state=args.split_seed,
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
            use_cache=True,
            excluded_keywords=args.excluded_keywords,
            require_prefix=args.require_prefix,
            disable_adaptive_filter=args.no_adaptive_filter,
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
        regression_weight=args.regression_weight,
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
        diag_every_iter=args.diag_every_iter,
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
    v_stability_path = output_dir / f'v_vector_seed{args.seed}.npy'
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
    _mu_v_2d = np.array(model.mu_v)
    train_logits = _theta_norm_train @ _mu_v_2d.T
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
