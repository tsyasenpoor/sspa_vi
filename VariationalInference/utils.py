"""
Utility Functions for Variational Inference
=============================================

This module provides generic utility functions for the VI pipeline.

Functions:
- Caching: save_cache, load_cache
- Array utilities: to_dense_array
- Normalization: size_factor_normalize
- Metrics: compute_metrics
- Results handling: save_results, load_model
- Analysis: get_top_genes_per_program
"""

import os
import sys
import pickle
import gzip
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


from typing import Optional, Dict, Any, List, Tuple, Union
from VariationalInference.gene_convertor import *



# =============================================================================
# Caching Utilities
# =============================================================================

def save_cache(data: Any, cache_file: Union[str, Path]) -> None:
    """
    Save data to a pickle cache file.

    Parameters
    ----------
    data : any
        Data to cache.
    cache_file : str or Path
        Path to cache file.
    """
    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved cached data to {cache_file}")

def load_cache(cache_file: Union[str, Path]) -> Optional[Any]:
    """
    Load data from a pickle cache file.

    Parameters
    ----------
    cache_file : str or Path
        Path to cache file.

    Returns
    -------
    data or None
        Cached data if file exists, None otherwise.
    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


# =============================================================================
# Array Utilities
# =============================================================================

def to_dense_array(X: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
    """
    Convert sparse matrix to dense array.

    Parameters
    ----------
    X : ndarray or sparse matrix
        Input array or sparse matrix.

    Returns
    -------
    ndarray
        Dense numpy array.
    """
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


# =============================================================================
# Normalization
# =============================================================================

def size_factor_normalize(X: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
    """
    Normalize by library size (size factor normalization).

    Scales each cell so that its total counts equal the median across cells.

    Parameters
    ----------
    X : ndarray or sparse matrix
        Count matrix (cells x genes).

    Returns
    -------
    ndarray
        Normalized matrix.
    """
    X = X.astype(float)

    if sp.issparse(X):
        UMI_counts_per_cell = np.array(X.sum(axis=1)).flatten()
    else:
        UMI_counts_per_cell = X.sum(axis=1)

    median_UMI = np.median(UMI_counts_per_cell)
    scaling_factors = median_UMI / UMI_counts_per_cell
    scaling_factors[np.isinf(scaling_factors)] = 0

    if sp.issparse(X):
        scaling_matrix = sp.diags(scaling_factors)
        X = scaling_matrix @ X
        X = X.toarray()
    else:
        X = X * scaling_factors[:, np.newaxis]

    return X


# Alias for backward compatibility
QCscRNAsizeFactorNormOnly = size_factor_normalize


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.
    y_proba : ndarray, optional
        Predicted probabilities for positive class.

    Returns
    -------
    dict
        Dictionary of metric names to values.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, average_precision_score,
        confusion_matrix
    )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            # Ensure arrays are 1D for consistent handling
            y_true_flat = np.asarray(y_true).ravel()
            y_proba_flat = np.asarray(y_proba).ravel()
            metrics['auc'] = roc_auc_score(y_true_flat, y_proba_flat)
            metrics['average_precision'] = average_precision_score(y_true_flat, y_proba_flat)
        except ValueError:
            # Handle case where only one class is present
            metrics['auc'] = float('nan')
            metrics['average_precision'] = float('nan')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])

    return metrics


# =============================================================================
# Model I/O
# =============================================================================

def save_model(model: Any, path: Union[str, Path]) -> None:
    """
    Save trained VI model to pickle file.

    Parameters
    ----------
    model : VI
        Trained model instance.
    path : str or Path
        Path to save file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to {path}")


def load_model(path: Union[str, Path]) -> Any:
    """
    Load trained VI model from pickle file.

    Parameters
    ----------
    path : str or Path
        Path to saved model.

    Returns
    -------
    VI
        Loaded model instance.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded model from {path}")
    return model


# =============================================================================
# Results Saving
# =============================================================================

def save_results(
    model: Any,
    output_dir: Union[str, Path],
    gene_list: List[str],
    splits: Dict[str, List[str]],
    prefix: str = 'vi',
    save_model: bool = True,
    compress: bool = True,
    save_full_model: bool = False,
    feature_type: str = 'gene',
    optimal_threshold: Union[float, Dict[str, float]] = 0.5,
    program_names: Optional[List[str]] = None,
    mode: str = 'unmasked'
) -> Dict[str, Path]:
    """
    Save VI model results to files.

    Parameters
    ----------
    model : VI
        Trained model instance.
    output_dir : str or Path
        Directory to save results.
    gene_list : list of str
        List of gene/pathway names.
    splits : dict
        Dictionary with train/val/test cell ID lists.
    prefix : str, default='vi'
        Prefix for output files.
    save_model : bool, default=True
        Whether to save the model (essential parameters only by default).
    compress : bool, default=True
        Whether to compress CSV files with gzip.
    save_full_model : bool, default=False
        If True, save entire model object. If False (default), save only
        essential parameters to reduce memory during save.
    feature_type : str, default='gene'
        Type of features ('gene' or 'pathway').
    optimal_threshold : float, default=0.5
        Optimal classification threshold tuned on validation set.
    program_names : list of str, optional
        Custom names for programs/factors (e.g., pathway names). If None,
        defaults to GP1, GP2, etc.
    mode : str, default='unmasked'
        Model mode ('unmasked', 'masked', or 'pathway_init').

    Returns
    -------
    dict
        Dictionary mapping result types to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}
    ext = '.csv.gz' if compress else '.csv'
    compression = 'gzip' if compress else None

    # Save model - either full or essential parameters only
    if save_model:
        if save_full_model:
            # Full model pickle (can be very large)
            model_path = output_dir / f'{prefix}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            saved_files['model'] = model_path
            print(f"Saved full model to {model_path}")
        else:
            # Save only essential parameters (memory-efficient)
            essential_params = {
                # Hyperparameters
                'n_factors': model.K,
                'a': model.a,
                'c': model.c,
                'sigma_v': model.sigma_v,
                'b_v': model.b_v,
                'v_prior': model.v_prior,
                'sigma_gamma': model.sigma_gamma,
                # Global parameters (needed for inference)
                'E_beta': model.E_beta,
                'E_log_beta': model.E_log_beta,
                # Classification weights
                'mu_v': model.mu_v,
                'sigma_v_diag': np.array(model.sigma_v_diag),
                'mu_gamma': model.mu_gamma,
                'Sigma_gamma': np.array(model.Sigma_gamma),
                # Dimensions
                'n': model.n,
                'p': model.p,
                'p_aux': model.p_aux,
            }

            # Optional: include training history
            if hasattr(model, 'elbo_history_'):
                essential_params['elbo_history'] = model.elbo_history_
            if hasattr(model, 'poisson_ll_history_'):
                essential_params['poisson_ll_history'] = model.poisson_ll_history_
            if hasattr(model, 'regression_ll_history_'):
                essential_params['regression_ll_history'] = model.regression_ll_history_
            if hasattr(model, 'training_time_'):
                essential_params['training_time'] = model.training_time_

            model_path = output_dir / f'{prefix}_model_params.npz'
            np.savez_compressed(model_path, **essential_params)
            saved_files['model_params'] = model_path
            print(f"Saved essential model parameters to {model_path}")

    # Save gene/pathway programs (beta matrix)
    # Use custom program names if provided (e.g., pathway names)
    if program_names is not None:
        prog_labels = program_names
    else:
        prog_labels = [f"GP{k+1}" for k in range(model.K)]
    
    program_label = f"{feature_type}_program" if feature_type == 'pathway' else "gene_program"
    beta_df = pd.DataFrame(
        model.E_beta.T,  # Transpose: programs x genes/pathways
        index=prog_labels,
        columns=gene_list
    )

    # Add classification weights if available
    if hasattr(model, 'mu_v'):
        n_outcomes = model.mu_v.shape[0]
        for k in range(n_outcomes):
            beta_df.insert(0, f'v_weight_class{k}', model.mu_v[k])

    beta_path = output_dir / f'{prefix}_{feature_type}_programs{ext}'
    beta_df.to_csv(beta_path, compression=compression)
    saved_files[f'{feature_type}_programs'] = beta_path
    print(f"Saved {feature_type} programs to {beta_path}")

    # Save training theta
    if hasattr(model, 'E_theta'):
        theta_train_df = pd.DataFrame(
            model.E_theta,
            index=splits['train'],
            columns=prog_labels
        )
        theta_path = output_dir / f'{prefix}_theta_train{ext}'
        theta_train_df.to_csv(theta_path, compression=compression)
        saved_files['theta_train'] = theta_path
        print(f"Saved training theta to {theta_path}")

    # Save summary
    summary = {
        'hyperparameters': {
            'n_factors': model.K,
            'a': model.a,
            'c': model.c,
            'sigma_v': model.sigma_v,
            'b_v': model.b_v,
            'v_prior': model.v_prior,
            'sigma_gamma': model.sigma_gamma,
        },
        'data_shapes': {
            f'n_{feature_type}s': len(gene_list),
            'n_train': len(splits['train']),
            'n_val': len(splits['val']),
            'n_test': len(splits['test']),
        },
        'feature_type': feature_type,
        'mode': mode,  # unmasked, masked, pathway_init, or combined
        'program_names': program_names,  # Pathway names if using pathway modes
        'n_pathway_factors': getattr(model, 'n_pathway_factors', None),  # For combined mode
        'training': {
            'final_elbo': model.elbo_history_[-1][1] if hasattr(model, 'elbo_history_') and model.elbo_history_ else None,
            'n_iterations': model.elbo_history_[-1][0] if hasattr(model, 'elbo_history_') and model.elbo_history_ else None,
            'final_holl': model.holl_history_[-1][1] if hasattr(model, 'holl_history_') and model.holl_history_ else None,
        },
        'classification': {
            'optimal_threshold': optimal_threshold,
        }
    }

    # === ELBO history ===
    if hasattr(model, 'elbo_history_'):
        summary['elbo_history'] = model.elbo_history_

    # === Held-out LL history ===
    if hasattr(model, 'holl_history_'):
        summary['holl_history'] = model.holl_history_

    # Convert JAX/NumPy types to JSON-serializable Python types
    def convert_to_json_serializable(obj):
        """Recursively convert JAX/NumPy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif hasattr(obj, '__array__'):  # JAX arrays and scalars
            arr = np.array(obj)
            if arr.ndim == 0:  # Scalar
                return arr.item()
            return arr.tolist()
        elif obj is None or isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)  # Fallback for unknown types
    
    summary = convert_to_json_serializable(summary)
    
    summary_path = output_dir / f'{prefix}_summary.json'
    if compress:
        summary_path = output_dir / f'{prefix}_summary.json.gz'
        with gzip.open(summary_path, 'wt') as f:
            json.dump(summary, f, indent=2)
    else:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    saved_files['summary'] = summary_path
    print(f"Saved summary to {summary_path}")
    
    # Save ELBO history to CSV for easy plotting
    if hasattr(model, 'elbo_history_') and model.elbo_history_:
        elbo_df = pd.DataFrame(model.elbo_history_, columns=['iteration', 'elbo'])
        elbo_history_path = output_dir / f'{prefix}_elbo_history{ext}'
        elbo_df.to_csv(elbo_history_path, index=False, compression=compression)
        saved_files['elbo_history'] = elbo_history_path
        print(f"Saved ELBO history to {elbo_history_path}")

    # Save held-out LL history (with breakdown if available)
    if hasattr(model, 'holl_history_') and model.holl_history_:
        # Support both old format (iter, holl) and new format (iter, holl, pois, reg)
        first = model.holl_history_[0]
        if len(first) == 4:
            columns = ['iteration', 'heldout_ll', 'heldout_pois_ll', 'heldout_reg_ll']
        else:
            columns = ['iteration', 'heldout_ll']
        holl_df = pd.DataFrame(model.holl_history_, columns=columns)
        holl_history_path = output_dir / f'{prefix}_holl_history{ext}'
        holl_df.to_csv(holl_history_path, index=False, compression=compression)
        saved_files['holl_history'] = holl_history_path
        print(f"Saved held-out LL history to {holl_history_path}")

    return saved_files


# =============================================================================
# Analysis Utilities
# =============================================================================

def get_top_genes_per_program(
    model: Any,
    gene_list: List[str],
    n_top: int = 10,
    threshold: float = 0.5,
    feature_type: str = 'gene'
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Get top genes/pathways for each gene program.

    Parameters
    ----------
    model : VI
        Trained model with E_beta attribute.
    gene_list : list of str
        List of gene/pathway names corresponding to beta rows.
    n_top : int, default=10
        Number of top genes/pathways to return per program.
    threshold : float, default=0.5
        Spike-and-slab threshold for considering genes/pathways active.
    feature_type : str, default='gene'
        Type of features ('gene' or 'pathway').

    Returns
    -------
    dict
        Dictionary mapping program names to lists of (gene/pathway, loading) tuples.
    """
    results = {}

    for k in range(model.K):
        program_name = f"GP{k+1}"

        # Get gene loadings for this program
        loadings = model.E_beta[:, k].copy()

        # Get top genes by E[beta] loading
        top_indices = np.argsort(loadings)[::-1][:n_top]
        top_genes = [(gene_list[i], float(loadings[i])) for i in top_indices]

        results[program_name] = top_genes

    return results


def get_active_programs(model: Any) -> Dict[str, Any]:
    """
    Get summary of factor activity (all factors active — no spike-and-slab).

    Parameters
    ----------
    model : CAVI
        Trained model.

    Returns
    -------
    dict
        Summary statistics about active components.
    """
    results = {
        'n_factors': model.K,
        'E_beta_range': (float(model.E_beta.min()), float(model.E_beta.max())),
        'mu_v_range': (float(model.mu_v.min()), float(model.mu_v.max())),
    }
    return results


def print_model_summary(model: Any, gene_list: Optional[List[str]] = None) -> None:
    """
    Print a summary of the trained model.

    Parameters
    ----------
    model : VI
        Trained model instance.
    gene_list : list of str, optional
        List of gene names for displaying top genes.
    """
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    print(f"\nHyperparameters (scHPF):")
    print(f"  n_factors (K): {model.K}")
    print(f"  a: {model.a}")
    print(f"  c: {model.c}")
    print(f"  v_prior: {model.v_prior}")
    if model.v_prior == 'normal':
        print(f"  sigma_v: {model.sigma_v}")
    else:
        print(f"  b_v: {model.b_v}")
    print(f"  sigma_gamma: {model.sigma_gamma}")

    if hasattr(model, 'seed_used_'):
        print(f"  random_state: {model.seed_used_} {'(random)' if model.seed_used_ is None else ''}")

    print(f"\nLearned Parameters:")
    print(f"  E[beta] shape: {model.E_beta.shape}")
    print(f"    range: [{model.E_beta.min():.4f}, {model.E_beta.max():.4f}]")
    print(f"    mean: {model.E_beta.mean():.4f}")

    print(f"  mu_v shape: {model.mu_v.shape}")
    print(f"    range: [{model.mu_v.min():.4f}, {model.mu_v.max():.4f}]")

    if hasattr(model, 'E_theta'):
        print(f"  E[theta] shape: {model.E_theta.shape}")
        print(f"    range: [{model.E_theta.min():.4f}, {model.E_theta.max():.4f}]")

    # Training info
    if hasattr(model, 'elbo_history_') and model.elbo_history_:
        print(f"\nTraining:")
        print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}")
        print(f"  Iterations: {model.elbo_history_[-1][0] + 1}")
    if hasattr(model, 'holl_history_') and model.holl_history_:
        print(f"  Best HO-LL: {max(entry[1] for entry in model.holl_history_):.4f}")

    # Top genes per program
    if gene_list is not None:
        top_genes = get_top_genes_per_program(model, gene_list, n_top=5)
        most_influential = np.argmax(np.abs(model.mu_v[0])) if model.mu_v.size > 0 else 0
        print(f"\nTop 5 genes in most influential program (GP{most_influential + 1}):")
        for gene, loading in top_genes[f'GP{most_influential + 1}']:
            print(f"    {gene}: {loading:.4f}")

    print("=" * 60)


# =============================================================================
# Protein Coding Filter (generic version)
# =============================================================================

def filter_protein_coding_genes(
    gene_list: List[str],
    gene_annotation: pd.DataFrame,
    gene_id_col: str = 'GeneID',
    type_col: str = 'Genetype'
) -> List[str]:
    """
    Filter gene list to keep only protein-coding genes.

    Parameters
    ----------
    gene_list : list of str
        List of gene IDs to filter.
    gene_annotation : pd.DataFrame
        Gene annotation DataFrame.
    gene_id_col : str, default='GeneID'
        Column name for gene IDs.
    type_col : str, default='Genetype'
        Column name for gene type.

    Returns
    -------
    list of str
        Filtered list of protein-coding genes.
    """
    protein_coding = gene_annotation[
        gene_annotation[type_col] == 'protein_coding'
    ][gene_id_col].tolist()

    filtered = [g for g in gene_list if g in protein_coding]
    print(f"Filtered to {len(filtered)}/{len(gene_list)} protein-coding genes")
    return filtered

def load_pathways(
    gmt_path: str = '/archive/projects/SSPA_BRAY/sspa/c2.cp.v2024.1.Hs.symbols.gmt',
    convert_to_ensembl: bool = True,
    species: str = 'human',
    gene_filter: Optional[List[str]] = None,
    min_genes: int = 5,
    max_genes: int = 5000,
    cache_dir: str = '/labs/Aguiar/SSPA_BRAY/cache',
    use_cache: bool = True,
    excluded_keywords: Optional[List[str]] = None,
    require_prefix: Optional[str] = 'REACTOME'
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load pathway definitions from GMT file into a binary matrix.
    
    GMT format: PATHWAY_NAME<tab>URL<tab>GENE1<tab>GENE2<tab>...
    
    Caches the Ensembl-converted pathways to avoid repeated conversion overhead.
    
    Parameters
    ----------
    gmt_path : str
        Path to GMT file.
    convert_to_ensembl : bool, default=True
        Convert gene symbols to Ensembl IDs.
    species : str, default='human'
        Species for gene ID conversion.
    gene_filter : list of str, optional
        If provided, only include genes in this list. Useful for filtering
        to genes present in expression data.
    min_genes : int, default=5
        Minimum number of genes for a pathway to be included.
    max_genes : int, default=500
        Maximum number of genes for a pathway to be included.
    cache_dir : str, default='/labs/Aguiar/SSPA_BRAY/cache'
        Directory for caching converted pathways.
    use_cache : bool, default=True
        Whether to use cached converted pathways.
    excluded_keywords : list of str, optional
        Exclude pathways containing any of these keywords (case-insensitive).
        Default: ["ADME", "DRUG", "MISCELLANEOUS", "EMT"]
    require_prefix : str, optional
        If provided, only keep pathways starting with this prefix.
        Default: 'REACTOME' (set to None to keep all sources).
    
    Returns
    -------
    pathway_mat : np.ndarray
        Binary matrix (n_pathways, n_genes) where pathway_mat[i,j]=1 if 
        gene j is in pathway i.
    pathway_names : list of str
        Pathway names corresponding to rows.
    gene_names : list of str
        Gene names (Ensembl or symbol) corresponding to columns.
    """
    import hashlib
    
    # Default excluded keywords
    if excluded_keywords is None:
        excluded_keywords = ["ADME", "DRUG", "MISCELLANEOUS", "EMT"]
    
    # Cache key based on GMT file and conversion settings
    gmt_hash = hashlib.md5(gmt_path.encode()).hexdigest()[:8]
    cache_suffix = f"_ensembl_{species}" if convert_to_ensembl else "_symbols"
    cache_file = Path(cache_dir) / f"pathways_{gmt_hash}{cache_suffix}.pkl"
    
    pathways = None
    all_genes = None
    
    # Try loading from cache
    if use_cache and cache_file.exists():
        print(f"Loading cached pathways from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        pathways = cached['pathways']
        all_genes = cached['all_genes']
        print(f"  Loaded {len(pathways)} pathways with {len(all_genes)} genes from cache")
    
    # Parse GMT and convert if not cached
    if pathways is None:
        # Parse GMT file
        pathways = {}  # pathway_name -> set of genes
        all_genes = set()
        
        print(f"Loading pathways from {gmt_path}...")
        with open(gmt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                pathway_name = parts[0]
                # Skip URL (parts[1]), genes start at parts[2]
                genes = set(parts[2:])
                pathways[pathway_name] = genes
                all_genes.update(genes)
        
        print(f"  Loaded {len(pathways)} pathways with {len(all_genes)} unique genes (symbols)")
        
        # Convert gene symbols to Ensembl if requested
        if convert_to_ensembl:
            converter = GeneIDConverter()
            symbol_list = list(all_genes)
            symbol_to_ensembl, ensembl_list = converter.symbols_to_ensembl(
                symbol_list, species=species
            )
            
            # Build reverse mapping for valid conversions
            valid_genes = set()
            symbol_to_final = {}
            for sym, ens in zip(symbol_list, ensembl_list):
                if ens is not None:
                    symbol_to_final[sym] = ens
                    valid_genes.add(ens)
            
            # Update pathway gene sets
            pathways_converted = {}
            for pathway_name, genes in pathways.items():
                converted = {symbol_to_final[g] for g in genes if g in symbol_to_final}
                if converted:
                    pathways_converted[pathway_name] = converted
            
            pathways = pathways_converted
            all_genes = valid_genes
            print(f"  After Ensembl conversion: {len(all_genes)} genes with valid mappings")
        
        # Save to cache
        if use_cache:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({'pathways': pathways, 'all_genes': all_genes}, f)
            print(f"  Cached converted pathways to {cache_file}")
    
    # Apply pathway filtering by prefix
    if require_prefix is not None:
        n_before = len(pathways)
        pathways = {k: v for k, v in pathways.items() if k.startswith(require_prefix)}
        print(f"  After prefix filter '{require_prefix}': {len(pathways)}/{n_before} pathways")
    
    # Apply pathway filtering by excluded keywords
    if excluded_keywords:
        excluded_upper = [kw.upper() for kw in excluded_keywords]
        n_before = len(pathways)
        pathways = {
            k: v for k, v in pathways.items()
            if not any(kw in k.upper() for kw in excluded_upper)
        }
        print(f"  After keyword exclusion {excluded_keywords}: {len(pathways)}/{n_before} pathways")
    
    # Recompute all_genes after pathway filtering
    all_genes = set()
    for genes in pathways.values():
        all_genes.update(genes)
    
    # Apply gene filter if provided
    if gene_filter is not None:
        gene_filter_set = set(gene_filter)
        all_genes = all_genes & gene_filter_set
        
        # Store original pathway sizes before gene filtering (for adaptive thresholding)
        original_pathway_sizes = {name: len(genes) for name, genes in pathways.items()}
        
        # Update pathways to only include filtered genes
        pathways_filtered = {}
        for pathway_name, genes in pathways.items():
            filtered = genes & gene_filter_set
            if filtered:
                pathways_filtered[pathway_name] = filtered
        pathways = pathways_filtered
        print(f"  After gene filter: {len(all_genes)} genes, {len(pathways)} pathways")
        
        # Adaptive filtering: different thresholds based on original pathway size
        # - Small pathways (<500 genes): keep if at least 2 genes in dataset
        # - Large pathways (>=500 genes): keep if at least half of genes in dataset
        SMALL_PATHWAY_THRESHOLD = 100
        MIN_GENES_SMALL = 5
        
        pathways_adaptive = {}
        n_dropped_small = 0
        n_dropped_large = 0
        for name, genes in pathways.items():
            orig_size = original_pathway_sizes.get(name, len(genes))
            n_overlap = len(genes)
            
            if orig_size < SMALL_PATHWAY_THRESHOLD:
                # Small pathway: require at least 2 genes
                if n_overlap >= MIN_GENES_SMALL:
                    pathways_adaptive[name] = genes
                else:
                    n_dropped_small += 1
            else:
                # Large pathway: require at least half of genes in dataset
                required = orig_size
                if n_overlap >= required:
                    pathways_adaptive[name] = genes
                else:
                    n_dropped_large += 1
        
        print(f"  After adaptive filter (small<{SMALL_PATHWAY_THRESHOLD}: ≥{MIN_GENES_SMALL}; "
              f"large: ≥50%): {len(pathways_adaptive)} pathways "
              f"(dropped {n_dropped_small} small, {n_dropped_large} large)")
        pathways = pathways_adaptive
    
    # Filter pathways by size (after gene filter overlap)
    pathways_sized = {
        name: genes for name, genes in pathways.items()
        if min_genes <= len(genes) <= max_genes
    }
    print(f"  After size filter [{min_genes}, {max_genes}]: {len(pathways_sized)} pathways")
    pathways = pathways_sized
    
    if len(pathways) == 0:
        raise ValueError(
            f"Zero pathways remain after filtering (min_genes={min_genes}, "
            f"max_genes={max_genes}). Likely cause: gene ID mismatch between "
            f"expression data (gene_filter) and pathway GMT file. Check whether "
            f"convert_to_ensembl should be True or False."
        )
    
    # Collect only genes that appear in remaining pathways
    genes_in_pathways = set()
    for genes in pathways.values():
        genes_in_pathways.update(genes)
    all_genes = genes_in_pathways
    
    # Create ordered lists
    gene_names = sorted(all_genes)
    pathway_names = sorted(pathways.keys())
    
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    # Build binary matrix
    n_pathways = len(pathway_names)
    n_genes = len(gene_names)
    pathway_mat = np.zeros((n_pathways, n_genes), dtype=np.int8)
    
    for i, pathway_name in enumerate(pathway_names):
        for gene in pathways[pathway_name]:
            if gene in gene_to_idx:
                pathway_mat[i, gene_to_idx[gene]] = 1
    
    # Summary stats
    genes_per_pathway = pathway_mat.sum(axis=1)
    pathways_per_gene = pathway_mat.sum(axis=0)
    print(f"\nPathway matrix: {n_pathways} pathways x {n_genes} genes")
    print(f"  Genes/pathway: min={genes_per_pathway.min()}, max={genes_per_pathway.max()}, "
          f"mean={genes_per_pathway.mean():.1f}")
    print(f"  Pathways/gene: min={pathways_per_gene.min()}, max={pathways_per_gene.max()}, "
          f"mean={pathways_per_gene.mean():.1f}")
    print(f"  Matrix density: {pathway_mat.mean()*100:.2f}%")
    
    return pathway_mat, pathway_names, gene_names