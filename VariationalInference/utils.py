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
    mode: str = 'unmasked',
    label_columns: Optional[List[str]] = None,
    aux_columns: Optional[List[str]] = None,
    val_test_data: Optional[Dict[str, Any]] = None,
    cell_metadata: Optional[Any] = None,
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
    optimal_threshold : float or dict, default=0.5
        Optimal classification threshold tuned on validation set.
    program_names : list of str, optional
        Custom names for programs/factors (e.g., pathway names). If None,
        defaults to GP1, GP2, etc.
    mode : str, default='unmasked'
        Model mode ('unmasked', 'masked', or 'pathway_init').
    label_columns : list of str, optional
        Names of label columns (e.g., ['CoVID-19 severity', 'Outcome']).
        Used for naming v_weight columns and gamma rows in output files.
    aux_columns : list of str, optional
        Names of auxiliary feature columns. Used for naming gamma weight
        columns in output files.
    val_test_data : dict, optional
        Dictionary with validation/test data for inferring theta:
        {'X_val': ..., 'X_aux_val': ..., 'X_test': ..., 'X_aux_test': ...}
    cell_metadata : DataFrame, optional
        DataFrame indexed by cell ID with metadata columns (e.g. majorType).
        If provided, metadata columns are prepended to theta DataFrames.

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
            if label_columns is not None and k < len(label_columns):
                col_name = f'v_weight_{label_columns[k]}'
            else:
                col_name = f'v_weight_class{k}'
            beta_df.insert(0, col_name, model.mu_v[k])

    beta_path = output_dir / f'{prefix}_{feature_type}_programs{ext}'
    beta_df.to_csv(beta_path, compression=compression)
    saved_files[f'{feature_type}_programs'] = beta_path
    print(f"Saved {feature_type} programs to {beta_path}")

    # --- Helper to add cell metadata columns to a theta DataFrame ---
    def _add_cell_metadata(theta_df):
        if cell_metadata is not None:
            # Match on index (cell IDs)
            common_ids = theta_df.index.intersection(cell_metadata.index)
            if len(common_ids) > 0:
                for col in cell_metadata.columns:
                    theta_df.insert(0, col, cell_metadata.reindex(theta_df.index)[col])
        return theta_df

    # Save training theta
    if hasattr(model, 'E_theta'):
        theta_train_df = pd.DataFrame(
            model.E_theta,
            index=splits['train'],
            columns=prog_labels
        )
        theta_train_df.index.name = 'cell_id'
        _add_cell_metadata(theta_train_df)
        theta_path = output_dir / f'{prefix}_theta_train{ext}'
        theta_train_df.to_csv(theta_path, compression=compression)
        saved_files['theta_train'] = theta_path
        print(f"Saved training theta to {theta_path}")

    # Save validation and test theta (inferred with frozen model parameters)
    if val_test_data is not None:
        for split_name in ('val', 'test'):
            X_key = f'X_{split_name}'
            X_aux_key = f'X_aux_{split_name}'
            if X_key not in val_test_data or split_name not in splits:
                continue
            X_split = val_test_data[X_key]
            X_aux_split = val_test_data.get(X_aux_key)
            result = model.transform(X_split, X_aux_new=X_aux_split)
            theta_split_df = pd.DataFrame(
                result['E_theta'],
                index=splits[split_name],
                columns=prog_labels
            )
            theta_split_df.index.name = 'cell_id'
            _add_cell_metadata(theta_split_df)
            theta_split_path = output_dir / f'{prefix}_theta_{split_name}{ext}'
            theta_split_df.to_csv(theta_split_path, compression=compression)
            saved_files[f'theta_{split_name}'] = theta_split_path
            print(f"Saved {split_name} theta to {theta_split_path}")

    # Save gamma (auxiliary feature weights) if available
    if hasattr(model, 'mu_gamma') and model.mu_gamma.size > 0:
        # mu_gamma shape: (kappa, p_aux)
        if aux_columns is not None and len(aux_columns) == model.mu_gamma.shape[1]:
            gamma_col_names = aux_columns
        else:
            gamma_col_names = [f'aux_{j}' for j in range(model.mu_gamma.shape[1])]

        if label_columns is not None and len(label_columns) == model.mu_gamma.shape[0]:
            gamma_row_names = label_columns
        else:
            gamma_row_names = [f'outcome_{k}' for k in range(model.mu_gamma.shape[0])]

        gamma_df = pd.DataFrame(
            np.array(model.mu_gamma),
            index=gamma_row_names,
            columns=gamma_col_names
        )
        gamma_df.index.name = 'label'
        gamma_path = output_dir / f'{prefix}_gamma_weights{ext}'
        gamma_df.to_csv(gamma_path, compression=compression)
        saved_files['gamma_weights'] = gamma_path
        print(f"Saved gamma weights to {gamma_path}")

        # Also save gamma variance (diagonal of Sigma_gamma) for reference
        if hasattr(model, 'Sigma_gamma') and model.Sigma_gamma.size > 0:
            Sigma_gamma = np.array(model.Sigma_gamma)
            # Extract diagonal variances: (kappa, p_aux)
            gamma_var = np.stack([np.diag(Sigma_gamma[k]) for k in range(Sigma_gamma.shape[0])])
            gamma_var_df = pd.DataFrame(
                gamma_var,
                index=gamma_row_names,
                columns=gamma_col_names
            )
            gamma_var_df.index.name = 'label'
            gamma_var_path = output_dir / f'{prefix}_gamma_variance{ext}'
            gamma_var_df.to_csv(gamma_var_path, compression=compression)
            saved_files['gamma_variance'] = gamma_var_path
            print(f"Saved gamma variance to {gamma_var_path}")

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
        'label_columns': label_columns,
        'aux_columns': aux_columns,
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
    
    # Save training curves plot (ELBO + HO-LL with breakdowns)
    if (hasattr(model, 'elbo_history_') and model.elbo_history_) or \
       (hasattr(model, 'holl_history_') and model.holl_history_):
        plot_training_curves(model, save_dir=output_dir)
        saved_files['training_curves'] = output_dir / 'training_curves.png'

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
    min_genes: int = 0,
    max_genes: int = 50000,
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


# =============================================================================
# Diagnostic Plotting
# =============================================================================

def plot_diagnostics(diagnostics, save_dir, fname="diagnostics.png"):
    """
    Generate diagnostic plots for model quality assessment.

    Plots (3x2 grid):
    1. Train theta L1 norm distribution over iterations
    2. zeta saturation and lambda(zeta) over iterations
    3. True validation logistic loss vs JJ bound
    4. eta vs gene total counts (mask consistency)
    5. E[eta] range over iterations (eta-beta collapse detector)
    6. E[beta] range over iterations

    Parameters
    ----------
    diagnostics : dict
        The model's diagnostics_ dictionary.
    save_dir : str or Path
        Directory to save figure.
    fname : str
        Filename for the saved figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    diag = diagnostics
    if diag is None:
        print("No diagnostics available. Run fit() first.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # --- Plot 1: Train theta L1 norms ---
    ax = axes[0, 0]
    if diag['theta_l1_train']:
        iters, means, stds, mins, maxs = zip(*diag['theta_l1_train'])
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(iters, means, 'b-', label='train mean')
        ax.fill_between(iters, means - stds, means + stds, alpha=0.2, color='b')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||E[theta_i]||_1')
    ax.set_title('Train theta L1 norms')
    ax.legend()

    # --- Plot 2: zeta saturation and lambda(zeta) ---
    ax = axes[0, 1]
    if diag['zeta_stats']:
        iters, zmins, zmeds, zmaxs, frac_caps = zip(*diag['zeta_stats'])
        ax.plot(iters, zmeds, 'g-', label='zeta median')
        ax.fill_between(iters, zmins, zmaxs, alpha=0.15, color='g')
        ax2 = ax.twinx()
        ax2.plot(iters, frac_caps, 'r--', label='frac at cap')
        ax2.set_ylabel('Fraction at zeta_max', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('zeta')
    ax.set_title('zeta saturation')

    # --- Plot 3: True Bernoulli LL vs JJ bound ---
    ax = axes[1, 0]
    if diag['true_val_ll'] and diag['jj_val_ll']:
        iters_t, vals_t = zip(*diag['true_val_ll'])
        iters_j, vals_j = zip(*diag['jj_val_ll'])
        ax.plot(iters_t, vals_t, 'b-o', markersize=3, label='True Bernoulli LL')
        ax.plot(iters_j, vals_j, 'r-s', markersize=3, label='JJ bound LL')
        ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation LL / sample')
    ax.set_title('True logistic loss vs JJ bound')

    # --- Plot 4: eta vs gene total counts (mask consistency) ---
    ax = axes[1, 1]
    if diag['eta_vs_counts'] is not None:
        E_eta, gene_counts, active_mask = diag['eta_vs_counts']
        if active_mask is not None:
            n_active = active_mask.sum(axis=1)
            scatter = ax.scatter(gene_counts, E_eta, c=n_active,
                                 cmap='viridis', s=3, alpha=0.5)
            plt.colorbar(scatter, ax=ax, label='n_active_factors')
        else:
            ax.scatter(gene_counts, E_eta, s=3, alpha=0.5)
    ax.set_xlabel('Gene total counts')
    ax.set_ylabel('E[eta_j]')
    ax.set_title('eta vs gene counts (mask consistency)')
    ax.set_xscale('log')

    # --- Plot 5: E[eta] over iterations (eta-beta collapse detector) ---
    ax = axes[2, 0]
    if diag.get('eta_stats'):
        iters, emins, emeds, emaxs = zip(*diag['eta_stats'])
        ax.semilogy(iters, emeds, 'b-', label='median')
        ax.fill_between(iters, emins, emaxs, alpha=0.15, color='b')
        ax.semilogy(iters, emaxs, 'r--', alpha=0.5, label='max')
    bp_dp = diag.get('bp_dp')
    if bp_dp:
        ax.set_title(f'E[eta] over iterations  (bp={bp_dp[0]:.4f}, dp={bp_dp[1]:.4f})')
    else:
        ax.set_title('E[eta] over iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E[eta_j]')
    ax.legend()

    # --- Plot 6: E[beta] over iterations ---
    ax = axes[2, 1]
    if diag.get('beta_stats'):
        iters, bmins, bmeds, bmaxs = zip(*diag['beta_stats'])
        ax.semilogy(iters, bmeds, 'g-', label='median')
        ax.fill_between(iters, bmins, bmaxs, alpha=0.15, color='g')
        ax.semilogy(iters, bmaxs, 'r--', alpha=0.5, label='max')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E[beta_jk]')
    ax.set_title('E[beta] over iterations')
    ax.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(str(save_dir), fname)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostics saved to {save_path}")
    plt.close(fig)


def plot_training_curves(model, save_dir, fname="training_curves.png"):
    """
    Generate ELBO and held-out LL convergence plots with component breakdowns.

    Produces a 2x2 grid:
      Top row:    full range (all iterations) for ELBO and HO-LL
      Bottom row: zoomed in (skip iter 0 burn-in) to show real dynamics

    Each panel shows component breakdowns (Poisson, Regression, Bernoulli).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_elbo = hasattr(model, 'elbo_history_') and bool(model.elbo_history_)
    has_holl = hasattr(model, 'holl_history_') and bool(model.holl_history_)
    if not has_elbo and not has_holl:
        print("No training history available. Run fit() first.")
        return

    # ── Extract data ──
    elbo_data = {}
    if has_elbo:
        first = model.elbo_history_[0]
        elbo_data['iters'] = [e[0] for e in model.elbo_history_]
        elbo_data['elbo'] = [e[1] for e in model.elbo_history_]
        if len(first) >= 4:
            elbo_data['pois'] = [e[2] for e in model.elbo_history_]
            elbo_data['reg'] = [e[3] for e in model.elbo_history_]

    holl_data = {}
    if has_holl:
        first = model.holl_history_[0]
        holl_data['iters'] = [e[0] for e in model.holl_history_]
        holl_data['total'] = [e[1] for e in model.holl_history_]
        if len(first) >= 3:
            holl_data['pois'] = [e[2] for e in model.holl_history_]
        if len(first) >= 4:
            holl_data['reg'] = [e[3] for e in model.holl_history_]
        if len(first) >= 5:
            holl_data['bern'] = [e[4] for e in model.holl_history_]

    n_cols = int(has_elbo) + int(has_holl)
    # Two rows: full range + zoomed (skip burn-in)
    need_zoom = (has_elbo and len(elbo_data['iters']) > 2) or \
                (has_holl and len(holl_data['iters']) > 2)
    n_rows = 2 if need_zoom else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows),
                             squeeze=False)

    def _plot_elbo(ax, iters, data, title_suffix=''):
        ax.plot(iters, data['elbo'], 'k-', lw=1.5, marker='.', ms=3,
                label='ELBO (total)')
        if 'pois' in data:
            ax.plot(iters, data['pois'], 'steelblue', lw=1, ls='--',
                    marker='.', ms=2, label='Poisson LL')
            ax2 = ax.twinx()
            ax2.plot(iters, data['reg'], 'coral', lw=1, ls='--',
                     marker='.', ms=2, label='Regression LL')
            ax2.set_ylabel('Regression LL', fontsize=9, color='coral')
            ax2.tick_params(axis='y', labelcolor='coral')
            ax2.legend(loc='center right', fontsize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.set_title(f'ELBO{title_suffix}')
        ax.legend(loc='upper left', fontsize=8)

    def _plot_holl(ax, iters, data, title_suffix=''):
        ax.plot(iters, data['total'], 'k-', lw=1.5, marker='.', ms=3,
                label='Total HO-LL')
        if 'pois' in data:
            ax.plot(iters, data['pois'], 'steelblue', lw=1, ls='--',
                    marker='.', ms=2, label='HO Poisson')
        if 'reg' in data or 'bern' in data:
            ax2 = ax.twinx()
            if 'reg' in data:
                ax2.plot(iters, data['reg'], 'coral', lw=1, ls='--',
                         marker='.', ms=2, label='HO Reg (JJ)')
            if 'bern' in data:
                ax2.plot(iters, data['bern'], 'mediumpurple', lw=1, ls=':',
                         marker='.', ms=2, label='HO Bernoulli (true)')
            ax2.set_ylabel('Regression LL / sample', fontsize=9)
            ax2.legend(loc='center right', fontsize=8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Held-out LL / sample')
        ax.set_title(f'Held-out Log-Likelihood{title_suffix}')
        ax.legend(loc='upper left', fontsize=8)

    # ── Row 0: full range ──
    col = 0
    if has_elbo:
        _plot_elbo(axes[0, col], elbo_data['iters'],
                   {k: elbo_data[k] for k in elbo_data if k != 'iters'},
                   ' (full range)')
        col += 1
    if has_holl:
        _plot_holl(axes[0, col], holl_data['iters'],
                   {k: holl_data[k] for k in holl_data if k != 'iters'},
                   ' (full range)')

    # ── Row 1: skip first point (burn-in) to reveal dynamics ──
    if need_zoom:
        col = 0
        if has_elbo and len(elbo_data['iters']) > 2:
            zoomed = {k: v[1:] for k, v in elbo_data.items()}
            _plot_elbo(axes[1, col], zoomed['iters'],
                       {k: zoomed[k] for k in zoomed if k != 'iters'},
                       ' (iter 0 excluded)')
            col += 1
        if has_holl and len(holl_data['iters']) > 2:
            zoomed = {k: v[1:] for k, v in holl_data.items()}
            _plot_holl(axes[1, col], zoomed['iters'],
                       {k: zoomed[k] for k in zoomed if k != 'iters'},
                       ' (iter 0 excluded)')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(str(save_dir), fname)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close(fig)