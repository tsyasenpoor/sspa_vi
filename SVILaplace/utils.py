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

import pickle
import gzip
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import os


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
    optimal_threshold: float = 0.5
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
                'n_factors': model.d,
                'alpha_theta': model.alpha_theta,
                'alpha_beta': model.alpha_beta,
                'b_laplace': model.b_laplace,
                'sigma_gamma': model.sigma_gamma,
                # Global parameters (needed for inference)
                'E_beta': model.E_beta,
                'E_log_beta': model.E_log_beta,
                'a_beta': model.a_beta,
                'b_beta': model.b_beta,
                # Classification weights (Laplace prior via scale mixture)
                'mu_v': model.mu_v,
                'sigma_v_sq': model.sigma_v_sq,
                'mu_gamma': model.mu_gamma,
                'Sigma_gamma': model.Sigma_gamma,
                # Dimensions
                'n': model.n,
                'p': model.p,
                'kappa': model.kappa,
                'p_aux': model.p_aux,
            }

            # Optional: include training history
            if hasattr(model, 'elbo_history_'):
                essential_params['elbo_history'] = model.elbo_history_
            if hasattr(model, 'training_time_'):
                essential_params['training_time'] = model.training_time_

            model_path = output_dir / f'{prefix}_model_params.npz'
            np.savez_compressed(model_path, **essential_params)
            saved_files['model_params'] = model_path
            print(f"Saved essential model parameters to {model_path}")

    # Save gene/pathway programs (beta matrix)
    program_label = f"{feature_type}_program" if feature_type == 'pathway' else "gene_program"
    beta_df = pd.DataFrame(
        model.E_beta.T,  # Transpose: programs x genes/pathways
        index=[f"GP{k+1}" for k in range(model.d)],
        columns=gene_list
    )

    # Add classification weights if available
    if hasattr(model, 'E_v'):
        for k in range(model.kappa):
            beta_df.insert(0, f'v_weight_class{k}', model.E_v[k])

    beta_path = output_dir / f'{prefix}_{feature_type}_programs{ext}'
    beta_df.to_csv(beta_path, compression=compression)
    saved_files[f'{feature_type}_programs'] = beta_path
    print(f"Saved {feature_type} programs to {beta_path}")

    # Save training theta
    if hasattr(model, 'E_theta'):
        theta_train_df = pd.DataFrame(
            model.E_theta,
            index=splits['train'],
            columns=[f"GP{k+1}" for k in range(model.d)]
        )
        theta_path = output_dir / f'{prefix}_theta_train{ext}'
        theta_train_df.to_csv(theta_path, compression=compression)
        saved_files['theta_train'] = theta_path
        print(f"Saved training theta to {theta_path}")

    # Save summary
    summary = {
        'hyperparameters': {
            'n_factors': model.d,
            'alpha_theta': model.alpha_theta,
            'alpha_beta': model.alpha_beta,
            'b_laplace': model.b_laplace,
            'sigma_gamma': model.sigma_gamma,
        },
        'data_shapes': {
            f'n_{feature_type}s': len(gene_list),
            'n_train': len(splits['train']),
            'n_val': len(splits['val']),
            'n_test': len(splits['test']),
        },
        'feature_type': feature_type,
        'training': {
            'training_time': getattr(model, 'training_time_', None),
            'final_elbo': model.elbo_history_[-1][1] if hasattr(model, 'elbo_history_') and model.elbo_history_ else None,
            'n_iterations': model.elbo_history_[-1][0] if hasattr(model, 'elbo_history_') and model.elbo_history_ else None,
            # === NEW: EMA and convergence diagnostics for paper ===
            'final_elbo_ema': getattr(model, 'elbo_ema_', None),
            'elbo_welford_mean': getattr(model, 'elbo_welford_mean_', None),
            'elbo_welford_std': (
                np.sqrt(model.elbo_welford_M2_ / (model.elbo_welford_n_ - 1)) 
                if hasattr(model, 'elbo_welford_n_') and model.elbo_welford_n_ > 1 
                else None
            ),
        },
        'classification': {
            'optimal_threshold': optimal_threshold,
        }
    }

    # === Raw ELBO history (mini-batch, scaled) ===
    if hasattr(model, 'elbo_history_'):
        summary['elbo_history'] = model.elbo_history_
    
    # === NEW: Convergence history with EMA trajectory ===
    # Format: [(epoch, ema, welford_mean, welford_std, rel_change), ...]
    if hasattr(model, 'convergence_history_') and model.convergence_history_:
        summary['convergence_history'] = model.convergence_history_

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

    return saved_files


# =============================================================================
# Analysis Utilities
# =============================================================================

def get_top_genes_per_program(
    model: Any,
    gene_list: List[str],
    n_top: int = 10,
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
    feature_type : str, default='gene'
        Type of features ('gene' or 'pathway').

    Returns
    -------
    dict
        Dictionary mapping program names to lists of (gene/pathway, loading) tuples.
    """
    results = {}

    for k in range(model.d):
        program_name = f"GP{k+1}"

        # Get gene loadings for this program
        loadings = model.E_beta[:, k].copy()

        # Get top genes by loading value
        top_indices = np.argsort(loadings)[::-1][:n_top]
        top_genes = [(gene_list[i], float(loadings[i])) for i in top_indices]

        results[program_name] = top_genes

    return results


def get_sparse_coefficients(
    model: Any,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Get summary of sparse v coefficients based on Laplace prior (Bayesian Lasso).

    The Laplace prior induces sparsity by shrinking small coefficients toward zero.
    This function reports which factors have non-negligible disease association.

    Parameters
    ----------
    model : VI
        Trained model with mu_v attribute.
    threshold : float, default=0.1
        Threshold for considering a coefficient non-negligible.

    Returns
    -------
    dict
        Summary statistics about sparse coefficients.
    """
    results = {}

    if hasattr(model, 'mu_v'):
        v_abs = np.abs(np.array(model.mu_v))
        active_v = v_abs > threshold

        results['v'] = {
            'n_active': int(active_v.sum()),
            'n_total': model.mu_v.size,
            'sparsity': float(1 - active_v.mean()),
            'active_per_class': [int(active_v[k].sum()) for k in range(model.kappa)],
            'threshold_used': float(threshold),
            'v_stats': {
                'min': float(model.mu_v.min()),
                'max': float(model.mu_v.max()),
                'mean': float(model.mu_v.mean()),
                'abs_mean': float(v_abs.mean())
            }
        }

        # Report E[τ⁻¹] statistics (adaptive shrinkage)
        if hasattr(model, 'E_tau_inv'):
            results['v']['tau_inv_stats'] = {
                'min': float(np.array(model.E_tau_inv).min()),
                'max': float(np.array(model.E_tau_inv).max()),
                'mean': float(np.array(model.E_tau_inv).mean())
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

    print(f"\nHyperparameters:")
    print(f"  n_factors: {model.d}")
    print(f"  alpha_theta: {model.alpha_theta}")
    print(f"  alpha_beta: {model.alpha_beta}")
    print(f"  b_laplace: {model.b_laplace}")
    print(f"  sigma_gamma: {model.sigma_gamma}")

    print(f"\nLearned Parameters:")
    print(f"  E[beta] shape: {model.E_beta.shape}")
    print(f"    range: [{model.E_beta.min():.4f}, {model.E_beta.max():.4f}]")
    print(f"    mean: {model.E_beta.mean():.4f}")

    if hasattr(model, 'mu_v'):
        print(f"  mu_v shape: {model.mu_v.shape}")
        print(f"    range: [{model.mu_v.min():.4f}, {model.mu_v.max():.4f}]")
        print(f"    mean: {model.mu_v.mean():.4f}")

    if hasattr(model, 'E_tau_inv'):
        print(f"  E[τ⁻¹] (adaptive shrinkage):")
        print(f"    range: [{np.array(model.E_tau_inv).min():.4f}, {np.array(model.E_tau_inv).max():.4f}]")

    # Sparsity info based on Laplace prior
    sparse_info = get_sparse_coefficients(model, threshold=0.1)
    if sparse_info and 'v' in sparse_info:
        v_info = sparse_info['v']
        print(f"\nSparsity Analysis (Laplace prior, threshold=0.1):")
        print(f"  Sparsity: {v_info['sparsity']*100:.1f}%")
        print(f"  Active factors: {v_info['n_active']}/{v_info['n_total']}")
        print(f"  v: min={v_info['v_stats']['min']:.4f}, max={v_info['v_stats']['max']:.4f}, "
              f"mean={v_info['v_stats']['mean']:.4f}")
        if 'tau_inv_stats' in v_info:
            print(f"  E[τ⁻¹]: min={v_info['tau_inv_stats']['min']:.4f}, "
                  f"max={v_info['tau_inv_stats']['max']:.4f}, mean={v_info['tau_inv_stats']['mean']:.4f}")

    # Training info
    if hasattr(model, 'elbo_history_') and model.elbo_history_:
        print(f"\nTraining:")
        print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}")
        print(f"  Iterations: {model.elbo_history_[-1][0] + 1}")
        if hasattr(model, 'training_time_'):
            print(f"  Time: {model.training_time_:.2f}s")

    # Top genes per program
    if gene_list is not None:
        top_genes = get_top_genes_per_program(model, gene_list, n_top=5)
        # Find most influential program based on v magnitude
        most_influential = np.argmax(np.abs(np.array(model.mu_v)[0])) if hasattr(model, 'mu_v') else 0
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
