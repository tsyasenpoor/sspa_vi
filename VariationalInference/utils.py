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
    feature_type: str = 'gene'
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
                'sigma_v': model.sigma_v,
                'sigma_gamma': model.sigma_gamma,
                'pi_v': model.pi_v,
                'pi_beta': model.pi_beta,
                # Global parameters (needed for inference)
                'E_beta': model.E_beta,
                'E_log_beta': model.E_log_beta,
                'rho_beta': model.rho_beta,
                # Classification weights
                'mu_v': model.mu_v,
                'Sigma_v_diag': np.array([np.diag(model.Sigma_v[k]) for k in range(model.kappa)]),  # Only diagonal
                'rho_v': model.rho_v,
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
            'sigma_v': model.sigma_v,
            'sigma_gamma': model.sigma_gamma,
            'pi_v': model.pi_v,
            'pi_beta': model.pi_beta,
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
            'final_elbo': model.elbo_history_[-1][1] if hasattr(model, 'elbo_history_') else None,
            'n_iterations': model.elbo_history_[-1][0] if hasattr(model, 'elbo_history_') else None,
        }
    }

    if hasattr(model, 'elbo_history_'):
        summary['elbo_history'] = model.elbo_history_

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

    for k in range(model.d):
        program_name = f"GP{k+1}"

        # Get gene loadings for this program
        loadings = model.E_beta[:, k]

        # Apply spike-and-slab threshold if available
        if hasattr(model, 'rho_beta'):
            active_mask = model.rho_beta[:, k] > threshold
            loadings = loadings * active_mask

        # Get top genes
        top_indices = np.argsort(loadings)[::-1][:n_top]
        top_genes = [(gene_list[i], float(loadings[i])) for i in top_indices]

        results[program_name] = top_genes

    return results


def get_active_programs(
    model: Any,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Get summary of active genes and factors based on spike-and-slab.

    Parameters
    ----------
    model : VI
        Trained model with rho_beta and rho_v attributes.
    threshold : float, default=0.5
        Threshold for considering an indicator active.

    Returns
    -------
    dict
        Summary statistics about active components.
    """
    results = {}

    if hasattr(model, 'rho_beta'):
        active_beta = model.rho_beta > threshold
        results['beta'] = {
            'n_active': int(active_beta.sum()),
            'n_total': model.rho_beta.size,
            'sparsity': float(1 - active_beta.mean()),
            'active_per_program': [int(active_beta[:, k].sum()) for k in range(model.d)]
        }

    if hasattr(model, 'rho_v'):
        active_v = model.rho_v > threshold
        results['v'] = {
            'n_active': int(active_v.sum()),
            'n_total': model.rho_v.size,
            'sparsity': float(1 - active_v.mean()),
            'active_per_class': [int(active_v[k].sum()) for k in range(model.kappa)]
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
    print(f"  sigma_v: {model.sigma_v}")
    print(f"  pi_v: {model.pi_v}")
    print(f"  pi_beta: {model.pi_beta}")

    if hasattr(model, 'seed_used_'):
        print(f"  random_state: {model.seed_used_} {'(random)' if model.seed_used_ is None else ''}")

    print(f"\nLearned Parameters:")
    print(f"  E[beta] shape: {model.E_beta.shape}")
    print(f"    range: [{model.E_beta.min():.4f}, {model.E_beta.max():.4f}]")
    print(f"    mean: {model.E_beta.mean():.4f}")

    if hasattr(model, 'E_v'):
        print(f"  E[v] shape: {model.E_v.shape}")
        print(f"    range: [{model.E_v.min():.4f}, {model.E_v.max():.4f}]")

    if hasattr(model, 'E_theta'):
        print(f"  E[theta] shape: {model.E_theta.shape}")
        print(f"    range: [{model.E_theta.min():.4f}, {model.E_theta.max():.4f}]")

    # Sparsity info
    active_info = get_active_programs(model)
    if active_info:
        print(f"\nSparsity (threshold=0.5):")
        if 'beta' in active_info:
            print(f"  Beta: {active_info['beta']['sparsity']*100:.1f}% sparse")
            print(f"    Active: {active_info['beta']['n_active']}/{active_info['beta']['n_total']}")
        if 'v' in active_info:
            print(f"  V: {active_info['v']['sparsity']*100:.1f}% sparse")
            print(f"    Active: {active_info['v']['n_active']}/{active_info['v']['n_total']}")

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
        most_influential = np.argmax(np.abs(model.E_v[0])) if hasattr(model, 'E_v') else 0
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
