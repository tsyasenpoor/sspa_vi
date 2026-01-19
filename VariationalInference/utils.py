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
    optimal_threshold: float = 0.5,
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
    # Use custom program names if provided (e.g., pathway names)
    if program_names is not None:
        prog_labels = program_names
    else:
        prog_labels = [f"GP{k+1}" for k in range(model.d)]
    
    program_label = f"{feature_type}_program" if feature_type == 'pathway' else "gene_program"
    beta_df = pd.DataFrame(
        model.E_beta.T,  # Transpose: programs x genes/pathways
        index=prog_labels,
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
            columns=prog_labels
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
        'mode': mode,  # unmasked, masked, pathway_init, or combined
        'program_names': program_names,  # Pathway names if using pathway modes
        'n_pathway_factors': getattr(model, 'n_pathway_factors', None),  # For combined mode
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

    # === NEW: Held-out log-likelihood history (Blei-style convergence) ===
    # Format: [(epoch, heldout_ll), ...]
    if hasattr(model, 'heldout_loglik_history_') and model.heldout_loglik_history_:
        summary['heldout_loglik_history'] = model.heldout_loglik_history_
        # Also store final held-out LL in training section for quick access
        summary['training']['final_heldout_loglik'] = model.heldout_loglik_history_[-1][1]

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
        # Use the full posterior mean E_beta which already incorporates spike-and-slab
        # E_beta = rho_beta * E_beta_slab + (1 - rho_beta) * spike_value
        loadings = model.E_beta[:, k].copy()

        # Apply soft weighting by rho_beta instead of hard threshold
        # This preserves relative ranking while downweighting low-probability genes
        if hasattr(model, 'rho_beta'):
            # Use rho_beta as a soft weight rather than binary mask
            # This ensures genes with high E_beta but low rho are still visible
            # but ranked lower than genes with both high E_beta and high rho
            loadings = loadings * model.rho_beta[:, k]

        # Get top genes
        top_indices = np.argsort(loadings)[::-1][:n_top]
        # Report the original E_beta values (not the weighted ones) for interpretability
        # along with the rho values for transparency
        if hasattr(model, 'rho_beta'):
            top_genes = [(gene_list[i], float(model.E_beta[i, k]), float(model.rho_beta[i, k]))
                        for i in top_indices]
        else:
            top_genes = [(gene_list[i], float(loadings[i])) for i in top_indices]

        results[program_name] = top_genes

    return results


def compute_adaptive_threshold(prior_prob: float, factor: float = 2.0) -> float:
    """
    Compute an adaptive threshold for spike-and-slab based on prior probability.

    For spike-and-slab priors, using threshold=0.5 is inappropriate when the prior
    is far from 0.5. Instead, use a threshold that considers when the posterior
    has "moved away" from the prior.

    Parameters
    ----------
    prior_prob : float
        Prior probability of being in the slab (e.g., pi_beta or pi_v).
    factor : float, default=2.0
        How much larger than the prior the posterior should be to be considered active.
        E.g., factor=2.0 means active if rho > 2 * prior_prob.

    Returns
    -------
    float
        Adaptive threshold value.
    """
    # Threshold at factor * prior, but cap at 0.5 to avoid being too lenient
    return min(prior_prob * factor, 0.5)


def get_active_programs(
    model: Any,
    threshold: float = None,
    use_adaptive: bool = True
) -> Dict[str, Any]:
    """
    Get summary of active genes and factors based on spike-and-slab.

    Parameters
    ----------
    model : VI
        Trained model with rho_beta and rho_v attributes.
    threshold : float, optional
        Threshold for considering an indicator active. If None and use_adaptive=True,
        an adaptive threshold based on the prior will be computed.
    use_adaptive : bool, default=True
        Whether to use adaptive thresholding based on the model's prior probabilities.

    Returns
    -------
    dict
        Summary statistics about active components.
    """
    results = {}

    if hasattr(model, 'rho_beta'):
        # Determine threshold for beta
        if threshold is not None:
            beta_threshold = threshold
        elif use_adaptive and hasattr(model, 'pi_beta'):
            beta_threshold = compute_adaptive_threshold(model.pi_beta)
        else:
            beta_threshold = 0.5

        active_beta = model.rho_beta > beta_threshold
        results['beta'] = {
            'n_active': int(active_beta.sum()),
            'n_total': model.rho_beta.size,
            'sparsity': float(1 - active_beta.mean()),
            'active_per_program': [int(active_beta[:, k].sum()) for k in range(model.d)],
            'threshold_used': float(beta_threshold),
            'prior_prob': float(model.pi_beta) if hasattr(model, 'pi_beta') else None,
            'rho_stats': {
                'min': float(model.rho_beta.min()),
                'max': float(model.rho_beta.max()),
                'mean': float(model.rho_beta.mean())
            }
        }

    if hasattr(model, 'rho_v'):
        # Determine threshold for v
        if threshold is not None:
            v_threshold = threshold
        elif use_adaptive and hasattr(model, 'pi_v'):
            v_threshold = compute_adaptive_threshold(model.pi_v)
        else:
            v_threshold = 0.5

        active_v = model.rho_v > v_threshold
        results['v'] = {
            'n_active': int(active_v.sum()),
            'n_total': model.rho_v.size,
            'sparsity': float(1 - active_v.mean()),
            'active_per_class': [int(active_v[k].sum()) for k in range(model.kappa)],
            'threshold_used': float(v_threshold),
            'prior_prob': float(model.pi_v) if hasattr(model, 'pi_v') else None,
            'rho_stats': {
                'min': float(model.rho_v.min()),
                'max': float(model.rho_v.max()),
                'mean': float(model.rho_v.mean())
            }
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

    # Sparsity info with adaptive threshold
    active_info = get_active_programs(model, use_adaptive=True)
    if active_info:
        print(f"\nSparsity (adaptive threshold):")
        if 'beta' in active_info:
            beta_info = active_info['beta']
            print(f"  Beta (threshold={beta_info['threshold_used']:.4f}, prior={beta_info['prior_prob']}):")
            print(f"    Sparsity: {beta_info['sparsity']*100:.1f}%")
            print(f"    Active: {beta_info['n_active']}/{beta_info['n_total']}")
            print(f"    rho_beta: min={beta_info['rho_stats']['min']:.4f}, "
                  f"max={beta_info['rho_stats']['max']:.4f}, mean={beta_info['rho_stats']['mean']:.4f}")
        if 'v' in active_info:
            v_info = active_info['v']
            print(f"  V (threshold={v_info['threshold_used']:.4f}, prior={v_info['prior_prob']}):")
            print(f"    Sparsity: {v_info['sparsity']*100:.1f}%")
            print(f"    Active: {v_info['n_active']}/{v_info['n_total']}")
            print(f"    rho_v: min={v_info['rho_stats']['min']:.4f}, "
                  f"max={v_info['rho_stats']['max']:.4f}, mean={v_info['rho_stats']['mean']:.4f}")

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
        for item in top_genes[f'GP{most_influential + 1}']:
            if len(item) == 3:
                gene, loading, rho = item
                print(f"    {gene}: {loading:.4f} (rho={rho:.4f})")
            else:
                gene, loading = item
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
        
        # Update pathways to only include filtered genes
        pathways_filtered = {}
        for pathway_name, genes in pathways.items():
            filtered = genes & gene_filter_set
            if filtered:
                pathways_filtered[pathway_name] = filtered
        pathways = pathways_filtered
        print(f"  After gene filter: {len(all_genes)} genes, {len(pathways)} pathways")
    
    # Filter pathways by size
    pathways_sized = {
        name: genes for name, genes in pathways.items()
        if min_genes <= len(genes) <= max_genes
    }
    print(f"  After size filter [{min_genes}, {max_genes}]: {len(pathways_sized)} pathways")
    pathways = pathways_sized
    
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