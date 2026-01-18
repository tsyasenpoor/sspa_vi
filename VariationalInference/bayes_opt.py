#!/usr/bin/env python3
"""
Bayesian Optimization for VI/SVI Hyperparameter Tuning

This standalone script uses Optuna to find optimal hyperparameters for the
Variational Inference model by maximizing validation AUC.

Supports multiple data formats:
- h5ad files (AnnData format)
- EMTAB CSV directories (preprocessed EMTAB11349 format)
- Simulated CSV files

Usage:
    # h5ad data
    python -m VariationalInference.bayes_opt \
        --data /path/to/data.h5ad \
        --label-column t2dm \
        --n-trials 100 \
        --output-dir ./bayes_opt_results

    # EMTAB directory (auto-detected)
    python -m VariationalInference.bayes_opt \
        --data /path/to/EMTAB11349/preprocessed \
        --label-column IBD \
        --aux-columns sex_female \
        --gene-annotation /path/to/gene_annotation.csv \
        --n-trials 100

Author: Generated for VI hyperparameter optimization
"""
import os 
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Add BRay/ to path for VariationalInference imports

# Suppress some warnings during optimization
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import VI components
from VariationalInference.data_loader import DataLoader
from VariationalInference.svi_corrected import SVI
from VariationalInference.utils import compute_metrics, load_pathways

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HYPERPARAMETER SEARCH SPACE DEFINITIONS
# =============================================================================

def get_default_search_space() -> Dict[str, Dict[str, Any]]:
    """
    Define the default search space for all hyperparameters.

    Returns a dict where each key is a hyperparameter name and value is a dict
    containing: 'type', 'low', 'high', 'log' (for float), 'choices' (for categorical).
    
    Key refinements:
    - Alpha parameters: ≥1.0 to ensure proper Gamma priors
    - Learning rate: Reduced upper bound for stability
    - pi_v: Narrowed to 0.5-0.95 for better disease signal retention
    - pi_beta: Increased sparsity (lower values) for gene programs
    - sigma_v: Tightened regularization range for disease effects
    """
    return {
        # Learning rate (for SVI) - refined range
        'learning_rate': {
            'type': 'float',
            'low': 1e-4,
            'high': 0.1,
            'log': True,
            'description': 'Natural gradient step size'
        },

        # Latent factor shape parameters (CORRECTED: ≥1 for proper priors)
        'alpha_theta': {
            'type': 'float',
            'low': 1.0,
            'high': 10.0,
            'log': False,
            'description': 'Sample factor sparsity (≥1 for proper prior)'
        },
        'alpha_beta': {
            'type': 'float',
            'low': 1.0,
            'high': 10.0,
            'log': False,
            'description': 'Gene loading sparsity (≥1 for proper prior)'
        },

        # Hierarchical shrinkage (CORRECTED)
        'alpha_xi': {
            'type': 'float',
            'low': 1.0,
            'high': 5.0,
            'log': False,
            'description': 'Sample depth heterogeneity'
        },
        'alpha_eta': {
            'type': 'float',
            'low': 1.0,
            'high': 5.0,
            'log': False,
            'description': 'Gene scale heterogeneity'
        },
        'lambda_xi': {
            'type': 'float',
            'low': 0.1,
            'high': 10.0,
            'log': True,
            'description': 'Prior mean for sample depth'
        },
        'lambda_eta': {
            'type': 'float',
            'low': 0.1,
            'high': 10.0,
            'log': True,
            'description': 'Prior mean for gene scaling'
        },

        # Spike-and-slab sparsity priors (CORRECTED)
        'pi_v': {
            'type': 'float',
            'low': 0.5,
            'high': 0.95,
            'log': False,
            'description': 'Disease coefficient inclusion prob'
        },
        'pi_beta': {
            'type': 'float',
            'low': 0.01,
            'high': 0.20,
            'log': True,
            'description': 'Gene-program sparsity'
        },

        # Regression regularization (REFINED)
        'sigma_v': {
            'type': 'float',
            'low': 0.1,
            'high': 2.0,
            'log': True,
            'description': 'Disease effect regularization'
        },
        'sigma_gamma': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0,
            'log': True,
            'description': 'Auxiliary covariate regularization'
        },

        # Classification-reconstruction tradeoff (UNCHANGED)
        'regression_weight': {
            'type': 'float',
            'low': 10.1,
            'high': 500.0,
            'log': True,
            'description': 'Disease vs reconstruction tradeoff'
        },

        # Model complexity
        'n_factors': {
            'type': 'int',
            'low': 500,
            'high': 2000,
            'step': 150,
            'description': 'Number of latent factors (d)'
        },

        # Numerical stability
        'count_scale': {
            'type': 'float',
            'low': 500.0,
            'high': 2000.0,
            'log': False,
            'description': 'Divide counts by this value for numerical stability with raw counts'
        },
    }


def get_simulation_search_space() -> Dict[str, Dict[str, Any]]:
    """
    Search space optimized for scdesign3 simulation data.
    
    Target: 1000 cells, 500 genes, balanced T2DM labels.
    Based on successful run (sim_svi_1279359.out): AUC=0.9991, F1=0.9851.
    
    Key observations from run:
    - n_factors=35 sufficient for 500 genes
    - alpha_theta=alpha_beta=2.0 worked well (moderate sparsity)
    - v coefficients evolved from ~0.07 to ~0.42 (disease signal present)
    - Learning rate 0.05 was slow; EMA lagged significantly
    
    Optimizations:
    - Aggressive learning rate schedule for faster convergence
    - Narrower n_factors (small gene set doesn't need large d)
    - Higher regression_weight (strong disease signal in simulation)
    """
    return {
        # =====================================================================
        # LEARNING RATE SCHEDULE (key for fast convergence)
        # =====================================================================
        'learning_rate': {
            'type': 'float',
            'low': 0.3,           # Much higher than default
            'high': 1.0,          # Aggressive upper bound
            'log': False,
            'description': 'Initial learning rate (higher for small data)'
        },
        'learning_rate_decay': {
            'type': 'float',
            'low': 0.5,           # Slower decay (ρ_t ~ t^{-decay})
            'high': 0.7,
            'log': False,
            'description': 'Robbins-Monro decay exponent'
        },
        'learning_rate_delay': {
            'type': 'float',
            'low': 1.0,
            'high': 10.0,
            'log': False,
            'description': 'Delay before decay kicks in (τ in (τ+t)^{-κ})'
        },
        'learning_rate_min': {
            'type': 'float',
            'low': 1e-4,
            'high': 5e-3,
            'log': True,
            'description': 'Minimum learning rate floor (prevents premature freezing)'
        },
        'batch_size': {
            'type': 'int',
            'low': 200,
            'high': 500,
            'step': 50,
            'description': 'Mini-batch size (larger = lower variance)'
        },
        
        # =====================================================================
        # MODEL COMPLEXITY (constrained for 500 genes)
        # =====================================================================
        'n_factors': {
            'type': 'int',
            'low': 20,
            'high': 80,
            'step': 10,
            'description': 'Latent factors (20-80 for 500 genes)'
        },
        'local_iterations': {
            'type': 'int',
            'low': 15,
            'high': 40,
            'step': 5,
            'description': 'Local CAVI iterations per mini-batch'
        },
        
        # =====================================================================
        # GAMMA PRIOR SHAPE (moderate sparsity worked well)
        # =====================================================================
        'alpha_theta': {
            'type': 'float',
            'low': 1.5,
            'high': 4.0,
            'log': False,
            'description': 'Sample factor sparsity'
        },
        'alpha_beta': {
            'type': 'float',
            'low': 1.5,
            'high': 4.0,
            'log': False,
            'description': 'Gene loading sparsity'
        },
        
        # =====================================================================
        # HIERARCHICAL SHRINKAGE
        # =====================================================================
        'alpha_xi': {
            'type': 'float',
            'low': 2.0,
            'high': 5.0,
            'log': False,
            'description': 'Sample depth heterogeneity'
        },
        'alpha_eta': {
            'type': 'float',
            'low': 2.0,
            'high': 5.0,
            'log': False,
            'description': 'Gene scale heterogeneity'
        },
        'lambda_xi': {
            'type': 'float',
            'low': 5.0,
            'high': 15.0,
            'log': False,
            'description': 'Prior mean for sample depth'
        },
        'lambda_eta': {
            'type': 'float',
            'low': 0.5,
            'high': 3.0,
            'log': False,
            'description': 'Prior mean for gene scaling'
        },
        
        # =====================================================================
        # REGRESSION (strong signal in simulation)
        # =====================================================================
        'sigma_v': {
            'type': 'float',
            'low': 0.3,
            'high': 1.0,
            'log': False,
            'description': 'Disease effect regularization'
        },
        'sigma_gamma': {
            'type': 'float',
            'low': 0.2,
            'high': 1.0,
            'log': False,
            'description': 'Auxiliary covariate regularization'
        },
        'regression_weight': {
            'type': 'float',
            'low': 1.0,
            'high': 20.0,
            'log': True,
            'description': 'Disease vs reconstruction tradeoff'
        },
        
        # =====================================================================
        # CONVERGENCE CRITERIA (tuned for noisy mini-batch gradients)
        # =====================================================================
        'ema_decay': {
            'type': 'float',
            'low': 0.85,
            'high': 0.95,
            'log': False,
            'description': 'EMA smoothing (lower=faster response to ELBO changes)'
        },
        'convergence_tol': {
            'type': 'float',
            'low': 5e-4,
            'high': 5e-3,
            'log': True,
            'description': 'Relative change threshold for stopping'
        },
    }


def get_emtab_search_space() -> Dict[str, Dict[str, Any]]:
    """
    Full search space for EMTAB11349 data with comprehensive hyperparameter exploration.
    
    Dataset characteristics:
    - ~590 cells, ~10k genes after filtering
    - ~59% sparse after library size normalization
    - Balanced binary labels (186/226 split)
    
    Based on successful run (ibd_svi_norm_1283703.out):
    - n_factors=150: AUC 0.74, training time 149.8s
    - Working hyperparameters provide starting point
    
    Full exploration strategy:
    - n_factors: 100-2000 to find optimal capacity for 10k genes
    - Spike-and-slab: Full range to discover optimal sparsity patterns
    - All hyperparameters: Broad ranges for comprehensive search
    """
    return {
        # =====================================================================
        # MODEL CAPACITY (10k genes → try up to 2000 factors)
        # =====================================================================
        'n_factors': {
            'type': 'int',
            'low': 100,
            'high': 2000,
            'step': 100,
            'description': 'Latent factors (100-2000 for 10k genes, exploring high capacity)'
        },
        
        # =====================================================================
        # LEARNING RATE SCHEDULE
        # =====================================================================
        'learning_rate': {
            'type': 'float',
            'low': 0.05,
            'high': 0.8,
            'log': True,
            'description': 'Initial learning rate (broad range for different capacities)'
        },
        'learning_rate_decay': {
            'type': 'float',
            'low': 0.5,
            'high': 0.8,
            'log': False,
            'description': 'Decay exponent (slower decay for large models)'
        },
        'learning_rate_min': {
            'type': 'float',
            'low': 0.001,
            'high': 0.05,
            'log': True,
            'description': 'Minimum learning rate floor'
        },
        
        # =====================================================================
        # GAMMA PRIOR SHAPE (full range for sparsity exploration)
        # =====================================================================
        'alpha_theta': {
            'type': 'float',
            'low': 1.0,
            'high': 8.0,
            'log': False,
            'description': 'Sample factor sparsity (1.0=minimal, 8.0=heavy shrinkage)'
        },
        'alpha_beta': {
            'type': 'float',
            'low': 1.0,
            'high': 10.0,
            'log': False,
            'description': 'Gene loading sparsity (explore full range)'
        },
        
        # =====================================================================
        # HIERARCHICAL SHRINKAGE
        # =====================================================================
        'alpha_xi': {
            'type': 'float',
            'low': 1.0,
            'high': 6.0,
            'log': False,
            'description': 'Sample depth heterogeneity'
        },
        'alpha_eta': {
            'type': 'float',
            'low': 1.0,
            'high': 6.0,
            'log': False,
            'description': 'Gene scale heterogeneity'
        },
        'lambda_xi': {
            'type': 'float',
            'low': 1.0,
            'high': 20.0,
            'log': True,
            'description': 'Prior mean for sample depth'
        },
        'lambda_eta': {
            'type': 'float',
            'low': 0.1,
            'high': 5.0,
            'log': True,
            'description': 'Prior mean for gene scaling'
        },
        
        # =====================================================================
        # REGRESSION REGULARIZATION
        # =====================================================================
        'sigma_v': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0,
            'log': True,
            'description': 'Disease effect regularization (broad range)'
        },
        'sigma_gamma': {
            'type': 'float',
            'low': 0.1,
            'high': 2.0,
            'log': True,
            'description': 'Auxiliary covariate regularization'
        },
        
        # =====================================================================
        # CLASSIFICATION-RECONSTRUCTION TRADEOFF
        # =====================================================================
        'regression_weight': {
            'type': 'float',
            'low': 10.0,
            'high': 500.0,
            'log': True,
            'description': 'Disease vs reconstruction balance (log scale for wide range)'
        },
        
        # =====================================================================
        # SPIKE-AND-SLAB SPARSITY (full exploration)
        # =====================================================================
        'pi_v': {
            'type': 'float',
            'low': 0.3,
            'high': 1.0,
            'log': False,
            'description': 'Disease coefficient inclusion prob (0.3=sparse, 1.0=dense)'
        },
        'pi_beta': {
            'type': 'float',
            'low': 0.01,
            'high': 1.0,
            'log': True,
            'description': 'Gene-program sparsity (0.01=very sparse, 1.0=dense)'
        },
        
        # =====================================================================
        # TRAINING DYNAMICS
        # =====================================================================
        'local_iterations': {
            'type': 'int',
            'low': 5,
            'high': 20,
            'step': 5,
            'description': 'Local CAVI iterations (more for complex models)'
        },
        'batch_size': {
            'type': 'int',
            'low': 64,
            'high': 256,
            'step': 32,
            'description': 'Mini-batch size (larger for stability with big models)'
        },
    }


def get_search_space_for_data(data_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get appropriate search space based on data type.

    Args:
        data_type: One of 'default', 'simulation', 'emtab', 'large'

    Returns:
        Search space dictionary
    """
    if data_type == 'simulation':
        return get_simulation_search_space()
    elif data_type == 'emtab':
        return get_emtab_search_space()
    elif data_type == 'large':
        # Large dataset space (conservative learning rate)
        space = get_default_search_space()
        space['learning_rate'] = {
            'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True,
            'description': 'Conservative learning rate for large data'
        }
        return space
    else:
        return get_default_search_space()


def get_search_space_for_mode(
    mode: str,
    data_type: str = 'default',
    n_pathways: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get search space adjusted for the model mode.

    In pathway_init and masked modes, n_factors is fixed to n_pathways,
    so we remove it from the search space.

    In combined mode, we tune n_drpgs (number of data-driven gene programs)
    instead of n_factors, since n_factors = n_pathways + n_drpgs.

    In unmasked mode, n_factors is tunable.

    Args:
        mode: Model mode ('unmasked', 'masked', 'pathway_init', 'combined')
        data_type: Data type for base search space
        n_pathways: Number of pathways (required for masked/pathway_init/combined)

    Returns:
        Search space dictionary adjusted for the mode
    """
    # Start with base search space for data type
    space = get_search_space_for_data(data_type)

    if mode in ['masked', 'pathway_init']:
        # n_factors is fixed to n_pathways, remove from search space
        if 'n_factors' in space:
            del space['n_factors']
        logger.info(f"Mode '{mode}': n_factors fixed to n_pathways ({n_pathways}), not tuning n_factors")

    elif mode == 'combined':
        # Replace n_factors with n_drpgs (data-driven gene programs)
        if 'n_factors' in space:
            del space['n_factors']

        # Add n_drpgs as tunable parameter
        space['n_drpgs'] = {
            'type': 'int',
            'low': 20,
            'high': 200,
            'step': 10,
            'description': 'Number of unconstrained data-driven gene programs (DRGPs)'
        }
        logger.info(f"Mode 'combined': tuning n_drpgs instead of n_factors")
        logger.info(f"  n_factors will be n_pathways ({n_pathways}) + n_drpgs")

    # In unmasked mode, n_factors remains tunable (no changes needed)

    return space


def round_to_precision(value: float, decimals: int = 2) -> float:
    """Round a float value to specified decimal places."""
    return round(value, decimals)


def sample_hyperparameters(
    trial: optuna.Trial,
    search_space: Dict[str, Dict[str, Any]],
    params_to_tune: Optional[List[str]] = None,
    float_precision: int = 2
) -> Dict[str, Any]:
    """
    Sample hyperparameters from the search space for a given Optuna trial.

    Float values are rounded to `float_precision` decimal places to avoid
    overly precise hyperparameter values that don't meaningfully affect results.

    Args:
        trial: Optuna trial object
        search_space: Dictionary defining the search space
        params_to_tune: List of parameter names to tune (None = all)
        float_precision: Number of decimal places for float values (default: 2)

    Returns:
        Dictionary of sampled hyperparameter values
    """
    params = {}

    for name, spec in search_space.items():
        # Skip if not in params_to_tune
        if params_to_tune is not None and name not in params_to_tune:
            continue

        param_type = spec['type']

        if param_type == 'float':
            # Sample the float value
            raw_value = trial.suggest_float(
                name,
                spec['low'],
                spec['high'],
                log=spec.get('log', False)
            )
            # Round to specified precision (default 2 decimal places)
            params[name] = round_to_precision(raw_value, float_precision)

        elif param_type == 'int':
            params[name] = trial.suggest_int(
                name,
                spec['low'],
                spec['high'],
                step=spec.get('step', 1)
            )
        elif param_type == 'categorical':
            params[name] = trial.suggest_categorical(name, spec['choices'])

    return params


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

class VIObjective:
    """
    Optuna objective class for VI hyperparameter optimization.

    This class encapsulates the data loading, model training, and evaluation
    logic needed for each trial.
    """

    def __init__(
        self,
        data_path: str,
        label_column: str,
        aux_columns: Optional[List[str]] = None,
        gene_annotation_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        method: str = 'vi',
        max_iter: int = 100,
        batch_size: int = 128,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: Optional[int] = None,
        search_space: Optional[Dict[str, Dict[str, Any]]] = None,
        params_to_tune: Optional[List[str]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        early_stopping_patience: int = 3,
        multi_objective: bool = False,
        subsample_ratio: Optional[float] = None,
        subsample_size: Optional[int] = None,
        # Pathway mode parameters
        mode: str = 'unmasked',
        pathway_mask: Optional[np.ndarray] = None,
        pathway_names: Optional[List[str]] = None,
        n_pathways: Optional[int] = None,
        n_drpgs: int = 50,
    ):
        """
        Initialize the objective function.

        Args:
            data_path: Path to h5ad file or EMTAB directory (auto-detected)
            label_column: Column name for binary labels (in adata.obs or responses.csv.gz)
            aux_columns: List of auxiliary feature column names
            gene_annotation_path: Path to gene annotation CSV (optional)
            cache_dir: Directory for caching preprocessed data (default: /labs/Aguiar/SSPA_BRAY/cache)
            method: 'vi' or 'svi'
            max_iter: Maximum iterations for training
            batch_size: Batch size for SVI
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            random_state: Random seed for reproducibility
            search_space: Custom search space (None = default)
            params_to_tune: List of params to optimize (None = all)
            fixed_params: Dict of fixed hyperparameter values
            verbose: Whether to print training progress
            early_stopping_patience: Patience for early stopping
            multi_objective: Optimize both AUC and accuracy (Pareto front)
            subsample_ratio: Fraction of training data to use per trial (0.0-1.0).
                            If None and subsample_size is None, uses full training data.
                            Stratified sampling preserves class proportions.
            subsample_size: Exact number of training samples per trial.
                           Takes precedence over subsample_ratio if both specified.
            mode: Model mode ('unmasked', 'masked', 'pathway_init', 'combined')
            pathway_mask: Binary matrix (n_pathways, n_genes) for pathway modes
            pathway_names: List of pathway names for reporting
            n_pathways: Number of pathways (for masked/pathway_init, this is n_factors)
            n_drpgs: Default number of data-driven gene programs for combined mode
        """
        self.data_path = data_path
        self.label_column = label_column
        self.aux_columns = aux_columns or []
        self.gene_annotation_path = gene_annotation_path
        self.cache_dir = cache_dir
        self.method = method
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.search_space = search_space or get_default_search_space()
        self.params_to_tune = params_to_tune
        self.fixed_params = fixed_params or {}
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.multi_objective = multi_objective
        self.subsample_ratio = subsample_ratio
        self.subsample_size = subsample_size

        # Pathway mode settings
        self.mode = mode
        self.pathway_mask = pathway_mask
        self.pathway_names = pathway_names
        self.n_pathways = n_pathways
        self.n_drpgs = n_drpgs

        # Validate pathway mode requirements
        if mode in ['masked', 'pathway_init', 'combined']:
            if pathway_mask is None:
                raise ValueError(f"pathway_mask required for mode='{mode}'")
            if n_pathways is None:
                raise ValueError(f"n_pathways required for mode='{mode}'")

        # Data will be loaded once and cached
        self._data_loaded = False
        self._X_train = None
        self._X_val = None
        self._X_test = None
        self._X_aux_train = None
        self._X_aux_val = None
        self._X_aux_test = None
        self._y_train = None
        self._y_val = None
        self._y_test = None
        self._gene_list = None

        # RNG for stratified subsampling (separate from data split seed)
        self._subsample_rng = np.random.RandomState(random_state)

    def _load_data(self):
        """Load and preprocess data (called once)."""
        if self._data_loaded:
            return

        logger.info(f"Loading data from {self.data_path}")

        loader = DataLoader(
            data_path=self.data_path,
            gene_annotation_path=self.gene_annotation_path,
            cache_dir=self.cache_dir,
            use_cache=True
        )

        data = loader.load_and_preprocess(
            label_column=self.label_column,
            aux_columns=self.aux_columns if self.aux_columns else None,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            random_state=self.random_state
        )

        self._X_train, self._X_aux_train, self._y_train = data['train']
        self._X_val, self._X_aux_val, self._y_val = data['val']
        self._X_test, self._X_aux_test, self._y_test = data['test']
        self._gene_list = data['gene_list']

        # Convert sparse matrices to dense JAX arrays ONCE for all trials
        # This saves significant time as conversion happens only once instead of per trial
        logger.info("Converting data to dense JAX arrays (one-time operation)...")
        if sp.issparse(self._X_train):
            self._X_train = jnp.array(self._X_train.toarray())
        else:
            self._X_train = jnp.array(self._X_train)
        
        if sp.issparse(self._X_val):
            self._X_val = jnp.array(self._X_val.toarray())
        else:
            self._X_val = jnp.array(self._X_val)
        
        if sp.issparse(self._X_test):
            self._X_test = jnp.array(self._X_test.toarray())
        else:
            self._X_test = jnp.array(self._X_test)
        
        # Convert auxiliary data to JAX arrays
        if self._X_aux_train is not None:
            self._X_aux_train = jnp.array(self._X_aux_train)
        if self._X_aux_val is not None:
            self._X_aux_val = jnp.array(self._X_aux_val)
        if self._X_aux_test is not None:
            self._X_aux_test = jnp.array(self._X_aux_test)
        
        # Convert labels to JAX arrays
        self._y_train = jnp.array(self._y_train)
        self._y_val = jnp.array(self._y_val)
        self._y_test = jnp.array(self._y_test)

        self._data_loaded = True

        logger.info(f"Data loaded and converted to JAX: {self._X_train.shape[0]} train, "
                   f"{self._X_val.shape[0]} val, {self._X_test.shape[0]} test samples")
        logger.info(f"Number of genes: {self._X_train.shape[1]}")

        # Log subsampling configuration
        if self.subsample_size is not None:
            target_size = min(self.subsample_size, self._X_train.shape[0])
            logger.info(f"Subsampling enabled: {target_size} samples per trial (stratified)")
        elif self.subsample_ratio is not None:
            target_size = int(self._X_train.shape[0] * self.subsample_ratio)
            logger.info(f"Subsampling enabled: {self.subsample_ratio:.1%} = {target_size} samples per trial (stratified)")
        else:
            logger.info("Subsampling disabled: using full training data per trial")

    def _get_subsampled_train_data(
        self,
        trial_number: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Get stratified subsample of training data for a trial.

        Uses different random seed per trial to add diversity while maintaining
        reproducibility. Stratification ensures class proportions are preserved
        (critical for imbalanced binary labels like T2DM).

        Args:
            trial_number: Optuna trial number (used for per-trial seeding)

        Returns:
            Tuple of (X_train_sub, X_aux_train_sub, y_train_sub)
        """
        n_train = self._X_train.shape[0]

        # Determine target subsample size
        if self.subsample_size is not None:
            target_size = min(self.subsample_size, n_train)
        elif self.subsample_ratio is not None:
            target_size = int(n_train * self.subsample_ratio)
        else:
            # No subsampling - return full training data
            return self._X_train, self._X_aux_train, self._y_train

        # Don't subsample if target is >= actual size
        if target_size >= n_train:
            return self._X_train, self._X_aux_train, self._y_train

        # Create per-trial seed for reproducibility with diversity
        # Combine base seed with trial number
        if self.random_state is not None:
            trial_seed = self.random_state + trial_number
        else:
            trial_seed = trial_number

        # Compute subsample ratio for train_test_split
        subsample_frac = target_size / n_train

        # Stratified subsampling using train_test_split
        # We keep the "train" part which has subsample_frac of the data
        # Convert JAX array to numpy for sklearn compatibility
        y_flat = np.array(self._y_train).ravel()

        try:
            # Use stratified split - keep only the selected subset
            indices = np.arange(n_train)
            selected_indices, _ = train_test_split(
                indices,
                train_size=subsample_frac,
                stratify=y_flat,
                random_state=trial_seed
            )

            # Index JAX arrays and keep as JAX arrays
            X_sub = self._X_train[selected_indices]
            y_sub = self._y_train[selected_indices]

            if self._X_aux_train is not None:
                X_aux_sub = self._X_aux_train[selected_indices]
            else:
                X_aux_sub = None

            if self.verbose:
                # Log class distribution in subsample
                pos_ratio = float(np.array(y_sub).sum()) / len(y_sub)
                orig_pos_ratio = y_flat.sum() / len(y_flat)
                logger.debug(
                    f"Trial {trial_number}: Subsampled {len(selected_indices)}/{n_train} samples "
                    f"(pos ratio: {orig_pos_ratio:.3f} -> {pos_ratio:.3f})"
                )

            return X_sub, X_aux_sub, y_sub

        except ValueError as e:
            # Fallback if stratification fails (e.g., not enough samples per class)
            logger.warning(
                f"Trial {trial_number}: Stratified subsampling failed ({e}), "
                f"using random subsampling"
            )
            rng = np.random.RandomState(trial_seed)
            selected_indices = rng.choice(n_train, size=target_size, replace=False)

            # Index JAX arrays and keep as JAX arrays
            X_sub = self._X_train[selected_indices]
            y_sub = self._y_train[selected_indices]
            X_aux_sub = self._X_aux_train[selected_indices] if self._X_aux_train is not None else None

            return X_sub, X_aux_sub, y_sub

    def __call__(self, trial: optuna.Trial):
        """
        Evaluate a single trial.

        Args:
            trial: Optuna trial object

        Returns:
            If multi_objective: tuple (val_auc, val_accuracy)
            Otherwise: val_auc (float)
        """
        # Ensure data is loaded
        self._load_data()

        # Sample hyperparameters
        sampled_params = sample_hyperparameters(
            trial,
            self.search_space,
            self.params_to_tune
        )

        # Merge with fixed parameters
        params = {**sampled_params, **self.fixed_params}

        # Determine n_factors based on mode
        if self.mode in ['masked', 'pathway_init']:
            # n_factors is fixed to number of pathways
            n_factors = self.n_pathways
            n_pathway_factors = None  # Not used in these modes
        elif self.mode == 'combined':
            # n_factors = n_pathways + n_drpgs
            # n_drpgs may be tuned or use default
            n_drpgs = params.pop('n_drpgs', self.n_drpgs)
            n_factors = self.n_pathways + n_drpgs
            n_pathway_factors = self.n_pathways
        else:
            # unmasked mode: n_factors is tunable
            n_factors = params.pop('n_factors', 50)
            n_pathway_factors = None

        learning_rate = params.pop('learning_rate', 0.01)

        # Get subsampled training data (stratified to preserve class proportions)
        # Validation always uses full holdout set
        X_train_sub, X_aux_train_sub, y_train_sub = self._get_subsampled_train_data(trial.number)

        if self.verbose:
            mode_info = f", mode={self.mode}"
            if self.mode == 'combined':
                mode_info += f", n_pathways={self.n_pathways}, n_drpgs={n_drpgs}"
            elif self.mode in ['masked', 'pathway_init']:
                mode_info += f", n_pathways={self.n_pathways}"
            logger.info(f"Trial {trial.number}: n_factors={n_factors}, "
                       f"learning_rate={learning_rate:.2f}, "
                       f"train_samples={X_train_sub.shape[0]}{mode_info}")

        try:
            # Create SVI model with mode-specific parameters
            model = SVI(
                n_factors=n_factors,
                batch_size=self.batch_size,
                learning_rate=learning_rate,
                alpha_theta=params.get('alpha_theta', 2.0),
                alpha_beta=params.get('alpha_beta', 2.0),
                alpha_xi=params.get('alpha_xi', 2.0),
                alpha_eta=params.get('alpha_eta', 2.0),
                lambda_xi=params.get('lambda_xi', 1.5),
                lambda_eta=params.get('lambda_eta', 1.5),
                sigma_v=params.get('sigma_v', 0.2),
                sigma_gamma=params.get('sigma_gamma', 0.5),
                pi_v=params.get('pi_v', 0.9),
                pi_beta=params.get('pi_beta', 0.05),
                regression_weight=params.get('regression_weight', 1.0),
                use_spike_slab=params.get('use_spike_slab', False),
                # Pathway mode parameters
                mode=self.mode,
                pathway_mask=self.pathway_mask,
                pathway_names=self.pathway_names,
                n_pathway_factors=n_pathway_factors,
            )

            # Train model on subsampled data
            model.fit(
                X=X_train_sub,
                y=y_train_sub,
                X_aux=X_aux_train_sub,
                max_epochs=self.max_iter,
                verbose=self.verbose
            )

            # Evaluate on validation set
            y_val_proba = model.predict_proba(
                self._X_val,
                self._X_aux_val
            )

            # Compute validation metrics (convert JAX arrays to numpy for sklearn)
            y_val_np = np.array(self._y_val).ravel()
            y_val_proba_np = np.array(y_val_proba).ravel()
            val_auc = roc_auc_score(y_val_np, y_val_proba_np)
            y_val_pred = (y_val_proba_np > 0.5).astype(int)
            metrics = compute_metrics(y_val_np, y_val_pred, y_val_proba_np)
            val_accuracy = metrics.get('accuracy', 0)

            # Log trial result
            if self.verbose:
                logger.info(f"Trial {trial.number}: AUC={val_auc:.4f}, Accuracy={val_accuracy:.4f}")

            # Store additional metrics as user attributes
            trial.set_user_attr('val_auc', val_auc)
            trial.set_user_attr('val_accuracy', val_accuracy)
            trial.set_user_attr('val_f1', metrics.get('f1', 0))
            trial.set_user_attr('val_precision', metrics.get('precision', 0))
            trial.set_user_attr('val_recall', metrics.get('recall', 0))

            if self.multi_objective:
                return val_auc, val_accuracy
            else:
                return val_auc

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            # Return a very low score for failed trials
            if self.multi_objective:
                return 0.0, 0.0
            else:
                return 0.0


# =============================================================================
# RESULT ANALYSIS AND EXPORT
# =============================================================================

def analyze_study_results(study: optuna.Study, multi_objective: bool = False) -> Dict[str, Any]:
    """
    Analyze the results of an Optuna study.

    Args:
        study: Completed Optuna study
        multi_objective: Whether this was a multi-objective study

    Returns:
        Dictionary with analysis results
    """
    results = {
        'multi_objective': multi_objective,
        'n_trials': len(study.trials),
        'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
    }

    if multi_objective:
        # Multi-objective: get Pareto front
        best_trials = study.best_trials
        results['n_pareto_solutions'] = len(best_trials)
        results['pareto_front'] = []
        
        for trial in best_trials:
            results['pareto_front'].append({
                'trial_number': trial.number,
                'values': trial.values,  # (AUC, Accuracy)
                'params': trial.params,
                'val_auc': trial.user_attrs.get('val_auc'),
                'val_accuracy': trial.user_attrs.get('val_accuracy'),
                'val_f1': trial.user_attrs.get('val_f1'),
                'val_precision': trial.user_attrs.get('val_precision'),
                'val_recall': trial.user_attrs.get('val_recall'),
            })
        
        # Select "best" based on highest mean of normalized objectives
        if best_trials:
            auc_values = [t.values[0] for t in best_trials]
            acc_values = [t.values[1] for t in best_trials]
            auc_max, auc_min = max(auc_values), min(auc_values)
            acc_max, acc_min = max(acc_values), min(acc_values)
            
            best_idx = 0
            best_score = -np.inf
            for i, trial in enumerate(best_trials):
                # Normalize and average
                norm_auc = (trial.values[0] - auc_min) / (auc_max - auc_min + 1e-10)
                norm_acc = (trial.values[1] - acc_min) / (acc_max - acc_min + 1e-10)
                score = (norm_auc + norm_acc) / 2
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            best_compromise = best_trials[best_idx]
            results['best_compromise_trial_number'] = best_compromise.number
            results['best_compromise_values'] = best_compromise.values
            results['best_compromise_params'] = best_compromise.params
        else:
            results['best_compromise_trial_number'] = None
            results['best_compromise_values'] = None
            results['best_compromise_params'] = {}
    else:
        # Single objective
        results['best_trial_number'] = study.best_trial.number
        results['best_value'] = study.best_value
        results['best_params'] = study.best_params

    # Get best trial's user attributes (additional metrics)
    best_trial = study.best_trial if not multi_objective else (best_trials[best_idx] if best_trials else None)
    if best_trial:
        results['best_metrics'] = {
            'val_auc': best_trial.user_attrs.get('val_auc'),
            'val_accuracy': best_trial.user_attrs.get('val_accuracy'),
            'val_f1': best_trial.user_attrs.get('val_f1'),
            'val_precision': best_trial.user_attrs.get('val_precision'),
            'val_recall': best_trial.user_attrs.get('val_recall'),
        }
    else:
        results['best_metrics'] = {}

    # Parameter importance (if enough trials and single objective)
    if len(study.trials) >= 10 and not multi_objective:
        try:
            importance = optuna.importance.get_param_importances(study)
            results['param_importance'] = importance
        except Exception:
            results['param_importance'] = {}
    else:
        results['param_importance'] = {}

    # All trials summary
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_data = {
                'number': trial.number,
                'params': trial.params,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            if multi_objective:
                trial_data['values'] = trial.values
            else:
                trial_data['value'] = trial.value
            trials_data.append(trial_data)
    results['all_trials'] = trials_data

    return results


def export_results(
    study: optuna.Study,
    output_dir: str,
    analysis: Dict[str, Any]
) -> None:
    """
    Export study results to files.

    Args:
        study: Completed Optuna study
        output_dir: Directory to save results
        analysis: Analysis results dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    multi_objective = analysis.get('multi_objective', False)

    # Save best parameters as JSON
    best_params_file = output_path / f'best_params_{timestamp}.json'
    
    if multi_objective:
        # Save Pareto front
        with open(best_params_file, 'w') as f:
            json.dump({
                'multi_objective': True,
                'n_pareto_solutions': analysis.get('n_pareto_solutions', 0),
                'best_compromise_trial_number': analysis.get('best_compromise_trial_number'),
                'best_compromise_values': analysis.get('best_compromise_values'),
                'best_compromise_params': analysis.get('best_compromise_params', {}),
                'pareto_front': analysis.get('pareto_front', []),
                'best_metrics': analysis.get('best_metrics', {}),
                'timestamp': timestamp
            }, f, indent=2)
    else:
        # Single objective
        with open(best_params_file, 'w') as f:
            json.dump({
                'multi_objective': False,
                'best_params': study.best_params,
                'best_value': study.best_value,
                'best_metrics': analysis.get('best_metrics', {}),
                'timestamp': timestamp
            }, f, indent=2)
    logger.info(f"Best parameters saved to {best_params_file}")

    # Save full analysis as JSON
    analysis_file = output_path / f'analysis_{timestamp}.json'
    with open(analysis_file, 'w') as f:
        # Convert any numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(analysis, f, indent=2, default=convert)
    logger.info(f"Full analysis saved to {analysis_file}")

    # Save study object for potential reloading
    study_file = output_path / f'study_{timestamp}.pkl'
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    logger.info(f"Study object saved to {study_file}")

    # Generate a simple text report
    report_file = output_path / f'report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BAYESIAN OPTIMIZATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Study completed at: {timestamp}\n")
        f.write(f"Total trials: {analysis['n_trials']}\n")
        f.write(f"Completed trials: {analysis['n_completed']}\n")
        f.write(f"Failed trials: {analysis['n_failed']}\n\n")

        if multi_objective:
            f.write("-" * 40 + "\n")
            f.write("MULTI-OBJECTIVE OPTIMIZATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Pareto front size: {analysis.get('n_pareto_solutions', 0)}\n\n")
            
            f.write("Best compromise solution:\n")
            f.write(f"Trial number: {analysis.get('best_compromise_trial_number')}\n")
            if analysis.get('best_compromise_values'):
                f.write(f"Validation AUC: {analysis['best_compromise_values'][0]:.4f}\n")
                f.write(f"Validation Accuracy: {analysis['best_compromise_values'][1]:.4f}\n\n")
            
            f.write("Best compromise hyperparameters:\n")
            for name, value in analysis.get('best_compromise_params', {}).items():
                if isinstance(value, float):
                    f.write(f"  {name}: {value:.6f}\n")
                else:
                    f.write(f"  {name}: {value}\n")
            
            f.write("\nAll Pareto-optimal solutions:\n")
            for i, sol in enumerate(analysis.get('pareto_front', [])[:10]):  # Show top 10
                f.write(f"\n  Solution {i+1} (Trial #{sol['trial_number']}):\n")
                f.write(f"    AUC: {sol['values'][0]:.4f}, Accuracy: {sol['values'][1]:.4f}\n")
        else:
            f.write("-" * 40 + "\n")
            f.write("BEST TRIAL\n")
            f.write("-" * 40 + "\n")
            f.write(f"Trial number: {analysis['best_trial_number']}\n")
            f.write(f"Validation AUC: {analysis['best_value']:.4f}\n\n")

            f.write("Best hyperparameters:\n")
            for name, value in analysis['best_params'].items():
                if isinstance(value, float):
                    f.write(f"  {name}: {value:.6f}\n")
                else:
                    f.write(f"  {name}: {value}\n")

        f.write("\nBest trial metrics:\n")
        for name, value in analysis.get('best_metrics', {}).items():
            if value is not None:
                f.write(f"  {name}: {value:.4f}\n")

        if analysis.get('param_importance'):
            f.write("\n" + "-" * 40 + "\n")
            f.write("PARAMETER IMPORTANCE\n")
            f.write("-" * 40 + "\n")
            for name, importance in sorted(
                analysis['param_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                f.write(f"  {name}: {importance:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    logger.info(f"Report saved to {report_file}")

    # Try to create visualization plots
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        if multi_objective:
            # Pareto front plot
            pareto_front = analysis.get('pareto_front', [])
            if pareto_front:
                aucs = [sol['values'][0] for sol in pareto_front]
                accs = [sol['values'][1] for sol in pareto_front]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(aucs, accs, s=100, alpha=0.6, edgecolors='black')
                ax.set_xlabel('Validation AUC', fontsize=12)
                ax.set_ylabel('Validation Accuracy', fontsize=12)
                ax.set_title('Pareto Front (AUC vs Accuracy)', fontsize=14)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(output_path / f'pareto_front_{timestamp}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
        else:
            # Optimization history plot
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig.savefig(output_path / f'optimization_history_{timestamp}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Parameter importance plot (if available)
            if len(study.trials) >= 10:
                try:
                    fig = optuna.visualization.matplotlib.plot_param_importances(study)
                    fig.savefig(output_path / f'param_importance_{timestamp}.png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception:
                    pass

        logger.info("Visualization plots saved")

    except ImportError:
        logger.warning("matplotlib not available, skipping visualization plots")
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")

# =============================================================================
# CONFIGURATION GENERATOR (Legacy - kept for reference)
# =============================================================================

# Note: VIConfig is not imported; this function is provided for future use
# if VIConfig becomes available. Currently, best parameters are exported as JSON.

# def generate_config_from_params(
#     params: Dict[str, Any],
#     base_config = None
# ):
#     """Generate a VIConfig from optimized parameters."""
#     if base_config is None:
#         base_config = VIConfig()
#     updates = {}
#     param_mapping = {
#         'n_factors': 'n_factors',
#         'alpha_theta': 'alpha_theta',
#         'alpha_beta': 'alpha_beta',
#         'alpha_xi': 'alpha_xi',
#         'alpha_eta': 'alpha_eta',
#         'lambda_xi': 'lambda_xi',
#         'lambda_eta': 'lambda_eta',
#         'sigma_v': 'sigma_v',
#         'sigma_gamma': 'sigma_gamma',
#         'pi_v': 'pi_v',
#         'pi_beta': 'pi_beta',
#         'regression_weight': 'regression_weight',
#         'learning_rate': 'learning_rate',
#     }
#     for param_name, config_attr in param_mapping.items():
#         if param_name in params:
#             updates[config_attr] = params[param_name]
#     return base_config.copy(**updates)



# =============================================================================
# MAIN CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization for VI Hyperparameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with h5ad file
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --n-trials 50

  # EMTAB directory (auto-detected from file structure)
  python -m VariationalInference.bayes_opt \\
      --data /path/to/EMTAB11349/preprocessed \\
      --label-column IBD \\
      --aux-columns sex_female \\
      --gene-annotation /path/to/gene_annotation.csv \\
      --n-trials 100

  # RECOMMENDED: With 40% stratified subsampling for faster trials
  # (trains on ~4k samples, validates on full holdout)
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --subsample-ratio 0.4 \\
      --n-trials 100

  # Alternative: Fixed subsample size (4000 samples per trial)
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --subsample-size 4000 \\
      --n-trials 100

  # With specific parameters to tune
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --params-to-tune n_factors sigma_v pi_v regression_weight \\
      --subsample-ratio 0.4 \\
      --n-trials 100

  # SVI with more iterations
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --method svi \\
      --max-iter 50 \\
      --n-trials 100

  # ====== PATHWAY MODE EXAMPLES ======

  # Pathway-initialized mode: n_factors = n_pathways (not tuned)
  # Beta initialized from pathway structure but free to deviate
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --mode pathway_init \\
      --gmt-file /path/to/pathways.gmt \\
      --n-trials 100

  # Masked mode: n_factors = n_pathways, beta fixed to pathway structure
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --mode masked \\
      --gmt-file /path/to/pathways.gmt \\
      --n-trials 100

  # Combined mode: pathway factors + tunable DRGPs
  # n_factors = n_pathways + n_drpgs, where n_drpgs is tunable
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --mode combined \\
      --gmt-file /path/to/pathways.gmt \\
      --n-drpgs 50 \\
      --n-trials 100

  # Unmasked mode (default): n_factors is tunable
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --mode unmasked \\
      --n-trials 100
        """
    )

    # Data arguments
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to h5ad file or EMTAB directory (auto-detected)'
    )
    parser.add_argument(
        '--label-column', '-l',
        required=True,
        help='Column name for binary labels (e.g., t2dm for h5ad, IBD for EMTAB)'
    )
    parser.add_argument(
        '--aux-columns', '-a',
        nargs='+',
        default=None,
        help='Auxiliary feature column names (e.g., Sex for h5ad, sex_female for EMTAB)'
    )
    parser.add_argument(
        '--gene-annotation',
        default=None,
        help='Path to gene annotation CSV file'
    )
    parser.add_argument(
        '--cache-dir',
        default='/labs/Aguiar/SSPA_BRAY/cache',
        help='Directory for caching preprocessed data (default: /labs/Aguiar/SSPA_BRAY/cache)'
    )

    # Optimization arguments
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds for entire optimization'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    parser.add_argument(
        '--sampler',
        choices=['tpe', 'random', 'cmaes'],
        default='tpe',
        help='Optuna sampler to use (default: tpe)'
    )
    parser.add_argument(
        '--pruner',
        choices=['median', 'halving', 'none'],
        default='none',
        help='Optuna pruner for early stopping (default: none)'
    )
    parser.add_argument(
        '--multi-objective',
        action='store_true',
        help='Optimize both AUC and accuracy (Pareto front)'
    )

    # Parameter selection
    parser.add_argument(
        '--params-to-tune',
        nargs='+',
        default=None,
        help='Specific parameters to tune (default: all)'
    )
    parser.add_argument(
        '--fixed-params',
        type=str,
        default=None,
        help='JSON string of fixed parameter values'
    )
    parser.add_argument(
        '--data-type',
        choices=['default', 'simulation', 'emtab', 'large'],
        default='default',
        help='Data type for search space selection: '
             'simulation (scdesign3 ~1k cells, 500 genes), '
             'emtab (~10k cells, 10k genes), '
             'large (>50k cells), '
             'default (generic). Default: auto-detect or default.'
    )

    # Model mode arguments
    parser.add_argument(
        '--mode',
        choices=['unmasked', 'masked', 'pathway_init', 'combined'],
        default='unmasked',
        help='Model mode: unmasked (standard, n_factors tunable), '
             'masked (beta fixed to pathway structure, n_factors=n_pathways), '
             'pathway_init (beta initialized from pathways, n_factors=n_pathways), '
             'combined (pathway-constrained + unconstrained DRGPs)'
    )
    parser.add_argument(
        '--gmt-file',
        default='/archive/projects/SSPA_BRAY/sspa/c2.cp.v2024.1.Hs.symbols.gmt',
        help='GMT file for pathway definitions (used in masked/pathway_init/combined modes)'
    )
    parser.add_argument(
        '--n-drpgs',
        type=int,
        default=50,
        help='Number of unconstrained data-driven gene programs for combined mode (default: 50)'
    )
    parser.add_argument(
        '--min-pathway-genes',
        type=int,
        default=5,
        help='Minimum number of genes for a pathway to be included (default: 5)'
    )
    parser.add_argument(
        '--max-pathway-genes',
        type=int,
        default=5000,
        help='Maximum number of genes for a pathway to be included (default: 5000)'
    )
    parser.add_argument(
        '--pathway-prefix',
        default='REACTOME',
        help='Only include pathways starting with this prefix (default: REACTOME, use "None" for all)'
    )

    # Training arguments
    parser.add_argument(
        '--method',
        choices=['vi', 'svi'],
        default='vi',
        help='Training method (default: vi)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=100,
        help='Maximum training iterations per trial (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for SVI (default: 128)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training data ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation data ratio (default: 0.15)'
    )

    # Subsampling arguments for faster trials
    parser.add_argument(
        '--subsample-ratio',
        type=float,
        default=None,
        help='Fraction of training data to use per trial (0.0-1.0). '
             'Stratified sampling preserves class proportions. '
             'Validation always uses full holdout. Recommended: 0.4 for large datasets.'
    )
    parser.add_argument(
        '--subsample-size',
        type=int,
        default=None,
        help='Exact number of training samples per trial. '
             'Takes precedence over --subsample-ratio if both specified. '
             'Recommended: 4000 for ~10k datasets.'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir', '-o',
        default='./bayes_opt_results',
        help='Output directory for results (default: ./bayes_opt_results)'
    )
    parser.add_argument(
        '--study-name',
        default=None,
        help='Name for the Optuna study'
    )
    parser.add_argument(
        '--storage',
        default=None,
        help='Optuna storage URL (e.g., sqlite:///study.db)'
    )

    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress most output'
    )

    return parser.parse_args()


def main():
    """Main entry point for the Bayesian optimization script."""
    args = parse_args()

    # Set up logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse fixed parameters if provided
    fixed_params = {}
    if args.fixed_params:
        try:
            fixed_params = json.loads(args.fixed_params)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for fixed-params: {e}")
            sys.exit(1)

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)

    # Handle pathway prefix argument
    pathway_prefix = args.pathway_prefix
    if pathway_prefix and pathway_prefix.lower() == 'none':
        pathway_prefix = None

    # Load pathways if needed for pathway modes
    pathway_mask = None
    pathway_names = None
    n_pathways = None
    pathway_genes = None

    if args.mode in ['masked', 'pathway_init', 'combined']:
        logger.info("=" * 60)
        logger.info(f"Loading pathways for {args.mode.upper()} mode")
        logger.info("=" * 60)

        # First, we need to know which genes are in the data
        # Load data once to get gene list, then filter pathways
        logger.info("Pre-loading data to get gene list for pathway filtering...")

        loader = DataLoader(
            data_path=args.data,
            gene_annotation_path=args.gene_annotation,
            cache_dir=args.cache_dir,
            use_cache=True
        )

        # Load data to get gene list (this will be cached for the objective function)
        temp_data = loader.load_and_preprocess(
            label_column=args.label_column,
            aux_columns=args.aux_columns if args.aux_columns else None,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_state=args.seed
        )
        data_gene_list = temp_data['gene_list']
        logger.info(f"Data contains {len(data_gene_list)} genes")

        # Load pathways, filtering to genes in the data
        logger.info(f"Loading pathways from {args.gmt_file}...")
        pathway_mat, pathway_names_raw, pathway_genes = load_pathways(
            gmt_path=args.gmt_file,
            gene_filter=data_gene_list,
            min_genes=args.min_pathway_genes,
            max_genes=args.max_pathway_genes,
            cache_dir=args.cache_dir,
            require_prefix=pathway_prefix
        )

        # Reorder pathway_mat columns to match data gene order
        # pathway_genes is the column order from load_pathways
        gene_to_idx = {g: i for i, g in enumerate(pathway_genes)}
        data_gene_indices = []
        for gene in data_gene_list:
            if gene in gene_to_idx:
                data_gene_indices.append(gene_to_idx[gene])
            else:
                # Gene not in pathways - will be masked out
                data_gene_indices.append(-1)

        # Build reordered pathway mask (n_pathways x n_data_genes)
        n_pathways = pathway_mat.shape[0]
        n_data_genes = len(data_gene_list)
        pathway_mask = np.zeros((n_pathways, n_data_genes), dtype=np.float32)

        for j, idx in enumerate(data_gene_indices):
            if idx >= 0:
                pathway_mask[:, j] = pathway_mat[:, idx]

        pathway_names = pathway_names_raw
        logger.info(f"Loaded {n_pathways} pathways covering {(pathway_mask.sum(axis=0) > 0).sum()}/{n_data_genes} genes")

        if args.mode == 'combined':
            logger.info(f"Combined mode: will add {args.n_drpgs} tunable DRGPs")
            logger.info(f"  Pathway factors: {n_pathways} (fixed)")
            logger.info(f"  DRGP factors: {args.n_drpgs} (tunable)")

    # Select search space based on data type AND mode
    search_space = get_search_space_for_mode(
        mode=args.mode,
        data_type=args.data_type,
        n_pathways=n_pathways
    )

    # Create study name
    study_name = args.study_name or f"vi_hyperopt_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=" * 60)
    logger.info("BAYESIAN OPTIMIZATION FOR VI HYPERPARAMETERS")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Model mode: {args.mode}")
    if args.mode in ['masked', 'pathway_init', 'combined']:
        logger.info(f"  Pathways: {n_pathways} loaded from {args.gmt_file}")
        if args.mode in ['masked', 'pathway_init']:
            logger.info(f"  n_factors: {n_pathways} (fixed = n_pathways)")
        elif args.mode == 'combined':
            logger.info(f"  n_factors: {n_pathways} + n_drpgs (n_drpgs tunable)")
    else:
        logger.info(f"  n_factors: tunable")
    logger.info(f"Search space: {len(search_space)} parameters")
    logger.info(f"Label column: {args.label_column}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Multi-objective: {args.multi_objective}")
    if args.multi_objective:
        logger.info("Optimizing: AUC + Accuracy (Pareto front)")
    else:
        logger.info("Optimizing: AUC")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Max iterations per trial: {args.max_iter}")
    if args.params_to_tune:
        logger.info(f"Parameters to tune: {args.params_to_tune}")
    else:
        logger.info("Parameters to tune: ALL")

    # Log subsampling configuration
    if args.subsample_size is not None:
        logger.info(f"Training subsampling: {args.subsample_size} samples per trial (stratified)")
    elif args.subsample_ratio is not None:
        logger.info(f"Training subsampling: {args.subsample_ratio:.0%} per trial (stratified)")
    else:
        logger.info("Training subsampling: DISABLED (using full training data)")
    logger.info("Validation: full holdout set (no subsampling)")
    logger.info("=" * 60)

    # Create sampler
    if args.sampler == 'tpe':
        sampler = TPESampler(seed=args.seed)
    elif args.sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    elif args.sampler == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=args.seed)

    # Create pruner
    if args.pruner == 'median':
        pruner = MedianPruner()
    elif args.pruner == 'halving':
        pruner = SuccessiveHalvingPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    # Create objective function with pathway mode parameters
    objective = VIObjective(
        data_path=args.data,
        label_column=args.label_column,
        aux_columns=args.aux_columns,
        gene_annotation_path=args.gene_annotation,
        cache_dir=args.cache_dir,
        method=args.method,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed,
        search_space=search_space,  # Use mode-aware search space
        params_to_tune=args.params_to_tune,
        fixed_params=fixed_params,
        verbose=args.verbose,
        multi_objective=args.multi_objective,
        subsample_ratio=args.subsample_ratio,
        subsample_size=args.subsample_size,
        # Pathway mode parameters
        mode=args.mode,
        pathway_mask=pathway_mask,
        pathway_names=pathway_names,
        n_pathways=n_pathways,
        n_drpgs=args.n_drpgs,
    )

    # Create or load study
    if args.multi_objective:
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            directions=['maximize', 'maximize'],  # Maximize AUC and Accuracy
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            direction='maximize',  # Maximize validation AUC
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

    # Run optimization
    logger.info("\nStarting optimization...")
    start_time = time.time()

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            show_progress_bar=not args.quiet
        )
    except KeyboardInterrupt:
        logger.info("\nOptimization interrupted by user")

    elapsed_time = time.time() - start_time
    logger.info(f"\nOptimization completed in {elapsed_time/60:.1f} minutes")

    # Analyze results
    logger.info("\nAnalyzing results...")
    analysis = analyze_study_results(study, multi_objective=args.multi_objective)

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 60)
    
    if args.multi_objective:
        print(f"\nPareto front size: {analysis.get('n_pareto_solutions', 0)}")
        print(f"\nBest compromise solution (Trial #{analysis.get('best_compromise_trial_number')}):")
        if analysis.get('best_compromise_values'):
            print(f"  Validation AUC: {analysis['best_compromise_values'][0]:.4f}")
            print(f"  Validation Accuracy: {analysis['best_compromise_values'][1]:.4f}")
        print("\nBest compromise hyperparameters:")
        for name, value in analysis.get('best_compromise_params', {}).items():
            if isinstance(value, float):
                print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")
        
        print("\nTop 5 Pareto-optimal solutions:")
        for i, sol in enumerate(analysis.get('pareto_front', [])[:5]):
            print(f"\n  {i+1}. Trial #{sol['trial_number']}")
            print(f"     AUC: {sol['values'][0]:.4f}, Accuracy: {sol['values'][1]:.4f}")
    else:
        print(f"\nBest Validation AUC: {study.best_value:.4f}")
        print(f"Best Trial: #{study.best_trial.number}")
        print("\nBest Hyperparameters:")
        for name, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.6f}")
            else:
                print(f"  {name}: {value}")

    if analysis.get('param_importance'):
        print("\nParameter Importance:")
        for name, importance in sorted(
            analysis['param_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {name}: {importance:.4f}")

    # Export results
    export_results(study, args.output_dir, analysis)

    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 60)

    # Generate config code snippet
    if args.multi_objective:
        best_params = analysis.get('best_compromise_params', {})
        print("\nTo use the best compromise parameters in your training:")
    else:
        best_params = study.best_params
        print("\nTo use these parameters in your training:")
    
    print("-" * 40)
    print("from VariationalInference.config import VIConfig")
    print("from VariationalInference.vi import VI")
    print()
    print("config = VIConfig(")
    for name, value in best_params.items():
        if isinstance(value, float):
            print(f"    {name}={value:.6f},")
        else:
            print(f"    {name}={value},")
    print(")")
    print()
    print("model = VI(**config.model_params())")
    print("model.fit(X, y, X_aux, **config.training_params())")
    print("-" * 40)

    return study


if __name__ == '__main__':
    main()
