#!/usr/bin/env python3
"""
Bayesian Optimization for VI/SVI Hyperparameter Tuning
======================================================

Uses Optuna to find optimal hyperparameters for either batch VI (CAVI)
or stochastic VI (SVI) by maximizing validation AUC.

Works with the config-based architecture (VIConfig / SVIConfig) and
supports all data formats handled by DataLoader.

Usage:
    # SVI on h5ad data
    python -m VariationalInference.bayes_opt \
        --data /path/to/data.h5ad \
        --method svi \
        --label-column t2dm \
        --n-trials 100

    # VI on EMTAB directory
    python -m VariationalInference.bayes_opt \
        --data /path/to/EMTAB11349/preprocessed \
        --method vi \
        --label-column IBD \
        --aux-columns sex_female \
        --gene-annotation /path/to/gene_annotation.csv \
        --n-trials 50

    # Only tune specific params
    python -m VariationalInference.bayes_opt \
        --data /path/to/data.h5ad \
        --method svi \
        --params-to-tune n_factors learning_rate regression_weight \
        --n-trials 30
"""
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import argparse
import json
import logging
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
from optuna.pruners import MedianPruner
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from VariationalInference.data_loader import DataLoader
from VariationalInference.config import VIConfig, SVIConfig

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# =========================================================================
# Search space definitions
# =========================================================================

# --- Shared model hyperparameters (used by both VI and SVI) ---
SHARED_SEARCH_SPACE = {
    'n_factors': ('int', 50, 2000, 50),       # (type, low, high, step)
    'alpha_theta': ('float', 0.5, 4.0),
    'alpha_beta': ('float', 0.5, 4.0),
    'alpha_xi': ('float', 0.5, 4.0),
    'alpha_eta': ('float', 0.5, 4.0),
    'lambda_xi': ('float', 0.5, 3.0),
    'lambda_eta': ('float', 0.5, 3.0),
    'sigma_v': ('log_float', 0.05, 2.0),
    'sigma_gamma': ('log_float', 0.1, 2.0),
    'pi_v': ('float', 0.3, 0.99),
    'pi_beta': ('float', 0.01, 0.20),
    'regression_weight': ('log_float', 0.1, 500.0),
}

# --- VI-specific search space ---
VI_SEARCH_SPACE = {
    'theta_damping': ('float', 0.3, 0.95),
    'beta_damping': ('float', 0.3, 0.95),
    'v_damping': ('float', 0.2, 0.9),
    'gamma_damping': ('float', 0.2, 0.9),
    'xi_damping': ('float', 0.5, 0.99),
    'eta_damping': ('float', 0.5, 0.99),
}

# --- SVI-specific search space ---
SVI_SEARCH_SPACE = {
    'batch_size': ('categorical', [64, 128, 256, 512]),
    'learning_rate': ('log_float', 1e-3, 0.5),
    'learning_rate_decay': ('float', 0.5, 1.0),
    'learning_rate_delay': ('float', 1.0, 20.0),
    'local_iterations': ('int', 3, 20, 1),
    'regression_lr_multiplier': ('log_float', 1.0, 50.0),
}

# --- Dataset-specific presets ---
DATASET_PRESETS = {
    'pbmc': {
        'n_factors': ('int', 500, 2000, 150),
        'regression_weight': ('log_float', 10.0, 500.0),
        'alpha_theta': ('float', 1.5, 4.0),
        'alpha_beta': ('float', 1.5, 4.0),
    },
    'simulation': {
        'n_factors': ('int', 20, 80, 5),
        'regression_weight': ('log_float', 0.5, 50.0),
        'alpha_theta': ('float', 1.5, 4.0),
        'alpha_beta': ('float', 1.5, 4.0),
        'local_iterations': ('int', 15, 40, 1),
    },
    'emtab': {
        'n_factors': ('int', 100, 2000, 100),
        'regression_weight': ('log_float', 1.0, 200.0),
    },
}


def _suggest_param(trial: optuna.Trial, name: str, spec: tuple):
    """Suggest a single hyperparameter from an Optuna trial."""
    ptype = spec[0]
    if ptype == 'int':
        low, high = spec[1], spec[2]
        step = spec[3] if len(spec) > 3 else 1
        return trial.suggest_int(name, low, high, step=step)
    elif ptype == 'float':
        return trial.suggest_float(name, spec[1], spec[2])
    elif ptype == 'log_float':
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif ptype == 'categorical':
        return trial.suggest_categorical(name, spec[1])
    else:
        raise ValueError(f"Unknown param type: {ptype}")


def build_search_space(method: str, dataset_preset: Optional[str] = None) -> Dict[str, tuple]:
    """
    Build the full search space for a given method and optional dataset preset.

    Parameters
    ----------
    method : str
        'vi' or 'svi'.
    dataset_preset : str, optional
        One of 'pbmc', 'simulation', 'emtab'. Overrides defaults for
        dataset-specific ranges.

    Returns
    -------
    dict
        Mapping param_name -> (type, *args) for _suggest_param.
    """
    space = dict(SHARED_SEARCH_SPACE)

    if method == 'vi':
        space.update(VI_SEARCH_SPACE)
    else:
        space.update(SVI_SEARCH_SPACE)

    if dataset_preset and dataset_preset in DATASET_PRESETS:
        space.update(DATASET_PRESETS[dataset_preset])

    return space


# =========================================================================
# Trial objective
# =========================================================================

class TrialObjective:
    """
    Optuna objective that trains a VI or SVI model and returns validation AUC.

    Parameters
    ----------
    method : str
        'vi' or 'svi'.
    X_train, y_train, X_aux_train : array-like
        Training data.
    X_val, y_val, X_aux_val : array-like
        Validation data.
    search_space : dict
        Search space from build_search_space.
    params_to_tune : list of str, optional
        If given, only tune these params; use defaults for the rest.
    fixed_params : dict, optional
        Fixed params that override defaults and are not tuned.
    mode : str
        Model mode ('unmasked', 'masked', etc.).
    pathway_mask : np.ndarray, optional
        Pathway mask for constrained modes.
    pathway_names : list of str, optional
        Names of pathways.
    n_pathway_factors : int, optional
        Number of pathway factors for 'combined' mode.
    max_iter : int
        Max iterations for VI (default 200).
    max_epochs : int
        Max epochs for SVI (default 100).
    random_state : int, optional
        Random seed.
    subsample_ratio : float, optional
        If set, subsample training data to this fraction for faster trials.
    """

    def __init__(
        self,
        method: str,
        X_train, y_train, X_aux_train,
        X_val, y_val, X_aux_val,
        search_space: Dict[str, tuple],
        params_to_tune: Optional[List[str]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        mode: str = 'unmasked',
        pathway_mask=None,
        pathway_names=None,
        n_pathway_factors=None,
        max_iter: int = 200,
        max_epochs: int = 100,
        random_state: Optional[int] = None,
        subsample_ratio: Optional[float] = None,
    ):
        self.method = method
        self.X_train = X_train
        self.y_train = y_train
        self.X_aux_train = X_aux_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_aux_val = X_aux_val
        self.search_space = search_space
        self.params_to_tune = params_to_tune
        self.fixed_params = fixed_params or {}
        self.mode = mode
        self.pathway_mask = pathway_mask
        self.pathway_names = pathway_names
        self.n_pathway_factors = n_pathway_factors
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.subsample_ratio = subsample_ratio

    def _subsample(self):
        """Return (optionally subsampled) training data."""
        if self.subsample_ratio is None or self.subsample_ratio >= 1.0:
            return self.X_train, self.y_train, self.X_aux_train

        n = self.X_train.shape[0]
        n_sub = max(int(n * self.subsample_ratio), 10)
        idx = np.random.choice(n, n_sub, replace=False)

        import scipy.sparse as sp
        if sp.issparse(self.X_train):
            X_sub = self.X_train[idx]
        else:
            X_sub = self.X_train[idx]
        y_sub = self.y_train[idx]
        X_aux_sub = self.X_aux_train[idx]
        return X_sub, y_sub, X_aux_sub

    def _sample_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from the search space."""
        params = {}
        for name, spec in self.search_space.items():
            if self.params_to_tune is not None and name not in self.params_to_tune:
                continue
            if name in self.fixed_params:
                continue
            params[name] = _suggest_param(trial, name, spec)
        # Apply fixed params
        params.update(self.fixed_params)
        return params

    def _build_config(self, params: Dict[str, Any]):
        """Build a VIConfig or SVIConfig from sampled params."""
        if self.method == 'vi':
            config_cls = VIConfig
            # Set training params
            config_kwargs = {
                'n_factors': params.get('n_factors', 50),
                'max_iter': self.max_iter,
                'verbose': False,
                'debug': False,
            }
        else:
            config_cls = SVIConfig
            config_kwargs = {
                'n_factors': params.get('n_factors', 50),
                'max_epochs': self.max_epochs,
                'verbose': False,
                'debug': False,
            }

        if self.random_state is not None:
            config_kwargs['random_state'] = self.random_state

        # Map sampled params into config
        for key, val in params.items():
            if hasattr(config_cls, key) or key == 'n_factors':
                config_kwargs[key] = val

        return config_cls(**config_kwargs)

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single trial: train model, return validation AUC."""
        try:
            params = self._sample_hyperparams(trial)
            config = self._build_config(params)

            # Import model class
            if self.method == 'vi':
                from VariationalInference.vi_cavi import CAVI as ModelClass
            else:
                from VariationalInference.svi_corrected import SVI as ModelClass

            # Build model kwargs from config
            model_kwargs = config.model_params()
            model_kwargs['mode'] = self.mode
            if self.pathway_mask is not None:
                model_kwargs['pathway_mask'] = self.pathway_mask
            if self.pathway_names is not None:
                model_kwargs['pathway_names'] = self.pathway_names
            if self.n_pathway_factors is not None:
                model_kwargs['n_pathway_factors'] = self.n_pathway_factors

            model = ModelClass(**model_kwargs)

            # Get training data (possibly subsampled)
            X_tr, y_tr, X_aux_tr = self._subsample()

            # Build fit kwargs from config
            fit_kwargs = config.training_params()
            fit_kwargs.update({
                'X_train': X_tr,
                'y_train': y_tr,
                'X_aux_train': X_aux_tr,
                'X_val': self.X_val,
                'y_val': self.y_val,
                'X_aux_val': self.X_aux_val,
            })

            model.fit(**fit_kwargs)

            # Evaluate on validation set
            y_val_proba = model.predict_proba(
                self.X_val, self.X_aux_val, n_iter=15
            )
            y_val_proba = np.asarray(y_val_proba).ravel()

            # Compute AUC (primary metric)
            y_val_flat = np.asarray(self.y_val).ravel()
            if len(np.unique(y_val_flat)) < 2:
                logger.warning("Validation set has only one class — returning 0.5")
                return 0.5

            auc = roc_auc_score(y_val_flat, y_val_proba)

            # Log additional metrics
            y_val_pred = (y_val_proba > 0.5).astype(int)
            acc = accuracy_score(y_val_flat, y_val_pred)
            f1 = f1_score(y_val_flat, y_val_pred, zero_division=0)
            trial.set_user_attr('val_accuracy', float(acc))
            trial.set_user_attr('val_f1', float(f1))
            trial.set_user_attr('val_auc', float(auc))

            logger.info(
                f"Trial {trial.number}: AUC={auc:.4f}, Acc={acc:.4f}, "
                f"F1={f1:.4f}, n_factors={params.get('n_factors', '?')}"
            )
            return auc

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0


# =========================================================================
# Main optimizer
# =========================================================================

class HyperparameterOptimizer:
    """
    Orchestrates Optuna-based hyperparameter search for VI / SVI.

    Parameters
    ----------
    method : str
        'vi' or 'svi'.
    data_path : str
        Path to data (h5ad file or EMTAB directory).
    label_column : str
        Column name for labels.
    aux_columns : list of str, optional
        Auxiliary feature columns.
    gene_annotation_path : str, optional
        Path to gene annotation CSV.
    dataset_preset : str, optional
        'pbmc', 'simulation', or 'emtab' for dataset-specific ranges.
    mode : str
        Model mode (default 'unmasked').
    gmt_file : str, optional
        GMT file for pathway modes.
    n_pathway_factors : int, optional
        Number of pathway factors for 'combined' mode.
    n_trials : int
        Number of Optuna trials.
    params_to_tune : list of str, optional
        Subset of params to tune (others use defaults).
    fixed_params : dict, optional
        Params fixed at specific values (not tuned).
    max_iter : int
        Max iterations for VI trials (default 200).
    max_epochs : int
        Max epochs for SVI trials (default 100).
    subsample_ratio : float, optional
        Subsample training data for faster trials.
    train_ratio : float
        Train split ratio (default 0.7).
    val_ratio : float
        Val split ratio (default 0.15).
    random_state : int, optional
        Random seed.
    output_dir : str
        Where to save results.
    study_name : str, optional
        Optuna study name.
    """

    def __init__(
        self,
        method: str = 'svi',
        data_path: str = None,
        label_column: str = 't2dm',
        aux_columns: Optional[List[str]] = None,
        gene_annotation_path: Optional[str] = None,
        dataset_preset: Optional[str] = None,
        mode: str = 'unmasked',
        gmt_file: Optional[str] = None,
        n_pathway_factors: Optional[int] = None,
        n_trials: int = 100,
        params_to_tune: Optional[List[str]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        max_iter: int = 200,
        max_epochs: int = 100,
        subsample_ratio: Optional[float] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: Optional[int] = None,
        output_dir: str = './bayes_opt_results',
        study_name: Optional[str] = None,
    ):
        self.method = method.lower()
        assert self.method in ('vi', 'svi'), f"method must be 'vi' or 'svi', got '{method}'"

        self.data_path = data_path
        self.label_column = label_column
        self.aux_columns = aux_columns
        self.gene_annotation_path = gene_annotation_path
        self.dataset_preset = dataset_preset
        self.mode = mode
        self.gmt_file = gmt_file
        self.n_pathway_factors = n_pathway_factors
        self.n_trials = n_trials
        self.params_to_tune = params_to_tune
        self.fixed_params = fixed_params or {}
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.subsample_ratio = subsample_ratio
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.study_name = study_name or f'{self.method}_bayes_opt_{datetime.now():%Y%m%d_%H%M%S}'

        # Will be populated by load_data / load_pathways
        self.data = None
        self.pathway_mask = None
        self.pathway_names = None
        self.search_space = None

    def load_data(self):
        """Load and split data using DataLoader."""
        logger.info(f"Loading data from {self.data_path}...")
        loader = DataLoader(
            data_path=self.data_path,
            gene_annotation_path=self.gene_annotation_path,
            verbose=True,
        )
        self.data = loader.load_and_preprocess(
            label_column=self.label_column,
            aux_columns=self.aux_columns,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            stratify_by=self.label_column,
            random_state=self.random_state,
        )
        X_train, X_aux_train, y_train = self.data['train']
        X_val, X_aux_val, y_val = self.data['val']
        X_test, X_aux_test, y_test = self.data['test']

        logger.info(
            f"Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} val, "
            f"{X_test.shape[0]} test, {self.data['n_genes']} genes, "
            f"{self.data['n_aux']} aux features"
        )
        return self.data

    def load_pathways(self):
        """Load pathway mask from GMT file if needed."""
        if self.mode == 'unmasked' or self.gmt_file is None:
            return

        from VariationalInference.utils import load_pathways

        gene_list = self.data['gene_list']
        logger.info(f"Loading pathways from {self.gmt_file}...")

        pathway_mat, pw_names, pw_genes = load_pathways(
            gmt_path=self.gmt_file,
            gene_filter=gene_list,
        )

        # Map pathway matrix columns to data gene order
        n_pathways = pathway_mat.shape[0]
        n_data_genes = len(gene_list)
        self.pathway_mask = np.zeros((n_pathways, n_data_genes), dtype=np.float32)

        pw_gene_to_idx = {g: i for i, g in enumerate(pw_genes)}
        for j, gene in enumerate(gene_list):
            if gene in pw_gene_to_idx:
                idx = pw_gene_to_idx[gene]
                self.pathway_mask[:, j] = pathway_mat[:, idx]

        self.pathway_names = pw_names
        logger.info(
            f"Loaded {n_pathways} pathways covering "
            f"{(self.pathway_mask.sum(axis=0) > 0).sum()}/{n_data_genes} genes"
        )

    def run(self) -> optuna.Study:
        """
        Run the full optimization pipeline.

        Returns
        -------
        optuna.Study
            The completed Optuna study.
        """
        # Load data
        if self.data is None:
            self.load_data()

        # Load pathways if needed
        self.load_pathways()

        # Build search space
        self.search_space = build_search_space(self.method, self.dataset_preset)

        # Filter search space if params_to_tune is set
        if self.params_to_tune:
            unknown = set(self.params_to_tune) - set(self.search_space.keys())
            if unknown:
                logger.warning(f"Unknown params requested for tuning: {unknown}")

        X_train, X_aux_train, y_train = self.data['train']
        X_val, X_aux_val, y_val = self.data['val']

        # Create objective
        objective = TrialObjective(
            method=self.method,
            X_train=X_train, y_train=y_train, X_aux_train=X_aux_train,
            X_val=X_val, y_val=y_val, X_aux_val=X_aux_val,
            search_space=self.search_space,
            params_to_tune=self.params_to_tune,
            fixed_params=self.fixed_params,
            mode=self.mode,
            pathway_mask=self.pathway_mask,
            pathway_names=self.pathway_names,
            n_pathway_factors=self.n_pathway_factors,
            max_iter=self.max_iter,
            max_epochs=self.max_epochs,
            random_state=self.random_state,
            subsample_ratio=self.subsample_ratio,
        )

        # Create Optuna study
        sampler = TPESampler(
            seed=self.random_state,
            n_startup_trials=10,
            multivariate=True,
        )
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
        )

        # Run optimization
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Starting {self.method.upper()} Bayesian optimization: "
            f"{self.n_trials} trials"
        )
        t0 = time.time()
        study.optimize(objective, n_trials=self.n_trials)
        elapsed = time.time() - t0

        # Report results
        self._report_results(study, elapsed)
        self._save_results(study)

        return study

    def _report_results(self, study: optuna.Study, elapsed: float):
        """Print a summary of the optimization results."""
        best = study.best_trial

        print("\n" + "=" * 70)
        print(f"  {self.method.upper()} BAYESIAN OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"  Trials:        {len(study.trials)}")
        print(f"  Best trial:    #{best.number}")
        print(f"  Best AUC:      {best.value:.4f}")
        if 'val_accuracy' in best.user_attrs:
            print(f"  Best Accuracy: {best.user_attrs['val_accuracy']:.4f}")
        if 'val_f1' in best.user_attrs:
            print(f"  Best F1:       {best.user_attrs['val_f1']:.4f}")
        print(f"  Time elapsed:  {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print()
        print("  Best hyperparameters:")
        for k, v in sorted(best.params.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.6g}")
            else:
                print(f"    {k}: {v}")
        print("=" * 70)

    def _save_results(self, study: optuna.Study):
        """Save optimization results to output directory."""
        best = study.best_trial
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save best params as JSON
        results = {
            'method': self.method,
            'study_name': self.study_name,
            'n_trials': len(study.trials),
            'best_trial': best.number,
            'best_auc': best.value,
            'best_accuracy': best.user_attrs.get('val_accuracy'),
            'best_f1': best.user_attrs.get('val_f1'),
            'best_params': best.params,
            'fixed_params': self.fixed_params,
            'data_path': str(self.data_path),
            'label_column': self.label_column,
            'aux_columns': self.aux_columns,
            'mode': self.mode,
            'dataset_preset': self.dataset_preset,
            'timestamp': timestamp,
        }

        json_path = self.output_dir / f'best_params_{self.method}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Best params saved to {json_path}")

        # Save all trials as CSV
        try:
            df = study.trials_dataframe()
            csv_path = self.output_dir / f'all_trials_{self.method}_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"All trials saved to {csv_path}")
        except Exception as e:
            logger.warning(f"Could not save trials CSV: {e}")

        # Save best config as ready-to-use JSON
        best_config_dict = dict(best.params)
        best_config_dict.update(self.fixed_params)
        if 'n_factors' not in best_config_dict:
            best_config_dict['n_factors'] = 50
        # Add non-tuned training params
        if self.method == 'vi':
            best_config_dict.setdefault('max_iter', self.max_iter)
        else:
            best_config_dict.setdefault('max_epochs', self.max_epochs)

        config_path = self.output_dir / f'best_config_{self.method}_{timestamp}.json'
        with open(config_path, 'w') as f:
            json.dump(best_config_dict, f, indent=2)
        logger.info(f"Best config saved to {config_path}")

        # Print usage hint
        print(f"\nTo use the best config:")
        if self.method == 'vi':
            print(f"  config = VIConfig.from_json('{config_path}')")
            print(f"  model = CAVI(**config.model_params())")
        else:
            print(f"  config = SVIConfig.from_json('{config_path}')")
            print(f"  model = SVI(**config.model_params())")

    def evaluate_best(self, study: optuna.Study) -> Dict[str, float]:
        """
        Re-train with best params and evaluate on the held-out test set.

        Parameters
        ----------
        study : optuna.Study
            Completed study.

        Returns
        -------
        dict
            Test metrics (auc, accuracy, f1).
        """
        best_params = dict(study.best_params)
        best_params.update(self.fixed_params)
        config = self._build_best_config(best_params)

        if self.method == 'vi':
            from VariationalInference.vi_cavi import CAVI as ModelClass
        else:
            from VariationalInference.svi_corrected import SVI as ModelClass

        model_kwargs = config.model_params()
        model_kwargs['mode'] = self.mode
        if self.pathway_mask is not None:
            model_kwargs['pathway_mask'] = self.pathway_mask
        if self.pathway_names is not None:
            model_kwargs['pathway_names'] = self.pathway_names
        if self.n_pathway_factors is not None:
            model_kwargs['n_pathway_factors'] = self.n_pathway_factors

        model = ModelClass(**model_kwargs)

        X_train, X_aux_train, y_train = self.data['train']
        X_val, X_aux_val, y_val = self.data['val']
        X_test, X_aux_test, y_test = self.data['test']

        fit_kwargs = config.training_params()
        fit_kwargs.update({
            'X_train': X_train,
            'y_train': y_train,
            'X_aux_train': X_aux_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_aux_val': X_aux_val,
            'verbose': True,
        })

        logger.info("Re-training with best hyperparameters on full training data...")
        model.fit(**fit_kwargs)

        y_test_proba = model.predict_proba(X_test, X_aux_test, n_iter=20)
        y_test_proba = np.asarray(y_test_proba).ravel()
        y_test_flat = np.asarray(y_test).ravel()

        test_auc = roc_auc_score(y_test_flat, y_test_proba)
        y_test_pred = (y_test_proba > 0.5).astype(int)
        test_acc = accuracy_score(y_test_flat, y_test_pred)
        test_f1 = f1_score(y_test_flat, y_test_pred, zero_division=0)

        metrics = {'auc': test_auc, 'accuracy': test_acc, 'f1': test_f1}

        print("\n" + "=" * 50)
        print("  TEST SET EVALUATION (best hyperparameters)")
        print("=" * 50)
        print(f"  AUC:      {test_auc:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  F1:       {test_f1:.4f}")
        print("=" * 50)

        # Save model
        model_path = self.output_dir / f'best_model_{self.method}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Best model saved to {model_path}")

        return metrics

    def _build_best_config(self, params: Dict[str, Any]):
        """Build config from best parameters."""
        if self.method == 'vi':
            config_cls = VIConfig
            config_kwargs = {
                'n_factors': params.get('n_factors', 50),
                'max_iter': self.max_iter,
                'verbose': True,
            }
        else:
            config_cls = SVIConfig
            config_kwargs = {
                'n_factors': params.get('n_factors', 50),
                'max_epochs': self.max_epochs,
                'verbose': True,
            }

        if self.random_state is not None:
            config_kwargs['random_state'] = self.random_state

        for key, val in params.items():
            if hasattr(config_cls, key) or key == 'n_factors':
                config_kwargs[key] = val

        return config_cls(**config_kwargs)


# =========================================================================
# CLI
# =========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Bayesian optimization for VI/SVI hyperparameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument('--data', required=True, help='Path to data (h5ad or directory)')
    parser.add_argument('--label-column', default='t2dm', help='Label column')
    parser.add_argument('--aux-columns', nargs='+', default=None, help='Auxiliary feature columns')
    parser.add_argument('--gene-annotation', default=None, help='Gene annotation CSV path')

    # Method
    parser.add_argument(
        '--method', default='svi', choices=['vi', 'svi'],
        help='Inference method to optimize'
    )

    # Dataset preset
    parser.add_argument(
        '--dataset-preset', default=None, choices=['pbmc', 'simulation', 'emtab'],
        help='Dataset-specific search space preset'
    )

    # Mode / pathways
    parser.add_argument(
        '--mode', default='unmasked',
        choices=['unmasked', 'masked', 'pathway_init', 'combined'],
        help='Model mode'
    )
    parser.add_argument('--gmt-file', default=None, help='GMT file for pathway modes')
    parser.add_argument('--n-pathway-factors', type=int, default=None,
                        help='Number of pathway factors for combined mode')

    # Optimization
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument(
        '--params-to-tune', nargs='+', default=None,
        help='Only tune these params (use defaults for others)'
    )

    # Fixed params (JSON string)
    parser.add_argument(
        '--fixed-params', type=str, default=None,
        help='JSON string of params to fix (not tuned). '
             'E.g. \'{"n_factors": 500, "batch_size": 224}\''
    )

    # Training limits for each trial
    parser.add_argument('--max-iter', type=int, default=200,
                        help='Max iterations per VI trial')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Max epochs per SVI trial')

    # Subsampling
    parser.add_argument('--subsample-ratio', type=float, default=None,
                        help='Subsample training data for faster trials (e.g. 0.5)')

    # Splits
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation ratio')

    # Output
    parser.add_argument('--output-dir', default='./bayes_opt_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    # Evaluation
    parser.add_argument('--evaluate-best', action='store_true',
                        help='Re-train best params and evaluate on test set')

    # Verbosity
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    optuna.logging.set_verbosity(
        optuna.logging.DEBUG if args.verbose else optuna.logging.WARNING
    )

    # Parse fixed params JSON
    fixed_params = {}
    if args.fixed_params:
        try:
            fixed_params = json.loads(args.fixed_params)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid --fixed-params JSON: {e}")
            return 1

    optimizer = HyperparameterOptimizer(
        method=args.method,
        data_path=args.data,
        label_column=args.label_column,
        aux_columns=args.aux_columns,
        gene_annotation_path=args.gene_annotation,
        dataset_preset=args.dataset_preset,
        mode=args.mode,
        gmt_file=args.gmt_file,
        n_pathway_factors=args.n_pathway_factors,
        n_trials=args.n_trials,
        params_to_tune=args.params_to_tune,
        fixed_params=fixed_params,
        max_iter=args.max_iter,
        max_epochs=args.max_epochs,
        subsample_ratio=args.subsample_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed,
        output_dir=args.output_dir,
    )

    study = optimizer.run()

    if args.evaluate_best:
        optimizer.evaluate_best(study)

    return 0


if __name__ == '__main__':
    sys.exit(main())
