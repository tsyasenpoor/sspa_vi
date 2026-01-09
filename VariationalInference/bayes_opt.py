#!/usr/bin/env python3
"""
Bayesian Optimization for VI/SVI Hyperparameter Tuning

This standalone script uses Optuna to find optimal hyperparameters for the
Variational Inference model by maximizing validation AUC.

Usage:
    python -m VariationalInference.bayes_opt \
        --data /path/to/data.h5ad \
        --label-column t2dm \
        --n-trials 100 \
        --output-dir ./bayes_opt_results

Author: Generated for VI hyperparameter optimization
"""

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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Suppress some warnings during optimization
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import VI components
from VariationalInference.data_loader import DataLoader
from VariationalInference.vi import VI
from VariationalInference.config import VIConfig
from VariationalInference.utils import compute_metrics

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
    """
    return {
        # Learning rate (for SVI) - log scale
        'learning_rate': {
            'type': 'float',
            'low': 1e-4,
            'high': 100,
            'log': True,
            'description': 'Learning rate for stochastic optimization'
        },

        # Prior shape parameters (alpha)
        'alpha_theta': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0,
            'log': False,
            'description': 'Prior shape for sample factor activities'
        },
        'alpha_beta': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0,
            'log': False,
            'description': 'Prior shape for gene loadings'
        },
        'alpha_xi': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0,
            'log': False,
            'description': 'Prior shape for sample depth'
        },
        'alpha_eta': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0,
            'log': False,
            'description': 'Prior shape for gene scaling'
        },

        # Prior rate parameters (lambda)
        'lambda_xi': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0,
            'log': False,
            'description': 'Prior rate for sample depth'
        },
        'lambda_eta': {
            'type': 'float',
            'low': 0.5,
            'high': 5.0,
            'log': False,
            'description': 'Prior rate for gene scaling'
        },

        # Classification weight regularization
        'sigma_v': {
            'type': 'float',
            'low': 0.01,
            'high': 4.0,
            'log': True,
            'description': 'Prior std for classification weights'
        },
        'sigma_gamma': {
            'type': 'float',
            'low': 0.01,
            'high': 4.0,
            'log': True,
            'description': 'Prior std for auxiliary effects'
        },

        # Spike-and-slab sparsity priors
        'pi_v': {
            'type': 'float',
            'low': 0.1,
            'high': 0.99,
            'log': False,
            'description': 'Prior prob of v being active (slab)'
        },
        'pi_beta': {
            'type': 'float',
            'low': 0.01,
            'high': 1.0,
            'log': True,
            'description': 'Prior prob of beta being active'
        },

        # Classification-reconstruction tradeoff
        'regression_weight': {
            'type': 'float',
            'low': 0.1,
            'high': 10.0,
            'log': True,
            'description': 'Weight for classification objective'
        },

        # Model complexity
        'n_factors': {
            'type': 'int',
            'low': 50,
            'high': 500,
            'step': 10,
            'description': 'Number of latent factors (d)'
        },
    }


def sample_hyperparameters(
    trial: optuna.Trial,
    search_space: Dict[str, Dict[str, Any]],
    params_to_tune: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Sample hyperparameters from the search space for a given Optuna trial.

    Args:
        trial: Optuna trial object
        search_space: Dictionary defining the search space
        params_to_tune: List of parameter names to tune (None = all)

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
            params[name] = trial.suggest_float(
                name,
                spec['low'],
                spec['high'],
                log=spec.get('log', False)
            )
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
    ):
        """
        Initialize the objective function.

        Args:
            data_path: Path to the h5ad data file
            label_column: Column name for binary labels
            aux_columns: List of auxiliary feature column names
            gene_annotation_path: Path to gene annotation CSV (optional)
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
        """
        self.data_path = data_path
        self.label_column = label_column
        self.aux_columns = aux_columns or []
        self.gene_annotation_path = gene_annotation_path
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

    def _load_data(self):
        """Load and preprocess data (called once)."""
        if self._data_loaded:
            return

        logger.info(f"Loading data from {self.data_path}")

        loader = DataLoader(
            data_path=self.data_path,
            gene_annotation_path=self.gene_annotation_path
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

        self._data_loaded = True

        logger.info(f"Data loaded: {self._X_train.shape[0]} train, "
                   f"{self._X_val.shape[0]} val, {self._X_test.shape[0]} test samples")
        logger.info(f"Number of genes: {self._X_train.shape[1]}")

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

        # Extract n_factors and learning_rate separately
        n_factors = params.pop('n_factors', 50)
        learning_rate = params.pop('learning_rate', 0.01)

        if self.verbose:
            logger.info(f"Trial {trial.number}: n_factors={n_factors}, "
                       f"learning_rate={learning_rate:.4f}")

        try:
            # Create model
            model = VI(
                n_factors=n_factors,
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
            )

            # Train model
            if self.method == 'vi':
                model.fit(
                    X=self._X_train,
                    y=self._y_train,
                    X_aux=self._X_aux_train,
                    max_iter=self.max_iter,
                    patience=self.early_stopping_patience,
                    verbose=self.verbose
                )
            else:
                # SVI - import if needed
                from VariationalInference.svi import SVI
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
                )
                model.fit(
                    X=self._X_train,
                    y=self._y_train,
                    X_aux=self._X_aux_train,
                    max_epochs=self.max_iter,
                    verbose=self.verbose
                )

            # Evaluate on validation set
            y_val_proba = model.predict_proba(
                self._X_val,
                self._X_aux_val,
                verbose=False
            )

            # Compute validation metrics
            val_auc = roc_auc_score(self._y_val.ravel(), y_val_proba.ravel())
            y_val_pred = (y_val_proba.ravel() > 0.5).astype(int)
            metrics = compute_metrics(self._y_val.ravel(), y_val_pred, y_val_proba.ravel())
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
# CONFIGURATION GENERATOR
# =============================================================================

def generate_config_from_params(
    params: Dict[str, Any],
    base_config: Optional[VIConfig] = None
) -> VIConfig:
    """
    Generate a VIConfig from optimized parameters.

    Args:
        params: Dictionary of optimized hyperparameters
        base_config: Optional base config to modify

    Returns:
        VIConfig with optimized parameters
    """
    if base_config is None:
        base_config = VIConfig()

    # Create updates dict
    updates = {}

    # Map parameter names to config attributes
    param_mapping = {
        'n_factors': 'n_factors',
        'alpha_theta': 'alpha_theta',
        'alpha_beta': 'alpha_beta',
        'alpha_xi': 'alpha_xi',
        'alpha_eta': 'alpha_eta',
        'lambda_xi': 'lambda_xi',
        'lambda_eta': 'lambda_eta',
        'sigma_v': 'sigma_v',
        'sigma_gamma': 'sigma_gamma',
        'pi_v': 'pi_v',
        'pi_beta': 'pi_beta',
        'regression_weight': 'regression_weight',
        'learning_rate': 'learning_rate',
    }

    for param_name, config_attr in param_mapping.items():
        if param_name in params:
            updates[config_attr] = params[param_name]

    return base_config.copy(**updates)


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
  # Basic usage
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --n-trials 50

  # With specific parameters to tune
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --params-to-tune n_factors sigma_v pi_v regression_weight \\
      --n-trials 100

  # SVI with more iterations
  python -m VariationalInference.bayes_opt \\
      --data data.h5ad \\
      --label-column t2dm \\
      --method svi \\
      --max-iter 50 \\
      --n-trials 100
        """
    )

    # Data arguments
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to h5ad data file'
    )
    parser.add_argument(
        '--label-column', '-l',
        required=True,
        help='Column name for binary labels'
    )
    parser.add_argument(
        '--aux-columns', '-a',
        nargs='+',
        default=None,
        help='Auxiliary feature column names'
    )
    parser.add_argument(
        '--gene-annotation',
        default=None,
        help='Path to gene annotation CSV file'
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

    # Create study name
    study_name = args.study_name or f"vi_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=" * 60)
    logger.info("BAYESIAN OPTIMIZATION FOR VI HYPERPARAMETERS")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")
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

    # Create objective function
    objective = VIObjective(
        data_path=args.data,
        label_column=args.label_column,
        aux_columns=args.aux_columns,
        gene_annotation_path=args.gene_annotation,
        method=args.method,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed,
        params_to_tune=args.params_to_tune,
        fixed_params=fixed_params,
        verbose=args.verbose,
        multi_objective=args.multi_objective
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
