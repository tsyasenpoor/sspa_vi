"""
Hyperparameter Optimization for VI Model using Bayesian Optimization (Optuna)

This script performs Bayesian optimization to find the best hyperparameters
for the VI model by maximizing validation F1 score.

Usage:
    python hyperparameter_optimization.py --n_trials 100 --n_jobs 1
"""

import json
import numpy as np
import pandas as pd
import os
import sys
import gc
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import pickle
import argparse
from datetime import datetime

# Determine base directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)  # Parent of VariationalInference/
sys.path.append(base_dir)

from VariationalInference.vi import VI

# Override base_dir for data loading if it exists
data_base_dir = '/labs/Aguiar/SSPA_BRAY/BRay'
if not os.path.exists(data_base_dir):
    # Fall back to local data directory
    data_base_dir = base_dir


def load_data():
    """Load and prepare the data (same as quick_reference.py)."""
    print("Loading data...")
    df = pd.read_pickle(os.path.join(data_base_dir, 'sspa_bcell/df.pkl'))
    features = pd.read_pickle(os.path.join(data_base_dir, 'sspa_bcell/features.pkl'))

    # Load splits
    with open(os.path.join(data_base_dir, 'sspa_bcell/data_split_cell_ids.json'), 'r') as f:
        splits = json.load(f)

    # Load gene list
    with open(os.path.join(data_base_dir, 'sspa_bcell/gene_list.txt'), 'r') as f:
        gene_list = [line.strip() for line in f]

    print(f"Training cells: {len(splits['train'])}")
    print(f"Validation cells: {len(splits['val'])}")
    print(f"Test cells: {len(splits['test'])}")
    print(f"Genes: {len(gene_list)}")

    return df, features, splits, gene_list


def prepare_matrices(df, features, cell_ids, gene_list):
    """Extract X, X_aux, y for given cell IDs."""
    df_subset = df.loc[df.index.isin(cell_ids)]
    features_subset = features.loc[features.index.isin(cell_ids)]

    # Align
    common_idx = df_subset.index.intersection(features_subset.index)
    df_subset = df_subset.loc[common_idx]
    features_subset = features_subset.loc[common_idx]

    # Extract matrices - use float32 to reduce memory by 50%
    X = df_subset[gene_list].values.astype(np.float32)
    X_aux = np.zeros((X.shape[0], 0), dtype=np.float32)  # No auxiliary features
    y_col = 't2dm' if 't2dm' in features_subset.columns else features_subset.columns[0]
    y = features_subset[y_col].values.astype(int)

    return X, X_aux, y


def objective(trial, X_train, X_aux_train, y_train, X_val, X_aux_val, y_val, args):
    """
    Objective function for Optuna optimization.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object for suggesting hyperparameters
    X_train, X_aux_train, y_train : Training data
    X_val, X_aux_val, y_val : Validation data
    args : Command line arguments

    Returns:
    --------
    val_f1 : float
        Validation F1 score (to be maximized)
    """

    # Suggest hyperparameters with reasonable ranges
    hyperparams = {
        # Number of latent factors - reduced range to save memory
        # With 9791 genes: n_factors=500 creates ~5M element matrices
        # Original range 50-1500 was too memory-intensive
        'n_factors': trial.suggest_int('n_factors', 25, 300, step=25),

        # Gamma prior parameters (shape parameters - typically 0.1 to 10)
        'alpha_theta': trial.suggest_float('alpha_theta', 0.1, 5.0, log=True),
        'alpha_beta': trial.suggest_float('alpha_beta', 0.1, 5.0, log=True),
        'alpha_xi': trial.suggest_float('alpha_xi', 0.1, 5.0, log=True),
        'alpha_eta': trial.suggest_float('alpha_eta', 0.1, 5.0, log=True),

        # Rate parameters for xi and eta
        'lambda_xi': trial.suggest_float('lambda_xi', 0.5, 5.0, log=True),
        'lambda_eta': trial.suggest_float('lambda_eta', 0.5, 5.0, log=True),

        # Gaussian prior variances for v and gamma (smaller = more regularization)
        'sigma_v': trial.suggest_float('sigma_v', 0.1, 5.0, log=True),
        'sigma_gamma': trial.suggest_float('sigma_gamma', 0.1, 5.0, log=True),

        # Spike-and-slab parameters
        'pi_v': trial.suggest_float('pi_v', 0.05, 0.5),  # Prior prob of v being active
        'pi_beta': trial.suggest_float('pi_beta', 0.01, 0.2),  # Prior prob of beta being active
        'spike_variance_v': trial.suggest_float('spike_variance_v', 1e-8, 1e-4, log=True),
        'spike_value_beta': trial.suggest_float('spike_value_beta', 1e-8, 1e-4, log=True),
    }

    # Damping parameters (if we want to optimize these too)
    damping_params = {
        'theta_damping': trial.suggest_float('theta_damping', 0.5, 0.95),
        'beta_damping': trial.suggest_float('beta_damping', 0.5, 0.95),
        'v_damping': trial.suggest_float('v_damping', 0.3, 0.9),
        'gamma_damping': trial.suggest_float('gamma_damping', 0.3, 0.9),
        'xi_damping': trial.suggest_float('xi_damping', 0.7, 0.95),
        'eta_damping': trial.suggest_float('eta_damping', 0.7, 0.95),
    }

    print(f"\n{'='*80}")
    print(f"Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*80}")
    for key, value in hyperparams.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    try:
        # Initialize model with suggested hyperparameters
        model = VI(
            n_factors=hyperparams['n_factors'],
            alpha_theta=hyperparams['alpha_theta'],
            alpha_beta=hyperparams['alpha_beta'],
            alpha_xi=hyperparams['alpha_xi'],
            alpha_eta=hyperparams['alpha_eta'],
            lambda_xi=hyperparams['lambda_xi'],
            lambda_eta=hyperparams['lambda_eta'],
            sigma_v=hyperparams['sigma_v'],
            sigma_gamma=hyperparams['sigma_gamma'],
            pi_v=hyperparams['pi_v'],
            pi_beta=hyperparams['pi_beta'],
            spike_variance_v=hyperparams['spike_variance_v'],
            spike_value_beta=hyperparams['spike_value_beta'],
            random_state=42
        )

        # Fit model on training data
        model.fit(
            X=X_train,
            y=y_train,
            X_aux=X_aux_train,
            max_iter=args.max_iter,
            tol=10.0,
            rel_tol=2e-4,
            elbo_freq=10,
            min_iter=30,
            patience=5,
            verbose=False,  # Suppress verbose output during optimization
            **damping_params,
            debug=False
        )

        # Predict on validation set
        y_val_proba = model.predict_proba(X_val, X_aux_val, max_iter=100, verbose=False)
        y_val_proba_pos = y_val_proba.ravel()
        y_val_pred = (y_val_proba_pos > 0.5).astype(int)

        # Calculate validation metrics
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_proba_pos)

        # Report intermediate results to Optuna
        trial.set_user_attr('val_accuracy', val_acc)
        trial.set_user_attr('val_auc', val_auc)
        trial.set_user_attr('final_elbo', model.elbo_history_[-1][1] if model.elbo_history_ else -np.inf)

        print(f"\nValidation Results:")
        print(f"  F1 Score: {val_f1:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  AUC: {val_auc:.4f}")
        print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}" if model.elbo_history_ else "  Final ELBO: N/A")

        # Clean up model to free memory
        del model
        gc.collect()

        return val_f1

    except Exception as e:
        print(f"\n❌ Trial {trial.number} failed with error: {str(e)}")
        # Clean up on failure too
        gc.collect()
        # Return a very low score for failed trials
        return 0.0


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for VI model')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    parser.add_argument('--max_iter', type=int, default=200,
                        help='Maximum iterations for VI training (default: 200)')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for the study (default: auto-generated)')
    parser.add_argument('--output_dir', type=str, default='./optuna_results',
                        help='Output directory for results (default: ./optuna_results)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds for optimization (default: None)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate study name if not provided
    if args.study_name is None:
        args.study_name = f"vi_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*80}")
    print(f"Starting Bayesian Hyperparameter Optimization")
    print(f"{'='*80}")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Max iterations per trial: {args.max_iter}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Load data
    df, features, splits, gene_list = load_data()

    # Prepare training and validation data
    X_train, X_aux_train, y_train = prepare_matrices(df, features, splits['train'], gene_list)
    X_val, X_aux_val, y_val = prepare_matrices(df, features, splits['val'], gene_list)

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_train distribution: {np.bincount(y_train)}")
    print(f"  y_val distribution: {np.bincount(y_val)}\n")

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # Maximize F1 score
        sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Don't prune for first 5 trials
            n_warmup_steps=10    # Wait 10 steps before pruning
        )
    )

    # Run optimization
    print(f"Starting optimization with {args.n_trials} trials...\n")
    study.optimize(
        lambda trial: objective(trial, X_train, X_aux_train, y_train,
                                X_val, X_aux_val, y_val, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=True
    )

    # Print results
    print(f"\n{'='*80}")
    print(f"Optimization Complete!")
    print(f"{'='*80}")
    print(f"\nBest trial:")
    print(f"  Trial number: {study.best_trial.number}")
    print(f"  Validation F1: {study.best_trial.value:.4f}")
    if 'val_accuracy' in study.best_trial.user_attrs:
        print(f"  Validation Accuracy: {study.best_trial.user_attrs['val_accuracy']:.4f}")
    if 'val_auc' in study.best_trial.user_attrs:
        print(f"  Validation AUC: {study.best_trial.user_attrs['val_auc']:.4f}")

    print(f"\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save results
    results_file = os.path.join(args.output_dir, f'{args.study_name}_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(study, f)
    print(f"\n✓ Saved study object to: {results_file}")

    # Save best parameters as JSON
    best_params_file = os.path.join(args.output_dir, f'{args.study_name}_best_params.json')
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'best_trial_number': study.best_trial.number,
            'user_attrs': study.best_trial.user_attrs,
            'n_trials': len(study.trials),
            'datetime': datetime.now().isoformat()
        }, f, indent=2)
    print(f"✓ Saved best parameters to: {best_params_file}")

    # Save full trials dataframe
    df_trials = study.trials_dataframe()
    trials_csv = os.path.join(args.output_dir, f'{args.study_name}_trials.csv')
    df_trials.to_csv(trials_csv, index=False)
    print(f"✓ Saved trials dataframe to: {trials_csv}")

    # Generate and save visualizations
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_html(os.path.join(args.output_dir, f'{args.study_name}_history.html'))
        print(f"✓ Saved optimization history plot")

        # Parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_html(os.path.join(args.output_dir, f'{args.study_name}_importance.html'))
        print(f"✓ Saved parameter importance plot")

    except Exception as e:
        print(f"⚠ Could not generate visualizations: {str(e)}")

    print(f"\n{'='*80}")
    print(f"All results saved to: {args.output_dir}")
    print(f"{'='*80}\n")

    return study


if __name__ == "__main__":
    main()
