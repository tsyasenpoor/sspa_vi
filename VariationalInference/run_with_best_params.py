"""
Run VI model with best hyperparameters found by Optuna optimization.

This script loads the best hyperparameters from the optimization study
and trains a final model on the full training set, evaluating on both
validation and test sets.

Usage:
    python run_with_best_params.py --study_file optuna_results/vi_optimization_XXXXXX_best_params.json
"""

import json
import numpy as np
import pandas as pd
import os
import sys
import argparse
import pickle
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix

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
    """Load and prepare the data."""
    print("Loading data...")
    df = pd.read_pickle(os.path.join(data_base_dir, 'sspa_bcell/df.pkl'))
    features = pd.read_pickle(os.path.join(data_base_dir, 'sspa_bcell/features.pkl'))

    with open(os.path.join(data_base_dir, 'sspa_bcell/data_split_cell_ids.json'), 'r') as f:
        splits = json.load(f)

    with open(os.path.join(data_base_dir, 'sspa_bcell/gene_list.txt'), 'r') as f:
        gene_list = [line.strip() for line in f]

    return df, features, splits, gene_list


def prepare_matrices(df, features, cell_ids, gene_list):
    """Extract X, X_aux, y for given cell IDs."""
    df_subset = df.loc[df.index.isin(cell_ids)]
    features_subset = features.loc[features.index.isin(cell_ids)]

    common_idx = df_subset.index.intersection(features_subset.index)
    df_subset = df_subset.loc[common_idx]
    features_subset = features_subset.loc[common_idx]

    X = df_subset[gene_list].values
    X_aux = np.zeros((X.shape[0], 0))
    y_col = 't2dm' if 't2dm' in features_subset.columns else features_subset.columns[0]
    y = features_subset[y_col].values.astype(int)

    return X, X_aux, y


def print_metrics(y_true, y_pred, y_proba, split_name):
    """Print detailed metrics for a data split."""
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n{split_name} Results:")
    print(f"{'='*60}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description='Run VI model with best hyperparameters')
    parser.add_argument('--study_file', type=str, required=True,
                        help='Path to best_params.json file from optimization')
    parser.add_argument('--max_iter', type=int, default=300,
                        help='Maximum iterations for final training (default: 300)')
    parser.add_argument('--output_dir', type=str, default='./best_model_results',
                        help='Output directory for results (default: ./best_model_results)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load best parameters
    print(f"Loading best parameters from: {args.study_file}")
    with open(args.study_file, 'r') as f:
        study_results = json.load(f)

    best_params = study_results['best_params']
    print(f"\nBest parameters from optimization:")
    print(f"  Original F1 score: {study_results['best_value']:.4f}")
    print(f"  Trial number: {study_results['best_trial_number']}")
    print(f"\nHyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    # Load data
    df, features, splits, gene_list = load_data()
    X_train, X_aux_train, y_train = prepare_matrices(df, features, splits['train'], gene_list)
    X_val, X_aux_val, y_val = prepare_matrices(df, features, splits['val'], gene_list)
    X_test, X_aux_test, y_test = prepare_matrices(df, features, splits['test'], gene_list)

    # Separate VI hyperparameters from damping parameters
    vi_params = {k: v for k, v in best_params.items()
                 if k in ['n_factors', 'alpha_theta', 'alpha_beta', 'alpha_xi', 'alpha_eta',
                         'lambda_xi', 'lambda_eta', 'sigma_v', 'sigma_gamma',
                         'pi_v', 'pi_beta', 'spike_variance_v', 'spike_value_beta']}

    damping_params = {k: v for k, v in best_params.items()
                      if k.endswith('_damping')}

    # Initialize model with best hyperparameters
    print(f"\n{'='*80}")
    print("Training final model with best hyperparameters")
    print(f"{'='*80}\n")

    model = VI(
        n_factors=vi_params['n_factors'],
        alpha_theta=vi_params['alpha_theta'],
        alpha_beta=vi_params['alpha_beta'],
        alpha_xi=vi_params['alpha_xi'],
        alpha_eta=vi_params['alpha_eta'],
        lambda_xi=vi_params['lambda_xi'],
        lambda_eta=vi_params['lambda_eta'],
        sigma_v=vi_params['sigma_v'],
        sigma_gamma=vi_params['sigma_gamma'],
        pi_v=vi_params['pi_v'],
        pi_beta=vi_params['pi_beta'],
        spike_variance_v=vi_params['spike_variance_v'],
        spike_value_beta=vi_params['spike_value_beta'],
        random_state=42
    )

    # Train model
    model.fit(
        X=X_train,
        y=y_train,
        X_aux=X_aux_train,
        max_iter=args.max_iter,
        tol=10.0,
        rel_tol=2e-4,
        elbo_freq=10,
        min_iter=50,
        patience=5,
        verbose=True,
        **damping_params,
        debug=True
    )

    # Evaluate on all splits
    print(f"\n{'='*80}")
    print("Evaluating on all data splits")
    print(f"{'='*80}")

    # Training set
    y_train_proba = model.predict_proba(X_train, X_aux_train, max_iter=100, verbose=False)
    y_train_pred = (y_train_proba.ravel() > 0.5).astype(int)
    print_metrics(y_train, y_train_pred, y_train_proba.ravel(), "Training")

    # Validation set
    y_val_proba = model.predict_proba(X_val, X_aux_val, max_iter=100, verbose=False)
    y_val_pred = (y_val_proba.ravel() > 0.5).astype(int)
    print_metrics(y_val, y_val_pred, y_val_proba.ravel(), "Validation")

    # Test set
    y_test_proba = model.predict_proba(X_test, X_aux_test, max_iter=100, verbose=False)
    y_test_pred = (y_test_proba.ravel() > 0.5).astype(int)
    print_metrics(y_test, y_test_pred, y_test_proba.ravel(), "Test")

    # Save model
    model_file = os.path.join(args.output_dir, 'best_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Saved trained model to: {model_file}")

    # Save predictions
    for split_name, (y_true, y_proba, cell_ids) in [
        ('train', (y_train, y_train_proba, splits['train'])),
        ('val', (y_val, y_val_proba, splits['val'])),
        ('test', (y_test, y_test_proba, splits['test']))
    ]:
        pred_df = pd.DataFrame({
            'cell_id': cell_ids,
            'true_label': y_true,
            'pred_prob': y_proba.ravel(),
            'pred_label': (y_proba.ravel() > 0.5).astype(int)
        })
        pred_file = os.path.join(args.output_dir, f'{split_name}_predictions.csv')
        pred_df.to_csv(pred_file, index=False)
        print(f"✓ Saved {split_name} predictions to: {pred_file}")

    # Save summary
    summary = {
        'hyperparameters': best_params,
        'metrics': {
            'train': {
                'f1': f1_score(y_train, y_train_pred, zero_division=0),
                'accuracy': accuracy_score(y_train, y_train_pred),
                'auc': roc_auc_score(y_train, y_train_proba.ravel())
            },
            'val': {
                'f1': f1_score(y_val, y_val_pred, zero_division=0),
                'accuracy': accuracy_score(y_val, y_val_pred),
                'auc': roc_auc_score(y_val, y_val_proba.ravel())
            },
            'test': {
                'f1': f1_score(y_test, y_test_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, y_test_pred),
                'auc': roc_auc_score(y_test, y_test_proba.ravel())
            }
        },
        'elbo_history': [(int(i), float(e)) for i, e in model.elbo_history_],
        'optimization_trial': study_results['best_trial_number'],
        'optimization_f1': study_results['best_value']
    }

    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {summary_file}")

    print(f"\n{'='*80}")
    print(f"All results saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
