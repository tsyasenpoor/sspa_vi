"""
Efficient Random Search for n_factors (Gene Programs)
======================================================
This script performs a focused random search on the n_factors parameter,
which controls the number of latent gene programs in the VI model.

Strategy:
- Random sampling in log-space for better coverage
- Early stopping within each trial to save time (50 max iters vs 200 for full training)
- Different random seeds per trial for exploration of initialization space
- Checkpointing to resume interrupted searches
- Lightweight validation metrics (AUC only)
- Parallel-friendly design (run multiple instances)

Speed optimizations for hyperparameter search:
- Reduced max iterations (50 vs 200) - we only need relative ranking, not perfect convergence
- Reduced min iterations (15 vs 50) - faster early stopping
- Tighter patience (3 vs 5) - quit non-promising configs sooner
- More frequent ELBO checks (every 5 vs 10 iters) - catch early stopping earlier
- Each trial uses different random seed for diverse initializations

Expected speedup: ~3-5x faster per trial compared to full training
"""

import json
import numpy as np
import pandas as pd
import os
import sys
import pickle
import gzip
from datetime import datetime
from sklearn.metrics import roc_auc_score
import argparse

base_dir = '/labs/Aguiar/SSPA_BRAY/BRay'
sys.path.append(base_dir)

from VariationalInference.vi import VI


def prepare_matrices(df, features, cell_ids, gene_list):
    """Extract X, X_aux, y for given cell IDs."""
    df_subset = df.loc[df.index.isin(cell_ids)]
    features_subset = features.loc[features.index.isin(cell_ids)]
    
    # Align
    common_idx = df_subset.index.intersection(features_subset.index)
    df_subset = df_subset.loc[common_idx]
    features_subset = features_subset.loc[common_idx]
    
    # Extract matrices
    X = df_subset[gene_list].values
    X_aux = np.zeros((X.shape[0], 0))
    y_col = 't2dm' if 't2dm' in features_subset.columns else features_subset.columns[0]
    y = features_subset[y_col].values.astype(int)
    
    return X, X_aux, y


def evaluate_n_factors(
    n_factors,
    X_train, y_train, X_aux_train,
    X_val, y_val, X_aux_val,
    trial_id,
    max_training_iter=50,  # Reduced for faster hyperparameter search
    early_stop_patience=3   # Reduced patience for faster trials
):
    """
    Train model with specific n_factors and evaluate on validation set.
    
    Returns:
        dict with trial results including AUC, training time, etc.
    """
    start_time = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"Trial {trial_id}: n_factors = {n_factors}")
    print(f"{'='*80}")
    
    # Fixed hyperparameters (you can adjust these based on your previous best config)
    # NOTE: Each trial uses a different random_state to explore different initialization points
    # This helps ensure we're not just finding a local optimum for one initialization
    model = VI(
        n_factors=n_factors,
        alpha_theta=0.5,
        alpha_beta=2.0,
        alpha_xi=2.0,
        lambda_xi=2.0,
        sigma_v=2.0,
        sigma_gamma=1.0,
        random_state=42 + trial_id  # Different seed per trial for diverse exploration
    )
    
    # Train with early stopping
    try:
        model.fit(
            X=X_train,
            y=y_train,
            X_aux=X_aux_train,
            max_iter=max_training_iter,
            tol=10.0,
            rel_tol=2e-4,
            elbo_freq=5,     # Check more frequently for early stopping
            min_iter=15,     # Reduced minimum iterations for faster trials
            patience=early_stop_patience,
            verbose=False,  # Reduce output during search
            theta_damping=0.8,
            beta_damping=0.8,
            v_damping=0.7,
            gamma_damping=0.7,
            xi_damping=0.9,
            eta_damping=0.9,
            debug=False
        )
        
        training_converged = True
        n_iterations = len(model.elbo_history_)
        
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        return {
            'trial_id': trial_id,
            'n_factors': n_factors,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    
    # Evaluate on validation set
    try:
        y_val_proba = model.predict_proba(X_val, X_aux_val, max_iter=50, verbose=False)  # Reduced from 100
        val_auc = roc_auc_score(y_val, y_val_proba.ravel())
        
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return {
            'trial_id': trial_id,
            'n_factors': n_factors,
            'status': 'eval_failed',
            'error': str(e),
            'n_iterations': n_iterations,
            'timestamp': datetime.now().isoformat()
        }
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Calculate efficiency metrics
    final_elbo = model.elbo_history_[-1] if model.elbo_history_ else None
    elbo_improvement = (model.elbo_history_[-1] - model.elbo_history_[0]) if len(model.elbo_history_) > 1 else None
    
    result = {
        'trial_id': trial_id,
        'n_factors': n_factors,
        'status': 'success',
        'val_auc': val_auc,
        'training_time_sec': duration,
        'n_iterations': n_iterations,
        'final_elbo': final_elbo,
        'elbo_improvement': elbo_improvement,
        'converged': training_converged,
        'timestamp': datetime.now().isoformat(),
        # Model complexity metrics
        'theta_mean': float(model.E_theta.mean()),
        'theta_std': float(model.E_theta.std()),
        'beta_mean': float(model.E_beta.mean()),
        'beta_std': float(model.E_beta.std()),
        'v_mean': float(model.E_v.mean()),
        'v_std': float(model.E_v.std()),
    }
    
    # Add sparsity metrics if available
    if hasattr(model, 'rho_beta') and hasattr(model, 'rho_v'):
        result['beta_sparsity'] = float(1 - (model.rho_beta > 0.5).mean())
        result['v_sparsity'] = float(1 - (model.rho_v > 0.5).mean())
    
    print(f"  ✓ Val AUC: {val_auc:.4f}")
    print(f"  ✓ Training time: {duration:.1f}s")
    print(f"  ✓ Iterations: {n_iterations}")
    
    return result


def run_random_search(
    n_trials=20,
    n_factors_min=100,
    n_factors_max=2000,
    log_sampling=True,
    checkpoint_file='nfactors_search_results.json',
    seed=42
):
    """
    Run random search for n_factors.
    
    Args:
        n_trials: Number of random configurations to try
        n_factors_min: Minimum number of factors
        n_factors_max: Maximum number of factors
        log_sampling: If True, sample in log-space for better coverage
        checkpoint_file: File to save results after each trial
        seed: Random seed for reproducibility
    """
    
    # Load data
    print("Loading data...")
    df = pd.read_pickle(os.path.join(base_dir, 'sspa_bcell/df.pkl'))
    features = pd.read_pickle(os.path.join(base_dir, 'sspa_bcell/features.pkl'))
    
    with open(os.path.join(base_dir, 'sspa_bcell/data_split_cell_ids.json'), 'r') as f:
        splits = json.load(f)
    
    with open(os.path.join(base_dir, 'sspa_bcell/gene_list.txt'), 'r') as f:
        gene_list = [line.strip() for line in f]
    
    print(f"  Training cells: {len(splits['train'])}")
    print(f"  Validation cells: {len(splits['val'])}")
    print(f"  Genes: {len(gene_list)}")
    
    # Prepare data once
    print("\nPreparing training and validation data...")
    X_train, X_aux_train, y_train = prepare_matrices(df, features, splits['train'], gene_list)
    X_val, X_aux_val, y_val = prepare_matrices(df, features, splits['val'], gene_list)
    
    # Sample n_factors values
    rng = np.random.RandomState(seed)
    
    if log_sampling:
        # Sample in log-space for better coverage of range
        log_min = np.log10(n_factors_min)
        log_max = np.log10(n_factors_max)
        log_samples = rng.uniform(log_min, log_max, size=n_trials)
        n_factors_samples = np.round(10 ** log_samples).astype(int)
    else:
        # Uniform sampling
        n_factors_samples = rng.randint(n_factors_min, n_factors_max + 1, size=n_trials)
    
    # Remove duplicates and sort for better monitoring
    n_factors_samples = sorted(set(n_factors_samples))
    actual_trials = len(n_factors_samples)
    
    print(f"\nRandom Search Configuration:")
    print(f"  Trials: {actual_trials} (unique)")
    print(f"  n_factors range: [{n_factors_samples[0]}, {n_factors_samples[-1]}]")
    print(f"  Sampling: {'log-space' if log_sampling else 'uniform'}")
    print(f"  Checkpoint file: {checkpoint_file}")
    
    # Load existing results if checkpoint exists
    all_results = []
    completed_n_factors = set()
    
    if os.path.exists(checkpoint_file):
        print(f"\nLoading existing checkpoint from '{checkpoint_file}'...")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                all_results = checkpoint.get('trials', [])
                completed_n_factors = {r['n_factors'] for r in all_results if r['status'] == 'success'}
            print(f"  Found {len(all_results)} previous trials ({len(completed_n_factors)} successful)")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ⚠ Checkpoint file corrupted or invalid: {e}")
            print(f"  Creating backup and starting fresh...")
            backup_file = checkpoint_file.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            os.rename(checkpoint_file, backup_file)
            print(f"  Old checkpoint saved as: {backup_file}")
    
    # Run trials
    print(f"\nStarting random search...")
    print(f"{'='*80}\n")
    
    for trial_id, n_factors in enumerate(n_factors_samples):
        if n_factors in completed_n_factors:
            print(f"Skipping trial {trial_id + 1}/{actual_trials}: n_factors={n_factors} (already completed)")
            continue
        
        result = evaluate_n_factors(
            n_factors=n_factors,
            X_train=X_train,
            y_train=y_train,
            X_aux_train=X_aux_train,
            X_val=X_val,
            y_val=y_val,
            X_aux_val=X_aux_val,
            trial_id=trial_id
        )
        
        all_results.append(result)
        
        # Save checkpoint after each trial
        checkpoint = {
            'search_config': {
                'n_trials': actual_trials,
                'n_factors_min': n_factors_min,
                'n_factors_max': n_factors_max,
                'log_sampling': log_sampling,
                'seed': seed
            },
            'data_info': {
                'n_train': X_train.shape[0],
                'n_val': X_val.shape[0],
                'n_genes': len(gene_list)
            },
            'trials': all_results,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"  Checkpoint saved ({len(all_results)} trials completed)\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print("Random Search Complete!")
    print(f"{'='*80}\n")
    
    successful_results = [r for r in all_results if r['status'] == 'success']
    
    if successful_results:
        results_df = pd.DataFrame(successful_results)
        results_df = results_df.sort_values('val_auc', ascending=False)
        
        print("Top 10 configurations by validation AUC:")
        print("-" * 80)
        for i, row in results_df.head(10).iterrows():
            print(f"  {row['n_factors']:4d} factors → AUC: {row['val_auc']:.4f} "
                  f"(time: {row['training_time_sec']:.1f}s, iters: {row['n_iterations']})")
        
        print(f"\nBest configuration:")
        best = results_df.iloc[0]
        print(f"  n_factors: {best['n_factors']}")
        print(f"  Val AUC: {best['val_auc']:.4f}")
        print(f"  Training time: {best['training_time_sec']:.1f}s")
        print(f"  Iterations: {best['n_iterations']}")
        
        # Save summary plot data
        summary_file = checkpoint_file.replace('.json', '_summary.csv')
        results_df.to_csv(summary_file, index=False)
        print(f"\n✓ Summary saved to: {summary_file}")
    
    else:
        print("⚠ No successful trials!")
    
    print(f"✓ Full results saved to: {checkpoint_file}")
    
    return all_results


def analyze_results(checkpoint_file='nfactors_search_results.json'):
    """
    Analyze and visualize results from a completed search.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    results = [r for r in checkpoint['trials'] if r['status'] == 'success']
    
    if not results:
        print("No successful trials to analyze!")
        return
    
    df = pd.DataFrame(results)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. AUC vs n_factors
    ax = axes[0, 0]
    ax.scatter(df['n_factors'], df['val_auc'], alpha=0.6, s=80)
    ax.set_xlabel('Number of Factors', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Validation AUC vs Number of Factors', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Training time vs n_factors
    ax = axes[0, 1]
    ax.scatter(df['n_factors'], df['training_time_sec'], alpha=0.6, s=80, c='orange')
    ax.set_xlabel('Number of Factors', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time vs Number of Factors', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. AUC vs Training time (efficiency)
    ax = axes[1, 0]
    scatter = ax.scatter(df['training_time_sec'], df['val_auc'], 
                        c=df['n_factors'], cmap='viridis', s=80, alpha=0.6)
    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Validation AUC', fontsize=12)
    ax.set_title('Efficiency: AUC vs Training Time', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='n_factors')
    ax.grid(True, alpha=0.3)
    
    # 4. Distribution of AUC scores
    ax = axes[1, 1]
    ax.hist(df['val_auc'], bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(df['val_auc'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["val_auc"].mean():.4f}')
    ax.axvline(df['val_auc'].max(), color='green', linestyle='--', linewidth=2, label=f'Best: {df["val_auc"].max():.4f}')
    ax.set_xlabel('Validation AUC', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Validation AUC', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = checkpoint_file.replace('.json', '_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Analysis plot saved to: {output_file}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*80)
    print("Search Statistics")
    print("="*80)
    print(f"Total trials: {len(df)}")
    print(f"n_factors range: [{df['n_factors'].min()}, {df['n_factors'].max()}]")
    print(f"\nValidation AUC:")
    print(f"  Best:   {df['val_auc'].max():.4f} (n_factors={df.loc[df['val_auc'].idxmax(), 'n_factors']:.0f})")
    print(f"  Mean:   {df['val_auc'].mean():.4f}")
    print(f"  Median: {df['val_auc'].median():.4f}")
    print(f"  Std:    {df['val_auc'].std():.4f}")
    print(f"\nTraining Time:")
    print(f"  Mean:   {df['training_time_sec'].mean():.1f}s")
    print(f"  Median: {df['training_time_sec'].median():.1f}s")
    print(f"  Range:  [{df['training_time_sec'].min():.1f}s, {df['training_time_sec'].max():.1f}s]")
    
    # Find sweet spot (good AUC, reasonable time)
    df['auc_per_minute'] = df['val_auc'] / (df['training_time_sec'] / 60)
    best_efficiency_idx = df['auc_per_minute'].idxmax()
    print(f"\nMost efficient configuration:")
    print(f"  n_factors: {df.loc[best_efficiency_idx, 'n_factors']:.0f}")
    print(f"  Val AUC: {df.loc[best_efficiency_idx, 'val_auc']:.4f}")
    print(f"  Training time: {df.loc[best_efficiency_idx, 'training_time_sec']:.1f}s")
    print(f"  AUC per minute: {df.loc[best_efficiency_idx, 'auc_per_minute']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random search for n_factors hyperparameter')
    parser.add_argument('--n_trials', type=int, default=20,
                      help='Number of random configurations to try (default: 20)')
    parser.add_argument('--min', type=int, default=100,
                      help='Minimum n_factors (default: 100)')
    parser.add_argument('--max', type=int, default=2000,
                      help='Maximum n_factors (default: 2000)')
    parser.add_argument('--linear', action='store_true',
                      help='Use linear sampling instead of log-space (default: log-space)')
    parser.add_argument('--checkpoint', type=str, default='nfactors_search_results.json',
                      help='Checkpoint file for saving results (default: nfactors_search_results.json)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--analyze', action='store_true',
                      help='Only analyze existing results, do not run new trials')
    
    args = parser.parse_args()
    
    if args.analyze:
        print("Analyzing existing results...")
        analyze_results(checkpoint_file=args.checkpoint)
    else:
        results = run_random_search(
            n_trials=args.n_trials,
            n_factors_min=args.min,
            n_factors_max=args.max,
            log_sampling=not args.linear,
            checkpoint_file=args.checkpoint,
            seed=args.seed
        )
        
        # Auto-analyze if we have results
        print("\nGenerating analysis plots...")
        try:
            analyze_results(checkpoint_file=args.checkpoint)
        except Exception as e:
            print(f"Could not generate plots: {e}")
            print("You can run analysis later with: python hyperparameter_search_nfactors.py --analyze")
