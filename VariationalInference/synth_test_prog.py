"""
Progressive Validation on Full Gene Set
Tests disease gene recovery on increasingly larger sample subsets.
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
import sys

sys.path.append('/labs/Aguiar/SSPA_BRAY/BRay/VariationalInference')
from vi import VI


def load_synthetic_data(data_dir):
    """Load full synthetic disease data."""
    data_dir = Path(data_dir)
    
    with open(data_dir / 'X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(data_dir / 'y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(data_dir / 'X_aux_train.pkl', 'rb') as f:
        X_aux_train = pickle.load(f)
    
    with open(data_dir / 'X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(data_dir / 'y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    with open(data_dir / 'X_aux_test.pkl', 'rb') as f:
        X_aux_test = pickle.load(f)
    
    with open(data_dir / 'ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    with open(data_dir / 'gene_list.txt', 'r') as f:
        gene_names = [line.strip() for line in f]
    
    return X_train, y_train, X_aux_train, X_test, y_test, X_aux_test, ground_truth, gene_names


def subsample_data(X, y, X_aux, n_samples, seed=42):
    """Subsample to n_samples, keeping disease proportion."""
    rng = np.random.RandomState(seed)
    
    if n_samples >= X.shape[0]:
        return X, y, X_aux
    
    # Stratified sampling to keep disease proportion
    y_flat = y.ravel()
    disease_idx = np.where(y_flat == 1)[0]
    control_idx = np.where(y_flat == 0)[0]
    
    # Sample proportionally
    n_disease = int(n_samples * len(disease_idx) / len(y_flat))
    n_control = n_samples - n_disease
    
    disease_sample = rng.choice(disease_idx, min(n_disease, len(disease_idx)), replace=False)
    control_sample = rng.choice(control_idx, min(n_control, len(control_idx)), replace=False)
    
    idx = np.concatenate([disease_sample, control_sample])
    rng.shuffle(idx)
    
    return X[idx], y[idx], X_aux[idx]


def compute_recovery_metrics(model, ground_truth, top_k=50):
    """Compute gene recovery metrics."""
    disease_genes = set(ground_truth['disease_gene_indices'])
    n_disease = len(disease_genes)
    
    beta = model.E_beta  # (p, d)
    
    # Find best factor for each disease gene
    best_f1 = 0
    best_factor = None
    best_metrics = None
    
    for k in range(beta.shape[1]):
        beta_k = beta[:, k]
        
        # Get top genes
        top_idx = set(np.argsort(beta_k)[-top_k:])
        
        # Compute metrics
        tp = len(top_idx & disease_genes)
        precision = tp / top_k
        recall = tp / n_disease
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_factor = k
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'v_weight': model.mu_v[0, k] if model.mu_v.shape[0] == 1 else model.mu_v[:, k].mean()
            }
    
    return best_factor, best_metrics


def train_and_evaluate(X_train, y_train, X_aux_train, X_test, y_test, X_aux_test,
                       ground_truth, stage_name, n_factors, max_iter, pi_beta=0.3):
    """Train model and evaluate recovery.
    
    Parameters:
    -----------
    pi_beta : float
        Probability that beta entry is non-zero (1 - sparsity).
        Default 0.3 means ~70% of beta entries are zero (sparse).
    """
    print(f"\n{'='*70}")
    print(f"{stage_name}")
    print(f"{'='*70}")
    print(f"Train: {X_train.shape[0]} samples × {X_train.shape[1]} genes")
    print(f"Test:  {X_test.shape[0]} samples × {X_test.shape[1]} genes")
    print(f"Disease: train={np.sum(y_train==1)}, test={np.sum(y_test==1)}")
    
    # Train
    print(f"\nTraining ({n_factors} factors, {max_iter} iters, pi_beta={pi_beta:.2f})...")
    # Spike-and-slab prior: pi_beta controls sparsity (probability of non-zero)
    model = VI(n_factors=n_factors, sigma_v=0.1, sigma_gamma=0.1, 
               pi_beta=pi_beta,
               random_state=42)
    
    start = time.time()
    model.fit(X_train, y_train, X_aux_train, 
              max_iter=max_iter, 
              elbo_freq=10, 
              verbose=True,
              min_iter=20,
              patience=3)
    elapsed = time.time() - start
    
    print(f"\n✓ Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Check if spike-and-slab indicators are collapsing
    print("rho_beta stats:", model.rho_beta.min(), model.rho_beta.max(), model.rho_beta.mean())
    print("rho_v stats:", model.rho_v.min(), model.rho_v.max(), model.rho_v.mean())

    # Check regression signal
    logits = model.E_theta @ model.E_v.T + X_aux_train @ model.E_gamma.T
    print("Logit range:", logits.min(), logits.max(), logits.std())

    # Test with pi_v=1.0, pi_beta=1.0 to disable spike-and-slab
    model_test = VI(n_factors=n_factors, sigma_v=0.1, sigma_gamma=0.1,
                    pi_v=1.0, pi_beta=1.0,
                    random_state=42)
    model_test.fit(X_train, y_train, X_aux_train,
                   max_iter=max_iter,
                   elbo_freq=10,
                   verbose=True,
                   min_iter=20,
                   patience=3)

    # Evaluate gene recovery
    print(f"\n{'='*70}")
    print("GENE RECOVERY ANALYSIS")
    print(f"{'='*70}")
    
    best_factor, metrics = compute_recovery_metrics(model, ground_truth, top_k=50)
    
    print(f"\nBest Factor: {best_factor}")
    print(f"  Precision@50: {metrics['precision']:.3f} ({metrics['tp']}/50)")
    print(f"  Recall@50:    {metrics['recall']:.3f} ({metrics['tp']}/{ground_truth['n_disease_genes']})")
    print(f"  F1 score:     {metrics['f1']:.3f}")
    print(f"  V-weight:     {metrics['v_weight']:.4f}")
    
    # Predict on train and test sets
    print(f"\n{'='*70}")
    print("PREDICTIONS")
    print(f"{'='*70}")
    
    from sklearn.metrics import roc_auc_score
    
    print("Computing train predictions...")
    train_probs = model.predict_proba(X_train, X_aux_train, verbose=False)
    train_auc = roc_auc_score(y_train.ravel(), train_probs.ravel())
    
    print("Computing test predictions...")
    test_probs = model.predict_proba(X_test, X_aux_test, verbose=False)
    test_auc = roc_auc_score(y_test.ravel(), test_probs.ravel())
    
    print(f"Train AUC: {train_auc:.3f}")
    print(f"Test AUC: {test_auc:.3f}")
    
    # V-weight statistics
    v_std = model.mu_v.std()
    v_range = (model.mu_v.min(), model.mu_v.max())
    print(f"\nV-weight stats:")
    print(f"  Std:   {v_std:.4f}")
    print(f"  Range: [{v_range[0]:.4f}, {v_range[1]:.4f}]")
    
    return {
        'stage': stage_name,
        'n_samples': X_train.shape[0],
        'time': elapsed,
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'train_auc': train_auc,
        'test_auc': test_auc,
        'best_factor': best_factor,
        'v_weight': metrics['v_weight'],
        'final_elbo': model.elbo_history_[-1][1] if model.elbo_history_ else None,
        'model': model,  # Include full model object
        # Predictions for exploration
        'train_probs': train_probs,
        'train_labels': y_train,
        'test_probs': test_probs,
        'test_labels': y_test
    }


def main(data_dir, n_stages=3):
    """
    Run progressive validation on sample subsets.
    
    Parameters:
    -----------
    data_dir : str
        Path to synthetic disease data
    n_stages : int
        Number of stages (1-4)
    """
    data_dir = Path(data_dir)
    output_dir = data_dir / 'progressive_validation'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("PROGRESSIVE VALIDATION - FULL GENES, SUBSET SAMPLES")
    print("="*70)
    
    # Load full data
    X_train, y_train, X_aux_train, X_test, y_test, X_aux_test, ground_truth, gene_names = load_synthetic_data(data_dir)
    
    print(f"\nFull data: {X_train.shape[0]} samples × {X_train.shape[1]} genes")
    print(f"Disease genes: {ground_truth['n_disease_genes']}")
    print(f"Disease samples: {np.sum(y_train==1)} train, {np.sum(y_test==1)} test")
    
    # Define stages
    # pi_beta = 0.3 means 70% of beta entries are zero (sparse prior)
    all_stages = [
        {
            'name': 'Stage 1: Small',
            'n_samples': 300,
            'n_factors': 15,
            'max_iter': 100,
            'pi_beta': 0.3,  # 70% sparsity
        },
        {
            'name': 'Stage 2: Medium',
            'n_samples': 600,
            'n_factors': 20,
            'max_iter': 150,
            'pi_beta': 0.3,  # 70% sparsity
        },
        {
            'name': 'Stage 3: Large',
            'n_samples': 900,
            'n_factors': 20,
            'max_iter': 150,
            'pi_beta': 0.3,  # 70% sparsity
        },
        {
            'name': 'Stage 4: Full',
            'n_samples': X_train.shape[0],
            'n_factors': 20,
            'max_iter': 200,
            'pi_beta': 0.3,  # 70% sparsity
        }
    ]
    
    stages = all_stages[:n_stages]
    
    # Run progressive validation
    results = []
    
    for stage in stages:
        # Subsample training data
        X_train_sub, y_train_sub, X_aux_train_sub = subsample_data(
            X_train, y_train, X_aux_train, stage['n_samples']
        )
        
        # Train and evaluate
        result = train_and_evaluate(
            X_train_sub, y_train_sub, X_aux_train_sub,
            X_test, y_test, X_aux_test,
            ground_truth,
            stage['name'],
            stage['n_factors'],
            stage['max_iter'],
            pi_beta=stage.get('pi_beta', 0.3)  # Default to 0.3 (70% sparsity)
        )
        
        results.append(result)
        
        # Save results and model
        stage_name_safe = stage["name"].lower().replace(" ", "_")
        results_path = output_dir / f'{stage_name_safe}_results.pkl'
        model_path = output_dir / f'{stage_name_safe}_model.pkl'
        
        with open(results_path, 'wb') as f:
            pickle.dump(result, f)
        
        # Save trained model with all parameters
        model_data = {
            'model': result.get('model'),  # Full model object
            'E_beta': result['model'].E_beta if 'model' in result else None,
            'E_theta': result['model'].E_theta if 'model' in result else None,
            'mu_v': result['model'].mu_v if 'model' in result else None,
            'sigma_v': result['model'].sigma_v if 'model' in result else None,
            'mu_gamma': result['model'].mu_gamma if 'model' in result else None,
            'sigma_gamma': result['model'].sigma_gamma if 'model' in result else None,
            'elbo_history': result['model'].elbo_history_ if 'model' in result else None,
            'stage_info': {
                'name': stage['name'],
                'n_samples': stage['n_samples'],
                'n_factors': stage['n_factors'],
                'max_iter': stage['max_iter']
            },
            # Predictions for exploration
            'train_probs': result.get('train_probs'),
            'train_labels': result.get('train_labels'),
            'test_probs': result.get('test_probs'),
            'test_labels': result.get('test_labels'),
            'train_auc': result.get('train_auc'),
            'test_auc': result.get('test_auc')
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Results saved to: {results_path}")
        print(f"✓ Model saved to: {model_path}")
        
        # Check if we should continue
        if result['f1'] < 0.2:
            print(f"\n{'!'*70}")
            print(f"! WARNING: F1 = {result['f1']:.3f} < 0.2")
            print(f"! Recovery is poor. Consider stopping or using stronger signal.")
            print(f"{'!'*70}")
            
            response = input("\nContinue to next stage? (y/n): ")
            if response.lower() != 'y':
                print("Stopping early.")
                break
    
    # Summary
    print(f"\n{'='*70}")
    print("PROGRESSIVE VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Stage':<20} {'Samples':<10} {'Time':<10} {'F1':<8} {'Test AUC':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['stage']:<20} {r['n_samples']:<10} {r['time']/60:>6.1f}m   {r['f1']:<8.3f} {r['test_auc']:<10.3f}")
    
    print(f"\n{'='*70}")
    
    # Trend analysis
    if len(results) > 1:
        f1_trend = "↑" if results[-1]['f1'] > results[0]['f1'] else "↓"
        f1_change = results[-1]['f1'] - results[0]['f1']
        
        print(f"\nF1 Trend: {f1_trend} {f1_change:+.3f} ({results[0]['f1']:.3f} → {results[-1]['f1']:.3f})")
        
        if f1_trend == "↑" and results[-1]['f1'] > 0.3:
            print("✓ Recovery improving with more data!")
            print("  Recommendation: Continue to next stage or full run")
        elif f1_trend == "↓":
            print("✗ Recovery decreasing with more data!")
            print("  Recommendation: Check model or use stronger signal")
        else:
            print("⚠ Recovery flat or marginal")
            print("  Recommendation: Use stronger signal (4x boost, 500 genes)")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                       help='Synthetic disease data directory')
    parser.add_argument('--n_stages', type=int, default=3,
                       help='Number of stages (1-4, default=3)')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.n_stages)