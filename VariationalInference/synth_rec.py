"""
Validate Synthetic Disease Recovery

Trains model on synthetic data and checks:
1. Does model recover disease genes in top beta weights?
2. Do v-weights become positive for disease-relevant factors?
3. What's the prediction AUC?
4. Gene recovery metrics (precision, recall)
"""

import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import sys

sys.path.append('/labs/Aguiar/SSPA_BRAY/BRay/VariationalInference')
from vi import VI


def load_synthetic_data(data_dir):
    """Load synthetic disease data."""
    data_dir = Path(data_dir)
    
    print(f"Loading synthetic data from: {data_dir}")
    
    # Load train data
    with open(data_dir / 'X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open(data_dir / 'y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open(data_dir / 'X_aux_train.pkl', 'rb') as f:
        X_aux_train = pickle.load(f)
    
    # Load test data
    with open(data_dir / 'X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(data_dir / 'y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    with open(data_dir / 'X_aux_test.pkl', 'rb') as f:
        X_aux_test = pickle.load(f)
    
    # Load ground truth
    with open(data_dir / 'ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    # Load gene list
    with open(data_dir / 'gene_list.txt', 'r') as f:
        gene_list = [line.strip() for line in f]
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: Controls={np.sum(y_train==0)}, Disease={np.sum(y_train==1)}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Disease genes: {len(ground_truth['disease_gene_names'])}")
    
    return X_train, y_train, X_aux_train, X_test, y_test, X_aux_test, ground_truth, gene_list


def train_on_synthetic(X_train, y_train, X_aux_train, n_factors=20, max_iter=200):
    """Train VI model on synthetic data."""
    print(f"\n{'='*70}")
    print("TRAINING VI MODEL")
    print(f"{'='*70}")
    
    model = VI(
        n_factors=n_factors,
        alpha_theta=0.5,
        alpha_beta=2.0,
        alpha_xi=2.0,
        lambda_xi=2.0,
        sigma_v=2.0,
        sigma_gamma=1.0
    )
    
    model.fit(
        X=X_train,
        y=y_train,
        X_aux=X_aux_train,
        max_iter=max_iter,
        tol=10.0,
        rel_tol=2e-4,
        elbo_freq=10,
        min_iter=50,
        patience=5,
        verbose=True,
        theta_damping=0.8,
        beta_damping=0.8,
        v_damping=0.7,
        gamma_damping=0.7,
        xi_damping=0.9,
        eta_damping=0.9
    )
    
    print(f"\nTraining complete!")
    print(f"  Final ELBO: {model.elbo_history_[-1][1]:.2f}")
    print(f"  Iterations: {len(model.elbo_history_)}")
    
    return model


def evaluate_gene_recovery(model, ground_truth, gene_list, top_k=50):
    """
    Check if model recovered disease genes.
    
    Metrics:
    - Precision@K: % of top K genes that are true disease genes
    - Recall@K: % of true disease genes in top K
    - Average rank of disease genes
    """
    print(f"\n{'='*70}")
    print("GENE RECOVERY ANALYSIS")
    print(f"{'='*70}")
    
    beta = model.E_beta  # Shape: (n_genes, n_factors)
    
    # True disease genes
    true_disease_indices = set(ground_truth['disease_gene_indices'])
    true_disease_names = set(ground_truth['disease_gene_names'])
    
    print(f"\nTrue disease genes: {len(true_disease_indices)}")
    
    # For each factor, get top genes
    results = {}
    
    for k in range(beta.shape[1]):
        beta_k = beta[:, k]
        
        # Rank genes by beta weight
        ranked_indices = np.argsort(beta_k)[::-1]  # Descending order
        ranked_genes = [gene_list[i] for i in ranked_indices]
        
        # Top K genes
        top_k_indices = set(ranked_indices[:top_k])
        top_k_genes = set(ranked_genes[:top_k])
        
        # Recovered disease genes in top K
        recovered_indices = top_k_indices & true_disease_indices
        recovered_genes = top_k_genes & true_disease_names
        
        # Metrics
        precision = len(recovered_indices) / top_k if top_k > 0 else 0
        recall = len(recovered_indices) / len(true_disease_indices) if len(true_disease_indices) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Average rank of disease genes
        disease_ranks = [np.where(ranked_indices == idx)[0][0] for idx in true_disease_indices]
        avg_rank = np.mean(disease_ranks)
        
        results[f'Factor_{k}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_rank': avg_rank,
            'recovered': list(recovered_genes),
            'n_recovered': len(recovered_genes)
        }
        
        print(f"\nFactor {k}:")
        print(f"  Precision@{top_k}: {precision:.3f} ({len(recovered_indices)}/{top_k})")
        print(f"  Recall@{top_k}:    {recall:.3f} ({len(recovered_indices)}/{len(true_disease_indices)})")
        print(f"  F1 score:         {f1:.3f}")
        print(f"  Avg rank:         {avg_rank:.1f}")
        if len(recovered_genes) > 0:
            print(f"  Recovered genes:  {list(recovered_genes)[:5]}...")
    
    # Find best factor (highest F1)
    best_factor = max(results.keys(), key=lambda k: results[k]['f1'])
    best_f1 = results[best_factor]['f1']
    
    print(f"\n{'='*70}")
    print(f"BEST FACTOR: {best_factor}")
    print(f"  F1 score: {best_f1:.3f}")
    print(f"  Precision: {results[best_factor]['precision']:.3f}")
    print(f"  Recall: {results[best_factor]['recall']:.3f}")
    print(f"{'='*70}")
    
    return results, best_factor


def evaluate_v_weights(model, ground_truth):
    """Check if v-weights are positive for disease-relevant factors."""
    print(f"\n{'='*70}")
    print("V-WEIGHT ANALYSIS")
    print(f"{'='*70}")
    
    v = model.mu_v  # Shape: (kappa, n_factors)
    
    print(f"\nV-weight statistics:")
    print(f"  Shape: {v.shape}")
    print(f"  Range: [{v.min():.4f}, {v.max():.4f}]")
    print(f"  Mean: {v.mean():.4f}")
    print(f"  Std: {v.std():.4f}")
    
    # Count positive vs negative
    n_positive = (v > 0).sum()
    n_negative = (v < 0).sum()
    
    print(f"\n  Positive v-weights: {n_positive} ({100*n_positive/v.size:.1f}%)")
    print(f"  Negative v-weights: {n_negative} ({100*n_negative/v.size:.1f}%)")
    
    # Per-factor v-weights
    print(f"\nV-weights per factor:")
    for k in range(v.shape[1]):
        v_k = v[:, k]
        print(f"  Factor {k}: mean={v_k.mean():7.4f}, std={v_k.std():7.4f}, "
              f"range=[{v_k.min():7.4f}, {v_k.max():7.4f}]")
    
    # Expected: factors with recovered disease genes should have positive v-weights
    return v


def evaluate_prediction_performance(model, X_test, y_test, X_aux_test):
    """Evaluate disease prediction performance."""
    print(f"\n{'='*70}")
    print("PREDICTION PERFORMANCE")
    print(f"{'='*70}")
    
    # Predict on test set
    y_pred_proba = model.predict_proba(X_test, X_aux_test)
    
    # AUC
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test.ravel(), y_pred_proba.ravel())
        print(f"\nTest AUC: {auc:.4f}")

        # Threshold at 0.5
        y_pred = (y_pred_proba.ravel() > 0.5).astype(int)
        accuracy = (y_pred == y_test.ravel()).mean()
        
        # Confusion matrix
        y_test_flat = y_test.ravel()
        tp = np.sum((y_pred == 1) & (y_test_flat == 1))
        tn = np.sum((y_pred == 0) & (y_test_flat == 0))
        fp = np.sum((y_pred == 1) & (y_test_flat == 0))
        fn = np.sum((y_pred == 0) & (y_test_flat == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nTest accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"\nConfusion matrix:")
        print(f"  TN={tn:4d}  FP={fp:4d}")
        print(f"  FN={fn:4d}  TP={tp:4d}")
        
        return auc, y_pred_proba
    else:
        print("Only one class in test set - cannot compute AUC")
        return None, y_pred_proba


def plot_results(model, ground_truth, gene_list, results, output_dir):
    """Create diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print("CREATING PLOTS")
    print(f"{'='*70}")
    
    # 1. Gene recovery heatmap
    beta = model.E_beta
    disease_gene_indices = ground_truth['disease_gene_indices']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Disease genes only
    beta_disease = beta[disease_gene_indices, :]
    sns.heatmap(beta_disease, cmap='YlOrRd', ax=axes[0],
                xticklabels=[f'F{i}' for i in range(beta.shape[1])],
                yticklabels=[gene_list[i] for i in disease_gene_indices],
                cbar_kws={'label': 'Beta weight'})
    axes[0].set_title('Disease Genes (Ground Truth)')
    axes[0].set_xlabel('Factor')
    
    # V-weights
    v = model.mu_v
    sns.heatmap(v.T, cmap='RdBu_r', center=0, ax=axes[1],
                xticklabels=[f'P{i}' for i in range(v.shape[0])],
                yticklabels=[f'F{i}' for i in range(v.shape[1])],
                cbar_kws={'label': 'V-weight'})
    axes[1].set_title('V-weights (Supervised Parameters)')
    axes[1].set_xlabel('Program')
    axes[1].set_ylabel('Factor')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_recovery.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: gene_recovery.png")
    plt.close()
    
    # 2. Recovery metrics by factor
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    factors = list(results.keys())
    precisions = [results[f]['precision'] for f in factors]
    recalls = [results[f]['recall'] for f in factors]
    f1s = [results[f]['f1'] for f in factors]
    
    axes[0].bar(range(len(factors)), precisions, color='steelblue')
    axes[0].set_xlabel('Factor')
    axes[0].set_ylabel('Precision@50')
    axes[0].set_title('Precision by Factor')
    axes[0].set_xticks(range(len(factors)))
    axes[0].set_xticklabels([f'F{i}' for i in range(len(factors))], rotation=45)
    
    axes[1].bar(range(len(factors)), recalls, color='coral')
    axes[1].set_xlabel('Factor')
    axes[1].set_ylabel('Recall@50')
    axes[1].set_title('Recall by Factor')
    axes[1].set_xticks(range(len(factors)))
    axes[1].set_xticklabels([f'F{i}' for i in range(len(factors))], rotation=45)
    
    axes[2].bar(range(len(factors)), f1s, color='forestgreen')
    axes[2].set_xlabel('Factor')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score by Factor')
    axes[2].set_xticks(range(len(factors)))
    axes[2].set_xticklabels([f'F{i}' for i in range(len(factors))], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recovery_metrics.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: recovery_metrics.png")
    plt.close()


def main(data_dir, output_dir=None, n_factors=20, max_iter=200, top_k=50):
    """Main validation workflow."""
    print("="*70)
    print("SYNTHETIC DISEASE VALIDATION")
    print("="*70)
    
    # Load data
    X_train, y_train, X_aux_train, X_test, y_test, X_aux_test, ground_truth, gene_list = \
        load_synthetic_data(data_dir)
    
    # Train model
    model = train_on_synthetic(X_train, y_train, X_aux_train, n_factors, max_iter)
    
    # Evaluate gene recovery
    results, best_factor = evaluate_gene_recovery(model, ground_truth, gene_list, top_k)
    
    # Evaluate v-weights
    v = evaluate_v_weights(model, ground_truth)
    
    # Evaluate prediction
    auc, y_pred_proba = evaluate_prediction_performance(model, X_test, y_test, X_aux_test)
    
    # Create plots
    if output_dir is None:
        output_dir = Path(data_dir) / 'validation_results'
    plot_results(model, ground_truth, gene_list, results, output_dir)
    
    # Save model
    model_path = Path(output_dir) / 'trained_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n  Saved model: {model_path}")
    
    # Save results
    summary = {
        'best_factor': best_factor,
        'best_f1': results[best_factor]['f1'],
        'best_precision': results[best_factor]['precision'],
        'best_recall': results[best_factor]['recall'],
        'test_auc': float(auc) if auc is not None else None,
        'v_weight_stats': {
            'mean': float(v.mean()),
            'std': float(v.std()),
            'min': float(v.min()),
            'max': float(v.max()),
            'n_positive': int((v > 0).sum()),
            'n_negative': int((v < 0).sum())
        },
        'factor_results': {k: {
            'precision': float(v['precision']),
            'recall': float(v['recall']),
            'f1': float(v['f1']),
            'avg_rank': float(v['avg_rank']),
            'n_recovered': int(v['n_recovered'])
        } for k, v in results.items()}
    }
    
    with open(Path(output_dir) / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Best F1 score: {results[best_factor]['f1']:.3f} ({best_factor})")
    print(f"✓ Test AUC: {auc:.3f}" if auc else "✗ Test AUC: N/A")
    print(f"✓ V-weight variance: {v.std():.4f}")
    
    if results[best_factor]['f1'] > 0.5:
        print(f"\n✓ SUCCESS: Model recovered disease genes!")
    else:
        print(f"\n⚠ WARNING: Low recovery - may need tuning")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate synthetic disease recovery')
    parser.add_argument('data_dir', type=str,
                       help='Synthetic disease data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--n_factors', type=int, default=20,
                       help='Number of factors')
    parser.add_argument('--max_iter', type=int, default=200,
                       help='Max iterations')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top K genes to check for recovery')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir, args.n_factors, args.max_iter, args.top_k)