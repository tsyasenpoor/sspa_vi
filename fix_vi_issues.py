"""
Diagnostic script to identify and fix VI training issues.

Issues identified:
1. ELBO decreases dramatically (iteration 11: -63M drop)
2. Predictions heavily skewed toward class 1 (mean 0.72 val, 0.65 test)
3. E[v] makes huge jumps (0.18 -> 1.6 in one iteration)

Root causes:
- v updates are too aggressive despite damping
- Prior on v (sigma_v=1.0) is too weak given the data scale
- Initialization doesn't account for data imbalance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def diagnose_training_issues(results_dir="."):
    """Analyze saved model and data to diagnose issues."""

    print("="*80)
    print("DIAGNOSTIC ANALYSIS")
    print("="*80)

    # Load model
    try:
        with open(f'{results_dir}/sspa_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Loaded trained model")
    except:
        print("✗ Could not load model")
        return

    # Check v statistics
    print("\n" + "="*80)
    print("V (Classification Weights) Analysis")
    print("="*80)
    E_v = model.E_v  # shape (kappa, d)
    print(f"Shape: {E_v.shape}")
    print(f"Mean: {E_v.mean():.4f}")
    print(f"Std: {E_v.std():.4f}")
    print(f"Min: {E_v.min():.4f}")
    print(f"Max: {E_v.max():.4f}")
    print(f"\nTop 5 most positive weights: {np.sort(E_v.flatten())[-5:]}")
    print(f"Top 5 most negative weights: {np.sort(E_v.flatten())[:5]}")

    # Check spike-and-slab activity
    rho_v = model.rho_v
    active_v = (rho_v > 0.5).sum()
    print(f"\nActive v components (rho > 0.5): {active_v}/{rho_v.size}")

    # Analyze predictions
    print("\n" + "="*80)
    print("Prediction Analysis")
    print("="*80)

    try:
        # Load theta matrices
        theta_train = pd.read_csv(f'{results_dir}/sspa_theta_train.csv.gz', index_col=0)
        theta_val = pd.read_csv(f'{results_dir}/sspa_theta_val.csv.gz', index_col=0)
        theta_test = pd.read_csv(f'{results_dir}/sspa_theta_test.csv.gz', index_col=0)

        print(f"Theta train mean: {theta_train.values.mean():.4f}")
        print(f"Theta val mean: {theta_val.values.mean():.4f}")
        print(f"Theta test mean: {theta_test.values.mean():.4f}")

        # Compute logits manually
        logits_train = theta_train.values @ E_v.T
        logits_val = theta_val.values @ E_v.T
        logits_test = theta_test.values @ E_v.T

        print(f"\nLogits train - mean: {logits_train.mean():.4f}, std: {logits_train.std():.4f}")
        print(f"Logits val - mean: {logits_val.mean():.4f}, std: {logits_val.std():.4f}")
        print(f"Logits test - mean: {logits_test.mean():.4f}, std: {logits_test.std():.4f}")

        # Break down which v components contribute most
        contribution = theta_val.values.mean(axis=0) * E_v.flatten()
        top_contributors = np.argsort(np.abs(contribution))[-10:]
        print(f"\nTop 10 gene programs contributing to predictions:")
        for idx in top_contributors[::-1]:
            print(f"  GP{idx}: theta_mean={theta_val.values.mean(axis=0)[idx]:.4f}, "
                  f"v={E_v.flatten()[idx]:.4f}, contribution={contribution[idx]:.4f}")

    except Exception as e:
        print(f"Could not analyze predictions: {e}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if E_v.max() > 2.0:
        print("⚠ V weights are very large (max > 2.0)")
        print("  → Reduce sigma_v (try 0.5 instead of 1.0)")
        print("  → Add stronger regularization via spike-and-slab")

    if active_v < E_v.size * 0.3:
        print("⚠ Very few v components are active")
        print("  → Model is using only a few gene programs for classification")
        print("  → This might be correct, but check if it makes biological sense")

    if logits_val.mean() > 0.5:
        print(f"⚠ Mean logit is positive ({logits_val.mean():.2f})")
        print("  → This biases predictions toward class 1")
        print("  → Check if training data is balanced")
        print("  → Consider adding intercept term or class weights")


def create_fixed_model_config():
    """Create improved model configuration to fix ELBO issues."""

    print("\n" + "="*80)
    print("RECOMMENDED MODEL CONFIGURATION")
    print("="*80)

    config = {
        # Stronger priors to prevent v from exploding
        'sigma_v': 0.3,  # Reduced from 1.0 - much stronger regularization
        'sigma_gamma': 0.5,

        # More selective spike-and-slab for v
        'pi_v': 0.3,  # Reduced from 0.5 - assume most programs irrelevant for classification
        'pi_beta': 0.05,

        # Tighter bounds on parameters
        'clip_v': 2.0,  # Clip v to [-2, 2] instead of [-3, 3]
        'clip_theta': 100.0,  # Prevent theta from exploding

        # More conservative damping
        'theta_damping': 0.3,  # Reduced from 0.5
        'beta_damping': 0.5,   # Reduced from 0.7
        'v_damping': 0.3,      # Reduced from 0.6 - CRITICAL
        'gamma_damping': 0.4,

        # Training parameters
        'max_iter': 300,
        'min_iter': 50,  # Don't stop too early
        'elbo_freq': 10,  # Check ELBO every 10 iterations

        # Numerical stability
        'min_variance': 1e-8,
        'max_precision': 1e8,
    }

    print("Key changes:")
    print(f"  - sigma_v: 1.0 → {config['sigma_v']} (stronger regularization)")
    print(f"  - v_damping: 0.6 → {config['v_damping']} (more conservative)")
    print(f"  - clip_v: 3.0 → {config['clip_v']} (tighter bounds)")
    print(f"  - pi_v: 0.5 → {config['pi_v']} (more selective)")

    return config


def check_data_balance(data_path="filtered_data_top3k.h5ad"):
    """Check if training data is balanced."""
    import scanpy as sc

    print("\n" + "="*80)
    print("DATA BALANCE CHECK")
    print("="*80)

    try:
        adata = sc.read_h5ad(data_path)

        # Check overall class balance
        if 'ap' in adata.obs.columns:
            class_counts = adata.obs['ap'].value_counts()
            print(f"\nOverall class distribution:")
            for cls, count in class_counts.items():
                pct = 100 * count / len(adata)
                print(f"  Class {cls}: {count} ({pct:.1f}%)")

            # Check if splits preserve balance
            if 'split' in adata.obs.columns:
                print(f"\nClass distribution by split:")
                for split in ['train', 'val', 'test']:
                    split_data = adata[adata.obs['split'] == split]
                    if len(split_data) > 0:
                        split_counts = split_data.obs['ap'].value_counts()
                        print(f"  {split}:")
                        for cls, count in split_counts.items():
                            pct = 100 * count / len(split_data)
                            print(f"    Class {cls}: {count} ({pct:.1f}%)")

            # Check imbalance
            max_count = class_counts.max()
            min_count = class_counts.min()
            imbalance_ratio = max_count / min_count

            if imbalance_ratio > 1.5:
                print(f"\n⚠ Data is imbalanced (ratio: {imbalance_ratio:.2f}:1)")
                print(f"  → This explains why predictions are biased toward majority class")
                print(f"  → Solutions:")
                print(f"     1. Use class weights in the model")
                print(f"     2. Oversample minority class")
                print(f"     3. Add intercept term to absorb base rate")
            else:
                print(f"\n✓ Data is reasonably balanced (ratio: {imbalance_ratio:.2f}:1)")

    except Exception as e:
        print(f"Could not load data: {e}")


if __name__ == "__main__":
    # Run diagnostics
    diagnose_training_issues()

    # Check data balance
    check_data_balance()

    # Print recommended config
    config = create_fixed_model_config()

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review the diagnostic output above")
    print("2. Check if data imbalance is the root cause")
    print("3. Re-run training with recommended configuration:")
    print("   - Update vi.py initialization with new sigma_v, pi_v values")
    print("   - Update fit() call with new damping parameters")
    print("   - Add stricter clipping in _update_v()")
    print("4. Monitor ELBO - it should increase monotonically")
    print("5. Check if predictions become more balanced")
