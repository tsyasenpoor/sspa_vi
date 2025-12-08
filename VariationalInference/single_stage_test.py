"""
Single Stage Test Runner

Run a single custom test without the full progressive workflow.
Useful for quick experiments and parameter tuning.

Usage:
    python test_single_stage.py --n_samples 200 --n_genes 800 --n_factors 8 --max_iter 100
"""

import argparse
import numpy as np
import pickle
import time
from pathlib import Path
import sys

sys.path.append('/labs/Aguiar/SSPA_BRAY/BRay/VariationalInference')
from vi import VI
from load_data_vi import load_control_data


def main():
    parser = argparse.ArgumentParser(description='Run single VI test stage')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                       default='/labs/Aguiar/SSPA_BRAY/BRay/ctrl_sspa_test',
                       help='Directory with training data')
    parser.add_argument('--output_dir', type=str,
                       default='/labs/Aguiar/SSPA_BRAY/BRay/single_stage_test',
                       help='Output directory')
    
    # Subset parameters
    parser.add_argument('--n_samples', type=int, default=300,
                       help='Number of samples to use')
    parser.add_argument('--n_genes', type=int, default=1000,
                       help='Number of genes to use (selects high-variance)')
    
    # Model parameters
    parser.add_argument('--n_factors', type=int, default=10,
                       help='Number of latent factors')
    parser.add_argument('--sigma_v', type=float, default=0.1,
                       help='Prior std for v parameters')
    parser.add_argument('--sigma_gamma', type=float, default=0.1,
                       help='Prior std for gamma parameters')
    
    # Training parameters
    parser.add_argument('--max_iter', type=int, default=100,
                       help='Maximum iterations')
    parser.add_argument('--elbo_freq', type=int, default=10,
                       help='Compute ELBO every N iterations')
    parser.add_argument('--theta_damping', type=float, default=0.5,
                       help='Damping factor for theta')
    parser.add_argument('--beta_damping', type=float, default=0.7,
                       help='Damping factor for beta')
    parser.add_argument('--v_damping', type=float, default=0.6,
                       help='Damping factor for v')
    parser.add_argument('--gamma_damping', type=float, default=0.6,
                       help='Damping factor for gamma')
    
    # Other
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (extra checks)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("SINGLE STAGE VI TEST")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.n_samples}")
    print(f"  Genes: {args.n_genes}")
    print(f"  Factors: {args.n_factors}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Sigma_v: {args.sigma_v}")
    print(f"  Sigma_gamma: {args.sigma_gamma}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    
    # Use custom data loader for user's format
    data = load_control_data(args.data_dir)
    X_full, y_full, X_aux_full = data[:3]
    
    print(f"Full dataset: {X_full.shape[0]} samples × {X_full.shape[1]} genes")
    
    # Create subset
    print(f"\nCreating subset...")
    rng = np.random.RandomState(args.random_state)
    
    # Sample cells
    if args.n_samples < X_full.shape[0]:
        idx_samples = rng.choice(X_full.shape[0], args.n_samples, replace=False)
        X = X_full[idx_samples]
        y = y_full[idx_samples]
        X_aux = X_aux_full[idx_samples]
    else:
        X, y, X_aux = X_full, y_full, X_aux_full
    
    # Sample genes (high-variance)
    if args.n_genes < X_full.shape[1]:
        gene_var = X.var(axis=0)
        top_genes = np.argsort(gene_var)[-args.n_genes:]
        X = X[:, top_genes]
        print(f"Selected {args.n_genes} most variable genes")
    
    print(f"Subset: {X.shape[0]} samples × {X.shape[1]} genes")
    print(f"Sparsity: {(X == 0).mean():.2%}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = VI(
        n_factors=args.n_factors,
        sigma_v=args.sigma_v,
        sigma_gamma=args.sigma_gamma,
        random_state=args.random_state
    )
    
    # Train
    print(f"\nTraining...")
    start_time = time.time()
    
    try:
        model.fit(
            X, y, X_aux,
            max_iter=args.max_iter,
            elbo_freq=args.elbo_freq,
            verbose=args.verbose,
            debug=args.debug,
            theta_damping=args.theta_damping,
            beta_damping=args.beta_damping,
            v_damping=args.v_damping,
            gamma_damping=args.gamma_damping
        )
        
        training_time = time.time() - start_time
        print(f"\n✓ Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")
        
    except Exception as e:
        print(f"\n✗ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Quick diagnostics
    print(f"\n{'='*60}")
    print("QUICK DIAGNOSTICS")
    print(f"{'='*60}")
    
    # V-weight check
    v_std = model.mu_v.std()
    v_min = model.mu_v.min()
    v_max = model.mu_v.max()
    v_unique = len(np.unique(model.mu_v.ravel()))
    
    print(f"\n[V-WEIGHTS]")
    print(f"  Std:    {v_std:.6f}")
    print(f"  Range:  [{v_min:.4f}, {v_max:.4f}]")
    print(f"  Unique: {v_unique}/{model.mu_v.size}")
    
    if v_std < 0.001:
        print(f"  ✗ CRITICAL: Near-zero variance!")
    elif v_std < 0.01:
        print(f"  ⚠ WARNING: Low variance")
    else:
        print(f"  ✓ Good variance")
    
    # Sparsity check (for spike-and-slab)
    if hasattr(model, 'rho_beta') and hasattr(model, 'rho_v'):
        threshold = 0.5
        beta_active = model.rho_beta > threshold
        v_active = model.rho_v > threshold
        
        print(f"\n[SPIKE-AND-SLAB SPARSITY]")
        print(f"  Beta:")
        print(f"    Active: {beta_active.sum()}/{model.rho_beta.size} ({(1-beta_active.mean())*100:.1f}% sparse)")
        print(f"    Active per factor: {beta_active.sum(axis=0)}")
        print(f"  V:")
        print(f"    Active: {v_active.sum()}/{model.rho_v.size} ({(1-v_active.mean())*100:.1f}% sparse)")
        print(f"    Active per factor: {v_active.sum(axis=0)}")
    
    # Convergence check
    if len(model.elbo_history_) > 2:
        elbo_vals = np.array([e[1] for e in model.elbo_history_])
        final_changes = np.abs(np.diff(elbo_vals[-3:]))
        mean_change = final_changes.mean()
        rel_change = mean_change / np.abs(elbo_vals[-1])
        
        print(f"\n[CONVERGENCE]")
        print(f"  Final ELBO: {elbo_vals[-1]:.2e}")
        print(f"  Mean change: {mean_change:.2e}")
        print(f"  Rel change: {rel_change:.6f}")
        
        if rel_change > 0.01:
            print(f"  ⚠ Still changing by {rel_change:.2%}")
        else:
            print(f"  ✓ Well converged")
    
    # Parameter ranges
    print(f"\n[PARAMETERS]")
    print(f"  E[theta]: [{model.E_theta.min():.4f}, {model.E_theta.max():.4f}]")
    print(f"  E[beta]:  [{model.E_beta.min():.4f}, {model.E_beta.max():.4f}]")
    print(f"  E[v]:     [{model.E_v.min():.4f}, {model.E_v.max():.4f}]")
    print(f"  E[gamma]: [{model.E_gamma.min():.4f}, {model.E_gamma.max():.4f}]")
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    results = {
        'args': vars(args),
        'training_time': training_time,
        'shape': X.shape,
        'v_weight_stats': {
            'mean': float(model.mu_v.mean()),
            'std': float(model.mu_v.std()),
            'min': float(model.mu_v.min()),
            'max': float(model.mu_v.max()),
        },
        'elbo_history': model.elbo_history_,
        'final_elbo': model.elbo_history_[-1][1] if model.elbo_history_ else None,
    }
    
    # Add sparsity info if spike-and-slab is used
    if hasattr(model, 'rho_beta') and hasattr(model, 'rho_v'):
        threshold = 0.5
        beta_active = model.rho_beta > threshold
        v_active = model.rho_v > threshold
        
        results['sparsity_info'] = {
            'beta_active_count': int(beta_active.sum()),
            'beta_total': int(model.rho_beta.size),
            'beta_sparsity': float(1.0 - beta_active.mean()),
            'beta_active_per_factor': beta_active.sum(axis=0).tolist(),
            'v_active_count': int(v_active.sum()),
            'v_total': int(model.rho_v.size),
            'v_sparsity': float(1.0 - v_active.mean()),
            'v_active_per_factor': v_active.sum(axis=0).tolist(),
        }
    
    # Save results
    result_path = output_dir / 'results.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Saved: {result_path}")
    
    # Save model
    model_path = output_dir / 'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved: {model_path}")
    
    # Save config
    config_path = output_dir / 'config.txt'
    with open(config_path, 'w') as f:
        f.write("Configuration:\n")
        f.write("-" * 40 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key:20s}: {value}\n")
        f.write("\n")
        f.write("Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Training time':20s}: {training_time:.1f}s\n")
        f.write(f"{'Final ELBO':20s}: {results['final_elbo']:.2e}\n")
        f.write(f"{'V-weight std':20s}: {v_std:.6f}\n")
    print(f"  Saved: {config_path}")
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())