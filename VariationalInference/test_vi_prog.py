"""
Progressive Testing Script for VI Implementation
Tests on increasingly larger subsets to catch bugs early before full run.
"""

import numpy as np
import pickle
import time
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import sys

# Adjust import based on your structure
sys.path.append('/labs/Aguiar/SSPA_BRAY/BRay/VariationalInference')
from vi import VI
from load_data_vi import load_control_data


def create_test_subset(X, y, X_aux, n_samples=300, n_genes=1000, random_state=42):
    """
    Create smaller dataset preserving characteristics.
    
    Strategy:
    - Sample cells randomly
    - Keep high-variance genes (most informative)
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Sample cells (rows)
    n_total = X.shape[0]
    if n_samples < n_total:
        idx_samples = rng.choice(n_total, n_samples, replace=False)
        X_sub = X[idx_samples]
        y_sub = y[idx_samples] if y is not None else None
        X_aux_sub = X_aux[idx_samples] if X_aux is not None else None
    else:
        X_sub, y_sub, X_aux_sub = X, y, X_aux
    
    # 2. Sample genes (columns) - keep high-variance genes
    if n_genes < X.shape[1]:
        gene_var = X_sub.var(axis=0)
        if issparse(gene_var):
            gene_var = np.asarray(gene_var).ravel()
        top_genes = np.argsort(gene_var)[-n_genes:]  # Keep most variable
        X_sub = X_sub[:, top_genes]
    else:
        top_genes = None
    
    print(f"  Subset: {X_sub.shape[0]} samples × {X_sub.shape[1]} genes")
    print(f"  Sparsity: {(X_sub == 0).mean():.2%}")
    if y_sub is not None:
        print(f"  Disease labels: {np.bincount(y_sub.astype(int).ravel())}")
    
    return X_sub, y_sub, X_aux_sub, top_genes


def quick_diagnostic(model, stage_name="model", verbose=True, threshold=0.5):
    """
    Fast diagnostic checks after training.
    
    Returns:
    --------
    passed : bool
        True if all checks pass
    issues : list
        List of detected issues
    """
    issues = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"=== {stage_name} Diagnostics ===")
        print(f"{'='*60}")
    
    # 1. V-WEIGHT BUG CHECK (CRITICAL)
    v_std = model.mu_v.std()
    v_min, v_max = model.mu_v.min(), model.mu_v.max()
    v_unique = len(np.unique(model.mu_v.ravel()))
    
    if verbose:
        print(f"\n[V-WEIGHT CHECK]")
        print(f"  Std:    {v_std:.6f} (want > 0.01)")
        print(f"  Range:  [{v_min:.4f}, {v_max:.4f}]")
        print(f"  Unique: {v_unique}/{model.mu_v.size}")
    
    if v_std < 0.01:
        issues.append("V-WEIGHT BUG: All v-weights identical or near-zero variance")
    if v_unique < model.kappa:
        issues.append(f"V-WEIGHT WARNING: Only {v_unique} unique values for {model.kappa} programs")
    
    # 2. SPARSITY CHECK (for spike-and-slab)
    if hasattr(model, 'rho_beta') and hasattr(model, 'rho_v'):
        if verbose:
            print(f"\n[SPIKE-AND-SLAB SPARSITY]")
        
        # Beta sparsity
        beta_active = model.rho_beta > threshold
        beta_sparsity = 1.0 - beta_active.mean()
        
        # V sparsity
        v_active = model.rho_v > threshold
        v_sparsity = 1.0 - v_active.mean()
        
        if verbose:
            print(f"  Beta (genes):")
            print(f"    Active: {beta_active.sum()}/{model.rho_beta.size} ({(1-beta_sparsity)*100:.1f}%)")
            print(f"    Sparse: {beta_sparsity*100:.1f}%")
            print(f"    Active per factor: {beta_active.sum(axis=0)}")
            print(f"  V (classification weights):")
            print(f"    Active: {v_active.sum()}/{model.rho_v.size} ({(1-v_sparsity)*100:.1f}%)")
            print(f"    Sparse: {v_sparsity*100:.1f}%")
            print(f"    Active per factor: {v_active.sum(axis=0)}")
        
        # Check if sparsity is too extreme
        if beta_sparsity > 0.95:
            issues.append(f"WARNING: Beta extremely sparse ({beta_sparsity*100:.1f}%)")
        if v_sparsity > 0.95:
            issues.append(f"WARNING: V extremely sparse ({v_sparsity*100:.1f}%)")
    
    # 3. CONVERGENCE CHECK
    if verbose:
        print(f"\n[CONVERGENCE CHECK]")
    
    if len(model.elbo_history_) > 2:
        elbo_vals = np.array([e[1] for e in model.elbo_history_])
        iterations = np.array([e[0] for e in model.elbo_history_])
        
        # Check final changes
        final_changes = np.abs(np.diff(elbo_vals[-3:]))
        mean_final_change = final_changes.mean()
        
        # Check if still decreasing significantly
        rel_change = mean_final_change / np.abs(elbo_vals[-1]) if elbo_vals[-1] != 0 else np.inf
        
        if verbose:
            print(f"  Final ELBO: {elbo_vals[-1]:.2e}")
            print(f"  Mean change (last 3): {mean_final_change:.2e}")
            print(f"  Relative change: {rel_change:.6f}")
            print(f"  Iterations: {iterations[-1]}")
        
        if rel_change > 0.01:  # Still changing by >1%
            issues.append(f"CONVERGENCE WARNING: Still changing by {rel_change:.2%}")
    
    # 3. PARAMETER SANITY CHECKS
    if verbose:
        print(f"\n[PARAMETER RANGES]")
    
    theta_min, theta_max = model.E_theta.min(), model.E_theta.max()
    beta_min, beta_max = model.E_beta.min(), model.E_beta.max()
    
    if verbose:
        print(f"  E[theta]: [{theta_min:.4f}, {theta_max:.4f}]")
        print(f"  E[beta]:  [{beta_min:.4f}, {beta_max:.4f}]")
        print(f"  E[v]:     [{model.E_v.min():.4f}, {model.E_v.max():.4f}]")
        
        # Check if E_gamma exists and is not empty
        if hasattr(model, 'E_gamma') and model.E_gamma.size > 0:
            print(f"  E[gamma]: [{model.E_gamma.min():.4f}, {model.E_gamma.max():.4f}]")
        else:
            print(f"  E[gamma]: [empty]")
    
    # Check for extreme values
    if theta_max > 1e6 or beta_max > 1e6:
        issues.append("OVERFLOW: Theta or beta values extremely large")
    if theta_min < 1e-10 or beta_min < 1e-10:
        issues.append("UNDERFLOW: Theta or beta values extremely small")
    
    # 4. NaN/Inf CHECK
    if verbose:
        print(f"\n[NUMERICAL STABILITY]")
    
    has_nan_theta = np.isnan(model.E_theta).any()
    has_nan_beta = np.isnan(model.E_beta).any()
    has_nan_v = np.isnan(model.mu_v).any()
    
    # Check E_gamma only if it exists and is not empty
    has_nan_gamma = False
    if hasattr(model, 'E_gamma') and model.E_gamma.size > 0:
        has_nan_gamma = np.isnan(model.E_gamma).any()
    
    # Check mu_gamma only if it exists and is not empty
    has_nan_mu_gamma = False
    if hasattr(model, 'mu_gamma') and model.mu_gamma.size > 0:
        has_nan_mu_gamma = np.isnan(model.mu_gamma).any()
    
    has_inf_theta = np.isinf(model.E_theta).any()
    has_inf_beta = np.isinf(model.E_beta).any()
    
    if verbose:
        print(f"  NaN in theta:  {has_nan_theta}")
        print(f"  NaN in beta:   {has_nan_beta}")
        print(f"  NaN in v:      {has_nan_v}")
        if hasattr(model, 'E_gamma') and model.E_gamma.size > 0:
            print(f"  NaN in E_gamma: {has_nan_gamma}")
        else:
            print(f"  NaN in E_gamma: [empty]")
        if hasattr(model, 'mu_gamma') and model.mu_gamma.size > 0:
            print(f"  NaN in mu_gamma: {has_nan_mu_gamma}")
        else:
            print(f"  NaN in mu_gamma: [empty]")
        print(f"  Inf in theta:  {has_inf_theta}")
        print(f"  Inf in beta:   {has_inf_beta}")
    
    if has_nan_theta or has_nan_beta or has_nan_v or has_nan_gamma or has_nan_mu_gamma:
        issues.append("CRITICAL: NaN values detected")
    if has_inf_theta or has_inf_beta:
        issues.append("CRITICAL: Inf values detected")
    
    # 5. GAMMA SHAPE/RATE SANITY
    if verbose:
        print(f"\n[GAMMA PARAMETERS]")
        print(f"  a_theta: [{model.a_theta.min():.2f}, {model.a_theta.max():.2f}]")
        print(f"  b_theta: [{model.b_theta.min():.2e}, {model.b_theta.max():.2e}]")
        print(f"  a_beta:  [{model.a_beta.min():.2f}, {model.a_beta.max():.2f}]")
        print(f"  b_beta:  [{model.b_beta.min():.2e}, {model.b_beta.max():.2e}]")
    
    # FINAL VERDICT
    passed = len(issues) == 0
    
    if verbose:
        print(f"\n{'='*60}")
        if passed:
            print("✓ ALL CHECKS PASSED")
        else:
            print("✗ ISSUES DETECTED:")
            for issue in issues:
                print(f"  - {issue}")
        print(f"{'='*60}\n")
    
    return passed, issues



def plot_elbo_trajectory(model, stage_name, output_dir):
    """Plot ELBO trajectory for inspection."""
    if not hasattr(model, 'elbo_history_') or len(model.elbo_history_) < 2:
        print(f"  Warning: No ELBO history to plot")
        return
    
    elbo_vals = np.array([e[1] for e in model.elbo_history_])
    iterations = np.array([e[0] for e in model.elbo_history_])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # ELBO trajectory
    ax1.plot(iterations, elbo_vals, 'o-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('ELBO')
    ax1.set_title(f'{stage_name} - ELBO Trajectory')
    ax1.grid(True, alpha=0.3)
    
    # ELBO change rate
    if len(elbo_vals) > 1:
        changes = np.diff(elbo_vals)
        ax2.plot(iterations[1:], changes, 'o-')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('ELBO Change')
        ax2.set_title(f'{stage_name} - ELBO Improvement Rate')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'{stage_name.lower().replace(" ", "_")}_elbo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_path}")


def run_test_stage(
    X, y, X_aux,
    stage_name,
    n_samples,
    n_genes,
    n_factors,
    max_iter,
    output_dir,
    elbo_freq=10,
    verbose=True
):
    """
    Run a single test stage.
    
    Returns:
    --------
    passed : bool
        Whether stage passed all checks
    model : VI
        Trained model (if successful)
    time_taken : float
        Training time in seconds
    """
    print(f"\n{'#'*60}")
    print(f"### {stage_name}")
    print(f"{'#'*60}\n")
    
    # Create subset
    print(f"Creating subset...")
    X_sub, y_sub, X_aux_sub, gene_idx = create_test_subset(
        X, y, X_aux, n_samples, n_genes
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    print(f"  Factors: {n_factors}")
    print(f"  Max iterations: {max_iter}")
    
    model = VI(
        n_factors=n_factors,
        sigma_v=0.1,
        sigma_gamma=0.1,
        random_state=42
    )
    
    # Train
    print(f"\nTraining...")
    start_time = time.time()
    
    try:
        model.fit(
            X_sub, y_sub, X_aux_sub,
            max_iter=max_iter,
            elbo_freq=elbo_freq,
            verbose=verbose,
            debug=False
        )
        time_taken = time.time() - start_time
        
        print(f"\n✓ Training completed in {time_taken:.1f}s ({time_taken/60:.1f} min)")
        
    except Exception as e:
        print(f"\n✗ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, 0
    
    # Diagnostic checks
    passed, issues = quick_diagnostic(model, stage_name, verbose=True)
    
    # Plot ELBO
    plot_elbo_trajectory(model, stage_name, output_dir)
    
    # Save results summary with sparsity info
    results = {
        'stage_name': stage_name,
        'n_samples': n_samples,
        'n_genes': n_genes,
        'n_factors': n_factors,
        'max_iter': max_iter,
        'time_taken': time_taken,
        'passed': passed,
        'issues': issues,
        'elbo_history': model.elbo_history_,
        'v_weight_stats': {
            'mean': float(model.mu_v.mean()),
            'std': float(model.mu_v.std()),
            'min': float(model.mu_v.min()),
            'max': float(model.mu_v.max()),
        },
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
    
    result_path = output_dir / f'{stage_name.lower().replace(" ", "_")}_results.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Saved results: {result_path}")
    
    # Save full model for inspection
    model_path = output_dir / f'{stage_name.lower().replace(" ", "_")}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved model: {model_path}")
    
    return passed, model, time_taken


def main():
    """Main progressive testing workflow."""
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Paths
    data_dir = Path('/labs/Aguiar/SSPA_BRAY/BRay/ctrl_sspa_test')
    output_dir = Path('/labs/Aguiar/SSPA_BRAY/BRay/progressive_tests')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load full data
    print("="*60)
    print("LOADING FULL DATASET")
    print("="*60)
    
    # Use custom data loader for user's format
    data = load_control_data(str(data_dir))
    X_train, y_train, X_aux_train = data[:3]
    X_val, y_val, X_aux_val = data[3:6]
    X_test, y_test, X_aux_test = data[6:9]
    gene_names = data[9]
    
    print(f"\nFull dataset: {X_train.shape[0]} samples × {X_train.shape[1]} genes")
    print(f"Disease labels (control-only): {np.unique(y_train)}")
    print(f"Sparsity: {(X_train == 0).mean():.2%}")
    
    # =========================================================================
    # TESTING STAGES
    # =========================================================================
    
    stages = [
        {
            'name': 'Stage 1: Tiny',
            'n_samples': 100,
            'n_genes': 500,
            'n_factors': 5,
            'max_iter': 200,
            'elbo_freq': 5,
            'description': 'Quick smoke test (~5 min)'
        },
        {
            'name': 'Stage 2: Small',
            'n_samples': 300,
            'n_genes': 1000,
            'n_factors': 10,
            'max_iter': 300,
            'elbo_freq': 10,
            'description': 'Verify convergence (~15 min)'
        },
        {
            'name': 'Stage 3: Medium',
            'n_samples': 600,
            'n_genes': 2000,
            'n_factors': 15,
            'max_iter': 500,
            'elbo_freq': 10,
            'description': 'Verify quality (~30 min)'
        },
        {
            'name': 'Stage 4: Full',
            'n_samples': X_train.shape[0],
            'n_genes': X_train.shape[1],
            'n_factors': 20,
            'max_iter': 200,
            'elbo_freq': 10,
            'description': 'Full dataset (~60 min)'
        }
    ]
    
    # =========================================================================
    # RUN PROGRESSIVE TESTS
    # =========================================================================
    
    all_passed = True
    results_summary = []
    
    for stage in stages:
        stage_result = run_test_stage(
            X_train, y_train, X_aux_train,
            stage_name=stage['name'],
            n_samples=stage['n_samples'],
            n_genes=stage['n_genes'],
            n_factors=stage['n_factors'],
            max_iter=stage['max_iter'],
            output_dir=output_dir,
            elbo_freq=stage['elbo_freq'],
            verbose=True
        )
        
        passed, model, time_taken = stage_result
        
        results_summary.append({
            'stage': stage['name'],
            'passed': passed,
            'time': time_taken,
            'description': stage['description']
        })
        
        # STOP IF STAGE FAILS
        if not passed:
            print(f"\n{'!'*60}")
            print(f"!!! {stage['name']} FAILED - STOPPING PROGRESSION !!!")
            print(f"{'!'*60}\n")
            all_passed = False
            break
        
        print(f"\n✓ {stage['name']} PASSED - Proceeding to next stage\n")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*60)
    print("PROGRESSIVE TESTING SUMMARY")
    print("="*60)
    
    for result in results_summary:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"{status} | {result['stage']:20s} | {result['time']:6.1f}s | {result['description']}")
    
    print("="*60)
    
    if all_passed and len(results_summary) == len(stages):
        print("\n ALL STAGES PASSED! Safe to run full dataset.")
    else:
        print("\n  Testing incomplete. Fix issues before proceeding.")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()