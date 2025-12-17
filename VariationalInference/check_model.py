"""
Quick Diagnostic Check for Already-Trained Models

Use this to check saved models without retraining.
"""

import pickle
import numpy as np
from pathlib import Path


def check_saved_model(model_path, gene_programs_path=None):
    """
    Quick diagnostic on saved model files.
    
    Usage:
        python check_model.py /path/to/model.pkl
        
    Or in Python:
        check_saved_model('/path/to/model.pkl')
    """
    
    print(f"\n{'='*60}")
    print(f"CHECKING MODEL: {Path(model_path).name}")
    print(f"{'='*60}\n")
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # If gene_programs dataframe provided separately
    if gene_programs_path:
        try:
            with open(gene_programs_path, 'rb') as f:
                gp = pickle.load(f)
            if hasattr(gp, 'v_weight'):
                v_weights = gp['v_weight'].values
            else:
                v_weights = model.mu_v if hasattr(model, 'mu_v') else None
        except:
            v_weights = model.mu_v if hasattr(model, 'mu_v') else None
    else:
        v_weights = model.mu_v if hasattr(model, 'mu_v') else None
    
    issues = []
    
    # =====================================================================
    # CHECK 1: V-WEIGHT BUG
    # =====================================================================
    print("\n[V-WEIGHT CHECK]")
    
    if v_weights is not None:
        if isinstance(v_weights, np.ndarray):
            v_std = v_weights.std()
            v_min = v_weights.min()
            v_max = v_weights.max()
            v_mean = v_weights.mean()
            v_unique = len(np.unique(v_weights.ravel()))
            v_total = v_weights.size
            
            print(f"  Mean:   {v_mean:.6f}")
            print(f"  Std:    {v_std:.6f}")
            print(f"  Range:  [{v_min:.4f}, {v_max:.4f}]")
            print(f"  Unique: {v_unique}/{v_total}")
            
            # BUG DETECTION
            if v_std < 0.001:
                print(f"  ✗ CRITICAL: V-weights have near-zero variance!")
                print(f"    All programs learning identical patterns.")
                issues.append("V-WEIGHT BUG: Zero variance")
            elif v_std < 0.01:
                print(f"  ⚠ WARNING: V-weights have very low variance ({v_std:.6f})")
                issues.append("V-WEIGHT WARNING: Low variance")
            else:
                print(f"  ✓ V-weights show proper variation")
            
            if v_unique < v_weights.shape[0]:
                print(f"  ⚠ WARNING: Only {v_unique} unique values for {v_weights.shape[0]} programs")
                issues.append("V-WEIGHT WARNING: Few unique values")
        else:
            print(f"  ✗ v_weights wrong type: {type(v_weights)}")
            issues.append("V-WEIGHT ERROR: Wrong type")
    else:
        print(f"  ✗ No v_weights found in model")
        issues.append("V-WEIGHT ERROR: Not found")
    
    # =====================================================================
    # CHECK 2: CONVERGENCE
    # =====================================================================
    print("\n[CONVERGENCE CHECK]")
    
    if hasattr(model, 'elbo_history_'):
        elbo_history = model.elbo_history_
        if len(elbo_history) > 2:
            elbo_vals = np.array([e[1] for e in elbo_history])
            iterations = np.array([e[0] for e in elbo_history])
            
            print(f"  Iterations: {iterations[-1]}")
            print(f"  Initial ELBO: {elbo_vals[0]:.2e}")
            print(f"  Final ELBO: {elbo_vals[-1]:.2e}")
            print(f"  Total change: {elbo_vals[-1] - elbo_vals[0]:.2e}")
            
            # Check final convergence
            if len(elbo_vals) >= 3:
                final_changes = np.abs(np.diff(elbo_vals[-3:]))
                mean_change = final_changes.mean()
                rel_change = mean_change / np.abs(elbo_vals[-1]) if elbo_vals[-1] != 0 else np.inf
                
                print(f"  Mean change (last 3): {mean_change:.2e}")
                print(f"  Relative change: {rel_change:.6f}")
                
                if rel_change > 0.05:
                    print(f"  ⚠ WARNING: Still changing by {rel_change:.2%} - may need more iterations")
                    issues.append("CONVERGENCE WARNING: Not fully converged")
                elif rel_change > 0.01:
                    print(f"  ⚠ Note: Changing by {rel_change:.2%} - could run longer")
                else:
                    print(f"  ✓ Well converged")
        else:
            print(f"  ⚠ Only {len(elbo_history)} ELBO values recorded")
    else:
        print(f"  ⚠ No ELBO history found")
    
    # =====================================================================
    # CHECK 3: PARAMETER RANGES
    # =====================================================================
    print("\n[PARAMETER RANGES]")
    
    if hasattr(model, 'E_theta'):
        print(f"  E[theta]: [{model.E_theta.min():.4f}, {model.E_theta.max():.4f}]")
        if model.E_theta.max() > 1e6:
            print(f"    ⚠ WARNING: Very large values detected")
            issues.append("OVERFLOW WARNING: Large theta values")
    
    if hasattr(model, 'E_beta'):
        print(f"  E[beta]:  [{model.E_beta.min():.4f}, {model.E_beta.max():.4f}]")
        if model.E_beta.max() > 1e6:
            print(f"    ⚠ WARNING: Very large values detected")
            issues.append("OVERFLOW WARNING: Large beta values")
    
    if hasattr(model, 'E_v'):
        print(f"  E[v]:     [{model.E_v.min():.4f}, {model.E_v.max():.4f}]")
    
    if hasattr(model, 'E_gamma'):
        if model.E_gamma.size == 0:
            print(f"  E[gamma]: (empty array - no covariates)")
        else:
            print(f"  E[gamma]: [{model.E_gamma.min():.4f}, {model.E_gamma.max():.4f}]")
    
    # =====================================================================
    # CHECK 4: NUMERICAL STABILITY
    # =====================================================================
    print("\n[NUMERICAL STABILITY]")
    
    has_nan = False
    has_inf = False
    
    for attr_name in ['E_theta', 'E_beta', 'mu_v', 'mu_gamma']:
        if hasattr(model, attr_name):
            attr = getattr(model, attr_name)
            if np.isnan(attr).any():
                print(f"  ✗ NaN detected in {attr_name}")
                has_nan = True
            if np.isinf(attr).any():
                print(f"  ✗ Inf detected in {attr_name}")
                has_inf = True
    
    if not has_nan and not has_inf:
        print(f"  ✓ No NaN or Inf values detected")
    else:
        issues.append("NUMERICAL ERROR: NaN or Inf detected")
    
    # =====================================================================
    # FINAL VERDICT
    # =====================================================================
    print(f"\n{'='*60}")
    
    if len(issues) == 0:
        print("✓ ALL CHECKS PASSED")
        print("="*60 + "\n")
        return True
    else:
        print("✗ ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        print("="*60 + "\n")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python check_model.py <model_path> [gene_programs_path]")
        print("\nExample:")
        print("  python check_model.py /path/to/model.pkl")
        print("  python check_model.py /path/to/model.pkl /path/to/gene_programs.pkl")
        sys.exit(1)
    
    model_path = sys.argv[1]
    gene_programs_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    passed = check_saved_model(model_path, gene_programs_path)
    sys.exit(0 if passed else 1)