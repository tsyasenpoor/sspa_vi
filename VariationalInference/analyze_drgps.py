import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def get_sparsity_info(model, threshold=0.5):
    """
    Extract sparsity information from spike-and-slab model.
    
    Parameters:
    -----------
    model : VI
        Trained VI model with spike-and-slab priors
    threshold : float
        Probability threshold for considering a parameter "active"
        
    Returns:
    --------
    info : dict or None
        Dictionary containing sparsity statistics, or None if model doesn't have spike-and-slab
    """
    if not hasattr(model, 'rho_beta') or not hasattr(model, 'rho_v'):
        return None
    
    beta_active = model.rho_beta > threshold
    v_active = model.rho_v > threshold
    
    info = {
        'beta_active': beta_active,
        'beta_sparsity': 1.0 - beta_active.mean(),
        'beta_active_per_factor': beta_active.sum(axis=0),
        'beta_active_per_gene': beta_active.sum(axis=1),
        'v_active': v_active,
        'v_sparsity': 1.0 - v_active.mean(),
        'v_active_per_factor': v_active.sum(axis=0),
        'v_active_per_class': v_active.sum(axis=1),
        'rho_beta': model.rho_beta,
        'rho_v': model.rho_v,
    }
    
    return info


def load_model_and_genes(model_path, data_dir='/labs/Aguiar/SSPA_BRAY/BRay/ctrl_sspa_test'):
    """Load trained model and gene names."""
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load gene list to get names
    gene_list_path = Path(data_dir) / 'gene_list.txt'
    if gene_list_path.exists():
        with open(gene_list_path, 'r') as f:
            all_genes = [line.strip() for line in f]
    else:
        all_genes = None
    
    return model, all_genes


def analyze_beta_diversity(model, gene_names=None):
    """
    Check if gene programs (beta) are learning different things.
    
    High correlation between programs = factor collapse
    """
    print("\n" + "="*60)
    print("BETA MATRIX (GENE PROGRAMS) ANALYSIS")
    print("="*60)
    
    beta = model.E_beta  # Shape: (p, d) or (n_genes, n_factors)
    n_genes, n_factors = beta.shape
    
    print(f"\nShape: {n_genes} genes × {n_factors} factors")
    print(f"Range: [{beta.min():.4f}, {beta.max():.4f}]")
    print(f"Mean: {beta.mean():.4f}, Std: {beta.std():.4f}")
    
    # Check correlations between programs
    print(f"\n{'='*60}")
    print("FACTOR DIVERSITY CHECK")
    print(f"{'='*60}")
    
    # Transpose to get (d, p) for correlation between factors
    corr = np.corrcoef(beta.T)  # Shape: (d, d)
    
    # Get upper triangle (exclude diagonal)
    triu_idx = np.triu_indices_from(corr, k=1)
    off_diag_corr = corr[triu_idx]
    
    print(f"\nPairwise correlations between factors:")
    print(f"  Mean: {off_diag_corr.mean():.4f}")
    print(f"  Std:  {off_diag_corr.std():.4f}")
    print(f"  Min:  {off_diag_corr.min():.4f}")
    print(f"  Max:  {off_diag_corr.max():.4f}")
    
    if off_diag_corr.mean() > 0.8:
        print(f"  ⚠ WARNING: High average correlation - possible factor collapse")
    elif off_diag_corr.max() > 0.95:
        print(f"  ⚠ WARNING: Some factors highly correlated")
    else:
        print(f"  ✓ Good diversity - factors learning different patterns")
    
    # Show correlation matrix
    print(f"\nFull correlation matrix:")
    print("Factor " + " ".join(f"  {i:2d}" for i in range(min(n_factors, 10))))
    for i in range(min(n_factors, 10)):
        row_str = f"  {i:2d}   " + " ".join(f"{corr[i,j]:5.2f}" for j in range(min(n_factors, 10)))
        print(row_str)
    
    return corr


def get_top_genes_per_program(model, gene_names, n_top=20):
    """Get top genes for each program."""
    beta = model.E_beta  # Shape: (p, d)
    
    print(f"\n{'='*60}")
    print("TOP GENES PER PROGRAM")
    print(f"{'='*60}")
    
    top_genes_dict = {}
    
    for k in range(beta.shape[1]):
        beta_k = beta[:, k]
        top_idx = np.argsort(beta_k)[-n_top:][::-1]  # Top n in descending order
        
        top_genes_dict[f'Program_{k}'] = {
            'indices': top_idx,
            'values': beta_k[top_idx],
            'genes': [gene_names[i] if gene_names else f"Gene_{i}" for i in top_idx] if gene_names else None
        }
        
        print(f"\nProgram {k}:")
        print(f"  Top {min(5, n_top)} genes:")
        for i, idx in enumerate(top_idx[:5]):
            gene_name = gene_names[idx] if gene_names else f"Gene_{idx}"
            print(f"    {i+1}. {gene_name:20s}  β = {beta_k[idx]:.4f}")
    
    return top_genes_dict


def analyze_theta_distribution(model):
    """Analyze sample factors (theta)."""
    print(f"\n{'='*60}")
    print("THETA MATRIX (SAMPLE FACTORS) ANALYSIS")
    print(f"{'='*60}")
    
    theta = model.E_theta  # Shape: (n, d)
    n_samples, n_factors = theta.shape
    
    print(f"\nShape: {n_samples} samples × {n_factors} factors")
    print(f"Range: [{theta.min():.4f}, {theta.max():.4f}]")
    print(f"Mean: {theta.mean():.4f}, Std: {theta.std():.4f}")
    
    # Check sparsity
    near_zero = (np.abs(theta) < 0.01).sum()
    sparsity = near_zero / theta.size
    print(f"\nSparsity (|θ| < 0.01): {sparsity:.2%}")
    
    # Per-factor statistics
    print(f"\nPer-factor statistics:")
    print(f"{'Factor':>8s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    for k in range(n_factors):
        theta_k = theta[:, k]
        print(f"{k:8d} {theta_k.mean():10.4f} {theta_k.std():10.4f} "
              f"{theta_k.min():10.4f} {theta_k.max():10.4f}")
    
    return theta


def analyze_v_weights(model):
    """Analyze supervised v parameters."""
    print(f"\n{'='*60}")
    print("V-WEIGHTS (SUPERVISED PARAMETERS) ANALYSIS")
    print(f"{'='*60}")
    
    v = model.mu_v  # Shape: (kappa, d)
    kappa, d = v.shape
    
    print(f"\nShape: {kappa} programs × {d} factors")
    print(f"Range: [{v.min():.4f}, {v.max():.4f}]")
    print(f"Mean: {v.mean():.4f}, Std: {v.std():.4f}")
    
    print(f"\nPer-program v-weights:")
    print(f"{'Program':>8s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    for k in range(kappa):
        v_k = v[k]
        print(f"{k:8d} {v_k.mean():10.4f} {v_k.std():10.4f} "
              f"{v_k.min():10.4f} {v_k.max():10.4f}")
    
    # Expected pattern for control-only data
    print(f"\n💡 INTERPRETATION (control-only data):")
    print(f"  - All y = 0 (no disease)")
    print(f"  - v-weights should be near 0 or negative")
    print(f"  - No true signal to learn")
    print(f"  - Will change dramatically with disease data")
    
    return v


def plot_beta_heatmap(model, output_path=None):
    """Plot heatmap of beta matrix."""
    beta = model.E_beta
    
    # If too many genes, sample them
    if beta.shape[0] > 100:
        idx = np.random.choice(beta.shape[0], 100, replace=False)
        beta_subset = beta[idx]
        title_suffix = " (100 random genes)"
    else:
        beta_subset = beta
        title_suffix = ""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(beta_subset, cmap='RdBu_r', center=0, ax=ax, 
                xticklabels=[f'F{i}' for i in range(beta.shape[1])],
                yticklabels=False)
    ax.set_xlabel('Factor')
    ax.set_ylabel('Gene')
    ax.set_title(f'Gene Program Matrix (β){title_suffix}')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  Saved plot: {output_path}")
    
    return fig


def plot_theta_distribution(model, output_path=None):
    """Plot distribution of theta values."""
    theta = model.E_theta
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Overall distribution
    axes[0].hist(theta.ravel(), bins=50, edgecolor='black')
    axes[0].set_xlabel('θ value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Sample Factors')
    axes[0].axvline(theta.mean(), color='red', linestyle='--', label='Mean')
    axes[0].legend()
    
    # Per-factor boxplot
    axes[1].boxplot([theta[:, k] for k in range(theta.shape[1])],
                    labels=[f'F{k}' for k in range(theta.shape[1])])
    axes[1].set_xlabel('Factor')
    axes[1].set_ylabel('θ value')
    axes[1].set_title('Sample Factors by Program')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {output_path}")
    
    return fig


def main(model_path, output_dir=None):
    """Main inspection workflow."""
    print("="*60)
    print("GENE PROGRAM INSPECTION")
    print("="*60)
    print(f"\nModel: {model_path}")
    
    # Load model
    model, gene_names = load_model_and_genes(model_path)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(model_path).parent / 'inspection'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Analyze beta (gene programs)
    corr = analyze_beta_diversity(model, gene_names)
    
    # Get top genes if gene names available
    if gene_names and len(gene_names) >= model.E_beta.shape[0]:
        top_genes = get_top_genes_per_program(model, gene_names, n_top=20)
    
    # Analyze theta (sample factors)
    theta = analyze_theta_distribution(model)
    
    # Analyze v-weights
    v = analyze_v_weights(model)
    
    # Compute and display sparsity information if spike-and-slab is used
    sparsity_info = get_sparsity_info(model, threshold=0.5)
    if sparsity_info is not None:
        print(f"\n{'='*60}")
        print("SPIKE-AND-SLAB SPARSITY ANALYSIS")
        print(f"{'='*60}")
        
        beta_active = sparsity_info['beta_active']
        v_active = sparsity_info['v_active']
        
        print(f"\nBeta (Gene Programs):")
        print(f"  Active elements: {beta_active.sum()}/{sparsity_info['rho_beta'].size}")
        print(f"  Sparsity: {(1 - beta_active.mean())*100:.1f}%")
        print(f"  Active per factor: {beta_active.sum(axis=0)}")
        print(f"  Active per gene: min={beta_active.sum(axis=1).min()}, max={beta_active.sum(axis=1).max()}")
        
        print(f"\nV (Classification Weights):")
        print(f"  Active elements: {v_active.sum()}/{sparsity_info['rho_v'].size}")
        print(f"  Sparsity: {(1 - v_active.mean())*100:.1f}%")
        print(f"  Active per factor: {v_active.sum(axis=0)}")
        print(f"  Active per class: {v_active.sum(axis=1)}")
    
    # Create plots
    print(f"\n{'='*60}")
    print("CREATING PLOTS")
    print(f"{'='*60}")
    
    fig_beta = plot_beta_heatmap(model, output_dir / 'beta_heatmap.png')
    fig_theta = plot_theta_distribution(model, output_dir / 'theta_distribution.png')
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=[f'F{i}' for i in range(corr.shape[0])],
                yticklabels=[f'F{i}' for i in range(corr.shape[0])],
                ax=ax, vmin=-1, vmax=1)
    ax.set_title('Factor Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'factor_correlation.png', dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {output_dir / 'factor_correlation.png'}")
    
    plt.close('all')
    
    print(f"\n{'='*60}")
    print("INSPECTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - beta_heatmap.png: Gene program visualization")
    print(f"  - theta_distribution.png: Sample factor distributions")
    print(f"  - factor_correlation.png: Factor diversity check")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inspect_gene_programs.py <model_path> [output_dir]")
        print("\nExample:")
        print("  python inspect_gene_programs.py progressive_tests/stage_1:_tiny_model.pkl")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(model_path, output_dir)