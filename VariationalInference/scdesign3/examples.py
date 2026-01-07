#!/usr/bin/env python3
"""
Example usage of scDesign3 integration for synthetic single-cell data generation.

scDesign3 generates realistic synthetic single-cell data by:
1. Learning statistical models from reference data (marginal distributions per gene)
2. Capturing gene-gene correlations via copula modeling
3. Generating new cells that preserve these learned properties

This module shows how to:
- Generate basic synthetic data from a reference h5ad
- Generate data with cell type-specific patterns
- Generate trajectory data (with pseudotime)
- Combine with disease signal injection for benchmarking
"""

import argparse
from pathlib import Path


def example_basic_simulation():
    """
    Example 1: Basic simulation from reference data.

    This generates synthetic cells that match the statistical properties
    of your reference dataset.
    """
    from VariationalInference.scdesign3 import ScDesign3Simulator

    print("=" * 60)
    print("Example 1: Basic Simulation")
    print("=" * 60)

    # Initialize simulator
    simulator = ScDesign3Simulator()

    # Check installation
    if not simulator.check_installation():
        print("\nscDesign3 not installed. Run:")
        print("  cd VariationalInference/scdesign3 && bash setup_r_env.sh")
        return None

    # Run simulation
    result = simulator.simulate(
        input_file="path/to/reference.h5ad",  # Your reference data
        output_dir="./scdesign3_output",
        n_cells=5000,          # Generate 5000 cells (0 = same as input)
        family="nb",           # Negative binomial (good for count data)
        copula="gaussian",     # Gaussian copula for correlations
        n_cores=4,             # Use 4 CPU cores
        seed=42,               # For reproducibility
    )

    print(f"\nGenerated {result.n_cells} cells with {result.n_genes} genes")
    print(f"Output saved to: {result.output_dir}")

    # Convert to AnnData for downstream analysis
    adata = result.to_anndata()
    print(f"AnnData shape: {adata.shape}")

    return result


def example_celltype_simulation():
    """
    Example 2: Simulation preserving cell type-specific patterns.

    If your reference data has cell type annotations, scDesign3 can
    learn and preserve cell type-specific expression patterns.
    """
    from VariationalInference.scdesign3 import ScDesign3Simulator

    print("=" * 60)
    print("Example 2: Cell Type-Aware Simulation")
    print("=" * 60)

    simulator = ScDesign3Simulator()

    result = simulator.simulate(
        input_file="path/to/reference.h5ad",
        output_dir="./scdesign3_celltype_output",
        celltype_column="cell_type",  # Column in obs with cell type labels
        n_cells=10000,
        family="nb",
        copula="gaussian",
        n_cores=4,
        seed=42,
    )

    print(f"\nGenerated data with cell type structure preserved")

    # Check cell type distribution in output
    if result.metadata is not None and "cell_type" in result.metadata.columns:
        print("\nCell type distribution:")
        print(result.metadata["cell_type"].value_counts())

    return result


def example_trajectory_simulation():
    """
    Example 3: Simulation with developmental trajectory (pseudotime).

    For trajectory data, scDesign3 can model gene expression as a
    function of pseudotime, preserving developmental patterns.
    """
    from VariationalInference.scdesign3 import ScDesign3Simulator

    print("=" * 60)
    print("Example 3: Trajectory/Pseudotime Simulation")
    print("=" * 60)

    simulator = ScDesign3Simulator()

    result = simulator.simulate(
        input_file="path/to/trajectory_data.h5ad",
        output_dir="./scdesign3_trajectory_output",
        celltype_column="cell_type",
        pseudotime_column="pseudotime",  # Column with pseudotime values
        # Custom formula for mean model (GAM with cubic regression spline)
        mu_formula="s(pseudotime, bs = 'cr', k = 10)",
        n_cells=5000,
        family="nb",
        copula="gaussian",
        n_cores=4,
        seed=42,
    )

    print(f"\nGenerated trajectory data with pseudotime structure")

    return result


def example_disease_benchmark():
    """
    Example 4: Generate benchmark data with known disease signal.

    This combines scDesign3's realistic data generation with
    controlled disease signal injection for method benchmarking.
    """
    from VariationalInference.scdesign3.simulator import simulate_with_disease_signal

    print("=" * 60)
    print("Example 4: Disease Benchmark Data")
    print("=" * 60)

    output = simulate_with_disease_signal(
        reference_file="path/to/reference.h5ad",
        output_dir="./scdesign3_disease_output",
        celltype_column="cell_type",
        n_cells=5000,
        # Disease parameters
        n_disease_genes=30,      # 30 genes will be disease-associated
        disease_fraction=0.5,    # 50% of cells will be "diseased"
        boost_factor=2.0,        # 2x expression increase in disease genes
        # scDesign3 parameters
        family="nb",
        seed=42,
    )

    result = output["result"]
    y = output["y"]
    ground_truth = output["ground_truth"]

    print(f"\nGenerated {result.n_cells} cells")
    print(f"  Control cells: {(y == 0).sum()}")
    print(f"  Disease cells: {(y == 1).sum()}")
    print(f"  Disease genes: {ground_truth['n_disease_genes']}")
    print(f"  Boost factor: {ground_truth['boost_factor']}x")

    print(f"\nGround truth saved to: {result.output_dir}/ground_truth.json")
    print("Use this to evaluate gene recovery in your model!")

    return output


def example_from_anndata():
    """
    Example 5: Simulate directly from an in-memory AnnData object.
    """
    from VariationalInference.scdesign3 import ScDesign3Simulator
    import scanpy as sc

    print("=" * 60)
    print("Example 5: Simulate from AnnData in Memory")
    print("=" * 60)

    # Load your data with scanpy
    adata = sc.read_h5ad("path/to/data.h5ad")

    # Optional: preprocess (but scDesign3 works best with raw counts)
    # adata = adata.raw.to_adata()  # Get raw counts if available

    simulator = ScDesign3Simulator()

    result = simulator.simulate_from_anndata(
        adata=adata,
        output_dir="./scdesign3_from_adata_output",
        celltype_column="cell_type",
        n_cells=5000,
        family="nb",
        seed=42,
    )

    # Get back as AnnData
    simulated_adata = result.to_anndata()
    print(f"\nSimulated AnnData shape: {simulated_adata.shape}")

    return result


def example_quick_test():
    """
    Example 6: Quick test with subset of genes (for development/testing).
    """
    from VariationalInference.scdesign3 import ScDesign3Simulator

    print("=" * 60)
    print("Example 6: Quick Test (Gene Subset)")
    print("=" * 60)

    simulator = ScDesign3Simulator()

    # Use only 500 most variable genes for quick testing
    result = simulator.simulate(
        input_file="path/to/reference.h5ad",
        output_dir="./scdesign3_quick_test",
        n_cells=1000,
        n_genes=500,               # Only use 500 genes
        gene_selection="variable", # Select by variance
        min_cells_expressing=0.05, # Filter lowly expressed genes
        family="nb",
        copula="gaussian",
        n_cores=2,
        seed=42,
    )

    print(f"\nQuick test: {result.n_genes} genes, {result.n_cells} cells")

    return result


def example_integration_with_vi():
    """
    Example 7: Full pipeline - scDesign3 data -> VI model training.

    This shows how to use scDesign3-generated data with the
    variational inference models in this project.
    """
    print("=" * 60)
    print("Example 7: Integration with VI Models")
    print("=" * 60)

    # Step 1: Generate synthetic data with disease signal
    from VariationalInference.scdesign3.simulator import simulate_with_disease_signal
    import pickle
    import numpy as np

    output = simulate_with_disease_signal(
        reference_file="path/to/reference.h5ad",
        output_dir="./vi_benchmark_data",
        n_cells=5000,
        n_disease_genes=30,
        disease_fraction=0.5,
        boost_factor=2.0,
        seed=42,
    )

    result = output["result"]
    y = output["y"]

    # Step 2: Prepare data for VI model (cells x genes format)
    X = result.counts.T  # Transpose to cells x genes

    # Split into train/val/test
    n_samples = X.shape[0]
    indices = np.random.RandomState(42).permutation(n_samples)

    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Save in format expected by VI models
    output_dir = Path("./vi_benchmark_data")

    with open(output_dir / "X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(output_dir / "y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open(output_dir / "X_val.pkl", "wb") as f:
        pickle.dump(X_val, f)
    with open(output_dir / "y_val.pkl", "wb") as f:
        pickle.dump(y_val, f)
    with open(output_dir / "X_test.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with open(output_dir / "y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)

    print(f"\nData prepared for VI model:")
    print(f"  Train: {X_train.shape} (disease: {y_train.sum()})")
    print(f"  Val:   {X_val.shape} (disease: {y_val.sum()})")
    print(f"  Test:  {X_test.shape} (disease: {y_test.sum()})")

    # Step 3: Train VI model (example code)
    print("\nTo train the VI model:")
    print("  from VariationalInference.vi import BatchVI")
    print("  model = BatchVI(X_train, y_train, K=10)")
    print("  model.fit(max_iter=200)")

    # Step 4: Evaluate gene recovery
    print("\nTo evaluate gene recovery:")
    print("  ground_truth = output['ground_truth']")
    print("  disease_genes = ground_truth['disease_gene_indices']")
    print("  # Compare with model's top beta weights")

    return output


def main():
    """Run examples based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="scDesign3 integration examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples.py --check        Check R/scDesign3 installation
  python examples.py --list         List available examples
  python examples.py --run basic    Run basic simulation example
  python examples.py --run all      Run all examples
        """,
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check scDesign3 installation",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available examples",
    )
    parser.add_argument(
        "--run",
        type=str,
        choices=["basic", "celltype", "trajectory", "disease", "anndata", "quick", "vi", "all"],
        help="Run specific example",
    )

    args = parser.parse_args()

    if args.check:
        from VariationalInference.scdesign3 import check_r_installation
        status = check_r_installation()
        print("\nInstallation Status:")
        print(f"  R installed: {status['r_installed']}")
        print(f"  R version: {status['r_version']}")
        print(f"  scDesign3 installed: {status['scdesign3_installed']}")
        print(f"  scDesign3 version: {status['scdesign3_version']}")
        if status['missing_packages']:
            print(f"  Missing packages: {status['missing_packages']}")
        return

    if args.list:
        print("\nAvailable Examples:")
        print("  basic     - Basic simulation from reference data")
        print("  celltype  - Cell type-aware simulation")
        print("  trajectory- Pseudotime/trajectory simulation")
        print("  disease   - Disease benchmark data generation")
        print("  anndata   - Simulate from in-memory AnnData")
        print("  quick     - Quick test with gene subset")
        print("  vi        - Full pipeline with VI model integration")
        print("  all       - Run all examples")
        return

    if args.run:
        examples = {
            "basic": example_basic_simulation,
            "celltype": example_celltype_simulation,
            "trajectory": example_trajectory_simulation,
            "disease": example_disease_benchmark,
            "anndata": example_from_anndata,
            "quick": example_quick_test,
            "vi": example_integration_with_vi,
        }

        if args.run == "all":
            for name, func in examples.items():
                print(f"\n{'#' * 60}")
                print(f"# Running: {name}")
                print(f"{'#' * 60}\n")
                try:
                    func()
                except Exception as e:
                    print(f"Example {name} failed: {e}")
        else:
            examples[args.run]()
        return

    # Default: print help
    parser.print_help()


if __name__ == "__main__":
    main()
