"""
scDesign3 Integration Module

This module provides a Python interface to scDesign3, an R package for generating
realistic synthetic single-cell omics data.

scDesign3 learns statistical models from reference datasets and generates synthetic
data that preserves complex biological relationships including:
- Gene expression distributions (Poisson, Negative Binomial, Zero-Inflated)
- Gene-gene correlations via copula modeling
- Cell type-specific expression patterns
- Developmental trajectories (pseudotime)
- Spatial patterns

Reference:
    Song, D., Wang, Q., Yan, G. et al. scDesign3 generates realistic in silico data
    for multimodal single-cell and spatial omics. Nature Biotechnology 42, 247-252 (2024).

Example usage:
    from VariationalInference.scdesign3 import ScDesign3Simulator

    # Initialize simulator
    simulator = ScDesign3Simulator()

    # Run simulation from h5ad reference data
    result = simulator.simulate(
        input_file='reference.h5ad',
        celltype_column='cell_type',
        n_cells=5000,
        family='nb',
        output_dir='./simulated_data'
    )

    # Load results as AnnData
    adata = result.to_anndata()
"""

from .simulator import ScDesign3Simulator, ScDesign3Result
from .utils import check_r_installation, install_scdesign3

__all__ = [
    'ScDesign3Simulator',
    'ScDesign3Result',
    'check_r_installation',
    'install_scdesign3',
]
