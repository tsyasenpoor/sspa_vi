#!/usr/bin/env python
"""
Preprocess control h5ad file using same gene filtering pipeline as data_loader.
"""
import sys
from pathlib import Path

# Add parent dir to path for imports
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import argparse
import scanpy as sc
from BRay.VariationalInference.data_loader import DataLoader


def preprocess_control_data(
    input_path: str,
    output_path: str,
    gene_annotation_path: str,
    min_fraction: float = 0.02
) -> None:
    """
    Apply gene filtering pipeline to control h5ad file.
    
    Applies same filters as data_loader:
    1. Convert gene names to Ensembl IDs
    2. Remove duplicates
    3. Filter protein-coding genes
    4. Remove all-zero genes
    5. Filter by minimum cell expression fraction
    
    Parameters
    ----------
    input_path : str
        Input h5ad file (control cells only)
    output_path : str
        Output h5ad file path
    gene_annotation_path : str
        Path to gene annotation CSV
    min_fraction : float
        Minimum fraction of cells expressing a gene (default: 0.02)
    """
    print(f"Loading control data: {input_path}")
    
    # Initialize DataLoader
    loader = DataLoader(
        data_path=input_path,
        gene_annotation_path=gene_annotation_path,
        species='human',
        use_cache=False,
        verbose=True
    )
    
    # Load and preprocess
    loader.load_adata()
    loader.get_expression_df(layer='raw')
    loader.convert_genes_to_ensembl()
    loader.remove_duplicate_genes()
    loader.filter_protein_coding()
    loader.filter_zero_genes()
    loader.filter_min_cells(min_fraction=min_fraction)
    
    # Reconstruct AnnData with filtered genes
    print(f"\nReconstructing AnnData with {loader.raw_df.shape[1]} filtered genes...")
    
    # Update var_names to match processed gene IDs
    # loader.raw_df.columns contains Ensembl IDs after conversion
    # We need to map back or reconstruct AnnData from scratch
    
    # Rebuild AnnData with filtered genes
    filtered_gene_names = loader.raw_df.columns.tolist()
    
    # Find indices of filtered genes in original var_names (after conversion)
    # Since conversion happened, we need to use the converted names
    from BRay.VariationalInference.gene_convertor import GeneIDConverter
    converter = GeneIDConverter(cache_file=str(loader.cache_dir / 'gene_id_cache.json'))
    
    # Map original var_names to Ensembl
    original_to_ensembl, _ = converter.symbols_to_ensembl(
        loader.adata.var_names.tolist(), 
        species=loader.species
    )
    ensembl_var_names = [original_to_ensembl.get(g, g) for g in loader.adata.var_names]
    
    # Find mask for genes to keep
    gene_mask = [g in filtered_gene_names for g in ensembl_var_names]
    adata_filtered = loader.adata[:, gene_mask].copy()
    
    # Update var_names to Ensembl IDs
    adata_filtered.var_names = [ensembl_var_names[i] for i, keep in enumerate(gene_mask) if keep]
    
    print(f"Final shape: {adata_filtered.n_obs} cells x {adata_filtered.n_vars} genes")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_filtered.write_h5ad(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess control h5ad with gene filtering')
    parser.add_argument('--input', required=True, help='Input h5ad file')
    parser.add_argument('--output', required=True, help='Output h5ad file')
    parser.add_argument('--gene-annotation', required=True, help='Gene annotation CSV')
    parser.add_argument('--min-fraction', type=float, default=0.02, help='Min fraction of cells')
    
    args = parser.parse_args()
    preprocess_control_data(
        args.input,
        args.output,
        args.gene_annotation,
        args.min_fraction
    )
