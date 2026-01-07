#!/usr/bin/env python
"""
Filter h5ad to keep only control cells (t2dm==0) for scDesign3 simulation.
"""
import argparse
from pathlib import Path
import scanpy as sc


def filter_control_cells(
    input_path: str,
    output_path: str,
    label_column: str = 't2dm',
    control_value: int = 0
) -> None:
    """
    Filter h5ad file to keep only control cells based on label column.
    
    Parameters
    ----------
    input_path : str
        Input h5ad file path
    output_path : str
        Output h5ad file path for filtered data
    label_column : str
        Column name in obs containing disease labels
    control_value : int
        Value indicating control samples (default: 0)
    """
    print(f"Loading: {input_path}")
    adata = sc.read_h5ad(input_path)
    print(f"  Original shape: {adata.n_obs} cells x {adata.n_vars} features")
    
    if label_column not in adata.obs.columns:
        raise ValueError(f"Column '{label_column}' not found in obs. Available: {list(adata.obs.columns)}")
    
    print(f"  {label_column} distribution: {adata.obs[label_column].value_counts().to_dict()}")
    
    # Filter to control cells
    mask = adata.obs[label_column] == control_value
    adata_filtered = adata[mask, :].copy()
    
    print(f"  Filtered to {label_column}=={control_value}: {adata_filtered.n_obs} cells")
    
    # Save filtered data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_filtered.write_h5ad(output_path)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter h5ad to control cells only')
    parser.add_argument('--input', required=True, help='Input h5ad file')
    parser.add_argument('--output', required=True, help='Output h5ad file')
    parser.add_argument('--label-column', default='t2dm', help='Label column name')
    parser.add_argument('--control-value', type=int, default=0, help='Control value')
    
    args = parser.parse_args()
    filter_control_cells(args.input, args.output, args.label_column, args.control_value)
