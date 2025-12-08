import scanpy as sc
import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc
import psutil
from scipy import sparse
from scipy.io import mmread
import gzip


def get_dataset_info(data_path="/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19"):
    """
    Get basic information about the dataset without loading it into memory.
    """
    import gzip
    
    # Count cells
    barcodes_file = os.path.join(data_path, "barcodes.tsv.gz")
    with gzip.open(barcodes_file, 'rt') as f:
        n_cells = sum(1 for _ in f)
    
    # Count genes
    features_file = os.path.join(data_path, "features.tsv.gz")
    with gzip.open(features_file, 'rt') as f:
        n_genes = sum(1 for _ in f)
    
    print(f"Dataset info:")
    print(f"  Total cells: {n_cells:,}")
    print(f"  Total genes: {n_genes:,}")
    print(f"  Estimated memory for full dataset: ~{(n_cells * n_genes * 4) / (1024**3):.2f} GB")
    
    return n_cells, n_genes


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def load_covid19_chunked(
    data_path="/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19",
    chunk_size=50000,
    max_cells=None,
    min_genes=200,
    min_cells=3,
    max_pct_mito=20,
    cache_file=None,
    verbose=True
):
    """
    Load COVID-19 dataset in chunks to handle large datasets efficiently.
    This is the most memory-efficient approach for very large datasets.
    
    Parameters:
    -----------
    chunk_size : int, default=50000
        Number of cells to process in each chunk
    max_cells : int, optional
        Maximum total cells to load (for testing)
    min_genes : int, default=200
        Minimum genes per cell
    min_cells : int, default=3
        Minimum cells per gene
    max_pct_mito : float, default=20
        Maximum mitochondrial percentage
    cache_file : str, optional
        Path to save final processed data
    verbose : bool, default=True
        Print progress information
        
    Returns:
    --------
    adata : AnnData
        Processed and filtered dataset
    """
    
    if verbose:
        print(f"Loading COVID-19 data in chunks of {chunk_size:,} cells")
        print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Get total number of cells
    barcodes_file = os.path.join(data_path, "barcodes.tsv.gz")
    with gzip.open(barcodes_file, 'rt') as f:
        total_cells = sum(1 for _ in f)
    
    if max_cells:
        total_cells = min(total_cells, max_cells)
    
    if verbose:
        print(f"Total cells to process: {total_cells:,}")
    
    # Read all barcodes and features once
    with gzip.open(barcodes_file, 'rt') as f:
        all_barcodes = [line.strip() for line in f][:total_cells]
    
    features_file = os.path.join(data_path, "features.tsv.gz")
    with gzip.open(features_file, 'rt') as f:
        features = [line.strip().split('\t') for line in f]
    
    gene_names = [f[1] if len(f) > 1 else f[0] for f in features]
    n_genes = len(gene_names)
    
    if verbose:
        print(f"Total genes: {n_genes:,}")
    
    # Process in chunks
    processed_cells = []
    processed_matrices = []
    n_chunks = (total_cells + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_cells)
        chunk_barcodes = all_barcodes[start_idx:end_idx]
        
        if verbose:
            print(f"\nProcessing chunk {chunk_idx + 1}/{n_chunks} "
                  f"(cells {start_idx:,}-{end_idx:,})")
            print(f"Memory before chunk: {get_memory_usage():.2f} GB")
        
        # Load matrix chunk
        matrix_file = os.path.join(data_path, "matrix.mtx.gz")
        with gzip.open(matrix_file, 'rb') as f:
            matrix = mmread(f).tocsr()
        
        # Subset to current chunk
        matrix_chunk = matrix[:, start_idx:end_idx]
        del matrix  # Free memory immediately
        gc.collect()
        
        # Create AnnData for this chunk
        adata_chunk = sc.AnnData(
            X=matrix_chunk.T.tocsr(),
            obs=pd.DataFrame(index=chunk_barcodes),
            var=pd.DataFrame(index=gene_names)
        )
        
        # Calculate QC metrics for this chunk
        adata_chunk.var['mt'] = adata_chunk.var_names.str.startswith(('MT-', 'Mt-'))
        sc.pp.calculate_qc_metrics(adata_chunk, qc_vars=['mt'], inplace=True)
        
        # Filter cells in this chunk
        initial_cells = adata_chunk.n_obs
        sc.pp.filter_cells(adata_chunk, min_genes=min_genes)
        
        if max_pct_mito and 'pct_counts_mt' in adata_chunk.obs.columns:
            adata_chunk = adata_chunk[adata_chunk.obs['pct_counts_mt'] < max_pct_mito, :]
        
        if verbose:
            print(f"  Chunk {chunk_idx + 1}: {initial_cells:,} → {adata_chunk.n_obs:,} cells "
                  f"({adata_chunk.n_obs/initial_cells*100:.1f}% kept)")
            print(f"  Memory after chunk: {get_memory_usage():.2f} GB")
        
        # Store processed chunk
        if adata_chunk.n_obs > 0:
            processed_cells.append(adata_chunk.obs.index.tolist())
            processed_matrices.append(adata_chunk.X)
        
        # Clean up
        del adata_chunk
        gc.collect()
    
    if verbose:
        print(f"\nCombining {len(processed_matrices)} processed chunks...")
        print(f"Memory before combining: {get_memory_usage():.2f} GB")
    
    # Combine all processed chunks
    if not processed_matrices:
        raise ValueError("No cells passed filtering criteria")
    
    # Combine matrices
    combined_matrix = sparse.vstack(processed_matrices)
    combined_barcodes = [barcode for chunk_barcodes in processed_cells for barcode in chunk_barcodes]
    
    # Create final AnnData
    adata = sc.AnnData(
        X=combined_matrix,
        obs=pd.DataFrame(index=combined_barcodes),
        var=pd.DataFrame(index=gene_names)
    )
    
    # Final gene filtering
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if verbose:
        print(f"Final dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    # Save cache if requested
    if cache_file:
        if verbose:
            print(f"Saving to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        adata.write_h5ad(cache_file)
    
    return adata


def load_covid19_backed(
    data_path="/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19",
    cache_file=None,
    min_genes=200,
    min_cells=3,
    max_pct_mito=20,
    subset_cells=None,
    verbose=True
):
    """
    Load COVID-19 dataset using backed mode to minimize memory usage.
    This keeps the data on disk and only loads what's needed into memory.
    
    Parameters:
    -----------
    data_path : str
        Path to 10x format files
    cache_file : str, optional
        Path to save backed h5ad file
    min_genes : int, default=200
        Minimum genes per cell
    min_cells : int, default=3
        Minimum cells per gene
    max_pct_mito : float, default=20
        Maximum mitochondrial percentage
    subset_cells : int, optional
        Randomly subset to this many cells
    verbose : bool, default=True
        Print progress information
        
    Returns:
    --------
    adata : AnnData
        Backed AnnData object
    """
    
    if verbose:
        print("Loading COVID-19 data in backed mode (disk-based)")
        print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Load with backed mode
    adata = sc.read_10x_mtx(
        data_path,
        var_names='gene_symbols',
        cache=True,
        gex_only=True,
        backed='r'  # Read-only backed mode
    )
    
    # Transpose to cells × genes
    adata = adata.T
    
    if verbose:
        print(f"Loaded data: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
        print(f"Memory usage after load: {get_memory_usage():.2f} GB")
    
    # Subset cells if requested (do this early to save memory)
    if subset_cells and subset_cells < adata.n_obs:
        if verbose:
            print(f"Randomly subsetting to {subset_cells:,} cells...")
        sc.pp.subsample(adata, n_obs=subset_cells)
    
    # Make variable names unique
    adata.var_names_make_unique()
    
    # Calculate QC metrics (this will load data into memory temporarily)
    if verbose:
        print("Calculating QC metrics...")
    
    adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'Mt-'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    
    # Filter cells
    if verbose:
        print(f"Cells before filtering: {adata.n_obs:,}")
        if 'pct_counts_mt' in adata.obs.columns:
            print(f"Median mitochondrial percentage: {adata.obs['pct_counts_mt'].median():.2f}%")
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    if max_pct_mito and 'pct_counts_mt' in adata.obs.columns:
        adata = adata[adata.obs['pct_counts_mt'] < max_pct_mito, :]
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if verbose:
        print(f"After filtering: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    # Save backed file if requested
    if cache_file:
        if verbose:
            print(f"Saving backed file: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        adata.write_h5ad(cache_file)
    
    return adata


def load_covid19_data(
    data_path="/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19",
    cache_file=None,
    min_genes=200,
    min_cells=3,
    max_genes=None,
    max_pct_mito=None,
    n_top_genes=None,
    subset_cells=None,
    force_reload=False,
    verbose=True,
    backed=False
):
    """
    Load COVID-19 dataset from 10x format files with optional filtering and caching.
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz
    cache_file : str, optional
        Path to save/load a cached h5ad file. If exists and force_reload=False, 
        will load from cache instead of raw files.
    min_genes : int, default=200
        Minimum number of genes expressed for a cell to be kept
    min_cells : int, default=3
        Minimum number of cells expressing a gene for it to be kept
    max_genes : int, optional
        Maximum number of genes for a cell (helps filter doublets)
    max_pct_mito : float, optional
        Maximum percentage of mitochondrial genes (e.g., 0.05 for 5%)
    n_top_genes : int, optional
        Number of highly variable genes to subset to (reduces size significantly)
    subset_cells : int, optional
        Randomly subset to this many cells (for testing or smaller datasets)
    force_reload : bool, default=False
        Force reload from raw files even if cache exists
    verbose : bool, default=True
        Print progress information
    backed : bool, default=False
        Use backed mode to avoid loading full matrix into memory
        
    Returns:
    --------
    adata : AnnData
        Loaded and optionally filtered AnnData object
    """
    
    # Check if cache exists and should be used
    if cache_file and os.path.exists(cache_file) and not force_reload:
        if verbose:
            print(f"Loading from cache: {cache_file}")
        adata = sc.read_h5ad(cache_file, backed='r' if backed else None)
        if verbose:
            print(f"Loaded cached data: {adata.shape[0]} cells × {adata.shape[1]} genes")
            print(f"Memory usage: {get_memory_usage():.2f} GB")
        return adata
    
    if verbose:
        print(f"Loading COVID-19 data from: {data_path}")
        if subset_cells:
            print(f"Will subset to {subset_cells} cells to save memory")
        print("This may take a few minutes...")
    
    # IMPORTANT: If subsetting, we need to do it BEFORE loading to save memory
    # But scanpy's read_10x_mtx doesn't support this directly
    # So we'll use a more memory-efficient approach
    
    if subset_cells and subset_cells < 10000:  # For small subsets, use a different approach
        if verbose:
            print("Using memory-efficient loading for small subset...")
        
        # Load just the metadata first
        import gzip
        from scipy.io import mmread
        from scipy import sparse
        
        # Read barcodes
        barcodes_file = os.path.join(data_path, "barcodes.tsv.gz")
        with gzip.open(barcodes_file, 'rt') as f:
            all_barcodes = [line.strip() for line in f]
        
        total_cells = len(all_barcodes)
        if verbose:
            print(f"Total cells in dataset: {total_cells:,}")
        
        # Randomly select cell indices
        np.random.seed(42)
        selected_indices = np.sort(np.random.choice(total_cells, size=min(subset_cells, total_cells), replace=False))
        selected_barcodes = [all_barcodes[i] for i in selected_indices]
        
        if verbose:
            print(f"Loading matrix for {len(selected_indices)} selected cells...")
        
        # Load full matrix (this is unavoidable but we'll subset immediately)
        matrix_file = os.path.join(data_path, "matrix.mtx.gz")
        with gzip.open(matrix_file, 'rb') as f:
            matrix = mmread(f).tocsr()  # genes × cells
        
        # Subset to selected cells immediately
        matrix_subset = matrix[:, selected_indices]
        
        # Clear the full matrix from memory
        del matrix
        
        # Read features
        features_file = os.path.join(data_path, "features.tsv.gz")
        with gzip.open(features_file, 'rt') as f:
            features = [line.strip().split('\t') for line in f]
        
        gene_ids = [f[0] for f in features]
        gene_names = [f[1] if len(f) > 1 else f[0] for f in features]
        
        # Create AnnData with transposed matrix
        adata = sc.AnnData(
            X=matrix_subset.T.tocsr(),  # Now cells × genes
            obs=pd.DataFrame(index=selected_barcodes),
            var=pd.DataFrame(index=gene_names)
        )
        
        if verbose:
            print(f"Created AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    else:
        # For larger subsets or full data, use scanpy's function
        adata = sc.read_10x_mtx(
            data_path,
            var_names='gene_symbols',
            cache=True,
            gex_only=True
        )
        
        if verbose:
            print(f"Initial data loaded: {adata.shape[0]} genes × {adata.shape[1]} cells")
        
        # Transpose to standard format (cells × genes)
        adata = adata.T
        
        if verbose:
            print(f"After transpose: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Subset cells if requested
        if subset_cells and subset_cells < adata.n_obs:
            if verbose:
                print(f"Randomly subsetting to {subset_cells} cells...")
            sc.pp.subsample(adata, n_obs=subset_cells)
    
    # Make variable names unique
    adata.var_names_make_unique()
    
    # Calculate QC metrics
    if verbose:
        print("Calculating QC metrics...")
    
    # Identify mitochondrial genes (starts with 'MT-' or 'Mt-')
    adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'Mt-'))
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata, 
        qc_vars=['mt'], 
        percent_top=None, 
        log1p=False, 
        inplace=True
    )
    
    if verbose:
        print(f"Cells before filtering: {adata.n_obs}")
        print(f"Genes before filtering: {adata.n_vars}")
        if 'pct_counts_mt' in adata.obs.columns:
            print(f"Median mitochondrial percentage: {adata.obs['pct_counts_mt'].median():.2f}%")
        print(f"Median genes per cell: {adata.obs['n_genes_by_counts'].median():.0f}")
        print(f"Median counts per cell: {adata.obs['total_counts'].median():.0f}")
    
    # Filter cells based on QC metrics
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    if max_genes:
        adata = adata[adata.obs['n_genes_by_counts'] < max_genes, :]
        if verbose:
            print(f"Filtered cells with > {max_genes} genes: {adata.n_obs} cells remaining")
    
    if max_pct_mito and 'pct_counts_mt' in adata.obs.columns:
        adata = adata[adata.obs['pct_counts_mt'] < max_pct_mito, :]
        if verbose:
            print(f"Filtered cells with > {max_pct_mito}% mito: {adata.n_obs} cells remaining")
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if verbose:
        print(f"\nAfter QC filtering:")
        print(f"  Cells: {adata.n_obs}")
        print(f"  Genes: {adata.n_vars}")
    
    # Optional: Select highly variable genes to reduce size
    if n_top_genes and n_top_genes < adata.n_vars:
        if verbose:
            print(f"\nSelecting {n_top_genes} highly variable genes...")
        
        # Normalize and log transform for HVG selection
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')
        
        # Subset to HVGs
        adata = adata[:, adata.var['highly_variable']]
        
        if verbose:
            print(f"Subsetted to {adata.n_vars} highly variable genes")
    
    # Save to cache if requested
    if cache_file:
        if verbose:
            print(f"\nSaving to cache: {cache_file}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        adata.write_h5ad(cache_file)
        
        if verbose:
            cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"Cache file size: {cache_size_mb:.2f} MB")
    
    if verbose:
        print(f"\nFinal dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    return adata


# Example usage
if __name__ == "__main__":
    # First, get dataset info without loading
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    get_dataset_info()
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Loading a VERY small subset for testing (100 cells)")
    print("=" * 80)
    adata_tiny = load_covid19_data(
        subset_cells=100,  # Just 100 cells for ultra-fast testing
        min_genes=50,      # Lower threshold for testing
        min_cells=1,
        cache_file="/labs/Aguiar/SSPA_BRAY/BRay/cache/covid19_test_100.h5ad"
    )
    print(f"\nTiny test dataset shape: {adata_tiny.shape}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZED LOADING METHODS FOR LARGE DATASETS")
    print("=" * 80)
    
    print("\n" + "=" * 60)
    print("METHOD 1: Backed Mode (Recommended for exploration)")
    print("=" * 60)
    print("""
    # Load in backed mode - keeps data on disk, minimal memory usage
    adata_backed = load_covid19_backed(
        subset_cells=100000,  # 100k cells for testing
        min_genes=200,
        min_cells=3,
        max_pct_mito=20,
        cache_file="/labs/Aguiar/SSPA_BRAY/BRay/cache/covid19_100k_backed.h5ad"
    )
    """)
    
    print("\n" + "=" * 60)
    print("METHOD 2: Chunked Processing (For full dataset processing)")
    print("=" * 60)
    print("""
    # Process in chunks - most memory efficient for full dataset
    adata_chunked = load_covid19_chunked(
        chunk_size=25000,     # Process 25k cells at a time
        max_cells=200000,     # Limit for testing (remove for full dataset)
        min_genes=200,
        min_cells=3,
        max_pct_mito=20,
        cache_file="/labs/Aguiar/SSPA_BRAY/BRay/cache/covid19_200k_chunked.h5ad"
    )
    """)
    
    print("\n" + "=" * 60)
    print("METHOD 3: Original Method (For smaller subsets)")
    print("=" * 60)
    print("""
    # Original method - good for smaller subsets
    adata_original = load_covid19_data(
        subset_cells=50000,
        min_genes=200,
        min_cells=3,
        max_pct_mito=20,
        n_top_genes=2000,  # Select only 2k highly variable genes
        cache_file="/labs/Aguiar/SSPA_BRAY/BRay/cache/covid19_50k_hvg2k.h5ad"
    )
    """)
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR 1.4M CELL DATASET")
    print("=" * 60)
    print("""
    # For exploration and analysis:
    adata_explore = load_covid19_backed(
        subset_cells=500000,  # Start with 500k cells
        cache_file="/labs/Aguiar/SSPA_BRAY/BRay/cache/covid19_500k_backed.h5ad"
    )
    
    # For full dataset processing:
    adata_full = load_covid19_chunked(
        chunk_size=50000,     # 50k cells per chunk
        min_genes=200,
        min_cells=3,
        max_pct_mito=20,
        cache_file="/labs/Aguiar/SSPA_BRAY/BRay/cache/covid19_full_processed.h5ad"
    )
    """)