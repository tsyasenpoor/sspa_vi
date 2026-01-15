"""
Data Loader for Variational Inference
======================================

This module provides generic data loading and preprocessing utilities
for single-cell RNA-seq data in multiple formats.

Supported Formats:
- h5ad files (AnnData format)
- EMTAB CSV directories (preprocessed EMTAB11349 format)
- Simulated CSV files

Features:
- Automatic format detection based on path
- Gene symbol to Ensembl ID conversion
- Protein-coding gene filtering
- Quality control and preprocessing
- Dynamic train/val/test splitting (regenerated each run)
- Auxiliary feature encoding

Usage:
    from VariationalInference.data_loader import DataLoader

    # Load h5ad file
    loader = DataLoader(
        data_path='/path/to/data.h5ad',
        gene_annotation_path='/path/to/gene_annotation.csv'  # optional
    )

    # Load EMTAB data (auto-detected from directory structure)
    loader = DataLoader(
        data_path='/path/to/EMTAB11349/preprocessed',
        gene_annotation_path='/path/to/gene_annotation.csv'
    )

    # Get preprocessed data with random splits
    data = loader.load_and_preprocess(
        label_column='t2dm',  # or 'IBD' for EMTAB
        aux_columns=['Sex'],  # or ['sex_female'] for EMTAB
        min_cells_expressing=0.02
    )

    # Access splits
    X_train, y_train, X_aux_train = data['train']
    X_val, y_val, X_aux_val = data['val']
    X_test, y_test, X_aux_test = data['test']
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
from sklearn.model_selection import train_test_split
import hashlib
import pickle
import tempfile
import os


class DataLoader:
    """
    Generic data loader for single-cell h5ad files and EMTAB CSV directories.

    Parameters
    ----------
    data_path : str or Path
        Path to the h5ad file to load, or directory containing EMTAB CSV files.
        For EMTAB format, directory should contain:
        - gene_expression_raw_processed.csv.gz
        - responses.csv.gz
        - aux_data.csv.gz
    gene_annotation_path : str or Path, optional
        Path to gene annotation CSV for protein-coding filtering.
        Expected columns: 'GeneID' (Ensembl), 'Genetype'.
    cache_dir : str or Path, optional
        Directory for caching preprocessed data. Defaults to system temp.
    species : str, default='human'
        Species for gene ID conversion. Options: 'human', 'mouse'.
    use_cache : bool, default=True
        Whether to cache preprocessed data for faster reloading.
    verbose : bool, default=True
        Whether to print progress messages.
    """

    # EMTAB expected files
    EMTAB_REQUIRED_FILES = [
        'gene_expression_raw_processed.csv.gz',
        'responses.csv.gz',
        'aux_data.csv.gz'
    ]

    def __init__(
        self,
        data_path: Union[str, Path],
        gene_annotation_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        species: str = 'human',
        use_cache: bool = True,
        verbose: bool = True
    ):
        self.data_path = Path(data_path)
        self.gene_annotation_path = Path(gene_annotation_path) if gene_annotation_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / 'sspa_vi_cache'
        self.species = species
        self.use_cache = use_cache
        self.verbose = verbose

        # Detect data type from path
        self.is_emtab = self._detect_emtab_format()
        self.is_singscore = 'singscore' in str(self.data_path).lower() and not self.is_emtab
        self.is_simulated = 'simulated' in str(self.data_path).lower() and not self.is_emtab
        self.feature_type = 'pathway' if self.is_singscore else 'gene'

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated after loading
        self.adata = None
        self.raw_df = None
        self.gene_list = None
        self.cell_ids = None
        self._gene_converter = None

        # EMTAB-specific data containers
        self.responses_df = None
        self.aux_data_df = None

    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)

    def _detect_emtab_format(self) -> bool:
        """
        Detect if data_path is an EMTAB directory with required CSV files.

        Returns
        -------
        bool
            True if EMTAB format detected, False otherwise.
        """
        if not self.data_path.is_dir():
            return False

        # Check if all required files exist
        for required_file in self.EMTAB_REQUIRED_FILES:
            if not (self.data_path / required_file).exists():
                return False

        return True

    def _get_cache_key(self) -> str:
        """Generate cache key based on data file."""
        file_stat = self.data_path.stat()
        key_str = f"{self.data_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _get_gene_converter(self):
        """Get or create gene ID converter with cache in cache_dir."""
        if self._gene_converter is None:
            from .gene_convertor import GeneIDConverter
            cache_file = self.cache_dir / 'gene_id_cache.json'
            self._gene_converter = GeneIDConverter(cache_file=str(cache_file))
        return self._gene_converter

    def load_adata(self, layer: str = 'raw') -> 'AnnData':
        """
        Load AnnData object from h5ad file.

        Parameters
        ----------
        layer : str, default='raw'
            Which layer to use for expression data.

        Returns
        -------
        AnnData
            Loaded AnnData object.
        """
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("scanpy is required for loading h5ad files: pip install scanpy")

        self._log(f"Loading h5ad file: {self.data_path}")
        self.adata = sc.read_h5ad(str(self.data_path).strip())
        self._log(f"  Data type: {self.feature_type} (singscore={self.is_singscore})")
        self._log(f"  Shape: {self.adata.n_obs} cells x {self.adata.n_vars} {self.feature_type}s")
        self._log(f"  Obs columns: {list(self.adata.obs.columns)}")

        return self.adata

    def load_emtab_files(self) -> pd.DataFrame:
        """
        Load EMTAB preprocessed CSV files from directory.

        Loads:
        - gene_expression_raw_processed.csv.gz -> self.raw_df
        - responses.csv.gz -> self.responses_df
        - aux_data.csv.gz -> self.aux_data_df

        Returns
        -------
        pd.DataFrame
            Raw expression DataFrame.
        """
        self._log(f"Loading EMTAB data from directory: {self.data_path}")

        gene_expr_path = self.data_path / 'gene_expression_raw_processed.csv.gz'
        responses_path = self.data_path / 'responses.csv.gz'
        aux_path = self.data_path / 'aux_data.csv.gz'

        # Load all CSV files
        gene_expression = pd.read_csv(gene_expr_path, compression='gzip')
        self.responses_df = pd.read_csv(responses_path, compression='gzip')
        self.aux_data_df = pd.read_csv(aux_path, compression='gzip')

        self._log(f"  Gene expression: {gene_expression.shape}")
        self._log(f"  Responses: {self.responses_df.shape}")
        self._log(f"  Auxiliary: {self.aux_data_df.shape}")

        # Extract expression matrix (drop Sample_ID column)
        if 'Sample_ID' in gene_expression.columns:
            self.cell_ids = gene_expression['Sample_ID'].tolist()
            self.raw_df = gene_expression.drop(columns=['Sample_ID']).copy()
            self.raw_df.index = self.cell_ids
        else:
            self.cell_ids = gene_expression.index.tolist()
            self.raw_df = gene_expression.copy()

        self.gene_list = self.raw_df.columns.tolist()

        return self.raw_df

    def load_simulated_csv(self) -> pd.DataFrame:
        """
        Load simulated data from CSV (simulated_gex_t2dm_groundtruth.csv format).
        
        Expected format:
        - Rows: samples/cells
        - Columns: genes (Ensembl IDs) + 't2dm' label column
        - No preprocessing needed (already clean)
        
        Returns
        -------
        pd.DataFrame
            Raw expression DataFrame with label column.
        """
        self._log(f"Loading simulated CSV: {self.data_path}")
        
        # Load CSV
        df = pd.read_csv(self.data_path, index_col=0)
        
        # Check for t2dm column
        if 'covid' not in df.columns:
            raise ValueError(f"Simulated CSV must contain 'covid' label column. Found: {list(df.columns)}")
        
        # Separate labels from expression data
        self.label_data = df['covid'].copy()
        self.raw_df = df.drop(columns=['covid'])
        
        # Store metadata
        self.gene_list = self.raw_df.columns.tolist()
        self.cell_ids = self.raw_df.index.tolist()
        
        self._log(f"  Shape: {len(self.cell_ids)} cells x {len(self.gene_list)} genes")
        self._log(f"  Label distribution: {self.label_data.value_counts().to_dict()}")
        
        return self.raw_df

    def get_expression_df(self, layer: str = 'raw') -> pd.DataFrame:
        """
        Get expression data as a DataFrame.

        Parameters
        ----------
        layer : str, default='raw'
            Which layer to use. Use None for adata.X.

        Returns
        -------
        pd.DataFrame
            Expression DataFrame with cells as rows, genes/pathways as columns.
        """
        if self.adata is None:
            self.load_adata()

        if layer is not None and layer in self.adata.layers:
            self._log(f"Using layer: {layer}")
            self.raw_df = self.adata.to_df(layer=layer)
        else:
            self._log("Using adata.X")
            self.raw_df = self.adata.to_df()

        # Rescale singscore data to be all positive
        if self.is_singscore:
            min_val = self.raw_df.min().min()
            if min_val < 0:
                self._log(f"Rescaling singscore data: min value = {min_val:.4f}")
                self.raw_df = self.raw_df - min_val
                self._log(f"  After rescaling: min = {self.raw_df.min().min():.4f}, max = {self.raw_df.max().max():.4f}")

        return self.raw_df

    def convert_genes_to_ensembl(self) -> pd.DataFrame:
        """
        Convert gene names to Ensembl IDs.
        Skipped for singscore data (pathways).

        Returns
        -------
        pd.DataFrame
            Expression DataFrame with Ensembl ID columns (or unchanged for pathways).
        """
        if self.raw_df is None:
            self.get_expression_df()

        # Skip conversion for singscore/pathway data
        if self.is_singscore:
            self._log("Skipping Ensembl conversion for pathway data")
            return self.raw_df

        gene_names = self.raw_df.columns.tolist()
        converter = self._get_gene_converter()

        self._log("Converting gene symbols to Ensembl IDs...")
        ensembl_map, _ = converter.symbols_to_ensembl(gene_names, species=self.species)

        # Apply mapping (keep original name if no mapping found)
        ensembl_names = [ensembl_map.get(gene, gene) for gene in gene_names]
        self.raw_df.columns = ensembl_names

        return self.raw_df

    def remove_duplicate_genes(self) -> pd.DataFrame:
        """
        Remove duplicate gene/pathway columns (keep first occurrence).

        Returns
        -------
        pd.DataFrame
            Expression DataFrame without duplicates.
        """
        if self.raw_df is None:
            raise ValueError("Must load expression data first")

        n_before = self.raw_df.shape[1]
        mask = ~self.raw_df.columns.duplicated(keep='first')
        self.raw_df = self.raw_df.loc[:, mask]
        n_after = self.raw_df.shape[1]

        self._log(f"Removed {n_before - n_after} duplicate {self.feature_type}s: {n_before} -> {n_after}")
        return self.raw_df

    def filter_protein_coding(self) -> pd.DataFrame:
        """
        Filter to keep only protein-coding genes.
        Skipped for singscore data (pathways).

        Requires gene_annotation_path to be set.
        Uses the gene annotation file structure with 'Genename', 'GeneID', and 'Genetype' columns.

        Returns
        -------
        pd.DataFrame
            Expression DataFrame with only protein-coding genes (or unchanged for pathways).
        """
        # Skip protein-coding filter for pathway data
        if self.is_singscore:
            self._log("Skipping protein-coding filter for pathway data")
            return self.raw_df

        if self.gene_annotation_path is None:
            self._log("No gene annotation file provided - skipping protein-coding filter")
            return self.raw_df

        if self.raw_df is None:
            raise ValueError("Must load expression data first")

        self._log(f"Loading gene annotation from {self.gene_annotation_path}...")
        gene_annotation = pd.read_csv(self.gene_annotation_path)
        
        # Set Genename as index if it exists
        if 'Genename' in gene_annotation.columns:
            gene_annotation = gene_annotation.set_index('Genename')
        
        self._log(f"Loaded annotation for {len(gene_annotation)} genes")
        
        # Convert annotation gene names to Ensembl IDs if needed (index)
        gene_names = gene_annotation.index.tolist()
        converter = self._get_gene_converter()
        gene_annotation_ensembl_map, _ = converter.symbols_to_ensembl(gene_names, species=self.species)
        gene_annotation.index = [gene_annotation_ensembl_map.get(g, g) for g in gene_annotation.index]

        # Also convert the 'GeneID' column to Ensembl IDs if it exists
        geneid_col = 'GeneID' if 'GeneID' in gene_annotation.columns else 'gene_id'
        if geneid_col in gene_annotation.columns:
            geneid_ensembl_map, _ = converter.symbols_to_ensembl(gene_annotation[geneid_col].tolist(), species=self.species)
            gene_annotation[geneid_col] = [geneid_ensembl_map.get(g, g) for g in gene_annotation[geneid_col]]

        self._log("Filtering for protein-coding genes...")
        type_col = 'Genetype' if 'Genetype' in gene_annotation.columns else 'gene_type'

        # Use the converted index (which matches the data gene format) instead of the GeneID column
        # This ensures both data and annotation genes are in the same format (e.g., human Ensembl IDs)
        protein_coding_mask = gene_annotation[type_col] == 'protein_coding'
        protein_coding_genes = set(gene_annotation[protein_coding_mask].index.tolist())
        all_annotation_genes = set(gene_annotation.index.tolist())
        genes_in_data = set(self.raw_df.columns)
        n_before = len(genes_in_data)
        genes_not_in_annotation = genes_in_data - all_annotation_genes
        if genes_not_in_annotation:
            self._log(f"WARNING: {len(genes_not_in_annotation)} genes in data are not in annotation - removing them")
            self._log(f"  Examples: {list(genes_not_in_annotation)[:10]}")
        genes_in_annotation = genes_in_data & all_annotation_genes
        non_protein_coding_in_data = genes_in_annotation - protein_coding_genes
        if non_protein_coding_in_data:
            self._log(f"Removing {len(non_protein_coding_in_data)} non-protein-coding genes")
            examples = list(non_protein_coding_in_data)[:10]
            for gene_id in examples:
                if gene_id in gene_annotation.index:
                    gene_type = gene_annotation.loc[gene_id, type_col]
                    self._log(f"  {gene_id}: {gene_type}")
        protein_coding_in_data = genes_in_data & protein_coding_genes
        self.raw_df = self.raw_df[list(protein_coding_in_data)]
        self._log(f"Filtered to {len(protein_coding_in_data)} protein-coding genes (from {n_before})")
        self._log(f"  Removed {len(genes_not_in_annotation)} genes not in annotation")
        self._log(f"  Removed {len(non_protein_coding_in_data)} non-protein-coding genes")
        return self.raw_df

    def filter_zero_genes(self) -> pd.DataFrame:
        """
        Remove genes/pathways with zero expression across all cells.

        Returns
        -------
        pd.DataFrame
            Expression DataFrame without all-zero genes/pathways.
        """
        if self.raw_df is None:
            raise ValueError("Must load expression data first")

        n_before = self.raw_df.shape[1]
        self.raw_df = self.raw_df.loc[:, (self.raw_df != 0).any(axis=0)]
        n_after = self.raw_df.shape[1]

        self._log(f"Removed {n_before - n_after} all-zero {self.feature_type}s: {n_before} -> {n_after}")
        return self.raw_df

    def filter_min_cells(self, min_fraction: float = 0.02) -> pd.DataFrame:
        """
        Keep only genes/pathways expressed in at least min_fraction of cells.

        Parameters
        ----------
        min_fraction : float, default=0.02
            Minimum fraction of cells that must express the gene/pathway.

        Returns
        -------
        pd.DataFrame
            Expression DataFrame with filtered genes/pathways.
        """
        if self.raw_df is None:
            raise ValueError("Must load expression data first")

        min_cells = int(min_fraction * self.raw_df.shape[0])
        genes_to_keep = (self.raw_df > 0).sum(axis=0) >= min_cells

        n_before = self.raw_df.shape[1]
        self.raw_df = self.raw_df.loc[:, genes_to_keep]
        n_after = self.raw_df.shape[1]

        self._log(f"Kept {self.feature_type}s in >={min_cells} cells ({min_fraction*100:.1f}%): {n_before} -> {n_after}")
        return self.raw_df

    def normalize_counts(
        self,
        target_sum: float = 1e4,
        method: str = 'library_size'
    ) -> pd.DataFrame:
        """
        Normalize count data to reduce overdispersion.

        Parameters
        ----------
        target_sum : float, default=1e4
            Target library size (counts per cell will sum to ~this value).
        method : str, default='library_size'
            Normalization method:
            - 'library_size': Divide by cell total, multiply by target_sum.
            - 'median_ratio': DESeq2-style median-of-ratios.

        Returns
        -------
        pd.DataFrame
            Normalized count matrix with float values.
        """
        if self.raw_df is None:
            raise ValueError("Must load expression data first")

        X = self.raw_df.values.astype(np.float64)
        n_cells, n_genes = X.shape

        self._log(f"Normalizing counts (method={method}, target_sum={target_sum:.0f})")

        # Pre-normalization stats
        cell_sums_before = X.sum(axis=1)
        var_before = X.var()
        mean_before = X.mean()
        vmi_before = var_before / mean_before if mean_before > 0 else np.nan
        self._log(f"  Before: max={X.max():.0f}, mean_lib_size={cell_sums_before.mean():.0f}, var/mean={vmi_before:.1f}")

        if method == 'library_size':
            cell_sums = X.sum(axis=1, keepdims=True)
            cell_sums = np.maximum(cell_sums, 1)
            X_norm = (X / cell_sums) * target_sum

        elif method == 'median_ratio':
            log_X = np.log(X + 1)
            geo_mean = np.exp(log_X.mean(axis=0))
            geo_mean = np.maximum(geo_mean, 1e-10)
            ratios = X / geo_mean
            size_factors = np.median(ratios, axis=1, keepdims=True)
            size_factors = np.maximum(size_factors, 1e-10)
            X_norm = X / size_factors
            current_median_sum = np.median(X_norm.sum(axis=1))
            X_norm = X_norm * (target_sum / current_median_sum)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Keep normalized values as floats (no rounding)
        # Post-normalization stats
        cell_sums_after = X_norm.sum(axis=1)
        var_after = X_norm.var()
        mean_after = X_norm.mean()
        vmi_after = var_after / mean_after if mean_after > 0 else np.nan
        self._log(f"  After:  max={X_norm.max():.2f}, mean_lib_size={cell_sums_after.mean():.2f}, var/mean={vmi_after:.1f}")

        self.raw_df = pd.DataFrame(
            X_norm,
            index=self.raw_df.index,
            columns=self.raw_df.columns
        )

        return self.raw_df

    def preprocess(
        self,
        layer: str = 'raw',
        convert_to_ensembl: bool = True,
        filter_protein_coding: bool = True,
        min_cells_expressing: float = 0.02,
        normalize: bool = False,
        normalize_target_sum: float = 1e4,
        normalize_method: str = 'library_size'
    ) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.

        For singscore data (pathways), convert_to_ensembl and filter_protein_coding
        are automatically skipped regardless of parameter values.
        
        For simulated data (CSV), all preprocessing is skipped - data is used as-is.

        Parameters
        ----------
        layer : str, default='raw'
            Which layer to use for expression data.
        convert_to_ensembl : bool, default=True
            Whether to convert gene symbols to Ensembl IDs (ignored for pathways/simulated).
        filter_protein_coding : bool, default=True
            Whether to filter for protein-coding genes (ignored for pathways/simulated).
        min_cells_expressing : float, default=0.02
            Minimum fraction of cells expressing each gene/pathway.
        normalize : bool, default=False
            Whether to normalize counts (library size normalization + integer rounding).
        normalize_target_sum : float, default=1e4
            Target library size for normalization.
        normalize_method : str, default='library_size'
            Normalization method: 'library_size' or 'median_ratio'.

        Returns
        -------
        pd.DataFrame
            Preprocessed expression DataFrame.
        """
        # SIMULATED DATA: Load directly from CSV, no preprocessing
        if self.is_simulated:
            self._log("Detected simulated data - loading CSV directly (no preprocessing)")
            self.load_simulated_csv()
            self._log(f"Final shape: {self.raw_df.shape[0]} cells x {self.raw_df.shape[1]} genes")
            return self.raw_df

        # EMTAB DATA: Load from CSV directory with full preprocessing
        if self.is_emtab:
            # Check cache first
            if self.use_cache:
                cache_key = self._get_cache_key()
                cache_params = f"emtab_{convert_to_ensembl}_{filter_protein_coding}_{min_cells_expressing}_{normalize}_{normalize_target_sum}_{normalize_method}"
                cache_file = self.cache_dir / f"preprocessed_{cache_key}_{hashlib.md5(cache_params.encode()).hexdigest()[:8]}.pkl"

                if cache_file.exists():
                    self._log(f"Loading preprocessed EMTAB data from cache: {cache_file.name}")
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)
                        self.raw_df = cached['raw_df']
                        self.gene_list = cached['gene_list']
                        self.cell_ids = cached['cell_ids']
                        self.responses_df = cached.get('responses_df')
                        self.aux_data_df = cached.get('aux_data_df')
                    self._log(f"Loaded from cache: {self.raw_df.shape[0]} cells x {self.raw_df.shape[1]} genes")
                    return self.raw_df

            self._log("Detected EMTAB format - loading from CSV directory")
            self.load_emtab_files()

            if convert_to_ensembl:
                self.convert_genes_to_ensembl()

            self.remove_duplicate_genes()

            if filter_protein_coding:
                self.filter_protein_coding()

            self.filter_zero_genes()
            self.filter_min_cells(min_cells_expressing)

            if normalize:
                self.normalize_counts(
                    target_sum=normalize_target_sum,
                    method=normalize_method
                )

            self.gene_list = self.raw_df.columns.tolist()
            self._log(f"Final shape: {self.raw_df.shape[0]} cells x {self.raw_df.shape[1]} genes")

            # Save to cache
            if self.use_cache:
                self._log(f"Saving preprocessed EMTAB data to cache: {cache_file.name}")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'raw_df': self.raw_df,
                        'gene_list': self.gene_list,
                        'cell_ids': self.cell_ids,
                        'responses_df': self.responses_df,
                        'aux_data_df': self.aux_data_df
                    }, f)

            return self.raw_df

        # Override settings for singscore data
        if self.is_singscore:
            convert_to_ensembl = False
            filter_protein_coding = False

        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key()
            cache_params = f"{layer}_{convert_to_ensembl}_{filter_protein_coding}_{min_cells_expressing}_{self.is_singscore}"
            cache_file = self.cache_dir / f"preprocessed_{cache_key}_{hashlib.md5(cache_params.encode()).hexdigest()[:8]}.pkl"

            if cache_file.exists():
                self._log(f"Loading preprocessed data from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    self.raw_df = cached['raw_df']
                    self.gene_list = cached['gene_list']
                    self.cell_ids = cached['cell_ids']
                    self.adata = None  # Don't store full adata in cache
                return self.raw_df

        # Load data
        self.load_adata()
        self.get_expression_df(layer=layer)

        # Preprocessing steps
        if convert_to_ensembl:
            self.convert_genes_to_ensembl()

        self.remove_duplicate_genes()

        if filter_protein_coding:
            self.filter_protein_coding()

        self.filter_zero_genes()
        self.filter_min_cells(min_cells_expressing)

        if normalize:
            self.normalize_counts(
                target_sum=normalize_target_sum,
                method=normalize_method
            )

        # Store results
        self.gene_list = self.raw_df.columns.tolist()
        self.cell_ids = self.raw_df.index.tolist()

        self._log(f"Final shape: {self.raw_df.shape[0]} cells x {self.raw_df.shape[1]} {self.feature_type}s")

        # Save to cache
        if self.use_cache:
            self._log(f"Saving preprocessed data to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'raw_df': self.raw_df,
                    'gene_list': self.gene_list,
                    'cell_ids': self.cell_ids
                }, f)

        return self.raw_df

    def get_labels(self, label_column: str) -> np.ndarray:
        """
        Get labels from adata.obs, simulated data, or EMTAB responses.

        Parameters
        ----------
        label_column : str
            Column name in adata.obs or responses.csv.gz for labels.

        Returns
        -------
        np.ndarray
            Label array aligned with cell_ids.
        """
        # For simulated data, labels are already loaded
        if self.is_simulated:
            if not hasattr(self, 'label_data') or self.label_data is None:
                raise ValueError("Simulated data not loaded. Call preprocess() first.")
            return self.label_data.values.astype(int)

        # For EMTAB data, get labels from responses_df
        if self.is_emtab:
            if self.responses_df is None:
                raise ValueError("EMTAB data not loaded. Call preprocess() first.")
            if label_column not in self.responses_df.columns:
                raise ValueError(f"Label column '{label_column}' not found. Available: {list(self.responses_df.columns)}")
            return self.responses_df[label_column].values.astype(int)

        if self.adata is None:
            self.load_adata()

        if label_column not in self.adata.obs.columns:
            raise ValueError(f"Label column '{label_column}' not found. Available: {list(self.adata.obs.columns)}")

        labels = self.adata.obs.loc[self.cell_ids, label_column].values
        return labels.astype(int)

    def get_auxiliary_features(self, aux_columns: List[str]) -> np.ndarray:
        """
        Get auxiliary features from adata.obs or EMTAB aux_data.

        For simulated data, returns empty array (no aux features).

        Parameters
        ----------
        aux_columns : list of str
            Column names in adata.obs or aux_data.csv.gz for auxiliary features.

        Returns
        -------
        np.ndarray
            Auxiliary feature matrix (n_cells, n_aux).
        """
        # Simulated data has no auxiliary features
        if self.is_simulated:
            if not aux_columns:
                return np.zeros((len(self.cell_ids), 0))
            else:
                self._log(f"WARNING: Simulated data has no auxiliary features. Ignoring: {aux_columns}")
                return np.zeros((len(self.cell_ids), 0))

        # EMTAB data: get features from aux_data_df
        if self.is_emtab:
            if self.aux_data_df is None:
                raise ValueError("EMTAB data not loaded. Call preprocess() first.")

            if not aux_columns:
                return np.zeros((len(self.cell_ids), 0))

            missing = [col for col in aux_columns if col not in self.aux_data_df.columns]
            if missing:
                raise ValueError(f"Auxiliary columns not found: {missing}. Available: {list(self.aux_data_df.columns)}")

            return self.aux_data_df[aux_columns].values.astype(float)

        if self.adata is None:
            self.load_adata()

        if not aux_columns:
            return np.zeros((len(self.cell_ids), 0))

        aux_data = []
        for col in aux_columns:
            if col not in self.adata.obs.columns:
                raise ValueError(f"Auxiliary column '{col}' not found. Available: {list(self.adata.obs.columns)}")

            values = self.adata.obs.loc[self.cell_ids, col]

            # Encode categorical columns
            if values.dtype == 'object' or values.dtype.name == 'category':
                # One-hot encode (or dummy encode with drop_first)
                unique_vals = values.unique()
                if len(unique_vals) == 2:
                    # Binary: single column (0/1)
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    aux_data.append(values.map(mapping).values.reshape(-1, 1))
                else:
                    # Multi-class: one-hot encode
                    dummies = pd.get_dummies(values, prefix=col, drop_first=True)
                    aux_data.append(dummies.values)
            else:
                # Numeric: use directly
                aux_data.append(values.values.reshape(-1, 1))

        return np.hstack(aux_data).astype(float)

    def split_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stratify_by: Optional[str] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Create random train/val/test splits.

        Note: This generates NEW random splits each time (not saved).
        For reproducibility, pass a random_state.

        Parameters
        ----------
        train_ratio : float, default=0.7
            Proportion for training set.
        val_ratio : float, default=0.15
            Proportion for validation set.
        stratify_by : str, optional
            Column name in adata.obs for stratified splitting.
        random_state : int, optional
            Random seed for reproducibility. None = random.

        Returns
        -------
        dict
            Dictionary with 'train', 'val', 'test' keys containing cell ID lists.
        """
        if self.cell_ids is None:
            raise ValueError("Must preprocess data first")

        test_ratio = 1 - train_ratio - val_ratio

        # Get stratification labels if specified
        stratify = None
        if stratify_by is not None:
            # For simulated data, use pre-loaded labels
            if self.is_simulated:
                if not hasattr(self, 'label_data') or self.label_data is None:
                    raise ValueError("Simulated data not loaded. Call preprocess() first.")
                stratify = self.label_data.loc[self.cell_ids].values
            # For EMTAB data, use responses_df
            elif self.is_emtab:
                if self.responses_df is None:
                    raise ValueError("EMTAB data not loaded. Call preprocess() first.")
                stratify = self.responses_df[stratify_by].values
            else:
                if self.adata is None:
                    self.load_adata()
                stratify = self.adata.obs.loc[self.cell_ids, stratify_by].values

        # First split: train vs (val+test)
        train_ids, temp_ids = train_test_split(
            self.cell_ids,
            test_size=(val_ratio + test_ratio),
            stratify=stratify,
            random_state=random_state
        )

        # Second split: val vs test
        val_proportion = val_ratio / (val_ratio + test_ratio)
        if stratify is not None:
            # Need to subset stratify array for temp split
            temp_indices = [self.cell_ids.index(cid) for cid in temp_ids]
            temp_stratify = stratify[temp_indices]
        else:
            temp_stratify = None

        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1 - val_proportion),
            stratify=temp_stratify,
            random_state=random_state
        )

        splits = {
            'train': list(train_ids),
            'val': list(val_ids),
            'test': list(test_ids)
        }

        self._log(f"Split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        return splits

    def get_matrices(
        self,
        cell_ids: List[str],
        label_column: str,
        aux_columns: Optional[List[str]] = None,
        return_sparse: bool = True
    ) -> Tuple[Union[np.ndarray, sp.csr_matrix], np.ndarray, np.ndarray]:
        """
        Get X, X_aux, y matrices for given cell IDs.

        Parameters
        ----------
        cell_ids : list of str
            List of cell IDs to include.
        label_column : str
            Column name for labels.
        aux_columns : list of str, optional
            Column names for auxiliary features.
        return_sparse : bool, default=True
            If True, returns X as sparse CSR matrix. If False, returns dense array.

        Returns
        -------
        X : np.ndarray or scipy.sparse.csr_matrix
            Expression matrix (n_cells, n_genes). Sparse by default for efficiency.
        X_aux : np.ndarray
            Auxiliary feature matrix (n_cells, n_aux).
        y : np.ndarray
            Label array (n_cells,).
        """
        if self.raw_df is None:
            raise ValueError("Must preprocess data first")

        # Get expression matrix and convert to sparse if requested
        X = self.raw_df.loc[cell_ids].values
        if return_sparse:
            X = sp.csr_matrix(X)

        # Get labels (handles simulated vs EMTAB vs h5ad)
        if self.is_simulated:
            y = self.label_data.loc[cell_ids].values.astype(int)
        elif self.is_emtab:
            # For EMTAB, get labels from responses_df using cell indices
            cell_idx = [self.cell_ids.index(cid) for cid in cell_ids]
            y = self.responses_df[label_column].values[cell_idx].astype(int)
        else:
            if self.adata is None:
                self.load_adata()
            y = self.adata.obs.loc[cell_ids, label_column].values.astype(int)

        # Get auxiliary features
        if aux_columns and not self.is_simulated:
            X_aux = self.get_auxiliary_features(aux_columns)
            # Subset to requested cells
            cell_idx = [self.cell_ids.index(cid) for cid in cell_ids]
            X_aux = X_aux[cell_idx]
        else:
            X_aux = np.zeros((len(cell_ids), 0))

        # No intercept - gamma only models auxiliary variable effects
        # The model prediction is: theta @ v + X_aux @ gamma
        
        # Log sparsity statistics if returning sparse
        if return_sparse:
            sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))
            self._log(f"X matrix sparsity: {sparsity*100:.2f}% (saved memory)")
        
        return X, X_aux, y

    def load_and_preprocess(
        self,
        label_column: str = 't2dm',
        aux_columns: Optional[List[str]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stratify_by: Optional[str] = None,
        min_cells_expressing: float = 0.02,
        layer: str = 'raw',
        convert_to_ensembl: bool = True,
        filter_protein_coding: bool = True,
        random_state: Optional[int] = None,
        normalize: bool = False,
        normalize_target_sum: float = 1e4,
        normalize_method: str = 'library_size',
        return_sparse: bool = True
    ) -> Dict[str, Any]:
        """
        Complete data loading and preprocessing pipeline.

        Parameters
        ----------
        label_column : str, default='t2dm'
            Column name in adata.obs for labels.
        aux_columns : list of str, optional
            Column names for auxiliary features (e.g., ['Sex']).
        train_ratio : float, default=0.7
            Proportion for training set.
        val_ratio : float, default=0.15
            Proportion for validation set.
        stratify_by : str, optional
            Column for stratified splitting.
        min_cells_expressing : float, default=0.02
            Minimum fraction of cells expressing each gene.
        layer : str, default='raw'
            Which layer to use.
        convert_to_ensembl : bool, default=True
            Whether to convert gene symbols to Ensembl IDs.
        filter_protein_coding : bool, default=True
            Whether to filter for protein-coding genes.
        random_state : int, optional
            Random seed for splitting. None = random.
        normalize : bool, default=False
            Whether to normalize counts (library size + integer rounding).
        normalize_target_sum : float, default=1e4
            Target library size for normalization.
        normalize_method : str, default='library_size'
            Normalization method: 'library_size' or 'median_ratio'.
        return_sparse : bool, default=True
            If True, returns expression matrices as sparse CSR matrices for memory efficiency.

        Returns
        -------
        dict
            Dictionary with:
            - 'train': (X_train, X_aux_train, y_train)
            - 'val': (X_val, X_aux_val, y_val)
            - 'test': (X_test, X_aux_test, y_test)
            - 'gene_list': list of gene names
            - 'splits': dict of cell ID lists
            
        Note
        ----
        Gene expression data (X matrices) are returned as sparse matrices by default
        for memory efficiency, as scRNA-seq data is typically very sparse.
        """
        # Preprocess
        self.preprocess(
            layer=layer,
            convert_to_ensembl=convert_to_ensembl,
            filter_protein_coding=filter_protein_coding,
            min_cells_expressing=min_cells_expressing,
            normalize=normalize,
            normalize_target_sum=normalize_target_sum,
            normalize_method=normalize_method
        )

        # Split
        splits = self.split_data(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            stratify_by=stratify_by or label_column,
            random_state=random_state
        )

        # Get matrices for each split (sparse by default for memory efficiency)
        X_train, X_aux_train, y_train = self.get_matrices(
            splits['train'], label_column, aux_columns, return_sparse=return_sparse
        )
        X_val, X_aux_val, y_val = self.get_matrices(
            splits['val'], label_column, aux_columns, return_sparse=return_sparse
        )
        X_test, X_aux_test, y_test = self.get_matrices(
            splits['test'], label_column, aux_columns, return_sparse=return_sparse
        )

        return {
            'train': (X_train, X_aux_train, y_train),
            'val': (X_val, X_aux_val, y_val),
            'test': (X_test, X_aux_test, y_test),
            'gene_list': self.gene_list,
            'splits': splits,
            'n_genes': len(self.gene_list),
            'n_cells': len(self.cell_ids),
            'n_aux': X_aux_train.shape[1],
            'feature_type': self.feature_type,
            'is_singscore': self.is_singscore,
            'is_emtab': self.is_emtab
        }


def load_data(
    data_path: str,
    label_column: str = 't2dm',
    aux_columns: Optional[List[str]] = None,
    gene_annotation_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to load and preprocess data in one call.

    Supports multiple data formats:
    - h5ad files (AnnData format)
    - EMTAB directory with CSV files (gene_expression_raw_processed.csv.gz,
      responses.csv.gz, aux_data.csv.gz)
    - Simulated CSV files

    Parameters
    ----------
    data_path : str
        Path to h5ad file or EMTAB directory containing CSV files.
    label_column : str, default='t2dm'
        Column name for labels (in adata.obs or responses.csv.gz).
    aux_columns : list of str, optional
        Column names for auxiliary features (in adata.obs or aux_data.csv.gz).
    gene_annotation_path : str, optional
        Path to gene annotation CSV.
    **kwargs
        Additional arguments passed to load_and_preprocess().
        Key argument: return_sparse=True to return sparse matrices (default).

    Returns
    -------
    dict
        Preprocessed data dictionary with sparse expression matrices by default.

    Examples
    --------
    >>> # Load from h5ad with sparse matrices (default)
    >>> data = load_data(
    ...     '/path/to/Bcell_GEX.h5ad',
    ...     label_column='t2dm',
    ...     aux_columns=['Sex']
    ... )
    >>> X_train, X_aux_train, y_train = data['train']
    >>> # X_train is scipy.sparse.csr_matrix
    >>>
    >>> # Load with dense matrices
    >>> data = load_data(
    ...     '/path/to/Bcell_GEX.h5ad',
    ...     label_column='t2dm',
    ...     aux_columns=['Sex'],
    ...     return_sparse=False
    ... )
    >>>
    >>> # Load from EMTAB directory
    >>> data = load_data(
    ...     '/path/to/EMTAB11349/preprocessed',
    ...     label_column='IBD',
    ...     aux_columns=['sex_female']
    ... )
    """
    loader = DataLoader(
        data_path=data_path,
        gene_annotation_path=gene_annotation_path,
        verbose=kwargs.pop('verbose', True)
    )

    return loader.load_and_preprocess(
        label_column=label_column,
        aux_columns=aux_columns,
        **kwargs
    )
