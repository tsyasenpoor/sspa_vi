"""
EMTAB11349 Data Loader for Variational Inference
=================================================

Preprocessed CSV-based data loader for EMTAB11349 dataset.
Handles gene expression, labels (IBD), and auxiliary features.

Usage:
    from VariationalInference.emtab_data_loader import EMTAB11349DataLoader

    loader = EMTAB11349DataLoader(
        data_dir='/path/to/EMTAB11349/preprocessed',
        gene_annotation_path='/path/to/gene_annotation.csv'
    )

    data = loader.load_and_preprocess(
        label_column='IBD',
        aux_columns=['sex_female'],
        min_cells_expressing=0.02
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
from sklearn.model_selection import train_test_split


class EMTAB11349DataLoader:
    """
    Data loader for preprocessed EMTAB11349 CSV files.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing preprocessed CSV files:
        - gene_expression_raw_processed.csv.gz
        - responses.csv.gz
        - aux_data.csv.gz
    gene_annotation_path : str or Path
        Path to gene annotation CSV for protein-coding filtering.
        Expected columns: 'GeneID' (Ensembl), 'Genetype'.
    species : str, default='human'
        Species for gene ID conversion.
    verbose : bool, default=True
        Whether to print progress messages.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        gene_annotation_path: Union[str, Path],
        species: str = 'human',
        verbose: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.gene_annotation_path = Path(gene_annotation_path)
        self.species = species
        self.verbose = verbose
        
        # Data containers
        self.gene_expression = None
        self.responses = None
        self.aux_data = None
        self.raw_df = None
        self.gene_list = None
        self.cell_ids = None
        self._gene_converter = None
        
    def _log(self, message: str):
        """Print if verbose."""
        if self.verbose:
            print(message)
    
    def _get_gene_converter(self):
        """Get gene ID converter."""
        if self._gene_converter is None:
            from .gene_convertor import GeneIDConverter
            cache_file = self.data_dir / 'gene_id_cache.json'
            self._gene_converter = GeneIDConverter(cache_file=str(cache_file))
        return self._gene_converter
    
    def load_raw_files(self):
        """Load all CSV files."""
        self._log("Loading preprocessed CSV files...")
        
        gene_expr_path = self.data_dir / 'gene_expression_raw_processed.csv.gz'
        responses_path = self.data_dir / 'responses.csv.gz'
        aux_path = self.data_dir / 'aux_data.csv.gz'
        
        self.gene_expression = pd.read_csv(gene_expr_path, compression='gzip')
        self.responses = pd.read_csv(responses_path, compression='gzip')
        self.aux_data = pd.read_csv(aux_path, compression='gzip')
        
        self._log(f"  Gene expression: {self.gene_expression.shape}")
        self._log(f"  Responses: {self.responses.shape}")
        self._log(f"  Auxiliary: {self.aux_data.shape}")
        
        # Extract expression matrix (drop Sample_ID)
        self.raw_df = self.gene_expression.drop(columns=['Sample_ID']).copy()
        self.cell_ids = self.gene_expression['Sample_ID'].tolist()
        self.gene_list = self.raw_df.columns.tolist()
        
        return self
    
    def convert_genes_to_ensembl(self) -> pd.DataFrame:
        """Convert gene symbols to Ensembl IDs."""
        if self.raw_df is None:
            raise ValueError("Must load raw files first")
        
        gene_names = self.raw_df.columns.tolist()
        converter = self._get_gene_converter()
        
        self._log("Converting gene symbols to Ensembl IDs...")
        ensembl_map, _ = converter.symbols_to_ensembl(gene_names, species=self.species)
        
        ensembl_names = [ensembl_map.get(gene, gene) for gene in gene_names]
        self.raw_df.columns = ensembl_names
        
        return self.raw_df
    
    def remove_duplicate_genes(self) -> pd.DataFrame:
        """Remove duplicate gene columns."""
        if self.raw_df is None:
            raise ValueError("Must load raw files first")
        
        n_before = self.raw_df.shape[1]
        mask = ~self.raw_df.columns.duplicated(keep='first')
        self.raw_df = self.raw_df.loc[:, mask]
        n_after = self.raw_df.shape[1]
        
        self._log(f"Removed {n_before - n_after} duplicate genes: {n_before} -> {n_after}")
        return self.raw_df
    
    def filter_protein_coding(self) -> pd.DataFrame:
        """Filter for protein-coding genes only."""
        if self.raw_df is None:
            raise ValueError("Must load raw files first")
        
        self._log(f"Loading gene annotation from {self.gene_annotation_path}...")
        gene_annotation = pd.read_csv(self.gene_annotation_path)
        
        if 'Genename' in gene_annotation.columns:
            gene_annotation = gene_annotation.set_index('Genename')
        
        self._log(f"Loaded annotation for {len(gene_annotation)} genes")
        
        # Convert annotation to Ensembl
        gene_names = gene_annotation.index.tolist()
        converter = self._get_gene_converter()
        gene_annotation_ensembl_map, _ = converter.symbols_to_ensembl(gene_names, species=self.species)
        gene_annotation.index = [gene_annotation_ensembl_map.get(g, g) for g in gene_annotation.index]
        
        self._log("Filtering for protein-coding genes...")
        type_col = 'Genetype' if 'Genetype' in gene_annotation.columns else 'gene_type'
        
        protein_coding_mask = gene_annotation[type_col] == 'protein_coding'
        protein_coding_genes = set(gene_annotation[protein_coding_mask].index.tolist())
        genes_in_data = set(self.raw_df.columns)
        
        n_before = len(genes_in_data)
        protein_coding_in_data = genes_in_data & protein_coding_genes
        self.raw_df = self.raw_df[list(protein_coding_in_data)]
        
        self._log(f"Filtered to {len(protein_coding_in_data)} protein-coding genes (from {n_before})")
        return self.raw_df
    
    def filter_zero_genes(self) -> pd.DataFrame:
        """Remove genes with zero expression across all cells."""
        if self.raw_df is None:
            raise ValueError("Must load raw files first")
        
        n_before = self.raw_df.shape[1]
        self.raw_df = self.raw_df.loc[:, (self.raw_df != 0).any(axis=0)]
        n_after = self.raw_df.shape[1]
        
        self._log(f"Removed {n_before - n_after} all-zero genes: {n_before} -> {n_after}")
        return self.raw_df
    
    def filter_min_cells(self, min_fraction: float = 0.02) -> pd.DataFrame:
        """Keep genes expressed in at least min_fraction of cells."""
        if self.raw_df is None:
            raise ValueError("Must load raw files first")
        
        min_cells = int(min_fraction * self.raw_df.shape[0])
        genes_to_keep = (self.raw_df > 0).sum(axis=0) >= min_cells
        
        n_before = self.raw_df.shape[1]
        self.raw_df = self.raw_df.loc[:, genes_to_keep]
        n_after = self.raw_df.shape[1]
        
        self._log(f"Kept genes in >={min_cells} cells ({min_fraction*100:.1f}%): {n_before} -> {n_after}")
        return self.raw_df
    
    def preprocess(
        self,
        convert_to_ensembl: bool = True,
        filter_protein_coding: bool = True,
        min_cells_expressing: float = 0.02
    ) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.
        
        Parameters
        ----------
        convert_to_ensembl : bool, default=True
            Convert gene symbols to Ensembl IDs.
        filter_protein_coding : bool, default=True
            Filter for protein-coding genes.
        min_cells_expressing : float, default=0.02
            Minimum fraction of cells expressing each gene.
            
        Returns
        -------
        pd.DataFrame
            Preprocessed expression matrix.
        """
        self.load_raw_files()
        
        if convert_to_ensembl:
            self.convert_genes_to_ensembl()
        
        self.remove_duplicate_genes()
        
        if filter_protein_coding:
            self.filter_protein_coding()
        
        self.filter_zero_genes()
        self.filter_min_cells(min_cells_expressing)
        
        self.gene_list = self.raw_df.columns.tolist()
        
        self._log(f"Final shape: {self.raw_df.shape[0]} cells x {self.raw_df.shape[1]} genes")
        return self.raw_df
    
    def get_labels(self, label_column: str) -> np.ndarray:
        """
        Extract labels from responses dataframe.
        
        Parameters
        ----------
        label_column : str
            Column name in responses.csv.gz (e.g., 'IBD').
            
        Returns
        -------
        np.ndarray
            Label array.
        """
        if self.responses is None:
            raise ValueError("Must load raw files first")
        
        if label_column not in self.responses.columns:
            raise ValueError(f"Label column '{label_column}' not found. Available: {list(self.responses.columns)}")
        
        return self.responses[label_column].values
    
    def get_auxiliary_features(self, aux_columns: List[str]) -> np.ndarray:
        """
        Extract auxiliary features from aux_data dataframe.
        
        Parameters
        ----------
        aux_columns : list of str
            Column names in aux_data.csv.gz (e.g., ['sex_female']).
            
        Returns
        -------
        np.ndarray
            Auxiliary feature matrix (n_samples, n_features).
        """
        if self.aux_data is None:
            raise ValueError("Must load raw files first")
        
        missing = [col for col in aux_columns if col not in self.aux_data.columns]
        if missing:
            raise ValueError(f"Auxiliary columns not found: {missing}. Available: {list(self.aux_data.columns)}")
        
        return self.aux_data[aux_columns].values
    
    def split_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stratify_by: Optional[str] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Split indices into train/val/test sets.
        
        Parameters
        ----------
        train_ratio : float, default=0.7
            Fraction for training.
        val_ratio : float, default=0.15
            Fraction for validation.
        stratify_by : str, optional
            Label column to stratify by.
        random_state : int, optional
            Random seed.
            
        Returns
        -------
        dict
            'train', 'val', 'test' index arrays.
        """
        n_samples = self.raw_df.shape[0]
        indices = np.arange(n_samples)
        
        stratify = None
        if stratify_by:
            stratify = self.get_labels(stratify_by)
        
        # Train + val vs test
        test_ratio = 1.0 - train_ratio - val_ratio
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            stratify=stratify,
            random_state=random_state
        )
        
        # Train vs val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        stratify_train_val = stratify[train_val_idx] if stratify is not None else None
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio_adjusted,
            stratify=stratify_train_val,
            random_state=random_state
        )
        
        self._log(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    
    def load_and_preprocess(
        self,
        label_column: str = 'IBD',
        aux_columns: Optional[List[str]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stratify_by: Optional[str] = None,
        min_cells_expressing: float = 0.02,
        convert_to_ensembl: bool = True,
        filter_protein_coding: bool = True,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full pipeline: load, preprocess, split.
        
        Parameters
        ----------
        label_column : str, default='IBD'
            Column for labels in responses.csv.gz.
        aux_columns : list of str, optional
            Columns for auxiliary features in aux_data.csv.gz (e.g., ['sex_female']).
        train_ratio : float, default=0.7
            Training fraction.
        val_ratio : float, default=0.15
            Validation fraction.
        stratify_by : str, optional
            Label column to stratify by (default: same as label_column).
        min_cells_expressing : float, default=0.02
            Minimum cell fraction per gene.
        convert_to_ensembl : bool, default=True
            Convert gene symbols to Ensembl IDs.
        filter_protein_coding : bool, default=True
            Filter for protein-coding genes.
        random_state : int, optional
            Random seed.
            
        Returns
        -------
        dict
            Keys: 'train', 'val', 'test', 'genes', 'n_genes', 'n_samples', 'n_aux_features'
            Each split: (X, y, X_aux) where X=expression, y=labels, X_aux=auxiliary
        """
        # Preprocess
        self.preprocess(
            convert_to_ensembl=convert_to_ensembl,
            filter_protein_coding=filter_protein_coding,
            min_cells_expressing=min_cells_expressing
        )
        
        # Get labels and aux features
        y = self.get_labels(label_column)
        X_aux = self.get_auxiliary_features(aux_columns) if aux_columns else None
        X = self.raw_df.values
        
        # Split
        if stratify_by is None:
            stratify_by = label_column
        
        splits = self.split_data(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            stratify_by=stratify_by,
            random_state=random_state
        )
        
        # Package results
        result = {
            'train': (X[splits['train']], y[splits['train']], X_aux[splits['train']] if X_aux is not None else None),
            'val': (X[splits['val']], y[splits['val']], X_aux[splits['val']] if X_aux is not None else None),
            'test': (X[splits['test']], y[splits['test']], X_aux[splits['test']] if X_aux is not None else None),
            'genes': self.gene_list,
            'n_genes': len(self.gene_list),
            'n_samples': X.shape[0],
            'n_aux_features': X_aux.shape[1] if X_aux is not None else 0
        }
        
        self._log(f"\nLabel distribution:")
        self._log(f"  Train: {np.bincount(y[splits['train']])}")
        self._log(f"  Val:   {np.bincount(y[splits['val']])}")
        self._log(f"  Test:  {np.bincount(y[splits['test']])}")
        
        return result


def load_emtab11349_data(
    data_dir: str,
    gene_annotation_path: str,
    label_column: str = 'IBD',
    aux_columns: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for EMTAB11349 data loading.
    
    Parameters
    ----------
    data_dir : str
        Directory with preprocessed CSV files.
    gene_annotation_path : str
        Path to gene annotation CSV.
    label_column : str, default='IBD'
        Label column name.
    aux_columns : list of str, optional
        Auxiliary feature columns (e.g., ['sex_female']).
    **kwargs
        Additional arguments for load_and_preprocess().
        
    Returns
    -------
    dict
        Preprocessed data dictionary.
        
    Examples
    --------
    >>> data = load_emtab11349_data(
    ...     data_dir='/path/to/EMTAB11349/preprocessed',
    ...     gene_annotation_path='/path/to/ENS_mouse_geneannotation.csv',
    ...     label_column='IBD',
    ...     aux_columns=['sex_female']
    ... )
    >>> X_train, y_train, X_aux_train = data['train']
    """
    loader = EMTAB11349DataLoader(
        data_dir=data_dir,
        gene_annotation_path=gene_annotation_path,
        verbose=kwargs.pop('verbose', True)
    )
    
    return loader.load_and_preprocess(
        label_column=label_column,
        aux_columns=aux_columns,
        **kwargs
    )
