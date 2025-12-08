"""
AJM Cytokine Dataset Preprocessing and Variational Inference Pipeline

This module prepares the AJM dataset for binary classification using variational inference.

WORKFLOW:
1. Load and preprocess AJM dataset (all protein-coding genes)
2. Calculate cytokine scores using cytokine seed genes (for cell selection only)
3. Identify training cells based on quartile filtering:
   - Group 1: Label 0 cells with cyto_score <= 75th percentile (TC-0hr)
   - Group 2: Label 1 cells with cyto_score >= 25th percentile (TC-LPS-24hr)
4. Calculate Cyto-Beam scores for all cells (for validation/analysis)
5. Split data:
   - Training: Full gene expression of cells in group 1 (label 0) + group 2 (label 1)
   - Test/Validation: Remaining cells from ALL timepoints (0hr, 3hr, 6hr, 24hr, 48hr, 72hr)
   
MODEL TRAINING:
- Input: Full gene expression matrix (all protein-coding genes, NOT restricted to cyto seeds)
- Output: Binary labels (0 or 1)
- Model learns to predict probability of each label using Bernoulli distribution
"""

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from typing import Tuple, Dict, List, Optional
import sys

sys.path.append('/labs/Aguiar/SSPA_BRAY/BRay/VariationalInference')

from gene_convertor import GeneIDConverter

# Initialize gene converter globally
gene_converter = GeneIDConverter()


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_cytokine_seed_genes(csv_path: str) -> Tuple[Dict, List]:
    """
    Load cytokine seed genes and convert to Ensembl IDs.
    
    Args:
        csv_path: Path to CSV file containing cytokine seed genes
        
    Returns:
        Tuple of (ensembl_map, ensembl_ids)
    """
    print(f"Loading cytokine seed genes from {csv_path}...")
    cytoseeds_df = pd.read_csv(csv_path)
    cyto_seed_genes = cytoseeds_df['x'].tolist()
    
    cyto_ensembl_map, cyto_ensembl_ids = gene_converter.symbols_to_ensembl(cyto_seed_genes)
    print(f"Loaded {len(cyto_seed_genes)} cytokine genes")
    print(f"Converted to {len(cyto_ensembl_ids)} Ensembl IDs")
    
    return cyto_ensembl_map, cyto_ensembl_ids


def load_gene_annotation(annotation_path: str) -> Tuple[pd.DataFrame, Dict, List]:
    """
    Load gene annotation file and convert gene names to Ensembl IDs.
    
    Args:
        annotation_path: Path to gene annotation CSV file
        
    Returns:
        Tuple of (gene_annotation, ensembl_map, ensembl_ids)
    """
    print(f"Loading gene annotation from {annotation_path}...")
    gene_annotation = pd.read_csv(annotation_path)
    gene_annotation = gene_annotation.set_index('Genename')
    
    gene_names = gene_annotation.index.tolist()
    gene_annotation_ensembl_map, gene_annotation_ensembl_ids = gene_converter.symbols_to_ensembl(gene_names)
    
    print(f"Loaded annotation for {len(gene_annotation)} genes")
    
    return gene_annotation, gene_annotation_ensembl_map, gene_annotation_ensembl_ids


def load_ajm_sparse_matrix(
    sparse_data_path: str,
    row_names_path: str,
    col_names_path: str,
    metadata_path: str,
    gene_annotation: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load AJM dataset from sparse matrix files and convert to DataFrame.
    
    Args:
        sparse_data_path: Path to sparse matrix CSV
        row_names_path: Path to row names (genes) CSV
        col_names_path: Path to column names (cells) CSV
        metadata_path: Path to metadata CSV
        gene_annotation: Gene annotation DataFrame
        
    Returns:
        Tuple of (ajm_df, ajm_features)
    """
    print("Loading AJM dataset from sparse matrix...")
    
    # Load sparse matrix data
    sparse_data = pd.read_csv(sparse_data_path)
    row_names = pd.read_csv(row_names_path)["row_names"].tolist()  # Genes
    col_names = pd.read_csv(col_names_path)["col_names"].tolist()  # Cells
    
    nrows = len(row_names)
    ncols = len(col_names)
    
    print(f"Matrix dimensions: {nrows} genes x {ncols} cells")
    
    # Create sparse matrix
    row_indices = sparse_data["row"].values
    col_indices = sparse_data["col"].values
    values = sparse_data["value"].values
    
    sparse_matrix = sp.coo_matrix(
        (values, (row_indices, col_indices)), 
        shape=(nrows, ncols)
    )
    
    # Transpose to have cells as rows and genes as columns
    sparse_matrix = sparse_matrix.transpose().tocsr()
    
    # Convert to DataFrame
    print("Converting sparse matrix to DataFrame...")
    ajm_df = pd.DataFrame(
        sparse_matrix.toarray(), 
        index=col_names,
        columns=row_names
    )
    
    # Convert gene names to Ensembl IDs
    print("Converting gene names to Ensembl IDs...")
    ajm_gene_names = ajm_df.columns.tolist()
    ajm_ensembl_map, _ = gene_converter.symbols_to_ensembl(ajm_gene_names)
    ensembl_names = [ajm_ensembl_map.get(gene, gene) for gene in ajm_gene_names]
    ajm_df.columns = ensembl_names
    
    # Load metadata
    ajm_features = pd.read_csv(metadata_path, index_col=0)
    if 'cell_id' in ajm_features.columns:
        ajm_features.set_index('cell_id', inplace=True)
    
    # Align cells
    common_cells = ajm_df.index.intersection(ajm_features.index)
    print(f"Common cells: {len(common_cells)}")
    
    ajm_df = ajm_df.loc[common_cells]
    ajm_features = ajm_features.loc[common_cells]
    
    # Remove duplicate genes
    print(f"Shape before removing duplicates: {ajm_df.shape}")
    mask = ~ajm_df.columns.duplicated(keep='first')
    ajm_df = ajm_df.loc[:, mask]
    print(f"Shape after removing duplicates: {ajm_df.shape}")
    
    # Filter for protein-coding genes
    print("Filtering for protein-coding genes...")
    protein_coding_genes = gene_annotation[
        gene_annotation['Genetype'] == 'protein_coding'
    ]['GeneID'].tolist()
    
    common_genes = ajm_df.columns.intersection(protein_coding_genes)
    ajm_df = ajm_df[common_genes]
    print(f"Final shape after protein coding filter: {ajm_df.shape}")
    
    return ajm_df, ajm_features


# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================

def add_cyto_labels(ajm_features: pd.DataFrame) -> pd.DataFrame:
    """
    Add cytokine labels based on time points.
    
    Args:
        ajm_features: Feature DataFrame
        
    Returns:
        Updated feature DataFrame with 'cyto' column
    """
    ajm_label_mapping = {
        'TC-0hr':       {'cyto': 0},
        'TC-LPS-3hr':   {'cyto': -1},
        'TC-LPS-6hr':   {'cyto': -1},
        'TC-LPS-24hr':  {'cyto': 1},
        'TC-LPS-48hr':  {'cyto': -1},
        'TC-LPS-72hr':  {'cyto': -1},
    }
    
    ajm_features = ajm_features.copy()
    ajm_features['cyto'] = None
    
    for sample_value, labels in ajm_label_mapping.items():
        mask = ajm_features['sample'] == sample_value
        ajm_features.loc[mask, 'cyto'] = labels['cyto']
    
    return ajm_features


def calculate_cyto_scores_for_selection(
    ajm_df: pd.DataFrame, 
    ajm_features: pd.DataFrame,
    cyto_ensembl_ids: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate cytokine scores for cells with labels 0 or 1 (for cell selection only).
    
    NOTE: Cytokine seed genes are ONLY used to calculate cyto_score for identifying
    which cells to use for training. The actual model training will use the FULL
    gene expression matrix (all protein-coding genes).
    
    Args:
        ajm_df: Gene expression DataFrame (all protein-coding genes)
        ajm_features: Feature DataFrame with 'cyto' column
        cyto_ensembl_ids: List of cytokine seed gene Ensembl IDs
        
    Returns:
        Tuple of (cyto_scores_df, cyto_intersection)
        - cyto_scores_df: DataFrame with cyto_score for each cell (restricted to label 0/1 cells)
        - cyto_intersection: List of cytokine genes found in the dataset
    """
    print("\n=== Calculating Cytokine Scores for Cell Selection ===")
    
    # Filter for cells with cyto label of 0 or 1
    filtered_features = ajm_features[ajm_features['cyto'].isin([0, 1])].copy()
    
    print(f"Cells with label 0 or 1: {len(filtered_features)}")
    print(f"Label distribution:\n{filtered_features['cyto'].value_counts()}")
    
    # Find intersection with cytokine genes
    all_genes = ajm_df.columns.tolist()
    cyto_intersection = list(set(all_genes).intersection(set(cyto_ensembl_ids)))
    print(f"Found {len(cyto_intersection)} cytokine seed genes in the dataset")
    
    # Calculate cytokine score (sum of cytokine gene expression)
    cyto_expression = ajm_df.loc[filtered_features.index, cyto_intersection]
    cyto_scores = cyto_expression.sum(axis=1)
    
    # Create DataFrame with scores and labels
    cyto_scores_df = pd.DataFrame({
        'cyto_label': filtered_features['cyto'],
        'cyto_score': cyto_scores
    })
    
    print(f"Cyto score statistics:")
    print(f"  Mean: {cyto_scores.mean():.2f}")
    print(f"  Median: {cyto_scores.median():.2f}")
    print(f"  Range: [{cyto_scores.min():.2f}, {cyto_scores.max():.2f}]")
    
    return cyto_scores_df, cyto_intersection


def select_training_cells_by_quartiles(
    cyto_scores_df: pd.DataFrame
) -> Tuple[pd.Index, pd.Index, float, float]:
    """
    Select training cells based on quartile filtering of cytokine scores.
    
    Group 1 (label 0): Cells with label 0 and cyto_score <= 75th percentile
    Group 2 (label 1): Cells with label 1 and cyto_score >= 25th percentile
    
    Args:
        cyto_scores_df: DataFrame with 'cyto_label' and 'cyto_score' columns
        
    Returns:
        Tuple of (group_1_cells, group_2_cells, q3_label_0, q1_label_1)
    """
    print("\n=== Selecting Training Cells by Quartile Filtering ===")
    
    # Get quartile values
    label_0_data = cyto_scores_df[cyto_scores_df['cyto_label'] == 0]['cyto_score']
    label_1_data = cyto_scores_df[cyto_scores_df['cyto_label'] == 1]['cyto_score']
    
    q3_label_0 = label_0_data.quantile(0.75)  # 75th percentile for label 0
    q1_label_1 = label_1_data.quantile(0.25)  # 25th percentile for label 1
    
    # Select cells based on quartiles
    group_1_mask = (cyto_scores_df['cyto_label'] == 0) & (cyto_scores_df['cyto_score'] <= q3_label_0)
    group_2_mask = (cyto_scores_df['cyto_label'] == 1) & (cyto_scores_df['cyto_score'] >= q1_label_1)
    
    group_1_cells = cyto_scores_df[group_1_mask].index
    group_2_cells = cyto_scores_df[group_2_mask].index
    
    print(f"Label 0: Q3 (75th percentile) = {q3_label_0:.2f}")
    print(f"Label 1: Q1 (25th percentile) = {q1_label_1:.2f}")
    print(f"\nSelected training cells:")
    print(f"  Group 1 (label 0, cyto_score <= Q3): {len(group_1_cells)} cells")
    print(f"  Group 2 (label 1, cyto_score >= Q1): {len(group_2_cells)} cells")
    print(f"  Total training cells: {len(group_1_cells) + len(group_2_cells)}")
    
    return group_1_cells, group_2_cells, q3_label_0, q1_label_1


def create_reference_profiles_for_cytobeam(
    ajm_df: pd.DataFrame,
    group_1_cells: pd.Index,
    group_2_cells: pd.Index,
    cyto_intersection: List[str]
) -> Tuple[pd.Series, pd.Series]:
    """
    Create average reference profiles (cyto_0 and cyto_1) for Cyto-Beam score calculation.
    
    These profiles are ONLY used for calculating Cyto-Beam scores (for validation/analysis).
    They are NOT used for model training.
    
    Args:
        ajm_df: Full gene expression DataFrame
        group_1_cells: Cell IDs for group 1 (label 0)
        group_2_cells: Cell IDs for group 2 (label 1)
        cyto_intersection: List of cytokine gene IDs
        
    Returns:
        Tuple of (cyto_0, cyto_1) - average expression profiles of cytokine genes
    """
    print("\n=== Creating Reference Profiles for Cyto-Beam Calculation ===")
    print("NOTE: These are used ONLY for Cyto-Beam scores, NOT for model training")
    
    # Get cytokine gene expression for selected cells
    group_1_cyto_expr = ajm_df.loc[group_1_cells, cyto_intersection]
    group_2_cyto_expr = ajm_df.loc[group_2_cells, cyto_intersection]
    
    # Calculate average profiles
    cyto_0 = group_1_cyto_expr.mean(axis=0)
    cyto_1 = group_2_cyto_expr.mean(axis=0)
    
    print(f"cyto_0 (group 1 average): {len(group_1_cells)} cells, {len(cyto_intersection)} genes")
    print(f"cyto_1 (group 2 average): {len(group_2_cells)} cells, {len(cyto_intersection)} genes")
    
    return cyto_0, cyto_1


def calculate_cytobeam_scores(
    ajm_df: pd.DataFrame,
    ajm_features: pd.DataFrame,
    cyto_0: pd.Series,
    cyto_1: pd.Series,
    cyto_intersection: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Calculate Cyto-Beam scores for all cells in the dataset.
    
    Args:
        ajm_df: Full gene expression DataFrame
        ajm_features: Feature DataFrame
        cyto_0: Reference profile for cyto_0 state
        cyto_1: Reference profile for cyto_1 state
        cyto_intersection: List of cytokine gene IDs
        
    Returns:
        Tuple of (ajm_df with scores, ajm_features with scores, corr_0_1)
    """
    print("\n=== Calculating Cyto-Beam Scores ===")
    
    # Get expression data for all cells (cytokine genes only)
    all_cells_cyto_expression = ajm_df[cyto_intersection].copy()
    
    # Calculate correlations
    corr_with_cyto_0 = all_cells_cyto_expression.corrwith(cyto_0, axis=1)
    corr_with_cyto_1 = all_cells_cyto_expression.corrwith(cyto_1, axis=1)
    
    # Correlation between reference profiles
    corr_0_1 = np.corrcoef(cyto_0, cyto_1)[0, 1]
    print(f"Correlation between cyto_0 and cyto_1: {corr_0_1:.3f}")
    
    # Define reference points and beam vector
    P0 = np.array([1, corr_0_1])  # cyto_0 reference point
    Pmax = np.array([corr_0_1, 1])  # cyto_1 reference point
    beam_vector = Pmax - P0
    beam_length = np.linalg.norm(beam_vector)
    beam_unit = beam_vector / beam_length
    
    # Calculate Cyto-Beam scores
    cytobeam_scores = []
    
    for idx in all_cells_cyto_expression.index:
        cell_point = np.array([corr_with_cyto_0[idx], corr_with_cyto_1[idx]])
        cell_vector = cell_point - P0
        projection_length = np.dot(cell_vector, beam_unit)
        cytobeam_score = projection_length / beam_length
        cytobeam_scores.append(cytobeam_score)
    
    # Add scores to DataFrames
    ajm_df = ajm_df.copy()
    ajm_features = ajm_features.copy()
    
    ajm_df['cytobeam_score'] = cytobeam_scores
    ajm_features['cytobeam_score'] = cytobeam_scores
    
    print(f"Cyto-Beam scores calculated for {len(cytobeam_scores)} cells")
    print(f"Score range: [{min(cytobeam_scores):.3f}, {max(cytobeam_scores):.3f}]")
    
    return ajm_df, ajm_features, corr_0_1


# ============================================================================
# DATA SPLITTING FUNCTIONS
# ============================================================================

def split_train_test_validation(
    ajm_df: pd.DataFrame,
    ajm_features: pd.DataFrame,
    group_1_cells: pd.Index,
    group_2_cells: pd.Index,
    timepoints_for_vi: List[str] = None,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Split data into train, test, and validation sets using FULL gene expression.
    
    IMPORTANT: All splits use the FULL gene expression matrix (all protein-coding genes),
    NOT restricted to cytokine seed genes.
    
    Training set: 
        - Cells from group 1 (label 0) with full gene expression
        - Cells from group 2 (label 1) with full gene expression
        - Model will learn: gene expression → probability of label 0 or 1 → Bernoulli(label)
    
    Test/Validation: 
        - Remaining cells from filtered timepoints with full gene expression
    
    Args:
        ajm_df: FULL gene expression DataFrame (all protein-coding genes)
        ajm_features: Feature DataFrame (contains labels and cytobeam_score)
        group_1_cells: Cell IDs for group 1 (label 0)
        group_2_cells: Cell IDs for group 2 (label 1)
        timepoints_for_vi: Time points to include (default: all timepoints)
        test_ratio: Proportion of non-training cells for testing
        val_ratio: Proportion of non-training cells for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with:
            - 'train_df': Training gene expression (full genes)
            - 'train_features': Training metadata (includes 'cyto' column with labels 0/1)
            - 'test_df': Test gene expression (full genes)
            - 'test_features': Test metadata
            - 'val_df': Validation gene expression (full genes)
            - 'val_features': Validation metadata
            - 'full_df': All filtered cells gene expression
            - 'full_features': All filtered cells metadata
    """
    if timepoints_for_vi is None:
        timepoints_for_vi = ['TC-0hr', 'TC-LPS-3hr', 'TC-LPS-6hr', 'TC-LPS-24hr', 'TC-LPS-48hr', 'TC-LPS-72hr']
    
    print("\n=== Splitting Data into Train/Test/Validation ===")
    print(f"Time points for VI: {timepoints_for_vi}")
    print(f"Gene expression matrix: FULL (all protein-coding genes)")
    
    # Filter for specified time points
    ajm_features_filtered = ajm_features[ajm_features['sample'].isin(timepoints_for_vi)].copy()
    ajm_df_filtered = ajm_df.loc[ajm_features_filtered.index].copy()
    
    print(f"\nFiltered dataset shape: {ajm_df_filtered.shape}")
    print(f"  - Cells: {ajm_df_filtered.shape[0]}")
    print(f"  - Genes: {ajm_df_filtered.shape[1]}")
    print(f"\nSample distribution:")
    print(ajm_features_filtered['sample'].value_counts().sort_index())
    
    # Define training cells (group_1 + group_2)
    train_cells = group_1_cells.union(group_2_cells)
    train_cells = train_cells.intersection(ajm_df_filtered.index)  # Ensure they're in filtered set
    
    # Define non-training cells
    all_filtered_cells = ajm_df_filtered.index
    non_train_cells = all_filtered_cells.difference(train_cells)
    
    print(f"\n=== Training Set ===")
    print(f"Total training cells: {len(train_cells)}")
    print(f"  - Group 1 (label 0): {len(group_1_cells.intersection(train_cells))} cells")
    print(f"  - Group 2 (label 1): {len(group_2_cells.intersection(train_cells))} cells")
    
    # Split non-training cells into test and validation
    np.random.seed(random_seed)
    non_train_cells_shuffled = np.random.permutation(non_train_cells)
    
    n_test = int(len(non_train_cells) * test_ratio / (test_ratio + val_ratio))
    test_cells = non_train_cells_shuffled[:n_test]
    val_cells = non_train_cells_shuffled[n_test:]
    
    print(f"\n=== Test Set ===")
    print(f"Test cells: {len(test_cells)}")
    
    print(f"\n=== Validation Set ===")
    print(f"Validation cells: {len(val_cells)}")
    
    # Create data splits - FULL gene expression for all
    train_df = ajm_df_filtered.loc[train_cells].copy()
    train_features = ajm_features_filtered.loc[train_cells].copy()
    
    test_df = ajm_df_filtered.loc[test_cells].copy()
    test_features = ajm_features_filtered.loc[test_cells].copy()
    
    val_df = ajm_df_filtered.loc[val_cells].copy()
    val_features = ajm_features_filtered.loc[val_cells].copy()
    
    # Remove cytobeam_score from gene expression DataFrames (it should only be in features)
    for df in [train_df, test_df, val_df, ajm_df_filtered]:
        if 'cytobeam_score' in df.columns:
            df.drop('cytobeam_score', axis=1, inplace=True)
    
    print(f"\n=== Data Split Summary ===")
    print(f"Train: {train_df.shape[0]} cells × {train_df.shape[1]} genes")
    print(f"  - Label 0: {(train_features['cyto'] == 0).sum()} cells")
    print(f"  - Label 1: {(train_features['cyto'] == 1).sum()} cells")
    print(f"Test:  {test_df.shape[0]} cells × {test_df.shape[1]} genes")
    print(f"Val:   {val_df.shape[0]} cells × {val_df.shape[1]} genes")
    
    return {
        'train_df': train_df,
        'train_features': train_features,
        'test_df': test_df,
        'test_features': test_features,
        'val_df': val_df,
        'val_features': val_features,
        'full_df': ajm_df_filtered,
        'full_features': ajm_features_filtered
    }


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def prepare_ajm_cyto_dataset(
    data_dir: str = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/BRAY_AJM2/2_Data",
    output_dir: str = "/labs/Aguiar/SSPA_BRAY/BRay",
    save_outputs: bool = True,
    timepoints_for_vi: List[str] = None
) -> Dict:
    """
    Complete pipeline to prepare AJM cytokine dataset for variational inference.
    
    Args:
        data_dir: Base directory for AJM data
        output_dir: Directory to save output files
        save_outputs: Whether to save intermediate outputs
        timepoints_for_vi: Time points to include for VI
        
    Returns:
        Dictionary containing all processed data and metadata
    """
    print("="*80)
    print("AJM CYTOKINE DATASET PREPARATION PIPELINE")
    print("="*80)
    
    # Define paths
    paths = {
        'cyto_seeds': f"{data_dir}/1_ASGs/CYTOBEAM_genes.csv",
        'gene_annotation': "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/ENS_mouse_geneannotation.csv",
        'sparse_data': f"{data_dir}/2_SingleCellData/2_AJM_Parse_Timecourse/raw_matrix_sparse.csv",
        'row_names': f"{data_dir}/2_SingleCellData/2_AJM_Parse_Timecourse/raw_matrix_rownames.csv",
        'col_names': f"{data_dir}/2_SingleCellData/2_AJM_Parse_Timecourse/raw_matrix_colnames.csv",
        'metadata': f"{data_dir}/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"
    }
    
    # Step 1: Load cytokine seed genes
    cyto_ensembl_map, cyto_ensembl_ids = load_cytokine_seed_genes(paths['cyto_seeds'])
    
    # Step 2: Load gene annotation
    gene_annotation, gene_annotation_ensembl_map, _ = load_gene_annotation(paths['gene_annotation'])
    
    # Step 3: Load AJM dataset
    ajm_df, ajm_features = load_ajm_sparse_matrix(
        paths['sparse_data'],
        paths['row_names'],
        paths['col_names'],
        paths['metadata'],
        gene_annotation
    )
    
    # Step 4: Add cyto labels
    ajm_features = add_cyto_labels(ajm_features)
    
    # Step 5: Calculate cytokine scores for cell selection
    cyto_scores_df, cyto_intersection = calculate_cyto_scores_for_selection(
        ajm_df, ajm_features, cyto_ensembl_ids
    )
    
    # Step 6: Select training cells based on quartile filtering
    group_1_cells, group_2_cells, q3_label_0, q1_label_1 = select_training_cells_by_quartiles(
        cyto_scores_df
    )
    
    # Step 7: Create reference profiles (for Cyto-Beam calculation only, NOT for training)
    cyto_0, cyto_1 = create_reference_profiles_for_cytobeam(
        ajm_df, group_1_cells, group_2_cells, cyto_intersection
    )
    
    # Step 8: Calculate Cyto-Beam scores for all cells (for validation/analysis)
    ajm_df, ajm_features, corr_0_1 = calculate_cytobeam_scores(
        ajm_df, ajm_features, cyto_0, cyto_1, cyto_intersection
    )
    
    # Step 9: Split data (using FULL gene expression)
    data_splits = split_train_test_validation(
        ajm_df, ajm_features,
        group_1_cells, group_2_cells,
        timepoints_for_vi=timepoints_for_vi
    )
    
    # Compile results
    results = {
        'data_splits': data_splits,
        'reference_profiles': {
            'cyto_0': cyto_0,  # Average profile for Cyto-Beam (validation only)
            'cyto_1': cyto_1,  # Average profile for Cyto-Beam (validation only)
        },
        'training_cells': {
            'group_1_cells': group_1_cells,  # Label 0 training cells
            'group_2_cells': group_2_cells,  # Label 1 training cells
        },
        'metadata': {
            'cyto_intersection': cyto_intersection,  # Cytokine genes (for scoring only)
            'cyto_ensembl_map': cyto_ensembl_map,
            'gene_annotation_ensembl_map': gene_annotation_ensembl_map,
            'corr_0_1': corr_0_1,
            'q3_label_0': q3_label_0,
            'q1_label_1': q1_label_1,
            'n_genes': ajm_df.shape[1],  # Total genes used for training
            'n_cyto_genes': len(cyto_intersection),  # Cytokine genes (scoring only)
        },
        'full_data': {
            'ajm_df': ajm_df,
            'ajm_features': ajm_features
        }
    }
    
    # Save outputs
    if save_outputs:
        print("\n=== Saving Outputs ===")
        
        # Save data splits
        with open(f"{output_dir}/ajm_df_vi.pkl", 'wb') as f:
            pickle.dump(data_splits['full_df'], f)
        print(f"Saved full VI dataset to ajm_df_vi.pkl")
        
        with open(f"{output_dir}/ajm_features_vi.pkl", 'wb') as f:
            pickle.dump(data_splits['full_features'], f)
        print(f"Saved full VI features to ajm_features_vi.pkl")
        
        # Save reference profiles
        with open(f"{output_dir}/cyto_beam_reference_points.json", 'w') as f:
            import json
            json.dump({
                'P0': [1.0, float(corr_0_1)],
                'Pmax': [float(corr_0_1), 1.0],
                'corr_0_1': float(corr_0_1),
                'q3_label_0': float(q3_label_0),
                'q1_label_1': float(q1_label_1),
                'n_cyto_0_cells': len(group_1_cells),
                'n_cyto_1_cells': len(group_2_cells),
                'n_cyto_genes': len(cyto_intersection)
            }, f, indent=2)
        print(f"Saved Cyto-Beam reference points")
        
        # Save Cyto-Beam scores
        cytobeam_df = pd.DataFrame({
            'cell_id': data_splits['full_features'].index,
            'cytobeam_score': data_splits['full_features']['cytobeam_score'],
            'sample': data_splits['full_features']['sample']
        })
        cytobeam_df.to_csv(f"{output_dir}/cyto_beam_scores.csv", index=False)
        print(f"Saved Cyto-Beam scores to CSV")
        
        # Save train/test/val cell IDs and metadata
        with open(f"{output_dir}/data_split_cell_ids.json", 'w') as f:
            import json
            json.dump({
                'train_cells': data_splits['train_df'].index.tolist(),
                'test_cells': data_splits['test_df'].index.tolist(),
                'val_cells': data_splits['val_df'].index.tolist(),
                'group_1_cells': group_1_cells.tolist(),
                'group_2_cells': group_2_cells.tolist(),
                'train_labels': {
                    'n_label_0': int((data_splits['train_features']['cyto'] == 0).sum()),
                    'n_label_1': int((data_splits['train_features']['cyto'] == 1).sum())
                }
            }, f, indent=2)
        print(f"Saved data split cell IDs")
        
        # Save training data with labels
        train_data_with_labels = data_splits['train_df'].copy()
        train_data_with_labels['label'] = data_splits['train_features']['cyto']
        train_data_with_labels.to_csv(f"{output_dir}/train_data_full_genes.csv")
        print(f"Saved training data with labels (full gene expression)")
        
        # Save gene list
        with open(f"{output_dir}/gene_list.txt", 'w') as f:
            for gene in data_splits['train_df'].columns:
                f.write(f"{gene}\n")
        print(f"Saved gene list ({len(data_splits['train_df'].columns)} genes)")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\n📊 DATA SPLITS (all using FULL gene expression):")
    print(f"  Training:   {results['data_splits']['train_df'].shape[0]:>5} cells × {results['data_splits']['train_df'].shape[1]} genes")
    print(f"    - Label 0: {(results['data_splits']['train_features']['cyto'] == 0).sum():>4} cells")
    print(f"    - Label 1: {(results['data_splits']['train_features']['cyto'] == 1).sum():>4} cells")
    print(f"  Test:       {results['data_splits']['test_df'].shape[0]:>5} cells × {results['data_splits']['test_df'].shape[1]} genes")
    print(f"  Validation: {results['data_splits']['val_df'].shape[0]:>5} cells × {results['data_splits']['val_df'].shape[1]} genes")
    print(f"  Full:       {results['data_splits']['full_df'].shape[0]:>5} cells × {results['data_splits']['full_df'].shape[1]} genes")
    
    print(f"\n🧬 GENES:")
    print(f"  Total protein-coding genes (used for training): {results['metadata']['n_genes']}")
    print(f"  Cytokine seed genes (used for scoring only):    {results['metadata']['n_cyto_genes']}")
    
    print(f"\n📈 CYTO-BEAM METRICS (for validation):")
    print(f"  Reference correlation (cyto_0 ↔ cyto_1): {results['metadata']['corr_0_1']:.3f}")
    print(f"  Quartile thresholds:")
    print(f"    - Label 0 Q3: {results['metadata']['q3_label_0']:.2f}")
    print(f"    - Label 1 Q1: {results['metadata']['q1_label_1']:.2f}")
    
    print(f"\n🎯 MODEL TRAINING SETUP:")
    print(f"  Input:  Full gene expression ({results['metadata']['n_genes']} genes)")
    print(f"  Output: Binary labels (0 or 1)")
    print(f"  Model:  Gene expression → P(label) → Bernoulli(label)")
    print("="*80 + "\n")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline (using all timepoints)
    results = prepare_ajm_cyto_dataset(
        data_dir="/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/BRAY_AJM2/2_Data",
        output_dir="/labs/Aguiar/SSPA_BRAY/BRay",
        save_outputs=True,
        timepoints_for_vi=None  # None = use all timepoints
    )

