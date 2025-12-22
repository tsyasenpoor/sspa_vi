"""
BRAY LAB SPECIFIC: AJM Antigen Presentation Dataset Pipeline
==============================================================

This module is specific to Bray Lab's AJM dataset analysis.
For generic VI usage, see: data_loader.py, cli.py, quick_reference.py

This module prepares the AJM dataset for antigen presentation (AP) analysis
using variational inference with seed gene-based scoring.
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

def load_ap_seed_genes(csv_path: str) -> Tuple[Dict, List]:
    print(f"Loading ap seed genes from {csv_path}...")
    apseeds_df = pd.read_csv(csv_path)
    ap_seed_genes = apseeds_df['x'].tolist()
    
    ap_ensembl_map, ap_ensembl_ids = gene_converter.symbols_to_ensembl(ap_seed_genes)
    print(f"Loaded {len(ap_seed_genes)} ap genes")
    print(f"Converted to {len(ap_ensembl_ids)} Ensembl IDs")

    return ap_ensembl_map, ap_ensembl_ids


def load_gene_annotation(annotation_path: str) -> Tuple[pd.DataFrame, Dict, List]:
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

def add_ap_labels(ajm_features: pd.DataFrame) -> pd.DataFrame:
    ajm_label_mapping = {
        'TC-0hr':       {'ap': 0},
        'TC-LPS-3hr':   {'ap': -1},
        'TC-LPS-6hr':   {'ap': 1},
        'TC-LPS-24hr':  {'ap': 1},
        'TC-LPS-48hr':  {'ap': -1},
        'TC-LPS-72hr':  {'ap': -1},
    }
    
    ajm_features = ajm_features.copy()
    ajm_features['ap'] = None

    for sample_value, labels in ajm_label_mapping.items():
        mask = ajm_features['sample'] == sample_value
        ajm_features.loc[mask, 'ap'] = labels['ap']

    return ajm_features


def calculate_ap_scores_for_selection(
    ajm_df: pd.DataFrame, 
    ajm_features: pd.DataFrame,
    ap_ensembl_ids: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    print("\n=== Calculating AP Scores for Cell Selection ===")

    # Filter for cells with ap label of 0 or 1
    filtered_features = ajm_features[ajm_features['ap'].isin([0, 1])].copy()

    print(f"Cells with label 0 or 1: {len(filtered_features)}")
    print(f"Label distribution:\n{filtered_features['ap'].value_counts()}")

    # Find intersection with AP genes
    all_genes = ajm_df.columns.tolist()
    ap_intersection = list(set(all_genes).intersection(set(ap_ensembl_ids)))
    print(f"Found {len(ap_intersection)} AP seed genes in the dataset")

    # Calculate AP score (sum of AP gene expression)
    ap_expression = ajm_df.loc[filtered_features.index, ap_intersection]
    ap_scores = ap_expression.sum(axis=1)

    # Create DataFrame with scores and labels
    ap_scores_df = pd.DataFrame({
        'ap_label': filtered_features['ap'],
        'ap_score': ap_scores
    })

    print(f"AP score statistics:")
    print(f"  Mean: {ap_scores.mean():.2f}")
    print(f"  Median: {ap_scores.median():.2f}")
    print(f"  Range: [{ap_scores.min():.2f}, {ap_scores.max():.2f}]")

    return ap_scores_df, ap_intersection


def select_training_cells_by_quartiles(
    ap_scores_df: pd.DataFrame
) -> Tuple[pd.Index, pd.Index, float, float]:
    print("\n=== Selecting Training Cells by Quartile Filtering ===")

    # Get quartile values
    label_0_data = ap_scores_df[ap_scores_df['ap_label'] == 0]['ap_score']
    label_1_data = ap_scores_df[ap_scores_df['ap_label'] == 1]['ap_score']

    q3_label_0 = label_0_data.quantile(0.75)  # 75th percentile for label 0
    q1_label_1 = label_1_data.quantile(0.25)  # 25th percentile for label 1
    
    # Select cells based on quartiles
    group_1_mask = (ap_scores_df['ap_label'] == 0) & (ap_scores_df['ap_score'] <= q3_label_0)
    group_2_mask = (ap_scores_df['ap_label'] == 1) & (ap_scores_df['ap_score'] >= q1_label_1)

    group_1_cells = ap_scores_df[group_1_mask].index
    group_2_cells = ap_scores_df[group_2_mask].index

    print(f"Label 0: Q3 (75th percentile) = {q3_label_0:.2f}")
    print(f"Label 1: Q1 (25th percentile) = {q1_label_1:.2f}")
    print(f"\nSelected training cells:")
    print(f"  Group 1 (label 0, ap_score <= Q3): {len(group_1_cells)} cells")
    print(f"  Group 2 (label 1, ap_score >= Q1): {len(group_2_cells)} cells")
    print(f"  Total training cells: {len(group_1_cells) + len(group_2_cells)}")
    
    return group_1_cells, group_2_cells, q3_label_0, q1_label_1


def create_reference_profiles_for_apbeam(
    ajm_df: pd.DataFrame,
    group_1_cells: pd.Index,
    group_2_cells: pd.Index,
    ap_intersection: List[str]
) -> Tuple[pd.Series, pd.Series]:
    print("\n=== Creating Reference Profiles for AP-Beam Calculation ===")
    print("NOTE: These are used ONLY for AP-Beam scores, NOT for model training")

    # Get AP gene expression for selected cells
    group_1_ap_expr = ajm_df.loc[group_1_cells, ap_intersection]
    group_2_ap_expr = ajm_df.loc[group_2_cells, ap_intersection]

    # Calculate average profiles
    ap_0 = group_1_ap_expr.mean(axis=0)
    ap_1 = group_2_ap_expr.mean(axis=0)

    print(f"ap_0 (group 1 average): {len(group_1_cells)} cells, {len(ap_intersection)} genes")
    print(f"ap_1 (group 2 average): {len(group_2_cells)} cells, {len(ap_intersection)} genes")

    return ap_0, ap_1


def calculate_apbeam_scores(
    ajm_df: pd.DataFrame,
    ajm_features: pd.DataFrame,
    ap_0: pd.Series,
    ap_1: pd.Series,
    ap_intersection: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    print("\n=== Calculating AP-Beam Scores ===")

    # Get expression data for all cells (AP genes only)
    all_cells_ap_expression = ajm_df[ap_intersection].copy()

    # Calculate correlations
    corr_with_ap_0 = all_cells_ap_expression.corrwith(ap_0, axis=1)
    corr_with_ap_1 = all_cells_ap_expression.corrwith(ap_1, axis=1)

    # Correlation between reference profiles
    corr_0_1 = np.corrcoef(ap_0, ap_1)[0, 1]
    print(f"Correlation between ap_0 and ap_1: {corr_0_1:.3f}")

    # Define reference points and beam vector
    P0 = np.array([1, corr_0_1])  # ap_0 reference point
    Pmax = np.array([corr_0_1, 1])  # ap_1 reference point
    beam_vector = Pmax - P0
    beam_length = np.linalg.norm(beam_vector)
    beam_unit = beam_vector / beam_length

    # Calculate AP-Beam scores
    apbeam_scores = []

    for idx in all_cells_ap_expression.index:
        cell_point = np.array([corr_with_ap_0[idx], corr_with_ap_1[idx]])
        cell_vector = cell_point - P0
        projection_length = np.dot(cell_vector, beam_unit)
        apbeam_score = projection_length / beam_length
        apbeam_scores.append(apbeam_score)

    # Add scores to DataFrames
    ajm_df = ajm_df.copy()
    ajm_features = ajm_features.copy()

    ajm_df['apbeam_score'] = apbeam_scores
    ajm_features['apbeam_score'] = apbeam_scores

    print(f"AP-Beam scores calculated for {len(apbeam_scores)} cells")
    print(f"Score range: [{min(apbeam_scores):.3f}, {max(apbeam_scores):.3f}]")

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
    
    # Remove apbeam_score from gene expression DataFrames (it should only be in features)
    for df in [train_df, test_df, val_df, ajm_df_filtered]:
        if 'apbeam_score' in df.columns:
            df.drop('apbeam_score', axis=1, inplace=True)
    
    print(f"\n=== Data Split Summary ===")
    print(f"Train: {train_df.shape[0]} cells × {train_df.shape[1]} genes")
    print(f"  - Label 0: {(train_features['ap'] == 0).sum()} cells")
    print(f"  - Label 1: {(train_features['ap'] == 1).sum()} cells")
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

def prepare_ajm_ap_dataset(
    data_dir: str = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/BRAY_AJM2/2_Data",
    output_dir: str = "/labs/Aguiar/SSPA_BRAY/BRay/ap",
    save_outputs: bool = True,
    timepoints_for_vi: List[str] = None
) -> Dict:

    print("="*80)
    print("AJM AP DATASET PREPARATION PIPELINE")
    print("="*80)
    
    # Define paths
    paths = {
        'ap_seeds': f"{data_dir}/1_ASGs/APBEAM_genes.csv",
        'gene_annotation': "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/ENS_mouse_geneannotation.csv",
        'sparse_data': f"{data_dir}/2_SingleCellData/2_AJM_Parse_Timecourse/raw_matrix_sparse.csv",
        'row_names': f"{data_dir}/2_SingleCellData/2_AJM_Parse_Timecourse/raw_matrix_rownames.csv",
        'col_names': f"{data_dir}/2_SingleCellData/2_AJM_Parse_Timecourse/raw_matrix_colnames.csv",
        'metadata': f"{data_dir}/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"
    }

    # Step 1: Load AP seed genes
    ap_ensembl_map, ap_ensembl_ids = load_ap_seed_genes(paths['ap_seeds'])

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

    # Step 4: Add AP labels
    ajm_features = add_ap_labels(ajm_features)

    # Step 5: Calculate AP scores for cell selection
    ap_scores_df, ap_intersection = calculate_ap_scores_for_selection(
        ajm_df, ajm_features, ap_ensembl_ids
    )
    
    # Step 6: Select training cells based on quartile filtering
    group_1_cells, group_2_cells, q3_label_0, q1_label_1 = select_training_cells_by_quartiles(
        ap_scores_df
    )

    # Step 7: Create reference profiles (for AP-Beam calculation only, NOT for training)
    ap_0, ap_1 = create_reference_profiles_for_apbeam(
        ajm_df, group_1_cells, group_2_cells, ap_intersection
    )

    # Step 8: Calculate AP-Beam scores for all cells (for validation/analysis)
    ajm_df, ajm_features, corr_0_1 = calculate_apbeam_scores(
        ajm_df, ajm_features, ap_0, ap_1, ap_intersection
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
            'ap_0': ap_0,  # Average profile for AP-Beam (validation only)
            'ap_1': ap_1,  # Average profile for AP-Beam (validation only)
        },
        'training_cells': {
            'group_1_cells': group_1_cells,  # Label 0 training cells
            'group_2_cells': group_2_cells,  # Label 1 training cells
        },
        'metadata': {
            'ap_intersection': ap_intersection,  # AP genes (for scoring only)
            'ap_ensembl_map': ap_ensembl_map,
            'gene_annotation_ensembl_map': gene_annotation_ensembl_map,
            'corr_0_1': corr_0_1,
            'q3_label_0': q3_label_0,
            'q1_label_1': q1_label_1,
            'n_genes': ajm_df.shape[1],  # Total genes used for training
            'n_ap_genes': len(ap_intersection),  # AP genes (scoring only)
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
        with open(f"{output_dir}/ap_ajm_df_vi.pkl", 'wb') as f:
            pickle.dump(data_splits['full_df'], f)
        print(f"Saved full VI dataset to ap_ajm_df_vi.pkl")

        with open(f"{output_dir}/ap_ajm_features_vi.pkl", 'wb') as f:
            pickle.dump(data_splits['full_features'], f)
        print(f"Saved full VI features to ap_ajm_features_vi.pkl")

        # Save reference profiles
        with open(f"{output_dir}/ap_beam_reference_points.json", 'w') as f:
            import json
            json.dump({
                'P0': [1.0, float(corr_0_1)],
                'Pmax': [float(corr_0_1), 1.0],
                'corr_0_1': float(corr_0_1),
                'q3_label_0': float(q3_label_0),
                'q1_label_1': float(q1_label_1),
                'n_ap_0_cells': len(group_1_cells),
                'n_ap_1_cells': len(group_2_cells),
                'n_ap_genes': len(ap_intersection)
            }, f, indent=2)
        print(f"Saved ap-Beam reference points")
        
        # Save ap-Beam scores
        apbeam_df = pd.DataFrame({
            'cell_id': data_splits['full_features'].index,
            'apbeam_score': data_splits['full_features']['apbeam_score'],
            'sample': data_splits['full_features']['sample']
        })
        apbeam_df.to_csv(f"{output_dir}/ap_beam_scores.csv", index=False)
        print(f"Saved ap-Beam scores to CSV")
        
        # Save train/test/val cell IDs and metadata
        with open(f"{output_dir}/ap_data_split_cell_ids.json", 'w') as f:
            import json
            json.dump({
                'train_cells': data_splits['train_df'].index.tolist(),
                'test_cells': data_splits['test_df'].index.tolist(),
                'val_cells': data_splits['val_df'].index.tolist(),
                'group_1_cells': group_1_cells.tolist(),
                'group_2_cells': group_2_cells.tolist(),
                'train_labels': {
                    'n_label_0': int((data_splits['train_features']['ap'] == 0).sum()),
                    'n_label_1': int((data_splits['train_features']['ap'] == 1).sum())
                }
            }, f, indent=2)
        print(f"Saved data split cell IDs")
        
        # Save training data with labels
        train_data_with_labels = data_splits['train_df'].copy()
        train_data_with_labels['label'] = data_splits['train_features']['ap']
        train_data_with_labels.to_csv(f"{output_dir}/ap_train_data_full_genes.csv")
        print(f"Saved training data with labels (full gene expression)")
        
        # Save gene list
        with open(f"{output_dir}/gene_list.txt", 'w') as f:
            for gene in data_splits['train_df'].columns:
                f.write(f"{gene}\n")
        print(f"Saved gene list ({len(data_splits['train_df'].columns)} genes)")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline (using all timepoints)
    results = prepare_ajm_ap_dataset(
        data_dir="/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/BRAY_AJM2/2_Data",
        output_dir="/labs/Aguiar/SSPA_BRAY/BRay/ap",
        save_outputs=True,
        timepoints_for_vi=None  # None = use all timepoints
    )

