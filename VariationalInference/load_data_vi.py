import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path


def prepare_matrices(df, features, cell_ids, gene_list):
    """
    Extract X, X_aux, y for given cell IDs.
    
    This matches the user's exact code from quick_reference.py
    """
    # Get subsets
    df_subset = df.loc[df.index.isin(cell_ids)]
    features_subset = features.loc[features.index.isin(cell_ids)]
    
    # Align indices
    common_idx = df_subset.index.intersection(features_subset.index)
    df_subset = df_subset.loc[common_idx]
    features_subset = features_subset.loc[common_idx]
    
    # Extract matrices
    X = df_subset[gene_list].values
    
    # IMPORTANT: No auxiliary features in this example
    X_aux = np.zeros((X.shape[0], 0))
    
    # Use 't2dm' column (or first column if column name varies)
    y_col = 't2dm' if 't2dm' in features_subset.columns else features_subset.columns[0]
    y = features_subset[y_col].values
    y = y.astype(int)
    
    return X, X_aux, y


def load_control_data(data_dir='/labs/Aguiar/SSPA_BRAY/BRay/ctrl_sspa_test'):
    """
    Load data exactly as in user's quick_reference.py
    
    Returns:
    --------
    X_train, y_train, X_aux_train : train set (n=1434)
    X_val, y_val, X_aux_val : validation set (n=307)
    X_test, y_test, X_aux_test : test set (n=308)
    gene_names : list of gene names
    """
    data_dir = Path(data_dir)
    
    print(f"Loading data from: {data_dir}")
    
    # Load expression matrix
    print(f"  Loading df.pkl...")
    df = pd.read_pickle(data_dir / 'df.pkl')
    print(f"    Shape: {df.shape}")
    
    # Load features (contains disease labels)
    print(f"  Loading features.pkl...")
    features = pd.read_pickle(data_dir / 'features.pkl')
    print(f"    Shape: {features.shape}")
    print(f"    Columns: {features.columns.tolist()}")
    
    # Load gene list
    print(f"  Loading gene_list.txt...")
    with open(data_dir / 'gene_list.txt', 'r') as f:
        gene_list = [line.strip() for line in f]
    print(f"    Genes: {len(gene_list)}")
    
    # Load split indices
    print(f"  Loading data_split_cell_ids.json...")
    with open(data_dir / 'data_split_cell_ids.json', 'r') as f:
        splits = json.load(f)
    
    train_ids = splits['train']
    val_ids = splits['val']
    test_ids = splits['test']
    
    print(f"    Train: {len(train_ids)}")
    print(f"    Val:   {len(val_ids)}")
    print(f"    Test:  {len(test_ids)}")
    
    # Prepare matrices using user's exact function
    print(f"\n  Preparing matrices...")
    X_train, X_aux_train, y_train = prepare_matrices(df, features, train_ids, gene_list)
    X_val, X_aux_val, y_val = prepare_matrices(df, features, val_ids, gene_list)
    X_test, X_aux_test, y_test = prepare_matrices(df, features, test_ids, gene_list)
    
    print(f"\n  Final shapes:")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_aux_train: {X_aux_train.shape} (EMPTY - no auxiliary features)")
    print(f"    y_train: {y_train.shape}, distribution: {np.bincount(y_train)}")
    print(f"    X_val: {X_val.shape}")
    print(f"    X_test: {X_test.shape}")
    
    return (X_train, y_train, X_aux_train,
            X_val, y_val, X_aux_val,
            X_test, y_test, X_aux_test,
            gene_list)


if __name__ == "__main__":
    # Test the loader
    data = load_control_data()
    X_train, y_train, X_aux_train = data[:3]
    
    print(f"\n{'='*60}")
    print("DATA LOADED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Training set: {X_train.shape[0]} samples × {X_train.shape[1]} genes")
    print(f"X_aux shape: {X_aux_train.shape} (EMPTY - no auxiliary features)")
    print(f"y distribution: {np.bincount(y_train)}")
    print(f"Sparsity: {(X_train == 0).mean():.2%}")
    print(f"{'='*60}\n")