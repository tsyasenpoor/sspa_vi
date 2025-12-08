
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from scipy.stats import poisson


def load_control_data(data_dir='/labs/Aguiar/SSPA_BRAY/BRay/ctrl_sspa_test'):
    """Load control-only baseline data."""
    print(f"Loading control data from: {data_dir}")
    data_dir = Path(data_dir)
    
    # Load expression matrix and features
    df = pd.read_pickle(data_dir / 'df.pkl')
    features = pd.read_pickle(data_dir / 'features.pkl')
    
    # Load gene list
    with open(data_dir / 'gene_list.txt', 'r') as f:
        gene_list = [line.strip() for line in f]
    
    print(f"  Shape: {df.shape}")
    print(f"  Genes: {len(gene_list)}")
    
    # Check for patient IDs
    if 'pt_id' in features.columns:
        print(f"  Found pt_id column - using patient-level assignment")
        pt_ids = features['pt_id']
        unique_patients = pt_ids.unique()
        print(f"  Unique patients: {len(unique_patients)}")
        
        # Cell counts per patient
        cells_per_patient = pt_ids.value_counts()
        print(f"  Cells per patient: min={cells_per_patient.min()}, "
              f"max={cells_per_patient.max()}, mean={cells_per_patient.mean():.1f}")
    else:
        print(f"  WARNING: No pt_id column found - will use cell-level assignment")
        pt_ids = None
        unique_patients = None
    
    return df, features, gene_list, pt_ids


def select_disease_genes(gene_list, n_disease_genes=30, method='random', seed=42):
    """
    Select disease-relevant genes.
    
    Parameters:
    -----------
    gene_list : list
        All gene names
    n_disease_genes : int
        Number of disease genes to select
    method : str
        'random' - random selection
        'high_variance' - select high-variance genes (requires X)
    seed : int
        Random seed
    
    Returns:
    --------
    disease_gene_indices : array
        Indices of disease genes
    disease_gene_names : list
        Names of disease genes
    """
    rng = np.random.RandomState(seed)
    
    print(f"\nSelecting {n_disease_genes} disease genes...")
    
    if method == 'random':
        disease_gene_indices = rng.choice(len(gene_list), n_disease_genes, replace=False)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    disease_gene_names = [gene_list[i] for i in disease_gene_indices]
    
    print(f"  Method: {method}")
    print(f"  Selected {len(disease_gene_indices)} genes")
    print(f"  First 10: {disease_gene_names[:10]}")
    
    return disease_gene_indices, disease_gene_names


def apply_multiplicative_boost(X_control, disease_sample_indices, disease_gene_indices, 
                                boost_factor=2.0, seed=42):
    """
    Apply multiplicative lambda boost to disease genes in disease samples.
    
    Model: X_disease ~ Poisson(λ_control * boost_factor)
    
    Parameters:
    -----------
    X_control : array (n, p)
        Control expression matrix
    disease_sample_indices : array
        Indices of disease samples
    disease_gene_indices : array  
        Indices of disease genes
    boost_factor : float
        Multiplicative boost (e.g., 2.0 = 2x increase)
        Can also be array of shape (n_disease_samples, n_disease_genes)
    seed : int
        Random seed for Poisson sampling
    
    Returns:
    --------
    X_disease : array (n, p)
        Expression matrix with disease signal
    lambda_boost_matrix : array (n, p)
        Boost factors applied (1.0 for non-disease, boost_factor for disease)
    """
    rng = np.random.RandomState(seed)
    
    print(f"\nApplying multiplicative boost...")
    print(f"  Boost factor: {boost_factor}")
    print(f"  Disease samples: {len(disease_sample_indices)}")
    print(f"  Disease genes: {len(disease_gene_indices)}")
    
    # Start with control data
    X_disease = X_control.copy()
    
    # Create boost matrix (1.0 everywhere, boost_factor in disease region)
    lambda_boost_matrix = np.ones_like(X_control)
    
    # Apply boost to disease genes in disease samples
    for i in disease_sample_indices:
        for j in disease_gene_indices:
            # Lambda boost
            lambda_control = X_control[i, j]
            lambda_disease = lambda_control * boost_factor
            
            # Resample from Poisson with boosted lambda
            X_disease[i, j] = rng.poisson(lambda_disease)
            lambda_boost_matrix[i, j] = boost_factor
    
    # Report statistics
    disease_region = X_disease[np.ix_(disease_sample_indices, disease_gene_indices)]
    control_region = X_control[np.ix_(disease_sample_indices, disease_gene_indices)]
    
    print(f"\n  Disease region statistics:")
    print(f"    Control mean: {control_region.mean():.2f}")
    print(f"    Disease mean: {disease_region.mean():.2f}")
    print(f"    Fold change: {disease_region.mean() / (control_region.mean() + 1e-10):.2f}x")
    
    return X_disease, lambda_boost_matrix


def create_synthetic_dataset(data_dir='/labs/Aguiar/SSPA_BRAY/BRay/ctrl_sspa_test',
                             output_dir='/labs/Aguiar/SSPA_BRAY/BRay/synthetic_disease',
                             n_disease_patients=None,
                             n_disease_samples=None,
                             n_disease_genes=30,
                             boost_factor=2.0,
                             use_train_only=True,
                             patient_level=True,
                             seed=42):
    """
    Create synthetic disease dataset from control baseline.
    
    Parameters:
    -----------
    data_dir : str
        Path to control data
    output_dir : str
        Where to save synthetic data
    n_disease_patients : int or None
        Number of PATIENTS to make disease (patient-level assignment)
        Use this for single-cell data with multiple cells per patient
    n_disease_samples : int or None
        Number of SAMPLES/CELLS to make disease (cell-level assignment)
        Use this only if no patient structure or for testing
    n_disease_genes : int
        Number of disease-relevant genes
    boost_factor : float
        Multiplicative boost for disease genes
    use_train_only : bool
        If True, only use training samples/patients for disease conversion
    patient_level : bool
        If True, assign disease at patient level (all cells from patient)
        If False, assign disease at cell level (individual cells)
    seed : int
        Random seed
    
    Returns:
    --------
    Saves:
        - X_train.pkl, y_train.pkl, X_aux_train.pkl
        - X_val.pkl, y_val.pkl, X_aux_val.pkl
        - X_test.pkl, y_test.pkl, X_aux_test.pkl
        - ground_truth.json (disease genes and samples/patients)
        - synthetic_config.json (parameters used)
    """
    rng = np.random.RandomState(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("CREATING SYNTHETIC DISEASE DATASET")
    print("="*70)
    
    # Load control data
    df, features, gene_list, pt_ids = load_control_data(data_dir)
    
    # Load data splits
    with open(Path(data_dir) / 'data_split_cell_ids.json', 'r') as f:
        splits = json.load(f)
    
    train_ids = splits['train']
    val_ids = splits['val']
    test_ids = splits['test']
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_ids)}")
    print(f"  Val:   {len(val_ids)}")
    print(f"  Test:  {len(test_ids)}")
    
    # Select disease genes
    disease_gene_indices, disease_gene_names = select_disease_genes(
        gene_list, n_disease_genes, method='random', seed=seed
    )
    
    # Determine disease assignment strategy
    if pt_ids is not None and patient_level:
        # PATIENT-LEVEL ASSIGNMENT (CORRECT FOR SINGLE-CELL)
        print(f"\n{'='*70}")
        print("PATIENT-LEVEL DISEASE ASSIGNMENT")
        print(f"{'='*70}")
        
        # Get patient IDs for each split
        train_pt_ids = features.loc[train_ids, 'pt_id'].unique()
        val_pt_ids = features.loc[val_ids, 'pt_id'].unique()
        test_pt_ids = features.loc[test_ids, 'pt_id'].unique()
        
        print(f"\nPatients per split:")
        print(f"  Train patients: {len(train_pt_ids)}")
        print(f"  Val patients:   {len(val_pt_ids)}")
        print(f"  Test patients:  {len(test_pt_ids)}")
        
        # Select disease patients
        if use_train_only:
            available_patients = train_pt_ids
            print(f"\nSelecting disease patients from training set only...")
        else:
            available_patients = features['pt_id'].unique()
            print(f"\nSelecting disease patients from all data...")
        
        if n_disease_patients is None:
            raise ValueError("Must specify n_disease_patients for patient-level assignment")
        
        if n_disease_patients > len(available_patients):
            raise ValueError(f"Not enough patients: {n_disease_patients} > {len(available_patients)}")
        
        # Select disease patients
        disease_patients = rng.choice(available_patients, n_disease_patients, replace=False)
        
        # Get all cells from disease patients
        disease_mask = features['pt_id'].isin(disease_patients)
        disease_sample_ids = features[disease_mask].index.tolist()
        disease_sample_indices = [df.index.get_loc(sid) for sid in disease_sample_ids]
        
        n_disease_samples = len(disease_sample_ids)
        
        print(f"  Selected {n_disease_patients} disease patients")
        print(f"  Total disease cells: {n_disease_samples}")
        print(f"  Avg cells per disease patient: {n_disease_samples / n_disease_patients:.1f}")
        
        # Count disease patients per split
        train_disease_pts = features.loc[train_ids][features.loc[train_ids, 'pt_id'].isin(disease_patients)]['pt_id'].nunique()
        val_disease_pts = features.loc[val_ids][features.loc[val_ids, 'pt_id'].isin(disease_patients)]['pt_id'].nunique()
        test_disease_pts = features.loc[test_ids][features.loc[test_ids, 'pt_id'].isin(disease_patients)]['pt_id'].nunique()
        
        print(f"\nDisease patients per split:")
        print(f"  Train: {train_disease_pts}/{len(train_pt_ids)} patients")
        print(f"  Val:   {val_disease_pts}/{len(val_pt_ids)} patients")
        print(f"  Test:  {test_disease_pts}/{len(test_pt_ids)} patients")
        
    else:
        # CELL-LEVEL ASSIGNMENT (NOT RECOMMENDED FOR SINGLE-CELL)
        print(f"\n{'='*70}")
        print("CELL-LEVEL DISEASE ASSIGNMENT")
        if pt_ids is not None:
            print("⚠ WARNING: Patient structure exists but using cell-level assignment!")
            print("⚠ This may cause data leakage - consider patient_level=True")
        print(f"{'='*70}")
        
        if n_disease_samples is None:
            raise ValueError("Must specify n_disease_samples for cell-level assignment")
        
        # Select disease samples (cells)
        if use_train_only:
            available_samples = train_ids
            print(f"\nSelecting disease samples from training set only...")
        else:
            available_samples = list(df.index)
            print(f"\nSelecting disease samples from all data...")
        
        if n_disease_samples > len(available_samples):
            raise ValueError(f"Not enough samples: {n_disease_samples} > {len(available_samples)}")
        
        disease_sample_ids = rng.choice(available_samples, n_disease_samples, replace=False).tolist()
        disease_sample_indices = [df.index.get_loc(sid) for sid in disease_sample_ids]
        disease_patients = None  # Not applicable for cell-level
        
        print(f"  Selected {len(disease_sample_ids)} disease samples")
    
    # Apply boost to create disease data
    X_control = df.values
    X_disease, lambda_boost_matrix = apply_multiplicative_boost(
        X_control, disease_sample_indices, disease_gene_indices, 
        boost_factor, seed
    )
    
    # Create labels (y)
    y = np.zeros(len(df), dtype=int)
    y[disease_sample_indices] = 1
    
    print(f"\nLabel distribution:")
    print(f"  Controls: {(y == 0).sum()}")
    print(f"  Disease:  {(y == 1).sum()}")
    
    # Split into train/val/test
    train_indices = [df.index.get_loc(sid) for sid in train_ids]
    val_indices = [df.index.get_loc(sid) for sid in val_ids]
    test_indices = [df.index.get_loc(sid) for sid in test_ids]
    
    X_train = X_disease[train_indices]
    y_train = y[train_indices]
    X_aux_train = np.zeros((len(train_indices), 0))  # Empty auxiliary features
    
    X_val = X_disease[val_indices]
    y_val = y[val_indices]
    X_aux_val = np.zeros((len(val_indices), 0))
    
    X_test = X_disease[test_indices]
    y_test = y[test_indices]
    X_aux_test = np.zeros((len(test_indices), 0))
    
    print(f"\nTrain split:")
    print(f"  X: {X_train.shape}")
    print(f"  y: Controls={np.sum(y_train==0)}, Disease={np.sum(y_train==1)}")
    print(f"\nVal split:")
    print(f"  X: {X_val.shape}")
    print(f"  y: Controls={np.sum(y_val==0)}, Disease={np.sum(y_val==1)}")
    print(f"\nTest split:")
    print(f"  X: {X_test.shape}")
    print(f"  y: Controls={np.sum(y_test==0)}, Disease={np.sum(y_test==1)}")
    
    # Save data
    print(f"\nSaving to: {output_dir}")
    
    with open(output_dir / 'X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open(output_dir / 'y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(output_dir / 'X_aux_train.pkl', 'wb') as f:
        pickle.dump(X_aux_train, f)
    
    with open(output_dir / 'X_val.pkl', 'wb') as f:
        pickle.dump(X_val, f)
    with open(output_dir / 'y_val.pkl', 'wb') as f:
        pickle.dump(y_val, f)
    with open(output_dir / 'X_aux_val.pkl', 'wb') as f:
        pickle.dump(X_aux_val, f)
    
    with open(output_dir / 'X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open(output_dir / 'y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open(output_dir / 'X_aux_test.pkl', 'wb') as f:
        pickle.dump(X_aux_test, f)
    
    # Save ground truth
    ground_truth = {
        'disease_sample_ids': disease_sample_ids,
        'disease_sample_indices': disease_sample_indices,
        'disease_gene_indices': disease_gene_indices.tolist(),
        'disease_gene_names': disease_gene_names,
        'n_disease_samples': len(disease_sample_ids),
        'n_disease_genes': len(disease_gene_names),
        'assignment_level': 'patient' if (pt_ids is not None and patient_level) else 'cell',
    }
    
    if disease_patients is not None:
        ground_truth['disease_patients'] = disease_patients.tolist()
        ground_truth['n_disease_patients'] = len(disease_patients)
    
    with open(output_dir / 'ground_truth.json', 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    # Save configuration
    config = {
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'n_disease_patients': int(n_disease_patients) if n_disease_patients else None,
        'n_disease_samples': int(n_disease_samples),  # Actual number of cells
        'n_disease_genes': n_disease_genes,
        'boost_factor': boost_factor,
        'use_train_only': use_train_only,
        'patient_level': patient_level if pt_ids is not None else False,
        'assignment_level': ground_truth['assignment_level'],
        'seed': seed,
        'total_samples': len(df),
        'total_genes': len(gene_list),
        'train_disease_samples': int(np.sum(y_train == 1)),
        'val_disease_samples': int(np.sum(y_val == 1)),
        'test_disease_samples': int(np.sum(y_test == 1))
    }
    
    if disease_patients is not None:
        config['train_disease_patients'] = int(train_disease_pts)
        config['val_disease_patients'] = int(val_disease_pts)
        config['test_disease_patients'] = int(test_disease_pts)
        config['total_patients'] = len(features['pt_id'].unique())
    
    with open(output_dir / 'synthetic_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save gene list for reference
    with open(output_dir / 'gene_list.txt', 'w') as f:
        for gene in gene_list:
            f.write(f"{gene}\n")
    
    print(f"\n{'='*70}")
    print("SYNTHETIC DATASET CREATED")
    print(f"{'='*70}")
    print(f"\nFiles saved:")
    print(f"  - X/y/X_aux for train/val/test splits")
    print(f"  - ground_truth.json (disease genes and samples)")
    print(f"  - synthetic_config.json (parameters)")
    print(f"  - gene_list.txt (gene names)")
    
    print(f"\n{'='*70}")
    print("VALIDATION METRICS TO CHECK")
    print(f"{'='*70}")
    print(f"\nAfter training, check if model:")
    print(f"  1. Recovers disease genes in top beta weights")
    print(f"  2. Has positive v-weights for disease-relevant factors")
    print(f"  3. Achieves good AUC for disease prediction")
    print(f"  4. Separates disease/control samples in theta space")
    
    return ground_truth, config


def create_multiple_scenarios(base_dir='/labs/Aguiar/SSPA_BRAY/BRay/ctrl_sspa_test',
                              output_base='/labs/Aguiar/SSPA_BRAY/BRay/synthetic_scenarios',
                              patient_level=True):
    """
    Create multiple synthetic scenarios for comprehensive testing.
    
    Scenarios:
    1. Easy: Strong signal, few genes
    2. Moderate: Medium signal, medium genes
    3. Hard: Weak signal, many genes
    """
    # First load data to get patient count
    df, features, gene_list, pt_ids = load_control_data(base_dir)
    
    if pt_ids is not None and patient_level:
        # Get training patient count
        with open(Path(base_dir) / 'data_split_cell_ids.json', 'r') as f:
            splits = json.load(f)
        train_ids = splits['train']
        train_pt_ids = features.loc[train_ids, 'pt_id'].unique()
        n_train_patients = len(train_pt_ids)
        
        # Use ~50% of training patients for disease
        n_disease_patients = int(n_train_patients * 0.5)
        
        print(f"\nPatient-level assignment:")
        print(f"  Training patients: {n_train_patients}")
        print(f"  Disease patients: {n_disease_patients} (~50%)")
        
        scenarios = [
            {
                'name': 'easy',
                'n_disease_patients': n_disease_patients,
                'n_disease_genes': 20,
                'boost_factor': 3.0,
                'description': 'Strong signal (3x), 20 genes, patient-level'
            },
            {
                'name': 'moderate', 
                'n_disease_patients': n_disease_patients,
                'n_disease_genes': 30,
                'boost_factor': 2.0,
                'description': 'Medium signal (2x), 30 genes, patient-level'
            },
            {
                'name': 'hard',
                'n_disease_patients': n_disease_patients,
                'n_disease_genes': 50,
                'boost_factor': 1.5,
                'description': 'Weak signal (1.5x), 50 genes, patient-level'
            }
        ]
    else:
        # Cell-level scenarios (not recommended for single-cell)
        n_disease_samples = 300
        
        scenarios = [
            {
                'name': 'easy',
                'n_disease_samples': n_disease_samples,
                'n_disease_genes': 20,
                'boost_factor': 3.0,
                'description': 'Strong signal (3x), 20 genes, cell-level'
            },
            {
                'name': 'moderate', 
                'n_disease_samples': n_disease_samples,
                'n_disease_genes': 30,
                'boost_factor': 2.0,
                'description': 'Medium signal (2x), 30 genes, cell-level'
            },
            {
                'name': 'hard',
                'n_disease_samples': n_disease_samples,
                'n_disease_genes': 50,
                'boost_factor': 1.5,
                'description': 'Weak signal (1.5x), 50 genes, cell-level'
            }
        ]
    
    output_base = Path(output_base)
    output_base.mkdir(exist_ok=True, parents=True)
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Creating scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'='*70}")
        
        output_dir = output_base / scenario['name']
        
        create_synthetic_dataset(
            data_dir=base_dir,
            output_dir=str(output_dir),
            n_disease_patients=scenario.get('n_disease_patients'),
            n_disease_samples=scenario.get('n_disease_samples'),
            n_disease_genes=scenario['n_disease_genes'],
            boost_factor=scenario['boost_factor'],
            use_train_only=True,
            patient_level=patient_level,
            seed=42
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create synthetic disease data')
    parser.add_argument('--data_dir', type=str, 
                       default='/labs/Aguiar/SSPA_BRAY/BRay/ctrl_sspa_test',
                       help='Control data directory')
    parser.add_argument('--output_dir', type=str,
                       default='/labs/Aguiar/SSPA_BRAY/BRay/synthetic_disease',
                       help='Output directory')
    parser.add_argument('--n_disease_patients', type=int, default=None,
                       help='Number of disease PATIENTS (for patient-level assignment)')
    parser.add_argument('--n_disease_samples', type=int, default=None,
                       help='Number of disease SAMPLES/CELLS (for cell-level assignment)')
    parser.add_argument('--n_disease_genes', type=int, default=30,
                       help='Number of disease genes')
    parser.add_argument('--boost_factor', type=float, default=2.0,
                       help='Multiplicative boost factor')
    parser.add_argument('--patient_level', action='store_true', default=True,
                       help='Use patient-level assignment (default: True)')
    parser.add_argument('--cell_level', action='store_true',
                       help='Use cell-level assignment (not recommended for single-cell)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--multiple_scenarios', action='store_true',
                       help='Create easy/moderate/hard scenarios')
    
    args = parser.parse_args()
    
    # Determine assignment level
    if args.cell_level:
        patient_level = False
    else:
        patient_level = True
    
    if args.multiple_scenarios:
        create_multiple_scenarios(
            base_dir=args.data_dir,
            output_base=args.output_dir,
            patient_level=patient_level
        )
    else:
        # Check that appropriate parameter is set
        if patient_level and args.n_disease_patients is None:
            # Try to infer from data
            print("Patient-level assignment selected but n_disease_patients not specified.")
            print("Will use 50% of training patients as disease.")
            
            # Load data to get patient count
            df, features, gene_list, pt_ids = load_control_data(args.data_dir)
            if pt_ids is None:
                raise ValueError("No patient structure found. Use --n_disease_samples for cell-level assignment.")
            
            with open(Path(args.data_dir) / 'data_split_cell_ids.json', 'r') as f:
                splits = json.load(f)
            train_ids = splits['train']
            train_pt_ids = features.loc[train_ids, 'pt_id'].unique()
            n_disease_patients = int(len(train_pt_ids) * 0.5)
            
            print(f"Setting n_disease_patients = {n_disease_patients}")
        else:
            n_disease_patients = args.n_disease_patients
        
        if not patient_level and args.n_disease_samples is None:
            raise ValueError("Cell-level assignment requires --n_disease_samples")
        
        create_synthetic_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_disease_patients=n_disease_patients,
            n_disease_samples=args.n_disease_samples,
            n_disease_genes=args.n_disease_genes,
            boost_factor=args.boost_factor,
            patient_level=patient_level,
            seed=args.seed
        )