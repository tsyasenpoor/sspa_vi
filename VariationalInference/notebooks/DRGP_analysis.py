import numpy as np
import anndata as ad
import sys
import joblib
from pathlib import Path

sys.path.insert(0, '/labs/Aguiar/SSPA_BRAY/scHPF')

# Define the base path and experiment folders
base_path = Path("/labs/Aguiar/SSPA_BRAY/results/spectra")
experiments = ["exp0_easy", "exp1_medium", "exp2_hard", "exp3_intersectional"]

# Dictionary to store loaded data
spectra_data = {}

# Load data for each experiment
for exp in experiments:
    exp_path = base_path / exp
    
    spectra_data[exp] = {
        "cell_scores": np.load(exp_path / "spectra_cell_scores.npy"),
        "factors": np.load(exp_path / "spectra_factors.npy"),
        "adata": ad.read_h5ad(exp_path / "spectra_adata.h5ad")
    }
    
    print(f"Loaded {exp}:")
    print(f"  - cell_scores shape: {spectra_data[exp]['cell_scores'].shape}")
    print(f"  - factors shape: {spectra_data[exp]['factors'].shape}")
    print(f"  - adata shape: {spectra_data[exp]['adata'].shape}")

# Load scHPF results for each experiment
schpf_base_path = Path("/labs/Aguiar/SSPA_BRAY/results/schpf")
schpf_data = {}

for exp in experiments:
    exp_path = schpf_base_path / exp
    joblib_file = exp_path / f"{exp}.scHPF_K50_b0_5trials.joblib"
    
    model = joblib.load(joblib_file)
    schpf_data[exp] = model
    
    print(f"Loaded scHPF {exp}:")
    print(f"  - model type: {type(model)}")
    print(f"  - model attributes: {vars(model).keys()}")

