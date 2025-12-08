import pickle  
import mygene
from gseapy import read_gmt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from pathlib import Path
from .vi import *

def prepare_and_load_emtab():
    # data_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/SSPA/dataset/ArrayExpress/preprocessed"
    data_path = "/labs/Aguiar/SSPA_BRAY/dataset/EMTAB11349/preprocessed/"
    cache_file = os.path.join(data_path, "tmm_emtab_symbol.pkl")

    # If cached converted AnnData exists, load and return it
    if os.path.exists(cache_file):
        print(f"Loading cached Ensembl-converted AnnData from {cache_file}")
        with open(cache_file, "rb") as f:
            adata = pickle.load(f)
        print(f"Loaded AnnData with shape: {adata.shape}")
        return adata

    # Otherwise, load and process the data
    gene_expression_file = "tmm_tpm_gene_expression.csv.gz"
    responses_file = "responses.csv.gz"
    aux_data_file = "aux_data.csv.gz"

    gene_expression = pd.read_csv(os.path.join(data_path, gene_expression_file), index_col=0, compression='gzip')
    responses = pd.read_csv(os.path.join(data_path, responses_file), index_col=0, compression='gzip')
    aux_data = pd.read_csv(os.path.join(data_path, aux_data_file), index_col=0, compression='gzip')

    # Concatenate all three dataframes into a single dataframe
    combined_data = pd.concat([gene_expression, responses, aux_data], axis=1)

    # Separate gene expression data (X) from labels and auxiliary variables
    gene_cols = [col for col in combined_data.columns if col not in ["Crohn's disease", "ulcerative colitis", "age", "sex_female"]]
    X = combined_data[gene_cols]
    labels = combined_data[["Crohn's disease", "ulcerative colitis"]]
    aux_vars = combined_data[["age", "sex_female"]]

    # One-hot encode labels: 0 if no disease, 1 if either disease
    disease_label = ((labels["Crohn's disease"] == 1) | (labels["ulcerative colitis"] == 1)).astype(int)
    labels_encoded = pd.DataFrame({'disease': disease_label}, index=labels.index)

    # Create AnnData object using original gene expression data
    adata = ad.AnnData(X=X)

    # Add encoded labels as obs
    adata.obs = labels_encoded.copy()
    adata.obs_names = combined_data.index

    # Add auxiliary variables as obs
    adata.obs = pd.concat([adata.obs, aux_vars], axis=1)

    # Add gene symbols as var_names (keeping original names)
    adata.var_names = X.columns.tolist()

    print(f"AnnData object created (gene symbols):")
    print(f"  - Shape: {adata.shape}")
    print(f"  - Observations (samples): {adata.n_obs}")
    print(f"  - Variables (genes): {adata.n_vars}")
    print(f"  - Obs columns: {list(adata.obs.columns)}")
    print(f"  - Disease label distribution:")
    print(adata.obs['disease'].value_counts())
    print(f"  - First few obs values:")
    print(adata.obs.head())

    # Save the AnnData to cache for future use
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(adata, f)
        print(f"Saved AnnData to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}")

    return adata