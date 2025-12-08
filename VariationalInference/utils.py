import pickle  
import mygene
from gseapy import read_gmt
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
from pathlib import Path
from .config import *


def save_cache(data, cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved cached data to {cache_file}")

def load_cache(cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def filter_protein_coding_genes(data, gene_annotation_path):
    gene_annotation = pd.read_csv(gene_annotation_path)
    gene_annotation = gene_annotation.set_index('GeneName')
    protein_coding_genes = gene_annotation[
        gene_annotation['Genetype'] == 'protein_coding'
    ].index.tolist()
    
    common_genes = [g for g in data.gene_names if g in protein_coding_genes]
    
    print(f"Total genes: {len(data.gene_names)}")
    print(f"Protein-coding genes found: {len(common_genes)}")
    print(f"Filtered out: {len(data.gene_names) - len(common_genes)} non-protein-coding genes")
    
    filtered_data = data.subset_genes(common_genes)
    return filtered_data

def QCscRNAsizeFactorNormOnly(X):

    X = X.astype(float)

    if sp.issparse(X):
        UMI_counts_per_cell = np.array(X.sum(axis=1)).flatten()
    else:  
        UMI_counts_per_cell = X.sum(axis=1)

    median_UMI = np.median(UMI_counts_per_cell)
    scaling_factors = median_UMI / UMI_counts_per_cell
    scaling_factors[np.isinf(scaling_factors)] = 0  

    if sp.issparse(X):
        scaling_matrix = sp.diags(scaling_factors)
        X = scaling_matrix @ X  
    else:
        X = X * scaling_factors[:, np.newaxis]  

    return X


def to_dense_array(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)

def create_gene_id_mapping(target_format='ensembl', species='mouse', use_cache=True):
    """
    Create mapping between gene symbols and other ID formats (e.g., Ensembl).
    
    Args:
        target_format: Target ID format ('ensembl' or 'symbol')
        species: Species name (default: 'mouse')
        use_cache: Whether to use cached mapping if available
        
    Returns:
        Dictionary mapping from symbol to target format
    """
    # Check cache first
    cache_file = CACHE_DIR / f"gene_mapping_{species}_{target_format}.pkl"
    
    if use_cache and cache_file.exists():
        print(f"Loading gene mapping from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Load all unique genes from seed files to create comprehensive mapping
    cyto_genes = pd.read_csv(cytoseeds_csv_path)['V4'].tolist()
    ap_genes = pd.read_csv(apseeds_csv_path)['x'].tolist()
    ig_genes = pd.read_csv(igseeds_csv_path)['Symbol'].tolist()
    all_genes = list(set(cyto_genes + ap_genes + ig_genes))
    
    print(f"Creating gene ID mapping for {len(all_genes)} unique genes...")
    
    mg = mygene.MyGeneInfo()
    gene_info = mg.querymany(all_genes, 
                            scopes='symbol,ensembl.gene', 
                            fields='symbol,ensembl.gene', 
                            species=species, 
                            returnall=True)
    
    gene_mapping = {}
    
    for gene in gene_info['out']:
        if 'symbol' in gene and 'ensembl' in gene:
            symbol = gene['symbol']
            # Handle both dict and list formats for ensembl field
            if isinstance(gene['ensembl'], dict):
                ensembl_id = gene['ensembl'].get('gene')
            elif isinstance(gene['ensembl'], list) and len(gene['ensembl']) > 0:
                ensembl_id = gene['ensembl'][0].get('gene')
            else:
                continue
            
            if target_format.lower() == 'ensembl':
                gene_mapping[symbol] = ensembl_id
            elif target_format.lower() == 'symbol':
                gene_mapping[ensembl_id] = symbol
    
    print(f"Mapped {len(gene_mapping)} genes to {target_format}")
    
    # Cache the mapping
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump(gene_mapping, f)
        print(f"Saved gene mapping to cache: {cache_file}")
    
    return gene_mapping




