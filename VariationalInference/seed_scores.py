import pickle  
import mygene
from gseapy import read_gmt
import numpy as np
import pandas as pd
import os
from pathlib import Path
from .config import *
from .utils import *
from .data import *

def load_seed_genes(gene_format='symbol', use_cache=True):
    # Load raw gene lists (assumed to be in symbol format)
    cyto_seed_genes = pd.read_csv(cytoseeds_csv_path)['V4'].tolist()
    ap_seed_genes = pd.read_csv(apseeds_csv_path)['x'].tolist()
    ig_seed_genes = list(set(pd.read_csv(igseeds_csv_path)['Symbol'].tolist()))
    
    # If symbol format requested, return as-is (input is already symbols)
    if gene_format.lower() == 'symbol':
        return cyto_seed_genes, ap_seed_genes, ig_seed_genes
    
    # Create or load cached gene mapping for the requested format
    gene_mapping = create_gene_id_mapping(target_format=gene_format, use_cache=use_cache)
    
    # Apply mapping to convert gene names
    cyto_seed_genes = [gene_mapping.get(gene, gene) for gene in cyto_seed_genes]
    ap_seed_genes = [gene_mapping.get(gene, gene) for gene in ap_seed_genes]
    ig_seed_genes = [gene_mapping.get(gene, gene) for gene in ig_seed_genes]
    
    return cyto_seed_genes, ap_seed_genes, ig_seed_genes

