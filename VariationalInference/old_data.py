import pickle  
import mygene
from gseapy import read_gmt
import numpy as np
import pandas as pd
import os
from memory_tracking import get_memory_usage, log_memory, log_array_sizes, clear_memory


# Force JAX to use CPU only - must be set before importing jax
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Log initial memory
print(f"Initial memory usage: {get_memory_usage():.2f} MB")

from vi_model_complete import run_model_and_evaluate

log_memory("Before loading data files")

# cytoseeds_csv_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_FileTransfer/Seed genes/CYTOBEAM_Cytokines_KEGGPATHWAY_addedMif.csv"
cytoseeds_csv_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Seed genes/CYTOBEAM_Cytokines_KEGGPATHWAY_addedMif.csv"
CYTOSEEDS_df = pd.read_csv(cytoseeds_csv_path)
CYTOSEEDS = CYTOSEEDS_df['V4'].tolist() #173
log_memory("After loading CYTOSEEDS")

mg = mygene.MyGeneInfo()


def save_cache(data, cache_file):
    """Save data to a cache file using pickle."""
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved cached data to {cache_file}")

def load_cache(cache_file):
    """Load data from a cache file if it exists."""
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def convert_pathways_to_ensembl(pathways, cache_file="/labs/Aguiar/SSPA_BRAY/BRay/pathways_ensembl_cache.pkl"):
    # log_memory("Before convert_pathways_to_ensembl")
    
    # Try to load from cache first
    cached_pathways = load_cache(cache_file)
    if cached_pathways is not None:
        # log_memory("After loading pathways from cache")

        filtered_pathways = {}
        excluded_keywords = ["ADME", "DRUG", "MISCELLANEOUS", "EMT"]

        total_pathways = len(cached_pathways)
        reactome_count = sum(1 for k in cached_pathways if k.startswith('REACTOME'))
        
        for k, v in cached_pathways.items():
            if k.startswith('REACTOME'):
                if not any(keyword in k.upper() for keyword in excluded_keywords):
                    filtered_pathways[k] = v
        
        print(f"Original pathways count: {total_pathways}")
        print(f"Reactome pathways count: {reactome_count}")
        print(f"Filtered pathways count (REACTOME only, excluding keywords): {len(filtered_pathways)}")
        
        # Define specific pathways to exclude
        specific_exclude = [
            "REACTOME_GENERIC_TRANSCRIPTION_PATHWAY",
            "REACTOME_ADAPTIVE_IMMUNE_SYSTEM",
            "REACTOME_INNATE_IMMUNE_SYSTEM",
            "REACTOME_IMMUNE_SYSTEM",
            "REACTOME_METABOLISM"
        ]
        
        # Remove specific pathways
        for pathway in specific_exclude:
            if pathway in filtered_pathways:
                filtered_pathways.pop(pathway)
            
        excluded_count = reactome_count - len(filtered_pathways)
        print(f"Reactome pathways excluded due to keywords: {excluded_count}")
        cached_pathways = filtered_pathways
        return cached_pathways
    
    print("Cache not found. Converting pathways to Ensembl IDs...")
    mg = mygene.MyGeneInfo()
    unique_genes = set()
    for genes in pathways.values():
        unique_genes.update(genes)
    gene_list = list(unique_genes)
    print(f"Number of unique genes for conversion: {len(gene_list)}")
    
    mapping = {}
    batch_size = 100  # processing in batches for memory efficiency
    for i in range(0, len(gene_list), batch_size):
        batch = gene_list[i:i+batch_size]
        query_results = mg.querymany(batch, scopes='symbol', fields='ensembl.gene', species='mouse', returnall=False)
        for hit in query_results:
            if 'ensembl' in hit:
                if isinstance(hit['ensembl'], list):
                    mapping[hit['query']] = hit['ensembl'][0]['gene']
                else:
                    mapping[hit['query']] = hit['ensembl']['gene']
            else:
                mapping[hit['query']] = hit['query']  # keep original if no conversion found
    
    new_pathways = {}
    for pathway, genes in pathways.items():
        new_pathways[pathway] = [mapping.get(g, g) for g in genes]
    
    # Save to cache for future use
    save_cache(new_pathways, cache_file)
    
    # log_memory("After convert_pathways_to_ensembl")
    return new_pathways

#Load pre-filtered pathways directly from pickle
import pickle
pathways_pkl_path = "/labs/Aguiar/SSPA_BRAY/BRay/filtered_pathways_cache.pkl"
print(f"Loading pre-filtered pathways from {pathways_pkl_path}")
with open(pathways_pkl_path, 'rb') as f:
    pathways = pickle.load(f)

# print("DEBUG: type(pathways) =", type(pathways))

# # If pathways is a DataFrame, convert to dict of lists
# if isinstance(pathways, pd.DataFrame):
#     print("Converting pathways DataFrame to dict of lists...")
#     pathways_dict = {}
#     for col in pathways.columns:
#         gene_list = pathways.index[pathways[col] == 1].tolist()
#         pathways_dict[col] = gene_list
#     pathways = pathways_dict

# Robustly flatten any nested lists in pathways
# import collections.abc

# def flatten(l):
#     for el in l:
#         if isinstance(el, collections.abc.Iterable) and not isinstance(el, str):
#             yield from flatten(el)
#         else:
#             yield el

# for k, v in pathways.items():
#     pathways[k] = list(flatten(v))

# print(f"Loaded {len(pathways)} pre-filtered pathways")


def batch_query(genes, batch_size=100):
    results = []
    for i in range(0, len(genes), batch_size):
        batch = genes[i:i+batch_size]
        results.extend(mg.querymany(batch, scopes='symbol', fields='ensembl.gene', species='mouse'))
    return results

# Define a cache file for CYTOSEED conversions
# cytoseed_cache_file = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/cytoseed_ensembl_cache.pkl"
cytoseed_cache_file = "/labs/Aguiar/SSPA_BRAY/BRay/cytoseed_ensembl_cache.pkl"
# Try to load CYTOSEED mappings from cache
symbol_to_ensembl_asg = load_cache(cytoseed_cache_file)

if symbol_to_ensembl_asg is None:
    log_memory("Before batch query")
    query_results = batch_query(CYTOSEEDS, batch_size=100)
    log_memory("After batch query")
    
    symbol_to_ensembl_asg = {}
    for entry in query_results:
        if 'ensembl' in entry and 'gene' in entry['ensembl']:
            if isinstance(entry['ensembl'], list):
                symbol_to_ensembl_asg[entry['query']] = entry['ensembl'][0]['gene']
            else:
                symbol_to_ensembl_asg[entry['query']] = entry['ensembl']['gene']
        else:
            symbol_to_ensembl_asg[entry['query']] = None 
    
    # Save to cache for future use
    save_cache(symbol_to_ensembl_asg, cytoseed_cache_file)

CYTOSEED_ensembl = [symbol_to_ensembl_asg.get(gene) for gene in CYTOSEEDS if symbol_to_ensembl_asg.get(gene)]
print(f"CYTOSEED_ensembl length: {len(CYTOSEED_ensembl)}")

# log_memory("Before reading pathways")
# pathways_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_FileTransfer/human_pathways/ReactomePathways.gmt"
pathways_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/m2.cp.v2024.1.Mm.symbols.gmt"

pathways = read_gmt(pathways_path)  # 1730 pathways
print(f"Number of pathways: {len(pathways)}")
log_memory("After reading pathways")

# def filter_pathways(pathways_dict, min_genes=10, max_genes=500, exclude_disease=True, exclude_drug_metabolism=True):
#     """
#     Filter pathways based on various criteria
    
#     Args:
#         pathways_dict: Dictionary with pathway names as keys and gene lists as values
#         min_genes: Minimum number of genes required in pathway
#         max_genes: Maximum number of genes allowed in pathway  
#         exclude_disease: Whether to exclude disease-related pathways
#         exclude_drug_metabolism: Whether to exclude drug metabolism pathways
    
#     Returns:
#         Filtered pathways dictionary
#     """
#     filtered_pathways = {}
    
#     for pathway_name, genes in pathways_dict.items():
#         gene_count = len(genes)
        
#         # Size filter
#         if gene_count < min_genes or gene_count > max_genes:
#             continue
            
#         # Disease/defect filter
#         if exclude_disease:
#             disease_keywords = ['defective', 'disease', 'cancer', 'tumor', 'carcinoma', 'syndrome', 'infection',
#                               'deficiency', 'disorder', 'mutation', 'variant', 'resistant']
#             if any(keyword in pathway_name.lower() for keyword in disease_keywords):
#                 continue
        
#         # Drug metabolism filter
#         if exclude_drug_metabolism:
#             drug_keywords = ['drug', 'response to', 'aspirin', 'adme', 'metabolism', 'biotransformation', 
#                            'glucuronidation', 'cytochrome', 'cyp', 'sulfation', 'antiviral']
#             if any(keyword in pathway_name.lower() for keyword in drug_keywords):
#                 continue
        
#         # If pathway passes all filters, include it
#         filtered_pathways[pathway_name] = genes
    
#     print(f"Original pathways: {len(pathways_dict)}")
#     print(f"After filtering: {len(filtered_pathways)}")
    
#     # Show size distribution of filtered pathways
#     sizes = [len(genes) for genes in filtered_pathways.values()]
#     if sizes:
#         print(f"Gene count range: {min(sizes)} - {max(sizes)}")
#         print(f"Mean gene count: {sum(sizes)/len(sizes):.1f}")
    
#     return filtered_pathways

# filtered_pathways = filter_pathways(pathways, min_genes=10, max_genes=500, exclude_disease=True, exclude_drug_metabolism=True)

# ibd_keywords = [
#     'inflam', 'immune', 'cytokine', 'interleukin', 'interferon',
#     'tnf', 'nf-kb', 'nfkb', 'jak', 'stat', 'il-', 'il1', 'il6', 'il17', 'il23',
#     'crohn', 'colitis', 'bowel', 'intestin', 'gut', 'gastro',
#     'autophagy', 'apoptosis', 'epithelial', 'barrier',
#     't cell', 'b cell', 'lymphocyte', 'macrophage', 'neutrophil',
#     'chemokine', 'complement', 'toll', 'myd88',
#     'bacterial', 'pathogen', 'microbiome'
# ]

# def filter_ibd_pathways(pathway_names):
#     ibd_pathways = []
#     for i, pathway in enumerate(pathway_names):
#         pathway_lower = pathway.lower()
#         if any(keyword in pathway_lower for keyword in ibd_keywords):
#             ibd_pathways.append(i)
#     return ibd_pathways

# # Define a cache file for filtered pathways
# IBD_filtered_human_pathways_cache_file = "/labs/Aguiar/SSPA_BRAY/BRay/IBD_filtered_human_pathways_cache.pkl"
# # filtered_human_pathways_cache_file = "/labs/Aguiar/SSPA_BRAY/BRay/filtered_human_pathways_cache.pkl"

# # Try to load filtered pathways from cache
# IBD_filtered_pathways = load_cache(IBD_filtered_human_pathways_cache_file)

# if IBD_filtered_pathways is None:
#     # Apply pathway filtering to the loaded pathways
#     ibd_pathway_indices = filter_ibd_pathways(filtered_pathways.keys())
#     pathway_names = list(filtered_pathways.keys())
#     IBD_filtered_pathways = {pathway_names[i]: filtered_pathways[pathway_names[i]] for i in ibd_pathway_indices}
#     # Save filtered pathways to cache for future use
#     save_cache(IBD_filtered_pathways, IBD_filtered_human_pathways_cache_file)
#     print(f"Filtered pathways saved to cache: {IBD_filtered_human_pathways_cache_file}")
# else:
#     print(f"Loaded filtered pathways from cache: {IBD_filtered_human_pathways_cache_file}")

# # Use the filtered pathways
# pathways = IBD_filtered_pathways
# print(f"Using {len(pathways)} filtered pathways for analysis")

pathways = convert_pathways_to_ensembl(pathways)  
log_memory("After converting pathways to ensembl")

# nap_file_path_raw = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/GEX_NAP_filt_raw_modelingonly_2024-02-05.csv"
# nap_metadata_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/meta_NAP_unfilt_fullData_2024-02-05.csv"

# Updated to use RDS file instead of CSV
# ajm_file_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
# ajm_metadata_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"
ajm_file_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
ajm_metadata_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"

def prepare_ajm_dataset(cache_file="/labs/Aguiar/SSPA_BRAY/BRay/ajm_dataset_cache.h5ad"):
# def prepare_ajm_dataset(cache_file="/labs/Aguiar/SSPA_BRAY/BRay/ajm_dataset_cache.h5ad"):
    print("Loading AJM dataset...")
    log_memory("Before loading AJM dataset")
    
    # Import required modules at the function's top level
    import os
    import subprocess
    import scipy.sparse as sp
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"Loading AnnData object from cache file: {cache_file}")
        try:
            adata = ad.read_h5ad(cache_file)
            print(f"Successfully loaded cached AnnData object with shape: {adata.shape}")
            
            # Extract the dataset splits
            ajm_ap_samples = adata[adata.obs['dataset'] == 'ap']
            ajm_cyto_samples = adata[adata.obs['dataset'] == 'cyto']

            # Normalize and log-transform
            QCscRNAsizeFactorNormOnly(ajm_ap_samples)
            QCscRNAsizeFactorNormOnly(ajm_cyto_samples)
            
            print("AJM AP Samples distribution:")
            print(ajm_ap_samples.obs['ap'].value_counts())

            print("AJM CYTO Samples distribution:")
            print(ajm_cyto_samples.obs['cyto'].value_counts())
            
            log_memory("After loading AnnData from cache")
            return ajm_ap_samples, ajm_cyto_samples
        except Exception as e:
            print(f"Error loading cached AnnData: {e}")
            print("Proceeding with RDS conversion...")
    else:
        print(f"Cache file {cache_file} not found, converting RDS file...")
    
    # Run the R script to convert the RDS file to CSV files
    print("Converting RDS to anndata format using R...")
    r_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rds_to_anndata.R")
    
    try:
        result = subprocess.run(["Rscript", r_script_path], 
                               capture_output=True, 
                               text=True, 
                               check=True)
        print("R conversion output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("R conversion error:")
        print(e.stdout)
        print(e.stderr)
        raise RuntimeError("Failed to convert RDS file to CSV")
    
    log_memory("After running R script for conversion")
    
    # Load the sparse matrix data
    print("Loading sparse matrix data...")
    sparse_data = pd.read_csv("matrix_sparse.csv")
    
    # Load row and column names
    row_names = pd.read_csv("matrix_rownames.csv")["row_names"].tolist()  # These are gene names
    col_names = pd.read_csv("matrix_colnames.csv")["col_names"].tolist()  # These are cell names
    
    # Load matrix dimensions (if available)
    matrix_dims = None
    if os.path.exists("matrix_dims.csv"):
        matrix_dims = pd.read_csv("matrix_dims.csv")
        nrows = matrix_dims["rows"].iloc[0]
        ncols = matrix_dims["cols"].iloc[0]
        print(f"Matrix dimensions from file: {nrows} x {ncols}")
        # Verify that dimensions match the length of row and column names
        if nrows != len(row_names) or ncols != len(col_names):
            print(f"WARNING: Dimension mismatch! Row names: {len(row_names)}, Column names: {len(col_names)}")
    else:
        print("Matrix dimensions file not found, using length of row and column names")
        nrows = len(row_names)
        ncols = len(col_names)
    
    print(f"Sparse data shape: {sparse_data.shape}")
    print(f"Number of genes (rows in original matrix): {len(row_names)}")
    print(f"Number of cells (columns in original matrix): {len(col_names)}")
    
    # Create sparse matrix from the CSV data
    # The sparse data has format: row (gene), col (cell), value
    # Indices should already be 0-based from R script
    row_indices = sparse_data["row"].values  # Gene indices
    col_indices = sparse_data["col"].values  # Cell indices
    values = sparse_data["value"].values
    
    # Debug information
    print(f"Row indices range: {row_indices.min()} to {row_indices.max()}")
    print(f"Column indices range: {col_indices.min()} to {col_indices.max()}")
    
    # In AnnData, rows are cells (observations) and columns are genes (variables)
    # So we need to transpose the matrix from our R output
    print(f"Creating sparse matrix with shape: ({nrows}, {ncols}) and transposing to match AnnData format")
    
    # Create a sparse COO matrix with original shape
    sparse_matrix = sp.coo_matrix((values, (row_indices, col_indices)), 
                                 shape=(nrows, ncols))
    
    # Transpose the matrix to have cells as rows and genes as columns
    sparse_matrix = sparse_matrix.transpose().tocsr()
    
    # Now the shape is (ncols, nrows) - (cells, genes)
    print(f"Transposed matrix shape: {sparse_matrix.shape}")
    
    log_memory("After loading sparse matrix")
    
    # Create AnnData object with transposed matrix where:
    # - Rows (observations) are cells
    # - Columns (variables) are genes
    ajm_adata = ad.AnnData(X=sparse_matrix)
    
    # In AnnData:
    # - obs_names (rows) should be cell names
    # - var_names (columns) should be gene names
    ajm_adata.obs_names = col_names  # Cell names as observation names
    ajm_adata.var_names = row_names  # Gene names as variable names
    
    log_memory("After creating AnnData object")
    
    print(f"AnnData object created with shape: {ajm_adata.shape}")
    
    # Load metadata separately
    ajm_features = pd.read_csv(ajm_metadata_path, index_col=0)
    log_memory("After loading AJM metadata")
    
    print(f"AJM features shape: {ajm_features.shape}")
    
    # Create label mappings
    ajm_label_mapping = {
        'TC-0hr':       {'ap':0,'cyto':0,'ig':-1},
        'TC-LPS-3hr':   {'ap':-1,'cyto':0,'ig':-1},
        'TC-LPS-6hr':   {'ap':1,'cyto':-1,'ig':-1},
        'TC-LPS-24hr':  {'ap':1,'cyto':1,'ig':-1},
        'TC-LPS-48hr':  {'ap':-1,'cyto':-1,'ig':-1},
        'TC-LPS-72hr':  {'ap':-1,'cyto':-1,'ig':-1},
    }
    
    # Initialize label columns in metadata
    for col in ['ap','cyto','ig']:
        ajm_features[col] = None
    
    # Apply mapping based on sample values
    for sample_value, labels in ajm_label_mapping.items():
        mask = ajm_features['sample'] == sample_value
        for col in labels:
            ajm_features.loc[mask,col] = labels[col]

    # Set cell_id as index if available
    if 'cell_id' in ajm_features.columns:
        ajm_features.set_index('cell_id', inplace=True)
    
    # Ensure cell IDs match between anndata and metadata
    common_cells_ajm = ajm_adata.obs_names.intersection(ajm_features.index)
    print(f"Number of common cells: {len(common_cells_ajm)}")
    
    # Subset to common cells
    ajm_adata = ajm_adata[common_cells_ajm]
    ajm_features = ajm_features.loc[common_cells_ajm]
    
    # Add metadata to AnnData object
    for col in ajm_features.columns:
        ajm_adata.obs[col] = ajm_features[col].values
    
    # Ensure gene symbols are available
    if 'gene_symbols' not in ajm_adata.var:
        ajm_adata.var['gene_symbols'] = ajm_adata.var_names
    
    # Create subset AnnData objects for different analyses
    ajm_ap_samples = ajm_adata[ajm_adata.obs['ap'].isin([0,1])]
    ajm_cyto_samples = ajm_adata[ajm_adata.obs['cyto'].isin([0,1])]

    # Normalize and log-transform
    QCscRNAsizeFactorNormOnly(ajm_ap_samples)
    QCscRNAsizeFactorNormOnly(ajm_cyto_samples)

    # Add dataset identifier to help with cache loading
    ajm_ap_samples.obs['dataset'] = 'ap'
    ajm_cyto_samples.obs['dataset'] = 'cyto'
    
    # Combine both datasets for caching
    combined_adata = ad.concat(
        [ajm_ap_samples, ajm_cyto_samples],
        join='outer',
        merge='same'
    )
    
    # Make sure all values in .obs are serializable (convert to string if needed)
    for col in combined_adata.obs.columns:
        if combined_adata.obs[col].dtype == 'object':
            combined_adata.obs[col] = combined_adata.obs[col].astype(str)
        
        # Convert None/NaN values to strings to avoid serialization issues
        combined_adata.obs[col] = combined_adata.obs[col].fillna('NA')
    
    # Similarly ensure .var values are serializable
    for col in combined_adata.var.columns:
        if combined_adata.var[col].dtype == 'object':
            combined_adata.var[col] = combined_adata.var[col].astype(str)
        
        # Convert None/NaN values to strings
        combined_adata.var[col] = combined_adata.var[col].fillna('NA')
    
    # Save to cache file
    print(f"Saving AnnData object to cache file: {cache_file}")
    try:
        combined_adata.write_h5ad(cache_file)
        print(f"Successfully saved AnnData to cache")
    except TypeError as e:
        print(f"Error saving AnnData to H5AD: {e}")
        print("Will proceed without caching.")
    
    print("AJM AP Samples distribution:")
    print(ajm_ap_samples.obs['ap'].value_counts())

    print("AJM CYTO Samples distribution:")
    print(ajm_cyto_samples.obs['cyto'].value_counts())
    
    # Log memory usage of created AnnData objects
    log_array_sizes({
        'ajm_adata.X': ajm_adata.X,
        'ajm_ap_samples.X': ajm_ap_samples.X,
        'ajm_cyto_samples.X': ajm_cyto_samples.X
    })
    
    # Try to clear some memory
    clear_memory()
    
    # Clean up the temporary CSV files
    temp_files = ["matrix_sparse.csv", "matrix_rownames.csv", "matrix_colnames.csv", "matrix_dims.csv"]
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Temporary file {file} removed")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file}: {e}")
    
    return ajm_ap_samples, ajm_cyto_samples

def prepare_and_load_thyroid():
    """
    Load and prepare Thyroid Cancer dataset from preprocessed files.
    
    Returns:
        adata: AnnData object containing gene expression data with gene symbols as var_names,
               and Clinical_History label and auxiliary variables in .obs
    """
    import pickle
    import mygene
    import numpy as np
    import scipy.sparse as sp
    data_path = "/labs/Aguiar/SSPA_BRAY/BRay/Thyroid_Cancer/preprocessed"
    gene_expression_file = "gene_expression.csv.gz"
    responses_file = "responses.csv.gz"
    aux_data_file = "aux_data.csv.gz"

    # Load data with proper handling of mixed types and set Sample_ID as index
    gene_expression = pd.read_csv(os.path.join(data_path, gene_expression_file), compression='gzip', low_memory=False, index_col='Sample_ID')
    responses = pd.read_csv(os.path.join(data_path, responses_file), compression='gzip', index_col='Sample_ID')
    aux_data = pd.read_csv(os.path.join(data_path, aux_data_file), compression='gzip', index_col='Sample_ID')

    # Concatenate all three dataframes into a single dataframe
    combined_data = pd.concat([gene_expression, responses, aux_data], axis=1)

    # Separate gene expression data (X) from labels and auxiliary variables
    gene_cols = [col for col in combined_data.columns if col not in ["Clinical_History", "Age", "sex_female"]]
    X = combined_data[gene_cols]
    labels = combined_data[["Clinical_History"]]
    aux_vars = combined_data[["Age", "sex_female"]]

    # Convert gene expression to numeric, handling any non-numeric values
    print("Converting gene expression data to numeric...")
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Fill any NaN values that resulted from conversion with 0
    X = X.fillna(0)
    
    # Ensure X is float type for numerical operations
    X = X.astype(np.float64)
    
    print(f"Gene expression matrix shape: {X.shape}")
    print(f"Gene expression data type: {X.dtypes.iloc[0]}")
    print(f"Gene expression min: {X.min().min()}, max: {X.max().max()}")

    adata = ad.AnnData(X=X)

    # Add labels as obs
    adata.obs = labels.copy()
    adata.obs_names = combined_data.index

    # Add auxiliary variables as obs
    adata.obs = pd.concat([adata.obs, aux_vars], axis=1)
    
    # Ensure all numeric columns are proper numeric types (not strings)
    for col in ['Clinical_History', 'Age', 'sex_female']:
        if col in adata.obs.columns:
            adata.obs[col] = pd.to_numeric(adata.obs[col], errors='coerce')
    
    # Check for any NaN values that might cause issues
    print(f"NaN values in obs columns:")
    for col in adata.obs.columns:
        nan_count = adata.obs[col].isna().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaN values")
    
    # Fill any NaN values with appropriate defaults
    if 'Clinical_History' in adata.obs.columns:
        adata.obs['Clinical_History'] = adata.obs['Clinical_History'].fillna(0)
    if 'Age' in adata.obs.columns:
        adata.obs['Age'] = adata.obs['Age'].fillna(adata.obs['Age'].median())
    if 'sex_female' in adata.obs.columns:
        adata.obs['sex_female'] = adata.obs['sex_female'].fillna(0)

    # Make variable names unique to avoid warnings
    adata.var_names_make_unique()

    print(f"AnnData object created for Thyroid Cancer dataset:")
    print(f"  - Shape: {adata.shape}")
    print(f"  - Observations (samples): {adata.n_obs}")
    print(f"  - Variables (genes): {adata.n_vars}")
    print(f"  - Obs columns: {list(adata.obs.columns)}")
    print(f"  - Clinical_History label distribution:")
    print(adata.obs['Clinical_History'].value_counts())
    print(f"  - Clinical_History data type: {adata.obs['Clinical_History'].dtype}")
    print(f"  - First few obs values:")
    print(adata.obs.head())

    return adata

def prepare_and_load_emtab():
    """
    Load and prepare EMTAB dataset from preprocessed files, converting gene symbols to Ensembl IDs.
    Uses mygene to convert gene names, and caches the converted AnnData object as a pickle file.
    On subsequent runs, loads the converted AnnData from the pickle if it exists.
    
    Returns:
        adata: AnnData object containing gene expression data with Ensembl IDs as var_names,
               and labels and auxiliary variables in .obs
    """
    import pickle
    import mygene
    import numpy as np
    import scipy.sparse as sp
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

def load_data_simulation():
    """
    Load the simulation dataset and return as an AnnData object.
    - Loads gene expression, aux data, and real responses only.
    - Returns: AnnData object (gene expression as X, aux/response as obs columns).
    """

    # main_path = '/labs/Aguiar/SSPA_BRAY/dataset/simulation_base'
    main_path = '/labs/Aguiar/SSPA_BRAY/BRay/simulation_large_scale'
    gene_expression_path = os.path.join(main_path, 'gene_expression.csv.gz')
    aux_data_path = os.path.join(main_path, 'aux_data.csv.gz')
    responses_sampled_path = os.path.join(main_path, 'responses_sampled.csv.gz')
    responses_real_path = os.path.join(main_path, 'responses_real.csv.gz')

    # Load gene expression
    gene_expression_df = pd.read_csv(gene_expression_path, compression='gzip')
    obs_names = gene_expression_df['Sample_ID'].astype(str).values  # Convert to strings
    gene_expression_df = gene_expression_df.drop(columns=['Sample_ID'])
    X = gene_expression_df.values
    var_names = gene_expression_df.columns.tolist()

    # Load aux data
    aux_data_df = pd.read_csv(aux_data_path, compression='gzip')
    aux_data_df = aux_data_df.drop(columns=['Sample_ID'])

    # Load only real responses (not sampled ones)
    responses_real_df = pd.read_csv(responses_real_path, compression='gzip')
    responses_real_df = responses_real_df.drop(columns=['Sample_ID'])
    # Keep the original column names since we're only using real responses

    # Combine obs columns
    obs = pd.concat([aux_data_df, responses_real_df], axis=1)
    obs.index = obs_names

    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs)
    adata.obs_names = obs_names
    adata.var_names = var_names

    return adata

log_memory("Before loading gene annotations")
# gene_annotation_path = "/mnt/research/aguiarlab/proj/SSPA-BRAY/BRay/BRAY_FileTransfer/ENS_mouse_geneannotation.csv"
gene_annotation_path = "/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/ENS_mouse_geneannotation.csv"
gene_annotation = pd.read_csv(gene_annotation_path)
gene_annotation = gene_annotation.set_index('GeneID')
log_memory("After loading gene annotations")

def filter_protein_coding_genes(adata, gene_annotation):
    log_memory("Before filtering protein coding genes")
    protein_coding_genes = gene_annotation[gene_annotation['Genetype'] == 'protein_coding'].index
    
    common_genes = np.intersect1d(adata.var_names, protein_coding_genes)
    
    print(f"Total genes: {adata.n_vars}")
    print(f"Protein-coding genes found: {len(common_genes)}")
    
    adata_filtered = adata[:, common_genes].copy()

    log_memory("After filtering protein coding genes")
    log_array_sizes({
        'adata.X': adata.X,
        'adata_filtered.X': adata_filtered.X
    })
    
    return adata_filtered


def QCscRNAsizeFactorNormOnly(adata):
    """Normalize counts in an AnnData object using a median-based size factor per cell (row-wise)."""
    import numpy as np
    import scipy.sparse as sp

    X = adata.X.astype(float)

    if sp.issparse(X):
        UMI_counts_per_cell = np.array(X.sum(axis=1)).flatten()  # Sum over columns → per row (cell)
    else:
        UMI_counts_per_cell = X.sum(axis=1)

    median_UMI = np.median(UMI_counts_per_cell)
    scaling_factors = median_UMI / UMI_counts_per_cell
    scaling_factors[np.isinf(scaling_factors)] = 0  # Avoid inf if dividing by zero

    if sp.issparse(X):
        scaling_matrix = sp.diags(scaling_factors)
        X = scaling_matrix @ X  # Multiply from the left: row-wise scaling
    else:
        X = X * scaling_factors[:, np.newaxis]  # Broadcast scaling per row

    adata.X = X
    return adata


# def scale_to_reasonable_counts(X, target_max=100):
#     """Scale data to reasonable count range while preserving structure"""
#     percentile_99 = np.percentile(X, 99)
#     scale_factor = target_max / percentile_99
#     return np.round(X * scale_factor).astype(int)

def sample_adata(adata, n_cells=None, cell_fraction=None,
                 n_genes=None, gene_fraction=None, random_state=0):
    """Return a random subset of the AnnData object.

    Parameters
    ----------
    adata : AnnData
        Input dataset.
    n_cells : int, optional
        Number of cells to sample.  Mutually exclusive with ``cell_fraction``.
    cell_fraction : float, optional
        Fraction of cells to sample.
    n_genes : int, optional
        Number of genes to sample.  Mutually exclusive with ``gene_fraction``.
    gene_fraction : float, optional
        Fraction of genes to sample.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """

    rng = np.random.default_rng(random_state)

    if cell_fraction is not None:
        n_cells = max(1, int(adata.n_obs * cell_fraction))
    if n_cells is None or n_cells > adata.n_obs:
        n_cells = adata.n_obs
    cell_indices = rng.choice(adata.n_obs, size=n_cells, replace=False)

    if gene_fraction is not None:
        n_genes = max(1, int(adata.n_vars * gene_fraction))
    if n_genes is None or n_genes > adata.n_vars:
        n_genes = adata.n_vars
    gene_indices = rng.choice(adata.n_vars, size=n_genes, replace=False)

    return adata[cell_indices, :][:, gene_indices].copy()


def load_signal_noise_datasets(base_path="dataset/signal_noise_ratios"):
    """
    Load all signal-to-noise ratio datasets and return as a dictionary.
    
    Args:
        base_path: Path to the signal_noise_ratios directory
        
    Returns:
        Dictionary with dataset names as keys and (adata, label_col) tuples as values
    """
    import json
    
    # Load dataset information
    info_path = os.path.join(base_path, 'dataset_info.json')
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    datasets = {}
    
    for scenario_name, info in dataset_info.items():
        print(f"Loading dataset: {scenario_name}")
        print(f"  - Disease genes: {info['n_disease_genes']}")
        print(f"  - Background genes: {info['n_background_genes']}")
        print(f"  - Signal ratio: {info['disease_relevant_ratio']:.1%}")
        
        scenario_path = info['path']
        
        # Load gene expression data
        gene_expr_path = os.path.join(scenario_path, 'gene_expression.csv.gz')
        gene_expr_df = pd.read_csv(gene_expr_path, index_col='Sample_ID')
        
        # Load auxiliary data
        aux_data_path = os.path.join(scenario_path, 'aux_data.csv.gz')
        aux_data_df = pd.read_csv(aux_data_path, index_col='Sample_ID')
        
        # Load response data (use sampled responses for evaluation)
        response_path = os.path.join(scenario_path, 'responses_sampled.csv.gz')
        response_df = pd.read_csv(response_path, index_col='Sample_ID')
        
        # Create AnnData object
        X = gene_expr_df.values.astype(np.float64)
        adata = ad.AnnData(X=X)
        
        # Set observation and variable names
        adata.obs_names = gene_expr_df.index.astype(str)
        adata.var_names = gene_expr_df.columns.tolist()
        
        # Combine auxiliary data and responses as observations
        obs_data = pd.concat([aux_data_df, response_df], axis=1)
        # Ensure obs_data index is also string type to avoid AnnData warnings
        obs_data.index = obs_data.index.astype(str)
        adata.obs = obs_data
        
        # Ensure all numeric columns are proper numeric types
        for col in adata.obs.columns:
            if adata.obs[col].dtype == 'object':
                adata.obs[col] = pd.to_numeric(adata.obs[col], errors='coerce')
        
        # Fill any NaN values
        for col in adata.obs.columns:
            if adata.obs[col].isna().sum() > 0:
                if col == 'disease':
                    adata.obs[col] = adata.obs[col].fillna(0)
                else:
                    adata.obs[col] = adata.obs[col].fillna(adata.obs[col].median())
        
        print(f"  - AnnData shape: {adata.shape}")
        print(f"  - Disease label distribution:")
        print(adata.obs['disease'].value_counts())
        
        # Store dataset with label column
        datasets[scenario_name] = (adata, 'disease')
    
    return datasets


def load_signal_noise_dataset_single(scenario_name, base_path="dataset/signal_noise_ratios"):
    """
    Load a single signal-to-noise ratio dataset.
    
    Args:
        scenario_name: Name of the scenario (e.g., 'high_signal', 'balanced', etc.)
        base_path: Path to the signal_noise_ratios directory
        
    Returns:
        Tuple of (adata, label_col)
    """
    import json
    
    # Load dataset information
    info_path = os.path.join(base_path, 'dataset_info.json')
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    if scenario_name not in dataset_info:
        raise ValueError(f"Scenario '{scenario_name}' not found. Available scenarios: {list(dataset_info.keys())}")
    
    info = dataset_info[scenario_name]
    scenario_path = info['path']
    
    print(f"Loading dataset: {scenario_name}")
    print(f"  - Disease genes: {info['n_disease_genes']}")
    print(f"  - Background genes: {info['n_background_genes']}")
    print(f"  - Signal ratio: {info['disease_relevant_ratio']:.1%}")
    
    # Load gene expression data
    gene_expr_path = os.path.join(scenario_path, 'gene_expression.csv.gz')
    gene_expr_df = pd.read_csv(gene_expr_path, index_col='Sample_ID')
    
    # Load auxiliary data
    aux_data_path = os.path.join(scenario_path, 'aux_data.csv.gz')
    aux_data_df = pd.read_csv(aux_data_path, index_col='Sample_ID')
    
    # Load response data (use sampled responses for evaluation)
    response_path = os.path.join(scenario_path, 'responses_sampled.csv.gz')
    response_df = pd.read_csv(response_path, index_col='Sample_ID')
    
    # Create AnnData object
    X = gene_expr_df.values.astype(np.float64)
    adata = ad.AnnData(X=X)
    
    # Set observation and variable names
    adata.obs_names = gene_expr_df.index.astype(str)
    adata.var_names = gene_expr_df.columns.tolist()
    
    # Combine auxiliary data and responses as observations
    obs_data = pd.concat([aux_data_df, response_df], axis=1)
    # Ensure obs_data index is also string type to avoid AnnData warnings
    obs_data.index = obs_data.index.astype(str)
    adata.obs = obs_data
    
    # Ensure all numeric columns are proper numeric types
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object':
            adata.obs[col] = pd.to_numeric(adata.obs[col], errors='coerce')
    
    # Fill any NaN values
    for col in adata.obs.columns:
        if adata.obs[col].isna().sum() > 0:
            if col == 'disease':
                adata.obs[col] = adata.obs[col].fillna(0)
            else:
                adata.obs[col] = adata.obs[col].fillna(adata.obs[col].median())
    
    print(f"  - AnnData shape: {adata.shape}")
    print(f"  - Disease label distribution:")
    print(adata.obs['disease'].value_counts())
    
    return adata, 'disease'


def create_synthetic_pathways(n_gs1, n_gs2, pathway_accuracy=1.0):
    """
    Create synthetic pathway masks for testing pathway-guided configs
    
    Args:
        n_gs1: number of background genes (gs1_*)
        n_gs2: number of disease genes (gs2_*)  
        pathway_accuracy: fraction of correct pathway assignments (0.5-1.0)
        
    Returns:
        numpy array: pathway mask matrix with shape (n_genes, n_pathways)
    """
    total_genes = n_gs1 + n_gs2
    
    # Create synthetic pathways
    pathways = {
        'Inflammatory_Response': np.zeros(total_genes),
        'Immune_System': np.zeros(total_genes), 
        'Cell_Cycle': np.zeros(total_genes),
        'Metabolic_Process': np.zeros(total_genes),
        'Background_Process': np.zeros(total_genes)
    }
    
    # Disease-relevant pathways should contain mostly gs2 genes
    n_correct_disease = int(n_gs2 * pathway_accuracy)
    n_incorrect_disease = n_gs2 - n_correct_disease
    
    # Inflammatory Response pathway (should contain gs2 genes)
    disease_genes_idx = np.arange(n_gs1, n_gs1 + n_gs2)  # gs2 gene indices
    background_genes_idx = np.arange(0, n_gs1)  # gs1 gene indices
    
    # Correctly assign most gs2 genes to disease pathways
    correct_disease_genes = np.random.choice(disease_genes_idx, n_correct_disease, replace=False)
    pathways['Inflammatory_Response'][correct_disease_genes] = 1
    
    # Add some gs2 genes to Immune_System pathway too
    immune_genes = np.random.choice(disease_genes_idx, n_gs2//2, replace=False)
    pathways['Immune_System'][immune_genes] = 1
    
    # Add noise: some gs1 genes in disease pathways (incorrect assignments)
    if pathway_accuracy < 1.0:
        noise_genes = np.random.choice(background_genes_idx, n_incorrect_disease, replace=False)
        pathways['Inflammatory_Response'][noise_genes] = 1
    
    # Background pathways should contain mostly gs1 genes
    background_in_background = np.random.choice(background_genes_idx, n_gs1//2, replace=False)
    pathways['Background_Process'][background_in_background] = 1
    
    # Cell cycle and metabolic - mix of both (neutral pathways)
    mixed_genes = np.random.choice(np.arange(total_genes), total_genes//4, replace=False)
    pathways['Cell_Cycle'][mixed_genes] = 1
    
    return np.column_stack([pathways[p] for p in pathways.keys()])


def create_signal_noise_pathways(adata, scenario_name, pathway_accuracy=1.0):
    """
    Create synthetic pathway masks for signal-to-noise datasets based on ground truth
    
    Args:
        adata: AnnData object with signal-to-noise data
        scenario_name: Name of the scenario (e.g., 'high_signal', 'balanced')
        pathway_accuracy: fraction of correct pathway assignments (0.5-1.0)
        
    Returns:
        numpy array: pathway mask matrix with shape (n_genes, n_pathways)
    """
    import json
    import os
    
    # Load ground truth information
    base_path = "dataset/signal_noise_ratios"
    scenario_path = os.path.join(base_path, f"scenario_{scenario_name}")
    ground_truth_path = os.path.join(scenario_path, 'ground_truth.json')
    
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Get gene information
    disease_genes = ground_truth['disease_genes']
    background_genes = ground_truth['background_genes']
    all_genes = ground_truth['all_genes']
    
    # Create gene name to index mapping
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    
    # Get counts
    n_disease_genes = len(disease_genes)
    n_background_genes = len(background_genes)
    total_genes = len(all_genes)
    
    print(f"Creating pathways for {scenario_name}:")
    print(f"  - Disease genes: {n_disease_genes}")
    print(f"  - Background genes: {n_background_genes}")
    print(f"  - Pathway accuracy: {pathway_accuracy:.1%}")
    
    # Create synthetic pathways
    pathways = {
        'Disease_Relevant_1': np.zeros(total_genes),
        'Disease_Relevant_2': np.zeros(total_genes),
        'Background_Process_1': np.zeros(total_genes),
        'Background_Process_2': np.zeros(total_genes),
        'Mixed_Process': np.zeros(total_genes)
    }
    
    # Get disease and background gene indices
    disease_indices = [gene_to_idx[gene] for gene in disease_genes]
    background_indices = [gene_to_idx[gene] for gene in background_genes]
    
    # Disease-relevant pathways should contain mostly disease genes
    n_correct_disease = int(n_disease_genes * pathway_accuracy)
    n_incorrect_disease = n_disease_genes - n_correct_disease
    
    # Disease_Relevant_1 pathway
    correct_disease_genes = np.random.choice(disease_indices, n_correct_disease, replace=False)
    pathways['Disease_Relevant_1'][correct_disease_genes] = 1
    
    # Disease_Relevant_2 pathway (different subset)
    remaining_disease = list(set(disease_indices) - set(correct_disease_genes))
    if len(remaining_disease) > 0:
        subset_size = min(len(remaining_disease), n_disease_genes // 2)
        disease_subset = np.random.choice(remaining_disease, subset_size, replace=False)
        pathways['Disease_Relevant_2'][disease_subset] = 1
    
    # Add noise: some background genes in disease pathways (incorrect assignments)
    if pathway_accuracy < 1.0 and n_incorrect_disease > 0:
        noise_genes = np.random.choice(background_indices, n_incorrect_disease, replace=False)
        pathways['Disease_Relevant_1'][noise_genes] = 1
    
    # Background pathways should contain mostly background genes
    background_in_background = np.random.choice(background_indices, n_background_genes // 2, replace=False)
    pathways['Background_Process_1'][background_in_background] = 1
    
    # Mixed pathway - contains both types
    mixed_genes = np.random.choice(np.arange(total_genes), total_genes // 4, replace=False)
    pathways['Mixed_Process'][mixed_genes] = 1
    
    pathway_matrix = np.column_stack([pathways[p] for p in pathways.keys()])
    
    print(f"  - Created pathway matrix: {pathway_matrix.shape}")
    print(f"  - Pathway sparsity: {100 * (1 - np.count_nonzero(pathway_matrix) / pathway_matrix.size):.1f}%")
    
    return pathway_matrix


def test_pathway_accuracy():
    """
    Test different pathway qualities for signal-to-noise datasets
    
    Returns:
        Dictionary with results for different pathway accuracies
    """
    accuracies = [1.0, 0.8, 0.6, 0.4]  # Perfect to poor pathway knowledge
    results = {}
    
    # Load a sample dataset to test with
    try:
        adata, label_col = load_signal_noise_dataset_single('balanced')
        
        for accuracy in accuracies:
            print(f"\nTesting pathway accuracy: {accuracy:.1%}")
            
            # Create pathway mask with this accuracy
            pathway_mask = create_signal_noise_pathways(adata, 'balanced', accuracy)
            
            # Test pathway-guided config (placeholder for now)
            results[accuracy] = {
                'pathway_mask_shape': pathway_mask.shape,
                'pathway_sparsity': 100 * (1 - np.count_nonzero(pathway_mask) / pathway_mask.size),
                'accuracy_level': accuracy
            }
            
        return results
        
    except Exception as e:
        print(f"Error in test_pathway_accuracy: {e}")
        return {}


def get_signal_noise_pathway_mask(scenario_name, pathway_accuracy=1.0):
    """
    Get pathway mask for a specific signal-to-noise scenario
    
    Args:
        scenario_name: Name of the scenario (e.g., 'high_signal', 'balanced')
        pathway_accuracy: fraction of correct pathway assignments (0.5-1.0)
        
    Returns:
        numpy array: pathway mask matrix
    """
    # Load the dataset to get gene information
    adata, label_col = load_signal_noise_dataset_single(scenario_name)
    
    # Create pathway mask
    pathway_mask = create_signal_noise_pathways(adata, scenario_name, pathway_accuracy)
    
    return pathway_mask


def load_signal_noise_datasets_simple(base_path="dataset/signal_noise_ratios"):
    """
    Load all signal-to-noise ratio datasets efficiently without loading pathway data.
    
    Args:
        base_path: Path to the signal_noise_ratios directory
        
    Returns:
        Dictionary with dataset names as keys and (adata, label_col) tuples as values
    """
    import json
    
    # Load dataset information
    info_path = os.path.join(base_path, 'dataset_info.json')
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    datasets = {}
    
    for scenario_name, info in dataset_info.items():
        print(f"Loading dataset: {scenario_name}")
        print(f"  - Disease genes: {info['n_disease_genes']}")
        print(f"  - Background genes: {info['n_background_genes']}")
        print(f"  - Signal ratio: {info['disease_relevant_ratio']:.1%}")
        
        scenario_path = info['path']
        
        # Load gene expression data
        gene_expr_path = os.path.join(scenario_path, 'gene_expression.csv.gz')
        gene_expr_df = pd.read_csv(gene_expr_path, index_col='Sample_ID')
        
        # Load auxiliary data
        aux_data_path = os.path.join(scenario_path, 'aux_data.csv.gz')
        aux_data_df = pd.read_csv(aux_data_path, index_col='Sample_ID')
        
        # Load response data (use sampled responses for evaluation)
        response_path = os.path.join(scenario_path, 'responses_sampled.csv.gz')
        response_df = pd.read_csv(response_path, index_col='Sample_ID')
        
        # Create AnnData object
        X = gene_expr_df.values.astype(np.float64)
        adata = ad.AnnData(X=X)
        
        # Set observation and variable names
        adata.obs_names = gene_expr_df.index.astype(str)
        adata.var_names = gene_expr_df.columns.tolist()
        
        # Combine auxiliary data and responses as observations
        obs_data = pd.concat([aux_data_df, response_df], axis=1)
        # Ensure obs_data index is also string type to avoid AnnData warnings
        obs_data.index = obs_data.index.astype(str)
        adata.obs = obs_data
        
        # Ensure all numeric columns are proper numeric types
        for col in adata.obs.columns:
            if adata.obs[col].dtype == 'object':
                adata.obs[col] = pd.to_numeric(adata.obs[col], errors='coerce')
        
        # Fill any NaN values
        for col in adata.obs.columns:
            if adata.obs[col].isna().sum() > 0:
                if col == 'disease':
                    adata.obs[col] = adata.obs[col].fillna(0)
                else:
                    adata.obs[col] = adata.obs[col].fillna(adata.obs[col].median())
        
        print(f"  - AnnData shape: {adata.shape}")
        print(f"  - Disease label distribution:")
        print(adata.obs['disease'].value_counts())
        
        # Store dataset with label column
        datasets[scenario_name] = (adata, 'disease')
    
    return datasets

