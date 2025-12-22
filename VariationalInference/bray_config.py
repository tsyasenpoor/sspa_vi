"""
BRAY LAB SPECIFIC CONFIGURATION
================================

This module contains paths and constants specific to the Bray Lab datasets
including AJM, NAP, and SSPA B-cell analysis pipelines.

For generic VI usage, see: config.py, data_loader.py, cli.py

These paths are for internal Bray Lab use on the Aguiar cluster.
External users should use the generic data_loader.py instead.
"""

from pathlib import Path

# Bray Lab base directories
BRAY_BASE_DIR = Path("/labs/Aguiar/SSPA_BRAY/BRay")
BRAY_DATA_DIR = BRAY_BASE_DIR / "BRAY_FileTransfer"
BRAY_RESULTS_DIR = BRAY_BASE_DIR / "results"
BRAY_CACHE_DIR = BRAY_BASE_DIR / "cache"
BRAY_SEED_DIR = BRAY_DATA_DIR / "SeedGenes"

# Legacy aliases (for backward compatibility with existing scripts)
BASE_DIR = BRAY_BASE_DIR
DATA_DIR = BRAY_DATA_DIR
RESULTS_DIR = BRAY_RESULTS_DIR
CACHE_DIR = BRAY_CACHE_DIR
SEED_DIR = BRAY_SEED_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)


ajm_file_path = DATA_DIR / "BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
ajm_metadata_path = DATA_DIR / "BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"

nap_file_path = DATA_DIR / "BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/GEX_NAP_filt_raw_modelingonly_2024-02-05.csv"
nap_metadata_path = DATA_DIR / "BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/meta_NAP_unfilt_fullData_2024-02-05.csv"

gene_annotation_path = DATA_DIR / "ENS_mouse_geneannotation.csv"

cytoseeds_csv_path = SEED_DIR / "CYTOBEAM_Cytokines_KEGGPATHWAY_addedMif.csv"
apseeds_csv_path = SEED_DIR / "APBEAM_KEGGPATHWAY_mouse.csv"
igseeds_csv_path = SEED_DIR / "IgBEAM_Igs_GO_term_summary_20220907_163014.csv"

