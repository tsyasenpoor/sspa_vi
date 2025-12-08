from pathlib import Path

BASE_DIR = Path("/labs/Aguiar/SSPA_BRAY/BRay")
DATA_DIR = BASE_DIR / "BRAY_FileTransfer"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"
SEED_DIR = DATA_DIR / "SeedGenes"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


ajm_file_path = DATA_DIR / "BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/GEX_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.rds"
ajm_metadata_path = DATA_DIR / "BRAY_AJM2/2_Data/2_SingleCellData/2_AJM_Parse_Timecourse/meta_TC_LPSonly_Bcellonly_filt_raw_2024-02-05.csv"

nap_file_path = DATA_DIR / "BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/GEX_NAP_filt_raw_modelingonly_2024-02-05.csv"
nap_metadata_path = DATA_DIR / "BRAY_AJM2/2_Data/2_SingleCellData/1_GSE139565_NaiveAndPlasma/meta_NAP_unfilt_fullData_2024-02-05.csv"

gene_annotation_path = DATA_DIR / "ENS_mouse_geneannotation.csv"

cytoseeds_csv_path = SEED_DIR / "CYTOBEAM_Cytokines_KEGGPATHWAY_addedMif.csv"
apseeds_csv_path = SEED_DIR / "APBEAM_KEGGPATHWAY_mouse.csv"
igseeds_csv_path = SEED_DIR / "IgBEAM_Igs_GO_term_summary_20220907_163014.csv"

