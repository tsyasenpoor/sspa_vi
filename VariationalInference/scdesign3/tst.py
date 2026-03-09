import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from VariationalInference.scdesign3 import ScDesign3Simulator
simulator = ScDesign3Simulator()
result = simulator.simulate(input_file='/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19/control_adata.h5ad', output_dir="./scdesign3_PBMC_10kcells_2kgenes", n_cells=10000,  n_genes=2000, family="nb", celltype_column="majorType", copula="gaussian")