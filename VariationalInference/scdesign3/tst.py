import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from VariationalInference.scdesign3 import ScDesign3Simulator
simulator = ScDesign3Simulator()
result = simulator.simulate(input_file='/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19/gm_ctrl_covid.h5ad', output_dir="./scdesign3_10000cells", n_cells=10000,  n_genes=10000, gene_selection="variable", family="poisson")