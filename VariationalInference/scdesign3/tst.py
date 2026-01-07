import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from VariationalInference.scdesign3 import ScDesign3Simulator
simulator = ScDesign3Simulator()
result = simulator.simulate(input_file='/labs/Aguiar/SSPA_BRAY/BRay/ctrl_sspa_test/Bcell_GEX_20251201_control.h5ad', output_dir="./scdesign3_quick_test", n_cells=100,  n_genes=500, gene_selection="variable")