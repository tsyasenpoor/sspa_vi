import sys
from pathlib import Path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from VariationalInference.scdesign3 import ScDesign3Simulator
simulator = ScDesign3Simulator(
    r_executable="/home/FCAM/tyasenpoor/miniconda3/envs/bray_cpu/bin/Rscript",
)
result = simulator.simulate(
    input_file='/labs/Aguiar/SSPA_BRAY/BRay/BRAY_FileTransfer/Covid19/control_adata.h5ad',
    output_dir="./scdesign3_covid19_1kcells_2kgenes",
    n_cells=1000, n_genes=2000,
    celltype_column="majorType",
    family="nb", copula="gaussian",
)