"""
Utility functions for scDesign3 integration.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional


def check_r_installation(r_executable: str = "Rscript") -> dict:
    """
    Check R installation and available packages.

    Parameters
    ----------
    r_executable : str
        Path to Rscript executable

    Returns
    -------
    dict
        Installation status information
    """
    status = {
        "r_installed": False,
        "r_version": None,
        "scdesign3_installed": False,
        "scdesign3_version": None,
        "missing_packages": [],
    }

    # Check if R is available
    if not shutil.which(r_executable):
        print(f"ERROR: R not found. Please install R >= 4.3.0")
        print("  Ubuntu/Debian: sudo apt-get install r-base r-base-dev")
        print("  macOS: brew install r")
        return status

    status["r_installed"] = True

    # Get R version
    try:
        result = subprocess.run(
            [r_executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_line = result.stdout.split("\n")[0]
        status["r_version"] = version_line
    except Exception as e:
        print(f"Warning: Could not get R version: {e}")

    # Check for scDesign3
    try:
        result = subprocess.run(
            [
                r_executable,
                "-e",
                "cat(as.character(packageVersion('scDesign3')))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and "Error" not in result.stderr:
            status["scdesign3_installed"] = True
            status["scdesign3_version"] = result.stdout.strip()
    except Exception:
        pass

    # Check required packages
    required_packages = [
        "SingleCellExperiment",
        "mgcv",
        "gamlss",
        "Matrix",
        "jsonlite",
    ]

    for pkg in required_packages:
        try:
            result = subprocess.run(
                [r_executable, "-e", f"library({pkg})"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                status["missing_packages"].append(pkg)
        except Exception:
            status["missing_packages"].append(pkg)

    return status


def install_scdesign3(r_executable: str = "Rscript") -> bool:
    """
    Install scDesign3 and dependencies.

    Parameters
    ----------
    r_executable : str
        Path to Rscript executable

    Returns
    -------
    bool
        True if installation successful
    """
    # Get path to setup script
    setup_script = Path(__file__).parent / "setup_r_env.sh"

    if not setup_script.exists():
        print(f"Setup script not found: {setup_script}")
        return False

    print("Installing scDesign3 and dependencies...")
    print("This may take several minutes...")

    try:
        result = subprocess.run(
            ["bash", str(setup_script)],
            timeout=1800,  # 30 minute timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Installation timed out")
        return False
    except Exception as e:
        print(f"Installation failed: {e}")
        return False


def convert_h5ad_to_sce(
    h5ad_path: str,
    output_path: str,
    r_executable: str = "Rscript",
) -> bool:
    """
    Convert an h5ad file to SingleCellExperiment RDS format.

    This can be useful for pre-converting data or for debugging.

    Parameters
    ----------
    h5ad_path : str
        Input h5ad file path
    output_path : str
        Output RDS file path
    r_executable : str
        Path to Rscript

    Returns
    -------
    bool
        True if conversion successful
    """
    r_code = f'''
    suppressPackageStartupMessages({{
        library(rhdf5)
        library(SingleCellExperiment)
        library(Matrix)
    }})

    h5_data <- h5read("{h5ad_path}", "/")

    # Extract counts
    if ("X" %in% names(h5_data)) {{
        if (is.list(h5_data$X)) {{
            counts <- sparseMatrix(
                i = h5_data$X$indices + 1,
                p = h5_data$X$indptr,
                x = h5_data$X$data,
                dims = rev(h5_data$X$shape)
            )
            counts <- t(counts)
        }} else {{
            counts <- t(h5_data$X)
        }}
    }}

    # Get names
    gene_names <- h5_data$var$`_index`
    cell_names <- h5_data$obs$`_index`
    rownames(counts) <- gene_names
    colnames(counts) <- cell_names

    # Get metadata
    col_data <- data.frame(row.names = cell_names)
    if ("obs" %in% names(h5_data)) {{
        for (col_name in names(h5_data$obs)) {{
            if (col_name != "_index" && !is.list(h5_data$obs[[col_name]])) {{
                col_data[[col_name]] <- h5_data$obs[[col_name]]
            }}
        }}
    }}

    # Create SCE
    sce <- SingleCellExperiment(
        assays = list(counts = counts),
        colData = col_data
    )

    saveRDS(sce, "{output_path}")
    H5close()
    cat("Conversion complete\\n")
    '''

    try:
        result = subprocess.run(
            [r_executable, "-e", r_code],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"Converted {h5ad_path} to {output_path}")
            return True
        else:
            print(f"Conversion failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Conversion error: {e}")
        return False


def get_sce_info(
    sce_path: str,
    r_executable: str = "Rscript",
) -> Optional[dict]:
    """
    Get information about a SingleCellExperiment RDS file.

    Parameters
    ----------
    sce_path : str
        Path to RDS file
    r_executable : str
        Path to Rscript

    Returns
    -------
    dict or None
        Information about the SCE object
    """
    r_code = f'''
    suppressPackageStartupMessages(library(SingleCellExperiment))
    library(jsonlite)

    sce <- readRDS("{sce_path}")

    info <- list(
        n_genes = nrow(sce),
        n_cells = ncol(sce),
        assays = assayNames(sce),
        coldata_columns = colnames(colData(sce)),
        rowdata_columns = colnames(rowData(sce))
    )

    cat(toJSON(info, auto_unbox = TRUE))
    '''

    try:
        result = subprocess.run(
            [r_executable, "-e", r_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
        else:
            return None
    except Exception:
        return None


def print_installation_instructions():
    """Print instructions for installing scDesign3."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                 scDesign3 Installation Guide                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  scDesign3 is an R package. To use it from Python, you need:     ║
║                                                                   ║
║  1. R >= 4.3.0 installed on your system                          ║
║  2. scDesign3 R package and dependencies                         ║
║                                                                   ║
║  Quick Setup:                                                     ║
║  ───────────                                                      ║
║  cd VariationalInference/scdesign3                               ║
║  bash setup_r_env.sh                                             ║
║                                                                   ║
║  Manual R Installation:                                          ║
║  ─────────────────────                                           ║
║  # In R console:                                                 ║
║  install.packages("BiocManager")                                 ║
║  BiocManager::install(c("SingleCellExperiment", "BiocParallel"))║
║  install.packages("devtools")                                    ║
║  devtools::install_github("SONGDONGYUAN1994/scDesign3")         ║
║                                                                   ║
║  System Requirements:                                             ║
║  ───────────────────                                             ║
║  - Linux or macOS (Windows has limited support)                  ║
║  - 1+ GB RAM per CPU core                                        ║
║  - For large datasets: reduce n_genes or use n_cores > 1         ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")
