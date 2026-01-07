#!/bin/bash
# Setup script for scDesign3 R environment
# This script installs R dependencies and the scDesign3 package

set -e

echo "=============================================="
echo "Setting up R environment for scDesign3"
echo "=============================================="

# Check if R is installed
if ! command -v R &> /dev/null; then
    echo "ERROR: R is not installed. Please install R >= 4.3.0"
    echo "  Ubuntu/Debian: sudo apt-get install r-base r-base-dev"
    echo "  macOS: brew install r"
    exit 1
fi

# Check R version
R_VERSION=$(R --version | head -n 1 | grep -oP '\d+\.\d+\.\d+' || R --version | head -n 1)
echo "Found R version: $R_VERSION"

echo ""
echo "NOTE: If you see errors about missing system libraries (like libpng),"
echo "you may need to install them first:"
echo "  Ubuntu/Debian: sudo apt-get install libpng-dev libcurl4-openssl-dev libssl-dev libxml2-dev"
echo "  CentOS/RHEL: sudo yum install libpng-devel libcurl-devel openssl-devel libxml2-devel"
echo "  Conda: conda install -c conda-forge libpng r-png"
echo ""

# Install required R packages
echo "Installing R dependencies..."
echo ""

R --vanilla << 'EOF'
# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Function to install if not present
install_if_missing <- function(pkg, source = "cran") {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("Installing %s from %s...\n", pkg, source))
        if (source == "bioc") {
            BiocManager::install(pkg, ask = FALSE, update = FALSE)
        } else {
            install.packages(pkg)
        }
        # Verify it installed
        if (!requireNamespace(pkg, quietly = TRUE)) {
            cat(sprintf("WARNING: %s may not have installed correctly\n", pkg))
        }
    } else {
        cat(sprintf("%s already installed\n", pkg))
    }
}

# Install BiocManager first
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    cat("Installing BiocManager...\n")
    install.packages("BiocManager")
}

# Install remotes for GitHub packages
if (!requireNamespace("remotes", quietly = TRUE)) {
    cat("Installing remotes...\n")
    install.packages("remotes")
}
library(remotes)
cat("remotes loaded successfully\n")

cat("\n=== Installing Bioconductor core dependencies (in order) ===\n")
# Install Bioconductor dependencies in dependency order
bioc_core <- c(
    "BiocGenerics",
    "S4Vectors",
    "IRanges",
    "XVector",
    "zlibbioc",
    "GenomeInfoDbData",
    "GenomeInfoDb",
    "GenomicRanges",
    "SparseArray",
    "MatrixGenerics",
    "DelayedArray",
    "Biobase",
    "SummarizedExperiment",
    "SingleCellExperiment",
    "BiocParallel",
    "rhdf5filters",
    "Rhdf5lib",
    "rhdf5"
)

for (pkg in bioc_core) {
    install_if_missing(pkg, "bioc")
}

cat("\n=== Installing CRAN dependencies ===\n")
# Core CRAN packages
cran_packages <- c(
    "dplyr",
    "tibble",
    "Matrix",
    "mgcv",
    "gamlss",
    "mclust",
    "mvtnorm",
    "ggplot2",
    "viridis",
    "matrixStats",
    "coop",
    "pbmcapply",
    "jsonlite"
)

for (pkg in cran_packages) {
    install_if_missing(pkg, "cran")
}

# Try to install umap (optional - needs reticulate which needs png)
cat("\nTrying to install umap (optional)...\n")
tryCatch({
    install_if_missing("png", "cran")
    install_if_missing("reticulate", "cran")
    install_if_missing("umap", "cran")
    cat("umap installed successfully\n")
}, error = function(e) {
    cat("Note: Could not install umap. This is optional and scDesign3 will still work.\n")
})

# Install scDesign3 from GitHub
cat("\n============================================\n")
cat("Installing scDesign3 from GitHub...\n")
cat("============================================\n")

if (!requireNamespace("scDesign3", quietly = TRUE)) {
    tryCatch({
        remotes::install_github("SONGDONGYUAN1994/scDesign3", upgrade = "never")
    }, error = function(e) {
        cat("Error installing scDesign3:", conditionMessage(e), "\n")
        cat("\nTrying alternative installation...\n")
        # Try installing without vignettes
        remotes::install_github("SONGDONGYUAN1994/scDesign3",
                                upgrade = "never",
                                build_vignettes = FALSE)
    })
} else {
    cat("scDesign3 already installed\n")
}

# Verify installation
cat("\n============================================\n")
cat("Verifying installation...\n")
cat("============================================\n")

if (requireNamespace("scDesign3", quietly = TRUE)) {
    library(scDesign3)
    cat(sprintf("SUCCESS: scDesign3 version %s installed!\n", packageVersion("scDesign3")))
    cat("\nInstallation complete!\n")
} else {
    cat("\n")
    cat("============================================\n")
    cat("ERROR: scDesign3 installation failed.\n")
    cat("============================================\n")
    cat("\nThe most common causes are missing system libraries.\n")
    cat("\nPlease try:\n")
    cat("  1. Install system dependencies:\n")
    cat("     conda install -c conda-forge libpng r-png\n")
    cat("     # OR for Ubuntu:\n")
    cat("     sudo apt-get install libpng-dev\n")
    cat("\n  2. Then run this script again.\n")
    cat("\n  3. Or try manual installation in R:\n")
    cat("     BiocManager::install('SingleCellExperiment')\n")
    cat("     remotes::install_github('SONGDONGYUAN1994/scDesign3')\n")
    quit(status = 1)
}
EOF

echo ""
echo "=============================================="
echo "R environment setup complete!"
echo "=============================================="
echo ""
echo "You can now use scDesign3 from Python via the wrapper module."
echo "Example:"
echo "  from VariationalInference.scdesign3 import ScDesign3Simulator"
echo "  simulator = ScDesign3Simulator()"
echo "  result = simulator.simulate(input_file='reference.h5ad', n_cells=1000)"
