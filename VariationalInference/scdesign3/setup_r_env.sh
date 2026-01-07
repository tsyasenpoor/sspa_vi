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
R_VERSION=$(R --version | head -n 1 | grep -oP '\d+\.\d+\.\d+')
echo "Found R version: $R_VERSION"

# Install required R packages
echo ""
echo "Installing R dependencies..."
echo ""

R --vanilla << 'EOF'
# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Function to install if not present
install_if_missing <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("Installing %s...\n", pkg))
        install.packages(pkg)
    } else {
        cat(sprintf("%s already installed\n", pkg))
    }
}

# Install BiocManager for Bioconductor packages
install_if_missing("BiocManager")

# Install devtools for GitHub packages (required for install_github)
# Force install to ensure it's available
if (!requireNamespace("devtools", quietly = TRUE)) {
    cat("Installing devtools (this may take a few minutes)...\n")
    install.packages("devtools", dependencies = TRUE)
}
# Explicitly load devtools
library(devtools)
cat("devtools loaded successfully\n")

# Core dependencies from CRAN
cran_packages <- c(
    "dplyr",
    "tibble",
    "Matrix",
    "mgcv",
    "gamlss",
    "mclust",
    "mvtnorm",
    "ggplot2",
    "umap",
    "viridis",
    "matrixStats",
    "coop",
    "pbmcapply",
    "jsonlite"
)

for (pkg in cran_packages) {
    install_if_missing(pkg)
}

# Bioconductor packages
bioc_packages <- c(
    "SingleCellExperiment",
    "SummarizedExperiment",
    "BiocParallel",
    "rhdf5"  # For reading h5ad files
)

for (pkg in bioc_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("Installing %s from Bioconductor...\n", pkg))
        BiocManager::install(pkg, ask = FALSE, update = FALSE)
    } else {
        cat(sprintf("%s already installed\n", pkg))
    }
}

# Install scDesign3 from GitHub
cat("\n============================================\n")
cat("Installing scDesign3 from GitHub...\n")
cat("============================================\n")

if (!requireNamespace("scDesign3", quietly = TRUE)) {
    devtools::install_github("SONGDONGYUAN1994/scDesign3", upgrade = "never")
} else {
    cat("scDesign3 already installed\n")
}

# Verify installation
cat("\n============================================\n")
cat("Verifying installation...\n")
cat("============================================\n")

library(scDesign3)
cat(sprintf("scDesign3 version: %s\n", packageVersion("scDesign3")))

cat("\nInstallation complete!\n")
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
echo "  result = simulator.simulate(sce_path='reference.h5ad', n_cells=1000)"
