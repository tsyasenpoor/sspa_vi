#!/usr/bin/env Rscript
# scDesign3 simulation runner
# This script is called from Python to run scDesign3 simulations

suppressPackageStartupMessages({
    library(scDesign3)
    library(SingleCellExperiment)
    library(Matrix)
    library(jsonlite)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
    cat("Usage: Rscript run_scdesign3.R <config_json> <output_dir>\n")
    quit(status = 1)
}

config_file <- args[1]
output_dir <- args[2]

# Read configuration
cat("Reading configuration from:", config_file, "\n")
config <- fromJSON(config_file)

# Print configuration
cat("\n========================================\n")
cat("scDesign3 Simulation Configuration\n")
cat("========================================\n")
cat("Input file:", config$input_file, "\n")
cat("Cell type column:", config$celltype_column, "\n")
cat("Number of cells to generate:", config$n_cells, "\n")
cat("Family:", config$family, "\n")
cat("Copula:", config$copula, "\n")
cat("Cores:", config$n_cores, "\n")
cat("Seed:", config$seed, "\n")
cat("Output directory:", output_dir, "\n")
cat("========================================\n\n")

# Set seed for reproducibility
set.seed(config$seed)

# Load input data
cat("Loading input data...\n")
input_file <- config$input_file

# Determine input format and load accordingly
if (grepl("\\.h5ad$", input_file)) {
    # Load from h5ad (AnnData) format
    # Need to convert from h5ad to SingleCellExperiment
    cat("Loading from h5ad format...\n")

    # Read the h5ad file using rhdf5
    if (!requireNamespace("rhdf5", quietly = TRUE)) {
        BiocManager::install("rhdf5", ask = FALSE, update = FALSE)
    }
    library(rhdf5)

    # Read the data
    h5_data <- h5read(input_file, "/")

    # First, determine dimensions and names from obs/var groups
    # This must be done BEFORE building the sparse matrix to ensure consistency

    # Get gene names (var index)
    if ("var" %in% names(h5_data) && "_index" %in% names(h5_data$var)) {
        gene_names <- h5_data$var$`_index`
    } else if ("var_names" %in% names(h5_data)) {
        gene_names <- h5_data$var_names
    } else {
        gene_names <- NULL  # Will generate later
    }

    # Get cell names (obs index) - handle both string and numeric indices
    if ("obs" %in% names(h5_data) && "_index" %in% names(h5_data$obs)) {
        cell_names <- h5_data$obs$`_index`
        # Convert numeric indices to strings if needed
        if (is.numeric(cell_names)) {
            cell_names <- paste0("Cell_", cell_names)
        }
    } else if ("obs_names" %in% names(h5_data)) {
        cell_names <- h5_data$obs_names
        if (is.numeric(cell_names)) {
            cell_names <- paste0("Cell_", cell_names)
        }
    } else {
        cell_names <- NULL  # Will generate later
    }

    # Determine n_obs and n_var from names if available
    n_var_from_names <- if (!is.null(gene_names)) length(gene_names) else NULL
    n_obs_from_names <- if (!is.null(cell_names)) length(cell_names) else NULL

    cat("  Gene names found:", !is.null(gene_names), ifelse(!is.null(n_var_from_names), paste0(" (n=", n_var_from_names, ")"), ""), "\n")
    cat("  Cell names found:", !is.null(cell_names), ifelse(!is.null(n_obs_from_names), paste0(" (n=", n_obs_from_names, ")"), ""), "\n")

    # Extract count matrix (X)
    if ("X" %in% names(h5_data)) {
        if (is.list(h5_data$X)) {
            # Sparse matrix format - could be CSR or CSC
            # CSR (row-compressed): indptr length = n_rows + 1, indices = column indices
            # CSC (col-compressed): indptr length = n_cols + 1, indices = row indices

            indptr_len <- length(h5_data$X$indptr) - 1
            max_index <- max(h5_data$X$indices) + 1  # +1 for 0-indexing

            # Determine format based on indptr length matching obs or var count
            # AnnData X is obs x var (cells x genes)
            is_csr <- FALSE
            is_csc <- FALSE

            if (!is.null(n_obs_from_names) && !is.null(n_var_from_names)) {
                if (indptr_len == n_obs_from_names) {
                    is_csr <- TRUE
                    cat("  Detected CSR format (indptr matches n_obs=", n_obs_from_names, ")\n")
                } else if (indptr_len == n_var_from_names) {
                    is_csc <- TRUE
                    cat("  Detected CSC format (indptr matches n_var=", n_var_from_names, ")\n")
                } else {
                    # Fallback: guess based on which dimension the indices fit
                    cat("  WARNING: indptr_len (", indptr_len, ") matches neither n_obs (", n_obs_from_names, ") nor n_var (", n_var_from_names, ")\n")
                    if (max_index <= n_var_from_names && indptr_len <= n_obs_from_names * 2) {
                        is_csr <- TRUE
                        cat("  Assuming CSR format\n")
                    } else {
                        is_csc <- TRUE
                        cat("  Assuming CSC format\n")
                    }
                }
            } else {
                # No metadata, assume CSR (AnnData default for some versions)
                is_csr <- TRUE
                cat("  No metadata dimensions, assuming CSR format\n")
            }

            n_obs <- if (!is.null(n_obs_from_names)) n_obs_from_names else (if (is_csr) indptr_len else max_index)
            n_var <- if (!is.null(n_var_from_names)) n_var_from_names else (if (is_csc) indptr_len else max_index)

            cat("  Building sparse matrix: n_obs=", n_obs, ", n_var=", n_var, "\n")

            if (is_csr) {
                # CSR: indptr indexes rows (cells), indices are column indices (genes)
                # Build obs x var matrix directly
                row_indices <- rep(seq_len(indptr_len), diff(h5_data$X$indptr))
                col_indices <- h5_data$X$indices + 1  # R is 1-indexed

                counts <- sparseMatrix(
                    i = row_indices,
                    j = col_indices,
                    x = as.numeric(h5_data$X$data),
                    dims = c(n_obs, n_var)
                )
            } else {
                # CSC: indptr indexes columns (genes), indices are row indices (cells)
                # Build obs x var matrix
                col_indices <- rep(seq_len(indptr_len), diff(h5_data$X$indptr))
                row_indices <- h5_data$X$indices + 1  # R is 1-indexed

                counts <- sparseMatrix(
                    i = row_indices,
                    j = col_indices,
                    x = as.numeric(h5_data$X$data),
                    dims = c(n_obs, n_var)
                )
            }

            # Transpose: AnnData is obs x var (cells x genes), SCE needs var x obs (genes x cells)
            counts <- t(counts)
            cat("  After transpose (genes x cells): ", nrow(counts), " x ", ncol(counts), "\n")
        } else {
            # Dense matrix
            counts <- t(h5_data$X)  # Transpose for SCE format (genes x cells)
        }
    } else if ("layers" %in% names(h5_data) && "counts" %in% names(h5_data$layers)) {
        counts <- t(h5_data$layers$counts)
    } else {
        stop("Could not find count matrix in h5ad file")
    }

    cat("  Count matrix dimensions: ", nrow(counts), " genes x ", ncol(counts), " cells\n")

    # Generate names if not available
    if (is.null(gene_names)) {
        gene_names <- paste0("Gene", 1:nrow(counts))
    }
    if (is.null(cell_names)) {
        cell_names <- paste0("Cell", 1:ncol(counts))
    }

    # Ensure names match matrix dimensions
    if (length(gene_names) != nrow(counts)) {
        cat("  WARNING: gene_names length (", length(gene_names), ") != nrow(counts) (", nrow(counts), ")\n")
        if (length(gene_names) > nrow(counts)) {
            cat("  Truncating gene_names to match matrix rows\n")
            gene_names <- gene_names[1:nrow(counts)]
        } else {
            cat("  Padding gene_names to match matrix rows\n")
            extra_names <- paste0("Gene", (length(gene_names)+1):nrow(counts))
            gene_names <- c(gene_names, extra_names)
        }
    }
    if (length(cell_names) != ncol(counts)) {
        cat("  WARNING: cell_names length (", length(cell_names), ") != ncol(counts) (", ncol(counts), ")\n")
        if (length(cell_names) > ncol(counts)) {
            cat("  Truncating cell_names to match matrix columns\n")
            cell_names <- cell_names[1:ncol(counts)]
        } else {
            cat("  Padding cell_names to match matrix columns\n")
            extra_names <- paste0("Cell", (length(cell_names)+1):ncol(counts))
            cell_names <- c(cell_names, extra_names)
        }
    }

    rownames(counts) <- gene_names
    colnames(counts) <- cell_names

    # Extract cell metadata
    col_data <- data.frame(row.names = cell_names)
    if ("obs" %in% names(h5_data)) {
        obs <- h5_data$obs
        for (col_name in names(obs)) {
            if (col_name == "_index") next

            col_value <- obs[[col_name]]

            # Handle categorical encoding (h5ad stores categoricals with codes + categories)
            if (is.list(col_value) && "codes" %in% names(col_value) && "categories" %in% names(col_value)) {
                # Decode categorical: codes are 0-indexed
                categories <- col_value$categories
                codes <- col_value$codes
                # Convert codes to 1-indexed for R and handle -1 (NA)
                decoded <- ifelse(codes < 0, NA, categories[codes + 1])
                col_data[[col_name]] <- decoded
            } else if (!is.list(col_value)) {
                # Simple non-list column
                col_data[[col_name]] <- col_value
            }
            # Skip other complex list structures
        }
    }

    cat("  Metadata columns:", paste(colnames(col_data), collapse = ", "), "\n")

    # Create SingleCellExperiment
    sce <- SingleCellExperiment(
        assays = list(counts = counts),
        colData = col_data
    )

    H5close()

} else if (grepl("\\.rds$", input_file, ignore.case = TRUE)) {
    # Load from RDS format (native R)
    cat("Loading from RDS format...\n")
    sce <- readRDS(input_file)

} else if (grepl("\\.csv$", input_file, ignore.case = TRUE)) {
    # Load from CSV (count matrix)
    cat("Loading from CSV format...\n")
    counts <- as.matrix(read.csv(input_file, row.names = 1))

    # Check for metadata file
    meta_file <- sub("\\.csv$", "_metadata.csv", input_file)
    if (file.exists(meta_file)) {
        col_data <- read.csv(meta_file, row.names = 1)
    } else {
        col_data <- data.frame(row.names = colnames(counts))
    }

    sce <- SingleCellExperiment(
        assays = list(counts = counts),
        colData = col_data
    )

} else {
    stop("Unsupported input format. Use .h5ad, .rds, or .csv")
}

cat("Loaded data with", nrow(sce), "genes and", ncol(sce), "cells\n")

# Validate cell type column exists
celltype_col <- config$celltype_column
if (!is.null(celltype_col) && celltype_col != "" && celltype_col != "NULL") {
    if (!(celltype_col %in% colnames(colData(sce)))) {
        available_cols <- colnames(colData(sce))
        cat("Available columns:", paste(available_cols, collapse = ", "), "\n")
        stop(paste("Cell type column '", celltype_col, "' not found in data"))
    }
    cat("Using cell type column:", celltype_col, "\n")
    cat("Cell types:", paste(unique(colData(sce)[[celltype_col]]), collapse = ", "), "\n")
} else {
    # Create a dummy cell type column
    cat("No cell type column specified, using single group\n")
    colData(sce)$cell_type <- "all"
    celltype_col <- "cell_type"
}

# Build mu_formula based on configuration
if (!is.null(config$pseudotime_column) && config$pseudotime_column != "" && config$pseudotime_column != "NULL") {
    pseudotime_col <- config$pseudotime_column
    if (!(pseudotime_col %in% colnames(colData(sce)))) {
        stop(paste("Pseudotime column '", pseudotime_col, "' not found in data"))
    }
    mu_formula <- config$mu_formula
    if (is.null(mu_formula) || mu_formula == "") {
        mu_formula <- paste0("s(", pseudotime_col, ", bs = 'cr', k = 10)")
    }
    cat("Using pseudotime column:", pseudotime_col, "\n")
    cat("mu_formula:", mu_formula, "\n")
} else {
    pseudotime_col <- NULL
    mu_formula <- "1"  # Intercept only
    cat("No pseudotime, using intercept-only model\n")
}

# Handle spatial coordinates if provided
if (!is.null(config$spatial_columns) && length(config$spatial_columns) > 0) {
    spatial_cols <- config$spatial_columns
    # Verify columns exist
    for (col in spatial_cols) {
        if (!(col %in% colnames(colData(sce)))) {
            stop(paste("Spatial column '", col, "' not found in data"))
        }
    }
    spatial_data <- as.matrix(colData(sce)[, spatial_cols, drop = FALSE])
    cat("Using spatial columns:", paste(spatial_cols, collapse = ", "), "\n")
} else {
    spatial_data <- NULL
}

# Optional: filter genes with low expression
if (!is.null(config$min_cells_expressing) && config$min_cells_expressing > 0) {
    min_cells <- config$min_cells_expressing
    if (min_cells < 1) {
        # Interpret as fraction
        min_cells <- ceiling(min_cells * ncol(sce))
    }
    gene_expressed <- rowSums(assay(sce, "counts") > 0)
    keep_genes <- gene_expressed >= min_cells
    sce <- sce[keep_genes, ]
    cat("Filtered to", nrow(sce), "genes expressed in >=", min_cells, "cells\n")
}

# Optional: subsample genes for faster testing
if (!is.null(config$n_genes) && config$n_genes > 0 && config$n_genes < nrow(sce)) {
    # Select top variable genes or random subset
    if (!is.null(config$gene_selection) && config$gene_selection == "variable") {
        cat("Selecting top", config$n_genes, "variable genes...\n")
        gene_vars <- rowVars(as.matrix(assay(sce, "counts")))
        top_genes <- order(gene_vars, decreasing = TRUE)[1:config$n_genes]
        sce <- sce[top_genes, ]
    } else {
        cat("Randomly selecting", config$n_genes, "genes...\n")
        set.seed(config$seed)
        random_genes <- sample(1:nrow(sce), config$n_genes)
        sce <- sce[random_genes, ]
    }
    cat("Using", nrow(sce), "genes for simulation\n")
}

# Run scDesign3 simulation
cat("\n========================================\n")
cat("Running scDesign3 simulation...\n")
cat("========================================\n")

# Determine number of cells to generate
n_cells <- config$n_cells
if (is.null(n_cells) || n_cells <= 0) {
    n_cells <- ncol(sce)  # Same as input
}

# Get the family parameter
family_use <- config$family
if (is.null(family_use) || family_use == "") {
    family_use <- "nb"  # Default to negative binomial
}

# Get copula type
copula_type <- config$copula
if (is.null(copula_type) || copula_type == "") {
    copula_type <- "gaussian"
}

# Get number of cores
n_cores <- config$n_cores
if (is.null(n_cores) || n_cores <= 0) {
    n_cores <- 1
}

# Run simulation
tryCatch({
    # Null-coalesce return_model parameter
    return_model_val <- if (is.null(config$return_model)) FALSE else config$return_model
    
    result <- scdesign3(
        sce = sce,
        assay_use = "counts",
        celltype = celltype_col,
        pseudotime = pseudotime_col,
        spatial = spatial_data,
        other_covariates = NULL,
        corr_formula = "1",
        mu_formula = mu_formula,
        sigma_formula = "1",
        family_use = family_use,
        n_cores = n_cores,
        copula = copula_type,
        ncell = n_cells,
        return_model = return_model_val
    )

    cat("\nSimulation completed successfully!\n")
    cat("Generated", ncol(result$new_count), "cells with", nrow(result$new_count), "genes\n")

}, error = function(e) {
    cat("ERROR in scdesign3:", conditionMessage(e), "\n")
    quit(status = 1)
})

# Create output directory
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Save simulated counts
cat("\nSaving outputs to:", output_dir, "\n")

# Save as CSV (for easy Python loading)
counts_file <- file.path(output_dir, "simulated_counts.csv")
write.csv(as.matrix(result$new_count), counts_file, row.names = TRUE)
cat("  - simulated_counts.csv\n")

# Save covariates/metadata
if (!is.null(result$new_covariate)) {
    meta_file <- file.path(output_dir, "simulated_metadata.csv")
    write.csv(result$new_covariate, meta_file, row.names = TRUE)
    cat("  - simulated_metadata.csv\n")
}

# Save as RDS (native R format, preserves structure)
rds_file <- file.path(output_dir, "simulated_sce.rds")
simulated_sce <- SingleCellExperiment(
    assays = list(counts = result$new_count),
    colData = result$new_covariate
)
saveRDS(simulated_sce, rds_file)
cat("  - simulated_sce.rds\n")

# Save gene names
gene_file <- file.path(output_dir, "gene_names.txt")
writeLines(rownames(result$new_count), gene_file)
cat("  - gene_names.txt\n")

# Save simulation info as JSON
info <- list(
    n_cells = ncol(result$new_count),
    n_genes = nrow(result$new_count),
    family = family_use,
    copula = copula_type,
    celltype_column = celltype_col,
    pseudotime_column = pseudotime_col,
    mu_formula = mu_formula,
    seed = config$seed,
    input_file = config$input_file,
    model_aic = if(!is.null(result$model_aic)) mean(result$model_aic, na.rm=TRUE) else NA,
    model_bic = if(!is.null(result$model_bic)) mean(result$model_bic, na.rm=TRUE) else NA
)
info_file <- file.path(output_dir, "simulation_info.json")
write(toJSON(info, auto_unbox = TRUE, pretty = TRUE), info_file)
cat("  - simulation_info.json\n")

# Optionally save fitted models
if (!is.null(config$return_model) && config$return_model) {
    if (!is.null(result$marginal_list)) {
        marginal_file <- file.path(output_dir, "marginal_models.rds")
        saveRDS(result$marginal_list, marginal_file)
        cat("  - marginal_models.rds\n")
    }
    if (!is.null(result$corr_list)) {
        corr_file <- file.path(output_dir, "correlation_models.rds")
        saveRDS(result$corr_list, corr_file)
        cat("  - correlation_models.rds\n")
    }
}

cat("\n========================================\n")
cat("scDesign3 simulation complete!\n")
cat("========================================\n")
