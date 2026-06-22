#!/usr/bin/env Rscript
# Fit SPsimSeq on the GTEx Whole Blood reference counts and generate a realistic NULL background
# X0 (n_sim samples, no differential expression) preserving WB marginals + gene-gene copula.
# The disease signal / programs are injected LATER (seqgendiff thin_diff), not here.
#
# Usage:
#   /home/FCAM/tyasenpoor/miniconda3/envs/bulksim_r/bin/Rscript fit_spsimseq.R \
#       --ref data/Simulations/bulk_gtex_v1/wb_reference_counts.tsv.gz \
#       --n-sim 2000 --seed 0 --out data/Simulations/bulk_gtex_v1/X0_null.tsv.gz
#   [--n-genes N]  subset to N most-variable genes (for a fast smoke test)

suppressMessages({library(SPsimSeq); library(data.table)})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- match(flag, args); if (is.na(i)) return(default); args[[i + 1]]
}
ref_path <- getarg("--ref")
n_sim    <- as.integer(getarg("--n-sim", "2000"))
seed     <- as.integer(getarg("--seed", "0"))
out_path <- getarg("--out")
n_genes  <- getarg("--n-genes", NA)

cat(sprintf("SPsimSeq null background: ref=%s n.sim=%d seed=%d\n", ref_path, n_sim, seed))
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

# genes x samples counts (row names = versioned ENSG, col names = SAMPID)
dt <- fread(ref_path)
gene_ids <- dt[[1]]
mat <- as.matrix(dt[, -1]); rownames(mat) <- gene_ids
storage.mode(mat) <- "integer"
cat(sprintf("  reference: %d genes x %d samples\n", nrow(mat), ncol(mat)))

if (!is.na(n_genes)) {
  ng <- as.integer(n_genes)
  v <- apply(mat, 1, var)
  keep <- order(v, decreasing = TRUE)[seq_len(min(ng, nrow(mat)))]
  mat <- mat[sort(keep), , drop = FALSE]
  cat(sprintf("  subset to %d most-variable genes (smoke test)\n", nrow(mat)))
}

# NULL background: single group, no DE (pDE = 0). Gaussian copula keeps gene-gene correlation.
sim <- SPsimSeq(
  n.sim       = 1,
  s.data      = mat,
  group       = rep(1, ncol(mat)),
  n.genes     = nrow(mat),
  group.config = 1,
  pDE         = 0,
  tot.samples = n_sim,
  genewiseCor = TRUE,                 # Gaussian copula keeps gene-gene correlation (required)
  log.CPM.transform = TRUE,
  variable.lib.size = FALSE,          # TRUE triggers a SPsimSeq LL bug; depths re-set at injection
  result.format = "list",
  return.details = FALSE
)

X0 <- sim[[1]]$counts                      # genes x n_sim
rownames(X0) <- rownames(mat)
colnames(X0) <- paste0("sim_", seq_len(ncol(X0)))
cat(sprintf("  generated X0: %d genes x %d samples\n", nrow(X0), ncol(X0)))

out_dt <- data.table(gene = rownames(X0), X0)
fwrite(out_dt, out_path, sep = "\t", compress = "gzip")
cat(sprintf("DONE -> %s\n", out_path))
