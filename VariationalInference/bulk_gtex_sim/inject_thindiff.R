#!/usr/bin/env Rscript
# Inject ground-truth gene programs into the SPsimSeq null background via seqgendiff binomial
# thinning, then equalize library sizes to LABEL-INDEPENDENT target depths so total counts do
# NOT leak the label. Verified recipe (see README): thin_diff -> thin_lib(equalize) -> check.
#
# Usage:
#   Rscript inject_thindiff.R --x0 X0.tsv.gz --beta beta_coef.tsv.gz --theta theta_design.tsv.gz \
#       --out X_injected.tsv.gz --seed 0 [--lib-frac 0.90]
#
# beta  (gene x K)   = coef_fixed   (log2 fold-change loadings)
# theta (sample x K) = design_fixed (carrier activity)

suppressMessages({library(seqgendiff); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(f, d = NULL) { i <- match(f, args); if (is.na(i)) d else args[[i + 1]] }
x0_path  <- getarg("--x0"); beta_path <- getarg("--beta"); theta_path <- getarg("--theta")
out_path <- getarg("--out"); seed <- as.integer(getarg("--seed", "0"))
lib_frac <- as.numeric(getarg("--lib-frac", "0.90"))
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

read_mat <- function(p) { dt <- fread(p); m <- as.matrix(dt[, -1]); rownames(m) <- dt[[1]]; m }
X0 <- read_mat(x0_path); storage.mode(X0) <- "integer"
beta <- read_mat(beta_path)    # gene x K
theta <- read_mat(theta_path)  # sample x K
# align axes to X0
beta  <- beta[rownames(X0), , drop = FALSE]
theta <- theta[colnames(X0), , drop = FALSE]
cat(sprintf("inject: X0 %d x %d | beta %d x %d | theta %d x %d\n",
            nrow(X0), ncol(X0), nrow(beta), ncol(beta), nrow(theta), ncol(theta)))

# --- 1. thin_diff: plant low-rank program signal (log2 link) ---
res <- thin_diff(mat = X0, design_fixed = theta, coef_fixed = beta)
Xt <- res$mat

# --- 2. equalize library size to label-independent targets (<= global min, keeps mild spread) ---
lib <- colSums(Xt); Lmin <- min(lib)
target <- runif(ncol(Xt), lib_frac, 1.0) * Lmin   # independent of carrier/label
Xeq <- thin_lib(mat = Xt, thinlog2 = log2(lib / target))$mat
storage.mode(Xeq) <- "integer"
rownames(Xeq) <- rownames(X0); colnames(Xeq) <- colnames(X0)

# --- 3. leak diagnostics: lib-size correlation with each program's carrier indicator ---
libe <- colSums(Xeq)
carrier_cor <- sapply(seq_len(ncol(theta)), function(k) {
  cc <- (theta[, k] > 0)
  if (length(unique(cc)) < 2) NA else suppressWarnings(cor(libe, as.numeric(cc)))
})
cat(sprintf("library size: mean=%.0f sd/mean=%.3f | max |corr(lib, program-carrier)| = %.3f\n",
            mean(libe), sd(libe) / mean(libe), max(abs(carrier_cor), na.rm = TRUE)))

fwrite(data.table(gene = rownames(Xeq), Xeq), out_path, sep = "\t", compress = "gzip")
cat(sprintf("DONE -> %s\n", out_path))
