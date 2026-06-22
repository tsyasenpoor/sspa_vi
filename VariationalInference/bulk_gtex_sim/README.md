# GTEx Bulk RNA-seq Simulation (DRGP bulk arm)

Standalone scripts for the **bulk** semi-synthetic benchmark for the supervised Poisson
factorization (DRGP) model — the bulk counterpart to the single-cell `Simulations/` (`sim_flat_v1`)
arm. Imports nothing from `Simulations/` except `evaluate.py` for scoring.

Full design + cited research: `docs/superpowers/specs/2026-06-19-gtex-bulk-simulation-{design,research-findings}.md`.

## Goal

Semi-synthetic GTEx **Whole Blood** bulk RNA-seq with (a) known ground-truth gene programs,
(b) a known disease-label model, and (c) a **genetic CAD-PRS auxiliary feature that affects the
label but not expression**. Evaluate DRGP vs baselines on program recovery + disease-risk
prediction, and show the model attributes genetic risk to the aux channel (γ) without leaking it
into programs (β).

## Locked decisions (2026-06-19)

| Item | Decision |
|---|---|
| Reference | GTEx v8 **real read counts** (`gene_reads.gct.gz`), **Whole Blood** (755 RNA-seq subjects; 670 with WGS). |
| Null background | **SPsimSeq** (semi-parametric copula; learns real WB marginals + gene-gene correlation). |
| Program injection | **seqgendiff `thin_diff`** (binomial thinning; plants `θ*β*` low-rank signal on a log2 link). |
| Library-size leak | thinning lowers / additive inflates total counts → NEITHER auto-preserves. Use balanced design + **empirically verify `Σⱼxᵢⱼ ⊥ y`**. `type="thin"`, never `"mult"`. |
| Programs | K*=10 (3 disease-relevant + 7 nuisance). Effect calibrated to clear background copula modes. |
| Label | `y ~ Bernoulli(g(α + θ*υ* + PRS·γ* + clinical-aux))`, prevalence ≈ 26% (real heart_disease). |
| Genetic aux | real **CAD genetic score**, z-scored, **label-only**; γ* sweep {0,1.2,1.4,1.6,2.0} OR/SD. PENDING PI: a few top CAD SNPs vs a published PGS-Catalog score (both "GWAS numbers"; choice ≈ immaterial to sim). |
| n_sim | 2000 (PRS sampled from empirical/≈N(0,1)); +670 "matched-n" reference variant. |
| Model | DRGP `rw_base ≈ O(1)` (bulk dense: code auto-scales `rw *= nnz/n`); K_fit sweep (oracle + over-spec). |
| Additive A/B | DEFERRED — decide after a pilot (keep injection mechanism pluggable). |
| Baselines | gene L1-LR, scHPF→LR, PCA→LR (+ optional Spectra); all get PRS via `[X|X_aux]`. |
| Eval | `Simulations/evaluate.py` (Hungarian + support-AUPRC + activity-Spearman); subject-grouped AUC. |

## Pipeline (build order)

1. `build_wb_reference.py` — parse `gene_reads.gct.gz` → Whole Blood subset → protein-coding + min-expression QC → reference counts matrix (+ subject map, has_WGS flag). **[foundational]**
2. `compute_prs.py` — CAD genetic score on the WGS (670 subjects), z-scored + empirical distribution. **[PENDING PI on score type]**
3. `fit_spsimseq.R` (+ Python wrapper, mirrors `scdesign3/`) — fit SPsimSeq on WB counts → realistic null `X0`.
4. `inject_programs.py` — plant K* programs via `seqgendiff thin_diff` (pluggable: thinning | additive-Poisson); balanced design; verify `Σx ⊥ y`.
5. `make_labels.py` — label model `y ~ Bernoulli(g(α + θ*υ* + PRS·γ*))`; calibrate α to prevalence.
6. `emit_dataset.py` — write loader-compatible CSVs (ENSG gene cols + `heart_disease` + aux incl PRS).
7. Fit DRGP (`quick_reference.py`) + baselines (`comp/`); score with `evaluate.py`; aggregate.

## Environments

- Python: `jax_gpu` (`PYTHONPATH=/labs/Aguiar/SSPA_BRAY/BRay`).
- Genetics: `gtex_geno` (`bcftools`, `plink2`); PRS via `pgsc_calc` if using a published score.
- R (SPsimSeq + seqgendiff): new env (mirror `scdesign3/` setup); `Rscript` currently base-env only.

## Outputs → `data/Simulations/bulk_gtex_v1/`
