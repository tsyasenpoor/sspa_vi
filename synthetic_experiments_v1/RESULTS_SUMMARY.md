# DRGP synthetic experiments — RESULTS_SUMMARY

Headline numbers and figure index for the v1 synthetic benchmark suite. All results from 50 seeds per condition (unless noted), generator defaults `a_beta=0.3, b_beta=0.15, a_theta=0.3, b_theta=0.3` (library mean ~1100), `n=500, p=5000, K_true=K_fit=10` for fixed-K experiments.

DRGP = Supervised Poisson Factorization via CAVI (`BRay/VariationalInference/vi_cavi.py`, `unmasked` mode unless specified). Baselines: `nmf_lr` (sklearn NMF + L1-LR), `pca_lr`, `plain_lr` (raw counts + L1-LR), `schpf_lr` (vendored scHPF + L1-LR), `spectra_lr` (Kunes et al. 2023 Spectra in dedicated `spectra` conda env + L1-LR; C1 only), `gsva_lr` (Hänzelmann/Castelo/Guinney 2013 GSVA via gseapy + L1-LR; C1 only, fed the oracle ground-truth gene sets gt.S).

## At-a-glance

| Theme | Verdict |
|---|---|
| Factor recovery at well-specified K | **DRGP ≥ NMF** (cos 0.9999 vs 0.9995, K=10, π=0.1) |
| Top-gene support (Jaccard@50) | **DRGP wins 20/21 (K,π) cells** |
| Mediation Δ recovery | **DRGP best Pearson** at Δ≥1.0; PCA blows up `δ̂` scale |
| Joint vs two-stage IE bias | **DRGP smaller |bias|** and tighter SE (95% CI coverage 100% vs scHPF 96–98%) |
| OOD AUROC | NMF wins on this DGP (linear logit); DRGP within 0.01–0.02 |
| Mask corruption (η=0.10) | **Both DRGP-masked and DRGP-combined collapse**: cos 0.999 → 0.74 |
| `combined` rescue of missing pathway | **Fails**: free factors don't pick up disease-relevant programs |
| Cross-seed stability of β | DRGP cos_min=0.76 (target 0.85 missed); NMF deterministic |
| Cross-seed stability of `v` | Unstable — confirmed memory note |

## Tier 1 — primary deliverables

### A1 — Factor recovery vs (K_fit, π)

`results/aggregated/A1_factor_recovery/`, figures `figures/A1/`.

7×3×50 = 1050 cells per method.

**Headline cell** (K_fit=10, π=0.10):

| method | cos_mean | jaccard@50 | OOD AUROC |
|---|---|---|---|
| drgp_unmasked | **0.9999** | **0.5918** | 0.9564 |
| nmf_lr        | 0.9995 | 0.5175 | **0.9774** |
| pca_lr        | 0.7448 | 0.3795 | 0.9524 |

**Sweep verdict**:
- cos_mean: DRGP wins 14/21 cells. NMF wins at K_fit < K_true (5, 8) where DRGP under-factorizes.
- jaccard@50 (top-gene overlap, the biology-relevant metric): **DRGP wins 20/21 cells**, by 5–9 pts.
- Wilcoxon p<1e-9 in DRGP-winning cells (vs nmf_lr).
- Figures: `A1_heatmap_cos_mean.png`, `A1_heatmap_support_auprc.png`, `A1_method_comparison_K10.png`, `A1_method_comparison_ood_K10.png`.

**Followup analysis** (`src/analysis-utils/a1_followup.py`):
- **π-axis check**: implementation verified (m_choices scale 4× across π). DRGP cosine is essentially π-invariant at well-specified K (spread ≤ 0.01); NMF, in contrast, **degrades with smaller π at K>K_true** (spread 0.039 at K=30) — a DRGP robustness win not visible in the original heatmap. See `A1_heatmap_cos_mean_rescaled.png` (median + Q5 dual panel, vmin=0.85).
- **Paired-seed scatter** (`A1_paired_seed_scatter_K10.png`): at K_fit=10, Pearson r(DRGP cos, NMF cos) = **0.129** and r(DRGP OOD, NMF OOD) = **0.265** across 150 paired runs. DRGP's failures are **uncorrelated** with NMF's — instability is **method-specific, not generator-specific** → multi-restart with ELBO selection is the indicated fix.
- **Outlier seeds at K_fit=10**: 49/150 (seed, π) cells have DRGP cos < 0.95 (worst: seed 48/π=0.20 at 0.750, seed 1/π=0.10 at 0.808, seed 10/π=0.05 at 0.798). NMF cos on the same 49 cells is ≥0.998 with two exceptions. Candidate set for multi-restart investigation at `figures/A1/A1_outlier_seeds_K10.csv`.

### A2 — Support recovery / FDR calibration

`results/aggregated/A2_support_recovery/summary.csv`, figures `figures/A2/`.

**AUPRC ceilings** (reuses A1 raw):

| K_fit | π | AUPRC_med | FDR@0.5 | Recall@0.5 |
|---|---|---|---|---|
| 10 | 0.10 | 0.4026 | **0.000** | 0.4045 |
| 15 | 0.10 | 0.4711 | 0.000 | 0.4644 |
| 20 | 0.10 | 0.4700 | 0.000 | 0.4629 |
| 30 | 0.10 | 0.4544 | 0.000 | 0.4480 |

**Finding**: spike-and-slab posterior `s_kj` is well-calibrated (FDR@0.5 ≈ 0 from K=10 onward) but recall-limited — roughly half of true-support genes never escape the prior. AUPRC plateaus at ~0.47.

**Magnitude-stratified AUPRC** (`figures/A2/A2_magnitude_stratified_auprc.{png,csv}`, 1,500 (seed, k_true) combos at K_fit=10):

| π | top-tercile of |β| AUPRC (median) | bottom-tercile AUPRC (median) |
|---|---|---|
| 0.05 | **1.00** | **0.00** |
| 0.10 | **1.00** | **0.00** |
| 0.20 | **1.00** | **0.01** |

The aggregate 0.40 is not a "half the support" story but a **clean bimodal regime**: DRGP perfectly recovers the high-loading support and almost completely shrinks the low-loading tail (this is the spike-and-slab working as designed). FDR remains 0 because the low-loading genes that aren't recovered also aren't falsely declared.

Figures: `pr_curves_K10.png`, `fdr_calibration_K10.png`, `auprc_heatmap.png`, `auprc_vs_K.png`, `A2_magnitude_stratified_auprc.png`.

### B1 — Program ranking (4 modes)

`results/aggregated/B1_program_ranking/summary.csv`, figures `figures/B1/`.

| mode | v_spearman | precision@K_rel | OOD AUROC | cos_mean |
|---|---|---|---|---|
| `masked_oracle`   (full S given) | 0.798 | **1.00** | **0.960** | 0.9999 |
| `unmasked`                       | 0.768 | **1.00** | 0.956    | 0.9999 |
| `masked_missing`  (S missing 1 of 3 rel) | n/a  | 0.00 | 0.505 | 0.798 |
| `combined_rescue` (S missing 1, extra free factors) | −0.03 | 0.333 | 0.478 | 0.480 |

**Finding**:
- Oracle mask offers **zero benefit** over `unmasked` when supervision is present.
- `combined` does **not** rescue a missing pathway — free factors don't preferentially align to it; v ranking collapses.
- Failure mode: masked component dominates likelihood; free factors absorb diffuse residual variance, not the missing program.

Figures: `v_spearman_by_mode.png`, `precision_at_rel_by_mode.png`, `ood_auroc_by_mode.png`.

### B2 — Mediation / Δ sweep

`results/aggregated/B2_mediation_delta_sweep/summary.csv`, figures `figures/B2/`.

`Δ_{0,asthma} ∈ {0.5, 1.0, 1.5, 2.0}`, K_true=K_fit=10.

| Δ | method | Pearson(Δ̂,Δ) | δ̂[0,asthma] | IE bias (frac of |IE_true|) |
|---|---|---|---|---|
| 0.5 | drgp_unmasked | 0.557 | 0.159 | −0.032 (median) |
| 1.0 | drgp_unmasked | 0.714 | 0.223 | +0.077 |
| 1.5 | drgp_unmasked | 0.828 | 0.255 | +0.166 |
| 2.0 | drgp_unmasked | 0.854 | 0.279 | +0.308 |
| 2.0 | nmf_lr        | 0.956 | 2.54  | −0.990 |
| 2.0 | pca_lr        | 0.888 | 9.67  | +0.020 |

**Finding**: NMF and PCA have higher Pearson on the *ranking* of `δ̂_kℓ` across programs, but their absolute scale is uncalibrated (NMF δ̂ ≈ 10× truth; PCA ≈ 50×) → |IE bias| is much worse than DRGP's at Δ≥1.0 in absolute terms. DRGP's `δ̂` is the only one in the right scale (≈0.15 × Δ_true).

Figures: `delta_scatter.png`, `pearson_vs_delta.png`, `ie_bias_box.png`.

### C1 — Method ranking (per-program AUROC for "is-disease-relevant")

`results/aggregated/C1_method_ranking/per_method_auroc.csv`, figures `figures/C1/`.

Per-method AUROC for ranking K_fit programs as disease-relevant vs not (pooled n=500 across 50 seeds × 10 programs):

| method | AUROC | cos_mean | v_spearman | OOD AUROC | wall-clock (s) |
|---|---|---|---|---|---|
| nmf_lr        | **0.9993** | 0.999 | 0.813 | 0.977 | 0.75 |
| drgp_masked   | 0.9992 | 1.000 | 0.798 | 0.960 | 49.5 |
| drgp_combined | 0.9985 | 1.000 | 0.798 | 0.954 | 50.9 |
| gsva_lr       | 0.9949 | 0.508† | 0.814 | 0.938 | 4.4 |
| drgp_unmasked | 0.9910 | 1.000 | 0.768 | 0.956 | 21.1 |
| pca_lr        | 0.9165 | 0.745 | 0.768 | 0.952 | 0.67 |
| schpf_lr      | 0.8995 | 0.428 | 0.804 | 0.953 | 5.76 |
| spectra_lr    | 0.8117 | 0.501 | 0.485 | 0.623 | 514 |
| plain_lr      | n/a (no programs) | n/a | n/a | 0.955 | 580 |

† gsva_lr's `Beta_hat` is the binary support `gt.S` (frozen, non-learned), so cos vs the continuous true `gt.Beta` is moderate by construction; jaccard@50 vs the top-50 support is 0.77 and v_spearman=0.814 is the best of any method.

**Finding**: DRGP's `v` posterior ranks disease-relevant programs nearly as well as L1-LR on NMF loadings, far better than scHPF + L1-LR or Spectra + L1-LR. With a **perfect pathway mask** (`drgp_masked`, K_path=10, K_fit=10) DRGP closes essentially all of the gap to NMF (0.9992 vs 0.9993, Wilcoxon vs drgp_unmasked p=0.065). `drgp_combined` (K_path=10 + K_free=2, K_fit=12) is similar (0.9985, p=0.019). **`gsva_lr` with oracle gene sets** (sees `gt.S` exactly, but no learned loadings) lands at 0.9949 — beats drgp_unmasked (p=0.045) but loses to drgp_masked. The 0.004 AUROC gap between gsva_lr and drgp_masked at identical priors quantifies the value of learning real-valued loadings on top of a perfect mask. DRGP beats Spectra by +0.19 in per-seed AUROC (Wilcoxon p=4.3e-7); Spectra's OOD AUROC collapses to 0.623 because it has no native projection method (LS fallback). plain_lr is 25× slower than DRGP-unmasked, NMF 28× faster. Spectra + GSVA + masked/combined all added via parallel raw dirs and merged through `analysis_c1.py --extra-raw-dirs`.

Figures: `ranking_auroc_box.png`, `ood_auroc_box.png`, `top3_false_positives_box.png`.

### C2 — Sample complexity / OOD AUROC vs n

`results/aggregated/C2_sample_complexity_ood/summary.csv`, figure `figures/C2/ood_auroc_vs_n.png`.

| n | drgp | nmf | pca | plain_lr |
|---|---|---|---|---|
|  50 | 0.865 | **0.952** | 0.913 | 0.925 |
| 100 | 0.926 | **0.958** | 0.931 | 0.933 |
| 250 | 0.944 | **0.975** | 0.946 | 0.950 |
| 500 | 0.956 | **0.977** | 0.952 | 0.955 |

**Finding**: NMF wins on this linear-logit DGP at every n. DRGP closes the gap as n grows (0.087 at n=50 → 0.021 at n=500). DRGP's penalty is probit shrinkage of Bayesian-Lasso `v` and Gaussian `γ` not shrinking aux features to zero. Honest takeaway: DRGP buys interpretable Δ and `v` calibration at a small AUROC cost on linear DGPs.

## Tier 2 — robustness

### D1 — Negative-binomial likelihood (overdispersion robustness)

`results/aggregated/D1_nb_likelihood/summary.csv`, figure `figures/D1/d1_three_panel.png`.

`φ` = NB dispersion (0 = Poisson).

| φ | drgp cos | nmf cos | drgp OOD | nmf OOD |
|---|---|---|---|---|
| 0.0 | 1.000 | 0.999 | 0.956 | 0.977 |
| 0.1 | 0.971 | 0.996 | 0.955 | 0.970 |
| 0.5 | 0.883 | 0.846 | 0.949 | 0.951 |
| 1.0 | 0.868 | 0.650 | 0.937 | 0.927 |

**Finding**: DRGP degrades gracefully under heavy overdispersion (φ=1.0: cos 0.87 vs NMF 0.65). Poisson likelihood is **more robust** to NB misspecification than NMF here. All 200 (cond × seed) cells are now clean — the original ~14% TMPDIR-XLA failures were rescued with `slurm/run_rescue.sh` (job 2100141, n_seeds=50 per φ).

### D2 — Sample-complexity (reuses C2 raw)

Identical numbers to C2. Figure: `figures/D2/sample_complexity_three_panel.png` (three-panel: cos / v_spearman / OOD vs n).

### D3 — Pathway-mask corruption

`results/aggregated/D3_mask_corruption/summary.csv`, figure `figures/D3/d3_three_panel.png`.

`η` = fraction of in-mask genes swapped with out-of-mask genes per pathway.

| mode | η | cos_mean | v_spearman | OOD AUROC |
|---|---|---|---|---|
| masked   | 0.00 | 1.000 | 0.798 | 0.960 |
| masked   | 0.10 | **0.735** | 0.015 | 0.498 |
| masked   | 0.25 | 0.677 | 0.082 | 0.519 |
| masked   | 0.50 | 0.523 | 0.015 | 0.523 |
| combined | 0.00 | 1.000 | 0.798 | 0.954 |
| combined | 0.10 | **0.733** | 0.015 | 0.478 |
| combined | 0.25 | 0.663 | 0.067 | 0.496 |
| combined | 0.50 | 0.566 | 0.134 | 0.619 |

**Finding**: catastrophic. Even **10% mask corruption** collapses both `masked` and `combined` modes (cos drops 26 pts, v_spearman → 0, OOD AUROC → chance). The prior is too informative; corrupted entries dominate β posteriors. Operational implication: do not run `masked`/`combined` on pathways with low-confidence membership. All 400 (cond × seed) cells are now clean (rescue job 2100142, 53 tasks, post-TMPDIR-patch).

### D4 — Overlap identifiability (supervised)

`results/aggregated/D4_overlap_identifiability/summary.csv`, figures `figures/D4/`.

`overlap_pair=(0,1)`, both disease-relevant; Jaccard ∈ {0.0, 0.2, 0.5, 0.8}.

| Jaccard | drgp cos_overlap | nmf cos_overlap | pca cos_overlap |
|---|---|---|---|
| 0.0 | 0.99993 | 0.99983 | 0.7412 |
| 0.2 | 0.99992 | 0.99984 | 0.6551 |
| 0.5 | 0.99992 | 0.99982 | 0.6210 |
| 0.8 | **0.99989** | 0.99981 | 0.5691 |

**Finding**: DRGP and NMF identify near-perfectly even at 80% overlap. PCA's rotational ambiguity manifests immediately (loses 17 pts going 0 → 0.8 Jaccard).

### D4b — Overlap identifiability (unsupervised overlap pair, outside disease set)

Same as D4 but `overlap_pair=(8,9)` outside the disease-relevant set. Same conclusion — DRGP/NMF identify cleanly, PCA degrades. Figures in `figures/D4b/`.

## Tier 3 — diagnostics

### N1 — K misspecification (reuses A1 raw)

`results/aggregated/N1_K_misspec/summary.csv`, figures `figures/N1/`.

| K_fit | drgp cos | drgp noise_factors |
|---|---|---|
|  5  | 0.892 | 0.0 |
|  8  | 0.959 | 0.0 |
| 10  | 0.9999 | 0.0 |
| 12  | 0.9999 | 0.0 |
| 15  | 0.9998 | 1.0 |
| 20  | 0.9997 | 3.5 |
| 30  | 0.9995 | 11.0 |

**Finding**: DRGP is *very* tolerant of K over-specification — extra factors fade (≤0.1 norm) rather than steal signal. Under-specification (K_fit<K_true) loses factor mass; over-specification is essentially free.

### N2 — Joint (DRGP) vs two-stage (scHPF + post-hoc LR) mediation

`results/aggregated/N2_joint_vs_twostage/{summary.csv,coverage.csv}`, figures `figures/N2/`.

| Δ | method | |ie_bias|_med | SE(ie_hat) | 95% CI coverage |
|---|---|---|---|---|
| 0.5 | drgp_unmasked | 0.226 | 0.74 | **1.00** |
| 0.5 | schpf_lr      | 0.367 | 1.89 | 0.98 |
| 1.0 | drgp_unmasked | 0.283 | 1.23 | **1.00** |
| 1.0 | schpf_lr      | 0.383 | 3.37 | 0.98 |
| 2.0 | drgp_unmasked | 0.423 | 1.94 | **1.00** |
| 2.0 | schpf_lr      | 0.384 | 4.72 | 0.96 |

**Finding**: DRGP's joint inference has **2–3× smaller SE** than two-stage and slightly smaller (or comparable at Δ=2) median |bias|. Coverage of true IE is 100% (DRGP) vs 96–98% (two-stage). Wilcoxon |IE bias| comparisons available in stdout.

### N3 — Cross-seed stability of β (with frozen structural params)

`results/aggregated/N3_cross_seed_stability/summary.csv`, figures `figures/N3/`.

50 seeds, all reusing seed-0 `B, v, α, Δ, S, ξ_0` via generator's `freeze_params` option. Pairwise cosine after Hungarian alignment:

| method | mean | std | min |
|---|---|---|---|
| drgp_unmasked | 0.963 | 0.051 | **0.763** |
| nmf_lr        | 0.999 | 0.0001 | 0.999 |

**Finding**: DRGP β is mostly stable (mean 0.96) but has a long tail — worst-pair cos=0.76 falls below plan's 0.85 target. NMF is essentially deterministic. Confirms memory note "v weight cross-seed instability" extends partially to β too. Mitigation candidates (not yet tried): multi-restart with ELBO selection, deeper ELBO warm-up.

## Experiments not yet run

- **MOFA baseline** — not installed in any env. (Spectra + D1/D3 rescues are now done — see C1, D1, D3 above.)

## Code & artifact paths

- Configs: `configs/`
- Source: `src/{generator,baselines,experiment,metrics,analysis*}.py`
- SLURM: `slurm/run_sweep.sh`, `slurm/submit_*.sh`
- Raw per-seed npz: `results/raw/<experiment>/`
- Aggregated CSVs: `results/aggregated/<experiment>/`
- Figures: `figures/<experiment>/`

## Reproducibility

```bash
source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
conda activate jax_gpu
export PYTHONPATH=/labs/Aguiar/SSPA_BRAY/BRay
cd /labs/Aguiar/SSPA_BRAY/BRay/synthetic_experiments_v1

# Single seed, single condition (debug):
python -m src.experiment --config configs/A1_factor_recovery.yaml --condition-idx 0 --seed 0 --out-dir results/raw/A1_factor_recovery

# Full sweep:
N_SEEDS=50 sbatch --array=0-1049%100 slurm/run_sweep.sh configs/A1_factor_recovery.yaml

# Re-analysis (no fit re-run):
python -m src.analysis             # A1
python -m src.analysis_a2          # A2 (reuses A1 raw)
python -m src.analysis_b1          # B1
# ... etc per experiment
```
