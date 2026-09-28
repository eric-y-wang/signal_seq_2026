# Interaction Scoring: Empirical Null Calibration of the Gene-Level GLM

The gene-level scoring in `05_interaction_scoring/` fits a Gamma-Poisson GLM (glmGamPoi) per
ligand pair and calls single-ligand and `ligand1:ligand2` effects significant at BH
`adj_pval < 0.1`. This folder builds an empirical null to test directly whether those p-values
are calibrated, and whether the real interaction calls differ from what noise alone produces.

The null population is cells that carry only non-targeting linker VLPs, with exactly one
singlet linker barcode in each round. The screen has 9 distinct linker barcodes: 3 used only in
round 1 (`linker1-3_round1`) and 6 used only in round 2 (`linker4-9_round2`). Their identities
stand in for ligand identities. No barcode carries any biological signal, so every
"significant" hit in this population is a false positive by construction.

Both null scripts copy the real GLM design (same covariates, size factors and contrasts) and
change only the population and the "ligand" labels:

```text
single term:  counts ~ ligand + replicate + lane + percent.mito + s.score + g2m.score
interaction:  counts ~ ligand1 * ligand2 + lane + replicate + percent.mito + s.score + g2m.score
```

## Pipeline

- **`01_glmGamPoi_single_term_null_slurm.r`** / **`01_r_job_submission_null_single.sh`**:
  single-term null.
  - **Null population.** Cells with `ligand_call_round{1,2}_DSB7 == "linker"` and exactly one
    linker barcode per round in `feature_call_DSB7`. Writes QC crosstabs of linker x
    replicate/lane and round-1 x round-2 pair counts.
  - **Pseudo-conditions.** Each of the 9 barcodes is one pseudo-ligand. Cells carrying it are
    `ligand = 1`, and all other null cells are the reference. In the real design the reference
    is `linker_linker`, but here it is restricted to the null population.
  - **Fit.** One SLURM job per pseudo-condition through `future.batchtools`. Genes expressed in
    fewer than `filter_cutoff` of cells are dropped per pseudo-condition. Default cutoff 0.05.
- **`02_glmGamPoi_interaction_null_slurm.r`** / **`02_r_job_submission_null_interaction.sh`**:
  interaction null.
  - **Pseudo-combos.** Every round-1 x round-2 barcode pair (3 x 6 = 18). Every null cell
    carries one barcode per round, so for any pair the whole null population splits into the
    4 arms of the real 2x2 design: round-1 only, round-2 only, both, and neither. The population
    and gene filter are the same for every combo, so they are fixed once up front. Default
    cutoff 0.1.
  - **Coverage gate.** A combo is fit only if every arm has at least 20 cells. At 0.1filter,
    14 of 18 combos are fit. The 4 `linker1_round1` x `linker6-9_round2` combos have no
    double-positive cells and are skipped. Coverage is written to
    `interaction_null_combo_coverage_*.csv`.
  - **Contrasts.** `ligand1:ligand2`, plus `ligand1` and `ligand2`. The two single-ligand
    contrasts give the "singles" table used as the single-term null in `03`/`04`.
- **`03_interaction_scoring_null_diagnostics.Rmd`**: calibration at `filter_cutoff = 0.1`,
  comparing the interaction null with the real `glmGamPoi_interaction` fit at the same cutoff.
  Tests are pooled into two contrasts: `ligand` (either single coefficient, 222,124 null
  gene-tests) and `ligand1:ligand2` (111,062 null gene-tests). It reports:
  - p-value histograms, null only and real vs. null
  - QQ plots: null only with a 95% beta-order-statistic band, and real vs. null, thinned to the
    top 4,000 plus 4,000 random points
  - p-value vs. |LFC| scatter, and |LFC| of `adj_pval < 0.1` hits, real vs. null
  - the empirical false-positive rate at `adj_pval < 0.1`, the real/null hit-rate enrichment,
    and an empirical FDR (null hit rate / real hit rate)

  | Contrast | Null FPR | Real hit rate | Real / null | Empirical FDR |
  | --- | --- | --- | --- | --- |
  | `ligand` | 0.58% | 27.9% | 48x | 2.1% |
  | `ligand1:ligand2` | 0.06% | 2.5% | 42x | 2.4% |

  The null false-positive rate is well below the nominal 10%, so the GLM is conservative rather
  than inflated.
- **`04_synergy_buffering_magnitude_test.Rmd`**: tests the claim in
  `interaction_scoring_v3.R` that synergy occurs mostly where the expected additive effect
  (`lfc_ligand1 + lfc_ligand2`) is near zero, and buffering where it is large, against the null.
  - **Rebuilt quantities.** Rebuilds expected, total and `interaction_score` at 0.1filter for
    both sources from the raw interaction and singles tables. It does not reuse
    `interactions_scored_v3_glmGamPoi_0.05filter.csv`, which was built at 0.05. The class is the
    sign of `interaction_score` (negative = buffering, positive = synergy).
  - **Percentile scale.** `|expected|` is converted to a percentile within each source's own
    distribution, because null coefficients are smaller across the board.
  - **Rate-matched selection.** Each source contributes its top fraction of gene-tests by
    interaction p-value, with the fraction set to the null's `adj_pval <= 0.1` rate. This avoids
    a shared cutoff that would take only the extreme tail of the null.
  - **Comparison.** Within each class, real vs. null percentiles are compared with Wilcoxon
    (rank-biserial) and two-sample KS.
  - **Result.** Real synergy calls sit at lower expected-effect percentiles than null synergy
    calls (mean 0.77 vs. 0.91; rank-biserial -0.52, p = 6e-7; KS D = 0.44), so the synergy claim
    is supported relative to the null. Buffering calls sit at percentile ~0.99 in both sources
    and cannot be distinguished. The clearest difference is composition: the real matched set
    is 91.5% buffering, while the null set is about 50/50.

## Inputs

- `imports_stable/SIG13/scanpy_outs/SIG13_doublets_DSB7.h5ad`: `counts` layer, and the
  `feature_call_DSB7`, `ligand_call_round{1,2}_DSB7`, replicate, lane and cell-cycle/mito
  covariates in `obs` (`01`, `02`).
- `imports_stable/SIG13/analysis_outs/glmGamPoi/glmGamPoi_{interaction,singles}_lfc_0.1filter.csv`:
  the real fit, written by `05_interaction_scoring/glmGamPoi_interaction_slurm.r` (`03`, `04`).
- `imports_stable/SIG13/analysis_outs/glmGamPoi/glmGamPoi_interaction_null_{lfc,singles_lfc}_0.1filter.csv`
  and `interaction_null_combo_coverage_0.1filter.csv`: stable copies of the `02` null outputs
  that `03` and `04` read, in the same flat folder as the real fit (the `glmGamPoi_null/`
  subfolder is not kept there). Re-running `01`/`02` writes fresh files to `analysis_outs/` only.

## Outputs

All outputs are written under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/`, which
is gitignored. The null GLMs have been run
at 0.05, 0.1 and 0.2filter.

- `glmGamPoi/glmGamPoi_null/`:
  - `glmGamPoi_singleTerm_null_{lfc,lfc_sig,coefficients}_{cutoff}filter.csv`: from `01`.
  - `glmGamPoi_interaction_null_{lfc,lfc_sig,singles_lfc,coefficients}_{cutoff}filter.csv`:
    from `02`. `lfc` is the `ligand1:ligand2` test and `singles_lfc` holds the `ligand1` and
    `ligand2` tests.
  - `interaction_null_combo_coverage_{cutoff}filter.csv`: per-combo arm sizes and skip reasons.
  - `qc_round{1,2}_linker_x_{replicate,lane}.csv`, `qc_round1x_round2_linker_pair_counts.csv`:
    null-population QC. Both scripts write these, and the output is deterministic.
- `plots/interaction_scoring_null/`:
  - from `03`: `glm_pval_histogram_{null,real_vs_null}.pdf`, `glm_QQ_{null,real_vs_null}.pdf`,
    `glm_pval_vs_effect_scatter.{pdf,png}` and `glm_effect_boxplot_significant_hits.pdf`
  - from `04`: `05_rate_matched_abs_additive_{boxplot,ecdf}.pdf`
- GLM checkpoints: `glmGamPoi_single_term_null_{cutoff}filter_checkpoints/` and
  `interaction_glmGamPoi_null_{cutoff}filter_checkpoints/` under `checkpoints/` (or under
  `$SIGNALSEQ_SCRATCH` if it is set). These hold one
  `.rds` per pseudo-condition or combo, and a rerun skips any that already exist.

## Running

```bash
cd analysis/02_combinatorial_screen_signalseq_SIG13/06_interaction_scoring_null
sbatch 01_r_job_submission_null_single.sh 0.1        # optional cutoff arg; default 0.05
sbatch 02_r_job_submission_null_interaction.sh 0.1   # optional cutoff arg; default 0.1
```

The driver jobs (8 cores, 200G, up to 6 h) run in the `R-deseq2` env, which `reticulate`/`anndata`
need to read the `.h5ad` (`conda = "auto"`). They fan out one `cpushort` worker per fit (1 core,
50G, 2 h). The wrappers `cd` into this folder, and driver logs go to `R-out.%j`/`R-err.%j` in the
submit directory.

`03` and `04` need the 0.1filter interaction null and the real 0.1filter interaction fit. Knit
them from this directory with the `R-signalseq` env active:

```bash
RENV_CONFIG_AUTOLOADER_ENABLED=FALSE \
RSTUDIO_PANDOC="$CONDA_PREFIX/bin" \
Rscript \
  -e "rmarkdown::render('03_interaction_scoring_null_diagnostics.Rmd')"
```
