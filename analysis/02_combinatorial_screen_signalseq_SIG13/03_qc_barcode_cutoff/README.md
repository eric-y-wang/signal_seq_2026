# Effect Consistency Across Barcode-Calling Cutoffs (DSB1-DSB10)

The production dataset calls doublets (cells with one round-1 and one round-2 ligand
barcode) at a DSB-normalized UMI threshold of 7 (`SIG13_doublets_DSB7.h5ad`). The
doublet *rate* is nearly flat across thresholds, but the doublet *identity* is not
(Jaccard vs. DSB7: 0.53 at DSB4, 0.80 at DSB6, 0.77 at DSB8). So the DSB7 choice
determines which cells enter the analysis.

This folder asks whether that choice changes the results. It re-calls the doublet
population at every threshold from DSB1 to DSB10, refits the interaction GLM, rescores
the sPCA components, and compares each against DSB7.

`../04_qc_barcode_counts` asks a complementary question: within the fixed DSB7
population, does barcode expression level serve as a proxy for the amount of VLP
each cell received in a way that affects results? (It cannot, because it is confounded with total RNA content.)

## Pipeline

- **`01_barcode_calling_comparison.ipynb`**: characterizes barcode calling itself.
  - Singlet / doublet / multiplet / uncalled rates across thresholds 1-10.
  - Overlap (Jaccard) of the doublet population at DSB4/5/6/8 with DSB7.
  - Bias in barcode scores by round position and by ligand.
  - Barcode score vs. library size. Barcode score tracks barcode-library depth
    strongly (Spearman ~0.81) and RNA library size moderately (~0.28).
- **`02_barcode_calling_visualization.ipynb`**: UMAP of DSB7 doublets on barcode
  expression alone, colored by called round-1 and round-2 barcode. Needs a GPU.
- **`03_generate_cutoff_datasets.ipynb`**: rebuilds the doublet population at each
  threshold by replaying the production filter chain (`../01_processing`) with the
  threshold as a parameter. It writes one `counts`-only h5ad per threshold, plus a
  single sPCA-gene `log1p_norm` file with per-threshold membership flags for notebook 06.
  - **Validation:** the regenerated DSB7 cell set must exactly match the production
    `SIG13_doublets_DSB7.h5ad` (349,678 cells).
  - The conditions and units present at every threshold are written to
    `cutoff_common_conditions.csv` / `cutoff_common_units.csv`, so every downstream
    comparison stays paired.
- **`04_glmGamPoi_interaction_cutoff_slurm.r`** + **`r_job_submission.sh`**: the
  production interaction GLM (`../05_interaction_scoring`), except that it reads
  `SIG13_doublets_DSB{k}.h5ad` and hard-codes `filter_cutoff` 0.1 (no command-line
  argument; the production default is 0.05), so results are compared with the production
  0.1filter fit. One SLURM job per threshold. DSB7 reuses the
  production output.
- **`05_glm_coefficient_concordance_viz.Rmd`**: per ligand pair, Pearson r of
  single-ligand and interaction LFCs between each threshold and DSB7, plus a full
  threshold x threshold heatmap. Restricted to the reproducible ligand pairs defined
  from DSB7, held fixed across thresholds.
- **`06_spca_waggr_cutoff_scoring.ipynb`**: scores the existing sPCA components
  (loadings are not refit) on each threshold's pseudobulk with `decoupler` `waggr`,
  scaling within each threshold's own population. DSB7 must reproduce the production
  scores at r ~ 1.
- **`07_spca_score_concordance.Rmd`**: descriptive comparison of component scores
  vs. DSB7. It covers score concordance, a threshold x threshold heatmap, and whether
  each component's condition ranking holds. A depth check tests whether the
  `DSB10 - DSB1` score shift tracks the library-size shift that the threshold itself
  induces.

## Population sizes

| threshold | final cells | conditions | median cells/condition |
|---|---|---|---|
| DSB1 | 81,057 | 660 | 103 |
| DSB2 | 160,182 | 660 | 199 |
| DSB3 | 251,859 | 660 | 313 |
| DSB7 | 349,678 | 660 | 453 |
| DSB10 | 169,075 | 658 | 219 |

`cutoff_dataset_summary.csv` has all ten thresholds.

## Caveats

- **DSB10 loses two conditions** (`IL23_IL1A`, `IL23_IL9`). Neither is in the
  reproducible set, and all comparisons are restricted to the common conditions.
- **DSB1-2 are a non-representative slice.** At DSB1, 86% of cells carry 3+ barcodes
  and are excluded as multiplets. The surviving two-barcode cells have the lowest
  barcode and RNA depth, so poor concordance at these thresholds is partly a
  selection artifact.
- **Populations overlap.** Each threshold shares 53-92% of its cells with DSB7, so
  correlations are consistency measures, not null-calibrated tests.
- **Cell numbers differ by threshold**, so cell counts per condition are plotted
  against concordance.

## Inputs

- `imports_stable/SIG13/scanpy_outs/SIG13_full_bc_processed.h5mu`: all 628,244 cells before doublet filtering, with
  DSB-normalized barcode scores and RNA counts. Every threshold is derived from it. Download it
  from GEO ([GSE318270](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE318270)).
- `imports_stable/SIG13/analysis_outs/spca/zscore_degs_allLigands_0.1_alpha1.0_sPCA_loadings.csv`:
  sPCA loadings (78 components).
- Production DSB7 GLM output (`imports_stable/SIG13/analysis_outs/glmGamPoi/`)
  and waggr scores (`imports_stable/SIG13/analysis_outs/spca/degs_zscore_allLigands/`).
- `imports_stable/SIG13/analysis_outs/replicate_corr/replicate_correlation_interactionLfc_0.2filter.csv`:
  reproducibility filter (`pearson_corr > 0.25 & num_shared_genes > 1`).

The later steps read the stable copies of the earlier steps' outputs in
`imports_stable/SIG13/`: 05 reads the 03 tables, and 07 reads the 06 scores. Re-running an
earlier step writes fresh files to `analysis_outs/` only.

The cutoff-sweep `.h5ad`s (about 74 GB) are **not** in `imports_stable/` because of size
limitations. To re-run 04 or 06, first run 03, which writes them to
`analysis_outs/02_combinatorial_screen_signalseq_SIG13/scanpy_outs/cutoff_sweep/`; 04 and 06 read
them from there. 05 and 07 do not need them.

The per-threshold GLM output (`glmGamPoi_interaction_DSB{k}/`) and
`cutoff_common_units.csv` are also not in `imports_stable/`. To knit 05, run 03 and 04, then
copy `glmGamPoi_interaction_DSB{k}/` into `imports_stable/SIG13/analysis_outs/glmGamPoi/`,
where 05 reads it. 07 still runs from `imports_stable/`.

## Outputs

All under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/`:
- `qc_barcode_cutoff/`: barcode-calling tables (01), cutoff dataset summary and common
  sets (03), GLM concordance (05), sPCA scores and QC (06).
- `scanpy_outs/cutoff_sweep/`: one `.h5ad` per threshold and the sPCA-gene sweep `.h5ad` (03).
- `glmGamPoi/glmGamPoi_interaction_DSB{k}/`: GLM output per threshold (04). GLM checkpoints
  go to `checkpoints/` (or `$SIGNALSEQ_SCRATCH` if set).
- `plots/qc_barcode_cutoff/`: figures.

## Running

| step | environment |
|---|---|
| 01, 03, 06 | `scanpy_standard2` |
| 02 | `rapids_singlecell`, on a GPU node |
| 04 | `R-deseq2` |
| 05, 07 | `R-signalseq` |

```bash
# 04: one GLM job per threshold, each in its own run_DSB{k}/ directory
for k in 1 2 3 4 5 6 8 9 10; do sbatch r_job_submission.sh $k; done

Rscript -e "rmarkdown::render('05_glm_coefficient_concordance_viz.Rmd')"
Rscript -e "rmarkdown::render('07_spca_score_concordance.Rmd')"
```

Submit from this folder. The job runs in `run_DSB{k}/` here, and its driver logs go to
`R-out.%j`/`R-err.%j` in the submit directory.

A GLM job may stop partway with a `future.batchtools` log-file error. Per-pair
checkpoints make re-running safe: re-submit a threshold until its checkpoint
directory holds 594 `.rds` files.
