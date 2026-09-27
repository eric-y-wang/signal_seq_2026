# Is Barcode Expression Level a Proxy for VLP Dose In a Way That Affects Results?

The production pipeline calls doublets at a DSB-normalized UMI threshold of 7 (DSB7)
and fits the interaction GLM on all resulting cells. If a cell's barcode expression
level reflected how much VLP it received, cells with high and low barcode expression
should give different ligand and interaction effects. This folder tests whether
barcode expression level can serve as that proxy, within the fixed DSB7 population.

**Result: it cannot.** Barcode expression is confounded with the cell's total RNA
content. Cells with high barcode expression also have larger RNA libraries, so
differences between the subsets reflect sequencing depth, not VLP dose.

Cells are ranked by mean barcode score (the mean DSB score of their two called
barcodes). Each analysis is repeated on three subsets:

- `top30`: the 30% of cells with the highest barcode expression
- `bottom30`: the 30% with the lowest
- `random30`: a random 30%, as a size-matched comparator

Comparing `top30` and `bottom30` against `random30` separates the effect of barcode
expression level from the effect of simply having fewer cells.

Barcode-calling QC itself, and the effect of changing the DSB threshold, are in
`../03_qc_barcode_cutoff`.

## Pipeline

- **`01_glmGamPoi_interaction_countSubset_slurm.r`**: the production interaction GLM
  (`../05_interaction_scoring`) at a hard-coded `filter_cutoff` of 0.1 (no command-line
  argument; the production default is 0.05), with each of the four groups (ligand 1 alone, ligand
  2 alone, the pair, `linker_linker`) subset to 30% of its cells before fitting. The
  subset is set by the `COUNT_SUBSET` environment variable. `random30` is seeded per
  ligand pair.
- **`r_job_submission_{top30,bottom30,random30}.sh`**: sbatch wrappers, one per
  subset.
- **`02_glm_coefficient_concordance_viz.Rmd`**: per ligand pair, Pearson r of the
  single-ligand and interaction LFCs between each subset and the full-data fit.
  Restricted to reproducible ligand pairs
  (`replicate_correlation_interactionLfc_0.2filter.csv`,
  `pearson_corr > 0.25 & num_shared_genes > 1`).
- **`03_spca_waggr_countSubset_scoring.ipynb`**: rescores the existing sPCA components
  (loadings are not refit) with `decoupler` `waggr` on each subset's
  `(condition, replicate)` pseudobulk. Subsets are taken within each pseudobulk
  unit, and scaling is done once on the full population. The full population is
  also scored as a check against the production scores. Also writes per-subset
  library-size QC.
- **`04_spca_score_concordance.Rmd`**: descriptive comparison, with no hypothesis
  tests:
  1. concordance of each subset's component scores with the full-data scores;
  2. whether each component's condition ranking holds (Spearman rho), which
     determines whether downstream synergy calls would change;
  3. whether the per-unit `top30 - bottom30` score shift tracks the library-size
     shift. `top30` cells have a median 53% more total counts than `bottom30`. This
     is the depth confound behind the result above.

## Inputs

All from `imports_stable/SIG13/`:
- `scanpy_outs/SIG13_doublets_DSB7.h5ad`: the production DSB7 cell set (01, 03).
- `analysis_outs/glmGamPoi/glmGamPoi_interaction/`: production GLM output (02).
- `analysis_outs/replicate_corr/replicate_correlation_interactionLfc_0.2filter.csv` (02).
- `analysis_outs/spca/`: sPCA loadings and production waggr scores (03).
- Stable copies of this folder's own earlier-step outputs: the per-subset GLM output (read
  by 02) and the 03 scores (read by 04). Re-running 01 or 03 writes fresh files to
  `analysis_outs/` only.

## Outputs

All under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/`:
- `glmGamPoi/glmGamPoi_interaction_{top30,bottom30,random30}count/`: GLM output per
  subset (01), with the same filenames as the production GLM. GLM checkpoints go to
  `checkpoints/` (or `$SIGNALSEQ_SCRATCH` if set).
- `qc_barcode_counts/correlation_diff_barcode_cutoffs.csv` and
  `plots/qc_barcode_counts/correlation_diff_barcode_cutoffs.pdf` (02).
- `qc_barcode_counts/spca_waggr_countSubset_scores_long.csv.gz` and
  `spca_waggr_countSubset_qc.csv` (03).
- `plots/qc_barcode_counts/spca_score_*.pdf` (04).

## Running

Run 01 in the `R-deseq2` env, 03 in `scanpy_standard2`, and knit 02 and 04 with
`R-signalseq`. Submit the jobs from this folder; driver logs go to `R-out.%j`/`R-err.%j` in
the submit directory.

```bash
sbatch r_job_submission_top30.sh
sbatch r_job_submission_bottom30.sh
sbatch r_job_submission_random30.sh

Rscript -e "rmarkdown::render('02_glm_coefficient_concordance_viz.Rmd')"
# run 03 in Jupyter, then
Rscript -e "rmarkdown::render('04_spca_score_concordance.Rmd')"
```
