# SIG30 Knockouts: Differential Expression and Knockdown Checks

DESeq2 on the SIG30 arrayed CRISPR knockouts. Each of the 9 targets is compared with the
control guide, with replicate as a covariate:

```text
~ target + replicate      # control is the reference level
```

## Pipeline

- **`01_deg_analysis_SIG30.Rmd`**: DESeq2 on the UMI-deduplicated count matrix.
  - Keeps genes with at least 10 counts in at least 3 samples.
  - QC on VST values: sample-distance heatmaps (all samples and per replicate) and PCA
    (top 2,000 genes).
  - Extracts every `target` vs. `control` coefficient, with IHW-weighted p-values
    (`independentFiltering = FALSE`), and writes `res_targets_SIG30.csv`. Plots the
    number of DEGs per target (`padj < 0.1`).
- **`02_deg_visualization_SIG30.Rmd`**: plots of the DEG results.
  - Sample-correlation heatmaps on the union of DEGs (`padj < 0.01` in any target), using
    log2 normalized counts and, separately, z-scores against the control samples.
  - Knockdown check: bar plots of each target gene's own normalized counts in its KO vs.
    control, and a target-gene x KO-condition LFC heatmap with significance stars
    (`*` < 0.05, `**` < 0.01, `***` < 0.001). Off-diagonal cells show cross-target effects.

## Inputs

From `imports_stable/SIG30/processing_outs/` (produced outside this repo):

- `count_matrix_umiDeDup_SIG30.csv`, `processed_metadata_SIG30.csv` (`sample_ID`,
  `target`, `replicate`) and `featureNames_SIG30.csv` (`01`).
- `norm_counts_matrix_SIG30.csv` and `zscore_matrix_SIG30.csv` (`02`). These come from a
  SIG30 z-score processing step that is not in this repo.

## Outputs

Written under `analysis_outs/07_crispr_arrayed_validation_RNAseq_SIG30/`, which is gitignored:

- `res_targets_SIG30.csv` (`01`): DESeq2 results for every target vs. control. `02` and
  `../03_program_crossreg` read its stable copy,
  `imports_stable/SIG30/analysis_outs/res_targets_SIG30.csv`.
- `target_lfc_barplots_SIG30.pdf` and `target_lfc_heatmap_SIG30.pdf` (`02`).

## Running

Knit `01` first in the `R-signalseq` env (DESeq2, IHW). Then knit `02` in the `R-signalseq`
env. Both source `functions/r_custom/plotting_fxns.R`
from this repo.
