# Cytokine x TNF Interactions in Chromatin Accessibility (SIG16, bulk ATAC-seq)

Bulk ATAC-seq of mouse activated CD4 T cells stimulated with one of three `ligand1` cytokines
(`IL4`, `IL6`, `TGFb`) or none, crossed with `ligand2` = `TNF` or none, in 3 mice
(24 samples). The question is whether the ligand:TNF interactions seen transcriptionally in
SIG13 (`../02_combinatorial_screen_signalseq_SIG13`) also show up at the level of chromatin
accessibility, and which TF motifs mark the interacting peaks.

| ligand1 | ligand2 | conditions |
|---|---|---|
| `none` | `none` | `none_none` (control) |
| `IL4` / `IL6` / `TGFb` | `none` | `IL4_none`, `IL6_none`, `TGFb_none` |
| `none` | `TNF` | `none_TNF` |
| `IL4` / `IL6` / `TGFb` | `TNF` | `IL4_TNF`, `IL6_TNF`, `TGFb_TNF` |

Peaks are counted over the IDR 0.05 consensus peak atlas (`idr-0.05`) and modelled with
DESeq2 as a 2-factor design:

```text
~ mouse + ligand1 + ligand2 + ligand1:ligand2        # none / none are the reference levels
```

Each `ligand1 x TNF` pair is scored with the same definition as SIG13
(`interaction_scoring_v3.R`), using the coefficients `b1`, `b2`, `b_int` of that fit:

```text
lfc_additive_expected       = b1 + b2
lfc_total_interaction_model = b1 + b2 + b_int
interaction_score           = b_int / lfc_total_interaction_model   (0 if interaction n.s.)
```

Peaks with `padj_interaction <= 0.05` and `|b_int| > log2(1.5)` are called
`synergy positive` (score > 0, total > 0), `synergy negative` (score > 0, total < 0) or
`buffering` (score < 0). Peaks whose combined condition (`{ligand1}_TNF` vs `none_none`)
passes `padj <= 0.05` and `|LFC| > log2(1.5)` without a significant interaction are `none`;
all others are `NA`. The `|LFC|` gate is an ATAC-specific addition to the SIG13 rule; the
deviations are explained in `03_interaction_classification.Rmd`.

## Pipeline

- **`01_deseq2_qc.Rmd`**: builds the DESeq2 object over the peak atlas.
  - Sample metadata parsed from `{ligand1}_{ligand2}_{rep}` sample names (replicate = mouse).
  - Sets the condition / ligand colour palette (RColorBrewer `Paired`) used by `02`–`03`.
  - QC: FRiP from the `featureCounts` summary, dispersion plot, PCA on the VST of the top
    20% most variable peaks, and a sample-sample correlation heatmap.
  - Set `refit <- TRUE` to re-run `DESeq()`; otherwise the saved object is reloaded.
- **`02_differential_accessibility.Rmd`**: pulls ten unshrunken contrasts from the fit:
  each `ligand1` alone (mode 1, 3 contrasts), TNF alone (mode 3), the interaction term
  (mode 5, 3 contrasts) and each combination vs `none_none` (mode 6, `list()` contrast
  `b1 + b2 + b_int`, 3 contrasts). Peaks are annotated with the nearest gene (within 50 kb
  of a TSS). Writes per-contrast result tables, MA and normalized-count plots, and two
  multi-panel volcano figures. Significance: `padj < 0.05`, `|LFC| > log2(1.5)`.
- **`03_interaction_classification.Rmd`**: joins modes 1/3/5/6 per `ligand1 x TNF` pair,
  scores and classifies every peak (see above), and checks that mode 6 equals
  `b1 + b2 + b_int`. Plots class counts, additive-expectation vs combined-effect scatters,
  score vs total-effect scatters, and ComplexHeatmap heatmaps of per-peak z-scored VST
  accessibility split by class.
- **`04_motif_enrichment.Rmd`**: one-sided hypergeometric test of FIMO motif hits in 17
  region sets against the whole atlas: gained / lost peaks for each of TGFb, IL4, IL6 and
  TNF (8 sets), and the three interaction classes for each pair (9 sets). P-values are
  BH-adjusted within each region set. Plots the top motifs per set and bidirectional
  (gained vs lost, synergy positive vs negative) bar plots.

## Run order

```text
01 ──> 02 ──> 03 ──> 04
```

Each step reads the earlier steps' outputs from their stable copies in
`imports_stable/SIG16/analysis_outs/` (so any step can be knit on its own); re-running a step
writes fresh output to `analysis_outs/04_tnf_interaction_ATACseq_SIG16/`.

## Inputs

All read from `imports_stable/`. Produced outside this repo (peak calling, IDR atlas, `Rsubread::featureCounts`,
ChIPseeker and FIMO), under `imports_stable/SIG16/analysis_outs/`:

- `deseq2-merged-reps/idr-0.05/counts_mat.rds` — `featureCounts` output over the peak atlas.
- `chipseeker/idr-0.05/all.peakatlas/annotation.tsv` — nearest gene / distance to TSS per peak.
- `fimo/idr-0.05/motif_mtx.rds` — peaks x motifs binary FIMO hit matrix.
- `motif_info.rds` and `plots/pal_family.rds` — motif-to-TF/family map and family palette (`04`).

Stable copies of earlier-step outputs, in the same folder:

- `deseq2-merged-reps/idr-0.05/{deseq2_dataset,metadata,normalized_counts}.rds` and
  `plots/palette.rds` (`01`; `01` itself reloads the stored fit unless `refit <- TRUE`).
- `unshrunken-res.table.rds` for the ten contrasts `02` writes (`name-ligand1_<L>_vs_none`,
  `name-ligand2_TNF_vs_none`, `name-ligand1<L>.ligand2TNF`,
  `contrast-condition_<L>_TNF_vs_none_none`), read by `03` and `04`.
- `name-ligand1<L>.ligand2TNF/interaction-classification.csv` (`03`), read by `04`.

`03` also compares against the RNA interaction calls in
`imports_stable/SIG13/analysis_outs/glmGamPoi/interactions_scored_v3_glmGamPoi_0.05filter.csv`
and `imports_stable/SIG18/analysis_outs/deg/res_interaction_scored_SIG18.csv`.

## Outputs

Written under `analysis_outs/04_tnf_interaction_ATACseq_SIG16/` (not tracked):

- `deseq2-merged-reps/idr-0.05/`: `deseq2_dataset.rds`, `metadata.{rds,csv}`,
  `normalized_counts.rds` (`01`, only when `refit <- TRUE`); one subfolder per contrast
  (`name-<term>/` or `contrast-<condition>/`) with `unshrunken-res.table.{csv,rds}` (`02`);
  `interaction-classification{,-sig,-summary}.csv` in each `name-ligand1<L>.ligand2TNF/`
  folder, and `interScored_SIG16_ATAC_idr-0.05{,_sig}.csv` for all pairs (`03`).
- `motif_enrichment-hypergeometric-merged-reps/idr-0.05/<region set>.csv` (`04`).
- Plots under `plots/`, `plots_new/` and `plots/motif_enrichment-hypergeometric-merged-reps/`,
  including `plots/palette.rds` (`01`).

The original SIG16 output folder also holds `interScored_SIG16_ATAC.csv`, an archived table
from a different (nf-core consensus) peak atlas with `Interval_NNNNN` peak IDs. It is not used
here and was not copied into `imports_stable/`; do not join it to the `idr-0.05` results.

## Running

Knit the Rmds in numbered order, in the `R-signalseq` env (tidyverse, DESeq2, ComplexHeatmap,
circlize, patchwork, pheatmap, ggrepel). Each finds the repo root by searching upward for
`imports_stable/` and sources `functions/r_custom/plotting_fxns.R` from there, so knit them
from inside the repo (the default, from this folder, works).
