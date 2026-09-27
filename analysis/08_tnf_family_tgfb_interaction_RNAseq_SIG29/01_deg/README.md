# SIG29 Differential Expression and Gene-Level Interactions

DESeq2 on the SIG29 samples. It estimates every condition's effect relative to
`none_none`, then tests each TGFb + TNF-family combination for a non-additive interaction:

```text
~ condition + replicate                          # none_none is the reference level
interaction contrast: combo - TGFb_none - none_<member>
```

The second Rmd asks whether each TNF-family member's interaction with TGFb looks like the
`TGFb:TNF` interaction, gene by gene.

## Pipeline

- **`01_deg_analysis_SIG29.Rmd`**: DESeq2 on the UMI-deduplicated counts.
  - Keeps genes with at least 10 counts in at least 4 samples. Fits
    `~ condition + replicate` with `none_none` as the reference. IHW is used for p-value
    adjustment.
  - QC: VST sample-distance heatmaps (all samples and per replicate) and PCA on the top
    2,000 genes.
  - Condition effects: every condition vs. `none_none`.
  - Interaction scoring: for each `TGFb_<member>` combination, tests the interaction
    contrast, then classifies genes as `synergy positive`, `synergy negative`, `buffering`
    or `none` at `padj_interaction <= 0.1` (see `../README.md`).
- **`02_deg_visualization_SIG29.Rmd`**: FC-FC plots of each gene's `TGFb:<member>`
  interaction LFC against its `TGFb:TNF` interaction LFC, for `TL1A`, `OX40L` and `DTA1`.
  - Pearson r per member, computed over all genes.
  - Concordance at `padj < 0.1`: `concordant (both sig)` (same sign),
    `discordant (both sig)`, `sig in one` or `n.s. in both`.
  - Stacked bars showing what fraction of `TGFb:TNF`-significant genes are concordant in
    each member.

## Inputs

From `imports_stable/SIG29/processing_outs/` (produced outside this repo):
`count_matrix_umiDeDup_SIG29.csv`, `processed_metadata_SIG29.csv` (`condition`, `ligand1`,
`ligand2`, `replicate`) and `featureNames_SIG29.csv`.

## Outputs

Written under `analysis_outs/08_tnf_family_tgfb_interaction_RNAseq_SIG29/`, which is gitignored:

- `res_conditions_SIG29.csv`, `res_interaction_scored_SIG29.csv` and
  `SIG29_PCA_plots.pdf` (`01`). `02` reads the interaction table, and
  `../../07_crispr_arrayed_validation_RNAseq_SIG30/03_program_crossreg` reads both tables,
  from their stable copies in `imports_stable/SIG29/analysis_outs/`.
- `res_interaction_fcfc_SIG29.csv` and `res_interaction_fcfc{,_simple}_SIG29.pdf` (`02`).

## Running

Knit `01` (DESeq2, IHW), then `02`, in the `R-signalseq` env. Both source `functions/r_custom/plotting_fxns.R` from this
repo.
