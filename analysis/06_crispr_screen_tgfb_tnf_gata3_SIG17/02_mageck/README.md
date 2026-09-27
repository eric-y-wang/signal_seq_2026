# CRISPR Screen (TGFb + TNF, GATA3): Clone Counts and MAGeCK Hits

SIG17 is a pooled CRISPR knockout screen in murine CD4 T cells. This analysis uses cells cultured
with TGFb + TNF and then sorted into 4 bins by GATA3-reporter expression (bin 1 = lowest, bin 4 =
highest; samples `SIG17_1-4` / `TGFb_TNF_gata3_1-4`). The goal is to find genes whose knockout
shifts cells along the GATA3 axis, and to compare those hits with the TGFb + TNF interaction term
from the SIG18 bulk ligand-combination experiment.

Read counting and statistics are done in `../01_processing/`. Each sgRNA read carries a 20bp UMI
on R2. The pipeline collapses UMI duplicates per guide, so every surviving (guide, UMI-cluster)
pair counts as **one transduced cell**, and then runs `mageck test` for each higher bin against
bin 1:

```text
mageck test -t TGFb_TNF_gata3_{2,3,4} -c TGFb_TNF_gata3_1 \
  --control-sgrna mageck_control_id.txt --sort-criteria pos --remove-zero both
```

A positive gene LFC means the gene's sgRNAs are **enriched** in high-GATA3 cells (knockout pushes
cells toward high GATA3; gene is a negative regulator of GATA3 expression in this condition). A
negative LFC means they are **depleted** (knockout pushes cells toward low GATA3; gene is a
positive regulator of GATA3 expression in this condition). The library has 4 sgRNAs per targeting
gene and 47 sgRNAs in each of two control sets, `control_NT` and `control_cutting`.

## Pipeline

- **`01_umi_clone_analysis.Rmd`**: counts how many UMI clones (independent cells) support each
  gene and sgRNA in each bin.
  - **Saturation check.** Every bin has a duplication rate of about 99%, so each molecule was
    sequenced about 100 times. It also plots reads per clone against sgRNA abundance to confirm
    that even rare sgRNAs are well covered. Because sequencing is saturated, clone counts are
    treated as **absolute cell counts** and not as a subsample.
  - **Clones per bin.** Plots the mean ± 95% CI on two scales. Per sgRNA, the targeting genes are
    compared with the two control sets, and the controls show how many cells a non-perturbing guide
    puts in each bin. Per gene (the sum of 4 sgRNAs), the count sets the power of a gene-level test.
    Controls are never summed to gene level because they have 47 guides each.
  - **Distributions.** Violin/box and ECDF plots of clones per gene and per sgRNA for each bin,
    followed by a summary table.
- **`02_tgfb_tnf_gata3_mageckTest.Rmd`**: gene- and sgRNA-level hits.
  - **Direction/FDR call.** Uses `neg|lfc` as the gene LFC (it is identical to `pos|lfc`). The FDR
    comes from the test direction that matches the sign of the LFC. Genes with `fdr < 0.1` are
    called `enriched (high GATA3)` or `depleted (low GATA3)`. Bin 1 is added as the baseline
    (`lfc = 0`, no FDR).
  - **Volcano plot** for bin 4 vs bin 1.
  - **CRISPR vs SIG18.** Plots significant CRISPR hits (bin 4 vs bin 1) against the SIG18
    `TGFb-TNF` GLM interaction coefficient, coloured by the SIG18 interaction class (`synergy
    positive/negative`, `buffering`, `none`). When one gene symbol maps to several Ensembl IDs, the
    row with the lowest `padj_interaction` is kept.
  - **sgRNA density + tick plot.** Shows the density of all sgRNA LFCs for bin 4 vs bin 1 above a
    row per gene, where each sgRNA is drawn as a tick (red = depleted, blue = enriched). This is
    drawn first for the top 30 genes by |LFC| and then for a hand-picked set of positive and
    negative regulators (`plot_genes`, e.g. `Gata3`, `Tgfbr2`, `Rel`, `Il2ra`, `Stat5a/b`).
  - **Dose plot.** Plots gene LFC across bins 1-4 for the curated set and combines it with the tick
    plot to make the final figure (with-legend and no-legend versions).

## Inputs

All inputs are written by `../01_processing/`. The Rmds read their stable copies from
`imports_stable/SIG17/dedup_pipeline_output/` (set as `pipe_dir`); re-running
`../01_processing/` writes fresh outputs to `analysis_outs/` instead.

- `02_guide_dedup/SIG17_{1..4}.umi_per_guide.tsv`: per-sgRNA UMI clone and read counts (`01`)
- `02_guide_dedup/SIG17_{1..4}.dedup_stats.json`: per-sample dedup summary and clone-size
  histogram (`01`)
- `../01_processing/mageck_library.csv`: the sgRNA library, used to count missing guides (`01`)
- `05_mageck_test_dedup/TGFb_TNF_gata3_{2,3,4}_v_1.gene_summary.txt`: gene-level `mageck test`
  results (`02`)
- `05_mageck_test_dedup/TGFb_TNF_gata3_4_v_1.sgrna_summary.txt`: sgRNA-level LFCs for bin 4 vs
  bin 1 (`02`)

`02` also reads the SIG18 interaction scores written by
`../../05_in_vitro_differentiation_RNAseq_SIG18/01_deg/`, from their stable copy
`imports_stable/SIG18/analysis_outs/deg/res_interaction_scored_SIG18.csv`.

## Outputs

All outputs are PDFs written to
`analysis_outs/06_crispr_screen_tgfb_tnf_gata3_SIG17/crispr_screen_tgfb_tnf_gata3/` at the repo
root, which is gitignored.

- `01`: `clone_mean_per_bin.pdf`, `clone_distribution_per_bin.pdf`,
  `clone_sequencing_saturation.pdf`
- `02`: `gata3_4_vs_1_sgRNA_volcano.pdf`,
  `gata3_4_vs_1_lfc_vs_SIG18_TGFb_TNF_interaction.pdf`,
  `gata3_4_vs_1_sgRNA_lfc_density_dose{,_noLegend}_test.pdf`

## Running

Run `../01_processing/` first (`bash ../01_processing/submit_all.sh`). The two Rmds do not depend
on each other and can be knit in either order. Both find the repo root by searching upward for
`imports_stable/` (for `functions/r_custom/plotting_fxns.R`, the inputs and the output folder)
and read the library file as `../01_processing/mageck_library.csv`, so knit them from here. They need `tidyverse`, `patchwork`, `ggrepel`, `RColorBrewer`, `scales` and `jsonlite`,
which are all in the `R-signalseq` env.
