# Gene-Level Interaction Analysis: Class Overviews, Examples and Specificity

The GLM scoring in `05_interaction_scoring/` tests each gene in each ligand pair for a
non-additive interaction and assigns it a class. This folder summarizes and visualizes those
gene-level calls directly, without grouping genes into programs (that is done in `07_spca/`).
It asks how common each interaction class is, what representative synergies look like, how
saturated single-ligand signals are, and how specific the TGFB1 x TNF-family synergies are.

Every analysis starts from the `interactions_scored_v3_glmGamPoi_0.05filter` table. For each gene
it compares the observed condition LFC with the additive expectation:

```text
lfc_expected = lfc_ligand1 + lfc_ligand2      # observed = lfc_condition; lfc_interaction = observed - expected
```

## Shared filtering

`01`, `03` and `04` apply the same QC before any plotting:

- **Consistent conditions.** Keep conditions whose condition-level LFCs correlate between
  replicates at `pearson_corr > 0.5` (`replicate_correlation_singleLfc_0.2filter.csv`).
- **Reproducible interaction terms.** Drop interactions whose interaction-term LFCs have
  `pearson_corr <= 0.25` between replicates (`replicate_correlation_interactionLfc_0.2filter.csv`).
- **Round-2 TGFB1.** Drop `*_TGFB1` conditions (`01`, `04`). Recombinant TGFB1 was only added to
  round-1 wells, so round-2 TGFB1 is inactive.
- **Significant genes.** Keep genes with `adj_pval_condition < 0.1`, plus all `buffering` genes
  (`01`, `04`). A buffered gene can have a non-significant combined effect.
- **Conditions with effects.** Where a per-condition summary is plotted, keep conditions with more
  than 50 single-term DEGs at `adj_pval < 0.1`.

This consistency filter is stricter than the one used in `13_clustering/`, `10_spca_coherence/` and
`14_signal_trans_reinforcement/` (`pearson_corr > 0.25 & num_shared_genes > 1`).

## Pipeline

- **`01_interaction_deg_manuscript_viz.Rmd`**: manuscript figures for gene-level interactions.
  - **Most significant hit per gene.** For each class, takes each gene's most significant
    interaction (`adj_pval_interaction < 0.001`) and plots expected vs. observed LFC, colored by
    -log10(p_adj) capped at 50.
  - **Class frequency.** Share of each condition's DEGs in each class, shown as a stacked bar per
    condition and as per-class boxplots. Linker conditions and double-dose self-pairs (e.g.
    `IL2_IL2`) are excluded. It also reports min/median/max per class and the fraction of
    conditions with no interaction genes at all.
  - **Synergy vs. DEG count.** Per condition, the number of synergistic genes against the
    single-ligand DEG counts of each arm. Only conditions with more than 20 synergies and more
    than 50 DEGs are shown.
  - **Example genes.** Expected vs. observed LFC across conditions for `Gata3`, `Batf`, `Tagap` and
    `Ahnak` in cytokine x TNF-family pairs (TNFSF9/TNFSF14 dropped as non-performing), and for the
    Th-differentiation genes `Foxp3`, `Gata3` and `Bcl6`. Unsaved exploratory plots cover `Bcl6`,
    `Foxp3` and `Cxcr5`.
- **`02_dose_saturation_analysis.Rmd`**: uses the double-dose self-pairs (`X_X` vs. `X_linker`) to
  show how saturated single-ligand signals are. For each ligand it plots single-dose vs.
  double-dose condition LFC for genes significant (`p_adj < 0.05`) in either, colored by
  significance pattern and then by the double-dose interaction class. The strongest responses are
  mostly buffered (the double dose adds little), which is consistent with receptor saturation.
  Only the class-colored panel (8 ligands) is saved.
- **`03_interaction_vis_circos.Rmd`**: chord diagrams of interaction counts between ligands, one
  each for `buffering`, `synergy positive` and `synergy negative`. Counts genes with
  `adj_pval_interaction < 0.01` per pair after the consistency filter, excluding self-pairs.
  Swapped well-position orders (`A_B`, `B_A`) are averaged under one alphabetically sorted label.
  Sector colors are fixed per ligand across the three diagrams, and link color scales with the
  count. Written by Ian Zumpano. The figures appear only in the knitted HTML and are not saved to
  disk.
- **`04_tgfb_tnf_interaction.Rmd`**: asks how specific the TGFB1 x TNF-family interaction genes
  are.
  - **Groups.** `TGFb x TNF` (round-1 TGFB1 with TNF, TNFSF4, TNFSF15 or TNFSF18), `TGFb other`
    and `TNF other`. Linker conditions, self-pairs, TGFB3, MSTN, TNFSF9 and TNFSF14 are excluded.
    The four `TGFb x TNF` conditions all pass the >50 DEG bar.
  - **Uniqueness.** A gene is **unique** to a focal condition for a class if it carries that class
    there and in none of the `TGFb other` / `TNF other` conditions. Reports the percentage of
    unique genes per condition x class and for the `TGFb x TNF` set as a whole.
  - **Examples.** The top 4 unique `synergy positive` genes of `TGFB1_TNF` (by
    `adj_pval_interaction`), plotted across all TGFb/TNF conditions. Single-ligand arms are placed
    on the diagonal.

## Inputs

All paths are under `imports_stable/SIG13/analysis_outs/`.

- `glmGamPoi/interactions_scored_v3_glmGamPoi_0.05filter.csv`: gene x interaction scores and
  classes (`01`, `02`, `04`); `03` reads the `_sig` version.
- `glmGamPoi/glmGamPoi_single_term/glmGamPoi_singleTerm_lfc_0.05filter.csv`: single-term DEGs, for
  the >50 DEG condition filter.
- `replicate_corr/replicate_correlation_{singleLfc,interactionLfc}_0.2filter.csv`: consistency
  filters.

## Outputs

All outputs are written under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/`, which is
gitignored.

- `plots/` from `01`:
  - `most_sig_interaction_per_gene.pdf`
  - `class_frequency_per_interaction_barplots.pdf`, `class_frequency_boxplots.pdf`
  - `interactions_synergistic_vs_deg_{dotplot,scatterplot}.pdf`
  - `tnf_family_deg_examples.pdf`, `Th_diff_example_synergies_full_plots.pdf`
- `plots/qc/dose_saturation_plots.pdf`: from `02`.
- `plots/` from `04`: `tgfb_tnf_unique_class_percent_barplot.pdf`,
  `tgfb_tnf_family_unique_class_percent_barplot.pdf` and
  `tgfb_tnf_unique_synergy_gene_examples.pdf`.
- `tgfb_tnf/uniqueness_by_condition_class.csv`, `tgfb_tnf/unique_genes_tgfb_tnf.csv`: from `04`.

## Running

The notebooks are independent of each other and only need the upstream GLM and
replicate-correlation outputs, which they read from `imports_stable/`. Knit each `.Rmd` from its
own directory with the `R-signalseq` env active (each finds the repo root by searching upward
for `imports_stable/`):

```bash
cd analysis/02_combinatorial_screen_signalseq_SIG13/12_gene_level_analysis
RSTUDIO_PANDOC="$CONDA_PREFIX/bin" \
Rscript \
  -e "rmarkdown::render('01_interaction_deg_manuscript_viz.Rmd')"
```

`RSTUDIO_PANDOC` points R at the env's pandoc.
