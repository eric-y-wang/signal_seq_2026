# Single-Ligand Conditions: Signature Validation and DEG Counts

This folder looks at the single-ligand conditions of the screen on their own, before
any combinatorial analysis. It checks that each ligand induces the expected
signaling response, and it summarizes how many genes each ligand changes.

A single-ligand condition is a ligand paired with `linker` (no ligand) in the other
round. Each ligand appears twice: `X_linker` (ligand in round 1, called **Set A**) and
`linker_X` (ligand in round 2, called **Set B**). Both notebooks start from the
single-term GLM results in `../05_interaction_scoring`
(`glmGamPoi_singleTerm_lfc_0.1filter.csv`).

## Pipeline

- **`01_single_ligand_signature_validation.rmd`**: checks that single-ligand responses
  match known signaling pathways, using GSEA on the MSigDB Hallmark gene sets (mouse,
  `msigdbr`).
  - **GSEA.** Runs `fgsea` separately for each single-ligand condition, on all genes
    ranked by LFC. Significant pathways are those with `padj < 0.1`.
  - **Set A vs. Set B consistency.** For each ligand, the Jaccard index between the
    significant pathways of `X_linker` and `linker_X`. The two sets largely agree.
  - **Dotplots.** NES and significance for the Set A conditions, first for all
    Hallmark pathways, then for the pathways that match the screen's ligands
    (interferon, IL2, TNF, IL6/STAT, inflammatory, MYC, TGF-beta).
  - **Enrichment plots.** Running enrichment score for six ligand x pathway pairs:
    IFNG and IL27 (interferon gamma response), IL6 (IL6/JAK/STAT3), IL2 (IL2/STAT5),
    TNF (TNFA via NF-kB) and TGFB1 (TGF-beta signaling). Written by Ian Zumpano.
- **`02_single_ligand_analysis.ipynb`**: counts DEGs (`adj_pval < 0.1`) per
  single-ligand condition, labelled as Set A / Set B, and plots them as a log-scale
  bar chart. It also lists the ligands whose mean DEG count across the two sets is
  above 50.

## Inputs

All paths are under `imports_stable/SIG13/analysis_outs/`.

- `glmGamPoi/glmGamPoi_singleTerm_lfc_0.1filter.csv`:
  single-term condition LFCs and adjusted p-values (`01`, `02`).
- `glmGamPoi/glmGamPoi_singleTerm_coefficients_0.1filter.csv`: loaded by
  `02` but not used in the current plots.

`01` reads cached intermediate tables (`ligand_signature_validation_linkers.csv`,
`hallmark_gsea_results.csv`, `significant_hallmark_gsea_results.csv`) from
`imports_stable/SIG13/analysis_outs_zumpano/`. The code that builds them is left in the
notebook, commented out. Uncomment it to regenerate them from the single-term table; they are
then written to `analysis_outs/02_combinatorial_screen_signalseq_SIG13/`.

## Outputs

All outputs are written under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/`, which is
gitignored.

- `plots/gsea_res.pdf`: the six enrichment plots (`01`). The dotplots appear only in
  the knitted HTML.
- `plots/linkerOnly_nDEGs_barplot.pdf`: DEG counts per single-ligand condition (`02`).

## Running

Knit `01` from this folder (it finds the repo root and sources
`functions/r_custom/plotting_fxns.R` from there). Knit it in the `R-signalseq` env
(which has `fgsea`, `msigdbr` and `org.Mm.eg.db`):

```bash
Rscript -e "rmarkdown::render('01_single_ligand_signature_validation.rmd')"
```

Run `02` in any Jupyter kernel with `pandas`, `matplotlib` and `seaborn`.
