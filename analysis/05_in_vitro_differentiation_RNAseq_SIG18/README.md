# In Vitro Differentiation Bulk RNA-seq (SIG18)

Bulk RNA-seq of mouse CD4 T cells differentiated in vitro under Th-polarizing conditions,
with or without TNF. Samples come from individual mice (`replicate`). Mouse 4 and mouse 9
are dropped in `01` and in the sPCA fit because they sat on the plate edge and clustered
anomalously.

| factor | baseline | levels |
|---|---|---|
| `ligand1` (polarization) | `Th0` | `Th1`, `Th2`, `Th17`, `iTreg`, `TGFb` |
| `ligand2` | `none` | `TNF` |

The two factors form a 6 x 2 design, so each polarization + TNF combination can be tested
for synergy against the sum of its single effects, relative to `Th0_none`:

```text
lfc_interaction = lfc(X-TNF) - lfc(X) - lfc(TNF)     # X = Th1, Th2, Th17, iTreg or TGFb
```

## Pipeline

### 01_deg

- **`deg_analysis_SIG18.Rmd`**: DESeq2 on UMI-deduplicated counts. Genes need >= 10
  counts in >= 3 samples.
  - One condition-level model, `~ condition + replicate`, with `condition` built from the
    two ligand columns (`none`, `TNF`, `Th1`, `Th1-TNF`, ...). Every condition is compared
    with `none` (IHW-adjusted p-values, no independent filtering).
  - QC: VST sample-distance heatmaps (all samples and per replicate) and PCA.
  - Interaction scoring: `padj_interaction` comes from the contrast
    `X-TNF - (X + TNF)`. At `padj_interaction <= 0.1` a gene is `synergy positive` or
    `synergy negative` when the interaction has the same sign as the combined effect,
    and `buffering` when the signs are opposite. Everything else is `none`.

### 02_spca

This follows the SIG13 sPCA pipeline
(`../02_combinatorial_screen_signalseq_SIG13/07_spca/README.md`), applied to per-sample
bulk profiles instead of single-cell pseudobulk.

- **`01_spca_degs_zscore_expression.py`** / **`01_run_spca_alphas.sh`**: fits
  non-negative sparse PCA on the z-score matrix, with no further scaling. The matrix is
  subset to the `01_deg` DEGs: single-ligand conditions (`res_conditions`, rows with no
  `-` in `condition`) at `padj < 0.01`, plus interaction contrasts
  (`res_interaction_scored`) at `padj_interaction < 0.01`. Uses 100 bootstraps at `n_components=100`, then UMAP (Hellinger) +
  HDBSCAN (80% coherence) to set the program number, and one final fit
  (`random_state=100`). The shell script submits one SLURM job per alpha. The manuscript
  uses `alpha=10.0`, which gave 58 programs over 6,239 genes. That fit used an earlier DEG
  list; the current `01_deg` list (6,431 genes, 97% overlap) gives slightly different
  numbers.
- **`02_spca_waggr_scoring.ipynb`**: scores programs per sample with `decoupler`'s `waggr`
  (`tmin=5`, no permutations) on normalized, `log1p`, scaled counts. The scoring network
  is built directly from `sPCA_components.csv` (nonzero loadings, programs named
  `comp_{i}`), with the loadings as weights.
- **`03_spca_annotation.Rmd`**: fits
  `score ~ ligand1 * ligand2 + replicate + total_reads_counts + seq_saturation` per
  program on the scaled waggr scores (HC3 robust SEs, global BH; Huber `rlm` as a check).
  Drops programs whose most significant term is `replicate` (one, `comp_57`). Calls
  interactions with the same rule and `p_adj_interaction <= 0.1` threshold as `01_deg`,
  and adds the fraction of each program's genes in each gene-level class (`freq_*`),
  taken from `01_deg`'s `res_interaction_scored_SIG18.csv`.
- **`04_spca_pos_synergy_viz.Rmd`**: program x condition heatmaps (from the sPCA codes),
  interaction counts per combination, heatmaps of positive-synergy programs (all, and
  TGFb + TNF), and a TGFb + TNF coefficient scatter. A program counts as synergistic when
  its call is `synergy positive` and `freq_synergy positive > 0.25`. Also exports the
  clustered program order.
- **`05_spca_subset_gene_viz.Rmd`**: heatmaps of the top 50 genes (by loading) for
  selected programs, on normalized counts.
- **`06_spca_ora_analysis.Rmd`**: KEGG and MSigDB Hallmark over-representation for each
  clean program (`clusterProfiler::compareCluster`, genes mapped mouse to human with
  biomaRt), in the program order from `04`.

## Inputs

All inputs are read from `imports_stable/SIG18/`.

From `imports_stable/SIG18/processing_outs/`:

- `count_matrix_umiDeDup_SIG18.csv`, `featureNames_SIG18.csv`,
  `processed_metadata_SIG18.csv`: counts, gene names and sample metadata (01, 02, 03).
- `zscore_matrix_umiDeDup_SIG18.csv`: z-scored expression for the sPCA fit.
- `norm_counts_matrix_umiDeDup_SIG18.csv`: normalized counts (05).

Outputs of earlier steps are read from their stable copies in
`imports_stable/SIG18/analysis_outs/`, so each step can run on its own. Re-running an earlier
step writes fresh output to `analysis_outs/` instead. For example:

- `deg/res_conditions_SIG18.csv` and `deg/res_interaction_scored_SIG18.csv` (from 01):
  DEG list for the sPCA fit and gene-level classes for `03`.
- `spca/zscore_degs/*` (from 01/02 of `02_spca`) and `spca/*` (from 03/04), read by the
  later sPCA steps.

## Outputs

Written under `analysis_outs/05_in_vitro_differentiation_RNAseq_SIG18/`, which is
gitignored:

- `deg/res_conditions_SIG18.csv` and `res_interaction_scored_SIG18.csv` (01).
- `spca/zscore_degs/`: `zscore_degs_unscaled_alpha{alpha}_sPCA_{components,codes}.csv`
  and `zscore_degs_alpha10.0_waggr_score.csv`.
- `spca/`: `zscore_degs_alpha10.0_sPCA_loadings.csv` (long-format loadings with
  gene-level classes), `lm_fit_*_sPCA{,_clean}.csv`, `lm_scored_*_sPCA_clean.csv`,
  `spca_alpha10.0_full_hm_row_order.csv`, and `*_sPCA_{kegg,hallmark}.csv`.
- `plots/`: `SIG18_PCA.pdf`, sPCA heatmaps, `SIG18_comp_*_hm*.pdf` and
  `SIG18_spca_go_enrichment.pdf`.

## Running

Knit `01_deg/deg_analysis_SIG18.Rmd` in the `R-signalseq` env. For `02_spca`, run
`bash 01_run_spca_alphas.sh` (SLURM, `scanpy_standard` env, 24 cores, 64G), then
`02_spca_waggr_scoring.ipynb` in `scanpy_standard`, then knit `03` to `06` in order in
`R-signalseq`. Each script finds the repo root by
searching upward for `imports_stable/`, so run them (and submit `sbatch`) from inside the
repo.
