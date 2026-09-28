# sPCA Gene Programs: Discovery, Annotation and Interaction Calls

Gene-level GLM testing (`05_interaction_scoring/`) calls synergy and buffering one gene at a
time. This folder groups genes into **gene programs** with non-negative sparse PCA (sPCA) and asks
the same questions at the program level: which ligands drive each program, and in which ligand
pairs the combination deviates from the sum of the single-ligand effects.

sPCA is fit on **pseudobulk** profiles (the mean per ligand pair x replicate) of DEG-subset
expression that has been **z-scored to control cells**. This is not an ordinary per-gene z-score
over all cells. For each gene and each `replicate` x `lane` group, the mean and SD come only from
that group's `linker_linker` cells, and every cell in the group is transformed with them:

```text
z = (x - mean_ctrl) / sd_ctrl        # x = normalized (non-log) counts; ctrl = linker_linker cells in the same replicate_lane
```

So values are in units of control-cell SD away from the control mean. The loadings are then used to score cells, and each program's pseudobulk
score is regressed on the ligand design:

```text
score ~ ligand1 * ligand2 + replicate      # linker is the reference level for both slots
```

The final fit (`alpha=1.0`) has 78 programs over 8,057 genes. Ten are removed as
replicate-driven, which leaves 68 "clean" programs. Those 68 are what the downstream
`08_spca_null`, `09_spca_stability` and `10_spca_coherence` folders test.

## Pipeline

- **`01_spca_degs_zscore_expression_allLigands.py`** / **`01_run_spca_alphas.sh`**: fits the
  programs. The shell script submits one SLURM job per alpha (currently 1.0, 1.5, 2.0).
  - **Input.** Cells from `SIG13_doublets_DSB7_zscore_degs0.1cutoff.h5ad`, averaged by
    `ligand_call_DSB7` x `replicate`. The data are already control-normalized (see above), so no
    further scaling is applied.
  - **Consensus program number.** Runs 100 bootstraps (resampling pseudobulk rows) of
    `NonNegativeSparsePCA` at `n_components=100`. This subclass of sklearn's `SparsePCA` calls
    `dict_learning` with `positive_code=True`. The pooled atoms are embedded with UMAP (Hellinger
    metric) and clustered with HDBSCAN. `min_cluster_size` is 80% of the bootstraps, so only atoms
    that recur in most bootstraps form a cluster. The number of clusters sets `n_components`.
  - **Final fit.** Refits once on the full pseudobulk at that `n_components` (`random_state=100`)
    and writes the components, codes and model.
- **`02_spca_waggr_scoring.ipynb`**: scores programs with `decoupler`'s `waggr` (a weighted mean,
  `tmin=5`, no permutations). It builds the scoring network directly from `sPCA_components.csv` (nonzero loadings,
  programs named `comp_{i}`) and uses those loadings as weights on `log1p_norm` expression
  scaled over all cells (`sc.pp.scale`), then pseudobulked by ligand pair x replicate. These
  scores replace the raw sPCA codes in every step that follows.
- **`03_spca_annotation.Rmd`**: the statistical core.
  - **LM.** Fits `score ~ ligand1 * ligand2 + replicate` per program on the scaled waggr scores
    with HC3 robust SEs and global BH adjustment. Huber `rlm` estimates are compared with OLS as a
    sensitivity check and agree almost perfectly. A second model, `score ~ interaction +
    replicate`, gives condition-level effects (these are used for gating in `10_spca_coherence`).
  - **Replicate filter.** Drops any program whose most significant term is `replicate`.
  - **Interaction calls.** At `p_adj_interaction <= 0.1`, the call is `synergy positive` or
    `synergy negative` when the interaction term has the same sign as the total effect
    (`ligand1 + ligand2 + interaction`), and `buffering` when the signs are opposite. Everything
    else is `none`.
  - **Gene-level merge.** For each program x interaction, records the fraction of the program's
    genes that carry each gene-level GLM interaction class (`freq_*` columns).
- **`04_spca_visualization.Rmd`**: figures for the clean programs:
  - program x interaction heatmaps (all interactions, and only reproducible ones that pass
    `pearson_corr > 0.25 & num_shared_genes > 1`)
  - single-ligand heatmaps with interaction-class bar plots
  - single vs. combinatorial top-3 score comparison
  - TNF-family synergy heatmaps
  - per-program top-gene expression heatmaps (`spca_gene_hm/`)

  Also exports the clustered program order.
- **`05_spca_ora_analysis.Rmd`**: runs over-representation analysis for each clean program's genes
  with `clusterProfiler::compareCluster`. Genes are converted mouse→human with `babelgene`, then
  tested against KEGG and MSigDB Hallmark. Programs are plotted in the clustered order from `04`.

## Inputs

- `imports_stable/SIG13/scanpy_outs/SIG13_doublets_DSB7_zscore_degs0.1cutoff.h5ad` (and
  `_pb.h5ad`): control-normalized DEG-subset expression built by
  `01_processing/02_zscore_deg_processing.ipynb`. The genes are single-term GLM DEGs at
  `adj_pval < 0.01`. The normalization is
  `perturbseq.normalize_to_control_adata` (`functions/perturbseq/expression_normalization.py`),
  run with `control_cells_query='ligand_call_DSB7 == "linker_linker"'` and
  `groupby_column='replicate_lane'`.
- `imports_stable/SIG13/scanpy_outs/SIG13_doublets_DSB7.h5ad`: `log1p_norm` expression for
  waggr scoring.
- `imports_stable/SIG13/analysis_outs/glmGamPoi/interactions_scored_v3_glmGamPoi_0.05filter_sig.csv`:
  gene-level interaction classes, used for the `freq_*` columns.
- `imports_stable/SIG13/analysis_outs/replicate_corr/replicate_correlation_interactionLfc_0.2filter.csv`:
  the reproducible-interaction filter used in `04`.
- `imports_stable/SIG13/analysis_outs/spca/`: stable copies of this folder's earlier-step outputs
  (`01` components, `02` waggr scores, `03` fits and loadings, `04` row order), which the later
  steps read. Re-running a step writes fresh files to `analysis_outs/` only.

## Outputs

All outputs are written under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/`, which is
gitignored.

- `spca/degs_zscore_allLigands/zscore_degs_allLigands_0.1_alpha{alpha}_`:
  - `sPCA_components.csv`: gene x program loadings
  - `sPCA_codes.csv`
  - `sparse_pca_model.joblib`
  - `waggr_score.csv`: ligand pair x replicate program scores
- `spca/zscore_degs_allLigands_0.1_alpha1.0_`:
  - `sPCA_loadings.csv`: long format, nonzero loadings only. This is the scoring network used by
    downstream folders (`02` builds the same network itself from `sPCA_components.csv`).
  - `sPCA_component_num.csv`: genes per program
  - `sPCA_single_comb_top3.csv`
- `spca/lm_fit_zscore_degs_allLigands_0.1_alpha1.0_sPCA{,_clean}.csv`: interaction-model
  coefficients for all programs and for clean programs.
- `spca/lm_fit_condition_zscore_degs_allLigands_0.1_alpha1.0_sPCA{,_clean}.csv`: condition-model
  coefficients.
- `spca/lm_scored_zscore_degs_allLigands_0.1_alpha1.0_sPCA_clean.csv`: per program x interaction
  calls with the gene-level `freq_*` columns. The clean program list is taken from this file.
- `spca/spca_alpha1.0_full_hm{,_reproducible}_row_order.csv`: clustered program order.
- `spca/zscore_degs_alpha1.0_sPCA_{kegg,hallmark}.csv`: ORA results.
- `plots/`: `spca_*.pdf` heatmaps and diagnostics, `spca_gene_hm/`, and
  `go_enrichment_zscore_degs_alpha1.0_sPCA.pdf`.

## Running

```bash
cd analysis/02_combinatorial_screen_signalseq_SIG13/07_spca
bash 01_run_spca_alphas.sh      # one sbatch job per alpha, 16 cores, 80G, ~1-2 h each
```

Then run `02_spca_waggr_scoring.ipynb` in the `scanpy_standard` kernel. After that, knit `03`,
`04` and `05` in order (or render them from the terminal per the repo convention). `04` loads the
`.h5ad` through `reticulate` in the `R-deseq2` conda env (`conda = "auto"`).
