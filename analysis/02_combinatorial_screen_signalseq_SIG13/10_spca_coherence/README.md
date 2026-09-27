# sPCA Program Coherence: One Program or Two?

sPCA was fit on **pseudobulk** profiles. If, in a combinatorial condition `L1_L2`, ligand L1
induces genes A,B in half the cells and L2 induces genes C,D in the other half, the pseudobulk
sees A,B,C,D all elevated and bundles them into one component, even though no single cell
expresses the whole program. This folder asks whether that happened: is the gene set behind each
sPCA component carried together by the same cells, or is it two sub-programs living in different
cells that pseudobulk averaging glued together?

For each gated (interaction, component) pair, the component's genes are split into two halves,
each half is scored per cell with the original sPCA loadings, and the two scores are
Spearman-correlated within that condition's own cells:

```text
r = spearman(score_A, score_B)      # high = one program, low/negative = two
```

Control cells are never pooled in, since they would be jointly low in both halves and inflate `r`.
`r` is reported raw, with no thresholds or categorical calls.

## Pipeline

- **`01_program_coherence_simple.ipynb`**: runs the whole analysis.
  - **Gating.** Keeps interactions that pass the repo-canonical replicate-consistency filter
    (`pearson_corr > 0.25 & num_shared_genes > 1`) and whose **condition-level** lm coefficient
    has `p_adj < 0.01` and `|estimate| > 0.5`. Self-pairs are dropped. Gating on the overall
    condition effect rather than the interaction term keeps every interaction class in the set, so
    the by-class comparison is not circular. This yields 5,882 pairs (138 interactions x 68
    components).
  - **Scaling.** Applies `sc.pp.scale` to the `log1p_norm` expression over all cells, matching
    the upstream per-cell scoring. Cells are scored with the existing (not refit) sPCA loadings.
  - **Gene partition.** For each component, splits its genes with k-means (k = 2) on the gene x
    gene correlation across the 1,320 condition x replicate pseudobulk profiles, the level sPCA
    was fit on. The split is fixed before any cell is scored. A split found in the scored cells
    would always find a low-correlation cut, even in pure noise.
  - **Sweep.** Computes `r` for every gated pair. Interactions with fewer than 100 cells are
    skipped (only `IL6_IFNB1`).
  - **Summaries.** Mirror concordance (`L1_L2` vs `L2_L1`, which are independent wells), `r` vs
    effect size and cell count, `r` by interaction class (Mann-Whitney vs `none`), and per-cell
    scatterplots of the lowest- and highest-`r` synergy pairs.

## Inputs

All from `imports_stable/SIG13/`:
- `scanpy_outs/SIG13_doublets_DSB7.h5ad`: per-cell expression for scoring
- `scanpy_outs/SIG13_doublets_DSB7_zscore_degs0.1cutoff.h5ad`: z-scored DEG input, pseudobulked
  for the gene partition
- `analysis_outs/spca/zscore_degs_allLigands_0.1_alpha1.0_sPCA_loadings.csv`: sPCA loadings
- `analysis_outs/spca/lm_fit_condition_zscore_degs_allLigands_0.1_alpha1.0_sPCA_clean.csv`:
  condition-level lm fit, used for gating
- `analysis_outs/spca/lm_scored_zscore_degs_allLigands_0.1_alpha1.0_sPCA_clean.csv`: interaction
  class and score annotation
- `analysis_outs/replicate_corr/replicate_correlation_interactionLfc_0.2filter.csv`: consistency
  filter

## Outputs

All outputs are written under
`analysis_outs/02_combinatorial_screen_signalseq_SIG13/spca_coherence_simple/`, which is gitignored.

- `program_coherence.csv`: one row per gated pair with `real_r`, interaction class/score and
  `evaluable`.
- `pseudobulk_index.csv`, `pseudobulk_partitions.csv`, `partition_genes.csv`: pseudobulk row
  labels, per-component partition summary (module sizes, seed-to-seed ARI) and per-gene module
  assignment.
- `mirror_concordance.csv`: mirrored (`L1_L2`, `L2_L1`) pairs.
- `coherence_by_interaction_class.csv`: per-class n, median `r` and Mann-Whitney/CLES vs `none`.
- `figures/`: `coherence_vs_power`, `coherence_by_interaction_class`,
  `coherence_vs_interaction_strength_synergy`, `synergy_{lowest,highest}_r_in_cells` (PDFs, made
  with `cnsplots`).

## Running

Run `01_program_coherence_simple.ipynb` interactively in the `scanpy_standard2` kernel
(`environments/scanpy_standard2.yaml`). It is CPU-only and needs no SLURM. The
whole notebook takes a few minutes; the sweep itself takes about 1 minute on 8 threads.

Helvetica and Arial are not installed on this system, so the notebook sets the cnsplots font to
Nimbus Sans (the URW Helvetica clone) and turns off `axes.unicode_minus` to keep tick labels in
the same font.
