# SIG13 Combinatorial Signal-seq Screen

Processing and analysis of the SIG13 screen: 648 single and pairwise ligand conditions
(the 12 round-1 × 55 round-2 barcode grid gives 660 called conditions; the 12 homotypic
`X_X` double-dose conditions, including `linker_linker`, are not counted) in CD4 T cells,
read out by single-cell RNA-seq with ligand barcodes (Signal-seq).

Each cell receives two rounds of barcoded ligand VLPs. Round-1 ligands (Set A) are
always paired with round-2 ligands (Set B), and a cell is kept only if it carries
exactly one round-1 and one round-2 barcode, called at a DSB-normalized UMI threshold
of 7 (DSB7). Conditions are named `{round1 ligand}_{round2 ligand}`. `linker` is a
non-targeting VLP (no ligand), so `IL4_linker` is IL4 alone, `IL4_TNF` is the pair and
`linker_linker` is the control. The screen was run as two technical replicates.

The central question is whether a ligand pair does something different from the sum of
its parts. For each pair, the four groups `L1_linker`, `linker_L2`, `L1_L2` and
`linker_linker` are fit with

```text
counts ~ ligand1 * ligand2 + covariates
```

and the `ligand1:ligand2` term is the non-additive interaction. Interactions are called
per gene (Gamma-Poisson GLMs, `glmGamPoi`) and per gene program (non-negative sparse PCA
programs scored with `decoupler` `waggr`), and classified as `synergy positive`,
`synergy negative` or `buffering`. The sPCA programs from this screen are also the basis
of the ligand-activity model in `../03_activity_inference_model`.

## Folders

| folder | what it does |
|---|---|
| `01_processing` | cellranger outputs to a QC-filtered, DSB7-called AnnData (RNA QC, DSB barcode normalization, round-aware doublet calling, normalization, cell-cycle scores), then control-z-scored DEG-subset expression for sPCA |
| `02_qc_general` | cell and condition counts, RNA quality, replicate and A_B vs B_A orientation correlations of GLM LFCs (the reproducibility filters used downstream), GLM concordance with the SIG07 screen, and LFC concordance with recombinant-protein stimulations (SIG14 bulk RNA-seq) |
| `03_qc_barcode_cutoff` | re-calls cells at DSB1-DSB10 and checks whether GLM coefficients and sPCA scores change relative to DSB7 |
| `04_qc_barcode_counts` | tests whether barcode expression level is a proxy for VLP dose (it is not; it is confounded with RNA depth) |
| `05_interaction_scoring` | single-term and interaction `glmGamPoi` GLMs for every condition (pooled and per-replicate), and per-gene interaction classes |
| `06_interaction_scoring_null` | empirical null for the gene-level GLMs from linker-only cells, and a test of synergy vs. buffering by expected effect size |
| `07_spca` | fits the sPCA gene programs (78, 68 after removing replicate-driven ones), scores them, and calls program-level interactions with a linear model |
| `08_spca_null` | empirical null for the program-level linear model from linker-only cells |
| `09_spca_stability` | program stability across sPCA alpha values and cell-level bootstraps |
| `10_spca_coherence` | tests whether each program is carried by the same cells or is two sub-programs merged by pseudobulking |
| `11_single_ligand` | Hallmark GSEA validation of single-ligand responses and DEG counts per ligand |
| `12_gene_level_analysis` | gene-level interaction class frequencies, example genes, dose saturation, circos plots and TGFB1 x TNF-family specificity |
| `13_clustering` | HDBSCAN clustering of conditions on interaction and single-ligand GLM coefficients, with bootstrap stability |
| `14_signal_trans_reinforcement` | tests whether ligand pairs that induce each other's receptors show more transcriptional synergy |

## Run order

```text
01 processing (01) ──> 05 interaction scoring ──┬──> 01 processing (02) ──> 07 sPCA ──┬──> 08 sPCA null
                                                │                                     ├──> 09 sPCA stability
                                                │                                     └──> 10 sPCA coherence
                                                ├──> 02 QC general
                                                ├──> 06 interaction null
                                                ├──> 11 single ligand
                                                ├──> 12 gene level
                                                ├──> 13 clustering
                                                └──> 14 reinforcement

03, 04 barcode QC: after 05 and 07
```

`01_processing/01` writes the DSB7 cell set that every other folder reads. The GLMs in
05 come next. `01_processing/02` needs the single-term DEGs from 05 to build the
z-scored input for sPCA (07), and 08-10 build on the 07 programs. `02_qc_general`
writes the replicate-correlation tables (`replicate_corr/`) that 03, 04,
07, 10, 12, 13 and 14 use as reproducibility filters. 03 and 04 rerun the 05 GLM and
rescore the 07 programs on alternative cell sets, so they need both.

Each folder's README has its own steps, inputs and outputs. 01 and 02 have no README;
see the notebooks.

## Environments

| folder | Python | R |
|---|---|---|
| 01 | `scanpy_standard2` (01), `scanpy_standard` (02) | |
| 02 | `scanpy_standard2` | `R-signalseq` |
| 03 | `scanpy_standard2`, `rapids_singlecell` (GPU, 02) | `R-deseq2` (GLM), `R-signalseq` (reports) |
| 04 | `scanpy_standard2` | `R-deseq2` (GLM), `R-signalseq` (reports) |
| 05 | | `R-deseq2` |
| 06 | | `R-deseq2` (GLM), `R-signalseq` (reports) |
| 07 | `scanpy_standard` | `R-signalseq`, `R-deseq2` (04, via `reticulate`) |
| 08 | `scanpy_standard2` | `R-signalseq` |
| 09 | `scanpy_standard` (SLURM fits), `scanpy_standard2` (notebooks) | |
| 10 | `scanpy_standard2` | |
| 11 | `scanpy_standard2` | `R-signalseq` |
| 12 | | `R-signalseq` |
| 13 | `scanpy_standard2` | |
| 14 | `scanpy_standard2` | `R-signalseq` |

The GLM scripts run in `R-deseq2` because `reticulate`/`anndata` are needed to read the
`.h5ad`. Environment files are in `environments/` at the repo root.

## Inputs and outputs

Scripts find the repo root by searching upward for `imports_stable/`, so run notebooks and
Rmds, and submit `sbatch` jobs, from inside the repo.

Every input is read from `imports_stable/`:
- Cell Ranger files: `imports_stable/SIG13/cellranger/`
- processed `.h5ad`/`.h5mu`: `imports_stable/SIG13/scanpy_outs/`
- earlier-step results: `imports_stable/SIG13/analysis_outs/`. The glmGamPoi results are kept
  flat in `imports_stable/SIG13/analysis_outs/glmGamPoi/`, not in the per-model subfolders the GLM
  scripts write (see `05_interaction_scoring/README.md`).

This includes files that an earlier folder in this pipeline produces. Those were copied into
`imports_stable/` so each step runs on its own; re-running an earlier step writes a fresh copy
to `analysis_outs/` and leaves `imports_stable/` unchanged. Exceptions:
- `SIG13_doublets_DSB7.h5ad` (the input to most steps) and `SIG13_full_bc_processed.h5mu` are
  not in the Zenodo deposit. Download them from GEO
  ([GSE318270](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE318270)) into
  `imports_stable/SIG13/scanpy_outs/`.
- The cutoff-sweep h5ads and GLM outputs of `03_qc_barcode_cutoff`, the GLM outputs of
  `04_qc_barcode_counts`, the glmGamPoi outputs at
  filter cutoffs other than 0.05/0.1/0.2 (used only by the `13_clustering` cutoff sweep), and the
  `09_spca_stability` fits are not included; see those folders' READMEs.

All outputs are written under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/` (not
tracked). Each analysis has its own subfolder there (for example `glmGamPoi/`, `spca/`,
`replicate_corr/`, `clustering/`). Figures go to `plots/`, and processed `.h5ad`/`.h5mu` files
go to `scanpy_outs/`. The folder READMEs give output paths relative to this folder. Shared
code is loaded from the repo's `functions/`.
