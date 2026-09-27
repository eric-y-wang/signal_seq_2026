# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the analysis code repository for the 2026 Signal-seq manuscript. It contains Jupyter notebooks, Python/R scripts, Slurm submission scripts, and R Markdown documents that reproduce the figures/results of the paper, plus the custom function libraries they depend on. It is a research reproducibility repo, not a software package — there is no build system, package entry point, linter, or automated test suite. "Correctness" here means a notebook runs top-to-bottom in the right environment and reproduces the reported result, not passing CI.

## Repository layout

- `analysis/` — one module per manuscript analysis, named `NN_<description>_<experiment ID>` (experiment IDs like `SIG13` match the `imports_stable/` subfolders). Every module and most subfolders have a `README.md` describing design, models, inputs and outputs — read it before editing. Files/subfolders are numbered (`01_...`, `02_...`) to indicate execution order; later steps consume earlier outputs. Rmds are checked in alongside their rendered `.html`.
  - `01_barcode_comparison_signalseq_SIG02_SIG03/` — linear (SIG02) vs. circular (SIG03) VLP barcode capture comparison.
  - `02_combinatorial_screen_signalseq_SIG13/` — 648-condition combinatorial Signal-seq screen (the 12 × 55 barcode grid gives 660 called conditions; the 12 homotypic `X_X` double-dose conditions, including `linker_linker`, are not counted). `01_processing` (Cell Ranger import, z-scored DEGs) → `02_qc_general` / `03_qc_barcode_cutoff` / `04_qc_barcode_counts` (QC and barcode-calling robustness) → `05_interaction_scoring` (HPC glmGamPoi GLMs + interaction classes, see below) → `06_interaction_scoring_null` → `07_spca` (sparse PCA gene programs, `waggr` scoring, annotation, ORA) → `08_spca_null` / `09_spca_stability` / `10_spca_coherence` → `11_single_ligand`, `12_gene_level_analysis`, `13_clustering` (HDBSCAN bootstrap), `14_signal_trans_reinforcement`.
  - `03_activity_inference_model/` — ridge-regression ligand-activity model built on the SIG13 sPCA programs: `model_core/` (shared `model_core.py`/`model_core_sc.py`, ortholog map) → `01_` construction/calibration → `02_` mixture validation → `03_` bulk disease datasets → `04_` single-cell disease datasets (GPU) → `05_` AMP validation.
  - `04_tnf_interaction_ATACseq_SIG16/` — bulk ATAC-seq, cytokine (IL4/IL6/TGFb) × TNF: DESeq2 on the IDR peak atlas → differential accessibility → SIG13-style interaction classes per peak (compared with SIG13/SIG18 RNA) → FIMO motif enrichment.
  - `05_in_vitro_differentiation_RNAseq_SIG18/` — bulk RNA-seq, Th polarization × TNF: DESeq2 DEGs and sPCA programs (dGEPs).
  - `06_crispr_screen_tgfb_tnf_gata3_SIG17/` — pooled CRISPR screen for GATA3 regulators under TGFb + TNF: UMI dedup + MAGeCK (SLURM), then hit analysis.
  - `07_crispr_arrayed_validation_RNAseq_SIG30/` — arrayed CRISPR KO bulk RNA-seq: DESeq2, SIG18 dGEP projection, receptor cross-regulation.
  - `08_tnf_family_tgfb_interaction_RNAseq_SIG29/` — TNF-family ligands × TGFb bulk RNA-seq: DEGs/interaction scoring and SIG18 dGEP projection.
  - `09_treg_depletion_agonist_antibody_SIG19/` — Foxp3-DTR agonist antibody scRNA-seq (Figure 5): scVI/Leiden → cluster annotation → Milo → Palantir/CellRank.
- `functions/` — shared libraries imported by the analysis notebooks/scripts:
  - `functions/perturbseq/` — Python, Perturb-seq-specific utilities (expression normalization, multithreaded z-scoring).
  - `functions/scanpy_custom/` — Python, scanpy extensions (custom dotplots, DSB normalization, embedding/QC plotting, ligand activity scoring) used across most Python notebooks.
  - `functions/r_custom/` — R helper functions for Seurat/scRNA-seq wrangling and plotting, loaded via `source()` rather than as a package.
  - `functions/cell_cycle_mouse_cc2019_seurat.csv` — mouse cell-cycle gene list.
- `environments/` — conda/renv files to reproduce the computational environments (see below).
- `imports_stable/` — frozen input data (entirely git-ignored and downloaded separately; see "Data location").

## Environments

Each notebook/script uses one of the environments below (module READMEs list which); it is usually also clear from its imports/libraries:

- `scanpy_standard2` (`environments/scanpy_standard2.yaml`) — Python stack for scanpy/anndata-based single-cell processing, QC, sPCA, decoupler/pertpy workflows. Create with `conda env create -f environments/scanpy_standard2.yaml` (or `mamba env create -f ...`).
- `scanpy_standard` (`environments/scanpy_standard.yaml`) — older Python stack used for the sPCA SLURM fits and `waggr` scoring notebooks (SIG13 `07_spca`, `09_spca_stability`; SIG18 `02_spca`), and `03_activity_inference_model` folders 01 and 03 (calibration and bulk scoring, same decoupler build).
- `scvi_standard` (`environments/scvi_standard.yaml`) — Python stack with `scvi-tools` for the SIG19 scVI step (`09_treg_depletion_agonist_antibody_SIG19/01_scvi_model_processing.py`).
- `rapids_singlecell` (`environments/rapids_singlecell.yaml`) — GPU stack (`rapids_singlecell`, cupy, CUDA 13) for `analysis/03_activity_inference_model/04_inference_model_disease_sc`.
- `R-signalseq` (`environments/R_signalseq.yaml`) — R 4.5.3 stack (tidyverse, DESeq2 + IHW, ComplexHeatmap, clusterProfiler/msigdbr/fgsea, …) that runs every Rmd in the repo except SIG13 `07_spca/04_spca_visualization.Rmd`.
- `R-deseq2` (`environments/R_deseq2.yaml`) — R 4.4.2 stack (DESeq2, glmGamPoi, future.batchtools, tidyverse) plus `reticulate`/`anndata` so R can read `.h5ad` files directly; used by the glmGamPoi SLURM scripts and by SIG13 `07_spca/04` (which reads an `.h5ad`; the plotting packages it needs are installed). Create with `conda env create -f environments/R_deseq2.yaml`. `environments/renv.lock` is an older renv lockfile (R 4.4.2) kept for reference; it lacks DESeq2 and IHW, and no README relies on it.
- `fastq_processing` (`environments/fastq_processing.yaml`) and `mageck` (`environments/mageck.yaml`) — used only by `analysis/06_crispr_screen_tgfb_tnf_gata3_SIG17/01_processing` (`umi_tools`/`cutadapt` for UMI deduplication, `mageck` for counting and testing).

Python notebooks make the shared libraries importable via `sys.path.insert(0, "<repo>/functions")` followed by `import scanpy_custom as scc` / `import perturbseq`. R scripts load `functions/r_custom/*.R` via `source()`.

## Running the HPC glmGamPoi pipeline

`analysis/02_combinatorial_screen_signalseq_SIG13/05_interaction_scoring/` (see its `README.md`) runs two Slurm-parallelized Gamma-Poisson GLMs with `glmGamPoi` + `future.batchtools` on the `counts` layer of `imports_stable/SIG13/scanpy_outs/SIG13_doublets_DSB7.h5ad` (required `.obs`: `ligand_call_DSB7`, `lane`, `replicate`, `pct_counts_mt`, `S_score`, `G2M_score`):

- `glmGamPoi_single_term_slurm.r` — every condition vs. `linker_linker` (`counts ~ ligand + replicate + lane + percent.mito + s.score + g2m.score`).
- `glmGamPoi_interaction_slurm.r` — 2×2 factorial test per ligand pair (`counts ~ ligand1 * ligand2 + <same covariates>`).
- `interaction_scoring_v3.R` — joins both models and assigns `synergy positive` / `synergy negative` / `buffering` / `none` classes.
- `glmGamPoi_{single_term,interaction}_independent_replicates_slurm.r` — the same two models fit separately per replicate (no `replicate` term); `02_qc_general/02` uses their 0.2filter output for the rep1 vs rep2 reproducibility filter. Interaction variant defaults to `filter_cutoff` 0.2, single-term to 0.05.

Paths are resolved automatically; the only parameter is `filter_cutoff`, passed on the command line (default 0.05 for the single-term, interaction and scoring scripts; 0.2 for the interaction independent-replicates variant). Submit from inside the repo: `sbatch r_job_submission.sh [script] [filter_cutoff]` (defaults to the single-term script) or `sbatch r_job_submission_scoring.sh [filter_cutoff]`; both activate `R-deseq2`. The driver fans out one job per condition/pair and checkpoints each as `.rds`, so reruns skip finished jobs. The same pattern (with their own `*_slurm.r` + `*r_job_submission*.sh`) is used in `03_qc_barcode_cutoff`, `04_qc_barcode_counts` and `06_interaction_scoring_null`, with different cutoffs: the `03`/`04` interaction GLMs hard-code `filter_cutoff` 0.1 (no command-line argument; the variant is chosen by the `DSB_CUTOFF` / `COUNT_SUBSET` env vars), and the `06` null scripts default to 0.05 (single-term) and 0.1 (interaction).

## Data location

Scripts use no absolute paths. Each one finds the repo root by walking up from its working directory (or `__file__` / `$SLURM_SUBMIT_DIR`) to the folder containing `imports_stable/`, then:

- reads every input from `imports_stable/`: frozen copies of study data and of earlier pipeline-step outputs, grouped by experiment (`SIG13/`, `SIG18/`, …). Public non-study datasets (`imports_stable/external/`, used by `03_activity_inference_model` disease scoring) are not included; the main `README.md` ("Running the repository", step 2) lists what to add and where. The whole folder (~207 GB) is git-ignored and downloaded separately; the main `README.md` "Running the repository" section explains how to set it up.
- writes every output to `analysis_outs/<analysis module folder>/…` (git-ignored), keeping the sub-structure of the original output folders.
- imports shared code from the repo's `functions/` (`sys.path.insert(0, f"{REPO_DIR}/functions")`, `source(file.path(repo_dir, "functions/r_custom/..."))`).

Run notebooks, Rmds and `sbatch` from inside the repo. The glmGamPoi checkpoint dirs default to `analysis_outs/<module>/checkpoints/`; set `SIGNALSEQ_SCRATCH` to use scratch space instead.
