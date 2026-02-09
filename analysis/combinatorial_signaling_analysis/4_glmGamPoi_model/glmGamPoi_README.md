# glmGamPoi Differential Expression Pipelines

## Overview

This repository contains two complementary R pipelines for high-throughput differential expression (DE) analysis of single-cell RNA-seq data. Jobs are submitted via r_job_submission.sh

Designed for High-Performance Computing (HPC) environments, both scripts utilize **glmGamPoi** to fit Gamma-Poisson Generalized Linear Models (GLMs) and **future.batchtools** to parallelize analysis across hundreds of conditions via the Slurm workload manager.

## The Pipelines

### 1. Interaction Analysis Pipeline (`glmGamPoi_interaction_slurm.R`)
**Goal:** Identify interaction effects between paired ligands (e.g., synergistic or antagonistic effects).

**Design:** A **2x2 Factorial Design**. For every ligand pair (A & B) found in the metadata, the script subsets cells into four groups:
1.  **Doublet:** `A_B` (Both ligands present)
2.  **Single A:** `A_linker` (Ligand A only)
3.  **Single B:** `linker_B` (Ligand B only)
4.  **Control:** `linker_linker` (Neither ligand)

**Statistical Model:**
$$Expression \sim Ligand1 * Ligand2 + Lane + Replicate + \%Mito + S.Score + G2M.Score$$

It explicitly tests the interaction term (`Ligand1:Ligand2`) to determine if the combinatorial effect differs significantly from the sum of the individual effects.

### 2. Single Term Analysis Pipeline (`glmGamPoi_single_term_slurm.R`)
**Goal:** Standard differential expression testing of specific conditions against a common reference group (linker_linker).

**Design:** A **One-vs-Reference** comparison. It iterates through every unique condition in the metadata and compares it directly to a defined reference group (e.g., `linker_linker`).

**Statistical Model:**
$$Expression \sim Condition + Lane + Replicate + \%Mito + S.Score + G2M.Score$$

where `Condition` is a binary variable (0 = Reference, 1 = Specific Condition).

---

## Inputs

Both pipelines expect a standard Scanpy **AnnData (`.h5ad`)** file, which is read into R directly via `reticulate`.

### 1. Data Structure
* **File Format:** `.h5ad`
* **Expression Layer:** Raw integer counts must be present (default layer: `"counts"`).
    * *Note: glmGamPoi requires raw counts, not log-normalized data.*

### 2. Required Metadata (`adata.obs`)
The `.obs` dataframe **must** contain the following columns:

| Column Name | Description | Pipeline Specifics |
| :--- | :--- | :--- |
| `ligand_call_DSB7` | The primary perturbation label. | **Interaction Pipeline:** Must follow specific patterns: `A_B`, `A_linker`, `linker_B`, `linker_linker`.<br>**Single Term Pipeline:** Can be any unique string. Comparison is made against the defined `ref_level`. |
| `lane` | Batch covariate. | Categorical (e.g., `L1`, `L2`). |
| `replicate` | Batch covariate. | Categorical (e.g., `rep1`, `rep2`). |
| `pct_counts_mt` | Quality covariate. | Numeric (0-100). |
| `S_score` | Cell cycle covariate. | Numeric. |
| `G2M_score` | Cell cycle covariate. | Numeric. |

### 3. System Requirements
* **Slurm Workload Manager:** Scripts are configured to submit jobs via `sbatch`.
* **Conda Environment:** A Conda environment (R-deseq2) containing R, python, reticulate, anndata, and other necessary libraries.

---

## Configuration

Before running either script, you must edit the **"1) Define Parameters"** section at the top of the R file.

### Common Parameters (Both Pipelines)
* `h5ad_file`: The absolute path to your input `.h5ad` file.
* `output_dir`: The directory where final CSV results will be saved.
* `checkpoint_dir`: A scratch directory for temporary intermediate files (must be writable by all compute nodes).
* `conda`: The path to your conda environment binary (e.g., `/home/user/miniforge3/bin/conda`).
* `filter_cutoff`: The minimum fraction of cells (0.0 - 1.0) that must express a gene for it to be included in the test (Default: `0.1`).
* `exp_layer`: The name of the layer in `adata.layers` containing raw counts (Default: `"counts"`).

### Single Term Pipeline Specifics
* `ref_level`: The specific label in `ligand_call_DSB7` that serves as the baseline control (e.g., `"linker_linker"`).

---

## Outputs

The pipelines generate CSV files in the specified `output_dir`. Filenames include the filter cutoff used (e.g., `0.1filter`).

### Interaction Analysis Outputs
| File Name | Description |
| :--- | :--- |
| `glmGamPoi_interaction_lfc_*.csv` | Full differential expression statistics (log2FC, p-values) for the **interaction term**. |
| `glmGamPoi_interaction_lfc_sig_*.csv` | A filtered subset of the interaction results containing only significant genes (adj_pval < 0.1). |
| `glmGamPoi_singles_lfc_*.csv` | Differential expression statistics for the single ligand main effects (Ligand vs Control). |
| `glmGamPoi_coefficients_*.csv` | Raw beta coefficients for all model terms (intercept, covariates, ligands). |

### Single Term Analysis Outputs
| File Name | Description |
| :--- | :--- |
| `glmGamPoi_singleTerm_lfc_*.csv` | Full differential expression statistics comparing the specific condition vs. reference. |
| `glmGamPoi_singleTerm_lfc_sig_*.csv` | A filtered subset containing only significant genes (adj_pval < 0.1). |
| `glmGamPoi_coefficients_*.csv` | Raw beta coefficients for all model terms. |
