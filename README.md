# signal_seq_2026
This repository contains the analysis code, custom functions, and reproducibility environments for the 2026 Signal-seq manuscript.

## Repository Structure

The repository is organized into three main directories: `analysis`, `functions`, and `environments`.

### 1. Analysis
The `analysis/` directory contains the core workflows used to generate the results and figures in the manuscript. Scripts are generally numbered to indicate the order of execution.

* **`agonist_antibody_analysis/`**
    * Analysis of Foxp3-DTR agonist antibody experiment (Figure 4).
    * **Workflows:** `scVI` model processing, Leiden clustering, `Milo` differential abundance testing, and linear model testing of Differentiation Gene Expression Programs (dGEP) module scores between clusters.
* **`combinatorial_signaling_analysis/`**
    * Analysis of 648 condition combinatorial signal-seq screen (Figure 2)
    * **Preprocessing:** Cell Ranger processing, barcode/RNA QC, and inter-replicate QC.
    * **Differential Expression (glmGamPoi):**
        * Implements a custom pipeline using `glmGamPoi` and `future.batchtools` for high-throughput interaction testing on HPC systems.
        * **Interaction Model:** Tests for synergistic/antagonistic effects using a 2x2 factorial design ($Ligand1 * Ligand2$).
        * **Single Term Model:** Standard one-vs-reference testing.
    * **Sparse PCA (sPCA):** Z-score processing, running sPCA to identify GEPs, scoring GEPs, linear modeling to connect ligands to GEP effects.
    * **Downstream:** Interaction effect clustering and single ligand analysis.
* **`regularized_regression_model/`**
    * Calibration and application of of ridge regression model to predict signaling activity.
    * **Workflows:** Explanatory matrix construction, model calibration on mouse and human data, ROC analysis for sensitivity/specificity testing, and application to external datasets.

### 2. Functions
Custom libraries and helper functions used across the analysis notebooks.

* **`functions/perturbseq/`**: Python modules for Perturb-seq specific tasks, including multi-threaded Z-score calculation.
* **`functions/scanpy_custom/`**: Extensions for `scanpy`, including custom dotplots and DSB normalization.
* **`functions/r_custom/`**: R scripts for plotting and scRNA-seq analysis utilities.

### 3. Environments
Files to reproduce the computational environments used in this study.

* **`environments/R-deseq2.yaml`**: Conda environment file containing R and dependencies, for use in some workflows.
* **`environments/scanpy_standard.yaml`**: Conda environment file containing Python dependencies.
* **`environments/renv.lock`**: Renv lockfile capturing R dependencies.
