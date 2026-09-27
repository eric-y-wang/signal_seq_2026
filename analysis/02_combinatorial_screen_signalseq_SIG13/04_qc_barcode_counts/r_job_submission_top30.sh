#!/bin/bash

#SBATCH --job-name=glm-top30count
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=R-out.%j
#SBATCH --error=R-err.%j

# load bulkseq conda environment
source ~/.bashrc
mamba activate R-deseq2

# set directory (with fail safe in case it fails)
# Locate the repo root (folder containing imports_stable/) from the submission directory
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_DIR" != "/" ] && [ ! -d "$REPO_DIR/imports_stable" ]; do REPO_DIR=$(dirname "$REPO_DIR"); done
cd "$REPO_DIR/analysis/02_combinatorial_screen_signalseq_SIG13/04_qc_barcode_counts" || { echo "Failure"; exit 1; }

export COUNT_SUBSET=top30
Rscript 01_glmGamPoi_interaction_countSubset_slurm.r
