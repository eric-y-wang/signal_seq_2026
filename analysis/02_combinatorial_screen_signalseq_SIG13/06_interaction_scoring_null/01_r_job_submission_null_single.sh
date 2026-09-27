#!/bin/bash

#SBATCH --job-name=glm_null_single
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

# Locate the repo root (the folder containing imports_stable/) from the submit directory
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_DIR" != "/" ] && [ ! -d "$REPO_DIR/imports_stable" ]; do REPO_DIR=$(dirname "$REPO_DIR"); done

# set directory (with fail safe in case it fails)
cd "$REPO_DIR/analysis/02_combinatorial_screen_signalseq_SIG13/06_interaction_scoring_null" || { echo "Failure"; exit 1; }

# Usage: sbatch 01_r_job_submission_null_single.sh [filter_cutoff]  (defaults to 0.05)
Rscript 01_glmGamPoi_single_term_null_slurm.r ${1:+"$1"}
