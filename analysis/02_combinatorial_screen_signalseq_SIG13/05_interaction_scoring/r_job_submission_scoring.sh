#!/bin/bash

#SBATCH --job-name=glm_score
#SBATCH --partition=cpushort
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=R-out.%j
#SBATCH --error=R-err.%j

# load bulkseq conda environment
source ~/.bashrc
mamba activate R-deseq2

# set directory (with fail safe in case it fails)
# Locate the repo root (folder containing imports_stable/) from the submission directory
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_DIR" != "/" ] && [ ! -d "$REPO_DIR/imports_stable" ]; do REPO_DIR=$(dirname "$REPO_DIR"); done
cd "$REPO_DIR/analysis/02_combinatorial_screen_signalseq_SIG13/05_interaction_scoring" || { echo "Failure"; exit 1; }

# Usage: sbatch r_job_submission_scoring.sh [filter_cutoff]  (defaults to 0.05)
Rscript interaction_scoring_v3.R ${1:+"$1"}
