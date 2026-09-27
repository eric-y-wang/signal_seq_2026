#!/bin/bash

#SBATCH --job-name=glm
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
cd "$REPO_DIR/analysis/02_combinatorial_screen_signalseq_SIG13/05_interaction_scoring" || { echo "Failure"; exit 1; }

# Usage: sbatch r_job_submission.sh [script_name] [filter_cutoff]
# Defaults to the single-term script; filter_cutoff defaults to 0.05, except 0.2 for
# glmGamPoi_interaction_independent_replicates_slurm.r.
Rscript "${1:-glmGamPoi_single_term_slurm.r}" ${2:+"$2"}
