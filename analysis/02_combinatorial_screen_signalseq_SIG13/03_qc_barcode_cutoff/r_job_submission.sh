#!/bin/bash

#SBATCH --job-name=glm-cutoff
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=R-out.%j
#SBATCH --error=R-err.%j

# usage: sbatch r_job_submission.sh <cutoff>   e.g. sbatch r_job_submission.sh 4

# load bulkseq conda environment
source ~/.bashrc
mamba activate R-deseq2

# Each driver runs from its own directory: future.batchtools puts its registry under `.future/`
# relative to the working directory, and sharing that between concurrent drivers makes them fail
# with "Log file ... for job with id 1 not available".
# Locate the repo root (folder containing imports_stable/) from the submission directory
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_DIR" != "/" ] && [ ! -d "$REPO_DIR/imports_stable" ]; do REPO_DIR=$(dirname "$REPO_DIR"); done
BASE="$REPO_DIR/analysis/02_combinatorial_screen_signalseq_SIG13/03_qc_barcode_cutoff"
RUN_DIR=$BASE/run_DSB$1
mkdir -p $RUN_DIR
cd $RUN_DIR || { echo "Failure"; exit 1; }

export DSB_CUTOFF=$1
Rscript $BASE/04_glmGamPoi_interaction_cutoff_slurm.r
