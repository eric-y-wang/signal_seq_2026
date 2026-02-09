#!/bin/bash

#SBATCH --job-name=glm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/data1/rudenska/EYW/git_projects/SIG13/analysis/interaction_scoring/R-out.%j
#SBATCH --error=/data1/rudenska/EYW/git_projects/SIG13/analysis/interaction_scoring/R-err.%j

# load bulkseq conda environment
source ~/.bashrc
mamba activate R-deseq2

# set directory (with fail safe in case it fails)
cd /data1/rudenska/EYW/git_projects/SIG13/analysis/interaction_scoring|| { echo "Failure"; exit 1; }

Rscript glmGamPoi_single_term_slurm.r
