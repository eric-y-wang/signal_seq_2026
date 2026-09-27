#!/bin/bash

# Step 03: run 02_score_disease_datasets.py on SLURM, one job per dataset.
#
#   sbatch 03_run_disease_scoring.sh              # all four datasets
#   sbatch --array=2 03_run_disease_scoring.sh    # just thomas_ibd (see DATASETS below)
#
# Resources are sized from observed usage (sacct, 16 cpus), dominated by reading
# the whole .h5ad into RAM:
#   inflammation_atlas  11 min   89 GB   (1.5M cells; 4 groupings)
#   thomas_ibd           6 min  100 GB   (dense counts + log1p layers, ~53 GB on disk)
#   amp_2023             2 min   50 GB
#   sig19_iln           <1 min   10 GB
# 150 GB / 1 h leaves headroom on the largest without over-requesting.

#SBATCH --job-name=disease_activity_scoring
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%A_%a-%x.out

DATASETS=(inflammation_atlas amp_2023 thomas_ibd sig19_iln)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# activate mamba environment
# scanpy_standard (not scanpy_standard2) to match the decoupler build the model
# was calibrated with in 01_inference_model_construction_validation
source ~/.bashrc
mamba activate scanpy_standard

# Ensure we use the conda env's libstdc++ before the system one
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# run from the submit directory (sbatch from this folder)
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "Failure"; exit 1; }

which python
echo "scoring dataset: $DATASET"
python 02_score_disease_datasets.py "$DATASET"
