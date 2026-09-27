#!/bin/bash

# Step 02: run 01_score_disease_datasets_sc.py on SLURM, one GPU job per dataset.
#
#   sbatch 02_run_sc_scoring.sh              # all four datasets
#   sbatch --array=2 02_run_sc_scoring.sh    # just thomas_ibd (see DATASETS below)
#
# Host RAM is driven by the CSR read; wall time by the permutation null.
#
# Only `layers/log1p_norm` is read, densified straight into the net's ~1.5k genes,
# so the working set is far smaller than the files -- the atlas is the largest at
# 1,505,203 x 1,449 float32 = 8.7 GB (measured peak RSS 39.2 GB), while thomas_ibd's
# file is ~53 GB because it carries several dense layers none of which are touched.
#
# The 1000-permutation null dominates wall time. Measured end to end (array 4872617,
# ALPHA_RANGE = logspace(-1, 5, 200)):
#
#   inflammation_atlas  6 m 52 s   44.3 GB   1,505,203 cells
#   thomas_ibd            41 s      5.3 GB     145,704 cells
#   amp_2023              19 s      4.9 GB      55,432 cells
#   sig19_iln             17 s      3.2 GB      19,898 cells
#
# Peak GPU use is the scaled expression array plus one permutation block
# (PERM_BUDGET), comfortably under 12 GB, so any card on the partition has headroom
# and there is no reason to pin h100/a100.
#
# 64 GB / 1 h leaves ample margin on the atlas, the largest on both axes.

#SBATCH --job-name=sc_activity_scoring
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/01_score_%A_%a.out
#SBATCH --error=logs/01_score_%A_%a.err

set -euo pipefail

DATASETS=(inflammation_atlas amp_2023 thomas_ibd sig19_iln)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# conda env built from environments/rapids_singlecell.yaml
set +u; source ~/.bashrc; mamba activate rapids_singlecell; set -u
# submit dir, so this runs correctly from the main checkout or a git worktree
DIR="${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p "$DIR/logs"
cd "$DIR"

nvidia-smi
echo "scoring dataset: $DATASET"

python 01_score_disease_datasets_sc.py "$DATASET"

echo "done: $(date)"
