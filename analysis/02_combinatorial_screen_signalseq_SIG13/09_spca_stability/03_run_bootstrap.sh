#!/bin/bash
#SBATCH --job-name=spca_bootstrap
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=140G
#SBATCH --time=06:00:00

# Full 2000-iteration cell-level bootstrap run (iteration i seeded with --seed_base + i). The
# worker script skips any iteration whose output file already exists, so an interrupted run can
# be resubmitted and only the missing iterations are computed.

source ~/.bashrc
mamba activate scanpy_standard

# Ensure we use the conda env's libstdc++ before the system one
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$SLURM_SUBMIT_DIR"

python 03_spca_cell_bootstrap_worker.py \
  --n_iter 2000 \
  --start_iter 0 \
  --n_components 78 \
  --alpha 1.0 \
  --seed_base 0
