#!/bin/bash

#SBATCH --job-name=calbration_scoring
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00

# activate mamba environment
source ~/.bashrc
mamba activate scanpy_standard

# Ensure we use the conda env's libstdc++ before the system one
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# run from the submit directory (sbatch from this folder)
cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "Failure"; exit 1; }

which python
python 02_calibration_input_scoring.py
