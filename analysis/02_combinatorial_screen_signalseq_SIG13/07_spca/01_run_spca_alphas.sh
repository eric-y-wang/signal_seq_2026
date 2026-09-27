#!/bin/bash
#SBATCH --partition=cpushort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

# Locate the repo root (the folder containing imports_stable/) from the submit directory
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_DIR" != "/" ] && [ ! -d "$REPO_DIR/imports_stable" ]; do REPO_DIR=$(dirname "$REPO_DIR"); done

# List of alpha values to try
alpha_values=(1.0 1.5 2.0)

for alpha in "${alpha_values[@]}"; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=spca_alpha_${alpha}
#SBATCH --partition=cpushort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16             
#SBATCH --mem=80G
#SBATCH --time=2:00:00

source ~/.bashrc
mamba activate scanpy_standard

# Ensure we use the conda env’s libstdc++ before the system one
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"


python ${REPO_DIR}/analysis/02_combinatorial_screen_signalseq_SIG13/07_spca/01_spca_degs_zscore_expression_allLigands.py \
  --alpha ${alpha}
EOF
done
