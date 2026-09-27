#!/bin/bash
#SBATCH --partition=cpushort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

# Locate the repo root (folder containing imports_stable/) from the submission directory
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_DIR" != "/" ] && [ ! -d "$REPO_DIR/imports_stable" ]; do REPO_DIR=$(dirname "$REPO_DIR"); done

# List of alpha values to try
alpha_values=(9.0 10.0)

for alpha in "${alpha_values[@]}"; do
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=spca_alpha_${alpha}
#SBATCH --partition=cpushort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24             
#SBATCH --mem=64G
#SBATCH --time=1:00:00

source ~/.bashrc
mamba activate scanpy_standard

# Ensure we use the conda env’s libstdc++ before the system one
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"


python ${REPO_DIR}/analysis/05_in_vitro_differentiation_RNAseq_SIG18/02_spca/01_spca_degs_zscore_expression.py \
  --alpha ${alpha}
EOF
done
