#!/bin/bash
#SBATCH --partition=cpushort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

# Dense alpha sweep, 0.1 to 3.0 in steps of 0.1, at the fixed consensus component count (78)
# from the existing alpha=1.0 analysis. alpha=1.0 itself is skipped and reused directly from
# analysis_outs/spca/degs_zscore_allLigands/zscore_degs_allLigands_0.1_alpha1.0_sPCA_components.csv
# since a single fixed-n_components fit at alpha=1.0 reproduces that existing result exactly
# (same data, same random_state=100).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

alpha_values=($(seq 0.1 0.1 3.0))

for alpha in "${alpha_values[@]}"; do
  if [ "$alpha" == "1.0" ]; then
    continue
  fi
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=spca_sweep_${alpha}
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/.bashrc
mamba activate scanpy_standard

# Ensure we use the conda env's libstdc++ before the system one
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"

python ${SCRIPT_DIR}/01_spca_alpha_sweep.py \
  --alpha ${alpha} \
  --n_components 78
EOF
done
