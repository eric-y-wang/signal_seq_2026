#!/bin/bash

#SBATCH --job-name=mageck_test
#SBATCH --partition=cpushort
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mageck_test_%j.out
#SBATCH --error=logs/mageck_test_%j.err

# submit from this folder (01_processing) so config.sh and logs/ resolve
PIPE_DIR=${SLURM_SUBMIT_DIR:-.}
source "${PIPE_DIR}/config.sh"

source ~/.bashrc
conda activate "$MAGECK_ENV"

# output folder name kept as 05_ to match the existing outputs read by ../02_mageck
TEST_DIR=${OUT_DIR}/05_mageck_test_dedup
mkdir -p "$TEST_DIR"

# each higher GATA3 bin vs bin 1. mageck test normalizes using only the -t/-c samples,
# so results do not depend on which other samples are in the count table.
for bin in 4 3 2; do
  mageck test -k "${COUNT_TABLE}" \
    -t "TGFb_TNF_gata3_${bin}" -c TGFb_TNF_gata3_1 \
    -n "${TEST_DIR}/TGFb_TNF_gata3_${bin}_v_1" \
    --control-sgrna "${PIPE_DIR}/mageck_control_id.txt" --sort-criteria pos --remove-zero both \
    --additional-rra-parameters "--permutation 100000"
done
