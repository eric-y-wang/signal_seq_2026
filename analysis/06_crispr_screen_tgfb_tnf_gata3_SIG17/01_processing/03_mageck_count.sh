#!/bin/bash

#SBATCH --job-name=mageck_count
#SBATCH --partition=cpushort
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mageck_count_%j.out
#SBATCH --error=logs/mageck_count_%j.err

# submit from this folder (01_processing) so config.sh and logs/ resolve
PIPE_DIR=${SLURM_SUBMIT_DIR:-.}
source "${PIPE_DIR}/config.sh"

source ~/.bashrc
conda activate "$MAGECK_ENV"

mkdir -p "${OUT_DIR}/03_mageck_dedup"

# one UMI-deduplicated read per (sgRNA, UMI cluster), so these counts are clone counts
fastqs=()
for s in "${SAMPLES[@]}"; do
  fastqs+=("${IN_DIR}/02_guide_dedup/${s}_R1.dedup.fastq.gz")
done

mageck count -l "${PIPE_DIR}/mageck_library.csv" \
  -n "${OUT_DIR}/03_mageck_dedup/SIG17_gata3_dedup" \
  --sample-label "$(IFS=,; echo "${LABELS[*]}")" \
  --fastq "${fastqs[@]}"
