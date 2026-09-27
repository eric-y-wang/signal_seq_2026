#!/bin/bash

#SBATCH --job-name=guide_dedup
#SBATCH --partition=cpushort
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/guide_dedup_%A_%a.out
#SBATCH --error=logs/guide_dedup_%A_%a.err

# submit from this folder (01_processing) so config.sh and logs/ resolve
PIPE_DIR=${SLURM_SUBMIT_DIR:-.}
source "${PIPE_DIR}/config.sh"

source ~/.bashrc
conda activate "$FASTQ_ENV"

SAMPLE=${SAMPLES[$((SLURM_ARRAY_TASK_ID - 1))]}

mkdir -p "${OUT_DIR}/02_guide_dedup"

python "${PIPE_DIR}/02_guide_dedup.py" \
  --sample "$SAMPLE" \
  --umi-fastq "${IN_DIR}/01_umi_extract/${SAMPLE}_R1.umi.fastq.gz" \
  --library "${PIPE_DIR}/mageck_library.csv" \
  --outdir "${OUT_DIR}/02_guide_dedup" \
  --statsdir "${OUT_DIR}/02_guide_dedup" \
  --threads "$SLURM_CPUS_PER_TASK"
