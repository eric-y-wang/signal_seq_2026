#!/bin/bash

#SBATCH --job-name=umi_extract
#SBATCH --partition=cpushort
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/umi_extract_%A_%a.out
#SBATCH --error=logs/umi_extract_%A_%a.err

# UMI structure:
#   R2 bp 1-20   = UMI
#   R2 bp 21-35  = constant adapter, observed as the complement of TTTTTTTCGTGGCTG,
#                  i.e. AAAAAAAGCACCGAC (poly-A run length varies +/-1bp from sequencing slippage)
# umi_tools requires the umi-carrying regex group to be named umi_<N>, not just umi.

# submit from this folder (01_processing) so config.sh and logs/ resolve
source "${SLURM_SUBMIT_DIR:-.}/config.sh"

source ~/.bashrc
conda activate "$FASTQ_ENV"

SAMPLE=${SAMPLES[$((SLURM_ARRAY_TASK_ID - 1))]}

R1_IN=${FASTQ_DIR}/${SAMPLE}_R1.fastq.gz
R2_IN=${FASTQ_DIR}/${SAMPLE}_R2.fastq.gz

OUTDIR=${OUT_DIR}/01_umi_extract
mkdir -p "$OUTDIR"

umi_tools extract \
  --extract-method=regex \
  --bc-pattern2="(?P<umi_1>.{20})A{6,8}GCACCGAC.*" \
  --stdin="$R1_IN" \
  --read2-in="$R2_IN" \
  --stdout="${OUTDIR}/${SAMPLE}_R1.umi.fastq.gz" \
  --read2-out="${OUTDIR}/${SAMPLE}_R2.umi.fastq.gz" \
  --filtered-out="${OUTDIR}/${SAMPLE}_R1.umi_failed.fastq.gz" \
  --filtered-out2="${OUTDIR}/${SAMPLE}_R2.umi_failed.fastq.gz" \
  --log="${OUTDIR}/${SAMPLE}.umi_extract.log"

# R2 is only needed to supply the UMI; discard it once extraction is done to save space.
rm -f "${OUTDIR}/${SAMPLE}_R2.umi.fastq.gz"
