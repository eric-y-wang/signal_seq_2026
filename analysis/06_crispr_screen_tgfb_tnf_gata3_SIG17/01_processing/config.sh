#!/bin/bash
# Shared paths/environments for the SIG17 Gata3-arm processing pipeline.
# Sourced by every step; edit here to re-run in a different environment.

# Repo root = nearest parent of the submit dir containing imports_stable/
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$REPO_DIR" != "/" ] && [ ! -d "$REPO_DIR/imports_stable" ]; do REPO_DIR=$(dirname "$REPO_DIR"); done

# raw sgRNA-seq fastqs: <FASTQ_DIR>/<sample>_R1.fastq.gz (sgRNA) and _R2.fastq.gz (UMI)
# (not included in imports_stable/; download from GEO GSE348675 into this folder)
FASTQ_DIR=${REPO_DIR}/imports_stable/SIG17/raw_fastq/merged

# stable copies of earlier-step outputs read by steps 02-03 (set IN_DIR=$OUT_DIR to chain from fresh outputs)
IN_DIR=${REPO_DIR}/imports_stable/SIG17/dedup_pipeline_output

# step-04 input: stable copy of step 03's count table (set to ${OUT_DIR}/03_mageck_dedup/... to chain
# from a fresh step 03)
COUNT_TABLE=${IN_DIR}/03_mageck_dedup/SIG17_gata3_dedup.count.txt

# pipeline outputs (one numbered subdirectory per step)
OUT_DIR=${REPO_DIR}/analysis_outs/06_crispr_screen_tgfb_tnf_gata3_SIG17/dedup_pipeline_output

# conda environments: umi_tools + cutadapt (steps 01-02) and mageck (steps 03-04)
FASTQ_ENV=fastq_processing
MAGECK_ENV=mageck

# Gata3 arm (TGFb + TNF): SIG17_1-4 == GATA3-reporter sort bins 1-4 (low -> high)
SAMPLES=(SIG17_1 SIG17_2 SIG17_3 SIG17_4)
LABELS=(TGFb_TNF_gata3_1 TGFb_TNF_gata3_2 TGFb_TNF_gata3_3 TGFb_TNF_gata3_4)
