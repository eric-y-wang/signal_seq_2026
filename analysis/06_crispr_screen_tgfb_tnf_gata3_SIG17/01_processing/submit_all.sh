#!/bin/bash
# Submits the Gata3-arm UMI-dedup MAGeCK pipeline as a SLURM dependency chain:
#   01_umi_extract.sh (array 1-4)
#     -> 02_guide_dedup.sh (array 1-4)
#       -> 03_mageck_count.sh
#         -> 04_mageck_test.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

extract_id=$(sbatch --parsable 01_umi_extract.sh)
echo "umi_extract array: $extract_id"

dedup_id=$(sbatch --parsable --dependency=afterok:${extract_id} 02_guide_dedup.sh)
echo "guide_dedup array: $dedup_id"

count_id=$(sbatch --parsable --dependency=afterok:${dedup_id} 03_mageck_count.sh)
echo "mageck_count: $count_id"

test_id=$(sbatch --parsable --dependency=afterok:${count_id} 04_mageck_test.sh)
echo "mageck_test: $test_id"
