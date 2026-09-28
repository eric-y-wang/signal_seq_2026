# CRISPR Screen (TGFb + TNF, GATA3): UMI Deduplication and MAGeCK

This folder turns raw sgRNA-seq reads from the four GATA3-reporter sort bins (`SIG17_1-4`, bin 1 =
lowest, bin 4 = highest) into the clone counts and `mageck test` results that `../02_mageck/`
reads.

The key step is **UMI deduplication**. Each sgRNA read (R1) is paired with a 20bp UMI on R2. Reads
that share a guide and a UMI are PCR or optical duplicates of one original molecule, so they are
collapsed before counting. Each surviving (guide, UMI-cluster) pair therefore stands for **one
transduced cell**, and MAGeCK counts cells, not reads:

```text
R2:  [UMI, 20bp][AAAAAAAGCACCGAC ...]          # adapter poly-A run varies +/-1bp
R1:  ...ACACCG[guide, 20bp]GTTTAAGAGC...
```

Each higher bin is then compared with bin 1:

```text
mageck test -t TGFb_TNF_gata3_{2,3,4} -c TGFb_TNF_gata3_1 \
  --control-sgrna mageck_control_id.txt --sort-criteria pos --remove-zero both \
  --additional-rra-parameters "--permutation 100000"
```

The original screen also sequenced two RORγt arms (`SIG17_5-12`), which were processed in the same
run. Only the Gata3 arm is kept here. Every step before `mageck test` treats each sample on its
own, and `mageck test` normalizes using only the `-t`/`-c` samples. Removing the other samples
therefore does not change the Gata3 results.

## Pipeline

- **`config.sh`**: paths and environments shared by all steps. `REPO_DIR` (the repo root, found
  by searching upward from the submit folder for `imports_stable/`), `FASTQ_DIR` (raw fastqs),
  `IN_DIR` (stable copies of earlier-step outputs), `COUNT_TABLE` (step 04's input), `OUT_DIR`
  (outputs), the two conda environments (by name), and the sample list and MAGeCK labels
  (`SIG17_{1..4}` → `TGFb_TNF_gata3_{1..4}`).
- **`01_umi_extract.sh`**: `umi_tools extract` in regex mode (SLURM array, one task per sample).
  It takes the 20bp UMI from R2, requires the adapter to match
  (`(?P<umi_1>.{20})A{6,8}GCACCGAC.*`), and tags the UMI onto each R1 read name. Reads without the
  adapter go to `*_umi_failed.fastq.gz`. The UMI-tagged R2 is deleted once extraction finishes.
- **`02_guide_dedup.py`** / **`02_guide_dedup.sh`**: per-sample deduplication (SLURM array).
  - **Guide isolation.** `cutadapt -g ACACCG...GTTTAAGAGC -e 0.15` trims the vector flanks to
    leave the 20bp guide. Reads where the flanks are not found are kept so the read order still
    matches the input, and are skipped later because they are longer than 25bp after trimming.
  - **Guide matching.** Each guide is matched to `mageck_library.csv`, first exactly and then with
    a single-mismatch fallback. The fallback leaves out any 1-mismatch sequence shared by two
    library guides, so no read is assigned ambiguously.
  - **UMI clustering.** Within each guide, `umi_tools`' `UMIClusterer(cluster_method="directional",
    threshold=1)` merges UMIs that probably differ only by sequencing error. This is the same
    algorithm `umi_tools dedup` uses, with guide identity taking the place of alignment position.
  - **Output.** Writes one representative full-length read per (guide, UMI-cluster) pair, plus
    per-guide and per-sample QC tables.
- **`03_mageck_count.sh`**: `mageck count` on the four deduplicated fastqs (read from `IN_DIR`).
- **`04_mageck_test.sh`**: `mageck test` for bins 4, 3 and 2 against bin 1 on `COUNT_TABLE` (see
  `config.sh`; by default the stable copy of step 03's `SIG17_gata3_dedup.count.txt`), using the 94
  `control_NT` / `control_cutting` sgRNAs in `mageck_control_id.txt` as negative controls.
- **`submit_all.sh`**: submits steps 01-04 as one SLURM dependency chain.

## Inputs

- `${FASTQ_DIR}/SIG17_{1..4}_R{1,2}.fastq.gz`: raw paired sgRNA-seq reads
  (`imports_stable/SIG17/raw_fastq/merged/`). These are not included in `imports_stable/`;
  download them from GEO ([GSE348675](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE348675))
  into that folder before running step 01.
- Steps 02-04 read the previous step's outputs from `IN_DIR`
  (default `imports_stable/SIG17/dedup_pipeline_output/`): `01_umi_extract/SIG17_{1..4}_R1.umi.fastq.gz`
  (02), `02_guide_dedup/SIG17_{1..4}_R1.dedup.fastq.gz` (03) and
  `03_mageck_dedup/SIG17_gata3_dedup.count.txt` (04, via `COUNT_TABLE`). Of these, only the
  step 03 count table is in `imports_stable/`, so only step 04 can run on its own. The step 01-02
  fastqs are not included (the `02_guide_dedup/` QC tables are); regenerate them by running
  steps 01-02 on the GEO fastqs with `IN_DIR=$OUT_DIR` (and `COUNT_TABLE` pointed at
  `${OUT_DIR}/03_mageck_dedup/SIG17_gata3_dedup.count.txt` to chain through 04).
- `mageck_library.csv`: sgRNA library (`sgRNA_id,sequence,gene`; 1,974 sgRNAs, 4 per targeting
  gene, plus 47 `control_NT` and 47 `control_cutting`).
- `mageck_control_id.txt`: negative-control sgRNA IDs.

## Outputs

All outputs are written under `OUT_DIR`
(`analysis_outs/06_crispr_screen_tgfb_tnf_gata3_SIG17/dedup_pipeline_output/`, gitignored).
`../02_mageck/` reads the stable copies of these outputs in
`imports_stable/SIG17/dedup_pipeline_output/`.

- `01_umi_extract/`: UMI-tagged R1 fastqs, reads that failed the adapter match, and `umi_tools`
  logs.
- `02_guide_dedup/`:
  - `SIG17_{1..4}_R1.dedup.fastq.gz`: deduplicated reads, the input to step 03
  - `SIG17_{1..4}.umi_per_guide.tsv`: per sgRNA, the matched reads, raw unique UMIs, UMI clusters
    (clones) and mean/max clone size
  - `SIG17_{1..4}.dedup_stats.json`: per-sample totals, duplication rate and clone-size histogram
  - `cutadapt` logs and trimmed-guide intermediates
- `03_mageck_dedup/SIG17_gata3_dedup.count.txt`: sgRNA x sample clone-count table, plus
  `mageck count` summaries.
- `05_mageck_test_dedup/TGFb_TNF_gata3_{2,3,4}_v_1.{gene,sgrna}_summary.txt`: `mageck test`
  results. The folder keeps its `05_` name so it matches the existing outputs.

SLURM stdout/stderr go to `logs/` in this folder, which is gitignored (`*.out`, `*.err`).

## Running

```bash
cd analysis/06_crispr_screen_tgfb_tnf_gata3_SIG17/01_processing
bash submit_all.sh
```

Each step can also be submitted with `sbatch`. Submit from this folder, because the scripts find
`config.sh`, the library files and `logs/` through `$SLURM_SUBMIT_DIR`. The `#SBATCH` partition
(`cpushort`) is specific to this cluster.

Two conda environments are used. Both are exported in `environments/`:

- `fastq_processing` (`environments/fastq_processing.yaml`, steps 01-02): Python 3.12,
  `umi_tools` 1.1.6, `cutadapt` 5.2. The newest bioconda `umi_tools` build only supports Python
  3.12 and older.
- `mageck` (`environments/mageck.yaml`, steps 03-04): `mageck` 0.5.9.5.

Create them with `conda env create -f environments/<name>.yaml`. `config.sh` activates them by
name (`FASTQ_ENV=fastq_processing`, `MAGECK_ENV=mageck`).
