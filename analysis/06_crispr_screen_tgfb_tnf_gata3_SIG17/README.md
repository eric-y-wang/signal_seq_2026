# CRISPR Screen (SIG17): TGFb + TNF Regulators of GATA3

SIG17 is a pooled CRISPR knockout screen in murine CD4 T cells. Cells cultured with
TGFb + TNF were sorted into 4 bins by GATA3-reporter expression (bin 1 = lowest, bin 4 =
highest). The screen asks which genes, when knocked out, shift cells along the GATA3 axis.
The hits are then compared with the TGFb + TNF interaction term from the SIG18 bulk
ligand-combination experiment.

Each sgRNA read carries a 20bp UMI, so reads are collapsed to one count per transduced
cell before running MAGeCK. Each higher bin is tested against bin 1:

```text
mageck test -t TGFb_TNF_gata3_{2,3,4} -c TGFb_TNF_gata3_1 \
  --control-sgrna mageck_control_id.txt --sort-criteria pos --remove-zero both
```

A positive gene LFC means the gene's sgRNAs are enriched in high-GATA3 cells, so the
gene is a negative regulator of GATA3 in this condition.

## Folders

| folder | what it does |
|---|---|
| `01_processing` | UMI extraction (`umi_tools`), guide matching and UMI deduplication, `mageck count` and `mageck test` (SLURM) |
| `02_mageck` | clone-count and sequencing-saturation QC, gene- and sgRNA-level hits, and comparison with the SIG18 TGFb-TNF interaction |

## Run order

```text
01_processing ──> 02_mageck
```

Run `01_processing` first (`bash submit_all.sh` from that folder). `02_mageck` reads the
stable copies of its outputs in `imports_stable/SIG17/dedup_pipeline_output/`, so it can
also be knit without re-running `01_processing`. The two Rmds in `02_mageck` can be knit in
either order. All scripts find the repo root by searching upward for `imports_stable/`, so
run them from inside the repo.

Each folder's README has its own steps, inputs and outputs.

## Environments

| folder | environment |
|---|---|
| 01 | `fastq_processing` (steps 01-02), `mageck` (steps 03-04) |
| 02 | `R-signalseq` |

Environment files are in `../../environments/`.

## Outputs

`01_processing` writes to its `OUT_DIR` (set in `config.sh`),
`analysis_outs/06_crispr_screen_tgfb_tnf_gata3_SIG17/dedup_pipeline_output/`. `02_mageck`
writes PDFs to `analysis_outs/06_crispr_screen_tgfb_tnf_gata3_SIG17/crispr_screen_tgfb_tnf_gata3/`.
Both are gitignored.
