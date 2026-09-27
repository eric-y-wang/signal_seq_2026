# Linear vs. Circular RNA Barcodes (SIG02, SIG03)

Compares two barcode designs for Signal-seq VLPs by how well each barcode (oBC) is
captured in single-cell data.

| experiment | chemistry | barcodes | VLPs per cell | conditions |
|---|---|---|---|---|
| SIG02 | `pSL`, ENTER-seq style linear barcodes | 18 oBCs across 8 pooled cargo genes (`{gene}_BC{n}`, 1-4 per gene) | 2, each uniquely barcoded | `lib` (pooled cargo library) and `control` (linker-only `NtermF` VLP), at 6 h and 24 h |
| SIG03 | `p139`, circular barcodes | `p139-BC4`, `p139-BC5` | 1 | CD4 T cells at 4 VLP titers (`2e11` to `6e9`) and a `0`-titer control, at 6 h and 22 h |

SIG02 cells received two uniquely barcoded VLPs, while SIG03 cells received one. 
The SIG02 `control` is a real virus (the linker-only `NtermF` construct), not a no-virus
condition, so its cells also carry viral transcript and `NtermF` barcode signal. Only
SIG03 has a no-virus control.

## Pipeline

- **`01_linear_barcode_viz_SIG02.ipynb`**: per-cell viral transcript (`pSL-*`) UMIs vs.
  oBC UMIs in SIG02.
  - Joint plots of viral vs. total oBC UMIs, by `condition` x `timepoint`.
  - Joint plots per cargo gene (`lib` cells), and for `NtermF` in `lib` vs. `control`
    (the one construct delivered in both).
  - Violin plots of oBC UMIs per cargo gene (`lib` cells), and a per-gene summary of
    median UMI and detection rate (> 0, > 2 and > 5 UMIs).
- **`02_circular_barcode_signalseq_SIG03.Rmd`**: circular barcode detection in SIG03 CD4
  T cells.
  - Reads the `.h5mu` directly with `rhdf5`.
  - Barcode UMI histograms per titer, and BC4 vs. BC5 density scatters at 6 h and 22 h.
  - Classification: for each barcode x timepoint, each titer is scored against the
    matched `0`-titer control, with raw barcode UMIs as the predictor (ROC AUC, ROC and PR
    curves, `yardstick`).
- **`03_linear_vs_circular_counts_SIG02_SIG03.ipynb`**: puts both chemistries side by side
  (SIG03 CD4 cells at `2e11` titer, all SIG02 cells). Scatter plots of viral vs. oBC UMIs,
  both summed per cell and per barcode, ridge plots of the viral and oBC UMI distributions
  (`cnsplots`), and mean UMIs per chemistry.
- **`04_linear_barcode_DSB_cutoff_sweep_SIG02.ipynb`**: sweeps the DSB threshold used to
  call SIG02 barcodes (1-10) and records the fraction and number of singlet, doublet,
  multiplet and uncalled cells at each cutoff. It follows the approach of SIG13's
  `../02_combinatorial_screen_signalseq_SIG13/03_qc_barcode_cutoff/01_barcode_calling_comparison.ipynb`.

## Inputs

Read from `imports_stable/`:

- `imports_stable/SIG02/scanpy_outs/SIG02_full.h5mu`: SIG02 `rna` and `bc` modalities
  (`01`, `03`, `04`). `bc.X` holds DSB-normalized counts and `bc.layers['counts']` holds
  raw UMIs.
- `imports_stable/SIG03/scanpy_outs/SIG03_oBCDirect_full.h5mu`: SIG03 `rna` and `bc`
  modalities (`02`, `03`).

## Outputs

Written under `analysis_outs/01_barcode_comparison_signalseq_SIG02_SIG03/`, which is
gitignored:

- top level (`01`, `03`, `04`): joint, violin, scatter and ridge plots,
  `ligand_umi_summary_lib*.csv`, and `barcode_calling_rate_vs_cutoff_SIG02.csv` (with its
  plot in `plots/`).
- `01_barcode_analysis_oBCDirect/` (`02`): histograms, BC4 vs. BC5 density plots,
  `umi_cutoff_auc_metrics_{BC4,BC5}_{6h,22h}.csv`, `roc_curves.pdf` and `pr_curves.pdf`.

## Running

The files do not depend on each other and can be run in any order. Run the notebooks in
the `scanpy_standard2` env. `03` also needs `cnsplots`, which is
included (pip) in `environments/scanpy_standard2.yaml`. Knit `02` in the `R-signalseq` env. It sources
`functions/r_custom/plotting_fxns.R` and `functions/r_custom/scRNA_seq_analysis_functions.R`
from this repo. Each file finds the repo root by searching upward for `imports_stable/`,
so run them from inside the repo.
