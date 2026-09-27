# imports_stable

Frozen copies of every input file the analysis scripts read. Scripts read from here and write to `analysis_outs/<module>/`.

This includes both inputs generated in this study (Cell Ranger outputs, FASTQs, bulk RNA-seq count matrices, ATAC-seq peak data) and files produced by earlier steps of this repo's pipelines. The earlier-step outputs are included so each script can be run on its own against the exact files used in the manuscript. Re-running an upstream step writes a fresh copy to `analysis_outs/` and does not change `imports_stable/`.

This folder itself (~215 GB) is not tracked in git (see `.gitignore`); only this README is tracked. The data is distributed separately via Zenodo, one record (zip) per experiment, as described below.

## Recreating `imports_stable/`

Each experiment's inputs are archived as a separate Zenodo record. Download the record(s) you need, unzip, and place the contents directly under `imports_stable/<experiment>/` at the root of the cloned repo (so the layout is `signal_seq_2026/imports_stable/SIG13/cellranger/...`, `signal_seq_2026/imports_stable/SIG18/...`, and so on). Each zip's internal folder structure matches the destination path in the table below exactly, so it can be unzipped straight into `imports_stable/`.

You don't need all of them — only the folders read by the modules you plan to run (each module's own `README.md`, and the main repo `README.md`, list which `imports_stable/<experiment>/` paths it reads).

| Folder | Contents | Zenodo record |
|---|---|---|
| `SIG13/cellranger/` | Cell Ranger `per_sample_outs` files (filtered/raw h5, protospacer calls) | `<DOI TBD>` |
| `SIG13/scanpy_outs/` | processed h5mu/h5ad objects, z-score and cutoff-sweep h5ads | `<DOI TBD>` |
| `SIG13/analysis_outs/` | glmGamPoi results, sPCA, clustering, QC and activity-model outputs | `<DOI TBD>` |
| `SIG13/analysis_outs_zumpano/` | ligand signature validation and GSEA tables | `<DOI TBD>` |
| `SIG07/analysis_outs/` | SIG07 glmGamPoi results (inter-assay comparison) | `<DOI TBD>` |
| `SIG02/`, `SIG03/` | barcode comparison h5mu objects | `<DOI TBD>` |
| `SIG14/`, `SIG18/`, `SIG26/`, `SIG29/`, `SIG30/` | bulk RNA-seq `processing_outs` and `analysis_outs` | `<DOI TBD>` |
| `SIG16/` | bulk ATAC-seq peak-atlas counts, ChIPseeker annotation, FIMO motif matrix, and DESeq2 / interaction outputs of the ATAC steps | `<DOI TBD>` |
| `SIG17/` | CRISPR screen dedup pipeline outputs (raw FASTQs not included, see Not included) | `<DOI TBD>` |
| `SIG19/` | Treg-depletion h5ads and cluster DE tables | `<DOI TBD>` |

Links will be filled in as each record is published (see the main `README.md` for the top-level Zenodo landing page). Because `SIG13` is large (~181 GB) it is split across multiple Zenodo records by subfolder, as shown above; every other experiment is one record per folder.

A symbolic link named `imports_stable` pointing to a copy stored elsewhere (e.g. scratch space) also works in place of a real folder, as long as the same subfolder structure is kept underneath it.

## Not included

- **External datasets.** The public, non-study datasets used by `analysis/03_activity_inference_model` (disease scoring in `03` and `04`, AMP validation in `05`) are not included, because they were not generated in this study and are not part of the Zenodo deposit. To run those steps, obtain the processed CD4 T cell objects and place them at the paths the scripts expect:
  - `imports_stable/external/AMP_2023/amp_2023_cd4_processed.h5ad` and `imports_stable/external/AMP_2023/2023_AMP2_CTAP.csv` (AMP 2023 rheumatoid arthritis synovium)
  - `imports_stable/external/inflammation_atlas_2026/inflammation_atlas_cd4_subset.h5ad` (Inflammation Atlas)
  - `imports_stable/external/thomas_IBD_2024/thomas_IBD_2024_cd4tcells_processed.h5ad` (Thomas et al. 2024, IBD)

  The scripts read the `log1p_norm` layer, `var` and the `obs` columns listed in each script's dataset config. The activity scores computed from these datasets are included (`SIG13/analysis_outs/inference_model_disease_bulk/`, `SIG13/analysis_outs/inference_model_disease_sc/`), so the downstream regression and visualization steps run without them.
- **SIG13 cutoff-sweep h5ads** (`SIG13/scanpy_outs/cutoff_sweep/`, ~74 GB) are not included in the Zenodo deposit because of size limitations. `analysis/02_combinatorial_screen_signalseq_SIG13/03_qc_barcode_cutoff/03_generate_cutoff_datasets.ipynb` regenerates them into `analysis_outs/`, where steps 04 and 06 of that folder read them. Their downstream outputs (per-threshold GLM tables and sPCA scores) are included.
- **SIG17 CRISPR screen raw FASTQs** (`SIG17/raw_fastq/merged/SIG17_{1-4}_R{1,2}.fastq.gz`) are available on GEO ([GSE348675](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE348675)), not on Zenodo. The dedup pipeline outputs derived from them (`SIG17/dedup_pipeline_output/`) are still included.
