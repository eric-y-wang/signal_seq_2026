# signal_seq_2026
This repository contains the analysis code, custom functions, and reproducibility environments for the 2026 Signal-seq manuscript. The `imports_stable` data directory is distributed separately as per-experiment Zenodo records (see `imports_stable/README.md` for the list): **will be made available upon publication**  

## Repository Structure

The repository is organized into three main code directories (`analysis`, `functions`, and `environments`) plus a stable data directory (`imports_stable`) for reproducing analyses.


### 1. Analysis
The `analysis/` directory contains the core workflows used to generate the results and figures in the manuscript. Each module is named `NN_<description>_<experiment ID>` and has its own `README.md` describing the design, inputs, outputs and how to run it. Files and subfolders are numbered to indicate the order of execution.

* **`01_barcode_comparison_signalseq_SIG02_SIG03/`**
    * Comparison of linear (SIG02) vs. circular (SIG03) RNA barcode designs for Signal-seq VLPs, by how well each barcode is captured in single-cell data.
* **`02_combinatorial_screen_signalseq_SIG13/`**
    * Analysis of 648 condition combinatorial Signal-seq screen (Figure 2; 660 barcode-called conditions minus the 12 homotypic double-dose conditions).
    * **Preprocessing:** Cell Ranger processing, z-scored DEGs, RNA/barcode QC, inter-replicate QC, validation against SIG07 and recombinant-protein (SIG14) stimulations, and barcode-calling robustness.
    * **Differential Expression (glmGamPoi):**
        * Implements a custom pipeline using `glmGamPoi` and `future.batchtools` for high-throughput interaction testing on HPC systems.
        * **Interaction Model:** Tests for synergistic/antagonistic effects using a 2x2 factorial design ($Ligand1 * Ligand2$).
        * **Single Term Model:** Standard one-vs-reference testing.
        * Interaction scoring and classification, with an empirical null built from linker-only (non-targeting) cells.
    * **Sparse PCA (sPCA):** Running sPCA on z-scored DEGs to identify GEPs, scoring GEPs, linear modeling to connect ligands to GEP effects, plus null, stability and coherence checks.
    * **Downstream:** Single ligand analysis, gene-level interaction analysis, HDBSCAN clustering of interaction effects, and signal transduction reinforcement testing.
* **`03_activity_inference_model/`**
    * Ridge regression model, built on the SIG13 sPCA programs, that infers ligand signaling activity from gene expression.
    * **Workflows:** Explanatory matrix construction, model calibration and ROC testing, mixture validation, and application to bulk and single-cell disease datasets (including AMP validation).
* **`04_tnf_interaction_ATACseq_SIG16/`**
    * Bulk ATAC-seq of CD4 T cells stimulated with IL4, IL6 or TGFb, with or without TNF.
    * **Workflows:** DESeq2 on the IDR peak atlas, differential accessibility, classification of ligand x TNF interaction effects per peak (compared with SIG13 and SIG18 RNA), and FIMO motif enrichment.
* **`05_in_vitro_differentiation_RNAseq_SIG18/`**
    * Bulk RNA-seq of CD4 T cells differentiated under Th-polarizing conditions, with or without TNF.
    * **Workflows:** DESeq2 differential expression and interaction scoring, and sPCA to derive differentiation gene expression programs (dGEPs).
* **`06_crispr_screen_tgfb_tnf_gata3_SIG17/`**
    * Pooled CRISPR knockout screen for regulators of GATA3 under TGFb + TNF.
    * **Workflows:** UMI deduplication and `MAGeCK` counting/testing (HPC), then hit analysis compared with the SIG18 TGFb + TNF interaction.
    * Raw FASTQs are not included in `imports_stable/`; they are available on GEO ([GSE348675](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE348675)).
* **`07_crispr_arrayed_validation_RNAseq_SIG30/`**
    * Arrayed CRISPR knockout bulk RNA-seq of 9 target genes plus a control guide.
    * **Workflows:** DESeq2 differential expression, projection onto the SIG18 dGEPs, and receptor cross-regulation.
* **`08_tnf_family_tgfb_interaction_RNAseq_SIG29/`**
    * Bulk RNA-seq of TGFb combined with TNF-family ligands (TNF, TL1A, OX40L, DTA1).
    * **Workflows:** DESeq2 differential expression and interaction scoring, and projection onto the SIG18 dGEPs.
* **`09_treg_depletion_agonist_antibody_SIG19/`**
    * Analysis of Foxp3-DTR agonist antibody experiment (Figure 5).
    * **Workflows:** `scVI` model processing, Leiden clustering and cluster annotation, `Milo` differential abundance testing, and `Palantir`/`CellRank` trajectory analysis.

### 2. Functions
Custom libraries and helper functions used across the analysis notebooks.

* **`functions/perturbseq/`**: Python modules for Perturb-seq specific tasks, including multi-threaded Z-score calculation.
* **`functions/scanpy_custom/`**: Extensions for `scanpy`, including custom dotplots, DSB normalization and ligand activity scoring.
* **`functions/r_custom/`**: R scripts for plotting and scRNA-seq analysis utilities.
* **`functions/cell_cycle_mouse_cc2019_seurat.csv`**: Mouse cell-cycle gene list.

### 3. Environments
Files to reproduce the computational environments used in this study. Each module's README lists which environment each script uses.

* **`environments/scanpy_standard2.yaml`**: Conda environment file containing Python dependencies, used by most Python notebooks.
* **`environments/scanpy_standard.yaml`**: Conda environment file containing Python dependencies, used for the sPCA fits and scoring, and activity inference model calibration and bulk scoring.
* **`environments/scvi_standard.yaml`**: Conda environment file containing `scvi-tools`, for the SIG19 scVI model step.
* **`environments/rapids_singlecell.yaml`**: Conda environment file containing `rapids_singlecell` and CUDA 13, for GPU single-cell activity scoring.
* **`environments/R_signalseq.yaml`**: Conda environment file containing R 4.5.3 and dependencies; runs every R Markdown document in the repo except SIG13 `07_spca/04_spca_visualization.Rmd`.
* **`environments/R_deseq2.yaml`**: Conda environment file containing R 4.4.2, `glmGamPoi`, `future.batchtools` and `reticulate`/`anndata`, for the HPC glmGamPoi pipelines and SIG13 `07_spca/04_spca_visualization.Rmd`.
* **`environments/renv.lock`**: Older renv lockfile (R 4.4.2), kept for reference; it does not include DESeq2 or IHW.
* **`environments/fastq_processing.yaml`**: Conda environment file containing `umi_tools` and `cutadapt`, for CRISPR screen UMI deduplication.
* **`environments/mageck.yaml`**: Conda environment file containing `mageck`, for CRISPR screen counting and testing.

### 4. Data
Scripts use no absolute paths. Each one finds the repository root by searching upward for `imports_stable/`, so notebooks, R Markdown documents and `sbatch` jobs must be run from inside the repository.

* **`imports_stable/`**: Frozen copies of the input files the scripts read (~207 GB), grouped by experiment (`SIG13/`, `SIG18/`, ...). This includes data generated in this study and the outputs of earlier pipeline steps, so each step can be run on its own. Public datasets not generated in this study (used by `03_activity_inference_model`) are not included and must be added separately. The folder is not tracked in git and is downloaded separately (see [Running the repository](#running-the-repository)).
* **`analysis_outs/`**: Outputs written by the scripts, one subfolder per analysis module (not tracked in git).

## Running the repository

The code in this repository runs against the `imports_stable/` data folder, which is downloaded separately.

1. **Get the data.** `imports_stable/` (~207 GB total) is distributed as separate Zenodo records, one per experiment (or, for the large SIG13 experiment, one per subfolder). Download the record(s) covering the modules you plan to run, unzip each into the root of the cloned repository, so that the layout is `signal_seq_2026/imports_stable/SIG13/`, `signal_seq_2026/imports_stable/SIG18/`, and so on. A symbolic link named `imports_stable` pointing to a copy stored elsewhere also works. See `imports_stable/README.md` for the full list of folders and their Zenodo records, and the layout [below](#imports_stable-layout).
2. **Add the data that is not included, if needed.**
    * The public datasets used for disease scoring and AMP validation in `03_activity_inference_model` (AMP 2023, Inflammation Atlas, Thomas IBD) were not generated in this study. Obtain the processed CD4 T cell objects and place them at the paths the scripts expect:
        * `imports_stable/external/AMP_2023/amp_2023_cd4_processed.h5ad` and `imports_stable/external/AMP_2023/2023_AMP2_CTAP.csv` (AMP 2023 rheumatoid arthritis synovium)
        * `imports_stable/external/inflammation_atlas_2026/inflammation_atlas_cd4_subset.h5ad` (Inflammation Atlas)
        * `imports_stable/external/thomas_IBD_2024/thomas_IBD_2024_cd4tcells_processed.h5ad` (Thomas et al. 2024, IBD)

      The scripts read the `log1p_norm` layer, `var` and the `obs` columns listed in each script's dataset config. The activity scores computed from these datasets are included (`SIG13/analysis_outs/inference_model_disease_bulk/`, `SIG13/analysis_outs/inference_model_disease_sc/`), so the downstream regression and visualization steps run without them.
    * The raw SIG17 CRISPR screen FASTQs, needed only for the first step of `06_crispr_screen_tgfb_tnf_gata3_SIG17/01_processing`, are on GEO ([GSE348675](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE348675)). Place them in `imports_stable/SIG17/raw_fastq/merged/`.
    * The SIG13 cutoff-sweep h5ads (`SIG13/scanpy_outs/cutoff_sweep/`, ~74 GB) are not included because of their size. `02_combinatorial_screen_signalseq_SIG13/03_qc_barcode_cutoff/03_generate_cutoff_datasets.ipynb` regenerates them into `analysis_outs/`; their downstream outputs are included.
3. **Create the environments.** Create the conda environments in `environments/` that you need (for example, `conda env create -f environments/scanpy_standard2.yaml`). Each module's `README.md` lists which environment each notebook or script uses.
4. **Run from inside the repository.** Scripts use no absolute paths: each one finds the repository root by searching upward for `imports_stable/`, so open notebooks, knit R Markdown documents and submit `sbatch` jobs from inside the repository. Every script reads its inputs from `imports_stable/` and writes its outputs to `analysis_outs/<analysis module>/` (not tracked in git). Because `imports_stable/` includes the outputs of earlier pipeline steps, each notebook or script can be run on its own without re-running the steps before it.

### `imports_stable/` layout

Files are grouped by the experiment or source they came from:

| Folder | Contents |
|---|---|
| `SIG13/cellranger/` | Cell Ranger `per_sample_outs` files (filtered/raw h5, protospacer calls) |
| `SIG13/scanpy_outs/` | processed h5mu/h5ad objects and z-score h5ads (cutoff-sweep h5ads not included, see step 2) |
| `SIG13/analysis_outs/` | glmGamPoi results, sPCA, clustering, QC and activity-model outputs |
| `SIG13/analysis_outs_zumpano/` | ligand signature validation and GSEA tables |
| `SIG07/analysis_outs/` | SIG07 glmGamPoi results (inter-assay comparison) |
| `SIG02/`, `SIG03/` | barcode comparison h5mu objects |
| `SIG14/`, `SIG18/`, `SIG26/`, `SIG29/`, `SIG30/` | bulk RNA-seq `processing_outs` and `analysis_outs` |
| `SIG16/` | bulk ATAC-seq peak-atlas counts, ChIPseeker annotation, FIMO motif matrix, and DESeq2 / interaction outputs of the ATAC steps |
| `SIG17/` | CRISPR screen dedup pipeline outputs (raw FASTQs not included, see step 2) |
| `SIG19/` | Treg-depletion h5ads and cluster DE tables |

## Figure panels

Each figure panel generated in this repo, the script that makes it, and the output file corresponding to the figure.

- **Script** paths are relative to `analysis/`.
- **Output file** is the file name the script saves, including any subfolder written in the save call, under `analysis_outs/<analysis module folder>/`. A `{}` placeholder shows the value used for the figure (e.g. `{FILTER}` = `0.1filter`).
- Consecutive panels made by the same script share a row, and "same as" points to the row where that script is first listed.

### Main figures

#### Figure 1

| Panel | Script | Output file |
|---|---|---|
| I | `01_barcode_comparison_signalseq_SIG02_SIG03/03_linear_vs_circular_counts_SIG02_SIG03.ipynb` | `SIG02_vs_SIG03_ridgeplot_oBC_total.pdf` |
| K–M | `01_barcode_comparison_signalseq_SIG02_SIG03/02_circular_barcode_signalseq_SIG03.Rmd` | K: `barcode_histogram_CD4_overlay.pdf` (6h panels)<br>L: `roc_curves.pdf` (6h panels)<br>M: `bc4_bc5_6h_density_plot.pdf` |

#### Figure 2

| Panel | Script | Output file |
|---|---|---|
| B–D | `02_combinatorial_screen_signalseq_SIG13/12_gene_level_analysis/01_interaction_deg_manuscript_viz.Rmd` | B: `plots/most_sig_interaction_per_gene.pdf`<br>C: `plots/class_frequency_per_interaction_barplots.pdf`<br>D: `plots/interactions_synergistic_vs_deg_dotplot.pdf` |
| E | `02_combinatorial_screen_signalseq_SIG13/13_clustering/06_coefficient_corr_heatmap_unique.ipynb` | `coefficient_corr_heatmap_interactions_clustered_unique.pdf` |
| F–J | `02_combinatorial_screen_signalseq_SIG13/07_spca/04_spca_visualization.Rmd` | F: `plots/spca_gene_hm/comp_50_IL4_TNF.pdf`<br>G: `plots/spca_gene_hm/comp_16_IFNG_IL27_TNF.pdf`<br>H: `plots/spca_gene_hm/comp_31_IL21_CCL19_21A.pdf`<br>I: `plots/spca_gene_hm/comp_4_IL6_IL2.pdf`<br>J: `plots/spca_alpha1.0_single_v_comb_plot.pdf` |
| L | `02_combinatorial_screen_signalseq_SIG13/14_signal_trans_reinforcement/02_reinforcement_synergy_testing.ipynb` | `boxplots_combined_{FILTER}.pdf` (`0.1filter`) |
| M, N | `02_combinatorial_screen_signalseq_SIG13/14_signal_trans_reinforcement/03_representative_example_viz.Rmd` | M: `signal_reinforcement_lfc_mean/figures/IL21_CCL_receptor_dotplot.pdf`<br>N: `signal_reinforcement_lfc_mean/figures/IL21_CCL_receptor_scatter.pdf` |

#### Figure 3

| Panel | Script | Output file |
|---|---|---|
| D | `03_activity_inference_model/01_inference_model_construction_validation/04_calibration_model_testing.Rmd` | `plots/inference_model_calibration/model_testing_ROC_joint_SIG14_SIG26_weighted_facet_interaction.pdf` |

#### Figure 4

| Panel | Script | Output file |
|---|---|---|
| B, C | `02_combinatorial_screen_signalseq_SIG13/12_gene_level_analysis/04_tgfb_tnf_interaction.Rmd` | B: `plots/tgfb_tnf_family_unique_class_percent_barplot.pdf`<br>C: `plots/tgfb_tnf_unique_synergy_gene_examples.pdf` |
| D | `07_crispr_arrayed_validation_RNAseq_SIG30/03_program_crossreg/Th_receptor_crossregulation.Rmd` | `program_crossreg/crossreg_6h_subset.pdf` |
| E | `04_tnf_interaction_ATACseq_SIG16/03_interaction_classification.Rmd` | `{dir.itxn}/bar-interaction-classification.pdf` (TGFβ × TNF) |
| F | `04_tnf_interaction_ATACseq_SIG16/04_motif_enrichment.Rmd` | `barplots-top10-motifs-padj-0.05.pdf` |
| G | same as E | `heatmap-synergy-positive-all-conditions.pdf` |
| J, K | `05_in_vitro_differentiation_RNAseq_SIG18/02_spca/04_spca_pos_synergy_viz.Rmd` | J: `plots/SIG18_spca_tgfb_tnf_coefficient_scatter.pdf`<br>K: `plots/SIG18_spca_tgfb_tnf_interaction_programs_replicates_Th_only.pdf` |
| L | `05_in_vitro_differentiation_RNAseq_SIG18/02_spca/05_spca_subset_gene_viz.Rmd` | `plots/SIG18_comp_38_hm_subset.pdf`, `plots/SIG18_comp_54_hm_subset.pdf` |
| P, Q | `06_crispr_screen_tgfb_tnf_gata3_SIG17/02_mageck/02_tgfb_tnf_gata3_mageckTest.Rmd` | P: `crispr_screen_tgfb_tnf_gata3/gata3_4_vs_1_lfc_vs_SIG18_TGFb_TNF_interaction.pdf`<br>Q: `crispr_screen_tgfb_tnf_gata3/gata3_4_vs_1_sgRNA_lfc_density_dose_noLegend_test.pdf` |
| U | `07_crispr_arrayed_validation_RNAseq_SIG30/02_SIG18_dGEP_projection/02_dGEP_analysis_SIG30_waggr.Rmd` | `scoreHeatmap_classEstimate_SIG30_waggr.pdf` |
| V | same as D | `program_crossreg/crossreg_hm_crispr_perturbations_all_SIG29.pdf` |

#### Figure 5

| Panel | Script | Output file |
|---|---|---|
| B, C | `09_treg_depletion_agonist_antibody_SIG19/02_cluster_annotation.ipynb` | B: `SIG19_DTR_iLN_scvi_umap.pdf` / `.png`<br>C: `leiden0.5_manually_curated_dotplot.pdf` |
| D–G | `09_treg_depletion_agonist_antibody_SIG19/04_palantir_cellrank_iLN.ipynb` | D, E: `SIG19_DTR_iLN_palantir_pseudotime_umap.pdf` / `.png` (both panels in one file)<br>F: `SIG19_DTR_iLN_palantir_trajectories_umap.pdf` / `.png`<br>G: `SIG19_DTR_iLN_palantir_trajectory_binned_cluster_freq.pdf` |
| H | `03_activity_inference_model/03_inference_model_disease_bulk/11_viz_sig19.Rmd` | `sig19_BGo_arm_by_cluster.pdf` |
| I–K | `09_treg_depletion_agonist_antibody_SIG19/03_milo_iLN.ipynb` | I: `SIG19_DTR_iLN_DA_BGo_nhoods.pdf`<br>J: `SIG19_DTR_iLN_DA_synergy_beeswarm.pdf` (left panel)<br>K: `SIG19_DTR_iLN_DA_BGo_individual_clusters.pdf` |
| L, M | same as B, C | L: `SIG19_DTR_iLN_Gata3_vs_Ikzf2_kde_cluster1_3.pdf`<br>M: `SIG19_DTR_iLN_Gata3_Ikzf2_doublepos_freq_boxplot_by_mouse.pdf` |
| N | same as D–G | `SIG19_DTR_iLN_cellrank_gene_trends_activity_scores.pdf` |

#### Figure 6

| Panel | Script | Output file |
|---|---|---|
| C | `03_activity_inference_model/03_inference_model_disease_bulk/09_viz_amp_2023.Rmd` | `amp_tph_comparison_TF_CTAP.pdf` |
| D | `03_activity_inference_model/05_inference_model_validation_AMP/01_activity_sc_visualization.ipynb` | `inference_model_validation_AMP/pseudobulk_receptor_T3_T7_TFonly_joint_boxplots.pdf` (per-receptor version with test brackets: `pseudobulk_receptor_T3_T7_TFonly.pdf`) |
| F | `03_activity_inference_model/05_inference_model_validation_AMP/02_deg_gene_sets_sc.ipynb` | `ulm_replicates_barplot_by_condition.pdf` (subset of conditions; full plot in ED Fig. 19G) |
| H | `03_activity_inference_model/05_inference_model_validation_AMP/03_sig19_ulm_unique_sets.ipynb` | `sig19_{TAG}_trio_boxplot_tukey.pdf` (`TAG` = `ulm_top100_unique`) |

### Extended Data figures

#### Extended Data Figure 1

| Panel | Script | Output file |
|---|---|---|
| H | `01_barcode_comparison_signalseq_SIG02_SIG03/01_linear_barcode_viz_SIG02.ipynb` | `violin_oBC_umi_by_ligand_lib_{tp}.pdf` (`tp` = `6h`) |
| L–N | `01_barcode_comparison_signalseq_SIG02_SIG03/02_circular_barcode_signalseq_SIG03.Rmd` | L: `barcode_histogram_CD4_overlay.pdf` (22h panels)<br>M: `roc_curves.pdf` (22h panels)<br>N: `bc4_bc5_22h_density_plot.pdf` |

#### Extended Data Figure 2

| Panel | Script | Output file |
|---|---|---|
| H, I | `02_combinatorial_screen_signalseq_SIG13/02_qc_general/01_rna_and_perturbation_counts.ipynb` | H: `plots/qc/rna_quality_metrics.pdf`<br>I: `plots/qc/barcode_calling_DSB7_callCounts_histogram.pdf` |

#### Extended Data Figure 3

| Panel | Script | Output file |
|---|---|---|
| A | `02_combinatorial_screen_signalseq_SIG13/07_spca/03_spca_annotation.Rmd` | `plots/spca_replicate_association_dotplot.pdf` |
| B | `02_combinatorial_screen_signalseq_SIG13/09_spca_stability/04_cell_bootstrap_viz.ipynb` | `bootstrap_stability_best_available_box.pdf` |
| C | `02_combinatorial_screen_signalseq_SIG13/09_spca_stability/02_alpha_sweep_viz.ipynb` | `alpha_sweep_similarity_vs_alpha_best_available.pdf` |
| E–H | `02_combinatorial_screen_signalseq_SIG13/10_spca_coherence/01_program_coherence_simple.ipynb` | E: `coherence_by_interaction_class.pdf`<br>F: `coherence_vs_interaction_strength_synergy.pdf`<br>G: `synergy_lowest_r_in_cells.pdf`<br>H: `synergy_highest_r_in_cells.pdf` |

#### Extended Data Figure 4

| Panel | Script | Output file |
|---|---|---|
| A–F | `02_combinatorial_screen_signalseq_SIG13/03_qc_barcode_cutoff/01_barcode_calling_comparison.ipynb` | A: `barcode_calling_representative_histograms.pdf`<br>B: `barcode_calling_rate_vs_cutoff_combined.pdf`<br>C: `barcode_spread_called_vs_cutoff.pdf`<br>D: `barcode_score_vs_library_size.pdf`<br>E: `barcode_position_bias_boxplot.pdf`<br>F: `dsb_by_ligand_boxplot.pdf` |
| G | `02_combinatorial_screen_signalseq_SIG13/03_qc_barcode_cutoff/02_barcode_calling_visualization.ipynb` | `barcode_calling_DSB7_ligand1.png`, `barcode_calling_DSB7_ligand2.png` (`.pdf` also saved) |
| I, J | `02_combinatorial_screen_signalseq_SIG13/03_qc_barcode_cutoff/05_glm_coefficient_concordance_viz.Rmd` | I: `glm_coefficient_concordance_cutoffs.pdf`<br>J: `glm_concordance_vs_dataset_size.pdf` |
| K | `02_combinatorial_screen_signalseq_SIG13/03_qc_barcode_cutoff/07_spca_score_concordance.Rmd` | `spca_score_concordance_cutoffs.pdf` |
| M | `02_combinatorial_screen_signalseq_SIG13/04_qc_barcode_counts/02_glm_coefficient_concordance_viz.Rmd` | `plots/qc_barcode_counts/correlation_diff_barcode_cutoffs.pdf` |
| N | `02_combinatorial_screen_signalseq_SIG13/04_qc_barcode_counts/04_spca_score_concordance.Rmd` | `spca_score_correlation_boxplots.pdf` |

#### Extended Data Figure 5

| Panel | Script | Output file |
|---|---|---|
| A–D | `02_combinatorial_screen_signalseq_SIG13/08_spca_null/02_spca_null_diagnostics.Rmd` | A, C: `plots/spca_null/spca_pval_histogram_real_vs_null.pdf`<br>B, D: `plots/spca_null/spca_pval_vs_effect_scatter.pdf` / `.png` |
| E–H | `02_combinatorial_screen_signalseq_SIG13/06_interaction_scoring_null/03_interaction_scoring_null_diagnostics.Rmd` | E, G: `plots/interaction_scoring_null/glm_pval_histogram_real_vs_null.pdf`<br>F, H: `plots/interaction_scoring_null/glm_pval_vs_effect_scatter.pdf` / `.png` |

#### Extended Data Figure 6

| Panel | Script | Output file |
|---|---|---|
| A–C | `02_combinatorial_screen_signalseq_SIG13/02_qc_general/02_intra_assay_correlation.ipynb` | A, B: `plots/qc/replicate_corr_all_deg.pdf`<br>C: `plots/qc/orientation_corr_all_deg.pdf` |
| D | `02_combinatorial_screen_signalseq_SIG13/02_qc_general/03_inter_assay_correlation_SIG07_SIG13.Rmd` | `plots/qc/SIG07_SIG13_pearson_corr_single_term.pdf` |
| F, G | `02_combinatorial_screen_signalseq_SIG13/02_qc_general/04_recomb_protein_correlation_SIG14_SIG13.Rmd` | F: `plots/qc/SIG13_SIG14_fc_total_lfc_corr_violin.pdf`<br>G: `plots/qc/SIG13_SIG14_fc_interaction_corr_plots.pdf` |

#### Extended Data Figure 7

| Panel | Script | Output file |
|---|---|---|
| (single panel) | `02_combinatorial_screen_signalseq_SIG13/12_gene_level_analysis/02_dose_saturation_analysis.Rmd` | `plots/qc/dose_saturation_plots.pdf` |

#### Extended Data Figure 8

| Panel | Script | Output file |
|---|---|---|
| A | `02_combinatorial_screen_signalseq_SIG13/11_single_ligand/01_single_ligand_signature_validation.rmd` | `plots/gsea_res.pdf` |
| B | `02_combinatorial_screen_signalseq_SIG13/11_single_ligand/02_single_ligand_analysis.ipynb` | `plots/linkerOnly_nDEGs_barplot.pdf` |
| C | `02_combinatorial_screen_signalseq_SIG13/13_clustering/05_coefficient_corr_heatmap.ipynb` | `coefficient_corr_heatmap_singleLigand_full.pdf` |
| D–F | `02_combinatorial_screen_signalseq_SIG13/13_clustering/02_hdbscan_bootstrap_singleLigand.ipynb` | D: `bootstrap_ari_ngenes_singleLigand_minSamples1.pdf`<br>E: `bootstrap_ari_filters_singleLigand_minSamples1.pdf`<br>F: `bootstrap_ari_ref_params_singleLigand_minSamples1.pdf` |

#### Extended Data Figure 9

| Panel | Script | Output file |
|---|---|---|
| A | `02_combinatorial_screen_signalseq_SIG13/12_gene_level_analysis/01_interaction_deg_manuscript_viz.Rmd` | `plots/Th_diff_example_synergies_full_plots.pdf` |
| B | `02_combinatorial_screen_signalseq_SIG13/12_gene_level_analysis/03_interaction_vis_circos.Rmd` | not saved to a file (drawn inline; see the rendered `.html`) |
| C | `02_combinatorial_screen_signalseq_SIG13/06_interaction_scoring_null/04_synergy_buffering_magnitude_test.Rmd` | `05_rate_matched_abs_additive_ecdf.pdf` |
| D | same as A | `plots/class_frequency_boxplots.pdf` |

#### Extended Data Figure 10

| Panel | Script | Output file |
|---|---|---|
| A–C | `02_combinatorial_screen_signalseq_SIG13/13_clustering/01_hdbscan_bootstrap_interactions.ipynb` | A: `bootstrap_ari_ngenes_minSamples1.pdf`<br>B: `bootstrap_ari_filters_minSamples1.pdf`<br>C: `bootstrap_ari_ref_params_minSamples1.pdf` |
| D | `02_combinatorial_screen_signalseq_SIG13/13_clustering/06_coefficient_corr_heatmap_unique.ipynb` | `coefficient_corr_heatmap_interactions_full_unique.pdf` |

#### Extended Data Figure 11

| Panel | Script | Output file |
|---|---|---|
| A | `02_combinatorial_screen_signalseq_SIG13/07_spca/04_spca_visualization.Rmd` | `plots/spca_alpha1.0_single_consistent_combined_hm.pdf` |
| B | `02_combinatorial_screen_signalseq_SIG13/07_spca/05_spca_ora_analysis.Rmd` | `plots/go_enrichment_zscore_degs_alpha1.0_sPCA.pdf` |

#### Extended Data Figure 12

`SIG14` = mouse and `SIG26` = human ground-truth stimulations.

| Panel | Script | Output file |
|---|---|---|
| C–F | `03_activity_inference_model/01_inference_model_construction_validation/04_calibration_model_testing.Rmd` | C: `plots/inference_model_calibration/model_testing_ROC_SIG14.pdf`<br>D: `plots/inference_model_calibration/model_testing_ROC_SIG26.pdf`<br>E: `plots/inference_model_calibration/model_testing_estimates_by_stimulation_scatter_labeled_SIG14.pdf`<br>F: `plots/inference_model_calibration/model_testing_estimates_by_stimulation_scatter_labeled_SIG26.pdf` |
| H–J | `03_activity_inference_model/02_inference_model_mixture_validation/02_mixture_discrimination_analysis.ipynb` | H: `01_combinatorial_activity_by_sample_type.pdf`<br>I, J: `03_per_pair_delta.pdf` |

#### Extended Data Figure 13

| Panel | Script | Output file |
|---|---|---|
| A | `05_in_vitro_differentiation_RNAseq_SIG18/01_deg/deg_analysis_SIG18.Rmd` | `plots/SIG18_PCA.pdf` |
| B | `05_in_vitro_differentiation_RNAseq_SIG18/02_spca/04_spca_pos_synergy_viz.Rmd` | `plots/SIG18_spca_tnf_family_interaction_programs.pdf` |
| C | `05_in_vitro_differentiation_RNAseq_SIG18/02_spca/05_spca_subset_gene_viz.Rmd` | `plots/SIG18_comp_{4,47,19}_hm.pdf` |
| D, E | same as B | D: `plots/spca_alpha10.0_interactions_per_combination_bar.pdf`<br>E: `plots/spca_alpha10.0_interactions_hm.pdf` |
| F | `05_in_vitro_differentiation_RNAseq_SIG18/02_spca/06_spca_ora_analysis.Rmd` | `plots/SIG18_spca_go_enrichment.pdf` |

#### Extended Data Figure 14

| Panel | Script | Output file |
|---|---|---|
| A, B | `04_tnf_interaction_ATACseq_SIG16/01_deseq2_qc.Rmd` | A: `plots_new/bar-FRiP-idr-{idr}.pdf` (`idr` = `0.05`)<br>B: `pca-name.pdf` |
| C | `04_tnf_interaction_ATACseq_SIG16/03_interaction_classification.Rmd` | `{dir.itxn}/bar-interaction-classification.pdf` (one per ligand × TNF folder) |
| D, E | `04_tnf_interaction_ATACseq_SIG16/04_motif_enrichment.Rmd` | `barplots-top10-motifs-padj-0.05.pdf` |
| F | same as C | `heatmap-interaction-classes-interactions-only.pdf` |

#### Extended Data Figure 15

| Panel | Script | Output file |
|---|---|---|
| A | `07_crispr_arrayed_validation_RNAseq_SIG30/03_program_crossreg/Th_receptor_crossregulation.Rmd` | `program_crossreg/crossreg_6h_all.pdf` |
| F, G | `08_tnf_family_tgfb_interaction_RNAseq_SIG29/01_deg/01_deg_analysis_SIG29.Rmd` | `SIG29_PCA_plots.pdf` |
| H | `08_tnf_family_tgfb_interaction_RNAseq_SIG29/01_deg/02_deg_visualization_SIG29.Rmd` | `res_interaction_fcfc_simple_SIG29.pdf` |
| I | `08_tnf_family_tgfb_interaction_RNAseq_SIG29/02_SIG18_dGEP_projection/02_dGEP_analysis_SIG29_waggr.Rmd` | `scoreHeatmap_classEstimate_SIG18vsSIG29_{score_method}.pdf` (`waggr`; left panel) |

#### Extended Data Figure 16

| Panel | Script | Output file |
|---|---|---|
| D | `06_crispr_screen_tgfb_tnf_gata3_SIG17/02_mageck/01_umi_clone_analysis.Rmd` | `crispr_screen_tgfb_tnf_gata3/clone_distribution_per_bin.pdf` |
| E, F | `06_crispr_screen_tgfb_tnf_gata3_SIG17/02_mageck/02_tgfb_tnf_gata3_mageckTest.Rmd` | E: `crispr_screen_tgfb_tnf_gata3/gata3_4_vs_1_sgRNA_volcano.pdf`<br>F: `crispr_screen_tgfb_tnf_gata3/gata3_4_vs_1_sgRNA_lfc_density_dose_test.pdf` (right panel) |
| G | `07_crispr_arrayed_validation_RNAseq_SIG30/01_deg/02_deg_visualization_SIG30.Rmd` | `target_lfc_barplots_SIG30.pdf` |
| Q | `07_crispr_arrayed_validation_RNAseq_SIG30/03_program_crossreg/Th_receptor_crossregulation.Rmd` | `program_crossreg/crossreg_hm_nr4a_ptpn_family_SIG29.pdf` |

#### Extended Data Figure 17

| Panel | Script | Output file |
|---|---|---|
| H | `03_activity_inference_model/03_inference_model_disease_bulk/11_viz_sig19.Rmd` | `sig19_treatment_bulk_volcano.pdf` |
| I, J | `09_treg_depletion_agonist_antibody_SIG19/02_cluster_annotation.ipynb` | I: `SIG19_DTR_iLN_scvi_umap.pdf`<br>J: `leiden0.5_manually_curated_dotplot.pdf` |
| K–M | `09_treg_depletion_agonist_antibody_SIG19/03_milo_iLN.ipynb` | K: `SIG19_DTR_iLN_DA_IL4_nhoods.pdf`<br>L: `SIG19_DTR_iLN_DA_synergy_beeswarm.pdf` (right panel)<br>M: `SIG19_DTR_iLN_DA_IL4_individual_clusters.pdf` |

#### Extended Data Figure 18

| Panel | Script | Output file |
|---|---|---|
| C–F | `03_activity_inference_model/03_inference_model_disease_bulk/08_viz_inflammation_atlas.Rmd` | C: `inflammation_atlas_cross_Th0.pdf`<br>D–F: `inflammation_atlas_combination_ranges_autoimmune.pdf` (all three panels in one file) |
| I–L | `03_activity_inference_model/03_inference_model_disease_bulk/10_viz_thomas_ibd.Rmd` | I, J: `ibd_pretreatment_elevated_ranges.pdf`<br>K: `ibd_celltype_non_vs_remission_grid.pdf`<br>L: `ibd_celltype_non_vs_remission.pdf` |

#### Extended Data Figure 19

| Panel | Script | Output file |
|---|---|---|
| B | `03_activity_inference_model/03_inference_model_disease_bulk/09_viz_amp_2023.Rmd` | `amp_ctap_celltype_significance_grid.pdf` |
| C | `03_activity_inference_model/05_inference_model_validation_AMP/01_activity_sc_visualization.ipynb` | `inference_model_validation_AMP/sc_viz_TF_{tag}_cells.pdf` (`tag` = `T7`; figure shows 4 of the 7 panels) |
| D–H | `03_activity_inference_model/05_inference_model_validation_AMP/02_deg_gene_sets_sc.ipynb` | D–F: `signature_comparison_foldchange_pairs_union.pdf`<br>G: `ulm_replicates_barplot_by_condition.pdf`<br>H: `gsea_running_es_{direction}.pdf` (`up`, `dn`) |
