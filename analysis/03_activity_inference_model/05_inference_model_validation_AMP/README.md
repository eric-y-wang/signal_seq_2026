# AMP 2023 Validation: RA Synovial CD4 Subsets vs In Vitro and In Vivo Signatures

The inference model predicts specific ligand activities in the Tph/Tfh CD4 subsets
(T-7 IFNG+, T-3 IFNG-) of the `T + F` CTAP in the AMP 2023 rheumatoid arthritis (RA)
synovium atlas. This folder tests that prediction three ways:

1. **Single cell.** Are the pseudobulk activity shifts visible cell by cell, and are
   the matching receptors expressed?
2. **Disease -> in vitro.** Do RA-vs-OA signatures of T-7, T-3 and T-4 score highest
   on the SIG26 ligand conditions the model assigns to them?
3. **In vivo -> disease.** Is the SIG19 mouse cluster 3 signature highest in AMP T-7?

All notebooks use `T + F` CTAP RA biopsies plus all osteoarthritis (OA) controls
(`CTAP = "control"`). Gene sets are signed (`weight = sign(logFC)`, top `TOP_N` genes
at `padj < 0.01`) and scored with decoupler ULM.

## Notebooks

- **`01_activity_sc_visualization.ipynb`**: per-cell activity scores from
  `../04_inference_model_disease_sc` in `T + F` vs control cells for T-7, T-3 and T-4
  (Mann-Whitney), and donor-pseudobulk receptor expression in T-3 vs T-7 (paired
  Wilcoxon, BH).
- **`02_deg_gene_sets_sc.ipynb`**: builds RA-vs-OA signatures for T-7, T-3 and T-4 and
  scores them on SIG26-24h with ULM (vs `none_none`) and GSEA (`gseapy.prerank` on
  SIG26 DESeq2 `stat`).
- **`03_sig19_ulm_unique_sets.ipynb`**: SIG19 `leiden_0.5` one-vs-rest signatures,
  mapped mouse -> human, top 100 genes per cluster with genes shared between clusters
  removed. Scored with ULM on donor x subset AMP pseudobulk, centred per donor, and
  tested with Welch's t-test (T-7 vs each other subset) and Tukey HSD.

Run the notebooks in the `scanpy_standard2` env. They are CPU-only. 01 needs the
activity scores from `../04_inference_model_disease_sc`. 02 and 03 are independent.

## Inputs

All inputs are read from `imports_stable/`:

- AMP 2023 CD4 cells (`amp_2023_cd4_processed.h5ad`, `log1p_norm` layer) and CTAP
  labels per biopsy (`2023_AMP2_CTAP.csv`), in `external/AMP_2023/`. These external files are not
  included in `imports_stable/` (not generated in this study); see the main `README.md` ("Running the repository", step 2)
- `SIG13/analysis_outs/inference_model_disease_sc/activity_scores_amp_2023.parquet`
  (01): stable copy of the `../04_inference_model_disease_sc` output
- SIG26-24h normalized counts and per-condition DESeq2 results, in
  `SIG26/processing_outs/` and `SIG26/analysis_outs/` (02)
- SIG19 `leiden_0.5` one-vs-rest pyDESeq2 contrasts, in
  `SIG19/analysis_outs/cluster_annotation/` (03)
- the BioMart ortholog snapshot
  `SIG13/analysis_outs/inference_model_validation_AMP/mouse_human_orthologs_biomart.csv`
  (03). If it is missing, 03 queries BioMart and writes the map to the output folder.

## Outputs

Written to `analysis_outs/03_activity_inference_model/inference_model_validation_AMP/`:

- **01**: `sc_viz_TF_{T7,T3,T4}_cells.pdf`, `pseudobulk_receptor_T3_T7_TFonly*.pdf`
- **02**: `signature_gene_sets_sc.csv`, `signature_gene_set_summary_sc.csv`,
  `enrichment_sig26.csv`, `ulm_enrichment_sig26_replicates.csv`, `gsea_sig26.csv`,
  and figures
- **03**: files prefixed `sig19_ulm_top100_unique_*` (gene sets, ULM and test tables,
  figures), plus a copy of the ortholog map `mouse_human_orthologs_biomart.csv`
