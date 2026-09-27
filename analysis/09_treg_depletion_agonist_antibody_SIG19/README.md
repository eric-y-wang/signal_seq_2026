# Agonist Antibody Experiment (SIG19, Foxp3-DTR)

Single-cell analysis of CD4 T cells from the inguinal lymph node (iLN) of Treg-depleted
(Foxp3-DTR) mice given antibody treatments that deliver TNF-family, IL4 or TGF-beta
signals, alone and in combination (Figure 5).

| group | signal |
|---|---|
| `isotype` | control |
| `DTA1` | TNF-family |
| `GK1.5-IL4` | IL4 |
| `GK1.5-BGo` | TGF-beta |
| `DTA1-IL4` | TNF-family + IL4 |
| `DTA1-BGo` | TNF-family + TGF-beta |

The single and combined treatments form a 2x2 design, so each combination can be
tested for synergy: `DTA1-X - DTA1 - GK1.5-X`, with each treatment's effect taken
relative to isotype.

## Pipeline

- **`01_scvi_model_processing.py`**: trains an scVI model on the iLN CD4 T cells
  (3,000 HVGs, `counts` layer), with cage as a categorical covariate and
  `pct_counts_mt`, `S_score` and `G2M_score` as continuous covariates. Computes a UMAP
  and Leiden clusters (resolutions 0.5, 0.75, 1.0) on the latent space and writes
  `analysis_outs/09_treg_depletion_agonist_antibody_SIG19/scvi_outs/SIG19_DTR_CD4T_iLN_scvi.h5ad`. The notebooks read its stable copy,
  `imports_stable/SIG19/scvi_outs/SIG19_DTR_CD4T_iLN_scvi.h5ad`.
- **`02_cluster_annotation.ipynb`**: annotates the `leiden_0.5` clusters.
  - UMAP and marker-gene dotplots and matrixplots.
  - Gata3 vs. Ikzf2 (Helios) co-expression in clusters 1 and 3 of isotype mice.
  - Pseudobulk pyDESeq2 in isotype mice, paired by mouse (`~ mouse + condition`):
    each cluster vs. the rest, and cluster 3 vs. cluster 1.
- **`03_milo_iLN.ipynb`**: Milo differential abundance testing (`pertpy`) on
  neighborhoods built from the scVI latent space and counted per mouse. Each
  treatment is compared with isotype (`~ cage + treatment`, edgeR). Two synergy
  contrasts are also tested: `DTA1-BGo - DTA1 - GK1.5-BGo` and
  `DTA1-IL4 - DTA1 - GK1.5-IL4`.
- **`04_palantir_cellrank_iLN.ipynb`**: trajectory analysis, with Tregs (cluster 6)
  removed.
  - Palantir pseudotime and fate trajectories, starting from cluster 0.
  - CellRank (pseudotime kernel, GPCCA) macrostates, terminal states and fate
    probabilities.
  - Tests whether cluster 2 is a transitory state leading to clusters 1 and 3.
  - Gene trends along pseudotime for Ikzf2 and Gata3, and for signaling activity
    scores from the inference model.

## Inputs

Read from `imports_stable/`:

- `imports_stable/SIG19/scanpy_outs/SIG19_DTR_CD4T_iLN_subset.h5ad`: iLN CD4 T cells
  (input to 01).
- `imports_stable/SIG19/scvi_outs/SIG19_DTR_CD4T_iLN_scvi.h5ad`: stable copy of 01's output
  (02, 03, 04).
- `imports_stable/SIG13/analysis_outs/inference_model_disease_sc/activity_scores_sig19_iln.parquet`:
  per-cell activity scores from
  `../03_activity_inference_model/04_inference_model_disease_sc` (04).

## Outputs

Written under `analysis_outs/09_treg_depletion_agonist_antibody_SIG19/` (gitignored):

- `cluster_annotation/` (02): figures and pyDESeq2 tables, including
  `leiden0.5_cluster{c}_vs_rest_pydeseq2.csv`, which
  `../03_activity_inference_model/05_inference_model_validation_AMP` uses (from its stable
  copy in `imports_stable/SIG19/analysis_outs/cluster_annotation/`).
- `milo_manuscript/` (03): differential abundance figures.
- `cellrank_manuscript/` (04): pseudotime, trajectory, fate and gene-trend figures.

## Running

Run `01` first (`scvi_standard` env, which has `scvi-tools`; a GPU is recommended).
Then run `02`, `03` and `04` in the `scanpy_standard2` env. `04` also needs the
activity scores from `../03_activity_inference_model/04_inference_model_disease_sc` (a stable
copy is in `imports_stable/`). All scripts find the repo root by searching upward for
`imports_stable/`, so run them from inside the repo.
