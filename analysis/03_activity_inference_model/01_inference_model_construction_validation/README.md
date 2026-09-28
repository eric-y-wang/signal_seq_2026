# Final Ligand-Activity Inference Model

Builds the SIG13 ligand-activity inference model used for the paper, and calibrates
it against two experiments with known stimulations.

Two products come out of this folder:

1. **The explanatory matrix** (`X`) -- the SIG13 activity x sPCA-component matrix that
   every downstream inference regresses against, with weak ligands removed and
   co-linear signaling activities merged into consensus activities.
2. **A model choice** -- 8 ridge variants are scored on SIG14 and SIG26 ground truth,
   and `ridge_zscore_50_weighted` wins.

The disease applications live in `../03_inference_model_disease_bulk`, which uses
the explanatory matrix built here (via `../model_core/model_core.py`) and does not
re-fit it.

## Steps

| step | what it does | where it runs |
|---|---|---|
| `01_explanatory_matrix_construction.ipynb` | activity selection, consensus clustering, explanatory matrix export | Jupyter, `scanpy_standard` |
| `02_calibration_input_scoring.py` | scores SIG14 + SIG26 pseudobulk with all 8 model variants | SLURM via 03 |
| `03_run_calibration_input_scoring.sh` | sbatch wrapper for 02 (16 cpus, 100G, 2h) | SLURM |
| `04_calibration_model_testing.Rmd` | ROC + per-condition accuracy, one model per curve | R (see below) |

```bash
cd analysis/03_activity_inference_model/01_inference_model_construction_validation
sbatch 03_run_calibration_input_scoring.sh
```

## 01: building the explanatory matrix

sPCA component scores are computed per cell with decoupler `waggr` on gene-scaled
`log1p_norm` expression, pseudobulked by ligand condition, then expressed as a
**difference from `linker_linker`** (the non-targeting control). 68 of 78 sPCA
components are used -- the other 10 are dropped upstream for being replicate-associated
(they are absent from `lm_scored_*_clean.csv`).

Activities are then filtered and merged:

| | single ligands | ligand pairs |
|---|---|---|
| reproducibility | inter-replicate Pearson > 0.25 on single-ligand LFCs | inter-replicate Pearson > 0.25 on interaction LFCs |
| effect on components | single-term `p_adj < 0.01` in the sPCA component LM | `p_adj_interaction < 0.001` |
| merging | modal HDBSCAN cluster (`analysis/02_combinatorial_screen_signalseq_SIG13/13_clustering`) | same |
| unclustered (`cluster == -1`) | kept as its own activity | dropped |

Each consensus cluster is named for its highest-row-mean member plus `_c`, so
`IL6_TNFSF18_c` is one activity standing for all the IL6/IL21 x TNF-family pairs that
clustered with it. Four HDBSCAN clusters are excluded by hand as **double-dose**
artifacts (the same signal paired with itself: TNF+TNF, IL2/4+IL2/4,
IFNG/IL27+IFNG/IL27, IL21/6+IL21/6), and `linker_TGFB1` is dropped as a bad
condition while `TGFB1_linker` is kept.

Result: **38 activities x 68 components** (11 single, 27 combinatorial).

## 02: scoring the calibration datasets

Each calibration dataset is scored the same way a disease dataset is -- `waggr`
component scores -> per-sample average -> z-score -> ridge against `X` -- for all 8
variants of the model:

| axis | options |
|---|---|
| genes per component | top 50 by loading, or all |
| loadings | used as weights, or discarded (unweighted) |
| activity value | ridge coefficients z-scored against a 1000x label-permutation null (`zscore`), or raw coefficients (`coeff`) |

`RidgeCV` is fit independently per sample (alpha `1e-1..1e3`, 5-fold CV).

| dataset | species | design |
|---|---|---|
| `SIG14` | mouse | in vitro mouse CD4 ligand pairs |
| `SIG26-6h`| human | in vitro human CD4 ligand pairs |

SIG26 is human, so the component net is mapped to human orthologs first
(`scc.convert_mouse_genes_to_human`, one human symbol per mouse gene).

## 04: choosing a model

Each condition in SIG14/SIG26 is hand-mapped to the model activity it *should* light
up (`mapping_tibble`). For every model variant, `lm(activity ~ condition)` is fit
against the `none_none` reference and each condition x activity estimate is labelled
`true_activity` or `other_activity`. Two readouts follow:

- **ROC / AUC** over the estimates, pooling all conditions -- overall sensitivity and
  specificity per model variant.
- **Per-condition percentile rank** of the true-activity estimate -- the pooled AUC
  can hide individual conditions the model misses, so within each condition every
  activity estimate is ranked and the true one should land at 1.

`ridge_zscore_50_weighted` is the selected model and gets the detailed per-condition
panels. 

## Inputs

All from `imports_stable/SIG13/analysis_outs/` (stable copies of the SIG13 screen outputs)
unless noted:

```
spca/zscore_degs_allLigands_0.1_alpha1.0_sPCA_loadings.csv     component loadings
spca/lm_scored_..._clean.csv, lm_fit_..._clean.csv             component LM results
clustering/hdbscan_bootstrap_modal_clusters[_singleLigand]_minSamples1.csv
replicate_corr/replicate_correlation_{interactionLfc,singleLfc}_0.2filter.csv
SIG13_doublets_DSB7.h5ad                    imports_stable/SIG13/scanpy_outs/
SIG14 and SIG26 counts                      imports_stable/{SIG14,SIG26}/processing_outs/
```

`SIG13_doublets_DSB7.h5ad` is not in the Zenodo deposit; download it from GEO
([GSE318270](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE318270)).

Step 02 reads the explanatory matrix from its stable copy in
`imports_stable/SIG13/analysis_outs/inference_model_final/`, and 04 reads the calibration
scores from `imports_stable/SIG13/analysis_outs/inference_model_calibration/`. Re-running
01 or 02 writes fresh copies to `analysis_outs/`, as listed below.

## Outputs

`analysis_outs/03_activity_inference_model/inference_model_final/` -- the model itself:

```
SIG13_waggr_scores_explanatory_mat.csv    38 activities x 68 components (X)
SIG13_waggr_scores_explanatory_long.csv   same, long form
SIG13_waggr_activity_clusters.csv         ligand condition -> consensus activity
```

`analysis_outs/03_activity_inference_model/inference_model_calibration/` -- the calibration:

```
activity_combined_{SIG14,SIG26-6h}.csv   sample x activity, all 8 models
r2_combined_{SIG14,SIG26-6h}.csv         per-sample ridge R2
```

Figures go to `analysis_outs/03_activity_inference_model/plots/inference_model_calibration/`. (The original directory
also contains a few `model_testing_*_SIG14.pdf` files and a
`model_testing_activity_out_SIG14.csv` written by an earlier version of 04 into
`inference_model_calibration/` itself; the current Rmd writes all figures to the plots
directory.)

## Notes

- **`SIG13_waggr_activity_clusters.csv` keeps the screen's linker scaffold in
  single-activity names** (`IL4_linker_c`, `linker_IL2_c`) while the explanatory
  matrix has it stripped (`IL4_c`, `IL2_c`), so the two files do not join directly --
  the clusters file lists 39 annotations, the matrix 38 rows. Strip `linker_|_linker`
  before joining; `03_inference_model_disease_bulk/01_activity_annotations.py` does this.
- **Component alignment is by name.** `X` and `Y` are both indexed by sPCA component
  and the ridge fit pairs them row-wise, so `score_ligand_activity()` reindexes `Y`
  onto `X`'s component order and raises if a component is missing. The orders already
  coincided (both lexicographic), so this does not change existing results.
- **Activity scores are not bitwise reproducible.** The permutation null draws from an
  unseeded RNG, so z-scores shift slightly between runs. R2 is deterministic.