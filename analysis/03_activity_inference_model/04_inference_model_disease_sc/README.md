# Disease Ligand-Activity Inference (Single Cell)

Applies the same final SIG13 ligand-activity inference model as
`../03_inference_model_disease_bulk` to the same four CD4 T cell datasets, but scores
every cell individually instead of averaging cells into samples first. Scoring runs
on the GPU.

The model is defined in `../model_core/model_core_sc.py`, separately from the bulk
model in `model_core.py`, so the two can use different alpha grids. The net, gene
set and explanatory matrix are identical to the bulk pipeline's.

## Differences from the bulk pipeline

| | bulk | single cell (here) |
|---|---|---|
| unit of observation | sample (patient, or patient x celltype) | one cell |
| waggr component scoring | CPU, `dc.mt.waggr` | GPU, `rsc.dcg.waggr` |
| target `Y` | component scores averaged per sample, then z-scored per sample | component scores z-scored per cell |
| ridge fit | one `RidgeCV` per sample | one batched GPU solve for all cells |
| alpha grid | `logspace(-1, 4, 500)` | `logspace(-1, 5, 200)` |
| permutation null | 1000 draws per sample | 1000 draws per cell |
| output format | CSV | Parquet |

**Groupings become filters.** Each cell is scored once. The obs columns the bulk
groupings split or filter on (`Level1`, `Level2`, `cluster_name`, `final_analysis`,
`leiden_0.5`) are carried into the output, so bulk-style groupings are recovered by
filtering. No cells are excluded before scoring. Each cell is fit independently, so
filtering afterwards gives the same scores as filtering first.

## How the per-cell fit works

The design matrix `X` (68 components x 38 activities) is the same for every cell.
The ridge solution is therefore a fixed operator applied to each cell's response:

```
beta(alpha) = H(alpha) @ y_centered,    H(alpha) = (Xc'Xc + alpha*I)^-1 Xc'
```

The operators for every alpha and CV fold are precomputed once
(`precompute_ridge_operators()`). Alpha selection and fitting then become batched
matrix multiplications over chunks of 200k cells. The permutation null is batched
the same way over cells and permutations. Each cell gets its own permutation set.

The permutation RNG is seeded from the cell-chunk offset and the alpha index, so a
rerun with the same `cell_chunk` reproduces its scores exactly.

## Steps

| file | what it does |
|---|---|
| `../model_core/model_core_sc.py` | the model: constants, net construction, expression read, GPU waggr, batched ridge + null |
| `01_score_disease_datasets_sc.py` | scores each dataset per cell |
| `02_run_sc_scoring.sh` | SLURM array wrapper, one GPU job per dataset |

```bash
sbatch 02_run_sc_scoring.sh                # all four datasets
sbatch --array=2 02_run_sc_scoring.sh      # just thomas_ibd
```

### Environment

The `rapids_singlecell` mamba env (`rapids_singlecell_cu13` 0.15.0rc4, decoupler
2.1.4, cupy 13.6.0), which `02_run_sc_scoring.sh` activates by name. A GPU is required.

The datasets are read from `imports_stable/external/` and `imports_stable/SIG19/scvi_outs/`,
the same files as in the bulk pipeline. The external datasets are not included in
`imports_stable/` (not generated in this study); see the main `README.md` ("Running the repository", step 2).

## Outputs

Written to `analysis_outs/03_activity_inference_model/inference_model_disease_sc/`:

```
activity_scores_<dataset>.parquet     cell x 38 ligand activity z-scores, + obs covariates
component_scores_<dataset>.parquet    cell x 68 sPCA component scores (the target Y)
ridge_diagnostics_<dataset>.parquet   per-cell ridge R2 and CV-selected alpha
```

## Results

1,000 permutations per cell, `ALPHA_RANGE = logspace(-1, 5, 200)`:

| dataset | cells | median R2 | median alpha | at alpha ceiling |
|---|---|---|---|---|
| `inflammation_atlas` | 1,505,203 | 0.151 | 387 | 15.9% |
| `thomas_ibd` | 145,704 | 0.144 | 415 | 16.2% |
| `amp_2023` | 55,432 | 0.159 | 361 | 16.4% |
| `sig19_iln` | 19,898 | 0.605 | 48.2 | 0.5% |

Notes for downstream analysis:

- **Alpha ceiling.** About 16% of human cells select the maximum alpha (1e5). Raising
  the ceiling from 1e4 to 1e5 barely changed this, which suggests these cells carry
  little coherent signal. Treat `alpha >= 1e5` as a quality flag.
- **Per-cell R2** is in-sample at the CV-chosen alpha and is not a validation metric.
  It is lower per cell than per sample, and much lower for the human datasets than
  for the mouse dataset, which is scored natively without ortholog mapping.

## Activity naming

The 38 activities are the same as in the bulk pipeline. Display names come from
`activity_annotations.csv`, written by step 01 of `../03_inference_model_disease_bulk`
(stable copy in `imports_stable/SIG13/analysis_outs/inference_model_disease_bulk/`).
