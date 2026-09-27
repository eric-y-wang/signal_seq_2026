# Ligand-Activity Inference Model

Builds a model that infers ligand signaling activities in CD4 T cells from gene
expression, validates it, and applies it to disease datasets.

The model is trained on the SIG13 combinatorial signal-seq screen
(`../02_combinatorial_screen_signalseq_SIG13`). Each ligand condition in the screen, single
or combinatorial, has a characteristic effect on the screen's sPCA gene expression
programs. Given a new sample, the model asks which combination of those ligand
effects best explains the sample's program scores.

## The model

| stage | definition |
|---|---|
| design `X` | 38 ligand activities x 68 sPCA components, from the SIG13 screen. Co-linear ligand conditions are merged into consensus activities (11 single, 27 combinatorial). |
| target `Y` | decoupler `waggr` scores of the same 68 components (top 50 genes each, loadings as weights) in the sample being scored, z-scored |
| fit | ridge regression of `Y` on `X`, alpha chosen by 5-fold CV |
| activity score | each ridge coefficient z-scored against a 1000x label-permutation null |

This variant (`ridge_zscore_50_weighted`) was selected from 8 candidates using
calibration experiments with known stimulations. Human datasets are scored against
the gene net mapped to human orthologs.

## Folders

| folder | what it does |
|---|---|
| `01_inference_model_construction_validation` | builds the explanatory matrix `X` and selects the model variant on SIG14 (mouse) and SIG26 (human) in vitro stimulations |
| `02_inference_model_mixture_validation` | tests whether the model distinguishes a true combinatorial stimulation from a 50:50 mixture of cells stimulated with each ligand alone |
| `03_inference_model_disease_bulk` | scores four CD4 T cell datasets per sample and tests which activities track disease or treatment |
| `04_inference_model_disease_sc` | scores the same four datasets per cell on the GPU |
| `05_inference_model_validation_AMP` | checks the RA synovium predictions against single-cell activity scores, receptor expression, and in vitro (SIG26) and in vivo (SIG19) signatures |
| `model_core` | shared model code (`model_core.py`, `model_core_sc.py`) and the pinned mouse->human ortholog map |

Datasets scored in 03 and 04:

| dataset | description | species |
|---|---|---|
| `inflammation_atlas` | Inflammation Atlas, cross-disease (immune-mediated inflammatory diseases) | human |
| `amp_2023` | AMP Phase 2, rheumatoid arthritis synovium | human |
| `thomas_ibd` | Thomas et al. 2024, ulcerative colitis and Crohn's disease colon | human |
| `sig19_iln` | SIG19 Treg depletion + cytokine blockade, inguinal lymph node | mouse |

## Run order

```text
01 construction ──> model_core ──┬──> 02 mixture validation
                                 ├──> 03 disease (bulk)
                                 └──> 04 disease (single cell) ──> 05 AMP validation
```

01 must run first: it writes the explanatory matrix that `model_core` reads. After
that, 02, 03 and 04 are independent of each other. 05 needs the per-cell scores
from 04. 04 takes activity display names from 03's step 01.

Each folder's README has its own steps, inputs and outputs.

## Environments

| folder | Python | R |
|---|---|---|
| 01 | `scanpy_standard` | `R-signalseq` |
| 02 | `scanpy_standard2` | |
| 03 | `scanpy_standard` | `R-signalseq` |
| 04 | `rapids_singlecell` (GPU) | |
| 05 | `scanpy_standard2` | |

01 and 03 use `scanpy_standard` so that scoring uses the same decoupler build the
model was calibrated with. Environment files are in the repo's `environments/`.

## Data locations

Inputs are read from `imports_stable/` at the repo root. It holds the bulk RNA-seq counts (`imports_stable/SIG14/`,
`imports_stable/SIG26/`), and the outputs of earlier steps (`imports_stable/SIG13/analysis_outs/`,
`imports_stable/SIG19/`). Later steps read the stable copies, so each step runs on its own. The external public
datasets (AMP 2023, Inflammation Atlas, Thomas IBD) are **not** included, because they were not
generated in this study; to re-run the disease scoring (`03`, `04`) or AMP validation (`05`), place
them under `imports_stable/external/` as described in the main `README.md` ("Running the repository", step 2). Their activity
scores are included, so the regression and visualization steps run without them.

All outputs are written under `analysis_outs/03_activity_inference_model/` (not tracked), in a subfolder named for
each analysis. Figures go to `analysis_outs/03_activity_inference_model/plots/`.

Scripts find the repo root by searching upward for `imports_stable/`, so notebooks, Rmds and
`sbatch` jobs must be run or submitted from inside the repo. Shared code comes from the repo's
`functions/`.
