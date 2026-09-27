# Disease Ligand-Activity Inference (Bulk Datasets)

Applies the final SIG13 ligand-activity inference model to four CD4 T cell
datasets (three external, plus the in-house SIG19 antibody experiment) and tests
which activities track disease or treatment. The model is built and calibrated in
`../01_inference_model_construction_validation`; nothing here re-fits it.

## The model

`ridge_zscore_50_weighted`, selected on SIG14 and SIG26 ground truth:

| stage | definition |
|---|---|
| signature net | SIG13 sPCA component loadings, replicate-associated components dropped, **top 50** genes per component, loadings as weights |
| target `Y` | decoupler `waggr` component scores on gene-scaled `log1p_norm` expression, averaged per sample, z-scored per sample |
| design `X` | SIG13 ligand x component explanatory matrix, z-scored per activity |
| fit | `RidgeCV` per sample (alpha `logspace(-1, 4, 500)`, 5-fold); coefficients z-scored against a 1000x label-permutation null |

`../model_core/model_core.py` defines the model. Human datasets are scored against
the net mapped to human orthologs, using the pinned map in
`../model_core/mouse_human_ortholog_map.csv`.

There are **38 ligand activities** (11 single, 27 combinatorial). Each is named for
one representative ligand condition of a consensus cluster. Display names come from
`inference_model_activity_lookup.csv`.

## Datasets

| key | dataset | species | groupings |
|---|---|---|---|
| `inflammation_atlas` | Inflammation Atlas 2026, cross-IMID | human | whole sample; naive/non-naive (Tregs excluded); fine celltype |
| `amp_2023` | AMP Phase 2, rheumatoid arthritis synovium | human | whole biopsy; celltype |
| `thomas_ibd` | Thomas et al. 2024, UC / CD colon | human | whole sample (Tregs excluded); celltype |
| `sig19_iln` | SIG19 Treg depletion (DTR) + cytokine blockade, iLN | mouse | whole iLN; leiden cluster |

## Steps

Run in order. Steps 01-03 are Python/SLURM; steps 04-11 are Rmd.

| step | what it does |
|---|---|
| `01_activity_annotations.py` | display names and single/combinatorial split for the 38 activities |
| `02_score_disease_datasets.py` | scores each dataset; writes activity scores, ridge R2, per-sample metadata |
| `03_run_disease_scoring.sh` | SLURM array wrapper for 02, one job per dataset |
| `04`-`07_regressions_*.Rmd` | disease/treatment association models, one per dataset |
| `08`-`11_viz_*.Rmd` | figures, one per dataset |

```bash
python 01_activity_annotations.py
sbatch 03_run_disease_scoring.sh              # all four datasets
sbatch --array=2 03_run_disease_scoring.sh    # just thomas_ibd
```

`regression_helpers.R` and `viz_helpers.R` hold the shared code. Each regression
fits one linear model per activity and then computes emmeans contrasts.

Step 02 reads the datasets from `imports_stable/external/` (Inflammation Atlas, AMP 2023,
Thomas IBD) and `imports_stable/SIG19/scvi_outs/` (SIG19). The three external datasets are not included in
`imports_stable/` (not generated in this study); place them there first, as described in
the main `README.md` ("Running the repository", step 2). The Rmds read the scores
from their stable copies in `imports_stable/SIG13/analysis_outs/inference_model_disease_bulk/`
(`SCORE_DIR` in `regression_helpers.R`). Steps 04-07 write regression results to `RES_DIR`
in `analysis_outs/`. Steps 08-11 read the stable copies of those results through
`RES_IN_DIR` (`.../inference_model_disease_bulk/regressions/` under `imports_stable/`).

### Environments

- **Scoring (steps 01-03):** `scanpy_standard`, the same env (and decoupler build)
  used for model calibration.
- **Rmds (steps 04-11):** `R-signalseq` (R 4.5.3). Knit from this folder, since the
  Rmds source helper files by relative path:

```bash
R_PROFILE_USER=/dev/null Rscript -e 'rmarkdown::render("04_regressions_inflammation_atlas.Rmd")'
```

## Outputs

Written to `analysis_outs/03_activity_inference_model/inference_model_disease_bulk/`:

```
activity_scores_<grouping>.csv    sample x ligand activity z-scores
r2_<grouping>.csv                 per-sample ridge R2 + CV-chosen alpha
sample_metadata_<grouping>.csv    per-sample covariates + cell counts
activity_annotations.csv          activity -> display name, single/combinatorial
regressions/<model>.csv           tidy contrasts: estimate, std.error, p.value, padj
```

Figures are written to `analysis_outs/03_activity_inference_model/plots/inference_model_disease_bulk/`.

## Covariates

Each dataset uses one covariate set for all of its models.

| dataset | covariates | excluded, and why |
|---|---|---|
| `inflammation_atlas` | `sex + age + chemistry` (age continuous, scaled) | `studyID`: collinear with disease, since each study contributes specific diseases |
| `amp_2023` | `sex + age + joint` | none |
| `thomas_ibd` | `sex + biopsy_site`, HC3 robust covariance | `Age`: fully separated from disease. `Batch`: all healthy donors are in one batch |
| `sig19_iln` | `sex` | `cage`: every cage is single-sex, so sex is nested within cage |

**Atlas samples without age are dropped.** Age is missing for 110 of 816 atlas
patients (~13%), so the atlas results describe ~706 patients. The loss is uneven:
all `flu` patients lack age (so `flu` is not estimable), and roughly half of
`COVID` and `sepsis` patients lack age. In the Level2 model, `sepsis x T_CD4_CM` and
`sepsis x Tregs` also become non-estimable.

## Conventions

- **Sample names** join the grouping's metadata columns with `__`.
  `load_activity_scores()` splits them back apart.
- **p-values** are extracted unadjusted from emmeans, then BH-corrected across the
  38 activities within each contrast (and within celltype for celltype models).
  Use `padj`.
- **Ridge R2** is in-sample at the CV-chosen alpha. It checks model fit and is not a
  validation metric. Per-celltype groupings fit worse than whole-sample ones.
- **Activity scores are reproducible.** The permutation null is seeded per sample.
