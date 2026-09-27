# SIG30 Knockouts: Projection of SIG18 Gene Programs

Scores the SIG18 sparse-PCA gene programs (dGEPs,
`../../05_in_vitro_differentiation_RNAseq_SIG18/02_spca`) in each SIG30 sample, then
tests how each knockout shifts each program:

```text
score ~ target + replicate      # control is the reference level
```

Adding a ligand and removing its receptor should move a program in opposite directions, so
the knockout effects are shown next to the SIG18 ligand effects.

## Pipeline

- **`01_dGEP_scoring_SIG30.ipynb`**: scores the SIG18 programs in each SIG30 sample.
  - Loads the SIG18 sPCA loadings (`alpha10.0`) and keeps the clean components.
  - Builds a weighted net from the top 50 genes per component, with loadings as weights.
    Both experiments are mouse, so no gene conversion is needed.
  - Normalizes counts (`normalize_total`, `log1p`), z-scores the program genes and runs
    decoupler `waggr` (`tmin = 5`, no permutations).
- **`02_dGEP_analysis_SIG30_waggr.Rmd`**: tests the effect of each knockout on each program.
  - Z-scales each program score across samples, as SIG18 does for its own scores, so both
    experiments report effects in SD units.
  - Fits `score ~ target + replicate` per program and takes emmeans contrasts of each target
    vs. control. One Benjamini-Hochberg correction is applied across all program x target
    tests; `p_adj < 0.1` counts as significant.
  - Heatmap of 15 programs chosen from SIG18's TGFb x TNF model: the top 5 positively
    synergistic, 5 TGFb-only and 5 TNF-only programs. The SIG30 KO estimates are shown next
    to the SIG18 `TGFb`, `TNF` and combined estimates.
  - Volcano plots per target, with points coloured and sized by the SIG18 `TGFb:TNF`,
    `TGFb` or `TNF` term.

## Inputs

- `imports_stable/SIG30/processing_outs/` (produced outside this repo):
  `count_matrix_umiDeDup_SIG30.csv`, `processed_metadata_SIG30.csv` and
  `featureNames_SIG30.csv`.
- SIG18 sPCA outputs from `../../05_in_vitro_differentiation_RNAseq_SIG18/02_spca`, read from
  their stable copies in `imports_stable/SIG18/analysis_outs/spca/`: component loadings, the
  clean-component list, and the `lm_scored`/`lm_fit` tables for the `alpha10.0` model.
- `02` reads `01`'s program scores from their stable copy,
  `imports_stable/SIG30/analysis_outs/inference_SIG18/waggr_scored_obs_SIG30.csv`.

## Outputs

Written to `analysis_outs/07_crispr_arrayed_validation_RNAseq_SIG30/inference_SIG18/`, which is gitignored:

- `waggr_scored_obs_SIG30.csv` (`01`): program scores per sample.
- `scoreAssoc_waggr_SIG30.csv` (`02`): knockout vs. control estimates per program.
- `scoreHeatmap_classEstimate_*.pdf` and `scoreVolcano_SIG18_*_waggr_SIG30.pdf` (`02`).

## Running

Run `01` in the `scanpy_standard2` env (decoupler 2.x), then knit `02` in the
`R-signalseq` env. Both create the output folder if it
does not exist.
