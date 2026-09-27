# SIG29: Projection of SIG18 Gene Programs

Scores the SIG18 sparse-PCA gene programs (dGEPs,
`../../05_in_vitro_differentiation_RNAseq_SIG18/02_spca`) in each SIG29 sample, then asks
at the program level whether each TNF-family member interacts with TGFb the way TNF does
in SIG18. Each program's score is fit with one term per ligand and one per ligand pair:

```text
score ~ lig_<L> + pair_<A>_<B> + replicate
```

`lig_<L>` marks a ligand in either slot and `pair_<A>_<B>` is the interaction term for an
unordered pair.

## Pipeline

- **`01_dGEP_scoring_SIG29.ipynb`**: scores the SIG18 sPCA programs in each SIG29 sample.
  - Uses the clean components from SIG18 (alpha 10). Each component's net is its top 50
    genes by loading, with the loadings as weights.
  - Normalizes the counts (`normalize_total`, `log1p`), z-scores the program genes and
    runs decoupler `waggr` (`tmin=5`, no permutations). SIG18 and SIG29 are both mouse,
    so no gene conversion is needed.
- **`02_dGEP_analysis_SIG29_waggr.Rmd`**: linear models on the z-scaled program scores.
  - Fits the model above per program with OLS, HC3 robust SEs and BH adjustment across
    all terms.
  - Drops programs whose most significant term is `replicate`. Classifies the pair terms
    as `synergy positive`, `synergy negative`, `buffering` or `none` (see `../README.md`).
  - Condition effects vs. `none_none` are linear contrasts from the same fits (`lig_<L>`
    alone, or `lig_<A> + lig_<B> + pair_<A>_<B>` for combinations).
  - Heatmap of SIG29 condition effects next to SIG18 `TGFb`, `TNF` and `TGFb+TNF`
    estimates. It shows up to 6 programs each for SIG18 TGFb:TNF `synergy positive`,
    TGFb-only and TNF-only.
  - Volcano plots of the pair terms, colored by the matching SIG18 program estimates
    (`TGFb:TNF`, `TGFb`, `TNF`).

## Inputs

- `imports_stable/SIG29/processing_outs/` (produced outside this repo):
  `count_matrix_umiDeDup_SIG29.csv`, `processed_metadata_SIG29.csv` and
  `featureNames_SIG29.csv`.
- `02` reads `01`'s program scores from their stable copy,
  `imports_stable/SIG29/analysis_outs/inference_SIG18/waggr_scored_obs_SIG29.csv`.
- SIG18 sPCA outputs from `../../05_in_vitro_differentiation_RNAseq_SIG18/02_spca`, read from
  their stable copies in `imports_stable/SIG18/analysis_outs/spca/`:
  - `zscore_degs/zscore_degs_unscaled_alpha10.0_sPCA_components.csv` and
    `lm_fit_zscore_degs_alpha10.0_sPCA_clean.csv` (`01`)
  - `lm_scored_zscore_degs_alpha10.0_sPCA_clean.csv` and
    `lm_fit_zscore_degs_alpha10.0_sPCA.csv` (`02`)

## Outputs

Written to `analysis_outs/08_tnf_family_tgfb_interaction_RNAseq_SIG29/inference_SIG18/`, which is gitignored:

- `waggr_scored_obs_SIG29.csv` (`01`): program scores per sample.
- `scoreInteractionLM_waggr_SIG29.csv`, `scoreInteractionScored_waggr_SIG29.csv` and
  `scoreConditionLM_waggr_SIG29.csv` (`02`).
- `scoreHeatmap_*_waggr.pdf` and `scoreVolcano_SIG18_*_waggr_SIG29.pdf` (`02`).

## Running

Run `01` in the `scanpy_standard2` env (decoupler 2.x), then knit `02` in the
`R-signalseq` env. Both find the repo root by searching
upward for `imports_stable/` and create the output folder if needed.
