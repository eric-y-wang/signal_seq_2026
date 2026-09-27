# sPCA Component LM: Empirical Null Calibration

`07_spca/03_spca_annotation.Rmd` fits, per sPCA component,
`loading ~ ligand1 * ligand2 + replicate` on the real waggr-scored pseudobulk
data and calls a `ligand1:ligand2` term significant from its BH-adjusted
p-value. This folder builds an empirical null to check directly,
p-value calibration of this model.

The null population is cells carrying only uniquely barcoded non-targeting linker VLPs,
with the 9 individual barcode identities standing in for ligand1/ligand2
identity in place of real ligands. Because none of these barcodes carry any
biological signal, any "significant" `ligand1:ligand2` term recovered from
this population is by construction a false positive - so the resulting
p-value distribution is a direct empirical readout of the real LM's
calibration (uniform p-values / correct FDR control vs. inflation).

## Pipeline

- **`01_spca_waggr_scoring_null.ipynb`** — Recovers the true-null population:
  singlet, double-linker cells with exactly one round-1 and one round-2
  linker barcode (via `feature_call_DSB7` / `ligand_call_{round}_DSB7`,
  mirroring the population definition used for the GLM null). Runs `decoupler`'s
  `waggr` scoring against the existing (not refit) sPCA component loadings and
  writes the null pseudobulk component scores.

- **`02_spca_null_diagnostics.Rmd`** — Fits the identical
  `loading ~ ligand1 * ligand2 + replicate` model (HC3 robust SEs, BH
  adjustment across all component/term p-values) on the null scores, using
  the most-abundant barcode per slot as the reference level (there's no
  principled "true zero" reference among 9 equally-null barcodes, unlike real
  data's `"linker"`). Loads the real analysis's own LM output
  (`lm_fit_zscore_degs_allLigands_0.1_alpha1.0_sPCA.csv`, already computed by
  `07_spca/03_spca_annotation.Rmd`) alongside the null fit for a direct real-vs-null
  comparison. Terms are pooled into two contrast categories, `ligand` (either main effect) or
  `ligand1:ligand2` (the interaction term that drives synergy/buffering calls) - and reports:
  - p-value histograms, null-only and real-vs-null overlay, per contrast
  - QQ plots (observed vs. expected `-log10(p)`), null-only and
    real-vs-null, per contrast
  - the empirical false-positive rate at `p_adj <= 0.1` (fraction of null
    component-terms called significant, per contrast)

## Files in this folder

- `01_spca_waggr_scoring_null.ipynb` — builds the null population, runs
  waggr scoring, writes the null pseudobulk component-score table.
- `02_spca_null_diagnostics.Rmd` / `.html` — fits the null LM, compares
  against the real LM fit, reports calibration diagnostics.

## Inputs

All from `imports_stable/SIG13/`:
- `scanpy_outs/SIG13_doublets_DSB7.h5ad` (`01`).
- `analysis_outs/spca/zscore_degs_allLigands_0.1_alpha1.0_sPCA_loadings.csv` (`01`) and
  `analysis_outs/spca/lm_fit_zscore_degs_allLigands_0.1_alpha1.0_sPCA.csv` (`02`), from `07_spca`.
- `analysis_outs/spca/spca_null/zscore_degs_null_0.1_alpha1.0_waggr_score.csv`: the stable copy
  of the `01` output that `02` reads. Re-running `01` writes a fresh copy to `analysis_outs/`
  only.

## Outputs

All under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/` (gitignored):
- `spca/spca_null/zscore_degs_null_0.1_alpha1.0_waggr_score.csv` — null pseudobulk waggr
  scores, one row per pseudo-ligand-pair x replicate, one column per sPCA component.
- `plots/spca_null/` — p-value histograms and QQ plots (null-only and real-vs-null), as PDFs.

## Running

The `07_spca` outputs it needs (component loadings and the real LM fit CSV) and
`SIG13_doublets_DSB7.h5ad` are read from `imports_stable/`, so `07_spca` does not have to be
re-run first.

```bash
cd analysis/02_combinatorial_screen_signalseq_SIG13/08_spca_null
jupyter nbconvert --to notebook --execute 01_spca_waggr_scoring_null.ipynb
```

Then knit `02_spca_null_diagnostics.Rmd` (or render from terminal per the
repo convention).
