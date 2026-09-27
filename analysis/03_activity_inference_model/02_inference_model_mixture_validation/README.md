# Mixture-vs-Combination Validation of the Inference Model

Tests whether the SIG13 ligand-activity inference model can tell a **true
combinatorial stimulation** from a **profile pooled from cells stimulated with each
ligand alone**. A combinatorial activity should mean "these cells saw A and B", not
just "this profile contains A signal and B signal". A 50:50 mix of A-only and
B-only cells contains both signals.

SIG14 and SIG26 are plate-based bulk experiments (one well = one sample), so a
mixed cell population can be simulated by mixing well-level count profiles.

**Result: the model separates real combinations from mixtures in both datasets.**

## Steps

| step | what it does |
|---|---|
| `01_build_and_score_mixtures.ipynb` | builds `mix_AB` profiles, scores real and synthetic samples together, QC against the calibration scores |
| `02_mixture_discrimination_analysis.ipynb` | statistics and figures |

Both notebooks run in the `scanpy_standard2` env in a few minutes. No SLURM job is needed.

## Design

For each ligand pair `(A, B)` and replicate:

| sample_type | construction |
|---|---|
| `real_control` | `none_none` well |
| `real_single_A` / `real_single_B` | A-alone / B-alone well |
| `real_combo` | true A+B well |
| `mix_AB` | 0.5 x A-alone + 0.5 x B-alone |

Mixtures are built in relative-abundance space. Each parent well is scaled to a
fraction of its library, the two are averaged, and the result is rescaled to the
parents' mean depth. Synthetic columns are appended to the raw count matrix, so
real and synthetic samples go through the same normalization, `waggr` and ridge
steps. They are scored in one run per dataset, because gene scaling is computed
across all samples.

Mixtures are built within a replicate (donor or mouse), which keeps batch out of
the comparison. The **ligand pair is the unit of analysis**: each pair's
activity profile is averaged across replicates after scoring.

| dataset | ligand pairs | replicates |
|---|---|---|
| `SIG26-6h` (human) | 11 | 3 donors |
| `SIG14` (mouse) | 11 | 3-4 mice |

Each pair needs A-alone, B-alone and A+B wells, plus a matching combinatorial
activity in the model. Ground truth follows the calibration mapping in
`../01_inference_model_construction_validation/04_calibration_model_testing.Rmd`.

The model is imported from `../model_core/model_core.py`, not
re-implemented. The SIG14 and SIG26-6h counts come from
`imports_stable/{SIG14,SIG26}/processing_outs/`, and the calibration scores for
comparison come from `imports_stable/SIG13/analysis_outs/inference_model_calibration/`.
Notebook 02 reads 01's outputs from their stable copy in
`imports_stable/SIG13/analysis_outs/inference_model_mixture_validation/`.

## Results

**Primary readout:** `z` of the mapped combinatorial activity, `real_combo` vs
`mix_AB`, paired within ligand pair.

| dataset | n pairs | median delta z | separated | paired Wilcoxon p |
|---|---|---|---|---|
| `SIG26-6h` | 11 | 1.49 | 9/11 | 0.014 |
| `SIG14` | 11 | 1.51 | 11/11 | 0.00098 |

A Tukey HSD across `real_control`, `mix_AB` and `real_combo` finds all pairwise
contrasts significant in both datasets. The mixture sits above control, and the real
combination sits above the mixture.

**Combinatorial vs single effect size:** activities are control-subtracted, and the
margin is `d(combinatorial) - d(best single)`.

| dataset | condition | median margin | margin > 0 |
|---|---|---|---|
| `SIG14` | `mix_AB` | -1.12 | 1/11 |
| `SIG14` | `real_combo` | +0.65 | 9/11 |
| `SIG26-6h` | `mix_AB` | +0.03 | 6/11 |
| `SIG26-6h` | `real_combo` | +0.64 | 7/11 |

In SIG14 the combinatorial activity falls below the singles in mixtures and rises
above them in real combinations (paired p = 0.0010). In SIG26-6h the shift is not
significant (p = 0.083).

**Per-pair breakdown:** only two pairs show no separation, both in SIG26-6h:
`IL27+IL21` and `TGFB+IL2`.

**Caveat:** `waggr` and ridge at fixed alpha are linear, so a linear expression
mixture gives an approximately linear mixture of activity scores. This test
therefore measures whether the true combinatorial response is non-additive and
whether the model reads that non-additivity out on the right activity.

## Outputs

Written to `analysis_outs/03_activity_inference_model/inference_model_mixture_validation/`:

```
activity_mixtures_{SIG26-6h,SIG14}.csv      sample x 38 activities
r2_mixtures_{SIG26-6h,SIG14}.csv            per-sample ridge R2 and alpha
sample_index_{SIG26-6h,SIG14}.csv           ligand pair x replicate x role -> sample_id
mixture_discrimination_summary.csv          primary readout table
combinatorial_vs_single_effect_size.csv     control-subtracted margins
per_ligand_pair_separation.csv              delta z per ligand pair
figures/01_combinatorial_activity_by_sample_type.pdf
figures/02_combinatorial_vs_single_effect_size.pdf
figures/03_per_pair_delta.pdf
```

`sample_index_*.csv` has one row per pair x replicate x role, because real wells
are shared across pairs. Join it to `activity_mixtures_*.csv` on `sample_id`, then
average over replicate.
