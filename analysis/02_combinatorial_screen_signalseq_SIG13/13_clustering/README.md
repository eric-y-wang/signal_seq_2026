# Condition Clustering: HDBSCAN on GLM Coefficient Profiles

Gene-level GLM testing (`05_interaction_scoring/`) gives, for every condition, a coefficient
per gene. This folder groups conditions whose coefficient profiles look alike, and asks how much
that grouping depends on the input choices. It does this separately for two kinds of coefficient:

1. **Interaction coefficients** (`ligand1:ligand2` from the interaction GLM): which ligand pairs
   deviate from additivity in the same way.
2. **Single-ligand coefficients** (`ligand` from the single-term GLM, linker-containing conditions
   only): which ligands have similar effects on their own.

Each condition is a row in a condition x gene matrix of coefficients, restricted to the top-N most
variable genes. The rows are clustered with HDBSCAN:

```text
hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='correlation')    # noise = -1
```

Stability is measured by resampling **genes** (columns) with replacement, reclustering, and
computing the adjusted Rand index (ARI) against the clustering on the full gene set. Noise points
are kept in the ARI.

Only replicate-consistent conditions are clustered, using the repo-canonical filter
`pearson_corr > 0.25 & num_shared_genes > 1` on the 0.2filter replicate-correlation tables. This naturally filters for conditions with notable transcriptional effects.

## `_unique` variants

A subset of conditions was tested in both well-position orders (e.g. `IL4_TNF` and `TNF_IL4`, or
`IL4_linker` and `linker_IL4`). In the base notebooks both orders are separate rows. The `_unique`
notebooks first average swapped-order duplicates under one alphabetically sorted label, per gene,
in the coefficients, the HVG-selection tables and the replicate-correlation table (before the
consistency threshold is applied). Each biological pair then contributes exactly one row.

| | Interactions | Single ligands |
| --- | --- | --- |
| Base: conditions / clustered / clusters (reference) | 146 / 106 / 32 | 38 / 31 / 10 |
| `_unique`: conditions / clustered / clusters (reference) | 117 / 91 / 31 | 29 / 25 / 8 |
| Reference median bootstrap ARI (base / `_unique`) | 0.812 / 0.774 | 0.845 / 0.889 |

## Pipeline

- **`01_hdbscan_bootstrap_interactions.ipynb`** (interactions) and
  **`02_hdbscan_bootstrap_singleLigand.ipynb`** (single ligands): the clustering and its
  stability. Both notebooks share the same structure.
  - **Gene selection.** Top-N genes by variance of the significant LFCs across consistent
    conditions: `lfc_interaction` from `interactions_scored_v3_*_sig.csv` for interactions, and
    `lfc` from `glmGamPoi_singleTerm_lfc_sig_*.csv` for single ligands. Missing coefficients are
    filled with 0.
  - **Gene-count sweep.** At 0.2filter, N = 100-3000 (interactions) or 100-6000 (single ligands),
    2000 bootstraps each. Each N is compared with its own reference clustering, not a global one.
  - **Filter-cutoff sweep.** At the reference N, runs the same bootstrap on each expression filter
    cutoff (0.05-0.4, step 0.05). Each cutoff is an independent glmGamPoi fit, so coefficients
    and dispersions are refit rather than reused.
  - **Final clustering.** Reference parameters are 0.2filter with N = 500 (interactions) or
    N = 3000 (single ligands). Runs 5000 bootstraps and aligns each bootstrap's labels to the
    reference with a modified Jonker-Volgenant algorithm (`scipy.linear_sum_assignment`), because HDBSCAN cluster IDs
    are arbitrary. It then reports per condition the fraction of bootstraps where it changes
    cluster (`frac_mismatch`), the number of distinct clusters it lands in, and its **modal
    cluster**. The modal cluster is what downstream notebooks use.
- **`03_hdbscan_bootstrap_interactions_unique.ipynb`** /
  **`04_hdbscan_bootstrap_singleLigand_unique.ipynb`**: `01` and `02` on order-averaged data (see
  above). The interaction gene-count sweep runs to N = 4000.
- **`05_coefficient_corr_heatmap.ipynb`**: Pearson correlation heatmaps of the coefficient
  profiles (same gene selection and consistency filter as `01`/`02`), ward-clustered and annotated
  with the modal HDBSCAN cluster (noise in grey). Interaction rows also carry log10(n+1) counts of
  significant `buffering`, `synergy positive` and `synergy negative` genes on a shared color
  scale. Single-ligand rows carry log10(n+1) significant DEG counts. For each coefficient type it
  draws a **full** heatmap (all consistent conditions) and a **clustered-only** heatmap (noise
  removed). Single-ligand conditions are relabelled `{ligand}_SetA` (`{ligand}_linker`) and
  `{ligand}_SetB` (`linker_{ligand}`).
- **`06_coefficient_corr_heatmap_unique.ipynb`**: `05` on order-averaged data, with clusters taken
  from the `03`/`04` outputs.
- **`07_zscore_gene_heatmap.ipynb`**: standalone overview figure. Averages the control-z-scored,
  DEG-subset expression by `ligand_call_DSB7` and draws a clustered condition x gene heatmap with no
  labels. It does not use the HDBSCAN results.

## Inputs

All paths are under `imports_stable/SIG13/analysis_outs/` unless noted.

- `glmGamPoi/glmGamPoi_interaction_coefficients_{cutoff}filter.csv`: interaction GLM
  coefficients; `01`-`04` read 8 cutoffs (see the note below).
- `glmGamPoi/glmGamPoi_singleTerm_coefficients_{cutoff}filter.csv`: single-term GLM
  coefficients; `01`-`04` read 8 cutoffs.
- `glmGamPoi/interactions_scored_v3_glmGamPoi_{cutoff}filter_sig.csv`: interaction HVG source, and
  the interaction-class counts in `05`/`06`.
- `glmGamPoi/glmGamPoi_singleTerm_lfc_sig_{cutoff}filter.csv`: single-ligand
  HVG source, and the DEG counts in `05`/`06`.
- `replicate_corr/replicate_correlation_{interactionLfc,singleLfc}_0.2filter.csv`: consistency
  filter.
- `imports_stable/SIG13/scanpy_outs/SIG13_doublets_DSB7_zscore_degs0.05cutoff.h5ad`: `07`
  only.
- `clustering/hdbscan_bootstrap_modal_clusters{suffix}_minSamples1[_unique].csv`: stable copies of
  the `01`-`04` modal clusters, read by `05`/`06`. Re-running `01`-`04` writes fresh files to
  `analysis_outs/` only.

Only the 0.05, 0.1 and 0.2 cutoffs of the GLM files above are in `imports_stable/`. To run the
filter-cutoff sweep in `01`-`04`, first regenerate 0.15, 0.25, 0.3, 0.35 and 0.4 with
`../05_interaction_scoring` (both GLMs and the scoring script, run with each cutoff as the
`filter_cutoff` argument), then copy the output flat into `imports_stable/SIG13/analysis_outs/glmGamPoi/`,
renaming each model's `glmGamPoi_coefficients_*` as described in `../05_interaction_scoring/README.md`.
`05`-`07` read only 0.2filter and still run from `imports_stable/`.

## Outputs

All outputs are written under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/`, which is
gitignored. `{suffix}` is empty for
interactions and `_singleLigand` for single ligands. `_unique` is appended for `03`/`04`/`06`.

- `clustering/hdbscan_ref_clusters{suffix}_minSamples1[_unique].csv`: reference (full gene set)
  cluster per condition.
- `clustering/hdbscan_bootstrap_modal_clusters{suffix}_minSamples1[_unique].csv`: modal cluster
  per condition across the 5000 bootstraps. This is the cluster assignment used in `05`/`06`.
- `clustering/hdbscan_bootstrap_cluster_variability{suffix}_minSamples1[_unique].csv`:
  `ref_cluster`, `frac_mismatch`, `n_unique_clusters`, `modal_cluster`.
- `clustering/plots/`:
  - `bootstrap_ari_{ngenes,filters,ref_params}{suffix}_minSamples1[_unique].pdf`
  - `n_clusters_{ngenes,filters}{suffix}_minSamples1[_unique].pdf`
  - `coefficient_corr_heatmap_{interactions,singleLigand}_{full,clustered}[_unique].pdf`
- `plots/zscore_gene_heatmap.png`: from `07`.

## Running

Run the notebooks interactively in the `scanpy_standard2` kernel. They are CPU-only and need no
SLURM. Run `01`-`04` before `05`/`06`, which read the modal-cluster CSVs. Each bootstrap notebook
takes about 5-10 minutes, most of it the two parameter sweeps.
