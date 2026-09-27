# Signal Reinforcement and Transcriptional Synergy

Tests whether ligand pairs that **induce each other's receptors** show **stronger
transcriptional synergy** than pairs that don't.

The hypothesis: if ligand A alone upregulates the receptor for its partner B (or vice
versa), the `A_B` combination should show more synergistic genes than combinations
without this cross-induction.

## Method

- **Ligand -> receptor map** (`ligand_receptor_map.csv`): a curated map of each of the
  54 screen ligands to its receptor gene subunit(s), one row per subunit (mouse
  symbols). The shared subunits `Il2rg` and `Il6st` are excluded as too broad to
  indicate reinforcement of any one partner. Three ligands (`CCL27`, `CX3CL1`, `OSM`)
  have no receptor gene in the expression panel and cannot be scored. Three (WNT1,
  MSTN, MIF) carry a low-confidence note.
- **Reinforcement score**: for each combination `A_B`, the mean single-ligand LFC of
  A on B's receptor subunits (`lfc_mean_1to2`), and of B on A's (`lfc_mean_2to1`).
  The two are summed into `lfc_mean_sum`.
- **Reinforcing pairs** (`sig_flag_final`): at least one direction has a positive
  mean LFC with at least one receptor gene at `adj_pval <= 0.01`, and
  `lfc_mean_sum > 0`.
- **Synergy metrics**: per combination, the fraction of its DEGs in each interaction
  class (`freq_synergy_positive`, `freq_synergy_negative`, `freq_synergy_total`,
  `freq_buffering`), plus the DEG count (`n_total_degs`). From
  `interactions_scored_v3_glmGamPoi_0.1filter.csv`.
- **Filters**: self-pairs (e.g. `IL2_IL2`) are excluded. Combinations are restricted
  to reproducible pairs (`replicate_correlation_interactionLfc_0.2filter.csv`,
  `pearson_corr > 0.25 & num_shared_genes > 1`). The two orientations of a pair
  (`IL2_IL4`, `IL4_IL2`) are collapsed into one unordered pair.
- **Tests**, one per metric:
  - Wilcoxon: reinforcing vs. non-reinforcing pairs (BH-corrected across the 5
    metrics).
  - Spearman: `lfc_mean_sum` vs. each metric (BH-corrected across the 5 metrics).
  - Permutation null: receptor identities are shuffled across ligands (5000
    permutations) and the Spearman correlation is recomputed, giving an empirical p.

## Notebooks

- **`01_build_ligand_receptor_map.ipynb`**: builds `ligand_receptor_map.csv`, computes
  reinforcement scores and synergy metrics, and writes `results_combined_0.1filter.csv`
  (138 reproducible combinations).
- **`02_reinforcement_synergy_testing.ipynb`**: collapses orientations to 110 unordered
  pairs, runs the three tests, and writes the result tables and figures.
- **`03_representative_example_viz.Rmd`**: an example. IL6 and IL21 both signal mainly
  through STAT3, but IL21 synergizes strongly with the CCR7 ligands (CCL19, CCL21A)
  and IL6 only weakly. This notebook plots the receptor cross-regulation (`Ccr7`,
  `Il21r`, `Il6ra`) and the reinforcement vs. synergy counts for those four pairs.

## Results

Of 110 unordered pairs, 38 are reinforcing and 72 are not.

| metric | Wilcoxon p_adj | Spearman r (p_adj) | permutation p |
|---|---|---|---|
| `freq_synergy_positive` | 5.8e-7 | 0.38 (2.1e-4) | 0.0046 |
| `freq_synergy_negative` | 1.8e-6 | 0.30 (2.3e-3) | 0.030 |
| `freq_synergy_total` | 4.5e-7 | 0.37 (2.1e-4) | 0.0084 |
| `n_total_degs` | 1.9e-4 | 0.29 (2.3e-3) | 0.036 |
| `freq_buffering` | 0.15 | 0.08 (0.38) | 0.60 |

Reinforcement tracks synergy: all three synergy metrics are significant in all three
tests, while buffering is not significant in any.

## Outputs

Written to `analysis_outs/02_combinatorial_screen_signalseq_SIG13/signal_reinforcement_lfc_mean/`.
`02` and `03` read the stable copy of `results_combined_0.1filter.csv` from
`imports_stable/SIG13/analysis_outs/signal_reinforcement_lfc_mean/`, so re-running `01` writes a
fresh copy to `analysis_outs/` only. Inputs (`SIG13_doublets_DSB7.h5ad`, the scored GLM
interactions and replicate correlations) are read from `imports_stable/SIG13/`.

- `results_combined_0.1filter.csv`, `merged_collapsed_0.1filter.csv`
- `wilcoxon_group_comparison_0.1filter.csv`, `spearman_correlation_0.1filter.csv`,
  `permutation_null_0.1filter.csv`
- `figures/`: boxplots, scatters and permutation nulls (02), and the IL21/CCL
  receptor plots (03)

`ligand_receptor_map.csv` is tracked in this folder.

## Running

Run 01 then 02 in the `scanpy_standard2` env. Knit 03 with `R-signalseq`.
