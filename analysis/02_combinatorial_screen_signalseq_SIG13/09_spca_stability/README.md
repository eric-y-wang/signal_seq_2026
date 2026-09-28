# sPCA Program Stability: Alpha Sweep and Cell-Level Bootstrap

The main sPCA analysis (`07_spca/01_spca_degs_zscore_expression_allLigands.py`) fits
78 gene programs at sparsity `alpha=1.0` on z-scored, DEG-subset pseudobulk profiles
(ligand x replicate means). This folder asks how robust those 78 programs are to two things:

1. **The choice of alpha.** Refit at alpha = 0.1-3.0, holding `n_components=78` fixed.
2. **Cell sampling noise.** Resample cells with replacement within each ligand x replicate
   group, re-pseudobulk and refit at alpha=1.0.

Each refit returns its components in arbitrary order, so every reference (alpha=1.0) program is
paired with its **best available match**: the refit component with the highest cosine similarity
to it. Similarity is computed on per-program z-scored gene loadings. Matching is unconstrained, so
two reference programs that share genes can pick the same refit component. A program's stability
is the distribution of its best-available cosine similarity across alphas or bootstrap
iterations.

## Pipeline

- **`01_spca_alpha_sweep.py`** / **`01_run_alpha_sweep.sh`**: submits one SLURM job per alpha
  (0.1-3.0, step 0.1) that fits the non-negative sPCA at `n_components=78`
  (`random_state=100`). alpha=1.0 is skipped because it would exactly reproduce the existing
  reference fit, which is reused instead.
- **`02_alpha_sweep_viz.ipynb`**: matches each alpha's programs to the alpha=1.0 reference and
  plots similarity across alpha: a heatmap (program x alpha), per-program boxplots, and a
  mean ± SD curve. Programs are annotated as `clean` (kept by `07_spca/03_spca_annotation.Rmd`)
  or `filtered`.
- **`03_spca_cell_bootstrap_worker.py`** / **`03_run_bootstrap.sh`**: runs the cell-level bootstrap
  (2000 iterations, iteration `i` seeded with `seed_base + i`, parallelized with joblib). It skips
  iterations whose output file already exists, so interrupted runs resume where they stopped.
- **`04_cell_bootstrap_viz.ipynb`**: matches each bootstrap iteration's programs to the reference,
  summarizes per-program stability (median/IQR/min/max similarity and the fraction of iterations
  where the best match falls below 0.3, `collision_rate`), and plots per-program boxplots
  annotated as clean/filtered.

## Inputs

All from `imports_stable/SIG13/`:
- `scanpy_outs/SIG13_doublets_DSB7_zscore_degs0.1cutoff.h5ad` (`01`, `03`)
- `analysis_outs/spca/degs_zscore_allLigands/zscore_degs_allLigands_0.1_alpha1.0_sPCA_components.csv`:
  reference loadings
- `analysis_outs/spca/lm_fit_zscore_degs_allLigands_0.1_alpha1.0_sPCA_clean.csv`: clean
  program list
- `analysis_outs/spca_stability/{alpha_sweep,bootstrap/components}/`: the `01` and `03` fits,
  which `02` and `04` read. These are not in `imports_stable/`. Run `01` and `03`, then copy
  `analysis_outs/02_combinatorial_screen_signalseq_SIG13/spca_stability/` to
  `imports_stable/SIG13/analysis_outs/spca_stability/` before running `02` and `04`.

## Outputs

All outputs are written under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/spca_stability/`,
which is gitignored.

- `alpha_sweep/`: per-alpha components, codes and models. Also
  `matched_similarity_long.csv` (one row per alpha x reference program).
- `bootstrap/components/`: per-iteration component loadings.
- `bootstrap/matched_similarity_long.csv`: one row per iteration x reference program.
- `bootstrap/component_stability_summary.csv`: per-program stability summary.
- `plots/`: `alpha_sweep_similarity_{heatmap,boxplot,vs_alpha_best_available}.pdf` and
  `bootstrap_stability_best_available_box.pdf`.

## Running

```bash
cd analysis/02_combinatorial_screen_signalseq_SIG13/09_spca_stability
bash 01_run_alpha_sweep.sh      # submits one sbatch job per alpha
sbatch 03_run_bootstrap.sh      # 32 cores, ~140G
```

After the jobs finish, run `02_alpha_sweep_viz.ipynb` and `04_cell_bootstrap_viz.ipynb` in the
`scanpy_standard2` kernel. Both notebooks are fast because they only read component tables and
compute similarities.
