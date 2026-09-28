# Interaction Scoring: Gene-Level GLMs and Interaction Classes

This folder produces the core gene-level results of the screen. For every gene in every ligand
condition it asks two questions. Is the gene differentially expressed vs. control? And, for
ligand pairs, is the combined effect different from the sum of the two single-ligand effects?
Two Gamma-Poisson GLMs (`glmGamPoi`) answer these, and a scoring script combines them into one
interaction class per gene x ligand pair. Most downstream folders (`12_gene_level_analysis/`,
`13_clustering/`, `07_spca/`, `06_interaction_scoring_null/`) start from these tables.

Conditions are named `{round1 ligand}_{round2 ligand}`. `linker` means no ligand in that round,
so `IL4_linker` is IL4 alone, `IL4_TNF` is the pair and `linker_linker` is the control. Both
models share the covariates (`replicate`, `lane`, `percent.mito`, `s.score`, `g2m.score`) and use
deconvolution size factors on raw `counts`:

```text
single term:  counts ~ ligand + replicate + lane + percent.mito + s.score + g2m.score
              # ligand = 1 for the condition, 0 for linker_linker
interaction:  counts ~ ligand1 * ligand2 + lane + replicate + percent.mito + s.score + g2m.score
              # 2x2 design over L1_linker, linker_L2, L1_L2 and linker_linker
```

Every term is tested with `glmGamPoi::test_de` (quasi-likelihood F test, BH-adjusted within each fit [combinatorial ligand condition]). LFCs are log2.

## Pipeline

- **`glmGamPoi_single_term_slurm.r`**: fits the single-term model for every condition (singles,
  pairs and self-pairs) against `linker_linker`. Its `ligand` coefficient is the **condition
  LFC**, the total observed effect of that well vs. control.
- **`glmGamPoi_interaction_slurm.r`**: for every ligand pair `L1_L2`, takes the four groups
  `L1_linker`, `linker_L2`, `L1_L2` and `linker_linker` and fits the interaction model. It tests
  `ligand1`, `ligand2` and `ligand1:ligand2`.
- Both GLM scripts share the same structure:
  - **Gene filter.** Within each fit, genes detected in fewer than `filter_cutoff` of the fit's
    cells are dropped, because very sparse genes distort the dispersion estimates.
  - **Fan-out.** The driver writes the counts matrix to a scratch checkpoint and submits one
    `cpushort` SLURM job per condition or pair through `future.batchtools` (up to 250 at once).
    Each job saves an `.rds` checkpoint, and a rerun skips any that already exist.
  - **Assembly.** Once all jobs finish, the checkpoints are combined into the output tables.
- **`glmGamPoi_single_term_independent_replicates_slurm.r`** and
  **`glmGamPoi_interaction_independent_replicates_slurm.r`**: the same two models fit
  separately within each replicate (`rep1`, `rep2`), so `replicate` is dropped from the design
  (`~ ligand + lane + ...` and `~ ligand1 * ligand2 + lane + ...`). Every output row carries a
  `replicate` column. `02_qc_general/02_intra_assay_correlation.ipynb` uses these 0.2filter
  tables for the rep1 vs rep2 reproducibility filter that later steps apply. They are not used
  by `interaction_scoring_v3.R`.
- **`interaction_scoring_v3.R`**: joins the two models at one `filter_cutoff` and assigns classes.
  - **Expected and total effect.** `lfc_total_interaction_model = lfc_ligand1 + lfc_ligand2 +
    lfc_interaction`. The additive expectation is `lfc_ligand1 + lfc_ligand2`.
  - **Interaction score.** `interaction_score = lfc_interaction / lfc_total_interaction_model`
    when `adj_pval_interaction <= 0.1`, and 0 otherwise.
  - **Classes.** Assigned in order:

    | Class | Rule |
    | --- | --- |
    | `synergy positive` | interaction significant, score > 0, total > 0 |
    | `synergy negative` | interaction significant, score > 0, total < 0 |
    | `buffering` | interaction significant, score < 0 |
    | `none` | no significant interaction, but `adj_pval_condition <= 0.1` |
    | `NA` | neither significant |

    Synergy means the interaction pushes the gene further in the direction of its total effect.
    Buffering means it pulls the gene back toward zero.
  - **Condition LFC.** The single-term condition LFC and p-value (`lfc_condition`,
    `adj_pval_condition`) are joined on, so each row also carries the observed total effect.

## Inputs

- `imports_stable/SIG13/scanpy_outs/SIG13_doublets_DSB7.h5ad`: the `counts` layer, and
  `ligand_call_DSB7`, `replicate`, `lane`, `pct_counts_mt`, `S_score` and `G2M_score` in `obs`.
- `interaction_scoring_v3.R` reads the stable copies of both GLMs' outputs from
  `imports_stable/SIG13/analysis_outs/glmGamPoi/`. Re-running the GLMs writes fresh files to
  `analysis_outs/` only.

The stable copies in `imports_stable/SIG13/analysis_outs/glmGamPoi/` are flat: the files from
the `glmGamPoi_single_term/`, `glmGamPoi_interaction/`, `_independent_replicates/` and
`glmGamPoi_null/` output subfolders below sit together in that one folder. Both GLMs write a
`glmGamPoi_coefficients_{cutoff}filter.csv`, so the stable copies are renamed
`glmGamPoi_interaction_coefficients_{cutoff}filter.csv` and
`glmGamPoi_singleTerm_coefficients_{cutoff}filter.csv`. To replace a stable copy with a fresh
fit, copy the files out of the subfolder, renaming the coefficients file the same way.

## Outputs

All outputs are written under `analysis_outs/02_combinatorial_screen_signalseq_SIG13/glmGamPoi/`,
which is gitignored. All three steps
have been run at `filter_cutoff` 0.05-0.4 (step 0.05). Downstream analyses mostly use 0.05filter
(gene-level classes) and 0.2filter (coefficients for clustering, replicate correlations). Only
the 0.05, 0.1 and 0.2 outputs are in `imports_stable/`; the others are needed only for the
`13_clustering` filter-cutoff sweep (see its README).

- `glmGamPoi_single_term/`:
  - `glmGamPoi_singleTerm_lfc_{cutoff}filter.csv`: condition LFCs, and `_sig_` for
    `adj_pval < 0.1`
  - `glmGamPoi_coefficients_{cutoff}filter.csv`
- `glmGamPoi_interaction/`:
  - `glmGamPoi_interaction_lfc_{cutoff}filter.csv`: the `ligand1:ligand2` test, and `_sig_` for
    `adj_pval < 0.1`
  - `glmGamPoi_singles_lfc_{cutoff}filter.csv`: the `ligand1` and `ligand2` tests, labelled
    `{ligand}_round1` and `{ligand}_round2`
  - `glmGamPoi_coefficients_{cutoff}filter.csv`
- `interactions_scored_v3_glmGamPoi_{cutoff}filter{suffix}.csv`: from `interaction_scoring_v3.R`.
  - no suffix: every gene x condition
  - `_sig`: rows with a class, i.e. significant in either model
  - `_sig_interactions`: rows with a class other than `none`
  - `_sig_summary`: per-condition class counts
- `glmGamPoi_single_term_independent_replicates/`: `glmGamPoi_singleTerm_lfc_independent_replicates_{cutoff}filter.csv`
  and `glmGamPoi_coefficients_independent_replicates_{cutoff}filter.csv`.
- `glmGamPoi_interaction_independent_replicates/`: `glmGamPoi_interaction_lfc_`, `glmGamPoi_singles_lfc_` and
  `glmGamPoi_coefficients_independent_replicates_{cutoff}filter.csv` (no `_sig_` tables). Both
  independent-replicate variants have been run at 0.05 and 0.2.
- GLM checkpoints: `glmGamPoi_single_term_{cutoff}filter_checkpoints/`,
  `interaction_glmGamPoi_slurm_{cutoff}filter_checkpoints/` and the matching
  `*_independent_replicates_{cutoff}filter_checkpoints/` under
  `analysis_outs/02_combinatorial_screen_signalseq_SIG13/checkpoints/`, or under
  `$SIGNALSEQ_SCRATCH` if it is set.

The other `glmGamPoi_*` directories (`_null`, `_DSB*`, `_*30count`) come from
`06_interaction_scoring_null/` and the QC sensitivity analyses, not from this folder.
`02_qc_general` reads the stable copies of the `_independent_replicates` tables from
`imports_stable/SIG13/analysis_outs/glmGamPoi/`.

## Running

The scoring step needs both GLMs at the same cutoff. All three scripts default to 0.05. To use a
different cutoff, pass the same value to every step:

```bash
cd analysis/02_combinatorial_screen_signalseq_SIG13/05_interaction_scoring
sbatch r_job_submission.sh glmGamPoi_single_term_slurm.r [cutoff]
sbatch r_job_submission.sh glmGamPoi_interaction_slurm.r [cutoff]
# after both finish:
sbatch r_job_submission_scoring.sh [cutoff]
```

The independent-replicate fits use the same wrapper. The interaction variant defaults to 0.2
and the single-term variant to 0.05; `02_qc_general` uses 0.2 for both:

```bash
sbatch r_job_submission.sh glmGamPoi_single_term_independent_replicates_slurm.r 0.2
sbatch r_job_submission.sh glmGamPoi_interaction_independent_replicates_slurm.r 0.2
```

`r_job_submission.sh` runs the single-term script if no script name is given. The GLM driver
jobs (8 cores, 200G, up to 6 h) and the scoring job (`cpushort`, 64G, 30 min) run in the
`R-deseq2` env, which `reticulate`/`anndata` need to read the `.h5ad` (`use_condaenv` uses
`conda = "auto"`). The wrappers `cd` into this folder, so submit them from anywhere inside the
repo. Driver logs are written to `R-out.*` and `R-err.*` in the submit directory, and worker
logs go to `.future/logs/` in this folder.
