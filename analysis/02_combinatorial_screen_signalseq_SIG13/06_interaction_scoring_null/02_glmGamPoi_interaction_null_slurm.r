require(tidyverse)
require(Matrix)
require(glmGamPoi)
require(data.table)
require(future)
require(future.batchtools)
require(anndata)
require(reticulate)

# Repo-relative paths (inputs: imports_stable/, outputs: analysis_outs/)
repo_dir <- (function(d = normalizePath(getwd())) {
  while (!dir.exists(file.path(d, "imports_stable"))) {
    if (dirname(d) == d) stop("Could not find repo root (folder containing imports_stable/)")
    d <- dirname(d)
  }
  d
})()
imports_dir <- file.path(repo_dir, "imports_stable")
outs_dir <- file.path(repo_dir, "analysis_outs", "02_combinatorial_screen_signalseq_SIG13")

# 1) Define Parameters ---------------------------------------------------
# filter_cutoff matches the real interaction script (glmGamPoi_interaction_slurm.r)
# so null and real results are directly comparable at the same cutoff. Pass a
# cutoff on the command line (Rscript 02_glmGamPoi_interaction_null_slurm.r 0.05) to
# run at a different value; defaults to 0.1 if none is given.
cli_args <- commandArgs(trailingOnly = TRUE)
filter_cutoff <- if (length(cli_args) >= 1) as.numeric(cli_args[1]) else 0.1
exp_layer <- "counts"
min_cells_per_arm <- 20 # minimum cells required in every 2x2 arm to attempt a fit
# h5ad file path
h5ad_file <- file.path(imports_dir, "SIG13/scanpy_outs/SIG13_doublets_DSB7.h5ad")
# round-1-only and round-2-only linker barcode names, as they appear in feature_call_DSB7
round1_linkers <- c("linker1_round1", "linker2_round1", "linker3_round1")
round2_linkers <- c("linker4_round2", "linker5_round2", "linker6_round2",
                     "linker7_round2", "linker8_round2", "linker9_round2")
# make checkpoint directory
# scratch location for checkpoints: set SIGNALSEQ_SCRATCH to use a scratch filesystem
checkpoint_dir <- file.path(Sys.getenv("SIGNALSEQ_SCRATCH", file.path(outs_dir, "checkpoints")), paste0("interaction_glmGamPoi_null_",filter_cutoff,"filter_checkpoints"))
if (!dir.exists(checkpoint_dir)) dir.create(checkpoint_dir, recursive = TRUE)
# make output directory
output_dir <- file.path(outs_dir, "glmGamPoi/glmGamPoi_null")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
# load reticulate conda environment
reticulate::use_condaenv("R-deseq2", conda = "auto", required = TRUE)
# set future options
options(future.globals.maxSize = 1 * 1024^3)  # set mem limits for future jobs

# 2) Load adata, build the true-null population, and extract counts -----------
cat("Loading adata...\n")
adata <- read_h5ad(h5ad_file)
obs <- adata$obs %>% as_tibble(rownames = "cell_barcode")

# Recover per-cell linker barcode identity: feature_call_DSB7 is pipe-delimited
# and contains the specific guide barcodes detected in a cell (real ligand
# guides and/or linker[1-9]_round[1-2] non-targeting guides)
get_linker_feats <- function(feature_call, linker_set) {
  feats <- str_split(feature_call, "\\|")[[1]]
  intersect(feats, linker_set)
}

obs <- obs %>%
  mutate(
    round1_linker_feats = map(feature_call_DSB7, get_linker_feats, linker_set = round1_linkers),
    round2_linker_feats = map(feature_call_DSB7, get_linker_feats, linker_set = round2_linkers),
    round1_n_linker_feats = lengths(round1_linker_feats),
    round2_n_linker_feats = lengths(round2_linker_feats)
  )

# Restrict to the true-null population: both rounds called "linker" by the
# existing pipeline AND exactly one singlet linker barcode detected per round
# (excludes doublets/ambiguous calls)
null_pop <- obs %>%
  filter(
    ligand_call_round1_DSB7 == "linker",
    ligand_call_round2_DSB7 == "linker",
    round1_n_linker_feats == 1,
    round2_n_linker_feats == 1
  ) %>%
  mutate(
    round1_linker_id = map_chr(round1_linker_feats, 1),
    round2_linker_id = map_chr(round2_linker_feats, 1)
  ) %>%
  select(
    cell_barcode, round1_linker_id, round2_linker_id,
    replicate, lane, pct_counts_mt, S_score, G2M_score
  )

cat("True-null (double-linker singlet) population size:", nrow(null_pop), "\n")

# QC crosstabs (deterministic given the input data, so it's harmless if this
# script and 01_glmGamPoi_single_term_null_slurm.r both write these concurrently)
xtab_round1_replicate <- null_pop %>% count(round1_linker_id, replicate, name = "n_cells")
xtab_round2_replicate <- null_pop %>% count(round2_linker_id, replicate, name = "n_cells")
xtab_round1_lane <- null_pop %>% count(round1_linker_id, lane, name = "n_cells")
xtab_round2_lane <- null_pop %>% count(round2_linker_id, lane, name = "n_cells")
xtab_pair_counts <- null_pop %>% count(round1_linker_id, round2_linker_id, name = "n_cells")

write_csv(xtab_round1_replicate, file.path(output_dir, "qc_round1_linker_x_replicate.csv"))
write_csv(xtab_round2_replicate, file.path(output_dir, "qc_round2_linker_x_replicate.csv"))
write_csv(xtab_round1_lane, file.path(output_dir, "qc_round1_linker_x_lane.csv"))
write_csv(xtab_round2_lane, file.path(output_dir, "qc_round2_linker_x_lane.csv"))
write_csv(xtab_pair_counts, file.path(output_dir, "qc_round1x_round2_linker_pair_counts.csv"))

cat("round1 x round2 linker pair cell counts:\n")
print(xtab_pair_counts %>% arrange(desc(n_cells)))

cat("Extracting counts matrix...\n")
# transpose to genes x cells, then subset immediately to the null population.
# Because the null population (round1 always one of 3 linkers, round2 always one
# of 6 linkers) is the same for every round1xround2 pseudo-combo tested below,
# the cell population and gene filter are fixed once here rather than
# recomputed per combo (unlike the real interaction script, whose population
# differs per real ligand pair).
counts <- adata$layers[[exp_layer]] %>% t()
counts <- counts[, null_pop$cell_barcode, drop = FALSE]
rm(adata, obs)
gc()
counts <- as.matrix(counts)
keep <- Matrix::rowSums(counts > 0) >= filter_cutoff * ncol(counts)
counts <- counts[keep, , drop = FALSE]

exp_dir <- file.path(checkpoint_dir, "expression")
if (!dir.exists(exp_dir)) dir.create(exp_dir, recursive = TRUE)
exp_mtx_path <- file.path(exp_dir, "exp_mtx.rds")
saveRDS(counts, exp_mtx_path)
cat("expression matrix saved at: ",exp_mtx_path, "\n")
rm(counts)
gc()

# 3) Set up slurm-based future backend -----------------------------------------
# Use explicit path to template file - check if it exists first
template_path <- file.path(checkpoint_dir, "slurm_template.tmpl")
# Create a basic template file if it doesn't exist
writeLines(
'#!/bin/bash
#SBATCH --job-name=<%= job.name %>
#SBATCH --partition=<%= resources$partition %>
#SBATCH --ntasks=<%= resources$ntasks %>
#SBATCH --cpus-per-task=<%= resources$cpus.per.task %>
#SBATCH --mem-per-cpu=<%= resources$mem.per.cpu %>
#SBATCH --time=<%= resources$time.limit %>
#SBATCH --output=<%= resources$log.file %>
#SBATCH --error=<%= resources$log.file %>

<%
# relative paths are not handled well by Slurm
log.file = fs::path_expand(log.file)
-%>

# load conda environment
source ~/.bashrc
mamba activate R-deseq2

## Export value of DEBUGME environemnt var to slave
export DEBUGME=<%= Sys.getenv("DEBUGME") %>

<%= sprintf("export OMP_NUM_THREADS=%i", resources$omp.threads) -%>
<%= sprintf("export OPENBLAS_NUM_THREADS=%i", resources$blas.threads) -%>
<%= sprintf("export MKL_NUM_THREADS=%i", resources$blas.threads) -%>

Rscript -e "batchtools::doJobCollection(\'<%= uri %>\')"
',
  template_path
)
message("Created SLURM template at: ", template_path)
options(future.batchtools.template = template_path)

# Configure slurm job submissions
workers <- tweak(batchtools_slurm,
                template = template_path,
                workers = 250,  # Maximum number of concurrent jobs
                resources = list(
                  job.name = "glmGamPoi_null_job",
                  partition = "cpushort",
                  ntasks = 1L,
                  cpus.per.task = 1L,
                  mem.per.cpu = "50G",
                  time.limit = "2:00:00",
                  log.file = ".future/logs/job-%j.log"
                ))
plan(workers)

# 4) Prepare round1 x round2 pseudo-combo metadata -------------------------------
cat("Preparing pseudo-combo metadata...\n")
round1_ids <- sort(unique(null_pop$round1_linker_id))
round2_ids <- sort(unique(null_pop$round2_linker_id))

pairwise_combinations <- expand_grid(round1_id = round1_ids, round2_id = round2_ids)

# Every cell in the null population carries exactly one round1 linker and one
# round2 linker, so for ANY given (round1_id, round2_id) pair the entire null
# population partitions into the same 4 arms used by the real interaction
# script's design: round1-alone, round2-alone, combo (both), and reference
# (neither) - no further cell-barcode subsetting is required per combo.
combo_entries <- pmap(pairwise_combinations, function(round1_id, round2_id) {
  ligand1 <- as.integer(null_pop$round1_linker_id == round1_id)
  ligand2 <- as.integer(null_pop$round2_linker_id == round2_id)

  n_ligand1_only <- sum(ligand1 == 1 & ligand2 == 0)
  n_ligand2_only <- sum(ligand1 == 0 & ligand2 == 1)
  n_combo         <- sum(ligand1 == 1 & ligand2 == 1)
  n_reference     <- sum(ligand1 == 0 & ligand2 == 0)

  combo_id <- paste0(round1_id, "_", round2_id)
  list(
    combo = combo_id,
    round1_id = round1_id,
    round2_id = round2_id,
    ligand1 = ligand1,
    ligand2 = ligand2,
    n_ligand1_only = n_ligand1_only,
    n_ligand2_only = n_ligand2_only,
    n_combo = n_combo,
    n_reference = n_reference,
    replicate = null_pop$replicate,
    lane = null_pop$lane,
    pct_counts_mt = null_pop$pct_counts_mt,
    s_score = null_pop$S_score,
    g2m_score = null_pop$G2M_score
  )
})

# 5) Log combo coverage (fit vs skipped) and drop underpowered combos ----------
coverage_log <- map_dfr(combo_entries, function(e) {
  arm_min <- min(e$n_ligand1_only, e$n_ligand2_only, e$n_combo, e$n_reference)
  will_fit <- arm_min >= min_cells_per_arm
  skip_reason <- if (will_fit) {
    NA_character_
  } else if (e$n_combo == 0) {
    "no cells with both linkers (biologically impossible / never observed)"
  } else {
    paste0("smallest arm has only ", arm_min, " cells (< min_cells_per_arm = ", min_cells_per_arm, ")")
  }
  tibble(
    combo = e$combo,
    round1_id = e$round1_id,
    round2_id = e$round2_id,
    n_ligand1_only = e$n_ligand1_only,
    n_ligand2_only = e$n_ligand2_only,
    n_combo = e$n_combo,
    n_reference = e$n_reference,
    will_fit = will_fit,
    skip_reason = skip_reason
  )
})
write_csv(coverage_log, file.path(output_dir, paste0("interaction_null_combo_coverage_", filter_cutoff, "filter.csv")))
cat("Combo coverage log written. Will attempt", sum(coverage_log$will_fit), "of", nrow(coverage_log), "round1xround2 combos.\n")
print(coverage_log)

combo_entries <- keep(combo_entries, ~ {
  min(.x$n_ligand1_only, .x$n_ligand2_only, .x$n_combo, .x$n_reference) >= min_cells_per_arm
})

# 6) Filter out combinations already done -------------------------------------
done_combos <- list.files(checkpoint_dir, pattern = "\\.rds$", full.names = FALSE) %>%
  tools::file_path_sans_ext()
entries_to_run <- keep(combo_entries, ~ !.x$combo %in% done_combos)
cat("Will run GLM for", length(entries_to_run), "new pseudo-combos.\n")

# 7) Define worker function ------------------------------------------
# This function will be serialized and run on the worker nodes
run_glm_for_entry <- function(entry, exp_mtx_path) {
  suppressPackageStartupMessages({
    require(Matrix)
    require(glmGamPoi)
    require(tidyverse)
    require(anndata)
    require(reticulate)
    require(forcats)
    require(dplyr)
  })

  combo <- entry$combo

  # counts matrix was already subset to the null population and gene-filtered
  # once, up front (fixed population/filter across all combos in this script)
  counts_sub <- readRDS(exp_mtx_path)

  model.df <- tibble(
    ligand1      = entry$ligand1,
    ligand2      = entry$ligand2,
    lane         = factor(entry$lane),
    replicate    = factor(entry$replicate, levels = c("rep1", "rep2")),
    percent.mito = entry$pct_counts_mt,
    s.score      = entry$s_score,
    g2m.score    = entry$g2m_score
  )

  tryCatch({
    fit <- glm_gp(
      counts_sub,
      design       = ~ ligand1 * ligand2 + lane + replicate + percent.mito + s.score + g2m.score,
      col_data     = model.df,
      size_factors = "deconvolution",
      on_disk      = FALSE,
      verbose      = TRUE,
    )

    res_inter <- test_de(fit, contrast = `ligand1:ligand2`) %>%
      as_tibble() %>% mutate(interaction = combo)
    res_l1   <- test_de(fit, contrast = `ligand1`) %>%
      as_tibble() %>% mutate(interaction = combo, single_ligand = paste0(entry$round1_id, "_round1"))
    res_l2   <- test_de(fit, contrast = `ligand2`) %>%
      as_tibble() %>% mutate(interaction = combo, single_ligand = paste0(entry$round2_id, "_round2"))
    coef_tbl <- as_tibble(fit$Beta, rownames = "genes") %>% mutate(interaction = combo)

    list(
      de_inter = res_inter,
      de_singles = bind_rows(res_l1, res_l2),
      coefficients = coef_tbl
    )
  }, error = function(e) {
  cat("ERROR in GLM fitting for", combo, ": ", conditionMessage(e), "\n")
  return(NULL)
  })
}

# 8) Submit jobs to slurm -----------------------------------------
cat("Submitting jobs to SLURM...\n")

futures <- list()

tryCatch({
  for (i in seq_along(entries_to_run)) {
    entry <- entries_to_run[[i]]

    futures[[entry$combo]] <- future({
      result <- try({
        run_glm_for_entry(entry, exp_mtx_path)
      }, silent = TRUE)

      if (inherits(result, "try-error")) {
        cat("ERROR processing", entry$combo, ":", conditionMessage(attr(result, "condition")), "\n")
        return(NULL)
      }

      checkpoint_file <- file.path(checkpoint_dir, paste0(entry$combo, ".rds"))
      saveRDS(result, checkpoint_file)

      entry$combo
    })

    cat("Submitted job for pseudo-combo:", entry$combo, "\n")
  }

  cat("Waiting for all jobs to complete...\n")
  results <- list()
  for (name in names(futures)) {
    cat("Checking results for", name, "\n")
    results[[name]] <- try(value(futures[[name]]), silent = TRUE)
    if (inherits(results[[name]], "try-error")) {
      cat("ERROR with job", name, ":", conditionMessage(attr(results[[name]], "condition")), "\n")
    }
  }

  successful <- sum(!sapply(results, inherits, "try-error"))
  cat("Jobs completed:", successful, "out of", length(futures), "pseudo-combos processed.\n")

}, error = function(e) {
  cat("ERROR during job submission/processing:", conditionMessage(e), "\n")
}, finally = {
  cat("Cleaning up futures...\n")
  try(future:::ClusterRegistry("stop"), silent = TRUE)
  try(plan(sequential), silent = TRUE)
  gc()
})


# 9) Combine results -----------------------------------------------------
cat("Combining all results...","\n")

gather_all_results <- function(result_type) {
  result_files <- list.files(checkpoint_dir, pattern = "\\.rds$", full.names = TRUE)

  results <- list()
  for (file in result_files) {
    res <- try(readRDS(file), silent = TRUE)
    if (!inherits(res, "try-error") && !is.null(res[[result_type]])) {
      results[[basename(file)]] <- res[[result_type]]
    } else {
      cat("WARNING: Could not read results from", basename(file), "\n")
    }
  }

  if (length(results) == 0) {
    cat("ERROR: No valid results found for", result_type, "\n")
    return(NULL)
  }

  bind_rows(results)
}

interaction_de_results <- gather_all_results("de_inter")
single_de_results <- gather_all_results("de_singles")
all_coefficients <- gather_all_results("coefficients")

# 10) Write out final tables -------------------------------------
if (!is.null(interaction_de_results)) {
  write_csv(interaction_de_results,
            file.path(output_dir, paste0("glmGamPoi_interaction_null_lfc_", filter_cutoff,"filter.csv")))
  write_csv(interaction_de_results %>% filter(adj_pval < 0.1),
            file.path(output_dir, paste0("glmGamPoi_interaction_null_lfc_sig_", filter_cutoff,"filter.csv")))
}

if (!is.null(single_de_results)) {
  write_csv(single_de_results,
            file.path(output_dir, paste0("glmGamPoi_interaction_null_singles_lfc_", filter_cutoff,"filter.csv")))
}

if (!is.null(all_coefficients)) {
  write_csv(all_coefficients,
            file.path(output_dir, paste0("glmGamPoi_interaction_null_coefficients_", filter_cutoff,"filter.csv")))
}

cat("All done!\n")
