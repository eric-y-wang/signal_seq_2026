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
# filter_cutoff matches the real single-term script (glmGamPoi_single_term_slurm.r)
# so null and real results are directly comparable at the same cutoff. Pass a
# cutoff on the command line (Rscript 01_glmGamPoi_single_term_null_slurm.r 0.1) to
# run at a different value; defaults to 0.05 if none is given.
cli_args <- commandArgs(trailingOnly = TRUE)
filter_cutoff <- if (length(cli_args) >= 1) as.numeric(cli_args[1]) else 0.05 # Minimum fraction of cells expressing a gene to be used in test
exp_layer <- "counts" # Expression layer to use
# h5ad file path
h5ad_file <- file.path(imports_dir, "SIG13/scanpy_outs/SIG13_doublets_DSB7.h5ad")
# round-1-only and round-2-only linker barcode names, as they appear in feature_call_DSB7
round1_linkers <- c("linker1_round1", "linker2_round1", "linker3_round1")
round2_linkers <- c("linker4_round2", "linker5_round2", "linker6_round2",
                     "linker7_round2", "linker8_round2", "linker9_round2")
# make checkpoint directory
# scratch location for checkpoints: set SIGNALSEQ_SCRATCH to use a scratch filesystem
checkpoint_dir <- file.path(Sys.getenv("SIGNALSEQ_SCRATCH", file.path(outs_dir, "checkpoints")), paste0("glmGamPoi_single_term_null_",filter_cutoff,"filter_checkpoints"))
if (!dir.exists(checkpoint_dir)) dir.create(checkpoint_dir, recursive = TRUE)
# set output directory
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
# script and 02_glmGamPoi_interaction_null_slurm.r both write these concurrently)
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
# transpose to genes x cells, then subset immediately to the null population
# (only these cells are ever used by this script, so no need to export the full matrix)
counts <- adata$layers[[exp_layer]] %>% t()
counts <- counts[, null_pop$cell_barcode, drop = FALSE]
rm(adata, obs)
gc()

# save counts to checkpoint
exp_dir <- file.path(checkpoint_dir, "expression")
if (!dir.exists(exp_dir)) dir.create(exp_dir, recursive = TRUE)
exp_mtx_path <- file.path(exp_dir, "exp_mtx.rds")
saveRDS(counts, exp_mtx_path)
cat("expression matrix saved at: ",exp_mtx_path, "\n")

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

# 4) Prepare pseudo-condition metadata -------------------------------
cat("Preparing pseudo-condition metadata...\n")
# Each of the 9 linker barcodes stands in as a "pseudo-ligand": ligand present in
# the cells carrying that barcode (in whichever round position it naturally
# occupies), reference is every other cell in the null population (carrying a
# different linker barcode in that same position). This mirrors the real
# single-term design (condition vs "linker_linker") but the reference here is
# restricted to the double-linker true-null population rather than pooling in
# real single-ligand cells.
pseudo_linkers <- c(unique(null_pop$round1_linker_id), unique(null_pop$round2_linker_id))

combo_entries <- map(pseudo_linkers, function(linker_id) {
  obs_sub <- null_pop %>%
    mutate(ligand_flag = as.integer(round1_linker_id == linker_id | round2_linker_id == linker_id))
  if (nrow(obs_sub) == 0) return(NULL)
  list(
    condition = linker_id,
    cell_barcodes = obs_sub$cell_barcode,
    ligand_flag = obs_sub$ligand_flag,
    replicate = obs_sub$replicate,
    lane = obs_sub$lane,
    pct_counts_mt = obs_sub$pct_counts_mt,
    s_score = obs_sub$S_score,
    g2m_score = obs_sub$G2M_score
  )
}) %>% compact()

# 5) Filter out combinations already done -----------------------------
done_conditions <- list.files(checkpoint_dir, pattern = "\\.rds$", full.names = FALSE) %>%
  tools::file_path_sans_ext()
entries_to_run <- keep(combo_entries, ~ !.x$condition %in% done_conditions)
cat("Will run GLM for", length(entries_to_run), "pseudo-conditions.\n")

# 6) Define worker function ----------------------------------
# This function will be serialized and run on the worker nodes
run_glm_for_entry <- function(entry, exp_mtx_path, filter_cutoff) {
  suppressPackageStartupMessages({
    require(Matrix)
    require(glmGamPoi)
    require(tidyverse)
    require(anndata)
    require(reticulate)
    require(forcats)
    require(dplyr)
  })

  # import counts (genes x cells)
  counts_full <- readRDS(exp_mtx_path)

  # define pseudo-ligand
  linker_name <- entry$condition

  # Subset the counts matrix and densify
  counts_sub <- counts_full[, entry$cell_barcodes, drop = FALSE]
  counts_sub <- as.matrix(counts_sub)

  # Filter out genes with predefined low expression
  keep <- Matrix::rowSums(counts_sub > 0) >= filter_cutoff*ncol(counts_sub)
  counts_sub <- counts_sub[keep, , drop = FALSE]

  rm(counts_full)
  gc()

  # Build model matrix from the entry data
  model.df <- tibble(
    ligand = entry$ligand_flag,
    lane = factor(entry$lane),
    percent.mito = entry$pct_counts_mt,
    s.score = entry$s_score,
    g2m.score = entry$g2m_score,
    replicate = factor(entry$replicate, levels = c("rep1","rep2"))
  )

  # Fit Gamma-Poisson GLM
  tryCatch({
    fit <- glm_gp(
      counts_sub,
      design = ~ ligand + replicate + lane + percent.mito + s.score + g2m.score,
      col_data = model.df,
      size_factors = "deconvolution",
      on_disk = FALSE
    )

    res_inter <- test_de(fit, contrast = `ligand`) %>%
      as_tibble() %>% mutate(condition = linker_name)
    coef_tbl <- as_tibble(fit$Beta, rownames = "name") %>% mutate(condition = linker_name)

    list(
      de_inter = res_inter,
      coefficients = coef_tbl
    ) %>% return()
  }, error = function(e) {
  cat("ERROR in GLM fitting for", linker_name, ": ", conditionMessage(e), "\n")
  return(NULL)
  })
}

# 7) Submit jobs to slurm -----------------------------------
cat("Submitting jobs to SLURM...\n")

futures <- list()

tryCatch({
  for (i in seq_along(entries_to_run)) {
    entry <- entries_to_run[[i]]

    futures[[entry$condition]] <- future({
      result <- try({
        run_glm_for_entry(entry, exp_mtx_path, filter_cutoff)
      }, silent = TRUE)

      if (inherits(result, "try-error")) {
        cat("ERROR processing", entry$condition, ":", conditionMessage(attr(result, "condition")), "\n")
        return(NULL)
      }

      checkpoint_file <- file.path(checkpoint_dir, paste0(entry$condition, ".rds"))
      saveRDS(result, checkpoint_file)

      entry$condition
    })

    cat("Submitted job for pseudo-condition:", entry$condition, "\n")
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
  cat("Jobs completed:", successful, "out of", length(futures), "pseudo-conditions processed.\n")

}, error = function(e) {
  cat("ERROR during job submission/processing:", conditionMessage(e), "\n")
}, finally = {
  cat("Cleaning up futures...\n")
  try(future:::ClusterRegistry("stop"), silent = TRUE)
  try(plan(sequential), silent = TRUE)
  gc()
})

# 8) Combine results -----------------------------------
cat("Combining all results...\n")

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
all_coefficients <- gather_all_results("coefficients")

# 9) Write out final tables
if (!is.null(interaction_de_results)) {
  write_csv(interaction_de_results,
            file.path(output_dir, paste0("glmGamPoi_singleTerm_null_lfc_", filter_cutoff,"filter.csv")))
  write_csv(interaction_de_results %>% filter(adj_pval < 0.1),
            file.path(output_dir, paste0("glmGamPoi_singleTerm_null_lfc_sig_", filter_cutoff,"filter.csv")))
}
if (!is.null(all_coefficients)) {
  write_csv(all_coefficients,
            file.path(output_dir, paste0("glmGamPoi_singleTerm_null_coefficients_", filter_cutoff,"filter.csv")))
}

cat("All done!\n")
