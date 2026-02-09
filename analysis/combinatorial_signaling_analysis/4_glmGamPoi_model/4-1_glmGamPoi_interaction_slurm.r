require(tidyverse)
require(Matrix)
require(glmGamPoi)
require(data.table)
require(future)
require(future.batchtools)
require(anndata)
require(reticulate)

# 1) Define Parameters ---------------------------------------------------
# Minimum fraction of cells expressing a gene to be used in test
filter_cutoff <- 0.1
# Expression layer to use
exp_layer <- "counts"
# h5ad file path
h5ad_file <- "/data1/rudenska/EYW/SIG13/scanpy_outs/SIG13_doublets_DSB7.h5ad"
# make checkpoint directory
checkpoint_dir <- file.path(paste0("/scratch/rudenska/interaction_glmGamPoi_slurm_",filter_cutoff,"filter_checkpoints"))
if (!dir.exists(checkpoint_dir)) dir.create(checkpoint_dir, recursive = TRUE)
# make output directory
output_dir <- "/data1/rudenska/EYW/git_projects/SIG13/analysis_outs/glmGamPoi/glmGamPoi_interaction"
# load reticulate conda environment
reticulate::use_condaenv("R-deseq2", conda = "/home/wange7/miniforge3/bin/conda", required = TRUE)
# set future options
options(future.globals.maxSize = 1 * 1024^3)  # set mem limits for future jobs

# 2) Load adata and counts -----------------------------------------

cat("Loading adata and metadata...\n")
adata <- read_h5ad(h5ad_file)
obs <- adata$obs %>% as_tibble(rownames = "cell_barcode")
cat("Metadata loaded.\n")

# 3) export counts matrix for future jobs -----------------------------------------
cat("Exporting counts matrix...\n")
# transpose to genes x cells
counts <- adata$layers[[exp_layer]] %>% t()
# save counts to checkpoint
exp_dir <- file.path(checkpoint_dir, "expression")
if (!dir.exists(exp_dir)) dir.create(exp_dir, recursive = TRUE)
exp_mtx_path <- file.path(exp_dir, "exp_mtx.rds")
saveRDS(counts, exp_mtx_path)
cat("expression matrix saved at: ",exp_mtx_path, "\n")

# 4) Set up slurm-based future backend -----------------------------------------
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
                  job.name = "glmGamPoi_job",
                  partition = "cpushort",
                  ntasks = 1L,
                  cpus.per.task = 1L,
                  mem.per.cpu = "50G", 
                  time.limit = "2:00:00",
                  log.file = ".future/logs/job-%j.log"
                ))
plan(workers)

# 5) Prepare combination metadata (indices + obs subsets) -------------------
cat("Preparing combination metadata...\n")
obs_dt <- as.data.table(obs)
setkey(obs_dt, ligand_call_DSB7)

pairwise_combinations <- obs_dt[!grepl("linker", ligand_call_DSB7), unique(ligand_call_DSB7)] %>%
  str_split("_", simplify = FALSE)

combo_entries <- map(pairwise_combinations, function(ligands) {
  combo_id <- paste0(ligands[1], "_", ligands[2])
  groupPull <- c(
    paste0(ligands[1], "_linker"),
    paste0("linker_", ligands[2]),
    combo_id,
    "linker_linker"
  )
  obs_sub <- obs_dt[ligand_call_DSB7 %in% groupPull]

  if (nrow(obs_sub) == 0) return(NULL)
  list(
    combo        = combo_id,
    ligands      = ligands,
    cell_barcodes = obs_sub$cell_barcode,
    ligand_calls = obs_sub$ligand_call_DSB7,
    lane         = obs_sub$lane,
    replicate    = obs_sub$replicate,
    pct_counts_mt = obs_sub$pct_counts_mt,
    s_score      = obs_sub$S_score,
    g2m_score    = obs_sub$G2M_score
  )
}) %>% compact()

# 6) Filter out combinations already done -------------------------------------
done_combos <- list.files(checkpoint_dir, pattern = "\\.rds$", full.names = FALSE) %>%
  tools::file_path_sans_ext()
entries_to_run <- keep(combo_entries, ~ !.x$combo %in% done_combos)
cat("Will run GLM for", length(entries_to_run), "new combinations.\n")

# 7) Define worker function ------------------------------------------
# This function will be serialized and run on the worker nodes
run_glm_for_entry <- function(entry, exp_mtx_path, filter_cutoff) {
  # Load libraries - explicitly include forcats to avoid the warning
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

  # define combination and ligands
  combo <- entry$combo
  ligands <- entry$ligands
  
  # Subset the counts matrix and densify
  counts_sub <- counts_full[, entry$cell_barcodes, drop = FALSE]
  counts_sub <- as.matrix(counts_sub)
  
  # Filter out genes with predefined low expression
  # necessary for GLM fitting because otherwise messes with dispersion estimates
  keep <- Matrix::rowSums(counts_sub > 0) >= filter_cutoff*ncol(counts_sub) 
  counts_sub <- counts_sub[keep, , drop = FALSE]
  
  # Clean up to free memory
  rm(counts_full)
  gc()
  
  # Build model matrix from the entry data
  ligand1 <- as.numeric(entry$ligand_calls %in% c(
    paste0(ligands[1], "_linker"), combo
  ))
  ligand2 <- as.numeric(entry$ligand_calls %in% c(
    paste0("linker_", ligands[2]), combo
  ))
  model.df <- tibble(
    ligand1     = ligand1,
    ligand2     = ligand2,
    lane        = factor(entry$lane),
    replicate   = factor(entry$replicate, levels = c("rep1", "rep2")),
    percent.mito= entry$pct_counts_mt,
    s.score     = entry$s_score,
    g2m.score   = entry$g2m_score
  )
  
  # Fit Gamma-Poisson GLM
  tryCatch({
    fit <- glm_gp(
      counts_sub,
      design       = ~ ligand1 * ligand2 + lane + replicate + percent.mito + s.score + g2m.score,
      col_data     = model.df,
      size_factors = "deconvolution",
      on_disk      = FALSE,
      verbose      = TRUE,
    )
    
    # Differential tests and coefficient extraction
    res_inter <- test_de(fit, contrast = `ligand1:ligand2`) %>%
      as_tibble() %>% mutate(interaction = combo)
    res_l1   <- test_de(fit, contrast = `ligand1`) %>%
      as_tibble() %>% mutate(interaction = combo, single_ligand = paste0(ligands[1],"_round1"))
    res_l2   <- test_de(fit, contrast = `ligand2`) %>%
      as_tibble() %>% mutate(interaction = combo, single_ligand = paste0(ligands[2],"_round2"))
    coef_tbl <- as_tibble(fit$Beta, rownames = "genes") %>% mutate(interaction = combo)
    
    # Return results
    list(
      de_inter = res_inter,
      de_singles = bind_rows(res_l1, res_l2), 
      coefficients = coef_tbl
    )
  }, error = function(e) {
  cat("ERROR in GLM fitting for", combo, ": ", conditionMessage(e), "\n")
  # You could also log the error into a file for better diagnosis
  return(NULL)
  })
}

# 8) Submit jobs to slurm -----------------------------------------
cat("Submitting jobs to SLURM...\n")

# Create list to store job futures
futures <- list()

# Error handling for job submission
tryCatch({
  # Submit each combination as a separate job
  for (i in seq_along(entries_to_run)) {
    entry <- entries_to_run[[i]]
    
    # Submit job and store future
    futures[[entry$combo]] <- future({
      result <- try({
        run_glm_for_entry(entry, exp_mtx_path, filter_cutoff)
      }, silent = TRUE)
      
      if (inherits(result, "try-error")) {
        cat("ERROR processing", entry$combo, ":", conditionMessage(attr(result, "condition")), "\n")
        return(NULL)
      }
      
      # Save result to checkpoint file
      checkpoint_file <- file.path(checkpoint_dir, paste0(entry$combo, ".rds"))
      saveRDS(result, checkpoint_file)
      
      # Return combo ID to indicate completion
      entry$combo
    })
    
    cat("Submitted job for combination:", entry$combo, "\n")
  }

  # Wait for all jobs to complete with better error handling
  cat("Waiting for all jobs to complete...\n")
  results <- list()
  for (name in names(futures)) {
    cat("Checking results for", name, "\n")
    results[[name]] <- try(value(futures[[name]]), silent = TRUE)
    if (inherits(results[[name]], "try-error")) {
      cat("ERROR with job", name, ":", conditionMessage(attr(results[[name]], "condition")), "\n")
    }
  }
  
  # Count successful results
  successful <- sum(!sapply(results, inherits, "try-error"))
  cat("Jobs completed:", successful, "out of", length(futures), "combinations processed.\n")
  
}, error = function(e) {
  cat("ERROR during job submission/processing:", conditionMessage(e), "\n")
}, finally = {
  # Clean up futures
  cat("Cleaning up futures...\n")
  try(future:::ClusterRegistry("stop"), silent = TRUE)
  try(plan(sequential), silent = TRUE)
  gc()
})


# 9) Combine results -----------------------------------------------------
cat("Combining all results...","\n")

# Function to gather all results with error handling
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
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
if (!is.null(interaction_de_results)) {
  write_csv(interaction_de_results,
            file.path(output_dir, paste0("glmGamPoi_interaction_lfc_", filter_cutoff,"filter.csv")))
  write_csv(interaction_de_results %>% filter(adj_pval < 0.1),
            file.path(output_dir, paste0("glmGamPoi_interaction_lfc_sig_", filter_cutoff,"filter.csv")))
}

if (!is.null(single_de_results)) {
  write_csv(single_de_results,
            file.path(output_dir, paste0("glmGamPoi_singles_lfc_", filter_cutoff,"filter.csv")))
}

if (!is.null(all_coefficients)) {
  write_csv(all_coefficients,
            file.path(output_dir, paste0("glmGamPoi_coefficients_", filter_cutoff,"filter.csv")))
}

cat("All done!\n")