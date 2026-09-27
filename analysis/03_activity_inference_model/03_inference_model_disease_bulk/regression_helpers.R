# Shared setup and model fitting for the disease-association regressions
# (analysis/03_activity_inference_model/03_inference_model_disease_bulk, steps 04-07).
#
# Every regression in this folder has the same shape: for each of the model's 38
# ligand activities, fit one linear model of activity ~ condition + covariates,
# then extract a set of estimated-marginal-mean contrasts. `fit_activity_models()`
# is that loop; the Rmds only declare the formula, the contrast, and how p-values
# should be grouped for FDR correction.

suppressPackageStartupMessages({
  library(tidyverse)
  library(emmeans)
  library(broom)
  library(sandwich)
  library(patchwork)   # covariate-QC panels in 04-07 use wrap_plots()
})

# Repo-relative paths (inputs: imports_stable/, outputs: analysis_outs/)
repo_dir <- (function(d = normalizePath(getwd())) {
  while (!dir.exists(file.path(d, "imports_stable"))) {
    if (dirname(d) == d) stop("Could not find repo root (folder containing imports_stable/)")
    d <- dirname(d)
  }
  d
})()
imports_dir <- file.path(repo_dir, "imports_stable")
outs_dir <- file.path(repo_dir, "analysis_outs", "03_activity_inference_model")

# repo-wide publication theme (theme_Publication)
source(file.path(repo_dir, "functions/r_custom/plotting_fxns.R"))
theme_set(theme_Publication())

# ---- paths -------------------------------------------------------------------
# inputs: step 01-03 score tables and step 04-07 regression results (imports_stable snapshot)
SCORE_DIR  <- file.path(imports_dir, "SIG13/analysis_outs/inference_model_disease_bulk")
RES_IN_DIR <- file.path(SCORE_DIR, "regressions")
# outputs: regression results written by 04-07, plots written by 08-11
RES_DIR    <- file.path(outs_dir, "inference_model_disease_bulk/regressions")
PLOT_DIR   <- file.path(outs_dir, "plots/inference_model_disease_bulk")

dir.create(RES_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

SAMPLE_DELIM <- "__"   # must match model_core.SAMPLE_DELIM


# ---- loading -----------------------------------------------------------------

#' Ligand-activity annotations: display name, single vs combinatorial, and which
#' ligand conditions were averaged into each consensus activity (step 01).
load_annotations <- function() {
  read_csv(file.path(SCORE_DIR, "activity_annotations.csv"), show_col_types = FALSE) %>%
    mutate(activity_type = factor(activity_type, c("single", "combinatorial")))
}

#' The model's activity names, i.e. the response variables of every regression.
activity_names <- function() load_annotations()$annotation

#' Load one grouping's activity scores (step 02), split the composite sample name
#' back into its metadata columns, and join the per-sample covariate table.
#'
#' @param label grouping label, e.g. "thomas_ibd" or "thomas_ibd_celltype"
#' @param sample_keys the obs columns that were joined into the sample name, in
#'   order -- must match the `sample_keys` (+ `extra_keys`) used in step 02
#' @param metadata whether to join `sample_metadata_<label>.csv` (patient-level
#'   covariates such as Age, Batch, disease-activity scores)
load_activity_scores <- function(label, sample_keys, metadata = TRUE) {
  scores <- read_csv(file.path(SCORE_DIR, paste0("activity_scores_", label, ".csv")),
                     show_col_types = FALSE)

  if (metadata) {
    covariates <- read_csv(file.path(SCORE_DIR, paste0("sample_metadata_", label, ".csv")),
                           show_col_types = FALSE) %>%
      select(-any_of(sample_keys))     # already recovered from the sample name
    scores <- left_join(scores, covariates, by = "sample")
  }

  separate_wider_delim(scores, sample, delim = SAMPLE_DELIM,
                       names = sample_keys, cols_remove = FALSE)
}

#' Per-sample ridge R2 (step 02) -- fit QC for a grouping.
load_model_fit <- function(label) {
  read_csv(file.path(SCORE_DIR, paste0("r2_", label, ".csv")), show_col_types = FALSE)
}


# ---- model fitting -----------------------------------------------------------

#' Fit one linear model per ligand activity and extract emmeans contrasts.
#'
#' @param data one row per sample, activities as columns (from load_activity_scores)
#' @param rhs right-hand side of the model as a string, e.g. "Disease + sex + site"
#' @param emm emmeans specification, e.g. ~ Disease or ~ Disease | celltype
#' @param contrast_args list passed to emmeans::contrast(), e.g.
#'   list(method = "pairwise"), list(method = "trt.vs.ctrl", ref = "healthy"),
#'   or list(method = <named list of custom contrast weights>)
#' @param robust use HC3 heteroskedasticity-robust covariance for the contrasts
#' @param padj_by columns to group by before BH correction; the default corrects
#'   across activities within each contrast, i.e. "of the 38 activities tested for
#'   this comparison, which survive FDR". Ignored when padj_source = "emmeans".
#' @param padj_source which multiple-testing family `padj` comes from:
#'   * `"BH"` (default) -- BH across the 38 activities within each `padj_by` group.
#'     Answers "of the activities tested for this comparison, which survive FDR".
#'   * `"emmeans"` -- emmeans' own adjustment across the contrast family within each
#'     activity (Dunnett for `trt.vs.ctrl`, Tukey for `pairwise`). Answers "for this
#'     activity, which of the comparisons survive". Used by the SIG19 analysis.
#'   These are orthogonal families, not stricter/looser versions of one another, so
#'   the choice changes which activities pass -- see README.
#' @param activities response variables (defaults to all model activities)
#' @param ... passed to emmeans(), e.g. rg.limit for grids with many levels
#'
#' @return tidy contrasts, one row per contrast x activity, with `component`
#'   (the activity), unadjusted `p.value`, and adjusted `padj`
fit_activity_models <- function(data, rhs, emm, contrast_args,
                                robust = FALSE,
                                padj_by = "contrast",
                                padj_source = c("BH", "emmeans"),
                                activities = activity_names(),
                                ...) {
  stopifnot(all(activities %in% names(data)))
  padj_source <- match.arg(padj_source)
  # for BH we need raw p-values out of emmeans; for "emmeans" we keep its default
  if (padj_source == "BH") contrast_args$adjust <- contrast_args$adjust %||% "none"

  res <- map_dfr(activities, function(activity) {
    fit <- lm(reformulate(rhs, response = sprintf("`%s`", activity)), data = data)
    grid <- if (robust) {
      emmeans(fit, emm, vcov. = sandwich::vcovHC(fit, type = "HC3"), ...)
    } else {
      emmeans(fit, emm, ...)
    }
    cts <- do.call(contrast, c(list(grid), contrast_args))

    out <- tidy(cts) %>% mutate(component = activity)
    if (padj_source == "emmeans") {
      # tidy() reports only the adjusted p when an adjustment was applied; carry the
      # unadjusted one alongside it so every results table has the same columns
      out$padj <- out$adj.p.value
      out$p.value <- summary(cts, adjust = "none")$p.value
    }
    out
  })

  if (padj_source == "BH") {
    res <- res %>%
      group_by(across(all_of(padj_by))) %>%
      mutate(padj = p.adjust(p.value, method = "BH")) %>%
      ungroup()
  }
  res
}

#' Fit, report how many activities survive FDR, and write the results table.
#' Results go to `analysis_outs/03_activity_inference_model/inference_model_disease_bulk/regressions/<name>.csv`.
fit_and_save <- function(data, rhs, emm, contrast_args, name, ...) {
  res <- fit_activity_models(data, rhs, emm, contrast_args, ...)
  write_csv(res, file.path(RES_DIR, paste0(name, ".csv")))
  cat(sprintf("%s: %d contrast x activity rows, %d with padj < 0.1\n",
              name, nrow(res), sum(res$padj < 0.1, na.rm = TRUE)))
  res
}

#' Contrast weights for a 2x2 interaction ("is the double treatment more than the
#' sum of its parts"), given the factor's level order:
#'   (both - single_a) - (single_b - control)
#' Used for the SIG19 antibody-combination synergy tests.
synergy_contrast <- function(levels, both, single_a, single_b, control) {
  as.numeric(levels == both) - as.numeric(levels == single_a) -
    as.numeric(levels == single_b) + as.numeric(levels == control)
}
