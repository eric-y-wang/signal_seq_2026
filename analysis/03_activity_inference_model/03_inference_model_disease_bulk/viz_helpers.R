# Shared plotting for the disease-association figures
# (analysis/03_activity_inference_model/03_inference_model_disease_bulk, steps 08-11).
#
# The figures across the four datasets reduce to four recurring forms:
#   volcano            estimate vs -log10(padj), by contrast
#   significance grid  how many activities move, by contrast x celltype
#   activity tiles     contrast x activity heatmap with significance stars
#   activity ranges    estimate +/- SE per activity, compared across groups
# Each has one function here, so the Rmds stay a list of "which comparison, which
# activities" rather than 60 lines of ggplot per panel.

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(ggrepel)
  library(RColorBrewer)
})

# paths, annotation loading, and the publication theme
source("regression_helpers.R")

# single vs combinatorial, used consistently in every figure
ACTIVITY_COLORS <- c(single = "black", combinatorial = "#D95F02FF")
SIG_CUTOFF <- 0.1


# ---- results preparation -----------------------------------------------------

#' Attach display names and the single/combinatorial split to a results table.
#' `component` (the activity name) is the join key throughout.
#' Idempotent: the plotting functions annotate defensively, so a table the Rmd has
#' already annotated is passed through untouched. Re-joining would suffix the
#' existing columns to `name.x` / `name.y` and break every reference to `name`.
annotate_results <- function(res, annotations = load_annotations()) {
  if (all(c("name", "activity_type") %in% names(res))) return(res)
  left_join(res, annotations, by = c("component" = "annotation"))
}

sig_stars <- function(padj) {
  case_when(padj < 0.001 ~ "***", padj < 0.01 ~ "**", padj < SIG_CUTOFF ~ "*", TRUE ~ "")
}

#' The n activities with the strongest evidence in `res` (smallest padj).
#' @param within optional grouping columns to rank inside (e.g. a single celltype)
#' @param combinatorial_only rank only combination activities, then add their
#'   single-ligand counterparts via `with_component_singles()`
top_activities <- function(res, n = 10, within = NULL, combinatorial_only = FALSE) {
  df <- annotate_results(res)
  if (combinatorial_only) df <- filter(df, activity_type == "combinatorial")
  if (!is.null(within)) df <- semi_join(df, within, by = names(within))

  top <- df %>%
    group_by(component) %>%
    summarise(padj = min(padj, na.rm = TRUE), .groups = "drop") %>%
    slice_min(padj, n = n, with_ties = FALSE) %>%
    pull(component)

  if (combinatorial_only) with_component_singles(top) else top
}

#' Expand a set of activities with the single-ligand activities of their
#' constituent ligands, so a combination can be plotted next to its parts.
#'
#' Uses the annotation table's `members` (the ligand conditions averaged into each
#' activity) rather than splitting activity names on "_", which mis-parses
#' consensus names like `IL6_TNFSF18_c`.
with_component_singles <- function(activities, annotations = load_annotations()) {
  ligands_of <- function(acts) {
    annotations %>%
      filter(annotation %in% acts) %>%
      pull(members) %>%
      str_split(";") %>% unlist() %>%
      str_split("_") %>% unlist() %>%
      unique()
  }
  singles <- annotations %>%
    filter(activity_type == "single") %>%
    filter(map_lgl(str_split(members, ";"), ~ any(.x %in% ligands_of(activities)))) %>%
    pull(annotation)

  unique(c(activities, singles))
}


# ---- figures -----------------------------------------------------------------

#' Volcano of contrast estimates. Points are coloured by single vs combinatorial;
#' activities passing `SIG_CUTOFF` are labelled with their display name.
plot_volcano <- function(res, facet = NULL, label = TRUE, ymax = NULL, title = NULL, ncol = 5) {
  p <- annotate_results(res) %>%
    ggplot(aes(x = estimate, y = -log10(padj))) +
    geom_point(aes(color = activity_type)) +
    geom_hline(yintercept = -log10(SIG_CUTOFF), linetype = "dotted", color = "red") +
    scale_color_manual(values = ACTIVITY_COLORS) +
    theme(aspect.ratio = 1) +
    labs(title = title, x = "estimate", y = "-log10(padj)")

  if (label) p <- p + geom_text_repel(aes(label = ifelse(padj < SIG_CUTOFF, name, "")),
                                      size = 2, max.overlaps = 20)
  if (!is.null(facet)) p <- p + facet_wrap(vars(!!sym(facet)), ncol = ncol)
  if (!is.null(ymax)) p <- p + ylim(0, ymax)
  p
}

#' How many activities move significantly, per contrast x celltype. Answers
#' "which cell types respond at all" before looking at individual activities.
plot_significance_grid <- function(res, celltype_col = "celltype", split_direction = FALSE,
                                   title = NULL) {
  df <- res %>%
    filter(padj < SIG_CUTOFF) %>%
    mutate(direction = ifelse(estimate > 0, "increased", "decreased")) %>%
    group_by(across(all_of(c("contrast", celltype_col, if (split_direction) "direction")))) %>%
    summarise(n = n(), .groups = "drop")

  p <- ggplot(df, aes(x = contrast, y = !!sym(celltype_col))) +
    geom_point(aes(fill = n, size = n), shape = 21) +
    scale_fill_viridis_c() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = title, x = "", y = "")

  if (split_direction) p <- p + facet_wrap(~direction)
  p
}

#' Contrast x activity heatmap with significance stars, split by single vs
#' combinatorial. `activities` fixes both which activities appear and their order.
plot_activity_tiles <- function(res, activities, limits = c(-3, 3),
                                order_by_estimate = TRUE, title = NULL) {
  df <- annotate_results(res) %>%
    filter(component %in% activities) %>%
    mutate(name = if (order_by_estimate) fct_reorder(name, estimate, .desc = TRUE)
                  else factor(name, levels = unique(name[order(match(component, activities))])))

  ggplot(df, aes(x = name, y = contrast)) +
    geom_tile(aes(fill = estimate)) +
    geom_text(aes(label = sig_stars(padj)), vjust = 0.8) +
    scale_fill_distiller(palette = "RdBu", limits = limits, oob = scales::oob_squish) +
    facet_grid(~activity_type, scales = "free", space = "free_x") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    labs(title = title, x = "", y = "")
}

#' Estimate +/- SE per activity, compared across groups (celltypes, timepoints,
#' diseases). Non-significant points are drawn hollow so the eye follows the
#' significant ones without dropping the rest.
plot_activity_ranges <- function(res, activities, group, palette = "Dark2",
                                 order_by = NULL, title = NULL) {
  df <- annotate_results(res) %>%
    filter(component %in% activities) %>%
    mutate(activity_type = fct_relevel(activity_type, "combinatorial"),
           significance = ifelse(padj < SIG_CUTOFF, "Significant", "NS"),
           name = if (is.null(order_by)) fct_reorder(name, estimate, .desc = TRUE)
                  else factor(name, levels = order_by))

  ggplot(df, aes(x = name, y = estimate)) +
    geom_pointrange(aes(ymin = estimate - std.error, ymax = estimate + std.error,
                        color = !!sym(group), alpha = significance, group = !!sym(group)),
                    position = position_dodge(0.5), linewidth = 1, size = 0) +
    geom_point(aes(fill = !!sym(group), shape = significance, size = significance,
                   group = !!sym(group)),
               position = position_dodge(0.5), color = "black") +
    geom_hline(yintercept = 0, linetype = "dotted", color = "red") +
    scale_color_brewer(palette = palette) +
    scale_fill_brewer(palette = palette) +
    scale_shape_manual(values = c(Significant = 23, NS = 21)) +
    scale_size_manual(values = c(Significant = 3, NS = 0)) +
    scale_alpha_manual(values = c(Significant = 1, NS = 0.75)) +
    facet_grid(~activity_type, scales = "free", space = "free_x") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
    labs(title = title, x = "")
}

#' The transpose of `plot_activity_ranges`: contrasts on x, a handful of related
#' activities compared within each. Used to ask "across all these diseases, does
#' the combination activity track its single-ligand parts?"
#'
#' @param activities activities to compare; the first one orders the contrasts
#' @param top_n keep only the `top_n` contrasts with the highest estimate for the
#'   first activity
plot_contrast_ranges <- function(res, activities, top_n = NULL, palette = "Dark2",
                                 title = NULL) {
  df <- annotate_results(res) %>%
    filter(component %in% activities) %>%
    group_by(contrast) %>%
    mutate(sort_value = estimate[match(activities[1], component)]) %>%
    ungroup() %>%
    filter(!is.na(sort_value))

  if (!is.null(top_n)) df <- filter(df, dense_rank(desc(sort_value)) <= top_n)

  df %>%
    mutate(contrast = fct_reorder(contrast, sort_value, .desc = TRUE),
           name = factor(name, levels = annotate_results(tibble(component = activities))$name),
           significance = ifelse(padj < SIG_CUTOFF, "Significant", "NS")) %>%
    ggplot(aes(x = contrast, y = estimate, color = name)) +
    geom_pointrange(aes(ymin = estimate - std.error, ymax = estimate + std.error,
                        alpha = significance, group = name),
                    position = position_dodge(0.5), linewidth = 1, size = 0) +
    geom_point(aes(fill = name, shape = significance, size = significance, group = name),
               position = position_dodge(0.5), color = "black") +
    geom_hline(yintercept = 0, linetype = "dotted", color = "red") +
    scale_color_brewer(palette = palette) +
    scale_fill_brewer(palette = palette) +
    scale_shape_manual(values = c(Significant = 23, NS = 21)) +
    scale_size_manual(values = c(Significant = 3, NS = 0)) +
    scale_alpha_manual(values = c(Significant = 1, NS = 0.75)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
    labs(title = title, x = "", color = "activity", fill = "activity")
}

#' Per-sample ridge R2 for a grouping -- the model-fit QC that accompanies each
#' dataset's figures.
plot_model_fit <- function(label, title = label) {
  load_model_fit(label) %>%
    ggplot(aes(x = r2_score)) +
    geom_histogram(bins = 40, fill = "grey70", color = "white") +
    geom_vline(aes(xintercept = median(r2_score)), linetype = "dashed", color = "red") +
    theme(aspect.ratio = 1) +
    labs(title = paste(title, "ridge fit"), x = "R2 (in-sample, at CV-chosen alpha)",
         y = "samples")
}

#' Per-sample CV-chosen ridge alpha (penalty) for a grouping -- accompanies
#' `plot_model_fit()` as model-fit QC. Alpha is drawn from a log-spaced grid
#' (1e-1..1e4), so the histogram is on a log10 x-axis.
plot_alpha_fit <- function(label, title = label) {
  load_model_fit(label) %>%
    ggplot(aes(x = best_alpha)) +
    geom_histogram(bins = 40, fill = "grey70", color = "white") +
    geom_vline(aes(xintercept = median(best_alpha)), linetype = "dashed", color = "red") +
    scale_x_log10() +
    theme(aspect.ratio = 1) +
    labs(title = paste(title, "ridge penalty"), x = "alpha (CV-chosen, log scale)",
         y = "samples")
}

#' ggsave into the folder's plot directory.
save_plot <- function(name, plot = last_plot(), ...) {
  ggsave(file.path(PLOT_DIR, paste0(name, ".pdf")), plot = plot, ...)
}
