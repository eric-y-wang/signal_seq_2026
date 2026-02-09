
#### IMPORT ####
library(tidyverse)

# import LFC from non-interaction model (glmGamPoi_single_term_slurm.R output)
base_dir <- "/data1/rudenska/EYW/git_projects/SIG13/analysis_outs/glmGamPoi/glmGamPoi_single_term/"
conditionLfc <- read_csv(paste0(base_dir,"glmGamPoi_singleTerm_lfc_0.2filter.csv"))

# import LFC coefficents from interaction model (glmGamPoi_interaction_slurm.R output)
base_dir <- "/data1/rudenska/EYW/git_projects/SIG13/analysis_outs/glmGamPoi/glmGamPoi_interaction/"
interactionsLfc <- read_csv(paste0(base_dir,"/glmGamPoi_interaction_lfc_0.2filter.csv"))
interSig <- interactionsLfc %>% filter(adj_pval < 0.1)
singlesLfc <- read_csv(paste0(base_dir,"glmGamPoi_singles_lfc_0.2filter.csv"))

#### CALCULATE INTERACTION SCORE ####

# make minimal interaction lfc tibble
interLfcMin <- interactionsLfc %>%
  select(name, interaction, lfc, adj_pval) %>%
  dplyr::rename(lfc_interaction = lfc,
                adj_pval_interaction = adj_pval)

# pivot single lfcs
singleLfcWide <- singlesLfc %>%
  mutate(single_ligand = case_when(grepl("_round1",single_ligand) ~ "ligand1",
                                   grepl("_round2",single_ligand) ~ "ligand2")) %>%
  dplyr::select(c(name,single_ligand,adj_pval,lfc,interaction)) %>%
  pivot_wider(
    id_cols = c(name, interaction),
    names_from = single_ligand,
    values_from = c(adj_pval, lfc),
    names_sep = "_"
  )

# combine single and interaction lfc and pval into one tibble
combLfc <- left_join(singleLfcWide, interLfcMin, by = c("name","interaction"))

# calculate interaction scores
interScored <- combLfc %>%
  mutate(lfc_total_interaction_model = (lfc_ligand1+lfc_ligand2+lfc_interaction),
         interaction_score = case_when(adj_pval_interaction > 0.1 ~ 0,
                                       adj_pval_interaction <= 0.1 ~ (lfc_interaction/(lfc_total_interaction_model)))) %>%
  arrange(desc(interaction_score))

# merge single condition model with interaction scores
output <- conditionLfc %>%
    select(name, condition, lfc, adj_pval) %>%
    rename(lfc_condition = lfc,
           adj_pval_condition = adj_pval) %>%
    left_join(interScored, by = c("name", "condition" = "interaction"))

#### CLASSIFY INTERACTION EFFECTS ####

# classify genes into categories
outputClassified <- output %>%
    mutate(interaction_class = case_when(
        # conditions are evaluated sequentially
        adj_pval_interaction <= 0.1 & lfc_total_interaction_model > 0 & interaction_score > 0 ~ "synergy positive",
        adj_pval_interaction <= 0.1 & lfc_total_interaction_model < 0 & interaction_score > 0 ~ "synergy negative",
        adj_pval_interaction <= 0.1 & interaction_score < 0 ~ "buffering",
        adj_pval_condition <= 0.1 ~ "none"
        ))

#### EXPORT ####

# make tibble with only significant degs based on single condition model
outputSig <- outputClassified %>%
  filter(!is.na(interaction_class))

# make tibble with only significant degs and interaction effects
outputSigInteractions <- outputSig %>%
  filter(interaction_class != "none")

# make tibble summarizing interaction classes for genes significant in each condition
outputSummary <- outputClassified %>%
    filter(!is.na(interaction_class)) %>%
    group_by(interaction, interaction_class) %>%
    summarise(n = n()) %>%
    arrange(interaction)

write_csv(outputClassified,"/data1/rudenska/EYW/git_projects/SIG13/analysis_outs/glmGamPoi/interactions_scored_v3_glmGamPoi_0.2filter.csv")
write_csv(outputSig,"/data1/rudenska/EYW/git_projects/SIG13/analysis_outs/glmGamPoi/interactions_scored_v3_glmGamPoi_0.2filter_sig.csv")
write_csv(outputSigInteractions,"/data1/rudenska/EYW/git_projects/SIG13/analysis_outs/glmGamPoi/interactions_scored_v3_glmGamPoi_0.2filter_sig_interactions.csv")
write_csv(outputSummary,"/data1/rudenska/EYW/git_projects/SIG13/analysis_outs/glmGamPoi/interactions_scored_v3_glmGamPoi_0.2filter_sig_summary.csv")