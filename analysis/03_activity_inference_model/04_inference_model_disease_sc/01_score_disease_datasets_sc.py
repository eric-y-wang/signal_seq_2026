#!/usr/bin/env python
"""Step 01: score the SIG13 inference model per cell on the disease datasets (GPU).

The single-cell counterpart of `03_inference_model_disease_bulk/02_score_disease_datasets.py`.
Same model, same four datasets, same reference net -- but every cell gets its own
ligand-activity z-scores instead of being averaged into a sample first:

  inflammation_atlas  human, cross-IMID atlas       1,505,203 cells
  amp_2023            human, rheumatoid arthritis      55,432 cells
  thomas_ibd          human, UC / CD                  145,704 cells
  sig19_iln           mouse, Treg-depletion + aCD4     19,898 cells

The dataset registry is defined here rather than imported from the bulk script, so
this folder stands on its own. It is deliberately simpler than bulk's: there is no
aggregation at single-cell resolution, so bulk's `sample_keys` / `extra_meta` /
`groupings` structure collapses to one flat `obs_columns` list per dataset. Those
columns -- the sample keys, the per-sample covariates, and every column a bulk
grouping split or filtered on (`Level1`, `Level2`, `cluster_name`,
`final_analysis`, `leiden_1.0`) -- are carried into the output, so the bulk
groupings are recoverable by filtering rather than by rescoring.

Outputs per dataset, in `analysis_outs/03_activity_inference_model/inference_model_disease_sc/`:

  activity_scores_<dataset>.parquet     cell x 38 ligand activities, + obs covariates
  component_scores_<dataset>.parquet    cell x 68 sPCA component scores (the target Y)
  ridge_diagnostics_<dataset>.parquet   per-cell ridge R2 and CV-selected alpha

Usage:
  python 01_score_disease_datasets_sc.py sig19_iln    # one dataset
  python 01_score_disease_datasets_sc.py all          # all four, in series

Requires a GPU; run via `02_run_sc_scoring.sh`.
"""
import warnings
warnings.filterwarnings("ignore")

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

# shared model module, in ../model_core
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model_core"))
import model_core_sc as msc  # noqa: E402


# ----------------------------------------------------------------------------
# 1. Dataset-specific obs preparation
# ----------------------------------------------------------------------------
AMP_CTAP = f"{msc.IMPORTS_DIR}/external/AMP_2023/2023_AMP2_CTAP.csv"


def prepare_amp_obs(obs):
    """Attach AMP cell-type-abundance phenotypes (CTAP) and binarize disease
    activity (DAS28-CRP3 >= 3.2 = high), as in the original AMP analysis."""
    ctap = pd.read_csv(AMP_CTAP)
    obs = obs.merge(ctap, on=["subject_id", "biopsy_id"], how="left")
    obs["das28_crp3_binary"] = np.select(
        [obs["treatment"] == "osteoarthritis control",
         obs["das28_crp3"] >= 3.2,
         obs["das28_crp3"] < 3.2],
        ["control", "high", "low"], default=None)
    obs["CTAP"] = obs["CTAP"].where(obs["treatment"] != "osteoarthritis control", "control")
    return obs


# ----------------------------------------------------------------------------
# 2. Dataset registry
#
#    var_names_key   var column holding gene symbols, where var is keyed on
#                    something else (the atlas is keyed on Ensembl ID)
#    prepare_obs     optional obs transform applied before anything is carried
#    obs_columns     covariates copied into the activity table. Includes every
#                    column a bulk grouping split or filtered on, so the bulk
#                    groupings reduce to filters on the output.
# ----------------------------------------------------------------------------
DATASETS = {
    "inflammation_atlas": dict(
        path=f"{msc.IMPORTS_DIR}/external/inflammation_atlas_2026/inflammation_atlas_cd4_subset.h5ad",
        species="human",
        var_names_key="symbol",
        obs_columns=["studyID", "sampleID", "patientID", "disease", "sex", "age",
                     "binned_age", "chemistry", "diseaseGroup", "institute",
                     "technology", "diseaseStatus", "treatmentStatus",
                     "Level1", "Level2"],
    ),
    "amp_2023": dict(
        path=f"{msc.IMPORTS_DIR}/external/AMP_2023/amp_2023_cd4_processed.h5ad",
        species="human",
        prepare_obs=prepare_amp_obs,
        obs_columns=["subject_id", "biopsy_id", "joint", "sex", "treatment", "CTAP",
                     "das28_crp3_binary", "age", "CDAI", "das28_crp3", "das28_esr3",
                     "krenn_inflammation", "krenn_lining", "ccp_result",
                     "disease_duration", "cluster_name"],
    ),
    "thomas_ibd": dict(
        path=f"{msc.IMPORTS_DIR}/external/thomas_IBD_2024/thomas_IBD_2024_cd4tcells_processed.h5ad",
        species="human",
        obs_columns=["Disease", "Patient", "Gender", "Remission_status", "Site",
                     "Inflammation", "Treatment", "Age", "Batch",
                     "Inflammation_score", "Ileum_vs_Colon", "Disease_duration",
                     "Ethnicity", "LibraryType", "final_analysis"],
    ),
    "sig19_iln": dict(
        path=f"{msc.IMPORTS_DIR}/SIG19/scvi_outs/SIG19_DTR_CD4T_iLN_scvi.h5ad",
        species="mouse",
        obs_columns=["treatment", "mouse", "cage", "sex", "experiment", "tissue",
                     "leiden_0.5"],
    ),
}


# ----------------------------------------------------------------------------
# 3. Scoring one dataset
# ----------------------------------------------------------------------------
def score_dataset(name, cfg, X, ops):
    print(f"\n==== {name} ({cfg['species']}) ====", flush=True)
    t_start = time.time()

    net, scoring_genes = msc.build_spca_net(convert_to_human=cfg["species"] == "human")
    print(f"net: {net['source'].nunique()} components, {len(scoring_genes)} genes", flush=True)

    expression, obs, gene_names = msc.read_component_expression(
        cfg["path"], scoring_genes, var_names_key=cfg.get("var_names_key"))

    if cfg.get("prepare_obs"):
        # merging on columns drops the index, so restore it before use
        index = obs.index
        obs = cfg["prepare_obs"](obs.reset_index(drop=True)).set_index(index)

    components = msc.score_components_gpu(expression, obs, gene_names, net)
    del expression
    gc.collect()

    Y = msc.build_cell_target_matrix(components, component_order=X.index)
    print(f"  target: {Y.shape[0]} components x {Y.shape[1]} cells -> ridge", flush=True)

    activity, diagnostics = msc.score_ligand_activity_gpu(X, Y, ops=ops)

    # attach the covariates so the table stands on its own
    wanted = cfg["obs_columns"]
    missing = [c for c in wanted if c not in obs.columns]
    if missing:
        # a silently dropped covariate would only surface much later, in a
        # regression that cannot find its variable
        raise KeyError(f"{name}: obs columns declared in the registry but absent "
                       f"from the dataset: {missing}")
    covariates = obs[wanted].reset_index(drop=True)
    activity = pd.concat([activity, covariates], axis=1)

    components = components.copy()
    components.columns = [str(c) for c in components.columns]
    components.index.name = "cell_id"
    components = components.reset_index()

    os.makedirs(msc.OUT_DIR, exist_ok=True)
    for df, prefix in [(activity, "activity_scores"),
                       (components, "component_scores"),
                       (diagnostics, "ridge_diagnostics")]:
        path = f"{msc.OUT_DIR}/{prefix}_{name}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        print(f"  wrote {os.path.basename(path)} {df.shape} "
              f"({os.path.getsize(path) / 1e6:.0f} MB)", flush=True)

    print(f"  median R2 = {diagnostics['r2_score'].median():.3f} | "
          f"median alpha = {diagnostics['alpha'].median():.3g} | "
          f"total {time.time() - t_start:.0f}s", flush=True)

    del Y, activity, components, diagnostics, obs
    gc.collect()


if __name__ == "__main__":
    requested = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(DATASETS) if requested == "all" else [requested]
    unknown = set(names) - set(DATASETS)
    if unknown:
        sys.exit(f"unknown dataset(s) {sorted(unknown)}; "
                 f"choose from {list(DATASETS)} or 'all'")

    os.makedirs(msc.OUT_DIR, exist_ok=True)
    X = msc.load_explanatory_matrix()
    print(f"explanatory matrix: {X.shape[0]} components x {X.shape[1]} ligand activities",
          flush=True)

    # X is shared by every cell of every dataset, so the ridge operators are built once
    ops = msc.precompute_ridge_operators(X, n_splits=5)
    print(f"ridge operators: {len(ops['alphas'])} alphas x 5 folds precomputed", flush=True)

    for name in names:
        score_dataset(name, DATASETS[name], X, ops)

    print("\nDONE", flush=True)
