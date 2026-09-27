#!/usr/bin/env python
"""Step 02: score the final SIG13 inference model on the bulk disease datasets.

Applies the model defined in `model_core.py` to four CD4 T cell datasets and
writes one ligand-activity table per grouping (whole-sample and per-celltype):

  inflammation_atlas  human, cross-IMID atlas       (Inflammation Atlas 2026)
  amp_2023            human, rheumatoid arthritis   (AMP Phase 2)
  thomas_ibd          human, UC / CD                (Thomas et al. 2024)
  sig19_iln           mouse, Treg-depletion + aCD4  (SIG19 DTR iLN)

Each dataset is scored once at the single-cell level, then component scores are
averaged into samples several ways ("groupings") -- e.g. per patient, or per
patient x celltype. Outputs per grouping, in
`analysis_outs/03_activity_inference_model/inference_model_disease_bulk/`:

  activity_scores_<grouping>.csv    sample x ligand activity z-scores
  r2_<grouping>.csv                 per-sample ridge R2 + CV-chosen alpha (model fit QC)
  sample_metadata_<grouping>.csv    per-sample covariates + cell counts

Usage:
  python 02_score_disease_datasets.py inflammation_atlas   # one dataset
  python 02_score_disease_datasets.py all                  # all four, in series
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

# shared model module, in ../model_core
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model_core"))
import model_core as mc  # noqa: E402


# ----------------------------------------------------------------------------
# 1. Dataset-specific obs preparation
# ----------------------------------------------------------------------------
AMP_CTAP = f"{mc.IMPORTS_DIR}/external/AMP_2023/2023_AMP2_CTAP.csv"


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
#    sample_keys   obs columns joined (with "__") into the sample name; the R
#                  regressions split this back apart, so order matters
#    extra_meta    further per-sample covariates exported for the regressions
#    groupings     each adds `extra_keys` to the sample key (celltype splits) and
#                  may `subset` the cells that go into it
# ----------------------------------------------------------------------------
ATLAS_KEYS = ["studyID", "sampleID", "patientID", "disease", "sex", "age",
              "binned_age", "chemistry", "diseaseGroup"]
AMP_KEYS = ["subject_id", "biopsy_id", "joint", "sex", "treatment", "CTAP", "das28_crp3_binary"]
IBD_KEYS = ["Disease", "Patient", "Gender", "Remission_status", "Site", "Inflammation", "Treatment"]
SIG19_KEYS = ["treatment", "mouse", "cage", "sex"]

DATASETS = {
    "inflammation_atlas": dict(
        path=f"{mc.IMPORTS_DIR}/external/inflammation_atlas_2026/inflammation_atlas_cd4_subset.h5ad",
        species="human",
        var_names_key="symbol",          # var_names are Ensembl IDs in this atlas
        sample_keys=ATLAS_KEYS,
        extra_meta=["institute", "technology", "diseaseStatus", "treatmentStatus"],
        groupings=[
            dict(label="inflammation_atlas"),
            dict(label="inflammation_atlas_celltypeLevel1", extra_keys=["Level1"]),
            # Naive/non-naive split with Tregs removed. The Treg labels live in
            # Level2 -- the old pipeline filtered on Level1 (values are only
            # T_CD4_Naive / T_CD4_NonNaive), so its "_noTregs" table was in fact
            # identical to the unfiltered one.
            dict(label="inflammation_atlas_celltypeLevel1_noTregs", extra_keys=["Level1"],
                 subset=lambda obs: ~obs["Level2"].isin(["Tregs", "Tregs_activated"])),
            dict(label="inflammation_atlas_celltypeLevel2", extra_keys=["Level2"]),
        ],
    ),
    "amp_2023": dict(
        path=f"{mc.IMPORTS_DIR}/external/AMP_2023/amp_2023_cd4_processed.h5ad",
        species="human",
        prepare_obs=prepare_amp_obs,
        sample_keys=AMP_KEYS,
        extra_meta=["age", "CDAI", "das28_crp3", "das28_esr3",
                    "krenn_inflammation", "krenn_lining", "ccp_result", "disease_duration"],
        groupings=[
            dict(label="amp_2023"),
            dict(label="amp_2023_celltype", extra_keys=["cluster_name"]),
        ],
    ),
    "thomas_ibd": dict(
        path=f"{mc.IMPORTS_DIR}/external/thomas_IBD_2024/thomas_IBD_2024_cd4tcells_processed.h5ad",
        species="human",
        sample_keys=IBD_KEYS,
        extra_meta=["Age", "Batch", "Inflammation_score", "Ileum_vs_Colon",
                    "Disease_duration", "Ethnicity", "LibraryType"],
        groupings=[
            # Tregs excluded from the whole-sample scores so bulk activity reflects
            # conventional CD4 T cells, as in the original analysis
            dict(label="thomas_ibd",
                 subset=lambda obs: ~obs["final_analysis"].astype(str).str.contains("Treg")),
            dict(label="thomas_ibd_celltype", extra_keys=["final_analysis"]),
        ],
    ),
    "sig19_iln": dict(
        path=f"{mc.IMPORTS_DIR}/SIG19/scvi_outs/SIG19_DTR_CD4T_iLN_scvi.h5ad",
        species="mouse",
        sample_keys=SIG19_KEYS,
        extra_meta=["experiment", "tissue"],
        groupings=[
            dict(label="sig19_iln"),
            dict(label="sig19_iln_celltype", extra_keys=["leiden_0.5"]),
        ],
    ),
}


# ----------------------------------------------------------------------------
# 3. Scoring one dataset
# ----------------------------------------------------------------------------
def score_dataset(name, cfg, X):
    print(f"\n==== {name} ({cfg['species']}) ====", flush=True)

    net, scoring_genes = mc.build_spca_net(convert_to_human=cfg["species"] == "human")
    print(f"net: {net['source'].nunique()} components, {len(scoring_genes)} genes", flush=True)

    rna = sc.read_h5ad(cfg["path"])
    if cfg.get("var_names_key"):
        rna.var_names = rna.var[cfg["var_names_key"]].astype(str)
        rna.var_names_make_unique()
    print(f"loaded {rna.shape[0]} cells x {rna.shape[1]} genes", flush=True)

    obs = mc.score_components(rna, net, scoring_genes)
    if cfg.get("prepare_obs"):
        # merging on columns drops the index, so restore it before use
        index = obs.index
        obs = cfg["prepare_obs"](obs.reset_index(drop=True)).set_index(index)
    del rna

    for grouping in cfg["groupings"]:
        label = grouping["label"]
        sample_keys = cfg["sample_keys"] + grouping.get("extra_keys", [])
        cells = obs if "subset" not in grouping else obs[grouping["subset"](obs)]
        print(f"\n  -- {label}: {len(cells)} cells", flush=True)

        Y = mc.build_target_matrix(cells, sample_keys, component_order=X.index)
        print(f"     {Y.shape[1]} samples x {Y.shape[0]} components -> ridge", flush=True)

        activity, r2 = mc.score_ligand_activity(X, Y)
        metadata = mc.build_sample_metadata(cells, sample_keys, cfg["extra_meta"])

        for df, prefix in [(activity, "activity_scores"), (r2, "r2"), (metadata, "sample_metadata")]:
            path = f"{mc.OUT_DIR}/{prefix}_{label}.csv"
            df.to_csv(path, index=False)
            print(f"     wrote {os.path.basename(path)} {df.shape}", flush=True)
        print(f"     median R2 = {r2['r2_score'].median():.3f}, "
              f"median alpha = {r2['best_alpha'].median():.3g}", flush=True)


if __name__ == "__main__":
    requested = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(DATASETS) if requested == "all" else [requested]
    unknown = set(names) - set(DATASETS)
    if unknown:
        sys.exit(f"unknown dataset(s) {sorted(unknown)}; choose from {list(DATASETS)} or 'all'")

    os.makedirs(mc.OUT_DIR, exist_ok=True)
    X = mc.load_explanatory_matrix()
    print(f"explanatory matrix: {X.shape[0]} components x {X.shape[1]} ligand activities", flush=True)

    for name in names:
        score_dataset(name, DATASETS[name], X)

    print("\nDONE", flush=True)
