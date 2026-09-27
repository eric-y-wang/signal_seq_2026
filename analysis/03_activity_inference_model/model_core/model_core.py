#!/usr/bin/env python
"""Core of the final SIG13 ligand-activity inference model.

This module is the single definition of "the model" for the bulk analyses
(`03_inference_model_disease_bulk`, `02_inference_model_mixture_validation`).
It is a faithful port of the winning variant from the calibration in
`01_inference_model_construction_validation` (`ridge_zscore_50_weighted`, selected in
`04_calibration_model_testing.Rmd` on SIG14 + SIG26 ground truth):

  net       : SIG13 sPCA component loadings, restricted to replicate-independent
              components, top 50 target genes per component by loading, loadings
              kept as weights (mouse -> human orthologs for human datasets)
  Y (target): decoupler weighted-aggregate (waggr) component scores on gene-scaled
              log1p-normalized expression, averaged per sample, z-scored per sample
  X (design): SIG13 ligand x component explanatory matrix from
              `01_inference_model_construction_validation/01_explanatory_matrix_construction.ipynb`,
              z-scored per ligand activity
  fit       : RidgeCV per sample (alpha grid logspace(-1, 4, 500), 5-fold), coefficients
              z-scored against a 1000x label-permutation null

Contents:
  1. Paths & constants
  2. Reference net + explanatory matrix   - build_spca_net, load_explanatory_matrix
  3. Component scoring                    - score_components
  4. Target matrix                        - build_target_matrix
  5. Ridge activity model                 - score_ligand_activity
"""
import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from joblib import Parallel, delayed

import sys
from pathlib import Path
# Repo-relative paths (inputs: imports_stable/, outputs: analysis_outs/)
_here = Path(__file__).resolve().parent
REPO_DIR = str(next(p for p in [_here, *_here.parents] if (p / "imports_stable").is_dir()))
IMPORTS_DIR = f"{REPO_DIR}/imports_stable"
OUTS_DIR = f"{REPO_DIR}/analysis_outs/03_activity_inference_model"
sys.path.insert(0, f"{REPO_DIR}/functions")
import scanpy_custom as scc


# ----------------------------------------------------------------------------
# 1. Paths & constants
# ----------------------------------------------------------------------------
SPCA_DIR = f"{IMPORTS_DIR}/SIG13/analysis_outs/spca"
MODEL_DIR = f"{IMPORTS_DIR}/SIG13/analysis_outs/inference_model_final"
OUT_DIR = f"{OUTS_DIR}/inference_model_disease_bulk"

SPCA_LOADINGS = f"{SPCA_DIR}/zscore_degs_allLigands_0.1_alpha1.0_sPCA_loadings.csv"
SPCA_LM_SCORED = f"{SPCA_DIR}/lm_scored_zscore_degs_allLigands_0.1_alpha1.0_sPCA_clean.csv"
EXPLANATORY_MAT = f"{MODEL_DIR}/SIG13_waggr_scores_explanatory_mat.csv"
ACTIVITY_CLUSTERS = f"{MODEL_DIR}/SIG13_waggr_activity_clusters.csv"
# Tracked in git alongside this module (not in analysis_outs): it pins the gene
# set the model is scored on, so it is reference data rather than an output.
ORTHOLOG_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "mouse_human_ortholog_map.csv")

N_TOP_GENES = 50          # target genes kept per sPCA component
ALPHA_RANGE = np.logspace(-1, 4, 500)
N_PERMS = 1000
SAMPLE_DELIM = "__"       # sample-key delimiter; metadata values contain "_"


# ----------------------------------------------------------------------------
# 2. Reference net + explanatory matrix
# ----------------------------------------------------------------------------
def ortholog_map(mouse_genes, n_attempts=5):
    """Mouse -> human gene symbol map for `mouse_genes`, cached on disk.

    `scc.convert_mouse_genes_to_human` queries Ensembl BioMart live, which fails
    transiently and can silently change between runs. The model's gene set has to
    be fixed across datasets, so the resolved mapping is written to
    ORTHOLOG_CACHE and reused; delete that file to deliberately refresh it.

    Genes with no human homolog are cached with a null `human_gene` so they are
    not re-queried.
    """
    mouse_genes = sorted(set(mouse_genes))

    if os.path.exists(ORTHOLOG_CACHE):
        cached = pd.read_csv(ORTHOLOG_CACHE)
        if not set(mouse_genes) - set(cached["mouse_gene"]):
            return cached
        print("ortholog cache does not cover all genes; re-querying BioMart", flush=True)

    for attempt in range(1, n_attempts + 1):
        try:
            mapping = scc.convert_mouse_genes_to_human(
                pd.DataFrame({"mouse_gene": mouse_genes}), "mouse_gene")
            break
        except Exception as error:
            print(f"BioMart attempt {attempt}/{n_attempts} failed: {error}", flush=True)
            if attempt == n_attempts:
                raise
            time.sleep(15 * attempt)

    mapping.to_csv(ORTHOLOG_CACHE, index=False)
    print(f"cached ortholog map for {len(mapping)} genes -> {ORTHOLOG_CACHE}", flush=True)
    return mapping


def build_spca_net(convert_to_human):
    """Build the decoupler net of sPCA component signatures.

    Components associated with replicate effects are dropped (only those scored
    in `lm_scored_*` are kept). For human datasets the mouse gene symbols are
    mapped to human orthologs *before* the top-50 cut, matching how the model
    was calibrated on human SIG26 data.

    Returns (net, scoring_genes).
    """
    loadings = pd.read_csv(SPCA_LOADINGS)
    lm_scored = pd.read_csv(SPCA_LM_SCORED)

    good_comps = lm_scored["component"].unique().tolist()
    net = (loadings[loadings["spca_component"].isin(good_comps)]
           .rename(columns={"gene": "target", "spca_component": "source", "loading": "weight"})
           .copy())

    if convert_to_human:
        # One human symbol per mouse gene, applied as a lookup exactly as
        # `01_inference_model_construction_validation/02_calibration_input_scoring.py` does -- a join would fan a mouse gene out
        # across all of its homologs and change the top-50 gene set.
        mapping = ortholog_map(net["target"].unique())
        net["target"] = net["target"].map(dict(zip(mapping["mouse_gene"], mapping["human_gene"])))
        net = net.drop_duplicates(subset=["source", "target"])

    net = net.groupby("source", group_keys=False).apply(lambda x: x.nlargest(N_TOP_GENES, "weight"))

    # Score on the top-50 gene set itself: these are the only genes waggr
    # aggregates over, so the expression subset is restricted to them.
    return net, net["target"].dropna().unique().tolist()


def load_explanatory_matrix():
    """Load the SIG13 ligand-activity explanatory matrix and z-score it.

    Returns X (components x ligand activities), z-scored within each activity.
    """
    activities = pd.read_csv(EXPLANATORY_MAT, index_col=0).T   # components x activities
    return pd.DataFrame(StandardScaler().fit_transform(activities),
                        index=activities.index, columns=activities.columns)


# ----------------------------------------------------------------------------
# 3. Component scoring
# ----------------------------------------------------------------------------
def score_components(rna, net, scoring_genes, min_cell_fraction=0.01, verbose=True):
    """Score every sPCA component on every cell of `rna`.

    Restricts to component genes, drops genes detected in fewer than
    `min_cell_fraction` of cells, gene-scales log1p-normalized expression, then
    runs decoupler's weighted aggregate (no permutations -- p-values unused).

    Returns `rna.obs` with one `waggr_<component>` column added per component.
    """
    sub = rna[:, rna.var_names.isin(scoring_genes)].copy()
    sub.X = sub.layers["log1p_norm"]

    n_before = sub.n_vars
    sc.pp.filter_genes(sub, min_cells=int(sub.n_obs * min_cell_fraction))
    if verbose:
        print(f"  scoring genes: kept {sub.n_vars}/{n_before} "
              f"(detected in >={min_cell_fraction:.0%} of {sub.n_obs} cells)", flush=True)

    sc.pp.scale(sub)
    dc.mt.waggr(sub, net, tmin=5, times=0)

    scores = sub.obsm["score_waggr"].add_prefix("waggr_")
    scores.index = sub.obs_names
    return pd.concat([rna.obs, scores], axis=1)


# ----------------------------------------------------------------------------
# 4. Target matrix
# ----------------------------------------------------------------------------
def sample_labels(obs, sample_keys):
    """Join the sample-key columns into the composite sample name.

    Key values are stripped of stray whitespace so downstream group labels are
    clean -- e.g. the Thomas IBD healthy donors carry `Remission_status` of
    "None " (with a trailing space).

    Returns (sample name Series, cleaned key columns DataFrame).
    """
    keys = obs[list(sample_keys)].astype(str).apply(lambda col: col.str.strip())
    return keys.agg(SAMPLE_DELIM.join, axis=1), keys


def build_target_matrix(obs, sample_keys, component_order):
    """Average component scores within each sample and z-score per sample.

    `sample_keys` are joined with SAMPLE_DELIM into the sample name that the
    downstream R regressions split back apart. Rows are reindexed to
    `component_order` (the explanatory matrix's component order) -- the ridge fit
    pairs X and Y row-wise, so this alignment is required, not cosmetic.

    Returns Y (components x samples).
    """
    waggr_cols = [c for c in obs.columns if c.startswith("waggr_")]
    labels, _ = sample_labels(obs, sample_keys)
    y = (obs.assign(sample=labels)
            .groupby("sample", observed=True)[waggr_cols]
            .mean()
            .T)
    y.index = y.index.str.removeprefix("waggr_")

    missing = set(component_order) - set(y.index)
    if missing:
        raise ValueError(f"components scored in SIG13 but absent from this dataset: {sorted(missing)}")
    y = y.loc[component_order]

    return pd.DataFrame(StandardScaler().fit_transform(y), index=y.index, columns=y.columns)


def build_sample_metadata(obs, sample_keys, extra_cols=()):
    """Per-sample metadata table: the sample key, its constituent columns, cell
    count, and any `extra_cols` (patient-level covariates needed by the
    regressions, e.g. Age, Batch, CTAP, disease-activity scores).

    Replaces the regressions' old dependency on the ~500 MB per-cell
    `analysis_outs/projection/*_obs.csv` exports.
    """
    labels, keys = sample_labels(obs, sample_keys)
    return (obs.drop(columns=list(sample_keys))          # replaced by cleaned copies
               .join(keys)
               .assign(sample=labels)
               .groupby("sample", observed=True)
               .agg(n_cells=("sample", "size"),
                    **{c: (c, "first") for c in dict.fromkeys(list(sample_keys) + list(extra_cols))})
               .reset_index())


# ----------------------------------------------------------------------------
# 5. Ridge activity model
# ----------------------------------------------------------------------------
def _ridge_permutation_zscore(y_obs, X, alpha_range, n_perms, seed):
    """Fit RidgeCV for one sample, then z-score its coefficients against a
    label-permutation null (CytoSig-style activity z-scores)."""
    cv_model = RidgeCV(alphas=alpha_range, fit_intercept=True, cv=5).fit(X, y_obs)
    best_alpha, beta_obs, r2_obs = cv_model.alpha_, cv_model.coef_, cv_model.score(X, y_obs)

    rng = np.random.default_rng(seed)
    Y_perm = np.array([rng.permutation(y_obs) for _ in range(n_perms)]).T
    beta_perms = Ridge(alpha=best_alpha, fit_intercept=True, random_state=67).fit(X, Y_perm).coef_

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (beta_obs - beta_perms.mean(axis=0)) / beta_perms.std(axis=0)
    return z, r2_obs, best_alpha


def score_ligand_activity(X, Y, alpha_range=ALPHA_RANGE, n_perms=N_PERMS, n_jobs=-1):
    """Fit the ridge activity model independently per sample (column of Y).

    Each sample's permutation null is seeded from its column position, so runs
    are bitwise reproducible while samples still draw independent permutations.

    Returns (activity_scores: samples x ligand activities, r2: per-sample R2 and
    CV-chosen ridge alpha).
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(_ridge_permutation_zscore)(Y.values[:, i], X.values, alpha_range, n_perms, seed=i)
        for i in range(Y.shape[1])
    )
    zs, r2s, alphas = zip(*results)

    activity = pd.DataFrame(dict(zip(Y.columns, zs)), index=X.columns).T
    activity.index.name = "sample"
    return activity.reset_index(), pd.DataFrame(
        {"sample": Y.columns, "r2_score": r2s, "best_alpha": alphas})
