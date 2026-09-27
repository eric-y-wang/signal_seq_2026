#!/usr/bin/env python
"""Step 02 of the publication inference-model pipeline (analysis/03_activity_inference_model/01_inference_model_construction_validation):
score SIG14 (mouse) and SIG26 (human, 6h & 24h) pseudobulk expression against the
SIG13 sPCA-component signatures built in 01_explanatory_matrix_construction.ipynb,
using 8 Ridge model variants each (top-50/all genes x weighted/unweighted net,
each fit with z-scored and raw-coefficient activity).

See README.md in this folder for a full description of the scoring approach.

Outputs (per dataset, written to analysis_outs/03_activity_inference_model/inference_model_calibration/):
  activity_combined_{SIG14,SIG26-6h,SIG26-24h}.csv
  r2_combined_{SIG14,SIG26-6h,SIG26-24h}.csv

Contents:
  1. Ridge activity model       - _ridge_permutation_zscore, score_ligand_activity
  2. sPCA net + explanatory mat - build_spca_net, build_ligand_explanatory_matrix
  3. Pseudobulk scoring         - score_pseudobulk_components, build_target_matrix
  4. Dataset pseudobulk prep    - prepare_pseudobulk_sig14, prepare_pseudobulk_sig26
  5. Orchestration              - score_dataset, __main__
"""
import warnings
warnings.filterwarnings("ignore")

import os
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
# Paths & shared constants
# ----------------------------------------------------------------------------
SPCA_DIR = f"{IMPORTS_DIR}/SIG13/analysis_outs/spca"
MODEL_DIR = f"{IMPORTS_DIR}/SIG13/analysis_outs/inference_model_final"
OUT_DIR = f"{OUTS_DIR}/inference_model_calibration"
os.makedirs(OUT_DIR, exist_ok=True)

SIG14_PROC = f"{IMPORTS_DIR}/SIG14/processing_outs"
SIG26_PROC = f"{IMPORTS_DIR}/SIG26/processing_outs"

ALPHA_RANGE = np.logspace(-1, 3, 100)

# ----------------------------------------------------------------------------
# 1. Ridge activity model: fit RidgeCV per condition, z-score coefficients
#    against a permuted null (or return raw coefficients if zscore_coeffs=False)
# ----------------------------------------------------------------------------
def _ridge_permutation_zscore(y_obs, X, alpha_range, n_perms=1000, zscore_coeffs=True):
    cv_model = RidgeCV(alphas=alpha_range, fit_intercept=True, cv=5).fit(X, y_obs)
    best_alpha, beta_obs, r2_obs = cv_model.alpha_, cv_model.coef_, cv_model.score(X, y_obs)

    rng = np.random.default_rng()
    Y_perm_matrix = np.array([rng.permutation(y_obs) for _ in range(n_perms)]).T

    perm_model = Ridge(alpha=best_alpha, fit_intercept=True, random_state=67).fit(X, Y_perm_matrix)
    beta_perms = perm_model.coef_

    with np.errstate(divide='ignore', invalid='ignore'):
        z = (beta_obs - np.mean(beta_perms, axis=0)) / np.std(beta_perms, axis=0) if zscore_coeffs else beta_obs

    return z, r2_obs, best_alpha


def score_ligand_activity(X_mat, Y_mat, alpha_range=ALPHA_RANGE, n_jobs=-1, verbose=0, zscore_coeffs=True):
    """Fit _ridge_permutation_zscore independently per column (condition) of
    Y_mat, in parallel. Returns (ligand x condition activity df, condition R2 series).

    X_mat and Y_mat are both indexed by sPCA component; the ridge fit pairs them
    row-wise, so Y is reindexed onto X's component order before dropping to numpy."""
    if isinstance(X_mat, pd.DataFrame) and isinstance(Y_mat, pd.DataFrame):
        missing = X_mat.index.difference(Y_mat.index)
        if len(missing):
            raise ValueError(f"components in the explanatory matrix but not in the "
                             f"target matrix: {sorted(missing)}")
        Y_mat = Y_mat.loc[X_mat.index]

    X = X_mat.values if isinstance(X_mat, pd.DataFrame) else X_mat
    Y = Y_mat.values if isinstance(Y_mat, pd.DataFrame) else Y_mat
    ligand_names = X_mat.columns if isinstance(X_mat, pd.DataFrame) else np.arange(X.shape[1])
    cond_names = Y_mat.columns if isinstance(Y_mat, pd.DataFrame) else np.arange(Y.shape[1])

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_ridge_permutation_zscore)(Y[:, i], X, alpha_range, zscore_coeffs=zscore_coeffs)
        for i in range(len(cond_names))
    )

    zs, r2s, alphas = zip(*results)
    return pd.DataFrame(dict(zip(cond_names, zs)), index=ligand_names), pd.Series(r2s, index=cond_names)


# ----------------------------------------------------------------------------
# 2. sPCA reference net + ligand explanatory matrix (built once, shared across datasets)
# ----------------------------------------------------------------------------
def build_spca_net(convert_to_human):
    """Load the SIG13 sPCA loadings + significance filter, format for decoupler,
    and derive the 4 net variants (top-50/all genes x weighted/unweighted).
    convert_to_human=True maps the mouse gene symbols in the net to human
    orthologs, needed to score human datasets against this mouse-derived
    component net."""
    spca_components = pd.read_csv(f"{SPCA_DIR}/zscore_degs_allLigands_0.1_alpha1.0_sPCA_loadings.csv")
    lm_scored = pd.read_csv(f"{SPCA_DIR}/lm_scored_zscore_degs_allLigands_0.1_alpha1.0_sPCA_clean.csv")

    good_comps = lm_scored['component'].unique().tolist()
    spca_components = spca_components[spca_components['spca_component'].isin(good_comps)]
    net = spca_components.rename(columns={'gene': 'target', 'spca_component': 'source', 'loading': 'weight'})

    if convert_to_human:
        net = scc.convert_mouse_genes_to_human(net, 'target')
        net.drop('target', axis=1, inplace=True)
        net.rename(columns={'human_gene': 'target'}, inplace=True)
        net.drop_duplicates(subset=['source', 'target'], inplace=True)

    nets = {
        '50_weighted':    net.groupby('source', group_keys=False).apply(lambda x: x.nlargest(50, 'weight')),
        '50_unweighted':  net.groupby('source', group_keys=False).apply(lambda x: x.nlargest(50, 'weight')).drop('weight', axis=1),
        'all_weighted':   net.copy(),
        'all_unweighted': net.drop('weight', axis=1),
    }
    comp_genes = net['target'].unique().tolist()
    return nets, comp_genes


def build_ligand_explanatory_matrix():
    """SIG13 waggr ligand-activity explanatory matrix (X): ligands x sPCA
    components, z-scored per component across ligands."""
    ligand_scores = pd.read_csv(f"{MODEL_DIR}/SIG13_waggr_scores_explanatory_mat.csv", index_col=0)
    ligands_df = ligand_scores.T
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(ligands_df), columns=ligands_df.columns, index=ligands_df.index)
    return X_scaled, scaler


# ----------------------------------------------------------------------------
# 3. Score sPCA components onto a dataset's pseudobulk, and build the ridge
#    target matrix (Y) from those scores
# ----------------------------------------------------------------------------
def score_pseudobulk_components(adata, net, scoring_genes):
    """Weighted-aggregate (decoupler waggr) score each sPCA component onto a
    pseudobulk AnnData, using only genes in scoring_genes and gene-scaled
    log1p-normalized expression."""
    adata = adata.copy()
    adata_sub = adata[:, adata.var_names.isin(scoring_genes)].copy()
    adata_sub.X = adata_sub.layers['log1p_norm']
    sc.pp.scale(adata_sub)
    dc.mt.waggr(adata_sub, net, tmin=5, times=0)
    adata.obsm['score_waggr'] = adata_sub.obsm['score_waggr'].copy()
    score_df = adata_sub.obsm['score_waggr']
    score_df.columns = [f'waggr_{col}' for col in score_df.columns]
    adata.obs = pd.concat([adata_sub.obs, score_df], axis=1)
    return adata


def build_target_matrix(adata_scored, sample_key_cols, scaler):
    """Average waggr component scores within each sample (defined by
    sample_key_cols) and per-component z-score across samples, to build the
    ridge model's target matrix (Y)."""
    y_df = (adata_scored.obs
            .assign(sample=lambda x: x[sample_key_cols].astype(str).agg('_'.join, axis=1))
            .groupby('sample')[[c for c in adata_scored.obs.columns if c.startswith('waggr_')]]
            .mean()
            .T)
    y_df.index = y_df.index.str.replace('waggr_', '')
    return pd.DataFrame(scaler.fit_transform(y_df), columns=y_df.columns, index=y_df.index)


# ----------------------------------------------------------------------------
# 4. Dataset-specific pseudobulk prep
# ----------------------------------------------------------------------------
def _load_normalized_counts(counts_path, feature_names_path):
    """Load a UMI count matrix, restrict to genes in feature_names_path, and
    build a log1p-normalized AnnData (samples x genes)."""
    counts = pd.read_csv(counts_path, index_col=0)
    feature_names = pd.read_csv(feature_names_path, index_col=0)

    counts_filtered = counts.loc[counts.index.isin(feature_names.index), :]
    counts_filtered = counts_filtered.merge(feature_names, left_index=True, right_index=True)
    counts_filtered = counts_filtered.set_index('gene').drop(columns=['category'])

    rna = sc.AnnData(counts_filtered.T)
    rna.var_names_make_unique()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    rna.layers['log1p_norm'] = rna.X.copy()
    return rna


def _aggregate_pseudobulk(rna, obs_df, group_cols):
    """Attach per-sample metadata (obs_df, keyed on 'sample') and average
    log1p-normalized expression within each group_cols group to build a
    pseudobulk AnnData."""
    rna.obs = rna.obs.merge(obs_df, left_index=True, right_on='sample')
    rna_pb = sc.get.aggregate(rna, by=group_cols, func='mean', layer='log1p_norm')
    rna_pb.layers['log1p_norm'] = rna_pb.layers['mean'].copy()
    return rna_pb


def prepare_pseudobulk_sig14():
    """SIG14 (mouse): sample key ligand1_ligand2_mouse_well_library_project,
    pseudobulked per condition/mouse/well."""
    rna = _load_normalized_counts(f"{SIG14_PROC}/count_matrix_umiDeDup_SIG14.csv",
                                   f"{SIG14_PROC}/featureNames_SIG14.csv")
    obs_df = pd.DataFrame({"sample": rna.obs_names})
    obs_df[['ligand1', 'ligand2', 'mouse', 'well', 'library', 'project']] = obs_df['sample'].str.split('_', expand=True)
    obs_df['condition'] = obs_df['ligand1'] + '_' + obs_df['ligand2']
    return _aggregate_pseudobulk(rna, obs_df, group_cols=['condition', 'ligand1', 'ligand2', 'mouse', 'well'])


def prepare_pseudobulk_sig26(tp):
    """SIG26 (human), timepoint tp in {'6h','24h'}: sample metadata from
    processed_metadata_SIG26-{tp}.csv, pseudobulked per condition/replicate/well."""
    rna = _load_normalized_counts(f"{SIG26_PROC}/count_matrix_umiDeDup_SIG26-{tp}.csv",
                                   f"{SIG26_PROC}/featureNames_SIG26-{tp}.csv")
    meta = pd.read_csv(f"{SIG26_PROC}/processed_metadata_SIG26-{tp}.csv")
    obs_df = (meta[['sample_ID', 'condition', 'ligand1', 'ligand2', 'replicate', 'well']]
              .rename(columns={'sample_ID': 'sample'}))
    return _aggregate_pseudobulk(rna, obs_df, group_cols=['condition', 'ligand1', 'ligand2', 'replicate', 'well'])


# ----------------------------------------------------------------------------
# 5. Orchestration: for each dataset, score all 4 net variants and fit both
#    zscore/raw-coefficient ridge activity for each -> 8 models total
# ----------------------------------------------------------------------------
def score_dataset(dataset_label, rna_pb, sample_key_cols, nets, comp_genes, X_scaled, scaler):
    print(f"==== {dataset_label} ====", flush=True)
    print("pseudobulk:", rna_pb.shape, flush=True)

    activity_rows, r2_rows = [], []
    for net_key, net in nets.items():
        print(f"  scoring net={net_key}", flush=True)
        scored = score_pseudobulk_components(rna_pb, net, comp_genes)
        Y_scaled = build_target_matrix(scored, sample_key_cols, scaler)

        for zscore_coeffs, kind in [(True, 'zscore'), (False, 'coeff')]:
            activity, r2 = score_ligand_activity(X_scaled, Y_scaled, zscore_coeffs=zscore_coeffs)
            model_name = f"ridge_{kind}_{net_key}"
            activity_rows.append(activity.T.reset_index(names='sample').assign(model=model_name))
            r2_rows.append(pd.DataFrame({'sample': r2.index, 'r2_score': r2.values, 'model': model_name}))

    activity_combined = pd.concat(activity_rows, ignore_index=True)
    r2_combined = pd.concat(r2_rows, ignore_index=True)

    activity_combined.to_csv(f"{OUT_DIR}/activity_combined_{dataset_label}.csv", index=False)
    r2_combined.to_csv(f"{OUT_DIR}/r2_combined_{dataset_label}.csv", index=False)
    print(f"saved activity_combined_{dataset_label}.csv {activity_combined.shape} "
          f"and r2_combined_{dataset_label}.csv {r2_combined.shape}", flush=True)


if __name__ == "__main__":
    # SIG14 (mouse): net stays in mouse gene symbols, no ortholog conversion needed
    nets_mouse, comp_genes_mouse = build_spca_net(convert_to_human=False)
    X_scaled_mouse, scaler_mouse = build_ligand_explanatory_matrix()
    score_dataset(
        "SIG14", prepare_pseudobulk_sig14(),
        sample_key_cols=['condition', 'mouse', 'well'],
        nets=nets_mouse, comp_genes=comp_genes_mouse,
        X_scaled=X_scaled_mouse, scaler=scaler_mouse,
    )

    # SIG26 (human, 6h): net mapped mouse -> human orthologs, shared across timepoints
    nets_human, comp_genes_human = build_spca_net(convert_to_human=True)
    for tp in ["6h"]:
        X_scaled_human, scaler_human = build_ligand_explanatory_matrix()
        score_dataset(
            f"SIG26-{tp}", prepare_pseudobulk_sig26(tp),
            sample_key_cols=['ligand1', 'ligand2', 'replicate', 'well'],
            nets=nets_human, comp_genes=comp_genes_human,
            X_scaled=X_scaled_human, scaler=scaler_human,
        )

    print("DONE", flush=True)
