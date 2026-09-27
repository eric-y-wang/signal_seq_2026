#!/usr/bin/env python
"""Single-cell port of the final SIG13 ligand-activity inference model (GPU).

This module is self-contained: it defines the net, the explanatory matrix and the
ridge semantics itself rather than importing them from `model_core.py`. It is still
a faithful port of that model -- the net construction and reference data are
identical -- but the two modules share no code, which is what allows the alpha grid
here to differ (see section 1).

Against the bulk pipeline, what changes is *where the work happens* and *what a
"sample" is*:

  bulk                                        single cell (here)
  ------------------------------------------  ----------------------------------------
  waggr on CPU (`dc.mt.waggr`)                waggr on GPU (`rsc.dcg.waggr`)
  component scores averaged into samples      no aggregation -- every cell is its own
  z-scored per sample                         z-scored per cell (same operation)
  one `RidgeCV` + permutation null per        one batched GPU solve for *all* cells
  sample (~1e2-1e3 fits)                      (~1e6 fits, as matmuls)
  permutation null per sample                 same null, batched over cells as well
                                              as permutations

Two facts make the per-cell fit tractable rather than 1.5M independent `sklearn`
calls; both are consequences of the model's shape, not approximations.

1. **The design matrix is shared.** X (68 components x 38 activities) is identical
   for every cell -- only the response y (that cell's 68 component scores) varies.
   Ridge with an intercept is closed-form, so

       beta(alpha) = H(alpha) @ y_centered,   H(alpha) = (Xc' Xc + alpha I)^-1 Xc'

   where `H(alpha)` is a fixed 38x68 operator. Every cell's fit is one matmul
   against a shared operator, so all cells fit as a single batched GEMM. The same
   holds for the CV folds: `RidgeCV(cv=5)` uses non-shuffled `KFold`, so the folds
   are the *same rows* for every cell and all 200x5 fold operators precompute once.

2. **The permutation null batches the same way.** Bulk already solves all 1000 of a
   sample's permutations at once (`Ridge().fit(X, Y_perm)` with a 68 x 1000
   response). Since the fit at a chosen alpha is just `beta = H @ y`, that solve is
   a GEMM, and the batch extends over cells too: `(b, m, n) @ (n, p)` returns the
   coefficients for `b` permutations of `m` cells in one call. Each cell draws its
   own independent permutation set, as bulk does per sample.

The permutation RNG is seeded from the cell-chunk offset and the alpha index, so
scores are reproducible run to run for a given `cell_chunk` (bulk seeds per sample
from its column index, to the same end).

Contents:
  1. Paths & constants
  2. Reference net + explanatory matrix   - build_spca_net, load_explanatory_matrix
  3. Expression loading                   - read_component_expression
  4. Component scoring (GPU)              - score_components_gpu
  5. Per-cell target matrix               - build_cell_target_matrix
  6. Batched ridge (GPU)                  - precompute_ridge_operators,
                                            score_ligand_activity_gpu
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.utils import make_index_unique
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# 1. Paths & constants
#
# This module is self-contained: it defines the model rather than importing it
# from `model_core.py`. The two modules still share upstream
# *data* -- the sPCA loadings and the explanatory matrix, both produced by other
# analyses and owned by neither -- but no longer share code or constants. That
# separation is what lets the alpha grid below differ from bulk's without
# silently changing bulk's results, since bulk's `score_ligand_activity` reads
# the same constant.
#
# Reference inputs are read from the repo's imports_stable/ snapshot and outputs
# are written under the repo's analysis_outs/.
# ----------------------------------------------------------------------------
# Repo-relative paths (inputs: imports_stable/, outputs: analysis_outs/)
_here = Path(__file__).resolve().parent
REPO_DIR = str(next(p for p in [_here, *_here.parents] if (p / "imports_stable").is_dir()))
IMPORTS_DIR = f"{REPO_DIR}/imports_stable"
OUTS_DIR = f"{REPO_DIR}/analysis_outs/03_activity_inference_model"
SPCA_DIR = f"{IMPORTS_DIR}/SIG13/analysis_outs/spca"
MODEL_DIR = f"{IMPORTS_DIR}/SIG13/analysis_outs/inference_model_final"
OUT_DIR = f"{OUTS_DIR}/inference_model_disease_sc"

SPCA_LOADINGS = f"{SPCA_DIR}/zscore_degs_allLigands_0.1_alpha1.0_sPCA_loadings.csv"
SPCA_LM_SCORED = f"{SPCA_DIR}/lm_scored_zscore_degs_allLigands_0.1_alpha1.0_sPCA_clean.csv"
EXPLANATORY_MAT = f"{MODEL_DIR}/SIG13_waggr_scores_explanatory_mat.csv"
# Tracked in git alongside this module: it pins the gene set the model is scored
# on, so it is reference data rather than an output. It is the same file
# `model_core.py` reads. Delete it to deliberately refresh from BioMart.
ORTHOLOG_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "mouse_human_ortholog_map.csv")

N_TOP_GENES = 50          # target genes kept per sPCA component
N_PERMS = 1000            # permutation draws per cell
TMIN = 5                  # decoupler target floor

# Alpha grid for the per-cell ridge -- extends past the bulk model's
# `logspace(-1, 4, 500)` to 1e5. A single cell's `y` is far noisier than a sample
# average, so cross-validation routinely wants more regularization: with a 1e3
# ceiling (the calibration grid, `logspace(-1, 3, 100)`), about a third of human
# cells pegged it. Extending the top is the point of diverging from bulk here.
ALPHA_RANGE = np.logspace(-1, 5, 200)

LAYER = "log1p_norm"
READ_CHUNK = 20_000       # rows per CSR read block
CELL_CHUNK = 200_000      # cells per ridge batch on GPU
PERM_BUDGET = 1_000_000   # max (permutations x cells) held at once in the null


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

    # Imported lazily: only this refresh path needs `functions/`, and the cache is
    # committed alongside the module, so the ordinary path has no dependency on it.
    import sys
    sys.path.insert(0, f"{REPO_DIR}/functions")
    import scanpy_custom as scc

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
    mapped to human orthologs *before* the top-50 cut, matching how the model was
    calibrated on human SIG26 data.

    Returns (net, scoring_genes).
    """
    loadings = pd.read_csv(SPCA_LOADINGS)
    lm_scored = pd.read_csv(SPCA_LM_SCORED)

    good_comps = lm_scored["component"].unique().tolist()
    net = (loadings[loadings["spca_component"].isin(good_comps)]
           .rename(columns={"gene": "target", "spca_component": "source", "loading": "weight"})
           .copy())

    if convert_to_human:
        # One human symbol per mouse gene, applied as a lookup -- a join would fan
        # a mouse gene out across all of its homologs and change the top-50 set.
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
# 3. Expression loading
# ----------------------------------------------------------------------------
def read_component_expression(path, scoring_genes, var_names_key=None,
                              layer=LAYER, chunk=READ_CHUNK, verbose=True):
    """Read `layer` for the component genes only, densified, plus full `obs`.

    Reads the CSR layer in row blocks and densifies straight into the
    `scoring_genes` subset. The destination is small (~1.5M x 1465 float32 =
    8.8 GB for the atlas) because the net is only the top-50 genes per component,
    so this avoids both a slow global CSR column-slice and loading the other
    layers -- some of these files carry several dense layers and are ~50 GB.

    `var_names_key` selects the var column holding gene symbols (the inflammation
    atlas keys var by Ensembl ID); names are made unique exactly as
    `var_names_make_unique` would, so a duplicated symbol resolves to its first
    occurrence, matching the bulk pipeline.

    Returns (X dense float32 (cells x genes), obs, gene_names).
    """
    t0 = time.time()
    with h5py.File(path, "r") as f:
        obs = ad.io.read_elem(f["obs"])
        var = ad.io.read_elem(f["var"])

        names = pd.Index(var[var_names_key].astype(str)) if var_names_key else var.index
        names = make_index_unique(pd.Index(names))

        keep = np.flatnonzero(names.isin(scoring_genes))
        gene_names = names[keep]
        if verbose:
            print(f"  genes: {len(keep)}/{len(scoring_genes)} of the net present "
                  f"in {len(names)} var entries", flush=True)

        grp = f[f"layers/{layer}"]
        n_obs, n_var = len(obs), len(names)
        indptr = grp["indptr"][:]

        X = np.zeros((n_obs, len(keep)), dtype=np.float32)
        for start in range(0, n_obs, chunk):
            end = min(start + chunk, n_obs)
            lo, hi = indptr[start], indptr[end]
            block = sp.csr_matrix(
                (grp["data"][lo:hi], grp["indices"][lo:hi], indptr[start:end + 1] - lo),
                shape=(end - start, n_var))
            X[start:end] = block[:, keep].toarray()
            del block
        del indptr

    if verbose:
        print(f"  loaded {X.shape} in {time.time() - t0:.0f}s "
              f"({X.nbytes / 1e9:.1f} GB dense)", flush=True)
    gc.collect()
    return X, obs, gene_names


# ----------------------------------------------------------------------------
# 4. Component scoring (GPU)
# ----------------------------------------------------------------------------
def score_components_gpu(X, obs, gene_names, net, min_cell_fraction=0.01, verbose=True):
    """Score every sPCA component on every cell, on GPU.

    Mirrors `model_core.score_components` step for step -- drop genes detected in
    fewer than `min_cell_fraction` of cells (scanpy keeps `n_cells >= threshold`),
    gene-scale (zero-centred, no clipping, statistics over all cells), then run
    the weighted aggregate -- with `rsc.dcg.waggr` in place of `dc.mt.waggr`.

    `empty=False` is not the decoupler default and matters: the default drops
    all-zero observations, which would silently misalign the cell axis against
    `obs`.

    Returns a DataFrame of per-cell component scores (cells x components).
    """
    import cupy as cp
    import rapids_singlecell as rsc

    n_before = X.shape[1]
    threshold = int(X.shape[0] * min_cell_fraction)
    n_cells_per_gene = np.count_nonzero(X, axis=0)
    keep = n_cells_per_gene >= threshold
    X, gene_names = X[:, keep], gene_names[keep]
    if verbose:
        print(f"  scoring genes: kept {X.shape[1]}/{n_before} "
              f"(detected in >={min_cell_fraction:.0%} of {X.shape[0]} cells)", flush=True)

    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=obs.index),
                       var=pd.DataFrame(index=gene_names))

    t0 = time.time()
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.scale(adata)
    if verbose:
        free, total = cp.cuda.Device(0).mem_info
        print(f"  to GPU + scale: {time.time() - t0:.0f}s "
              f"(GPU free {free / 1e9:.0f}/{total / 1e9:.0f} GB)", flush=True)

    t0 = time.time()
    rsc.dcg.waggr(adata, net, tmin=TMIN, times=0, empty=False, verbose=verbose)
    if verbose:
        print(f"  waggr: {time.time() - t0:.0f}s", flush=True)

    scores = adata.obsm["score_waggr"]
    if not isinstance(scores, pd.DataFrame):
        scores = pd.DataFrame(cp.asnumpy(scores), index=adata.obs_names)
    scores = scores.astype(np.float32)
    scores.index = obs.index

    del adata
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return scores


# ----------------------------------------------------------------------------
# 5. Per-cell target matrix
# ----------------------------------------------------------------------------
def build_cell_target_matrix(scores, component_order):
    """Z-score each cell's component scores and orient as (components x cells).

    The bulk pipeline averages component scores within a sample and then z-scores
    each sample's 68 values (`StandardScaler` on a components x samples matrix
    standardizes per column). Per cell the averaging step simply drops out; the
    z-scoring is the identical operation applied to each cell's 68 values.

    Rows are reindexed to `component_order` because the ridge pairs X and Y
    row-wise -- required, not cosmetic.
    """
    y = scores.T                                   # components x cells
    y.index = y.index.astype(str)

    missing = set(component_order) - set(y.index)
    if missing:
        raise ValueError(f"components scored in SIG13 but absent from this dataset: "
                         f"{sorted(missing)}")
    y = y.loc[list(component_order)]

    values = y.to_numpy(dtype=np.float32)
    # ddof=0, per cell (axis 0) -- matches StandardScaler
    values -= values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, keepdims=True)
    values /= np.where(sd == 0, 1.0, sd)
    return pd.DataFrame(values, index=y.index, columns=y.columns)


# ----------------------------------------------------------------------------
# 6. Batched ridge (GPU)
# ----------------------------------------------------------------------------
def precompute_ridge_operators(X_design, alphas=None, n_splits=5):
    """Build every alpha-dependent operator the per-cell fit needs, once.

    All of these depend only on X and alpha, never on y, so they are computed on
    CPU (they are tiny) and reused for every cell.

      fold_ops[f][i]  A, such that the held-out prediction for fold f at alpha i is
                      `A @ ytrain_centered + mean(ytrain)`. Derivation: ridge with an
                      intercept predicts `Xte @ beta + (ytr_mean - Xtr_mean @ beta)`
                      `= (Xte - Xtr_mean) @ beta + ytr_mean`, and
                      `beta = H_fold @ ytr_centered`, so `A = (Xte - Xtr_mean) @ H_fold`.
      H[i]            full-data operator (activities x components)

    Folds come from `sklearn`'s own `KFold(n_splits)` so they are identical to what
    `RidgeCV(cv=n_splits)` uses internally.

    Returns a dict of numpy arrays.
    """
    alphas = ALPHA_RANGE if alphas is None else alphas
    X = np.asarray(X_design, dtype=np.float64)
    n, p = X.shape

    Xc = X - X.mean(axis=0)
    gram = Xc.T @ Xc
    eye = np.eye(p)
    H = np.stack([np.linalg.solve(gram + a * eye, Xc.T) for a in alphas])   # (n_alphas, p, n)

    fold_ops, folds = [], list(KFold(n_splits=n_splits).split(np.arange(n)))
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        Xtr_mean = Xtr.mean(axis=0)
        Xtr_c = Xtr - Xtr_mean
        gram_tr = Xtr_c.T @ Xtr_c
        Xte_shift = Xte - Xtr_mean
        fold_ops.append(np.stack([
            Xte_shift @ np.linalg.solve(gram_tr + a * eye, Xtr_c.T) for a in alphas
        ]))                                                                # (n_alphas, n_te, n_tr)

    return dict(alphas=np.asarray(alphas), H=H, fold_ops=fold_ops, folds=folds,
                Xc=Xc, n=n, p=p)


def _permutation_null(H_a, Yc_sub, n_perms, seed, budget=PERM_BUDGET):
    """Permutation null for one alpha group, all permutations at once.

    The direct analogue of what bulk's `_ridge_permutation_zscore` does with
    `Ridge().fit(X, Y_perm)`: build the permuted responses and solve them
    simultaneously, as one multi-response ridge. Because the ridge solution is
    `beta = H @ y` for the already-chosen alpha, "solve" is a single GEMM and the
    batch extends over cells as well as permutations -- `(b, m, n) @ (n, p)` gives
    the coefficients for `b` permutations of `m` cells in one call.

    **Each cell gets its own independent permutation set**, matching bulk, which
    seeds a fresh RNG per sample. Sharing one set across cells would be cheaper (the
    permuted operators could then be hoisted out of the cell loop) but every cell's
    null would inherit the same Monte Carlo error, so a set that happened to over- or
    under-estimate the null sd for some activity would bias that activity for *every*
    cell in the same direction instead of averaging out.

    Only running moments are kept, so memory is bounded by `budget` = the largest
    permutations x cells product held at once, not by `n_perms * m`. The block size
    adapts to the group size, which keeps the iteration count proportional to total
    work rather than to the number of alpha groups (there can be up to 100 groups,
    most of them small).

    `seed` is derived from the cell-chunk offset and the alpha index by the caller,
    so results do not depend on how groups happen to be ordered or iterated.

    Returns (mean, sd), each (activities x cells).
    """
    import cupy as cp

    p, n = H_a.shape
    m = Yc_sub.shape[1]
    Yt = cp.ascontiguousarray(Yc_sub.T)              # (m, n) -- sort axis contiguous
    Ht = cp.ascontiguousarray(H_a.T)                 # (n, p)

    # float64 accumulators: cheap at (m, p) and removes any doubt about summing
    # n_perms squared terms in float32
    total = cp.zeros((m, p), dtype=cp.float64)
    total_sq = cp.zeros((m, p), dtype=cp.float64)

    rng = cp.random.default_rng(seed)
    block = max(1, min(int(n_perms), int(budget // max(m, 1))))
    done = 0
    while done < n_perms:
        b = min(block, n_perms - done)
        # independent permutation of the n components per (permutation, cell):
        # argsort of uniform keys along the contiguous last axis
        keys = rng.random((b, m, n), dtype=cp.float32)
        idx = cp.argsort(keys, axis=-1)
        del keys
        Yp = cp.take_along_axis(cp.broadcast_to(Yt, (b, m, n)), idx, axis=-1)
        del idx
        beta = cp.matmul(Yp, Ht)                     # (b, m, p) -- batched GEMM
        del Yp
        total += beta.sum(axis=0, dtype=cp.float64)
        total_sq += (beta.astype(cp.float64) ** 2).sum(axis=0)
        del beta
        done += b

    mean = total / n_perms
    var = cp.maximum(total_sq / n_perms - mean * mean, 1e-30)
    return mean.T.astype(cp.float32), cp.sqrt(var).T.astype(cp.float32)


def score_ligand_activity_gpu(X_design, Y, ops=None, cell_chunk=CELL_CHUNK,
                              n_perms=N_PERMS, seed=67, verbose=True):
    """Fit the ridge activity model for every cell (column of Y), batched on GPU.

    Same model as `model_core.score_ligand_activity`: per response, pick alpha by
    5-fold CV R^2 over `ALPHA_RANGE`, refit on all components, z-score the
    coefficients against a label-permutation null drawn independently per cell.
    Only the execution differs -- see this module's docstring.

    Returns (activity: cells x activities, diagnostics: cells x [r2_score, alpha]).
    """
    import cupy as cp

    ops = precompute_ridge_operators(X_design, n_splits=5) if ops is None else ops
    alphas, H_all = ops["alphas"], ops["H"]
    n_alphas, p, n = len(alphas), ops["p"], ops["n"]

    Xc_g = cp.asarray(ops["Xc"], dtype=cp.float32)
    H_g = cp.asarray(H_all, dtype=cp.float32)
    fold_ops_g = [cp.asarray(A, dtype=cp.float32) for A in ops["fold_ops"]]
    folds_g = [(cp.asarray(tr), cp.asarray(te)) for tr, te in ops["folds"]]

    Y_values = Y.to_numpy(dtype=np.float32) if isinstance(Y, pd.DataFrame) else np.asarray(Y, np.float32)
    n_cells = Y_values.shape[1]
    z_out = np.empty((n_cells, p), dtype=np.float32)
    r2_out = np.empty(n_cells, dtype=np.float32)
    # float64: alpha holds grid values up to 1e5, and float32 cannot round-trip them
    # (158.48931924611142 -> 158.48932), which makes them look unequal to the grid
    alpha_out = np.empty(n_cells, dtype=np.float64)

    t0 = time.time()
    for start in range(0, n_cells, cell_chunk):
        end = min(start + cell_chunk, n_cells)
        Yg = cp.asarray(Y_values[:, start:end])

        # -- alpha selection: mean held-out R^2 over the 5 folds, as GridSearchCV does
        cv_score = cp.zeros((n_alphas, Yg.shape[1]), dtype=cp.float32)
        for (tr, te), A in zip(folds_g, fold_ops_g):
            ytr = Yg[tr]
            ytr_mean = ytr.mean(axis=0, keepdims=True)
            ytr_c = ytr - ytr_mean
            yte = Yg[te]
            ss_tot = ((yte - yte.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
            for i in range(n_alphas):
                resid = yte - (A[i] @ ytr_c + ytr_mean)
                cv_score[i] += 1.0 - (resid ** 2).sum(axis=0) / ss_tot
        best = cp.argmax(cv_score, axis=0)          # ties -> lowest alpha, as sklearn
        del cv_score

        # -- refit on all components, then z-score against the null
        Yc = Yg - Yg.mean(axis=0, keepdims=True)
        ss_tot_full = (Yc ** 2).sum(axis=0)
        beta = cp.zeros((p, Yg.shape[1]), dtype=cp.float32)
        z = cp.zeros_like(beta)
        r2 = cp.zeros(Yg.shape[1], dtype=cp.float32)

        for ai in cp.asnumpy(cp.unique(best)):
            mask = best == int(ai)
            Yc_sub = Yc[:, mask]
            b = H_g[int(ai)] @ Yc_sub
            beta[:, mask] = b
            resid = Yc_sub - Xc_g @ b
            r2[mask] = 1.0 - (resid ** 2).sum(axis=0) / ss_tot_full[mask]

            # seed from (chunk offset, alpha index) so the draw does not depend on
            # group iteration order
            group_seed = (int(seed) * 1_000_003 + start * 10_007 + int(ai)) % (2 ** 63)
            mean_null, sd_null = _permutation_null(
                H_g[int(ai)], Yc_sub, n_perms, group_seed)
            z[:, mask] = (b - mean_null) / sd_null

        z_out[start:end] = cp.asnumpy(z).T
        r2_out[start:end] = cp.asnumpy(r2)
        alpha_out[start:end] = alphas[cp.asnumpy(best)]

        del Yg, Yc, beta, z, r2
        cp.get_default_memory_pool().free_all_blocks()
        if verbose:
            print(f"     ridge {end}/{n_cells} cells ({time.time() - t0:.0f}s)", flush=True)

    activity = pd.DataFrame(z_out, index=Y.columns,
                             columns=[f"activity_{c}" for c in X_design.columns])
    activity.index.name = "cell_id"
    diagnostics = pd.DataFrame({"cell_id": Y.columns, "r2_score": r2_out, "alpha": alpha_out})
    return activity.reset_index(), diagnostics
