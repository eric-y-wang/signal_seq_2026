# This module provides a ChatGPT optimized implementation for the Norman lab zscore normalization
# using multithreading. It normalizes each group in a single-cell dataset to its control cells,
# leveraging multiprocessing across groups for efficiency. 

import numpy as np
from scipy import sparse
import multiprocessing as mp
from tqdm import tqdm

# Global variables for shared data in workers
_X = None
_ctrl_mask = None
_grp_labels = None

def _init_worker(X, ctrl_mask, grp_labels):
    global _X, _ctrl_mask, _grp_labels
    _X = X
    _ctrl_mask = ctrl_mask
    _grp_labels = grp_labels


def _compute_control_stats(ctrl: sparse.spmatrix, dtype=np.float32):
    ctrl = ctrl.tocsr(copy=False)
    mean = np.asarray(ctrl.mean(axis=0)).ravel().astype(dtype)

    ctrl_sq = ctrl.copy()
    ctrl_sq.data **= 2
    mean_sq = np.asarray(ctrl_sq.mean(axis=0)).ravel().astype(dtype)

    var = mean_sq - mean**2
    std = np.sqrt(var, dtype=dtype)

    # instead of forcing 1.0, put NaN where std is zero
    std[std == 0] = np.nan
    return mean, std


def _normalize_matrix_to_control_sparse(
    mat: sparse.spmatrix,
    ctrl: sparse.spmatrix,
    dtype=np.float32
) -> sparse.csr_matrix:
    mean, std = _compute_control_stats(ctrl, dtype=dtype)
    mat = mat.tocoo()
    cols = mat.col

    # compute (x - mean) / std, yielding NaN wherever std==0
    with np.errstate(divide='ignore', invalid='ignore'):
        data = (mat.data.astype(dtype) - mean[cols]) / std[cols]

    return sparse.csr_matrix((data, (mat.row, cols)), shape=mat.shape)


def _process_group(grp: str):
    grp_pos = np.where(_grp_labels == grp)[0]
    ctrl_pos = grp_pos[_ctrl_mask[grp_pos]]

    if ctrl_pos.size == 0:
        return None

    grp_mat = _X[grp_pos]
    ctrl_mat = _X[ctrl_pos]

    norm_block = _normalize_matrix_to_control_sparse(grp_mat, ctrl_mat)
    coo = norm_block.tocoo()
    abs_rows = grp_pos[coo.row]

    return abs_rows, coo.col, coo.data


def normalize_to_control_adata_multithread(
    adata,
    control_cells_query: str,
    groupby_column: str,
    layer_name: str = "zscore",
    n_jobs: int = None
):
    """
    In-place add a z-score layer to adata.layers[layer_name],
    normalizing each group to its control cells using multiprocessing
    with a progress bar.
    """
    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr(copy=False)

    obs = adata.obs
    ctrl_mask = obs.eval(control_cells_query).values
    grp_labels = obs[groupby_column].values
    unique_grps = np.unique(grp_labels)

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    results = []

    with mp.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(X, ctrl_mask, grp_labels)
    ) as pool:
        with tqdm(total=len(unique_grps), desc="Normalizing groups") as pbar:
            for res in pool.imap_unordered(_process_group, unique_grps):
                if res is not None:
                    results.append(res)
                pbar.update()

    if not results:
        raise ValueError("No valid groups with control cells found.")

    all_rows, all_cols, all_data = zip(*results)
    all_rows = np.concatenate(all_rows)
    all_cols = np.concatenate(all_cols)
    all_data = np.concatenate(all_data)

    full_csr = sparse.csr_matrix(
        (all_data.astype(np.float32), (all_rows, all_cols)),
        shape=X.shape
    )

    adata.layers[layer_name] = full_csr
    adata.X = full_csr
    return adata
