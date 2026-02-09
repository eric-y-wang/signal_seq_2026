# From https://gist.github.com/hussenmi/8e30c924302c1c8670fd1123d57d54d6#file-dsb_python-py
# The original dsb_adapted code takes forever to run. This code was optimized for efficiency and also added the quantile thresholding included
# in the original paper. The optimized code is below.

import numpy as np
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from tqdm import tqdm

def remove_batch_effect(X, covariates):
    X = X.T  # cells × features
    model = LinearRegression().fit(covariates.reshape(-1, 1), X)
    corrected = X - model.predict(covariates.reshape(-1, 1))
    return corrected.T

def apply_quantile_clipping(norm_adt: np.ndarray, quantile_clip=(0.001, 0.9995)) -> np.ndarray:
    q_low = np.quantile(norm_adt, quantile_clip[0], axis=1)
    q_high = np.quantile(norm_adt, quantile_clip[1], axis=1)
    clipped = np.empty_like(norm_adt)
    for i in range(norm_adt.shape[0]):
        clipped[i, :] = np.clip(norm_adt[i, :], q_low[i], q_high[i])
    return clipped

def dsb_normalize_adata(
    adata_filtered: AnnData,
    adata_raw: AnnData,
    layer: str = None,
    denoise_counts: bool = True,
    use_isotype_control: bool = False,
    isotype_control_names: list = None,
    pseudocount: float = 10.0,
    scale_factor: str = 'standardize',
    quantile_clipping: bool = False,
    quantile_clip: tuple = (0.001, 0.9995),
    output_layer: str = 'dsb',
    return_stats: bool = False,
    n_jobs: int = -1
):
    """
    Parallelized DSB normalization for ADT data using two AnnData objects.

    Parameters
    ----------
    adata_filtered : AnnData
        ADT counts for cells (cell-containing droplets).
    adata_raw : AnnData
        ADT counts for empty droplets.
    layer : str or None
        Layer to read counts from (defaults to .X).
    denoise_counts : bool
        Whether to remove technical noise via GMM + PCA.
    use_isotype_control : bool
        Whether to use isotype controls in noise regression.
    isotype_control_names : list
        List of protein names to use as isotype controls.
    pseudocount : float
        Pseudocount added before log transform.
    scale_factor : str
        'standardize' or 'mean.subtract'.
    quantile_clipping : bool
        Whether to clip values at given quantiles.
    quantile_clip : tuple
        Tuple of (low, high) quantiles.
    output_layer : str
        Layer to save the normalized result.
    return_stats : bool
        Whether to return diagnostic stats.
    n_jobs : int
        Number of CPU cores to use for parallel GMM (default: all).

    Returns
    -------
    Optional[dict]
        If return_stats=True, returns a dictionary of normalization stats.
    """

    # --- Extract data matrices ---
    X = adata_filtered.layers[layer] if layer else adata_filtered.X
    X = X.toarray() if not isinstance(X, np.ndarray) else X

    X_bg = adata_raw.layers[layer] if layer else adata_raw.X
    X_bg = X_bg.toarray() if not isinstance(X_bg, np.ndarray) else X_bg

    # --- Validate and align ---
    assert adata_filtered.var_names.equals(adata_raw.var_names), "Protein names must match"
    X = X.T if X.shape[0] == adata_filtered.n_obs else X
    X_bg = X_bg.T if X_bg.shape[0] == adata_raw.n_obs else X_bg

    # --- Step I: Background correction ---
    adtu_log = np.log(X_bg + pseudocount)
    adt_log = np.log(X + pseudocount)
    mu_u = np.mean(adtu_log, axis=1)
    sd_u = np.std(adtu_log, axis=1)

    if scale_factor == 'standardize':
        norm_adt = ((adt_log.T - mu_u) / sd_u).T
    elif scale_factor == 'mean.subtract':
        norm_adt = (adt_log.T - mu_u).T
    else:
        raise ValueError("scale_factor must be 'standardize' or 'mean.subtract'")

    # --- Step II: Denoising with parallel GMM ---
    def fit_cellwise_gmm(cell_vector: np.ndarray) -> float:
        vals = cell_vector.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=100)
        gmm.fit(vals)
        return np.min(gmm.means_.flatten())

    if denoise_counts:
        print("Fitting Gaussian Mixture Models (parallelized)...")
        background_means = Parallel(n_jobs=n_jobs)(
            delayed(fit_cellwise_gmm)(norm_adt[:, i]) for i in tqdm(range(norm_adt.shape[1]), desc="GMM per cell")
        )
        background_means = np.array(background_means)

        if use_isotype_control:
            if isotype_control_names is None:
                raise ValueError("Must specify isotype_control_names when use_isotype_control is True")
            iso_indices = [adata_filtered.var_names.get_loc(name) for name in isotype_control_names]
            noise_matrix = np.vstack([
                norm_adt[iso_indices, :],
                background_means.reshape(1, -1)
            ])
            noise_vec = PCA(n_components=1).fit_transform(noise_matrix.T).flatten()
        else:
            noise_vec = background_means

        norm_adt = remove_batch_effect(norm_adt, covariates=noise_vec)

    # --- Step III: Quantile clipping ---
    if quantile_clipping:
        print(f"Applying quantile clipping: {quantile_clip}")
        norm_adt = apply_quantile_clipping(norm_adt, quantile_clip=quantile_clip)

    # --- Save output ---
    adata_filtered.X = norm_adt.T
    adata_filtered.layers[output_layer] = norm_adt.T

    if return_stats:
        return {
            "dsb_normalized_matrix": norm_adt,
            "background_mean": mu_u,
            "background_sd": sd_u,
            "noise_vector": noise_vec if denoise_counts else None
        }