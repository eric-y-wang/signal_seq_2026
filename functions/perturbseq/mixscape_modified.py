from typing import Literal
import numpy as np
from scipy.sparse import issparse, csr_matrix
from anndata import AnnData
from pynndescent import NNDescent
from scanpy.tools._utils import _choose_representation

# If _choose_representation is defined elsewhere in your package, import it:
# from your_package._utils import _choose_representation

def perturbation_signature_zscore(
        adata: AnnData,
        pert_key: str,
        control: str,
        *,
        ref_selection_mode: Literal["nn", "split_by"] = "nn",
        split_by: str | None = None,
        n_neighbors: int = 20,
        use_rep: str | None = None,
        n_dims: int | None = 15,
        n_pcs: int | None = None,
        batch_size: int | None = None,
        copy: bool = False,
        **kwargs,
    ):
    """Calculate zscore perturbation signature.

        This function is modified from the original pertpy implementaiton to use z-scores instead of means

        The perturbation signature is calculated by z-scoring the mRNA expression profile of each cell using the mean and 
        standard deviation of the control cells (selected according to `ref_selection_mode`).
        The implementation resembles https://satijalab.org/seurat/reference/runmixscape. Note that in the original implementation, the
        perturbation signature is calculated on unscaled data by default, and we therefore recommend to do the same.

        Args:
            adata: The annotated data object.
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Name of the control category from the `pert_key` column.
            ref_selection_mode: Method to select reference cells for the perturbation signature calculation. If `nn`,
                the `n_neighbors` cells from the control pool with the most similar mRNA expression profiles are selected. If `split_by`,
                the control cells from the same split in `split_by` (e.g. indicating biological replicates) are used to calculate the perturbation signature.
            split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
                the perturbation signature for every replicate separately.
            n_neighbors: Number of neighbors from the control to use for the perturbation signature.
            use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                If `None`, the representation is chosen automatically:
                For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                If 'X_pca' is not present, it's computed with default parameters.
            n_dims: Number of dimensions to use from the representation to calculate the perturbation signature.
                If `None`, use all dimensions.
            n_pcs: If PCA representation is used, the number of principal components to compute.
                If `n_pcs==0` use `.X` if `use_rep is None`.
            batch_size: Size of batch to calculate the perturbation signature.
                If 'None', the perturbation signature is calcuated in the full mode, requiring more memory.
                The batched mode is very inefficient for sparse data.
            copy: Determines whether a copy of the `adata` is returned.
            **kwargs: Additional arguments for the `NNDescent` class from `pynndescent`.

        Returns:
            If `copy=True`, returns the copy of `adata` with the perturbation signature in `.layers["X_pert"]`.
            Otherwise, writes the perturbation signature directly to `.layers["X_pert"]` of the provided `adata`.

        Examples:
            Calcutate perturbation signature for each cell in the dataset:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
        """
    if ref_selection_mode not in ["nn", "split_by"]:
        raise ValueError("ref_selection_mode must be either 'nn' or 'split_by'.")
    if ref_selection_mode == "split_by" and split_by is None:
        raise ValueError("split_by must be provided if ref_selection_mode is 'split_by'.")

    if copy:
        adata = adata.copy()

    X_pert = adata.X.copy()
    adata.layers["X_pert"] = X_pert
    X_pert_lil = X_pert.tolil() if issparse(X_pert) else X_pert

    control_mask = adata.obs[pert_key] == control

    # split_by mode: global control within each split
    if ref_selection_mode == "split_by":
        for split in adata.obs[split_by].unique():
            mask_split = adata.obs[split_by] == split
            ctrl = control_mask & mask_split

            ctrl_vals = adata.X[ctrl]
            ctrl_mean = np.mean(ctrl_vals, axis=0)
            ctrl_std  = np.std(ctrl_vals, axis=0)
            ctrl_std[ctrl_std == 0] = np.nan

            vals = X_pert_lil[mask_split]
            X_pert_lil[mask_split] = (vals - ctrl_mean) / ctrl_std

    # nn mode: per-cell kNN reference
    else:
        if split_by is None:
            split_masks = [np.ones(adata.n_obs, bool)]
        else:
            obs = adata.obs[split_by]
            split_masks = [(obs == grp).to_numpy() for grp in obs.unique()]

        representation = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
        if n_dims is not None and n_dims < representation.shape[1]:
            representation = representation[:, :n_dims]

        for mask_split in split_masks:
            ctrl_split = control_mask & mask_split
            R_split   = representation[mask_split]
            R_control = representation[ctrl_split]

            eps = kwargs.pop("epsilon", 0.1)
            nn_index = NNDescent(R_control, **kwargs)
            indices, _ = nn_index.query(R_split, k=n_neighbors, epsilon=eps)

            Xnorm = adata.X.toarray() if issparse(adata.X) else adata.X
            X_ctrl = Xnorm[ctrl_split]
            X_splt = Xnorm[mask_split]

            if batch_size is None:
                rows = np.repeat(np.arange(len(indices)), n_neighbors)
                cols = indices.ravel()
                M = csr_matrix((np.ones_like(cols), (rows, cols)),
                               shape=(len(indices), X_ctrl.shape[0]))
                M = M / n_neighbors

                mu  = M.dot(X_ctrl)
                Ex2 = M.dot(X_ctrl**2)
                var = Ex2 - mu**2
                std = np.sqrt(var)
                std[std == 0] = np.nan

                Z = (X_splt - mu) / std
                X_pert_lil[mask_split] = Z

            else:
                split_idx = np.where(mask_split)[0]
                for start in range(0, len(indices), batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]
                    cells     = split_idx[start:end]

                    neigh_expr = X_ctrl[batch_idx]          # (b, k, n_vars)
                    mu_b = np.mean(neigh_expr, axis=1)      # (b, n_vars)
                    std_b = np.std(neigh_expr, axis=1)
                    std_b[std_b == 0] = np.nan

                    X_batch = X_splt[start:end]
                    Zbatch = (X_batch - mu_b) / std_b
                    X_pert_lil[cells] = Zbatch

    if issparse(X_pert_lil):
        adata.layers["X_pert"] = X_pert_lil.tocsr()
    else:
        adata.layers["X_pert"] = X_pert_lil

    if copy:
        return adata

from typing import Sequence, Optional
import numpy as np
from scipy.sparse import issparse
from anndata import AnnData

def mixscape_gene_list(
    adata: AnnData,
    labels: str,
    control: str,
    gene_list: Sequence[str],
    *,
    layer: Optional[str] = None,
    scale: bool = True,
    score_name: str = "perturbation_score",
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Compute a per-cell perturbation score by projecting each cell’s
    perturbation‐signature onto the vector defined by:
        mean(non‐control cells) – mean(control cells)
    restricted to the user’s gene_list.

    Args:
        adata:        AnnData with a perturbation‐signature layer.
        labels:       .obs column giving each cell’s label (e.g. gRNA target).
        control:      The label string in `labels` denoting control cells.
        gene_list:    List of gene names (must be a subset of adata.var_names).
        layer:        Name of layer holding the per-gene perturbation signature.
                      If None, looks for "X_pert" in adata.layers.
        scale:        If True, z‐score each gene (column) across all cells before
                      building the vector and projecting.
        score_name:   Column name to store the resulting scalar score in adata.obs.
        copy:         If True, operate on a copy and return it; otherwise modify in place.

    Returns:
        If copy=True, the new AnnData; else None (adata is modified).
    """
    if copy:
        adata = adata.copy()

    # 1) Grab the perturbation‐signature matrix
    X = (
        adata.layers[layer]
        if layer is not None
        else adata.layers.get("X_pert")
    )
    if X is None:
        raise KeyError(
            f"No layer '{layer or 'X_pert'}' found in .layers; "
            "run perturbation_signature or specify layer."
        )

    # 2) Restrict to the user’s genes
    mask = np.isin(adata.var_names, gene_list)
    if not mask.any():
        raise ValueError("None of gene_list were found in adata.var_names.")
    gene_idx = np.where(mask)[0]

    # 3) Extract data & optionally z‐score per gene
    if issparse(X):
        dat = X[:, gene_idx].toarray()
        # set NaNs and infinities to 0
        dat = np.nan_to_num(dat, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        dat = X[:, gene_idx]
        # set NaNs and infinities to 0
        dat = np.nan_to_num(dat, nan=0.0, posinf=0.0, neginf=0.0)
    if scale:
        μ = np.mean(dat, axis=0)
        σ = np.std(dat, axis=0, ddof=1)
        σ[σ == 0] = 1.0
        dat = (dat - μ) / σ

    # 4) Define control vs. perturbed masks
    lbls = adata.obs[labels].astype(str)
    ctrl_mask = lbls.values == control
    pert_mask = ~ctrl_mask
    if not ctrl_mask.any():
        raise ValueError(f"No cells with {labels} == {control}")
    if not pert_mask.any():
        raise ValueError(f"No cells with {labels} != {control}")

    # 5) Build the perturbation vector
    mean_ctrl  = dat[ctrl_mask].mean(axis=0)
    mean_pert  = dat[pert_mask].mean(axis=0)
    vec        = mean_pert - mean_ctrl
    denom      = np.dot(vec, vec)
    if denom == 0:
        # no variation in signature across those genes
        denom = 1.0

    # 6) Project every cell onto vec
    #    dat shape: (n_cells, n_genes); vec shape: (n_genes,)
    scores = dat.dot(vec) / denom

    # 7) Store and return
    adata.obs[score_name] = scores
    if copy:
        return adata
