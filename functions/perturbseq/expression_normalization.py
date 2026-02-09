# Perturbseq library for loading and manipulating single-cell experiments
# Copyright (C) 2019  Thomas Norman

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import pandas as pd
import numpy as np
import scipy as sp

# MODIFIED FUNCTIONS BY EYW

def normalize_matrix_to_control(matrix, control_matrix, verbose=True):
    """ Normalize expression distribution relative to a population of control (unperturbed) cells.
    The normalization proceeds by first normalizing the UMI counts (if umi_count != None) within each cell to the median
    UMI count within the population. The control population is normalized to the same amount. 
    The expression within the population is then Z-normalized with respect to the control 
    distribution: i.e., for each gene the control mean is subtracted, and then these values are
    divided by the control standard deviation.
        
    Args:
        matrix: gene expression matrix to normalize (output from cellranger)
        control_matrix: gene expression matrix of control population
    
    Returns:
        DataFrame of normalized expression data
    """
    # Convert sparse matrices to dense if necessary
    if sp.sparse.issparse(matrix):
        if verbose:
            print('     Densifying matrix...')
        matrix = matrix.todense()
    if sp.sparse.issparse(control_matrix):
        if verbose:
            print('     Densifying control matrix...')
        control_matrix = control_matrix.todense()

    # Convert to numpy arrays for processing if needed
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    if isinstance(control_matrix, pd.DataFrame):
        control_matrix = control_matrix.values

    m = matrix.astype(np.float64)
    c_m = control_matrix.astype(np.float64)

    control_mean = c_m.mean(axis=0)
    control_std = c_m.std(axis=0)
    
    if verbose:
        print("     Scaling matrix to control")
    # Center and rescale the expression of each gene to average 0 and std 1
    m_out = (m - control_mean) / control_std
    
    if verbose:
        print("     Done.")
    return pd.DataFrame(m_out, columns=np.arange(m.shape[1]), index=np.arange(m.shape[0]))

def normalize_to_control_adata(adata, control_cells_query, groupby_column, layer_name="zscore", verbose=True, **kwargs):
    """
    Normalizes a multi-lane 10x experiment stored in an AnnData object.
    Cells within each group (defined by a column in `.obs`) are normalized to the control cells
    within the same group. The resulting normalized expression data is stored as a layer
    called `layer_name` (defaults to "zscore") in the returned AnnData object as a CSR sparse matrix of dtype 'float32'.
    
    Args:
        adata: AnnData object to normalize.
        control_cells_query: String query to identify control cell population to normalize with respect to.
        groupby_column: Column name in `adata.obs` to use for grouping cells (e.g., "gem_group").
        layer_name: Name of the layer where the normalized data will be stored (default "zscore").
        verbose: If True, prints status messages; if False, runs silently.
        **kwargs: Additional arguments passed to groupby on `adata.obs`, useful for refined slicing.
    
    Returns:
        AnnData object with the normalized expression data stored in adata.layers[layer_name] as a CSR sparse matrix.
    """
    import pandas as pd
    import numpy as np
    from scipy.sparse import csr_matrix
    
    # Make a copy to avoid modifying the original AnnData object
    adata = adata.copy()

    # Check if the groupby column exists in .obs
    if groupby_column not in adata.obs.columns:
        raise ValueError(f"'{groupby_column}' column not found in adata.obs.")
    
    unique_groups = adata.obs[groupby_column].unique()
    
    if len(unique_groups) == 1:
        if verbose:
            print(f"Single {groupby_column} detected. Normalizing without grouping.")
        group_pop = adata.obs
        control_indices = adata.obs.query(control_cells_query).index

        if control_indices.empty:
            raise ValueError(f"No control cells found for the query: {control_cells_query}")

        group_data = adata[group_pop.index].X.copy()
        control_data = adata[control_indices].X.copy()

        normalized_matrix = normalize_matrix_to_control(
            group_data,
            control_data,
            verbose=verbose
        )
        
        # Assign indices and columns
        normalized_matrix.index = group_pop.index
        normalized_matrix.columns = adata.var_names
        
    else:
        if verbose:
            print(f"Multiple {groupby_column} detected. Processing each group separately.")
        
        group_iterator = zip(
            adata.obs.groupby(groupby_column, observed=False),
            adata[adata.obs.query(control_cells_query).index].obs.groupby(groupby_column, observed=False)
        )
        
        group_matrices = dict()
        for (group_name, group_pop), (_, group_control_pop) in group_iterator:
            if verbose:
                print(f'Processing {groupby_column} {group_name}')
            
            # Check if control population is empty for the group
            if group_control_pop.empty:
                if verbose:
                    print(f"Warning: No control cells found for {groupby_column} {group_name}. Skipping normalization.")
                continue

            # Select the data for the current group
            group_data = adata[group_pop.index].X.copy()
            control_data = adata[group_control_pop.index].X.copy()

            if control_data.shape[0] == 0:
                if verbose:
                    print(f"Warning: Control data matrix is empty for {groupby_column} {group_name}. Skipping normalization.")
                continue

            norm_mat = normalize_matrix_to_control(
                group_data,
                control_data,
                verbose=verbose
            )
            
            # Assign the correct index and columns
            norm_mat.index = group_pop.index
            norm_mat.columns = adata.var_names
            group_matrices[group_name] = norm_mat
        
        if verbose:
            print('Merging submatrices...')
        
        # Merge all matrices into a DataFrame
        normalized_matrix = pd.concat(group_matrices.values(), axis=0)
        # Ensure the rows match the original AnnData index
        normalized_matrix = normalized_matrix.loc[adata.obs.index]
    
    # Convert the normalized matrix to a CSR sparse matrix with dtype 'float32'
    adata.layers[layer_name] = csr_matrix(normalized_matrix.values, dtype=np.float32)
    
    return adata