"""
scanpy_functions.py

This module provides utility functions for processing AnnData objects.
It currently includes a helper function to update the 'highly_variable'
column in an AnnData object by filtering out genes that match a specified
regex pattern.

Author: Eric Y. wang
Date: 2025-03-20
"""

## Function to modify HVG (primarily to remove TCR and BCR sequences from consideration) ##

def update_hvgs(adata, regex_pattern, variable_gene_column='highly_variable'):
    """
    Filters and updates the 'highly_variable' column in an AnnData object by removing genes matching a regex pattern.

    Parameters:
    - adata (AnnData): The AnnData object containing the data.
    - regex_pattern (str): The regex pattern to identify genes to be removed from 'highly_variable'.
    - variable_gene_column (str): The column in `adata.var` indicating highly variable genes.

    Returns:
    - adata (AnnData): The modified AnnData object with the updated 'highly_variable' column.
    - matching_genes (list): A list of genes that matched the regex pattern.
    """
    import re
    
    # Ensure the variable_gene_column exists
    if variable_gene_column not in adata.var:
        raise ValueError(f"'{variable_gene_column}' column not found in adata.var")

    # Extract the list of highly variable genes
    highly_variable_genes = adata.var_names[adata.var[variable_gene_column]]

    # Compile the regex pattern
    regex = re.compile(regex_pattern)

    # Find genes matching the regex pattern
    matching_genes = [gene for gene in highly_variable_genes if regex.match(gene)]

    # Update the 'highly_variable' column: set matching genes to False
    adata.var[variable_gene_column] = adata.var_names.isin(highly_variable_genes)
    adata.var.loc[adata.var_names.isin(matching_genes), variable_gene_column] = False

    # Return the updated AnnData object and the list of matching genes
    return adata, matching_genes

## Functions to calculate correlation matrix ##

def generate_corr(adata, aggregate_columns, correlation_method='pearson'):
    """
    Calculates the correlation matrix of mean expression values grouped by specified columns from an AnnData object.
    
    Parameters:
    - adata (AnnData): The AnnData object containing the data.
    - aggregate_columns (list): A list of column names in adata.obs used for grouping.
    - correlation_method (str): The correlation method to use; either 'pearson' or 'spearman'.
    
    Returns:
    - correlation_matrix (DataFrame): The correlation matrix calculated from the mean aggregated expression values.
    
    Raises:
    - ValueError: If `correlation_method` is not 'pearson' or 'spearman'.
    """
    import pandas as pd
    # Create a DataFrame from the AnnData object
    df = pd.DataFrame(adata.X.toarray(), index=adata.obs.index, columns=adata.var_names)
    df[aggregate_columns] = adata.obs[aggregate_columns]
    
    # Calculate mean expression grouped by aggregate_columns
    mean_aggregated = df.groupby(aggregate_columns, observed=True).mean()
    
    # Validate the correlation method
    if correlation_method not in ['pearson', 'spearman']:
        raise ValueError("correlation_method must be either 'pearson' or 'spearman'")
    
    # Calculate the correlation matrix using the specified method
    correlation_matrix = mean_aggregated.T.corr(method=correlation_method)
    
    return correlation_matrix

## Functions subset and subcluster data ##

def subset_umap_hvg(
    adata,
    classification_list,
    classification_column,
    preprocessed_layer='log1p_norm',
    flavor='seurat_v3',
    n_top_genes=2000,
    hvg_layer='counts'
):
    """
    Subset on `classification_list`, compute HVGs, PCA → neighbors → UMAP.
    
    Parameters
    ----------
    adata : AnnData
    classification_list : list
    classification_column : str
    preprocessed_layer : str
        Which .layers[] to pull into .X before PCA.
    flavor, n_top_genes, hvg_layer : passed to sc.pp.highly_variable_genes
    """
    import scanpy as sc

    # 1) subset + load chosen layer
    ad = adata[adata.obs[classification_column].isin(classification_list)].copy()
    ad.X = ad.layers[preprocessed_layer].copy()

    # 2) HVG selection
    sc.pp.highly_variable_genes(
        ad,
        flavor=flavor,
        n_top_genes=n_top_genes,
        layer=hvg_layer
    )

    # 3) PCA → neighbors → UMAP
    sc.pp.pca(ad, mask_var='highly_variable')
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)
    return ad

def subset_umap_genes(
    adata,
    classification_list,
    gene_list,
    classification_column,
    preprocessed_layer='log1p_norm'
):
    """
    Subset on `classification_list`, mark provided genes for PCA, 
    then run PCA → neighbors → UMAP.

    Parameters
    ----------
    adata : AnnData
        Input AnnData.
    classification_list : list
        Values in `.obs[classification_column]` to keep.
    gene_list : list of str
        Genes to use as features for PCA/UMAP.
    classification_column : str
        Column in `.obs` to subset on
    preprocessed_layer : str
        Which adata.layers[...] to load into .X before PCA (default: 'log1p_norm').

    Returns
    -------
    AnnData
        Subsetted AnnData with `.obsm['X_umap']`.
    """
    # 1) Subset and load specified layer into .X
    ad = adata[adata.obs[classification_column].isin(classification_list)].copy()
    ad.X = ad.layers[preprocessed_layer].copy()

    # 2) Create boolean mask in .var for your gene_list
    mask_key = 'use_for_pca'
    ad.var[mask_key] = False

    # Only keep genes actually present
    genes_in_data = [g for g in gene_list if g in ad.var_names]
    ad.var.loc[genes_in_data, mask_key] = True

    # 3) PCA → neighbors → UMAP using only those genes
    sc.pp.pca(ad, mask_var=mask_key)
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)

    return ad

## Function to convert mouse genes to human in an adata object ##

def convert_mouse_genes_to_human_adata(adata, mouse_gene_key=None, new_key='human_gene', drop_unmapped=False):
    """
    Convert mouse gene names in an AnnData object to human homolog gene names,
    replace the index of adata.var with the human gene names, and drop:
      - genes without a mapped human homolog (if drop_unmapped=True)
      - duplicated human gene names (keeping only the first occurrence)
      
    Parameters:
        adata (AnnData): AnnData object with mouse gene names in adata.var.
        mouse_gene_key (str or None): Column name in adata.var containing mouse gene symbols.
                                      If None, the index of adata.var is used.
        new_key (str): Column name in adata.var to store human gene names.
        drop_unmapped (bool): If True, drop genes that do not have a human homolog.
    
    Returns:
        AnnData: A copy of the original AnnData object with updated var DataFrame.
                The index of adata.var is replaced with unique human gene names.
    """
    import pandas as pd
    from pybiomart import Dataset

    # Query BioMart for the mapping between mouse and human gene names.
    dataset = Dataset(name='mmusculus_gene_ensembl', host='http://www.ensembl.org')
    mapping_df = dataset.query(attributes=['external_gene_name', 'hsapiens_homolog_associated_gene_name'])
    mapping_df.columns = ['mouse_gene', 'human_gene']
    mapping_df = mapping_df[mapping_df['human_gene'].notna() & (mapping_df['human_gene'] != '')]
    mapping = dict(zip(mapping_df['mouse_gene'], mapping_df['human_gene']))

    adata_new = adata.copy()

    # Use a specified column if available; otherwise, use the index.
    if mouse_gene_key and mouse_gene_key in adata_new.var.columns:
        genes = adata_new.var[mouse_gene_key].astype(str)
    else:
        genes = adata_new.var.index.astype(str)

    # Map mouse gene names to human gene names.
    human_names = genes.map(mapping)

    if drop_unmapped:
        # Keep only genes with a mapping.
        keep = human_names.notna()
        adata_new = adata_new[:, keep].copy()
        genes = genes[keep]
        human_names = human_names[keep]
    else:
        # For unmapped genes, retain the original mouse gene name.
        human_names = human_names.fillna(genes)

    # Drop duplicate human gene names (keep the first occurrence).
    non_duplicate = ~human_names.duplicated(keep='first')
    adata_new = adata_new[:, non_duplicate].copy()
    human_names = human_names[non_duplicate]

    # Update adata.var with the new human gene names.
    adata_new.var[new_key] = human_names
    adata_new.var.index = human_names

    return adata_new

def convert_mouse_genes_to_human(df, mouse_gene_col, new_col='human_gene'):
    """
    Convert mouse gene names in a specified column of a DataFrame to human homolog gene names.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing mouse gene names.
        mouse_gene_col (str): Column name in `df` that contains mouse gene symbols.
        new_col (str): Column name for the resulting human gene names.
                       Defaults to 'human_gene'.
    
    Returns:
        pd.DataFrame: A copy of the original DataFrame with an added column for human gene names.
    """
    import pandas as pd
    from pybiomart import Dataset
    
    # Create a Dataset object for the mouse dataset using the Dataset class.
    mouse_dataset = Dataset(name='mmusculus_gene_ensembl', host='http://www.ensembl.org')
    
    # Query BioMart for the mapping between mouse gene symbols and their human homologs.
    # 'external_gene_name' is the mouse gene symbol.
    # 'hsapiens_homolog_associated_gene_name' is the corresponding human gene symbol.
    mapping_df = mouse_dataset.query(attributes=[
        'external_gene_name',
        'hsapiens_homolog_associated_gene_name'
    ])
    
    # Rename the columns for clarity.
    mapping_df.columns = ['mouse_gene', 'human_gene']
    
    # Remove entries with missing or empty human homologs.
    mapping_df = mapping_df[mapping_df['human_gene'].notnull() & (mapping_df['human_gene'] != '')]
    
    # Create a mapping dictionary: keys are mouse gene names, values are human gene names.
    mapping_dict = dict(zip(mapping_df['mouse_gene'], mapping_df['human_gene']))
    
    # Make a copy of the DataFrame to avoid modifying the original data.
    df_new = df.copy()
    
    # Map the mouse gene names to human gene names.
    # This creates a new column with the specified name (default 'human_gene').
    df_new[new_col] = df_new[mouse_gene_col].map(mapping_dict)
    
    return df_new