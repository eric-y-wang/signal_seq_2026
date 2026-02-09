def plot_qc_plots(rna, umi_threshold, gene_threshold, mito_threshold, kde_hue="lane", pct_mt_limit=10, figsize=(15,10), subsample=None):
    """
    Generate QC plots from an AnnData object.

    The function creates a figure with five subplots:
      1. UMI count density per cell.
      2. Genes detected per cell density.
      3. Mitochondrial gene expression density per cell.
      4. Density of genes detected per UMI.
      5. Scatterplot of genes detected vs UMI counts.

    QC thresholds are indicated as red lines:
      - umi_threshold: UMI counts threshold.
      - gene_threshold: Genes detected threshold.
      - mito_threshold: Mitochondrial percentage threshold.

    Parameters:
        rna: AnnData object containing metadata in rna.obs.
        umi_threshold: Numeric threshold for UMI counts.
        gene_threshold: Numeric threshold for genes detected.
        mito_threshold: Numeric threshold for mitochondrial gene percentage.
        kde_hue: Column name in rna.obs to use for hue in KDE plots (default "lane"). 
                 Set to None to disable hue.
        pct_mt_limit: Maximum percentage of mitochondrial genes to plot (default: 10).
        figsize: Tuple defining the figure size (default: (18, 12)).
        subsample: Optional; if provided, subsample the data. Can be an integer (number of rows) 
                   or a float (fraction of rows, between 0 and 1).

    Returns:
        None. Displays a matplotlib figure with the QC plots.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    # Copy the metadata (important if subsampling)
    metadata = rna.obs.copy()

    # Optionally subsample the data
    if subsample is not None:
        if isinstance(subsample, float) and 0 < subsample < 1:
            metadata = metadata.sample(frac=subsample, random_state=42)
        elif isinstance(subsample, int) and subsample < len(metadata):
            metadata = metadata.sample(n=subsample, random_state=42)

    # Set up the figure layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)

    # Helper function to plot KDEs with common settings.
    # Note: we disable the legend here.
    def plot_kde(ax, x, threshold=None, xscale='linear', xlim=None, title=""):
        sns.kdeplot(
            data=metadata,
            x=x,
            hue=kde_hue if kde_hue is not None else None,
            fill=True,
            alpha=0.2,
            ax=ax,
            legend=False
        )
        if xscale:
            ax.set_xscale(xscale)
        if threshold is not None:
            ax.axvline(x=threshold, color='red')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_title(title)

    # Plot 1: UMI count density per cell
    ax1 = fig.add_subplot(gs[0, 0])
    plot_kde(ax1, 'total_counts', threshold=umi_threshold, xscale='log', title="UMI Counts per Cell")
    ax1.set_ylabel("Cell density")

    # Plot 2: Genes detected per cell density
    ax2 = fig.add_subplot(gs[0, 1])
    plot_kde(ax2, 'n_genes_by_counts', threshold=gene_threshold, xscale='log', title="Genes Detected per Cell")

    # Plot 3: Mitochondrial gene expression density per cell
    ax3 = fig.add_subplot(gs[0, 2])
    plot_kde(ax3, 'pct_counts_mt', threshold=mito_threshold, xscale='linear', xlim=(0, pct_mt_limit),
             title="Mitochondrial Gene Expression per Cell")

    # Plot 4: Genes detected per UMI density (no threshold line)
    ax4 = fig.add_subplot(gs[1, 0])
    plot_kde(ax4, 'log10_genes_per_umi', threshold=None, xscale='linear', title="Genes Detected per UMI")

    # Create a figure-level legend for the KDE plots if a hue is specified.
    if kde_hue is not None:
        # Get unique categories for the specified hue and sort them.
        unique_categories = sorted(metadata[kde_hue].dropna().unique())
        # Use the default seaborn palette (or adjust if needed).
        colors = sns.color_palette("deep", len(unique_categories))
        kde_handles = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(unique_categories))]
        kde_labels = unique_categories
        # Add the legend to the figure (position can be adjusted).
        fig.legend(kde_handles, kde_labels, loc='upper center', ncol=len(kde_labels), title=kde_hue)

    # Plot 5: Scatter plot for correlation between genes detected and UMI counts
    ax5 = fig.add_subplot(gs[1, 1])
    # Disable the per-axis legend here.
    sns.scatterplot(
        data=metadata,
        x='total_counts',
        y='n_genes_by_counts',
        hue="pct_counts_mt",
        palette="viridis",
        ax=ax5,
        alpha=0.5,
        legend=False
    )
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.axvline(x=umi_threshold, color='red')
    ax5.axhline(y=gene_threshold, color='red')
    ax5.set_title("Genes Detected vs UMI Counts")
    for coll in ax5.collections:
        coll.set_rasterized(True)

    # Create a colorbar to serve as the legend for the scatter plot.
    norm = mpl.colors.Normalize(vmin=metadata["pct_counts_mt"].min(), vmax=metadata["pct_counts_mt"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    # Add an axis for the colorbar on the right side of the figure.
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position as needed
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Mitochondrial % (pct_counts_mt)")

    # Adjust layout to leave room for the legends
    plt.tight_layout(rect=[0, 0, 0.9, 0.93])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import scanpy as sc  # assuming sc.pl.draw_graph is from scanpy


def plot_embedding_categories(ad, obs, ncol=5, dotsize=20, figsize=5, 
                    ratio=1, use_basis='X_umap', sort_categories=True, show_titles=True):
    """
    Plot subplots highlighting each unique category on an embedding of an AnnData object.

    Parameters:
        ad : AnnData
            The annotated data object.
        obs : str
            The key in ad.obs used to define the categories.
        ncol : int, optional
            Number of columns in the subplot grid (default is 5).
        plt_func : function, optional
            Plotting function to be used for each subplot (default is sc.pl.draw_graph).
        dotsize : int, optional
            Size of the dots in the plots (default is 20).
        figsize : int, optional
            Base size to scale the figure dimensions (default is 5).
        ratio : float, optional
            Multiplier to adjust the width relative to the height of each subplot (default is 1).
        use_basis : str, optional
            The basis to use for plotting (default is 'X_umap').
        sort_categories : bool, optional
            Whether to sort the categories alphabetically (default is True).
        show_titles : bool, optional
            Whether to display each category as the title of its subplot (default is True).

    Returns:
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        axes : numpy.ndarray
            Array of Axes objects that can be further customized.
    """
    # Get unique categories from the specified observation field
    categories = np.unique(ad.obs[obs].tolist())
    if sort_categories:
        categories = np.sort(categories)
    
    n_categories = len(categories)
    nrow = ceil(n_categories / ncol)

    # Create the subplot grid
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize * ncol * ratio, figsize * nrow), 
                             constrained_layout=True)
    
    # Flatten the axes array to simplify indexing (handles cases with one row/column gracefully)
    axes = np.ravel(axes)
    
    # Plot each category on its corresponding axis
    for ax, category in zip(axes, categories):
        sc.pl.embedding(ad, color=obs, groups=category, size=dotsize, ax=ax, basis=use_basis, show=False, legend_loc='none')
        if show_titles:
            ax.set_title(str(category))
    
    # Turn off any extra axes that were created but not used
    for ax in axes[n_categories:]:
        ax.axis('off')
    
    return fig, axes
