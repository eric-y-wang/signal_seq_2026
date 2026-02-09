import scanpy as sc
import scvi
import torch
torch.set_float32_matmul_precision("high")

rna_iLN = sc.read_h5ad('/data1/rudenska/EYW/SIG19/scanpy_outs/SIG19_DTR_CD4T_iLN_subset.h5ad')

rna_list = [rna_iLN]
for rna_sub in rna_list:
    # filter genes
    sc.pp.filter_genes(rna_sub, min_cells=10)
    # define HVGs
    sc.pp.highly_variable_genes(
        rna_sub,
        n_top_genes=3000,
        subset=False,
        layer="counts",
        flavor="seurat_v3"
    )
    # subset to HVGs
    rna_sub_hvg = rna_sub[:, rna_sub.var.highly_variable].copy()
    # define model
    scvi.model.SCVI.setup_anndata(
        rna_sub_hvg,
        layer="counts",
        categorical_covariate_keys=["cage"],
        continuous_covariate_keys=["pct_counts_mt",'S_score','G2M_score'],
    )
    # train model on hvg only anndata
    model = scvi.model.SCVI(rna_sub_hvg)
    model.train(max_epochs=1000, early_stopping=True)
    # save latent on non-subsetted data and perform dim reduction
    latent = model.get_latent_representation()
    rna_sub.obsm["X_scvi"] = latent
    # umap and leiden clustering on scvi latent space
    sc.pp.neighbors(rna_sub, use_rep='X_scvi')
    sc.tl.umap(rna_sub, key_added='X_umap_scvi')
    sc.tl.leiden(rna_sub, resolution=0.5, key_added='leiden_0.5')
    sc.tl.leiden(rna_sub, resolution=0.75, key_added='leiden_0.75')
    sc.tl.leiden(rna_sub, resolution=1.0, key_added='leiden_1.0')

# export
rna_iLN.write_h5ad('/data1/rudenska/EYW/SIG19/scvi_outs/SIG19_DTR_CD4T_iLN_scvi.h5ad')