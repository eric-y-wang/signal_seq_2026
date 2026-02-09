# multithread parameters
import os
n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from umap import UMAP
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import dict_learning
from sklearn.preprocessing import StandardScaler
import argparse
from umap import UMAP
from umap.distances import hellinger
from hdbscan import HDBSCAN

import scanpy as sc
import pandas as pd
import sys
sys.path.insert(0, '/data1/rudenska/EYW/git_projects/SIG13/functions')
import perturbseq as ps

parser = argparse.ArgumentParser(description="import alpha parameter for sparse PCA")
parser.add_argument("--alpha", type=float, required=True)

args = parser.parse_args()

# Define parameters ------------------------------------------------------------------------------

alpha=args.alpha  # Sparsity-controlling parameter

## export paramters
run_name = f'zscore_degs_allLigands_0.1_alpha{alpha}'
output_dir = f"/data1/rudenska/EYW/git_projects/SIG13/analysis_outs/spca/degs_zscore_allLigands"

## parameters for iterative sPCA
n_components=100
n_samples=100

## paramters for clustering
coherence = 0.8

## input files
adata = sc.read_h5ad("/data1/rudenska/EYW/SIG13/scanpy_outs/SIG13_doublets_DSB7_zscore_degs0.1cutoff.h5ad")

# Select ligands and genese for sparse PCA ---------------------------------------------------

## these anndata object contain z-scored expression values in .X and have already been subset to degs

## get mean average
adata_pb = sc.get.aggregate(adata, by=['ligand_call_DSB7','replicate'], func='mean')
adata_pb.obs['ligand_replicate'] = adata_pb.obs['ligand_call_DSB7'].astype(str) + '_' + adata_pb.obs['replicate'].astype(str)

# Build input df for sparse PCA ------------------------------------------------

input_df = pd.DataFrame(adata_pb.layers['mean'].copy(), 
                       index=adata_pb.obs.index, 
                       columns=adata_pb.var.index)

# don't scale the input data, since it is already z-scored
input_scaled = input_df.copy()

# Define sparse PCA -----------------------------------------------------------------------------
class NonNegativeSparsePCA(SparsePCA):
    def _fit(self, X, n_components, random_state):
        """Specialized 'fit' for Non-Negative SparsePCA."""

        code_init = self.V_init.T if self.V_init is not None else None
        dict_init = self.U_init.T if self.U_init is not None else None

        # Dictionary learning algorithm with non-negative dictionary atoms
        code, dictionary, E, self.n_iter_ = dict_learning(
            X.T,
            n_components,
            alpha=self.alpha,
            tol=self.tol,
            max_iter=self.max_iter,
            method=self.method,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=random_state,
            code_init=code_init,
            dict_init=dict_init,
            return_n_iter=True,
            positive_code=True,
        )

        self.components_ = code.T

        # Normalize components
        components_norm = np.linalg.norm(self.components_, axis=1)[:, np.newaxis]
        components_norm[components_norm == 0] = 1
        self.components_ /= components_norm
        self.n_components_ = len(self.components_)

        self.error_ = E
        return self

# Perform iterative sparse PCA ------------------------------------------------------------- 

# 3. Launch all bootstraps in parallel
def run_one(i, input_df, n_components, alpha):
    # 1) bootstrap with its own RNG
    boot = input_df.sample(n=len(input_df),
                          replace=True)

    # 2) fit a single-threaded SPCA
    spca = NonNegativeSparsePCA(
        n_components=n_components,
        alpha=alpha,
        random_state=None,
        n_jobs=1,
        method='cd',
        max_iter=10000,
        verbose=0
    )
    spca.fit(boot)

    comps = pd.DataFrame(spca.components_, columns=input_df.columns)
    comps.index = [f"{i}_{j}" for j in comps.index]
    return comps

results = Parallel(n_jobs=n_cores, verbose=10)(
    delayed(run_one)(
        i, input_scaled, n_components, alpha
    )
    for i in range(n_samples)
)

# 4. Reassemble into one DataFrame
atoms_df = pd.concat(results, axis=0)

# Cluster atoms --------------------------------------------------------------------------------------------

comps = dict()

embedding = UMAP(
    n_neighbors=15, n_components=5, metric=hellinger, random_state=100
).fit_transform(atoms_df)

clusterer = HDBSCAN(
    min_samples=int(n_samples*coherence),
    min_cluster_size=int(n_samples*coherence),
    cluster_selection_method="leaf",
    allow_single_cluster=True,
).fit(embedding)

labels = clusterer.labels_
membership_strengths = clusterer.probabilities_

result = np.empty((labels.max() + 1, atoms_df.shape[1]), dtype=np.float32)

for i in range(labels.max() + 1):
    mask = labels == i
    result[i] = (
        np.average(
            np.sqrt(atoms_df.values[mask]), axis=0, weights=membership_strengths[mask]
        )
        ** 2
    )
    result[i] /= result[i].sum()
    
comps['results'] = pd.DataFrame(result, columns=atoms_df.columns)
comps['results'] = comps['results'].div(comps['results'].apply(lambda x: np.linalg.norm(x), axis=1), axis=0)

# Rerun sparse PCA with consensus program number -----------------------------------------------------

## Fit NonNegativeSparsePCA
n_components = comps['results'].shape[0]
print(n_components)

sparse_pca = NonNegativeSparsePCA(n_components=n_components,  # number of sparse atoms to extract
                   alpha=alpha,  # Sparsity-controlling parameter
                   random_state=100,
                   n_jobs=1, verbose=1, method='cd', max_iter=10000)
sparse_pca.fit(input_df)

# create dfs where programs are columns
bulk_comps = pd.DataFrame(sparse_pca.components_, columns=input_df.columns).T.reset_index().rename(columns={'index': 'gene'})
bulk_codes = pd.DataFrame(sparse_pca.transform(input_df), index=input_df.index).reset_index().rename(columns={'index': 'interaction'})

# create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  

# save results
bulk_comps.to_csv(f"{output_dir}/{run_name}_sPCA_components.csv", index=False)
bulk_codes.to_csv(f"{output_dir}/{run_name}_sPCA_codes.csv", index=False)
joblib.dump(sparse_pca, f"{output_dir}/{run_name}_sparse_pca_model.joblib")
