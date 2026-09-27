# multithread parameters
import os
n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import dict_learning
import argparse

# Repo-relative paths (inputs: imports_stable/, outputs: analysis_outs/)
from pathlib import Path
_here = Path(__file__).resolve().parent
REPO_DIR = str(next(p for p in [_here, *_here.parents] if (p / "imports_stable").is_dir()))
IMPORTS_DIR = f"{REPO_DIR}/imports_stable"
OUTS_DIR = f"{REPO_DIR}/analysis_outs/02_combinatorial_screen_signalseq_SIG13"
import scanpy as sc

parser = argparse.ArgumentParser(
    description="Fit sPCA at a fixed component count for one alpha value (alpha parameter sweep)."
)
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument(
    "--n_components",
    type=int,
    default=78,
    help="Fixed component count, taken from the consensus cluster number of the alpha=1.0 analysis.",
)
args = parser.parse_args()

# Define parameters ------------------------------------------------------------------------------

alpha = args.alpha
n_components = args.n_components

run_name = f"zscore_degs_allLigands_0.1_alpha{alpha}"
output_dir = f"{OUTS_DIR}/spca_stability/alpha_sweep"

## input files (same DEG-subset, z-scored dataset as the original alpha=1.0 analysis)
adata = sc.read_h5ad(f"{IMPORTS_DIR}/SIG13/scanpy_outs/SIG13_doublets_DSB7_zscore_degs0.1cutoff.h5ad")

# Select ligands and genes for sparse PCA ---------------------------------------------------

## these anndata objects contain z-scored expression values in .X and have already been subset to degs

## get mean average
adata_pb = sc.get.aggregate(adata, by=["ligand_call_DSB7", "replicate"], func="mean")
adata_pb.obs["ligand_replicate"] = (
    adata_pb.obs["ligand_call_DSB7"].astype(str) + "_" + adata_pb.obs["replicate"].astype(str)
)

# Build input df for sparse PCA ------------------------------------------------

input_df = pd.DataFrame(
    adata_pb.layers["mean"].copy(), index=adata_pb.obs.index, columns=adata_pb.var.index
)

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

# Fit sPCA directly at the fixed consensus component count --------------------------------------
# No inner bootstrap / UMAP / HDBSCAN discovery step here: the program count is held fixed at
# n_components (the consensus number from the alpha=1.0 analysis) so that components can be
# matched to the alpha=1.0 reference programs across alpha values downstream.

sparse_pca = NonNegativeSparsePCA(
    n_components=n_components,
    alpha=alpha,
    random_state=100,
    n_jobs=n_cores,
    verbose=1,
    method="cd",
    max_iter=10000,
)
sparse_pca.fit(input_scaled)

# create dfs where programs are columns
bulk_comps = (
    pd.DataFrame(sparse_pca.components_, columns=input_scaled.columns)
    .T.reset_index()
    .rename(columns={"index": "gene"})
)
bulk_codes = (
    pd.DataFrame(sparse_pca.transform(input_scaled), index=input_scaled.index)
    .reset_index()
    .rename(columns={"index": "interaction"})
)

# create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save results
bulk_comps.to_csv(f"{output_dir}/{run_name}_sPCA_components.csv", index=False)
bulk_codes.to_csv(f"{output_dir}/{run_name}_sPCA_codes.csv", index=False)
joblib.dump(sparse_pca, f"{output_dir}/{run_name}_sparse_pca_model.joblib")
