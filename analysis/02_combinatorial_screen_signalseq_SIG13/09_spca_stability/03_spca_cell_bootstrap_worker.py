# multithread parameters
import os
n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import dict_learning

# Repo-relative paths (inputs: imports_stable/, outputs: analysis_outs/)
from pathlib import Path
_here = Path(__file__).resolve().parent
REPO_DIR = str(next(p for p in [_here, *_here.parents] if (p / "imports_stable").is_dir()))
IMPORTS_DIR = f"{REPO_DIR}/imports_stable"
OUTS_DIR = f"{REPO_DIR}/analysis_outs/02_combinatorial_screen_signalseq_SIG13"
import scanpy as sc

parser = argparse.ArgumentParser(
    description=(
        "Cell-level bootstrap of the alpha=1.0 sPCA analysis: resample cells with replacement "
        "(same size per ligand x replicate group) prior to pseudobulking, refit sPCA at the fixed "
        "consensus component count, and save the gene loadings for each bootstrap iteration."
    )
)
parser.add_argument("--n_iter", type=int, default=1000, help="Number of bootstrap iterations (the full run in 03_run_bootstrap.sh uses 2000; use a smaller value for quick tests).")
parser.add_argument("--start_iter", type=int, default=0, help="Starting iteration index (for chunking across array jobs).")
parser.add_argument("--n_components", type=int, default=78, help="Fixed component count (consensus number from the alpha=1.0 analysis).")
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--seed_base", type=int, default=0)
args = parser.parse_args()

output_dir = f"{OUTS_DIR}/spca_stability/bootstrap/components"
os.makedirs(output_dir, exist_ok=True)

# Load data once -----------------------------------------------------------------------------
# these anndata objects contain z-scored expression values in .X and have already been subset to degs
adata = sc.read_h5ad(f"{IMPORTS_DIR}/SIG13/scanpy_outs/SIG13_doublets_DSB7_zscore_degs0.1cutoff.h5ad")

var_names = adata.var.index.to_numpy()

# Densify once up front: a plain numpy ndarray (rather than a scipy sparse matrix) lets joblib's
# loky backend automatically memory-map it to disk and share it read-only across worker
# processes, instead of pickling/duplicating an 11GB+ matrix into every worker.
X = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)

# positional (not label) indices per ligand x replicate group, matching the pseudobulk grouping
# used by the main spca pipeline (07_spca/01_spca_degs_zscore_expression_allLigands.py)
group_indices = adata.obs.groupby(["ligand_call_DSB7", "replicate"], observed=True).indices
group_keys = sorted(group_indices.keys())
group_idx_arrays = [np.asarray(group_indices[k]) for k in group_keys]

print(f"Loaded {X.shape[0]} cells x {X.shape[1]} genes, {len(group_keys)} ligand x replicate groups")

# Define sparse PCA -----------------------------------------------------------------------------
class NonNegativeSparsePCA(SparsePCA):
    def _fit(self, X, n_components, random_state):
        """Specialized 'fit' for Non-Negative SparsePCA."""

        code_init = self.V_init.T if self.V_init is not None else None
        dict_init = self.U_init.T if self.U_init is not None else None

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

        components_norm = np.linalg.norm(self.components_, axis=1)[:, np.newaxis]
        components_norm[components_norm == 0] = 1
        self.components_ /= components_norm
        self.n_components_ = len(self.components_)

        self.error_ = E
        return self


def run_one_bootstrap(iter_id, X, group_idx_arrays, var_names, n_components, alpha, seed):
    out_path = f"{output_dir}/bootstrap_iter{iter_id:04d}_sPCA_components.csv"
    if os.path.exists(out_path):
        return iter_id  # allow resuming a partially-completed run

    rng = np.random.default_rng(seed)

    # resample cells with replacement within each ligand x replicate group, to that group's own
    # original size, then recompute the pseudobulk mean -- this preserves the pseudobulk design
    # (means are computed per group) while capturing cell-level sampling noise within each group
    n_genes = X.shape[1]
    pb = np.empty((len(group_idx_arrays), n_genes), dtype=np.float32)
    for gi, idx in enumerate(group_idx_arrays):
        resampled = rng.choice(idx, size=len(idx), replace=True)
        pb[gi] = X[resampled].mean(axis=0)

    sparse_pca = NonNegativeSparsePCA(
        n_components=n_components,
        alpha=alpha,
        random_state=int(seed),
        n_jobs=1,
        verbose=0,
        method="cd",
        max_iter=10000,
    )
    sparse_pca.fit(pb)

    comps = (
        pd.DataFrame(sparse_pca.components_, columns=var_names)
        .T.reset_index()
        .rename(columns={"index": "gene"})
    )
    comps.to_csv(out_path, index=False)
    return iter_id


results = Parallel(n_jobs=n_cores, verbose=10)(
    delayed(run_one_bootstrap)(
        i, X, group_idx_arrays, var_names, args.n_components, args.alpha, args.seed_base + i
    )
    for i in range(args.start_iter, args.start_iter + args.n_iter)
)

print(f"Completed bootstrap iterations: {results}")
