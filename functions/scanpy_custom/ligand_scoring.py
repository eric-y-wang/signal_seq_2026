# Function to calculate ligand activity z-scores based on a modified CytoSig approach
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV
from joblib import Parallel, delayed

# Ridge Version
def _process_single_condition(y_obs, X, alpha_range, n_perms=1000, zscore_coeffs=True):
    cv_model = RidgeCV(alphas=alpha_range, fit_intercept=True, cv=5).fit(X, y_obs)
    best_alpha, beta_obs, r2_obs = cv_model.alpha_, cv_model.coef_, cv_model.score(X, y_obs)

    if zscore_coeffs:
        rng = np.random.default_rng()
        Y_perm_matrix = np.array([rng.permutation(y_obs) for _ in range(n_perms)]).T

        perm_model = Ridge(alpha=best_alpha, fit_intercept=True, random_state=67).fit(X, Y_perm_matrix)
        beta_perms = perm_model.coef_ 

        with np.errstate(divide='ignore', invalid='ignore'):
            z = (beta_obs - np.mean(beta_perms, axis=0)) / np.std(beta_perms, axis=0)
        
        return z, r2_obs, best_alpha
    
    else:
        return beta_obs, r2_obs, best_alpha

def calculate_ligand_activity_parallel(X_mat, Y_mat, alpha_range=np.logspace(-1, 3, 100), n_jobs=-1, verbose=0, zscore_coeffs=True, n_perms=1000):
    X = X_mat.values if isinstance(X_mat, pd.DataFrame) else X_mat
    Y = Y_mat.values if isinstance(Y_mat, pd.DataFrame) else Y_mat
    ligand_names = X_mat.columns if isinstance(X_mat, pd.DataFrame) else np.arange(X.shape[1])
    cond_names = Y_mat.columns if isinstance(Y_mat, pd.DataFrame) else np.arange(Y.shape[1])
    
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_process_single_condition)(Y[:, i], X, alpha_range, n_perms=n_perms, zscore_coeffs=zscore_coeffs) for i in range(len(cond_names))
    )

    zs, r2s, alphas = zip(*results)
    
    if verbose > 0:
        for name, a, r in zip(cond_names, alphas, r2s):
            print(f"Condition: {name} | Chosen Alpha: {a:.5f} | R2: {r:.4f}")

    return pd.DataFrame(dict(zip(cond_names, zs)), index=ligand_names), pd.Series(r2s, index=cond_names)

# Elastic Net Version
def _process_single_condition_elastic(y_obs, X, alpha_range, l1_ratios, n_perms=1000, zscore_coeffs=False):
    cv_model = ElasticNetCV(
        l1_ratio=l1_ratios, alphas=alpha_range, cv=5, tol=1e-3, 
        fit_intercept=True, max_iter=5000, n_jobs=1
    ).fit(X, y_obs)
    
    best_alpha, best_l1, beta_obs, r2_obs = cv_model.alpha_, cv_model.l1_ratio_, cv_model.coef_, cv_model.score(X, y_obs)

    if zscore_coeffs:
        rng = np.random.default_rng()
        Y_perm_matrix = np.array([rng.permutation(y_obs) for _ in range(n_perms)]).T

        perm_model = ElasticNet(
            alpha=best_alpha, l1_ratio=best_l1, tol=1e-3, 
            fit_intercept=True, random_state=67, max_iter=5000
        ).fit(X, Y_perm_matrix)
        
        beta_perms = perm_model.coef_ 
    
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (beta_obs - np.mean(beta_perms, axis=0)) / np.std(beta_perms, axis=0) if zscore_coeffs else beta_obs

        return z, r2_obs, best_alpha, best_l1
    
    else:
        return beta_obs, r2_obs, best_alpha, best_l1

def calculate_ligand_activity_parallel_elastic(X_mat, Y_mat, 
                                               alpha_range=np.logspace(-1, 3, 100),
                                               l1_ratios=np.logspace(-1,0,20),
                                               n_jobs=-1, verbose=0, zscore_coeffs=False, n_perms=1000):
    X = X_mat.values if isinstance(X_mat, pd.DataFrame) else X_mat
    Y = Y_mat.values if isinstance(Y_mat, pd.DataFrame) else Y_mat
    ligand_names = X_mat.columns if isinstance(X_mat, pd.DataFrame) else np.arange(X.shape[1])
    cond_names = Y_mat.columns if isinstance(Y_mat, pd.DataFrame) else np.arange(Y.shape[1])
    
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_process_single_condition_elastic)(Y[:, i], X, alpha_range, l1_ratios, n_perms=n_perms, zscore_coeffs=zscore_coeffs) for i in range(len(cond_names))
    )

    zs, r2s, alphas, l1s = zip(*results)

    if verbose > 0:
        for name, a, l, r in zip(cond_names, alphas, l1s, r2s):
            print(f"Condition: {name} | Alpha: {a:.5f}, L1: {l:.5f} | R2: {r:.4f}")

    return pd.DataFrame(dict(zip(cond_names, zs)), index=ligand_names), pd.Series(r2s, index=cond_names)