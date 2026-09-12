import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from .data import _as_array

def compute_mmd_sq_df(sample1_idx, sample2_idx, dist_df, kernel="Gaussian", kernel_param=1.0):
    # Check kernel type
    if kernel not in ("Gaussian", "IMQ"):
        raise ValueError("kernel must be one of: 'Gaussian', 'IMQ'")

    N, M = len(sample1_idx), len(sample2_idx)

    # Extract distance submatrices using label-based indexing
    D_xx = dist_df.loc[sample1_idx, sample1_idx].values
    D_xy = dist_df.loc[sample1_idx, sample2_idx].values
    D_yy = dist_df.loc[sample2_idx, sample2_idx].values

    if kernel == "Gaussian":
        # Gaussian kernel terms
        term1 = np.exp(-(D_xx**2)/kernel_param).sum() / (N*N)
        term2 = -2*np.exp(-(D_xy**2)/kernel_param).sum() / (N*M)
        term3 = np.exp(-(D_yy**2)/kernel_param).sum() / (M*M)
    else:  # IMQ kernel
        term1 = (1/np.sqrt(kernel_param + D_xx**2)).sum() / (N*N)
        term2 = -2*(1/np.sqrt(kernel_param + D_xy**2)).sum() / (N*M)
        term3 = (1/np.sqrt(kernel_param + D_yy**2)).sum() / (M*M)

    return term1 + term2 + term3


def compute_perm_mmd_sq(sample1_idx, sample2_idx, dist_matrix, patient_data, kernel="Gaussian", kernel_param=1.0):
    patient_data = patient_data.copy()  # avoid side effects
    patient_data["point"] = patient_data.index  # preserve original row order

    # sort clusters by size (smallest first)
    indices = sorted([sample1_idx, sample2_idx], key=len)
    subset_data = patient_data.loc[list(indices[0]) + list(indices[1])].copy()

    # create block_id for each (region, polygon_id)
    subset_data["block_id"] = (
        subset_data.groupby(["region", "polygon_id"], observed=False).ngroup() + 1
    )

    # size of smaller cluster
    regions = subset_data["region"].unique()
    n_size = (subset_data["region"] == regions[0]).sum()

    # sample blocks until size constraint is met
    blk_ids, n_permuted = [], 0
    all_blocks = subset_data["block_id"].unique()
    while n_permuted < n_size:
        remaining = np.setdiff1d(all_blocks, blk_ids)
        if len(remaining) == 0:
            break
        new_block = np.random.choice(remaining, 1)[0]
        blk_ids.append(new_block)
        n_permuted = subset_data.loc[subset_data["block_id"].isin(blk_ids)].shape[0]

    # assign permuted region labels
    subset_data["permuted_region"] = np.where(subset_data["block_id"].isin(blk_ids), regions[0], regions[1])

    # compute MMD² on permuted labels
    perm_mmd_sq = compute_mmd_sq_df(
        subset_data.loc[subset_data["permuted_region"] == regions[0], "point"].values,
        subset_data.loc[subset_data["permuted_region"] == regions[1], "point"].values,
        dist_matrix, kernel=kernel, kernel_param=kernel_param
    )

    return perm_mmd_sq

def two_sample_mmd(sample1_idx, sample2_idx, dist_matrix, patient_data,
                   kernel="Gaussian", kernel_param=1, nperm=200):
    """Compute observed MMD², null distribution, and p-value."""
    
    obs = compute_mmd_sq_df(sample1_idx, sample2_idx, dist_matrix, kernel, kernel_param)  # observed MMD²
    null = np.array([compute_perm_mmd_sq(sample1_idx, sample2_idx, dist_matrix, patient_data, kernel, kernel_param)
                     for _ in range(nperm)])  # null distribution by permutation
    p_val = np.mean(null >= obs)  # permutation p-value
    
    return {"obs_mmd_sq": obs, "p_value": p_val, "null_dist": null}


def _caller_adata_name(adata):
    import inspect

    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back else None
    if caller is None:
        return "adata"

    for name, value in caller.f_locals.items():
        if value is adata:
            return name
    return "adata"


def multiple_comparison(adata, kernel="Gaussian", kernel_param=1.0, nperm=200,
                        adj_p="BH", region_key="labels",
                        block_key="repspat_block_id",
                        distance_key="repspat_distances"):
    """Pairwise two-sample MMD test with multiple testing correction."""
    for key in (region_key, block_key):
        if key not in adata.obs:
            raise KeyError(f"'{key}' not found in adata.obs.")
    if distance_key not in adata.obsp:
        raise KeyError(f"'{distance_key}' not found in adata.obsp.")

    patient_data = adata.obs[[region_key, block_key]].rename(
        columns={region_key: "region", block_key: "polygon_id"}
    ).copy()
    dist_matrix = pd.DataFrame(
        _as_array(adata.obsp[distance_key]),
        index=adata.obs_names,
        columns=adata.obs_names,
    )
    
    results = []
    # Loop over all unique cluster pairs
    for r1, r2 in combinations(patient_data["region"].unique(), 2):
        idx1 = patient_data[patient_data["region"] == r1].index.tolist()
        idx2 = patient_data[patient_data["region"] == r2].index.tolist()
        res = two_sample_mmd(idx1, idx2, dist_matrix, patient_data, 
                             kernel=kernel, kernel_param=kernel_param, nperm=nperm)
        
        results.append({
            "region_1": r1,
            "region_2": r2,
            "obs_mmd_sq": res["obs_mmd_sq"],
            "p_value": res["p_value"],
            "null_dist": res["null_dist"]
        })

    df = pd.DataFrame(results)
    
    # Adjust p-values for multiple comparisons
    method_map = {"BH": "fdr_bh", "bonferroni": "bonferroni", "holm": "holm"}
    if adj_p not in method_map:
        raise ValueError(f"adj_p must be one of: {list(method_map.keys())}")
    df["adj_p"] = multipletests(df["p_value"], method=method_map[adj_p])[1]
    adata.uns["repspat_mmd_results"] = df
    print(
        "MMD results are stored in "
        f'{_caller_adata_name(adata)}.uns["repspat_mmd_results"]'
    )
    
    return adata
