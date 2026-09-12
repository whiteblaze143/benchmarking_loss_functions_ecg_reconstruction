"""Multiple testing correction for pairwise episode comparisons.

Implements Step 3(c) of repSpat:
- Pairwise MMD^2 testing across all G*(G-1)/2 episode pairs.
- Benjamini-Hochberg (BH) False Discovery Rate (FDR) adjustment at alpha = 0.05.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from .block_permutation import two_sample_block_permutation_test


def adjust_pvalues_fdr(
    p_values: np.ndarray | list[float],
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> tuple[np.ndarray, np.ndarray]:
    """Adjusts p-values for multiple comparisons using FDR or FWER control.

    Parameters
    ----------
    p_values : array-like
        Raw p-values across pairwise hypothesis tests.
    alpha : float
        Significance level (default: 0.05).
    method : str
        Method for multiple testing correction ('fdr_bh', 'bonferroni', 'holm').

    Returns
    -------
    rejected : np.ndarray
        Boolean array indicating whether H0 is rejected (significant difference).
    adj_p : np.ndarray
        Adjusted p-values.
    """
    p_values = np.asarray(p_values, dtype=float)
    if len(p_values) == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    rejected, adj_p, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return rejected, adj_p


def pairwise_episode_testing(
    labels: np.ndarray,
    dist_matrix: np.ndarray,
    block_ids: np.ndarray,
    kernel: str = "IMQ",
    kernel_param: float = 1.0,
    n_permutations: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Performs pairwise block-permutation MMD tests across all episode pairs.

    Parameters
    ----------
    labels : np.ndarray
        Episode labels [n].
    dist_matrix : np.ndarray
        Pairwise distance matrix D [n, n].
    block_ids : np.ndarray
        Block assignments [n].
    kernel : str
        'IMQ' or 'Gaussian'.
    kernel_param : float
        Kernel parameter.
    n_permutations : int
        Permutations B per pair.
    alpha : float
        FDR significance threshold.
    random_state : int
        Seed.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame containing columns:
        ['episode_1', 'episode_2', 'obs_mmd_sq', 'p_value', 'adj_p', 'is_similar']
        where is_similar = (adj_p >= alpha).
    """
    unique_labels = sorted(np.unique(labels))
    pairs = list(itertools.combinations(unique_labels, 2))

    records = []
    for ep1, ep2 in pairs:
        idx1 = np.flatnonzero(labels == ep1)
        idx2 = np.flatnonzero(labels == ep2)

        res = two_sample_block_permutation_test(
            idx_g=idx1,
            idx_h=idx2,
            dist_matrix=dist_matrix,
            block_ids=block_ids,
            kernel=kernel,
            kernel_param=kernel_param,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        records.append({
            "episode_1": int(ep1),
            "episode_2": int(ep2),
            "obs_mmd_sq": float(res["obs_mmd_sq"]),
            "p_value": float(res["p_value"]),
            "null_dist": res["null_distribution"],
        })

    if not records:
        return pd.DataFrame(columns=["episode_1", "episode_2", "obs_mmd_sq", "p_value", "adj_p", "is_similar"])

    df = pd.DataFrame(records)
    _, adj_p = adjust_pvalues_fdr(df["p_value"].values, alpha=alpha, method="fdr_bh")
    df["adj_p"] = adj_p
    # Under repSpat: adj_p >= alpha means insufficient evidence to conclude distributions differ
    # i.e., episodes share statistically indistinguishable multivariate distributions
    df["is_similar"] = df["adj_p"] >= alpha

    return df
