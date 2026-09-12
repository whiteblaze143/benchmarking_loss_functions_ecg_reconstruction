"""Benjamini-Hochberg False Discovery Rate (FDR) multiple testing correction.

Implements Step 3(c) of repSpat:
- Corrects raw p-values across all binom(G, 2) pairwise two-sample tests.
- Re-orders p-values and applies Benjamini-Hochberg step-up adjustment:
  p_adj(i) = min_{j >= i} ( (M / j) * p_(j) )
- Edges in the similarity graph are added for cluster pairs where p_adj >= alpha
  (insufficient evidence that electrophysiological distributions differ).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fdr_correction_bh(
    p_values: np.ndarray | list[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array-like
        Raw p-values of length M.
    alpha : float
        Significance level (default: 0.05).

    Returns
    -------
    reject : np.ndarray of bool
        Boolean array where True indicates H_0 rejected (distributions differ).
    p_adj : np.ndarray of float
        Monotonically non-decreasing BH-adjusted p-values.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    M = len(p_arr)
    if M == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    # Sort p-values ascending
    order = np.argsort(p_arr)
    p_sorted = p_arr[order]

    # Step-up adjustment: p_adj = p_sorted * (M / rank)
    ranks = np.arange(1, M + 1, dtype=np.float64)
    adjusted_sorted = p_sorted * (float(M) / ranks)

    # Ensure monotonicity from right to left: p_adj(i) = min_{j >= i} adjusted(j)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    # Unsort back to original input order
    p_adj = np.empty_like(adjusted_sorted)
    p_adj[order] = adjusted_sorted

    reject = p_adj < alpha
    return reject, p_adj


def apply_fdr_to_pairwise_results(
    pairwise_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Applies Benjamini-Hochberg adjustment to pairwise testing DataFrame.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        DataFrame with column 'p_val'.
    alpha : float
        FDR alpha threshold.

    Returns
    -------
    annotated_df : pd.DataFrame
        DataFrame with added columns: 'adj_p', 'reject_h0', 'is_similar'.
    """
    if pairwise_df.empty:
        df = pairwise_df.copy()
        df["adj_p"] = []
        df["reject_h0"] = []
        df["is_similar"] = []
        return df

    df = pairwise_df.copy()
    reject, p_adj = fdr_correction_bh(df["p_val"].values, alpha=alpha)
    df["adj_p"] = p_adj
    df["reject_h0"] = reject
    # In repSpat: distributions are considered similar if we fail to reject H_0 (adj_p >= alpha)
    df["is_similar"] = ~reject
    return df
