"""Exact Shapley and Pairwise Interaction calculation on subset lattices.

Computes:
1. Exact Shapley value:
    phi_i = sum_{S subseteq N \\ {i}} (|S|! * (n - |S| - 1)! / n!) * [v(S union {i}) - v(S)]
2. Exact 2nd-order pairwise interaction index (synergy / redundancy):
    I_ij = sum_{S subseteq N \\ {i, j}} (|S|! * (n - |S| - 2)! / (n - 1)!) * [v(S union {i, j}) - v(S union {i}) - v(S union {j}) + v(S)]
"""

from __future__ import annotations

import math
from typing import Callable
import numpy as np


def compute_exact_shapley_values(
    n: int,
    value_function: Callable[[int], float],
) -> np.ndarray:
    """Computes exact Shapley values for all n players given a bitmask value function v(mask).

    Args:
        n: Number of players (e.g. 12).
        value_function: Function mapping integer bitmask (0 to 2^n - 1) to float value.

    Returns:
        phi: Array of shape (n,) containing exact Shapley values.
    """
    total_masks = 1 << n
    phi = np.zeros(n, dtype=np.float64)

    # Precompute values for all 2^n subsets
    v = np.zeros(total_masks, dtype=np.float64)
    for mask in range(total_masks):
        v[mask] = value_function(mask)

    # Factorial weights lookup table
    weights = np.zeros(n, dtype=np.float64)
    for s_size in range(n):
        weights[s_size] = (math.factorial(s_size) * math.factorial(n - s_size - 1)) / math.factorial(n)

    for i in range(n):
        bit_i = 1 << i
        val_i = 0.0
        # Iterate over all subsets S that do not contain player i
        for mask in range(total_masks):
            if not (mask & bit_i):
                s_size = bin(mask).count("1")
                marginal = v[mask | bit_i] - v[mask]
                val_i += weights[s_size] * marginal
        phi[i] = val_i

    return phi


def compute_exact_pairwise_interactions(
    n: int,
    value_function: Callable[[int], float],
) -> np.ndarray:
    """Computes exact 2nd-order interaction matrix I_ij (synergy/redundancy) for all pairs.

    Args:
        n: Number of players.
        value_function: Function mapping integer bitmask to float value.

    Returns:
        I: Symmetric matrix of shape (n, n), with diagonal set to 0.
    """
    total_masks = 1 << n
    I_mat = np.zeros((n, n), dtype=np.float64)

    v = np.zeros(total_masks, dtype=np.float64)
    for mask in range(total_masks):
        v[mask] = value_function(mask)

    # Factorial weights for 2nd order interaction:
    # weight(s) = s! * (n - s - 2)! / (n - 1)!
    weights = np.zeros(n - 1, dtype=np.float64)
    for s_size in range(n - 1):
        weights[s_size] = (math.factorial(s_size) * math.factorial(n - s_size - 2)) / math.factorial(n - 1)

    for i in range(n):
        bit_i = 1 << i
        for j in range(i + 1, n):
            bit_j = 1 << j
            val_ij = 0.0
            for mask in range(total_masks):
                if not (mask & bit_i) and not (mask & bit_j):
                    s_size = bin(mask).count("1")
                    # second-order difference: v(S + i + j) - v(S + i) - v(S + j) + v(S)
                    diff2 = v[mask | bit_i | bit_j] - v[mask | bit_i] - v[mask | bit_j] + v[mask]
                    val_ij += weights[s_size] * diff2
            I_mat[i, j] = val_ij
            I_mat[j, i] = val_ij

    return I_mat
