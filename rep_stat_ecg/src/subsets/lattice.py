"""Subset Lattice and Information Rank utilities for GRAIL-ECG.

Defines:
1. Canonical 12-lead ordering: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6].
2. Primary 8 independent basis: [I, II, V1, V2, V3, V4, V5, V6].
3. Linear transform matrix expressing all 12 leads in terms of the 8 basis leads.
4. Exhaustive 4,095 non-empty subset enumeration with bitmasks, cardinality, and independent measurement rank.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple
import numpy as np

LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_TO_IDX_12 = {name: i for i, name in enumerate(LEAD_NAMES_12)}

INDEPENDENT_LEAD_NAMES_8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
INDEPENDENT_LEAD_INDICES_8 = [0, 1, 6, 7, 8, 9, 10, 11]

# 12x8 matrix M such that Lead_12 = M @ Lead_8
# Lead_8 = [I, II, V1, V2, V3, V4, V5, V6]^T
# Lead I   = 1*I
# Lead II  = 1*II
# Lead III = -1*I + 1*II
# aVR      = -0.5*I - 0.5*II
# aVL      = 1*I - 0.5*II
# aVF      = -0.5*I + 1*II
# V1..V6   = 1*V1 .. 1*V6
BASIS_MATRIX_12_TO_8 = np.zeros((12, 8), dtype=np.float64)
# I
BASIS_MATRIX_12_TO_8[0, 0] = 1.0
# II
BASIS_MATRIX_12_TO_8[1, 1] = 1.0
# III = II - I
BASIS_MATRIX_12_TO_8[2, 0] = -1.0
BASIS_MATRIX_12_TO_8[2, 1] = 1.0
# aVR = -(I + II)/2
BASIS_MATRIX_12_TO_8[3, 0] = -0.5
BASIS_MATRIX_12_TO_8[3, 1] = -0.5
# aVL = (I - III)/2 = I - II/2
BASIS_MATRIX_12_TO_8[4, 0] = 1.0
BASIS_MATRIX_12_TO_8[4, 1] = -0.5
# aVF = (II + III)/2 = II - I/2
BASIS_MATRIX_12_TO_8[5, 0] = -0.5
BASIS_MATRIX_12_TO_8[5, 1] = 1.0
# V1..V6
for v_idx in range(6):
    BASIS_MATRIX_12_TO_8[6 + v_idx, 2 + v_idx] = 1.0


class SubsetInfo(NamedTuple):
    mask: int
    indices: list[int]
    lead_names: list[str]
    k_displayed: int
    r_independent: int
    n_limb: int
    limb_rank: int
    n_precordial: int


def compute_subset_rank(indices: list[int]) -> tuple[int, int]:
    """Computes total independent rank and limb rank for a set of lead indices in 0..11."""
    if not indices:
        return 0, 0
    submatrix = BASIS_MATRIX_12_TO_8[indices, :]
    total_rank = int(np.linalg.matrix_rank(submatrix))

    limb_indices = [idx for idx in indices if idx < 6]
    if limb_indices:
        limb_submatrix = BASIS_MATRIX_12_TO_8[limb_indices, :2]
        limb_rank = int(np.linalg.matrix_rank(limb_submatrix))
    else:
        limb_rank = 0

    return total_rank, limb_rank


def get_subset_info(mask: int) -> SubsetInfo:
    """Decodes an integer bitmask (1 to 4095) into SubsetInfo."""
    assert 1 <= mask < (1 << 12), f"Invalid mask {mask}"
    indices = [i for i in range(12) if (mask & (1 << i))]
    lead_names = [LEAD_NAMES_12[i] for i in indices]
    k_displayed = len(indices)
    r_independent, limb_rank = compute_subset_rank(indices)
    n_limb = sum(1 for i in indices if i < 6)
    n_precordial = sum(1 for i in indices if i >= 6)

    return SubsetInfo(
        mask=mask,
        indices=indices,
        lead_names=lead_names,
        k_displayed=k_displayed,
        r_independent=r_independent,
        n_limb=n_limb,
        limb_rank=limb_rank,
        n_precordial=n_precordial,
    )


def enumerate_all_4095_subsets() -> list[SubsetInfo]:
    """Enumerates all 4,095 non-empty subsets of the 12 displayed leads."""
    return [get_subset_info(mask) for mask in range(1, 1 << 12)]
