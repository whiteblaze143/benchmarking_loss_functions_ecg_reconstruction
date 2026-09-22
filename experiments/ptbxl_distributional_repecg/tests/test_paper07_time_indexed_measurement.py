from __future__ import annotations

import numpy as np

from repecg.common.kernels import biased_mmd2
from repecg.paper07_operator import response_atoms, time_indexed_response_atoms


def test_within_cell_temporal_permutation_is_invisible_without_tau_and_visible_with_tau() -> None:
    """Exact multiset equality is the time-agnostic null control."""
    beat = np.zeros((1, 256, 8), dtype=np.float64)
    beat[0, :, 0] = np.linspace(-1.0, 1.0, 256) ** 3
    q = np.eye(8)[0]
    old = response_atoms(beat, q, voltage_scale=1.0)[0, :16]
    new = time_indexed_response_atoms(beat, q, voltage_scale=1.0)[0, :16]
    permutation = np.asarray([3, 11, 0, 14, 6, 9, 1, 15, 7, 4, 13, 2, 10, 5, 12, 8])
    old_permuted = old[permutation]
    new_permuted = np.concatenate((new[:, :1], old_permuted), axis=1)
    # The old response cell is exactly the same empirical multiset.
    old_order = np.lexsort(old.T[::-1])
    old_permuted_order = np.lexsort(old_permuted.T[::-1])
    assert np.array_equal(old[old_order], old_permuted[old_permuted_order])
    # The new joint (tau, response) multiset is not the same.
    assert not np.array_equal(new[np.lexsort(new.T[::-1])], new_permuted[np.lexsort(new_permuted.T[::-1])])
    assert biased_mmd2(new, new_permuted) > 0.0
