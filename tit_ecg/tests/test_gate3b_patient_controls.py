import numpy as np
import pytest

from tit_ecg.scripts.run_empirical_multi_dataset_benchmark import (
    GATE3B_REQUIRED_PURITY,
    beat_ids_from_segmentation,
)
from tit_ecg.src.mmd_test import create_attribute_blocks


def test_beat_ids_are_defined_by_independent_qrs_occurrences():
    segmentation = np.array([0, 1, 2, 2, 3, 0, 1, 2, 2, 3, 0])

    beat_ids = beat_ids_from_segmentation(segmentation)

    assert np.array_equal(beat_ids, [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])


def test_gate3b_requires_exact_membership_instead_of_a_heuristic_threshold():
    assert GATE3B_REQUIRED_PURITY == 1.0


def test_unblocked_scheme_uses_individual_observations():
    blocks = create_attribute_blocks(
        X=np.arange(12).reshape(6, 2),
        labels=np.array([0, 0, 0, 1, 1, 1]),
        m_star=2,
        block_mode="unblocked",
    )

    assert [block.tolist() for block in blocks[0]] == [[0], [1], [2]]
    assert [block.tolist() for block in blocks[1]] == [[3], [4], [5]]


def test_unknown_block_scheme_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported block_mode"):
        create_attribute_blocks(
            X=np.arange(8).reshape(4, 2),
            labels=np.array([0, 0, 1, 1]),
            m_star=2,
            block_mode="typo",
        )
