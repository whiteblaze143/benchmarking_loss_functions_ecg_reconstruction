from pathlib import Path

import numpy as np
import pytest
import torch

from repecg.evaluation.delineation_labels import (
    ISP_CODE_TO_CLASS,
    intervals_to_mask,
    ludb_lead_mask,
    rdb_payload_masks,
)


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"


def test_intervals_to_mask_preserves_inclusive_native_coordinates() -> None:
    mask = intervals_to_mask([(0, 1, 2), (1, 4, 4), (2, 7, 8)], 10, code_to_class=ISP_CODE_TO_CLASS)
    assert mask.tolist() == [0, 1, 1, 0, 2, 0, 0, 3, 3, 0]


def test_intervals_to_mask_rejects_overlap_and_bounds() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        intervals_to_mask([(0, 1, 3), (1, 3, 4)], 10, code_to_class=ISP_CODE_TO_CLASS)
    with pytest.raises(ValueError, match="invalid inclusive"):
        intervals_to_mask([(0, 1, 10)], 10, code_to_class=ISP_CODE_TO_CLASS)


def test_ludb_annotation_triplets_yield_native_mask() -> None:
    mask = ludb_lead_mask(DATA / "ludb/1", "i", length=5000)
    assert set(np.unique(mask)).issubset({0, 1, 2, 3})
    assert int((mask == 1).sum()) > 0
    assert int((mask == 2).sum()) > 0
    assert int((mask == 3).sum()) > 0


def test_rdb_payload_masks_rejects_invalid_labeled_samples() -> None:
    payload = {
        "segmentation": torch.tensor([[0, 1, -1]] * 12),
        "seg_valid": torch.tensor([[True, True, False]] * 12),
    }
    segmentation, valid = rdb_payload_masks(payload)
    assert segmentation.shape == valid.shape == (12, 3)
    payload["seg_valid"][0, 2] = True
    with pytest.raises(ValueError, match="includes invalid"):
        rdb_payload_masks(payload)
