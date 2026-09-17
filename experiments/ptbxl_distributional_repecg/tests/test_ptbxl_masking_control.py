import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/dataset/build_ptbxl_masking_control.py"
SPEC = importlib.util.spec_from_file_location("build_ptbxl_masking_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mask_standardized_beats_keeps_i_ii_and_zeros_rest() -> None:
    beats = np.arange(3 * 256 * 8, dtype=np.float32).reshape(3, 256, 8)
    masked = MODULE.mask_standardized_beats(beats)
    np.testing.assert_array_equal(masked[..., :2], beats[..., :2])
    assert np.all(masked[..., 2:] == 0.0)
    assert not np.shares_memory(masked, beats)


def test_mask_standardized_beats_rejects_wrong_basis() -> None:
    with pytest.raises(ValueError, match=r"\[B,256,8\]"):
        MODULE.mask_standardized_beats(np.zeros((2, 256, 12), dtype=np.float32))
