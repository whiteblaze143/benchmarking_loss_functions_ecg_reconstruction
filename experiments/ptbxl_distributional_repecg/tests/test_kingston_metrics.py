import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/dataset/evaluate_kingston_negative_control.py"
SPEC = importlib.util.spec_from_file_location("evaluate_kingston_negative_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_binary_targets_use_afib_as_positive() -> None:
    labels = np.asarray(["SINUS", "AFIB_AFLT", "SINUS"])
    np.testing.assert_array_equal(MODULE.binary_targets(labels), [0, 1, 0])


def test_binary_targets_reject_unknown_label() -> None:
    with pytest.raises(ValueError, match="unexpected Kingston labels"):
        MODULE.binary_targets(np.asarray(["OTHER"]))
