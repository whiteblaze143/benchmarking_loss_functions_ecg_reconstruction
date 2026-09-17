import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/dataset/evaluate_ptbxl_masking_control.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ptbxl_masking_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_patient_equal_weights_give_each_patient_unit_mass() -> None:
    patient_ids = np.asarray([1, 1, 1, 2, 3, 3])
    weights = MODULE.patient_equal_weights(patient_ids)
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3, 1, 1 / 2, 1 / 2])
    assert np.isclose(weights[patient_ids == 1].sum(), 1.0)
    assert np.isclose(weights[patient_ids == 2].sum(), 1.0)
    assert np.isclose(weights[patient_ids == 3].sum(), 1.0)
