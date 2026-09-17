from __future__ import annotations

import numpy as np
import pytest

from repecg.common.preprocess import LeadScaler
from repecg.evaluation.kingston import zero_padded_basis

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def test_kingston_mapping_uses_only_named_i_and_ii_and_zero_pads_after_scaling() -> None:
    signal = np.column_stack([
        np.arange(5.0),
        np.arange(5.0) + 10,
        np.arange(5.0) + 20,
        np.arange(5.0) + 30,
    ])
    scaler = LeadScaler(mean=np.arange(8.0), std=np.arange(8.0) + 1)
    result = zero_padded_basis(signal, ("I", "II", "III", "V"), scaler)
    np.testing.assert_array_equal(result.observed_lead_mask, [1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(result.physical_mv[:, :2], signal[:, :2])
    np.testing.assert_allclose(result.standardized[:, 0], signal[:, 0])
    np.testing.assert_allclose(result.standardized[:, 1], (signal[:, 1] - 1) / 2)
    np.testing.assert_array_equal(result.physical_mv[:, 2:], 0)
    np.testing.assert_array_equal(result.standardized[:, 2:], 0)


def test_kingston_mapping_rejects_missing_named_detection_leads() -> None:
    scaler = LeadScaler(mean=np.zeros(8), std=np.ones(8))
    with pytest.raises(ValueError, match="requires explicitly named I and II"):
        zero_padded_basis(np.zeros((5, 2)), ("I", "V"), scaler)


def test_kingston_rhythm_vocabulary_matches_task_registry() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/dataset/build_kingston_negative_control.py"
    spec = spec_from_file_location("build_kingston_negative_control", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._rhythm_label("SINUS") == "SINUS"
    assert module._rhythm_label("AFIB/AFLT") == "AFIB_AFLT"
    with pytest.raises(ValueError, match="unsupported Kingston rhythm"):
        module._rhythm_label("UNKNOWN")
