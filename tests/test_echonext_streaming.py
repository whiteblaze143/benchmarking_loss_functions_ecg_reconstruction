from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_comprehensive_registry import LEAD_NAMES
from scripts.evaluate_echonext import (
    EchoNextWaveforms,
    repeat_condition_tabular,
    write_json_atomic,
)


def provenance(normalization="none") -> dict:
    return {
        "source": "fixture",
        "version": "1",
        "sampling_rate_hz": 250,
        "lead_order": LEAD_NAMES,
        "units": "mV",
        "normalization": normalization,
    }


def test_echonext_streams_time_lead_batches_to_500_hz() -> None:
    values = np.zeros((3, 2500, 12), dtype=np.float32)
    values[:, :, 1] = np.linspace(-1, 1, 2500)
    waveforms = EchoNextWaveforms(values, provenance())
    batch = waveforms.batch(1, 3)
    assert len(waveforms) == 3
    assert batch.shape == (2, 12, 5000)
    assert np.isfinite(batch).all()


def test_echonext_inverts_dataset_zscore_but_rejects_minmax() -> None:
    values = np.zeros((1, 12, 2500), dtype=np.float32)
    normalization = {
        "kind": "dataset_zscore",
        "mean": [2.0] * 12,
        "std": [0.5] * 12,
    }
    batch = EchoNextWaveforms(values, provenance(normalization)).batch(0, 1)
    assert np.allclose(batch.mean(), 2.0, atol=1e-3)
    with pytest.raises(ValueError, match="non-invertible normalization"):
        EchoNextWaveforms(values, provenance({"kind": "per_record_minmax"}))


def test_condition_tabular_repetition_matches_condition_major_signal_order() -> None:
    tabular = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    repeated = repeat_condition_tabular(tabular, 3)
    np.testing.assert_array_equal(
        repeated,
        np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )


def test_atomic_json_checkpoint_leaves_no_staging_file(tmp_path) -> None:
    output = tmp_path / "partial.json"
    write_json_atomic(output, {"models": {"cell": {"value": 1.0}}})
    assert output.read_text().startswith("{")
    assert not (tmp_path / "partial.json.tmp").exists()
