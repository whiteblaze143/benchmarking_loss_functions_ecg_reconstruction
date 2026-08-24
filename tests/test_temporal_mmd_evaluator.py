from __future__ import annotations

import json

import numpy as np
import torch

import scripts.evaluate_temporal_mmd as evaluator


class ImmediateExecutor:
    def map(self, function, values):
        return [function(value) for value in values]


class MustNotIterate:
    def __iter__(self):
        raise AssertionError("valid target cache should be reused without data iteration")


def test_target_feature_cache_is_content_bound_and_reused(tmp_path, monkeypatch):
    def fixed_features(signal):
        value = float(np.asarray(signal).mean())
        return {
            "R_Amp": {
                "sample_index": np.array([1, 5]),
                "value": np.array([value, value + 1.0]),
            }
        }

    monkeypatch.setattr(evaluator, "extract_clinical_features", fixed_features)
    signals = torch.zeros((2, 12, 8), dtype=torch.float32)
    signals[0, 8] = 2.0
    signals[0, 11] = 3.0
    signals[1, 8] = 4.0
    signals[1, 11] = 5.0
    loader = [(signals, torch.tensor([101, 202]))]
    leads = {"V3": 8, "V6": 11}

    lookup, metadata = evaluator.load_or_build_target_cache(
        "ptb_xl",
        loader,
        ImmediateExecutor(),
        leads,
        tmp_path,
        "test-content-root",
        "extractor-digest",
    )

    assert metadata["records"] == 2
    assert metadata["rows"] == 8
    np.testing.assert_allclose(
        lookup[("101", "V3", "R_Amp")]["value"], [2.0, 3.0]
    )
    np.testing.assert_array_equal(
        lookup[("101", "V3", "R_Amp")]["sample_index"], [1, 5]
    )
    np.testing.assert_allclose(
        lookup[("202", "V6", "R_Amp")]["value"], [5.0, 6.0]
    )
    parquet = tmp_path / "_target_ptb_xl_features.parquet"
    sidecar = tmp_path / "_target_ptb_xl_features.json"
    assert parquet.is_file() and sidecar.is_file()
    assert json.loads(sidecar.read_text())["parquet_sha256"] == metadata["parquet_sha256"]

    reused_lookup, reused_metadata = evaluator.load_or_build_target_cache(
        "ptb_xl",
        MustNotIterate(),
        ImmediateExecutor(),
        leads,
        tmp_path,
        "test-content-root",
        "extractor-digest",
    )
    assert reused_metadata == metadata
    np.testing.assert_allclose(
        reused_lookup[("101", "V6", "R_Amp")]["value"], [3.0, 4.0]
    )


def _events(times, values):
    return {
        "sample_index": np.asarray(times, dtype=np.int64),
        "value": np.asarray(values, dtype=float),
    }


def test_distribution_rbf_mmd_is_zero_for_identical_samples():
    result = evaluator.distribution_rbf_mmd2([0, 1, 2, 3], [0, 1, 2, 3])
    assert result["mmd2"] == 0.0
    assert result["real_samples"] == result["recon_samples"] == 4
    assert result["bandwidth"] > 0


def test_distribution_rbf_mmd_detects_shift_and_caps_deterministically():
    real = np.linspace(-1, 1, 1000)
    shifted = real + 2
    first = evaluator.distribution_rbf_mmd2(real, shifted, sample_cap=32)
    second = evaluator.distribution_rbf_mmd2(real, shifted, sample_cap=32)
    assert first == second
    assert first["mmd2"] > 0
    assert first["real_samples"] == first["recon_samples"] == 32


def test_distribution_rbf_mmd_rejects_nonfinite_input():
    with np.testing.assert_raises_regex(ValueError, "finite"):
        evaluator.distribution_rbf_mmd2([0, np.nan], [0, 1])


def test_time_matcher_maximizes_cardinality_then_minimizes_time_error():
    # A naive first-in-order match would pair real time 0 to reconstruction 6.
    # The optimal one-pair solution uses real time 10 because it is closer.
    result = evaluator.match_events_by_time(
        _events([0, 10], [100.0, 200.0]),
        _events([6], [300.0]),
        tolerance_samples=10,
    )
    assert result["matched"] == 1
    assert result["unmatched_real"] == 1
    assert result["unmatched_recon"] == 0
    np.testing.assert_allclose(result["real_values"], [200.0])
    np.testing.assert_allclose(result["recon_values"], [300.0])
    np.testing.assert_array_equal(result["absolute_time_error_samples"], [4])


def test_time_matcher_accounts_for_missed_and_extra_events():
    result = evaluator.match_events_by_time(
        _events([100, 500, 900], [1.0, 2.0, 3.0]),
        _events([95, 505, 700, 1200], [10.0, 20.0, 30.0, 40.0]),
        tolerance_samples=50,
    )
    assert result["matched"] == 2
    assert result["unmatched_real"] == 1
    assert result["unmatched_recon"] == 2
    np.testing.assert_allclose(result["real_values"], [1.0, 2.0])
    np.testing.assert_allclose(result["recon_values"], [10.0, 20.0])
    np.testing.assert_array_equal(result["absolute_time_error_samples"], [5, 5])


def test_time_matcher_rejects_unsorted_events():
    with np.testing.assert_raises_regex(ValueError, "sorted"):
        evaluator.match_events_by_time(
            _events([10, 5], [1.0, 2.0]),
            _events([6], [3.0]),
            tolerance_samples=10,
        )


def test_qt_extraction_uses_each_qrs_onset_at_most_once():
    result = evaluator.extract_qt_events(
        qrs_onsets=[100, 500, 900],
        r_peaks=[150, 550, 950],
        t_offsets=[300, 350, 700, 1100],
        fs=500,
    )
    np.testing.assert_array_equal(result["sample_index"], [100, 500, 900])
    # The first T offset within each beat cycle is used; later offsets cannot
    # reuse the same QRS onset and create multi-second pseudo-QT intervals.
    np.testing.assert_allclose(result["value"], [400.0, 400.0, 400.0])


def test_qt_extraction_does_not_cross_next_qrs_onset():
    result = evaluator.extract_qt_events(
        qrs_onsets=[100, 500],
        r_peaks=[150, 550],
        t_offsets=[600],
        fs=500,
    )
    np.testing.assert_array_equal(result["sample_index"], [500])
    np.testing.assert_allclose(result["value"], [200.0])


def test_qt_extraction_requires_t_offset_after_same_cycle_r_peak():
    result = evaluator.extract_qt_events(
        qrs_onsets=[100, 500],
        r_peaks=[150, 550],
        t_offsets=[120, 300, 520, 700],
        fs=500,
    )
    np.testing.assert_array_equal(result["sample_index"], [100, 500])
    np.testing.assert_allclose(result["value"], [400.0, 400.0])
