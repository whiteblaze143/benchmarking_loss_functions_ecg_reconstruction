from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from repecg.common import load_config
from repecg.common.beats import valid_rr_pairs
from repecg.common.kernels import NystromMap, WhiteningTransform, biased_mmd2
from repecg.common.metrics import expected_calibration_error, macro_f1, select_f1_thresholds
from repecg.common.phase import phase_cells, phase_normalize
from repecg.common.perturbations import (
    add_noise_at_snr,
    jitter_rpeaks,
    resample_raw_waveform,
    retain_random_beats,
    scale_global_amplitude,
)
from repecg.common.preprocess import LeadScalerAccumulator, fit_lead_scaler
from repecg.common.ptbxl import PTBXLStore
from repecg.common.recurrence import recurrence_operator, upper_triangle


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/common.yaml"


def test_metadata_contract_and_patient_disjointness() -> None:
    store = PTBXLStore(load_config(CONFIG))
    manifest = store.cohort_manifest()
    assert manifest["records"] == 21_799
    assert manifest["patients"] == 18_869
    assert manifest["splits"]["development_train"]["records"] == 15_245
    assert manifest["splits"]["development_select"]["records"] == 2_173
    assert manifest["splits"]["final_select"]["records"] == 2_183
    assert manifest["splits"]["locked_test"]["records"] == 2_198


def test_fold10_signal_is_locked() -> None:
    store = PTBXLStore(load_config(CONFIG))
    ecg_id = int(store.split_frame("locked_test").index[0])
    with pytest.raises(PermissionError, match="fold 10"):
        store.read_record(ecg_id)


def test_label_encoding_and_real_record_shape() -> None:
    store = PTBXLStore(load_config(CONFIG))
    ecg_id = int(store.split_frame("development_train").index[0])
    record = store.read_record(ecg_id)
    assert record.signal_mv.shape == (5000, 12)
    assert record.labels.shape == (5,)
    assert set(np.unique(record.labels)).issubset({0.0, 1.0})


def test_rr_and_phase_shapes() -> None:
    fs = 500
    time = np.arange(5000) / fs
    basis = np.stack([np.sin(2 * np.pi * (1 + i / 10) * time) for i in range(8)], axis=1)
    peaks = np.arange(250, 4751, 500)
    intervals = valid_rr_pairs(peaks, fs, 300, 2000)
    beats = phase_normalize(basis, intervals)
    assert beats.shape == (len(intervals), 256, 8)
    assert phase_cells(beats, 16).shape == (len(intervals), 16, 16, 8)
    assert phase_cells(beats, 8).shape == (len(intervals), 8, 32, 8)


def test_kernel_and_recurrence_contracts() -> None:
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(512, 8))
    whitening = WhiteningTransform.fit(raw)
    white = whitening.transform(raw)
    mapping = NystromMap.fit(white, landmarks=32, seed=42)
    a, b = white[:64], white[64:128]
    exact = biased_mmd2(a, b)
    approximate = float(np.square(mapping.mean(a) - mapping.mean(b)).sum())
    assert exact >= -1e-12
    assert approximate >= 0
    means = np.stack([mapping.mean(white[i : i + 16]) for i in range(0, 256, 16)])
    operator = recurrence_operator(means, tau=1.0)
    assert operator.shape == (16, 16)
    assert np.allclose(operator, operator.T)
    assert np.allclose(np.diag(operator), 0)
    assert upper_triangle(operator).shape == (120,)


def test_streaming_scaler_matches_in_memory() -> None:
    rng = np.random.default_rng(18)
    signals = [rng.normal(size=(17, 8)), rng.normal(size=(23, 8)), rng.normal(size=(5, 8))]
    expected = fit_lead_scaler(signals)
    accumulator = LeadScalerAccumulator()
    for signal in signals:
        accumulator.update(signal)
    observed = accumulator.finalize()
    assert np.allclose(observed.mean, expected.mean)
    assert np.allclose(observed.std, expected.std)

    merged = LeadScalerAccumulator()
    for signal in signals:
        mean = signal.mean(axis=0)
        merged.merge(len(signal), mean, np.square(signal - mean).sum(axis=0))
    combined = merged.finalize()
    assert np.allclose(combined.mean, expected.mean)
    assert np.allclose(combined.std, expected.std)


def test_calibration_and_validation_f1_metrics() -> None:
    target = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=np.float64)
    perfect = np.asarray([[0.0, 1.0], [0.1, 0.9], [0.9, 0.1], [1.0, 0.0]])
    thresholds = select_f1_thresholds(target, perfect)
    assert thresholds.shape == (2,)
    assert macro_f1(target, perfect, thresholds) == 1.0
    assert expected_calibration_error(target, target) == 0.0


def test_raw_waveform_perturbation_contracts() -> None:
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(5000, 8))
    assert resample_raw_waveform(signal, 500, 250).shape == (2500, 8)
    assert resample_raw_waveform(signal, 500, 100).shape == (1000, 8)
    noisy = add_noise_at_snr(signal, 20.0, np.random.default_rng(1))
    observed_snr = 20.0 * np.log10(
        np.sqrt(np.mean(signal**2, axis=0)) / np.sqrt(np.mean((noisy - signal) ** 2, axis=0))
    )
    assert np.allclose(observed_snr, 20.0)
    assert np.allclose(scale_global_amplitude(signal, 1.1), signal * 1.1)
    peaks = np.arange(250, 4751, 500)
    jittered = jitter_rpeaks(peaks, max_ms=20, sampling_hz=500, signal_length=5000, rng=np.random.default_rng(2))
    assert np.max(np.abs(jittered - peaks)) <= 10
    beats = np.arange(10 * 3).reshape(10, 3)
    retained, indices = retain_random_beats(beats, removal_fraction=0.5, rng=np.random.default_rng(3))
    assert len(retained) == 5
    assert np.array_equal(retained, beats[indices])
