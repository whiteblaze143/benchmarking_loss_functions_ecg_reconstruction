import numpy as np

from scripts.materialize_ptbxl_beat_bounds import beat_bounds, detect_r_peaks
from scripts.audit_ptbxl_beat_bounds import matched_f1


def _synthetic_ecg(invert=False):
    fs, length = 500, 5000
    time = np.arange(length) / fs
    signal = 0.02 * np.sin(2 * np.pi * 0.3 * time)
    for peak in range(300, length - 100, 500):
        signal += (-1 if invert else 1) * np.exp(-0.5 * ((np.arange(length) - peak) / 7) ** 2)
    return signal


def test_detector_is_polarity_invariant_and_bounds_are_valid():
    positive = detect_r_peaks(_synthetic_ecg())
    negative = detect_r_peaks(_synthetic_ecg(invert=True))
    np.testing.assert_allclose(positive, negative, atol=2)
    bounds = beat_bounds(positive, 5000)
    assert bounds[0][0] == 0 and bounds[-1][1] == 5000
    assert all(0 <= start < end <= 5000 for start, end in bounds)


def test_detector_rejects_flat_signal():
    try:
        detect_r_peaks(np.zeros(5000))
    except ValueError as exc:
        assert "fewer than two" in str(exc)
    else:
        raise AssertionError("flat signal must fail")


def test_peak_matching_is_one_to_one():
    f1, errors = matched_f1(np.array([100, 110]), np.array([105]), tolerance=10)
    assert f1 == 2 / 3
    assert len(errors) == 1
