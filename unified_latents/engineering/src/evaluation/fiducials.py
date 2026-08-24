"""Beat-level fiducial extraction utilities for ECG Bland-Altman analysis.

These helpers operate on single-lead waveforms sampled at 500 Hz
(real and reconstructed) and return beat-level fiducial measurements.

This module is intended for offline analysis (not training-critical).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.signal import find_peaks

FS_HZ = 500.0


@dataclass
class BeatFiducials:
    """Container for fiducial measurements of a single beat."""

    p_amplitude: float
    qrs_amplitude: float
    st_elevation: float
    t_amplitude: float
    pr_interval_ms: float
    qrs_duration_ms: float


def detect_r_peaks(lead: np.ndarray, fs: float = FS_HZ) -> np.ndarray:
    """Detect R-peaks in a single-lead ECG using simple peak detection.

    This is a pragmatic baseline for research analysis, not a clinical
    detector. It assumes reasonably clean QRS complexes.
    """

    # Enforce 250 ms refractory period between peaks
    distance = int(0.25 * fs)
    prominence = float(np.std(lead) * 0.7)
    peaks, _ = find_peaks(lead, distance=distance, prominence=prominence)
    return peaks


def _window_segment(arr: np.ndarray, center: int, fs: float, start_ms: float, end_ms: float) -> np.ndarray:
    """Return a window around `center` between [start_ms, end_ms] relative to R-peak."""

    start = center + int(start_ms * 1e-3 * fs)
    end = center + int(end_ms * 1e-3 * fs)
    start = max(0, start)
    end = min(len(arr), end)
    return arr[start:end]


def _amplitude(seg: np.ndarray) -> float:
    if seg.size == 0:
        return float("nan")
    return float(seg.max() - seg.min())


def _duration_ms(seg: np.ndarray, fs: float) -> float:
    """Crude duration estimate based on where derivative exceeds a threshold."""

    if seg.size < 2:
        return float("nan")
    deriv = np.abs(np.diff(seg))
    thr = float(0.02 * max(1e-6, np.std(seg)))
    idx = np.where(deriv > thr)[0]
    if idx.size == 0:
        return float("nan")
    dur_samples = idx[-1] - idx[0]
    return float(1000.0 * dur_samples / fs)


def extract_fiducials_for_beat(lead: np.ndarray, r_index: int, fs: float = FS_HZ) -> BeatFiducials:
    """Extract simple fiducial amplitudes and durations for a single beat.

    Windows are defined relative to the R-peak.
    """

    p_seg = _window_segment(lead, r_index, fs, -200.0, -50.0)
    qrs_seg = _window_segment(lead, r_index, fs, -50.0, 50.0)
    st_seg = _window_segment(lead, r_index, fs, 50.0, 100.0)
    t_seg = _window_segment(lead, r_index, fs, 100.0, 300.0)

    p_amp = _amplitude(p_seg)
    qrs_amp = _amplitude(qrs_seg)
    t_amp = _amplitude(t_seg)
    st_elev = float(st_seg.mean()) if st_seg.size else float("nan")

    if p_seg.size and qrs_seg.size:
        pq = np.concatenate([p_seg, qrs_seg])
        pr_int = _duration_ms(pq, fs)
    else:
        pr_int = float("nan")
    qrs_dur = _duration_ms(qrs_seg, fs)

    return BeatFiducials(
        p_amplitude=p_amp,
        qrs_amplitude=qrs_amp,
        st_elevation=st_elev,
        t_amplitude=t_amp,
        pr_interval_ms=pr_int,
        qrs_duration_ms=qrs_dur,
    )


def collect_beat_level_fiducials(real: np.ndarray, recon: np.ndarray, fs: float = FS_HZ) -> List[Dict[str, float]]:
    """Collect beat-level fiducials for real and reconstructed signals.

    Parameters
    ----------
    real, recon
        1D numpy arrays for a single ECG lead (same length).

    Returns
    -------
    List of dicts, each containing real_*/recon_* fiducials for a beat.
    """

    assert real.shape == recon.shape, "real and recon must have same shape"

    r_peaks = detect_r_peaks(real, fs)
    records: List[Dict[str, float]] = []
    for r_idx in r_peaks:
        real_f = extract_fiducials_for_beat(real, r_idx, fs)
        recon_f = extract_fiducials_for_beat(recon, r_idx, fs)
        rec: Dict[str, float] = {}
        for key, value in real_f.__dict__.items():
            rec[f"real_{key}"] = value
        for key, value in recon_f.__dict__.items():
            rec[f"recon_{key}"] = value
        records.append(rec)
    return records
