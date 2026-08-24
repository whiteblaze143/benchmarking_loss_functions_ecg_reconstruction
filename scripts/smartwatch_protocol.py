"""Protocol-safe helpers for the PhysioNet smartwatch simulator dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, correlate, find_peaks, resample_poly, sosfiltfilt

from scripts.evaluate_comprehensive_registry import LEAD_NAMES


def parse_protocol_target(relative_record: Path) -> dict[str, Any]:
    """Extract the calibrated simulator setting from a dataset-relative path."""
    parts = relative_record.parts
    stem = relative_record.name.casefold()
    if not parts:
        raise ValueError(f"Empty smartwatch record path: {relative_record}")
    experiment_dir = parts[0].casefold()
    if experiment_dir == "amp_test" and stem.startswith("amp"):
        value = int(stem.split("_", 1)[0][3:])
        return {
            "experiment_type": "r_wave_amplitude",
            "target_value": float(value),
            "target_unit": "uV",
        }
    if experiment_dir == "freq_test" and stem.startswith("f"):
        value = int(stem.split("_", 1)[0][1:])
        return {
            "experiment_type": "heart_rate",
            "target_value": float(value),
            "target_unit": "bpm",
        }
    if experiment_dir == "st-segment" and stem.startswith("st-"):
        setting = stem.split("_", 1)[0].removeprefix("st-")
        sign = -1 if setting.startswith("m") else 1
        magnitude = int(setting[1:]) * 100
        return {
            "experiment_type": "st_offset",
            "target_value": float(sign * magnitude),
            "target_unit": "uV",
        }
    if experiment_dir == "sqr-2hz":
        return {
            "experiment_type": "square_wave",
            "target_value": 2.0,
            "target_unit": "Hz",
        }
    raise ValueError(f"Unsupported smartwatch protocol record: {relative_record}")


def canonical_record_key(relative_record: Path) -> str:
    """Return the case-insensitive pairing key used across device directories."""
    return relative_record.as_posix().casefold()


def validate_single_watch_lead(signal_names: Sequence[str]) -> tuple[str, int]:
    if len(signal_names) != 1:
        raise ValueError(f"Expected one smartwatch signal, found {signal_names}")
    name = str(signal_names[0])
    if name not in LEAD_NAMES:
        raise ValueError(f"Unsupported smartwatch lead name: {name}")
    return name, LEAD_NAMES.index(name)


def resample_signals(
    signals: np.ndarray,
    source_fs: float,
    target_fs: int = 500,
) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Signals must be [time,lead], got {values.shape}")
    if source_fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    source_integer = int(round(float(source_fs)))
    if not np.isclose(source_fs, source_integer):
        raise ValueError(f"Non-integer source frequency is unsupported: {source_fs}")
    if source_integer != target_fs:
        values = resample_poly(values, target_fs, source_integer, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("WFDB record contains non-finite physical samples")
    return values.astype(np.float32)


def _alignment_trace(signal: np.ndarray, fs: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    sos = butter(3, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    if values.size > 3 * (2 * sos.shape[0] + 1):
        values = sosfiltfilt(sos, values)
    values = values - np.median(values)
    scale = np.std(values)
    return values / scale if scale > 1e-9 else values


def align_watch_to_reference(
    watch_lead: np.ndarray,
    reference_12lead: np.ndarray,
    *,
    reference_lead_index: int,
    fs: int = 500,
    target_len: int = 5000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align independent device captures of the same simulator waveform."""
    watch = np.asarray(watch_lead, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference_12lead, dtype=np.float32)
    if reference.ndim != 2 or reference.shape[1] != 12:
        raise ValueError(f"Reference must be [time,12], got {reference.shape}")
    if watch.size < target_len or reference.shape[0] < target_len:
        raise ValueError(
            f"Insufficient duration for 10-second alignment: "
            f"watch={watch.size}, reference={reference.shape[0]}"
        )
    reference_start = (reference.shape[0] - target_len) // 2
    reference_window = reference[reference_start:reference_start + target_len]
    query = _alignment_trace(reference_window[:, reference_lead_index], fs)
    search = _alignment_trace(watch, fs)
    scores = correlate(search, query, mode="valid", method="fft")
    coarse_start = int(np.argmax(scores))
    candidate_starts = range(
        max(0, coarse_start - 5),
        min(watch.size - target_len, coarse_start + 5) + 1,
    )
    local_correlations = {
        start: float(
            np.corrcoef(
                _alignment_trace(watch[start:start + target_len], fs),
                query,
            )[0, 1]
        )
        for start in candidate_starts
    }
    watch_start = max(local_correlations, key=local_correlations.get)
    watch_window = watch[watch_start:watch_start + target_len]
    aligned_corr = local_correlations[watch_start]
    return (
        watch_window,
        reference_window,
        {
            "watch_start_sample": watch_start,
            "reference_start_sample": reference_start,
            "alignment_pearson": aligned_corr,
            "alignment_filter_hz": [0.5, 40.0],
            "target_length": target_len,
            "sampling_rate_hz": fs,
        },
    )


def detect_r_peaks(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    try:
        _, info = nk.ecg_peaks(
            np.asarray(signal, dtype=np.float64),
            sampling_rate=fs,
            method="engzeemod2012",
            correct_artifacts=False,
        )
        peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
        if peaks.size >= 2:
            return peaks
    except (ValueError, IndexError, ZeroDivisionError):
        pass
    trace = _alignment_trace(signal, fs)
    peaks, _ = find_peaks(
        trace,
        distance=max(1, int(round(fs * 60.0 / 330.0))),
        prominence=0.45,
    )
    if peaks.size < 2:
        inverse, _ = find_peaks(
            -trace,
            distance=max(1, int(round(fs * 60.0 / 330.0))),
            prominence=0.45,
        )
        peaks = inverse
    return peaks.astype(np.int64)


def estimate_heart_rate_bpm(signal: np.ndarray, fs: int = 500) -> float | None:
    peaks = detect_r_peaks(signal, fs)
    if peaks.size < 2:
        return None
    rr = np.diff(peaks) / float(fs)
    rr = rr[(rr >= 60.0 / 330.0) & (rr <= 60.0 / 20.0)]
    return float(60.0 / np.median(rr)) if rr.size else None


def estimate_r_amplitude_uv(signal_mv: np.ndarray, fs: int = 500) -> float | None:
    signal = np.asarray(signal_mv, dtype=np.float64)
    peaks = detect_r_peaks(signal, fs)
    amplitudes: list[float] = []
    for peak in peaks:
        baseline_start = peak - int(round(0.20 * fs))
        baseline_stop = peak - int(round(0.08 * fs))
        if baseline_start < 0 or baseline_stop <= baseline_start:
            continue
        baseline = float(np.median(signal[baseline_start:baseline_stop]))
        amplitudes.append(abs(float(signal[peak]) - baseline) * 1000.0)
    return float(np.median(amplitudes)) if amplitudes else None


def estimate_st_offset_uv(signal_mv: np.ndarray, fs: int = 500) -> float | None:
    signal = np.asarray(signal_mv, dtype=np.float64)
    peaks = detect_r_peaks(signal, fs)
    offsets: list[float] = []
    for peak in peaks:
        baseline_start = peak - int(round(0.20 * fs))
        baseline_stop = peak - int(round(0.08 * fs))
        st_start = peak + int(round(0.12 * fs))
        st_stop = peak + int(round(0.24 * fs))
        if baseline_start < 0 or st_stop > signal.size:
            continue
        baseline = float(np.median(signal[baseline_start:baseline_stop]))
        st_level = float(np.median(signal[st_start:st_stop]))
        offsets.append((st_level - baseline) * 1000.0)
    return float(np.median(offsets)) if offsets else None


def measure_protocol_signal(
    signal_mv: np.ndarray,
    target: dict[str, Any],
    fs: int = 500,
) -> float | None:
    experiment_type = target["experiment_type"]
    if experiment_type == "heart_rate":
        return estimate_heart_rate_bpm(signal_mv, fs)
    if experiment_type == "r_wave_amplitude":
        return estimate_r_amplitude_uv(signal_mv, fs)
    if experiment_type == "st_offset":
        return estimate_st_offset_uv(signal_mv, fs)
    return None
