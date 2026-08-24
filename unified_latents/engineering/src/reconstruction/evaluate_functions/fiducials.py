"""Beat-level fiducial extraction utilities for ECG Bland-Altman analysis.

These helpers operate on single-lead waveforms sampled at 500 Hz
(real and reconstructed) and return beat-level fiducial measurements.

This module is intended for offline analysis (not training-critical).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import neurokit2 as nk
import numpy as np

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
    """Detect R-peaks using the NeuroKit pipeline used in `ecg_fm_integration`."""
    try:
        _, rpeaks = nk.ecg_peaks(lead, sampling_rate=int(fs))
    except Exception:
        return np.array([], dtype=int)
    peaks = rpeaks.get("ECG_R_Peaks", [])
    return np.asarray(peaks, dtype=int)


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


def _safe_wave_index(waves: Dict[str, List[float]], key: str, index: int) -> Optional[int]:
    values = waves.get(key)
    if values is None or index >= len(values):
        return None
    value = values[index]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)


def delineate_beats(lead: np.ndarray, fs: float = FS_HZ) -> List[Dict[str, Optional[int]]]:
    """Return beat delineations from NeuroKit for a single lead."""
    try:
        _, rpeaks = nk.ecg_peaks(lead, sampling_rate=int(fs))
        _, waves = nk.ecg_delineate(
            lead,
            rpeaks,
            sampling_rate=int(fs),
            method="dwt",
            show=False,
            show_type="all",
        )
    except Exception:
        return []

    r_indices = np.asarray(rpeaks.get("ECG_R_Peaks", []), dtype=int)
    beats: List[Dict[str, Optional[int]]] = []
    for i, r_index in enumerate(r_indices):
        beats.append(
            {
                "r_index": int(r_index),
                "p_onset": _safe_wave_index(waves, "ECG_P_Onsets", i),
                "p_offset": _safe_wave_index(waves, "ECG_P_Offsets", i),
                "q_peak": _safe_wave_index(waves, "ECG_Q_Peaks", i),
                "r_onset": _safe_wave_index(waves, "ECG_R_Onsets", i),
                "r_offset": _safe_wave_index(waves, "ECG_R_Offsets", i),
                "s_peak": _safe_wave_index(waves, "ECG_S_Peaks", i),
                "t_onset": _safe_wave_index(waves, "ECG_T_Onsets", i),
                "t_offset": _safe_wave_index(waves, "ECG_T_Offsets", i),
            }
        )
    return beats


def _segment_from_bounds(lead: np.ndarray, start: Optional[int], end: Optional[int]) -> np.ndarray:
    if start is None or end is None:
        return np.array([])
    start = max(0, int(start))
    end = min(len(lead), int(end))
    if end <= start:
        return np.array([])
    return lead[start:end]


def _beat_window_from_delineation(lead: np.ndarray, beat: Dict[str, Optional[int]]) -> np.ndarray:
    start = beat.get("p_onset")
    end = beat.get("t_offset")
    return _segment_from_bounds(lead, start, end)


def extract_fiducials_for_beat(lead: np.ndarray, beat: Dict[str, Optional[int]], fs: float = FS_HZ) -> BeatFiducials:
    """Extract fiducials for a beat using NeuroKit delineation boundaries."""
    p_seg = _segment_from_bounds(lead, beat.get("p_onset"), beat.get("p_offset"))

    qrs_start = beat.get("r_onset")
    if qrs_start is None:
        qrs_start = beat.get("q_peak")
    qrs_end = beat.get("r_offset")
    if qrs_end is None:
        qrs_end = beat.get("s_peak")
    qrs_seg = _segment_from_bounds(lead, qrs_start, qrs_end)

    st_seg = _segment_from_bounds(lead, beat.get("r_offset"), beat.get("t_onset"))
    t_seg = _segment_from_bounds(lead, beat.get("t_onset"), beat.get("t_offset"))

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

    beats = delineate_beats(real, fs)
    records: List[Dict[str, float]] = []
    for beat in beats:
        real_f = extract_fiducials_for_beat(real, beat, fs)
        recon_f = extract_fiducials_for_beat(recon, beat, fs)
        rec: Dict[str, float] = {}
        for key, value in real_f.__dict__.items():
            rec[f"real_{key}"] = value
        for key, value in recon_f.__dict__.items():
            rec[f"recon_{key}"] = value
        records.append(rec)
    return records


def compute_reconstruction_morphology_metrics(
    gt_12lead: np.ndarray,
    pred_12lead: np.ndarray,
    lead_indices: List[int],
    fs: float = FS_HZ,
) -> Dict[str, float]:
    """Summarize beat-level morphology metrics over one or more reconstructed leads."""
    gt_np = gt_12lead.detach().float().cpu().numpy() if hasattr(gt_12lead, "detach") else gt_12lead
    pred_np = pred_12lead.detach().float().cpu().numpy() if hasattr(pred_12lead, "detach") else pred_12lead

    if not lead_indices:
        return {
            "samples_with_beats": 0.0,
            "mean_beat_rmse": float("nan"),
            "mean_beat_corr": float("nan"),
            "mean_r_amp_error": float("nan"),
            "mean_r_peak_timing_error_ms": float("nan"),
        }

    beat_rmse: List[float] = []
    beat_corr: List[float] = []
    r_amp_error: List[float] = []
    r_peak_timing_error_ms: List[float] = []
    samples_with_beats = 0

    for b in range(gt_np.shape[0]):
        for lead_idx in lead_indices:
            gt_lead = gt_np[b, lead_idx]
            pred_lead = pred_np[b, lead_idx]
            beats = delineate_beats(gt_lead, fs=fs)
            pred_r_peaks = detect_r_peaks(pred_lead, fs=fs)
            if not beats:
                continue
            samples_with_beats += 1
            for beat in beats:
                gt_window = _beat_window_from_delineation(gt_lead, beat)
                pred_window = _beat_window_from_delineation(pred_lead, beat)
                if gt_window.size == 0 or pred_window.size == 0 or gt_window.shape != pred_window.shape:
                    continue
                beat_rmse.append(float(np.sqrt(np.mean((gt_window - pred_window) ** 2))))
                if np.std(gt_window) > 1e-6 and np.std(pred_window) > 1e-6:
                    beat_corr.append(float(np.corrcoef(gt_window, pred_window)[0, 1]))

                r_index = beat.get("r_index")
                if r_index is None:
                    continue
                gt_amp = abs(float(gt_lead[r_index]))
                pred_amp = abs(float(pred_lead[r_index]))
                if gt_amp > 1e-6:
                    r_amp_error.append(abs(gt_amp - pred_amp) / gt_amp)
                if pred_r_peaks.size:
                    nearest_pred_peak = pred_r_peaks[np.argmin(np.abs(pred_r_peaks - r_index))]
                    r_peak_timing_error_ms.append(abs(int(nearest_pred_peak) - int(r_index)) * 1000.0 / fs)

    return {
        "samples_with_beats": float(samples_with_beats),
        "mean_beat_rmse": float(np.mean(beat_rmse)) if beat_rmse else float("nan"),
        "mean_beat_corr": float(np.mean(beat_corr)) if beat_corr else float("nan"),
        "mean_r_amp_error": float(np.mean(r_amp_error)) if r_amp_error else float("nan"),
        "mean_r_peak_timing_error_ms": float(np.mean(r_peak_timing_error_ms)) if r_peak_timing_error_ms else float("nan"),
    }
