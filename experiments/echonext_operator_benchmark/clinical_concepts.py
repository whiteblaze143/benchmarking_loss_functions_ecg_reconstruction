#!/usr/bin/env python3
"""Module C: Observable Clinical Concept Layer & Algorithmic Grounding.

Defines the six core clinical ECG concepts, their physiological units, normal ranges,
spatial lead projection vectors v_j in R^8 (canonical leads [I, II, V1, V2, V3, V4, V5, V6]),
and algorithmic reference extractors from full-lead recordings.

Clinical Concept Definitions:
1. C_HR (Heart Rate, bpm):
   Ventricular rhythm frequency: 60 / RR_mean. Normal: 60 - 100 bpm.
   Support: Temporal, observable from any lead with SNR > 0.
2. C_QRS (QRS Duration, ms):
   Ventricular depolarization interval: Q_onset to J_point. Normal: 70 - 110 ms; >120 ms prolonged.
   Support: Spanned across precordial and limb leads; primary marker of ventricular dyssynchrony.
3. C_QTc (Corrected QT Interval, ms):
   Ventricular repolarization interval via Fridericia formula: QT / (RR^(1/3)). Normal: 360 - 450 ms.
   Support: Global repolarization timing.
4. C_AXIS (Frontal QRS Electrical Axis, degrees):
   Net ventricular depolarization vector in frontal plane: arctan2(2/sqrt(3)*II - 1/sqrt(3)*I, I).
   Normal: -30 to +90 degrees.
   Support: STRICTLY limb lead subspace (Leads I, II). Unobservable from precordial leads alone.
5. C_ST_ANT (Anterior ST Segment Elevation/Depression, uV):
   J-point displacement at J+60ms relative to PR baseline in V2/V3. Normal: -50 to +150 uV.
   Support: STRICTLY anterior precordial subspace (Leads V2, V3). Unobservable from limb leads alone.
6. C_R_PROG (Precordial R-Wave Progression, ratio):
   R-wave amplitude ratio: max(R_V5, R_V6) / (max(R_V1, R_V2) + eps). Normal: 1.5 - 6.0.
   Support: STRICTLY precordial subspace (V1-V6). Unobservable from limb leads alone.

Observability Criterion:
    A concept with spatial lead vector v_j in R^8 is:
    - OBSERVABLE:           ||(I - P_Q) v_j||^2 / ||v_j||^2 < 0.05
    - PARTIALLY_OBSERVABLE: 0.05 <= ||(I - P_Q) v_j||^2 / ||v_j||^2 < 0.90
    - UNOBSERVABLE:         ||(I - P_Q) v_j||^2 / ||v_j||^2 >= 0.90
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Literal
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks


# Canonical 8-lead ordering: [I, II, V1, V2, V3, V4, V5, V6]
CANONICAL_LEAD_NAMES = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass(frozen=True)
def ConceptMetadata:
    name: str
    symbol: str
    unit: str
    normal_low: float
    normal_high: float
    support_vector: np.ndarray  # Vector v in R^8 defining spatial lead dependence
    is_temporal: bool           # True if primarily timing-based rather than lead-specific amplitude


# Spatial support vectors in R^8
# Limb leads: indices 0 (I) and 1 (II)
v_limb = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
v_limb /= np.linalg.norm(v_limb)

# Anterior precordial: indices 3 (V2) and 4 (V3)
v_anterior = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
v_anterior /= np.linalg.norm(v_anterior)

# Full precordial: indices 2..7 (V1..V6)
v_precordial = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
v_precordial /= np.linalg.norm(v_precordial)

# Global isotropic support: all 8 leads
v_global = np.ones(8, dtype=np.float32) / np.sqrt(8.0)


CLINICAL_CONCEPTS: Dict[str, ConceptMetadata] = {
    "heart_rate_bpm": ConceptMetadata(
        name="Ventricular Heart Rate",
        symbol="C_HR",
        unit="bpm",
        normal_low=60.0,
        normal_high=100.0,
        support_vector=v_global,
        is_temporal=True,
    ),
    "qrs_duration_ms": ConceptMetadata(
        name="QRS Duration",
        symbol="C_QRS",
        unit="ms",
        normal_low=70.0,
        normal_high=110.0,
        support_vector=v_global,
        is_temporal=True,
    ),
    "qtc_interval_ms": ConceptMetadata(
        name="Corrected QT Interval",
        symbol="C_QTc",
        unit="ms",
        normal_low=360.0,
        normal_high=450.0,
        support_vector=v_global,
        is_temporal=True,
    ),
    "frontal_qrs_axis_deg": ConceptMetadata(
        name="Frontal QRS Electrical Axis",
        symbol="C_AXIS",
        unit="deg",
        normal_low=-30.0,
        normal_high=90.0,
        support_vector=v_limb,
        is_temporal=False,
    ),
    "st_elevation_v2_v3_uv": ConceptMetadata(
        name="Anterior ST Elevation/Depression",
        symbol="C_ST_ANT",
        unit="uV",
        normal_low=-50.0,
        normal_high=150.0,
        support_vector=v_anterior,
        is_temporal=False,
    ),
    "precordial_r_progression": ConceptMetadata(
        name="Precordial R-Wave Progression",
        symbol="C_R_PROG",
        unit="ratio",
        normal_low=1.5,
        normal_high=6.0,
        support_vector=v_precordial,
        is_temporal=False,
    ),
}

CONCEPT_KEYS = list(CLINICAL_CONCEPTS.keys())
NUM_CONCEPTS = len(CONCEPT_KEYS)


def extract_reference_ecg_concepts(
    signal_8lead: np.ndarray,
    sampling_rate: float = 250.0,
) -> Dict[str, float]:
    """Extract algorithmic reference clinical concepts from a full 8-lead ECG (8, T).

    Args:
        signal_8lead: Array of shape (8, T) or (T, 8) in microvolts/standard units.
        sampling_rate: Sampling frequency in Hz (EchoNext / PTB-XL typically 250Hz).

    Returns:
        Dict mapping concept_key -> measured reference value.
    """
    if signal_8lead.shape[0] != 8 and signal_8lead.shape[1] == 8:
        signal_8lead = signal_8lead.T  # Ensure (8, T)
    
    d, T = signal_8lead.shape
    dt_ms = 1000.0 / sampling_rate
    
    # 1. R-peak detection using Lead II or Lead V5
    ref_lead = signal_8lead[1]  # Lead II
    if np.std(ref_lead) < 1e-4:
        ref_lead = signal_8lead[6]  # Fallback to V5
        
    diff_lead = np.diff(ref_lead) ** 2
    height_thresh = np.mean(diff_lead) + 2.0 * np.std(diff_lead)
    min_dist = int(0.35 * sampling_rate)  # min 350 ms between beats (~170 bpm max)
    
    r_peaks, _ = find_peaks(diff_lead, height=height_thresh, distance=min_dist)
    if len(r_peaks) < 2:
        # Fallback peak finder on raw amplitude
        r_peaks, _ = find_peaks(ref_lead, height=np.mean(ref_lead) + 1.5 * np.std(ref_lead), distance=min_dist)

    # Concept 1: Heart Rate (bpm)
    if len(r_peaks) >= 2:
        rr_intervals = np.diff(r_peaks) / sampling_rate  # in seconds
        rr_mean = float(np.median(rr_intervals))
        hr_bpm = float(np.clip(60.0 / max(rr_mean, 0.25), 30.0, 220.0))
    else:
        rr_mean = 0.8
        hr_bpm = 75.0

    # Extract median beat around R peaks
    half_window = int(0.35 * sampling_rate)  # +/- 350ms
    valid_beats = []
    for r in r_peaks:
        if r - half_window >= 0 and r + half_window < T:
            valid_beats.append(signal_8lead[:, r - half_window : r + half_window])
    
    if len(valid_beats) > 0:
        median_beat = np.median(np.stack(valid_beats, axis=0), axis=0)  # (8, 2*half_window)
    else:
        median_beat = signal_8lead[:, : min(T, 2 * half_window)]

    r_idx = half_window if len(valid_beats) > 0 else int(median_beat.shape[1] / 2)

    # Concept 2: QRS Duration (ms)
    # Estimate energy around R peak across leads
    beat_energy = np.sum(median_beat ** 2, axis=0)
    baseline_energy = np.percentile(beat_energy, 20)
    energy_thresh = baseline_energy + 0.15 * (np.max(beat_energy) - baseline_energy)
    
    # Search backwards for Q onset
    q_onset = r_idx
    for i in range(r_idx, max(0, r_idx - int(0.12 * sampling_rate)), -1):
        if beat_energy[i] <= energy_thresh:
            q_onset = i
            break
            
    # Search forward for S/J offset
    j_offset = r_idx
    for i in range(r_idx, min(median_beat.shape[1], r_idx + int(0.15 * sampling_rate))):
        if beat_energy[i] <= energy_thresh:
            j_offset = i
            break
            
    qrs_duration_ms = float(np.clip((j_offset - q_onset) * dt_ms, 50.0, 220.0))

    # Concept 3: QTc Interval (ms) via Fridericia formula
    # Search for T-wave peak and offset after J point
    t_search_start = j_offset + int(0.06 * sampling_rate)
    t_search_end = min(median_beat.shape[1], j_offset + int(0.40 * sampling_rate))
    if t_search_end > t_search_start:
        lead_ii_segment = median_beat[1, t_search_start:t_search_end]
        t_peak_rel = np.argmax(np.abs(lead_ii_segment))
        t_peak_idx = t_search_start + t_peak_rel
        t_offset = min(median_beat.shape[1] - 1, t_peak_idx + int(0.08 * sampling_rate))
        qt_interval_ms = (t_offset - q_onset) * dt_ms
        qtc_interval_ms = float(np.clip(qt_interval_ms / (max(rr_mean, 0.3) ** (1.0 / 3.0)), 280.0, 620.0))
    else:
        qtc_interval_ms = 410.0

    # Concept 4: Frontal QRS Axis (degrees)
    # Using Lead I (index 0) and Lead II (index 1) peak amplitudes
    amp_I = float(median_beat[0, r_idx] - np.mean(median_beat[0, :max(1, q_onset)]))
    amp_II = float(median_beat[1, r_idx] - np.mean(median_beat[1, :max(1, q_onset)]))
    # Hexaxial projection: Axis = arctan2(2/sqrt(3)*II - 1/sqrt(3)*I, I)
    y_axis = (2.0 / np.sqrt(3.0)) * amp_II - (1.0 / np.sqrt(3.0)) * amp_I
    x_axis = amp_I
    axis_rad = np.arctan2(y_axis, x_axis)
    frontal_axis_deg = float(np.degrees(axis_rad))

    # Concept 5: Anterior ST Elevation (V2, V3) (uV)
    # ST level at J + 60ms relative to PR baseline (q_onset - 40ms to q_onset)
    pr_start = max(0, q_onset - int(0.04 * sampling_rate))
    pr_baseline_v2 = float(np.mean(median_beat[3, pr_start:q_onset]))
    pr_baseline_v3 = float(np.mean(median_beat[4, pr_start:q_onset]))
    
    st_sample = min(median_beat.shape[1] - 1, j_offset + int(0.06 * sampling_rate))
    st_v2 = float(median_beat[3, st_sample] - pr_baseline_v2)
    st_v3 = float(median_beat[4, st_sample] - pr_baseline_v3)
    st_elevation_v2_v3_uv = 0.5 * (st_v2 + st_v3)

    # Concept 6: Precordial R-Wave Progression
    # max(R_V5, R_V6) / (max(R_V1, R_V2) + eps)
    r_v1 = float(np.max(median_beat[2, q_onset:j_offset + 1]))
    r_v2 = float(np.max(median_beat[3, q_onset:j_offset + 1]))
    r_v5 = float(np.max(median_beat[6, q_onset:j_offset + 1]))
    r_v6 = float(np.max(median_beat[7, q_onset:j_offset + 1]))
    
    r_early = max(1e-3, max(r_v1, r_v2))
    r_late = max(1e-3, max(r_v5, r_v6))
    precordial_r_progression = float(np.clip(r_late / r_early, 0.1, 15.0))

    return {
        "heart_rate_bpm": hr_bpm,
        "qrs_duration_ms": qrs_duration_ms,
        "qtc_interval_ms": qtc_interval_ms,
        "frontal_qrs_axis_deg": frontal_axis_deg,
        "st_elevation_v2_v3_uv": st_elevation_v2_v3_uv,
        "precordial_r_progression": precordial_r_progression,
    }


def audit_concepts_observability(
    P_Q: Union[np.ndarray, torch.Tensor],
) -> Dict[str, Dict[str, Any]]:
    """Audit the observability of all 6 clinical concepts under measurement projector P_Q.

    Args:
        P_Q: Orthogonal projection matrix in R^(8, 8).

    Returns:
        Dict mapping concept_key -> {status, observable_fraction, nullspace_residue}.
    """
    is_torch = isinstance(P_Q, torch.Tensor)
    if is_torch:
        P_Q_np = P_Q.detach().cpu().numpy()
    else:
        P_Q_np = np.asarray(P_Q)

    # Symmetrize
    P_Q_np = 0.5 * (P_Q_np + P_Q_np.T)
    
    report = {}
    for key, meta in CLINICAL_CONCEPTS.items():
        v = meta.support_vector
        v_norm_sq = float(np.sum(v ** 2))
        P_v = P_Q_np @ v
        obs_norm_sq = float(np.sum(P_v ** 2))
        eta = max(0.0, min(1.0, obs_norm_sq / max(v_norm_sq, 1e-12)))
        rho = 1.0 - eta
        
        # Strict observability thresholds
        if rho < 0.05:
            status = "OBSERVABLE"
        elif rho < 0.85:
            status = "PARTIALLY_OBSERVABLE"
        else:
            status = "UNOBSERVABLE"
            
        report[key] = {
            "name": meta.name,
            "symbol": meta.symbol,
            "status": status,
            "observable_fraction": eta,
            "nullspace_residue": rho,
            "is_temporal": meta.is_temporal,
        }
        
    return report
