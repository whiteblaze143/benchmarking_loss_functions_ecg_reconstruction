from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from wfdb.processing import xqrs_detect

from .ptbxl import INDEPENDENT_LEADS


@dataclass(frozen=True)
class BeatDetection:
    lead: str
    rpeaks: np.ndarray
    valid_intervals: np.ndarray


def valid_rr_pairs(rpeaks: np.ndarray, fs: int, min_ms: int, max_ms: int) -> np.ndarray:
    peaks = np.asarray(rpeaks, dtype=np.int64)
    if len(peaks) < 2:
        return np.empty((0, 2), dtype=np.int64)
    delta_ms = np.diff(peaks) * 1000.0 / fs
    keep = (delta_ms >= min_ms) & (delta_ms <= max_ms)
    return np.column_stack((peaks[:-1][keep], peaks[1:][keep]))


def detect_rpeaks(
    basis_mv: np.ndarray,
    *,
    fs: int = 500,
    primary_lead: str = "II",
    fallback_leads: tuple[str, ...] = ("I", "V2", "V5"),
    min_rr_ms: int = 300,
    max_rr_ms: int = 2000,
) -> BeatDetection:
    if basis_mv.ndim != 2 or basis_mv.shape[1] != len(INDEPENDENT_LEADS):
        raise ValueError("basis signal must have shape (time,8)")
    candidates = (primary_lead,) + tuple(x for x in fallback_leads if x != primary_lead)
    results: list[BeatDetection] = []
    for lead in candidates:
        idx = INDEPENDENT_LEADS.index(lead)
        try:
            peaks = np.asarray(xqrs_detect(basis_mv[:, idx], fs=fs, verbose=False), dtype=np.int64)
        except Exception:
            peaks = np.empty(0, dtype=np.int64)
        pairs = valid_rr_pairs(peaks, fs, min_rr_ms, max_rr_ms)
        results.append(BeatDetection(lead=lead, rpeaks=peaks, valid_intervals=pairs))
        if lead == primary_lead and len(pairs) >= 2:
            return results[-1]
    return max(results, key=lambda item: (len(item.valid_intervals), -candidates.index(item.lead)))
