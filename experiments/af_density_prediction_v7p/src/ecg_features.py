"""Extraction of conventional ECG physiology and ventricular dynamics (F4).

Zero synthetic fallbacks: when beats are insufficient or absent, physiological metrics
are strictly np.nan with is_rr_valid=0.0. Never fabricate physiological numbers.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import wfdb

LTAFDB_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")


def compute_rr_stats(rr_s: np.ndarray) -> Dict[str, float]:
    """Computes physiological ventricular rate and variability features.

    If fewer than 2 valid beats are available, metrics are strictly np.nan.
    Never fabricates default numbers like 75 bpm or 30 ms.
    """
    if len(rr_s) < 2:
        return {
            "mean_hr_bpm": np.nan,
            "std_hr_bpm": np.nan,
            "rmssd_ms": np.nan,
            "cvrr": np.nan,
            "pnn50": np.nan,
            "num_beats": float(len(rr_s)),
            "is_rr_valid": 0.0,
        }

    # Plausible physiological RR range [0.25s, 2.5s] (24 to 240 bpm)
    valid_rr = rr_s[(rr_s >= 0.25) & (rr_s <= 2.5)]
    if len(valid_rr) < 2:
        return {
            "mean_hr_bpm": np.nan,
            "std_hr_bpm": np.nan,
            "rmssd_ms": np.nan,
            "cvrr": np.nan,
            "pnn50": np.nan,
            "num_beats": float(len(valid_rr)),
            "is_rr_valid": 0.0,
        }

    hr = 60.0 / valid_rr
    diff_ms = np.abs(np.diff(valid_rr)) * 1000.0
    rmssd = float(np.sqrt(np.mean(diff_ms ** 2)))
    pnn50 = float(np.mean(diff_ms > 50.0))
    cvrr = float(np.std(valid_rr) / np.mean(valid_rr))

    return {
        "mean_hr_bpm": float(np.mean(hr)),
        "std_hr_bpm": float(np.std(hr)),
        "rmssd_ms": rmssd,
        "cvrr": cvrr,
        "pnn50": pnn50,
        "num_beats": float(len(valid_rr)),
        "is_rr_valid": 1.0,
    }


class RRFeatureExtractor:
    """Caches record-level beat annotations and extracts window-level RR statistics."""

    def __init__(self):
        self._cache = {}

    def get_beats_in_range(self, dataset: str, record_id: str, start_s: float, end_s: float) -> np.ndarray:
        key = (dataset, record_id)
        if key not in self._cache:
            self._load_record(dataset, record_id)
        beat_times = self._cache.get(key, np.array([]))
        if len(beat_times) == 0:
            return np.array([])
        return beat_times[(beat_times >= start_s) & (beat_times <= end_s)]

    def _load_record(self, dataset: str, record_id: str):
        try:
            if dataset == "LTAFDB":
                rec_base = str(LTAFDB_DIR / record_id)
                ann = wfdb.rdann(rec_base, "atr")
                # fs = 128
                self._cache[(dataset, record_id)] = ann.sample / 128.0
            elif dataset == "SHDB-AF":
                rec_base = str(SHDB_DIR / record_id)
                ann = wfdb.rdann(rec_base, "qrs")
                # fs = 200
                self._cache[(dataset, record_id)] = ann.sample / 200.0
            else:
                self._cache[(dataset, record_id)] = np.array([])
        except Exception:
            self._cache[(dataset, record_id)] = np.array([])

    def extract_window_rr_features(self, dataset: str, record_id: str, t: float, window_s: float = 300.0) -> Dict[str, float]:
        beats = self.get_beats_in_range(dataset, record_id, max(0.0, t - window_s), t)
        if len(beats) < 2:
            return compute_rr_stats(np.array([]))
        rr = np.diff(beats)
        return compute_rr_stats(rr)
