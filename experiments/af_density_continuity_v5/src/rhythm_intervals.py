"""Rhythm intervals, harmonization standards, and run-censoring arithmetic for AF-DENSITY V5.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Canonical harmonized rhythm states
RHYTHM_AF = "AF"
RHYTHM_CONFIRMED_SINUS = "CONFIRMED_SINUS"
RHYTHM_ATRIAL_FLUTTER = "ATRIAL_FLUTTER"
RHYTHM_OTHER_ATRIAL_ARRHYTHMIA = "OTHER_ATRIAL_ARRHYTHMIA"
RHYTHM_OTHER_RHYTHM = "OTHER_RHYTHM"
RHYTHM_UNKNOWN = "UNKNOWN"

VALID_HARMONIZED_STATES = {
    RHYTHM_AF,
    RHYTHM_CONFIRMED_SINUS,
    RHYTHM_ATRIAL_FLUTTER,
    RHYTHM_OTHER_ATRIAL_ARRHYTHMIA,
    RHYTHM_OTHER_RHYTHM,
    RHYTHM_UNKNOWN
}

def harmonize_rhythm_label(native_label: str) -> str:
    """Harmonizes raw annotation label into canonical 6-state system."""
    clean = str(native_label).strip().replace('\x00', '')
    if clean.startswith('('):
        clean = clean[1:]
    clean_upper = clean.upper()

    if clean_upper in {"AF", "AFIB", "A_FIB"}:
        return RHYTHM_AF
    elif clean_upper in {"N", "SINUS", "CONFIRMED_SINUS", "NORMAL"}:
        return RHYTHM_CONFIRMED_SINUS
    elif clean_upper in {"AFL", "FLUTTER", "ATRIAL_FLUTTER"}:
        return RHYTHM_ATRIAL_FLUTTER
    elif clean_upper in {"AT", "SVTA", "SVT", "OTHER_ATRIAL_ARRHYTHMIA", "ATRIAL_TACHYCARDIA"}:
        return RHYTHM_OTHER_ATRIAL_ARRHYTHMIA
    elif clean_upper in {"CALIBRATION_PREFIX", "INITIAL_UNKNOWN", "UNKNOWN", "MISSING", "NOISE"}:
        return RHYTHM_UNKNOWN
    elif clean_upper in {"NON_AF", "PROBABLE_SINUS"}:
        # In IRIDIA, non-AF is probable sinus, not confirmed sinus
        return RHYTHM_UNKNOWN
    else:
        # Ventricular rhythms, junctional, heart block, etc.
        return RHYTHM_OTHER_RHYTHM


@dataclass
class RhythmInterval:
    dataset: str
    patient_id: str
    record_id: str
    source_file_id: str
    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    sampling_frequency: float
    rhythm_label_native: str
    rhythm_label_harmonized: str
    annotation_source: str
    is_left_censored: bool
    is_right_censored: bool
    signal_available: bool
    rhythm_known: bool

    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = self.duration_s()
        return d


@dataclass
class AFRun:
    dataset: str
    patient_id: str
    record_id: str
    run_id: str
    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    is_left_censored: bool
    is_right_censored: bool
    has_observed_onset: bool
    has_observed_termination: bool
    is_complete: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def audit_run_censoring(runs: List[AFRun]) -> Dict[str, int]:
    """Verifies inclusion-exclusion arithmetic:
    complete_runs = total_runs - left_censored - right_censored + both_censored.
    """
    total = len(runs)
    left_cens = sum(1 for r in runs if r.is_left_censored)
    right_cens = sum(1 for r in runs if r.is_right_censored)
    both_cens = sum(1 for r in runs if (r.is_left_censored and r.is_right_censored))
    observed_onsets = sum(1 for r in runs if r.has_observed_onset)
    observed_terminations = sum(1 for r in runs if r.has_observed_termination)
    complete = sum(1 for r in runs if r.is_complete)

    expected_complete = total - left_cens - right_cens + both_cens
    assert complete == expected_complete, (
        f"Arithmetic mismatch: complete={complete} vs expected={expected_complete}"
    )

    return {
        "total_runs": total,
        "left_censored_runs": left_cens,
        "right_censored_runs": right_cens,
        "both_censored_runs": both_cens,
        "observed_onsets": observed_onsets,
        "observed_terminations": observed_terminations,
        "complete_runs": complete
    }
