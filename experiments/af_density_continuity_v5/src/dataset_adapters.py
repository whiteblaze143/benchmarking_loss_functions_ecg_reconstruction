"""Dataset adapters for LTAFDB, SHDB-AF, and IRIDIA-AF.
Parses native rhythm annotations, beat annotations, and metadata into canonical intervals and AF runs.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import wfdb

from .rhythm_intervals import (
    RhythmInterval,
    AFRun,
    harmonize_rhythm_label,
    RHYTHM_AF,
    RHYTHM_CONFIRMED_SINUS,
    RHYTHM_UNKNOWN
)


def load_ltafdb_record(
    record_id: str,
    base_dir: str = "/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference"
) -> Tuple[List[RhythmInterval], List[AFRun], Dict[str, Any]]:
    """Loads a single LTAFDB record and produces canonical intervals and AF runs."""
    rec_path = os.path.join(base_dir, record_id)
    header = wfdb.rdheader(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')

    fs = float(header.fs)
    n_samples = header.sig_len
    total_duration_s = n_samples / fs
    patient_id = f"ltafdb_{record_id}"

    # Extract rhythm changes from aux_note
    rhythm_changes = [] # (sample, rhythm_str)
    for s, aux in zip(ann.sample, ann.aux_note):
        aux_clean = aux.strip().replace('\x00', '')
        if aux_clean.startswith('('):
            rhythm_changes.append((int(s), aux_clean[1:]))

    # If no rhythm changes, default to UNKNOWN or whole record
    intervals: List[RhythmInterval] = []
    af_runs: List[AFRun] = []

    if not rhythm_changes:
        # No rhythm annotations
        intervals.append(RhythmInterval(
            dataset="LTAFDB",
            patient_id=patient_id,
            record_id=record_id,
            source_file_id=f"{record_id}.dat",
            start_sample=0,
            end_sample=n_samples,
            start_time_s=0.0,
            end_time_s=total_duration_s,
            sampling_frequency=fs,
            rhythm_label_native="UNKNOWN",
            rhythm_label_harmonized=RHYTHM_UNKNOWN,
            annotation_source="WFDB_ATR",
            is_left_censored=False,
            is_right_censored=False,
            signal_available=True,
            rhythm_known=False
        ))
        return intervals, af_runs, {"record_id": record_id, "duration_s": total_duration_s, "af_runs": 0}

    # If the first rhythm change is at s > 0, handle [0, s)
    first_s, first_rhythm = rhythm_changes[0]
    if first_s > 0:
        # In LTAFDB, the rhythm note applies from the start or [0, first_s) is initial prefix
        # Convention: if first_s is small (< 5 seconds), the rhythm starts at 0; else UNKNOWN
        if first_s <= int(fs * 5.0):
            rhythm_changes[0] = (0, first_rhythm)
        else:
            intervals.append(RhythmInterval(
                dataset="LTAFDB",
                patient_id=patient_id,
                record_id=record_id,
                source_file_id=f"{record_id}.dat",
                start_sample=0,
                end_sample=first_s,
                start_time_s=0.0,
                end_time_s=first_s / fs,
                sampling_frequency=fs,
                rhythm_label_native="INITIAL_UNANNOTATED",
                rhythm_label_harmonized=RHYTHM_UNKNOWN,
                annotation_source="WFDB_ATR",
                is_left_censored=False,
                is_right_censored=False,
                signal_available=True,
                rhythm_known=False
            ))

    # Build sequential intervals
    run_idx = 0
    for idx, (curr_s, curr_rhythm) in enumerate(rhythm_changes):
        next_s = rhythm_changes[idx + 1][0] if idx + 1 < len(rhythm_changes) else n_samples
        if next_s <= curr_s:
            continue

        harmonized = harmonize_rhythm_label(curr_rhythm)
        is_left_c = (curr_s == 0)
        is_right_c = (next_s == n_samples)

        intervals.append(RhythmInterval(
            dataset="LTAFDB",
            patient_id=patient_id,
            record_id=record_id,
            source_file_id=f"{record_id}.dat",
            start_sample=curr_s,
            end_sample=next_s,
            start_time_s=curr_s / fs,
            end_time_s=next_s / fs,
            sampling_frequency=fs,
            rhythm_label_native=curr_rhythm,
            rhythm_label_harmonized=harmonized,
            annotation_source="WFDB_ATR",
            is_left_censored=is_left_c,
            is_right_censored=is_right_c,
            signal_available=True,
            rhythm_known=(harmonized != RHYTHM_UNKNOWN)
        ))

        if harmonized == RHYTHM_AF:
            run_id = f"ltafdb_{record_id}_af_{run_idx:04d}"
            af_runs.append(AFRun(
                dataset="LTAFDB",
                patient_id=patient_id,
                record_id=record_id,
                run_id=run_id,
                start_sample=curr_s,
                end_sample=next_s,
                start_time_s=curr_s / fs,
                end_time_s=next_s / fs,
                duration_s=(next_s - curr_s) / fs,
                is_left_censored=is_left_c,
                is_right_censored=is_right_c,
                has_observed_onset=(not is_left_c),
                has_observed_termination=(not is_right_c),
                is_complete=((not is_left_c) and (not is_right_c))
            ))
            run_idx += 1

    metadata = {
        "dataset": "LTAFDB",
        "record_id": record_id,
        "patient_id": patient_id,
        "fs": fs,
        "n_samples": n_samples,
        "duration_s": total_duration_s,
        "duration_h": total_duration_s / 3600.0,
        "af_runs": len(af_runs),
        "total_af_s": sum(r.duration_s for r in af_runs)
    }
    return intervals, af_runs, metadata


def load_shdb_record(
    record_id: str,
    base_dir: str = "/data/mithunmanivannan/datasets/shdb_af",
    metadata_df: Optional[pd.DataFrame] = None
) -> Tuple[List[RhythmInterval], List[AFRun], Dict[str, Any]]:
    """Loads a single SHDB-AF record and produces canonical intervals and AF runs."""
    rec_path = os.path.join(base_dir, record_id)
    header = wfdb.rdheader(rec_path)
    fs = float(header.fs)
    n_samples = header.sig_len
    total_duration_s = n_samples / fs

    # Map subject_id from metadata if available
    subject_id = record_id
    if metadata_df is not None:
        row = metadata_df[metadata_df["Data_ID"].astype(str) == str(int(record_id))]
        if not row.empty:
            subject_id = str(row["Subject_ID"].values[0])
    patient_id = f"shdb_{subject_id}"

    # Try reading .atr
    intervals: List[RhythmInterval] = []
    af_runs: List[AFRun] = []
    try:
        ann = wfdb.rdann(rec_path, 'atr')
    except Exception:
        # Unannotated or missing atr
        intervals.append(RhythmInterval(
            dataset="SHDB-AF",
            patient_id=patient_id,
            record_id=record_id,
            source_file_id=f"{record_id}.dat",
            start_sample=0,
            end_sample=n_samples,
            start_time_s=0.0,
            end_time_s=total_duration_s,
            sampling_frequency=fs,
            rhythm_label_native="UNKNOWN",
            rhythm_label_harmonized=RHYTHM_UNKNOWN,
            annotation_source="WFDB_ATR",
            is_left_censored=False,
            is_right_censored=False,
            signal_available=True,
            rhythm_known=False
        ))
        return intervals, af_runs, {
            "dataset": "SHDB-AF", "record_id": record_id, "patient_id": patient_id,
            "fs": fs, "duration_s": total_duration_s, "duration_h": total_duration_s / 3600.0,
            "af_runs": 0, "total_af_s": 0.0, "is_annotated": False
        }

    rhythm_changes = []
    for s, aux in zip(ann.sample, ann.aux_note):
        aux_clean = aux.strip().replace('\x00', '')
        if aux_clean.startswith('('):
            rhythm_changes.append((int(s), aux_clean[1:]))

    if not rhythm_changes:
        intervals.append(RhythmInterval(
            dataset="SHDB-AF",
            patient_id=patient_id,
            record_id=record_id,
            source_file_id=f"{record_id}.dat",
            start_sample=0,
            end_sample=n_samples,
            start_time_s=0.0,
            end_time_s=total_duration_s,
            sampling_frequency=fs,
            rhythm_label_native="UNKNOWN",
            rhythm_label_harmonized=RHYTHM_UNKNOWN,
            annotation_source="WFDB_ATR",
            is_left_censored=False,
            is_right_censored=False,
            signal_available=True,
            rhythm_known=False
        ))
        return intervals, af_runs, {
            "dataset": "SHDB-AF", "record_id": record_id, "patient_id": patient_id,
            "fs": fs, "duration_s": total_duration_s, "duration_h": total_duration_s / 3600.0,
            "af_runs": 0, "total_af_s": 0.0, "is_annotated": False
        }

    first_s, first_rhythm = rhythm_changes[0]
    if first_s > 0:
        if first_s <= int(fs * 5.0):
            rhythm_changes[0] = (0, first_rhythm)
        else:
            intervals.append(RhythmInterval(
                dataset="SHDB-AF",
                patient_id=patient_id,
                record_id=record_id,
                source_file_id=f"{record_id}.dat",
                start_sample=0,
                end_sample=first_s,
                start_time_s=0.0,
                end_time_s=first_s / fs,
                sampling_frequency=fs,
                rhythm_label_native="INITIAL_UNANNOTATED",
                rhythm_label_harmonized=RHYTHM_UNKNOWN,
                annotation_source="WFDB_ATR",
                is_left_censored=False,
                is_right_censored=False,
                signal_available=True,
                rhythm_known=False
            ))

    run_idx = 0
    for idx, (curr_s, curr_rhythm) in enumerate(rhythm_changes):
        next_s = rhythm_changes[idx + 1][0] if idx + 1 < len(rhythm_changes) else n_samples
        if next_s <= curr_s:
            continue

        harmonized = harmonize_rhythm_label(curr_rhythm)
        is_left_c = (curr_s == 0)
        is_right_c = (next_s == n_samples)

        intervals.append(RhythmInterval(
            dataset="SHDB-AF",
            patient_id=patient_id,
            record_id=record_id,
            source_file_id=f"{record_id}.dat",
            start_sample=curr_s,
            end_sample=next_s,
            start_time_s=curr_s / fs,
            end_time_s=next_s / fs,
            sampling_frequency=fs,
            rhythm_label_native=curr_rhythm,
            rhythm_label_harmonized=harmonized,
            annotation_source="WFDB_ATR",
            is_left_censored=is_left_c,
            is_right_censored=is_right_c,
            signal_available=True,
            rhythm_known=(harmonized != RHYTHM_UNKNOWN)
        ))

        if harmonized == RHYTHM_AF:
            run_id = f"shdb_{record_id}_af_{run_idx:04d}"
            af_runs.append(AFRun(
                dataset="SHDB-AF",
                patient_id=patient_id,
                record_id=record_id,
                run_id=run_id,
                start_sample=curr_s,
                end_sample=next_s,
                start_time_s=curr_s / fs,
                end_time_s=next_s / fs,
                duration_s=(next_s - curr_s) / fs,
                is_left_censored=is_left_c,
                is_right_censored=is_right_c,
                has_observed_onset=(not is_left_c),
                has_observed_termination=(not is_right_c),
                is_complete=((not is_left_c) and (not is_right_c))
            ))
            run_idx += 1

    metadata = {
        "dataset": "SHDB-AF",
        "record_id": record_id,
        "patient_id": patient_id,
        "fs": fs,
        "n_samples": n_samples,
        "duration_s": total_duration_s,
        "duration_h": total_duration_s / 3600.0,
        "af_runs": len(af_runs),
        "total_af_s": sum(r.duration_s for r in af_runs),
        "is_annotated": True
    }
    return intervals, af_runs, metadata


def load_iridia_record(
    record_id: str,
    record_info: pd.Series,
    af_intervals_df: pd.DataFrame,
    calibration_df: Optional[pd.DataFrame] = None
) -> Tuple[List[RhythmInterval], List[AFRun], Dict[str, Any]]:
    """Loads a single IRIDIA-AF record from audited metadata and interval files."""
    patient_id = str(record_info["patient_id"])
    fs = 200.0
    total_duration_s = float(record_info["record_seconds"])
    n_samples = int(record_info["record_samples"])

    # Find calibration prefix intervals to mark as UNKNOWN
    calib_s = 0.0
    if calibration_df is not None:
        calib_rows = calibration_df[calibration_df["record_id"] == record_id]
        if not calib_rows.empty:
            calib_s = float(calib_rows["excluded_seconds"].sum())
    if calib_s == 0.0:
        # Default 30s prefix per 24h file if recorded
        calib_s = 30.0

    intervals: List[RhythmInterval] = []
    af_runs: List[AFRun] = []

    # 1. Calibration prefix
    if calib_s > 0:
        intervals.append(RhythmInterval(
            dataset="IRIDIA-AF",
            patient_id=patient_id,
            record_id=record_id,
            source_file_id=f"{record_id}.h5",
            start_sample=0,
            end_sample=int(calib_s * fs),
            start_time_s=0.0,
            end_time_s=calib_s,
            sampling_frequency=fs,
            rhythm_label_native="CALIBRATION_PREFIX",
            rhythm_label_harmonized=RHYTHM_UNKNOWN,
            annotation_source="IRIDIA_CALIBRATION_AUDIT",
            is_left_censored=False,
            is_right_censored=False,
            signal_available=True,
            rhythm_known=False
        ))

    # 2. AF episodes
    rec_af = af_intervals_df[af_intervals_df["record_id"] == record_id].sort_values("start_seconds")
    curr_t = calib_s

    run_idx = 0
    for _, ep in rec_af.iterrows():
        ep_start_s = float(ep["start_seconds"])
        ep_end_s = float(ep["end_seconds"])

        # Non-AF interval before AF run
        if ep_start_s > curr_t:
            intervals.append(RhythmInterval(
                dataset="IRIDIA-AF",
                patient_id=patient_id,
                record_id=record_id,
                source_file_id=f"{record_id}.h5",
                start_sample=int(curr_t * fs),
                end_sample=int(ep_start_s * fs),
                start_time_s=curr_t,
                end_time_s=ep_start_s,
                sampling_frequency=fs,
                rhythm_label_native="NON_AF",
                rhythm_label_harmonized=RHYTHM_UNKNOWN, # probable sinus, not confirmed
                annotation_source="IRIDIA_EXPERT_ANNOTATION",
                is_left_censored=False,
                is_right_censored=False,
                signal_available=True,
                rhythm_known=True
            ))

        # AF run
        is_left_c = bool(ep.get("left_observation_censored", ep_start_s <= calib_s + 1.0))
        is_right_c = bool(ep.get("end_at_observation_boundary", ep_end_s >= total_duration_s - 1.0))

        intervals.append(RhythmInterval(
            dataset="IRIDIA-AF",
            patient_id=patient_id,
            record_id=record_id,
            source_file_id=f"{record_id}.h5",
            start_sample=int(ep_start_s * fs),
            end_sample=int(ep_end_s * fs),
            start_time_s=ep_start_s,
            end_time_s=ep_end_s,
            sampling_frequency=fs,
            rhythm_label_native="AF",
            rhythm_label_harmonized=RHYTHM_AF,
            annotation_source="IRIDIA_EXPERT_ANNOTATION",
            is_left_censored=is_left_c,
            is_right_censored=is_right_c,
            signal_available=True,
            rhythm_known=True
        ))

        run_id = f"iridia_{record_id}_af_{run_idx:04d}"
        af_runs.append(AFRun(
            dataset="IRIDIA-AF",
            patient_id=patient_id,
            record_id=record_id,
            run_id=run_id,
            start_sample=int(ep_start_s * fs),
            end_sample=int(ep_end_s * fs),
            start_time_s=ep_start_s,
            end_time_s=ep_end_s,
            duration_s=ep_end_s - ep_start_s,
            is_left_censored=is_left_c,
            is_right_censored=is_right_c,
            has_observed_onset=(not is_left_c),
            has_observed_termination=(not is_right_c),
            is_complete=((not is_left_c) and (not is_right_c))
        ))
        run_idx += 1
        curr_t = ep_end_s

    # Trailing interval
    if curr_t < total_duration_s:
        intervals.append(RhythmInterval(
            dataset="IRIDIA-AF",
            patient_id=patient_id,
            record_id=record_id,
            source_file_id=f"{record_id}.h5",
            start_sample=int(curr_t * fs),
            end_sample=n_samples,
            start_time_s=curr_t,
            end_time_s=total_duration_s,
            sampling_frequency=fs,
            rhythm_label_native="NON_AF",
            rhythm_label_harmonized=RHYTHM_UNKNOWN,
            annotation_source="IRIDIA_EXPERT_ANNOTATION",
            is_left_censored=False,
            is_right_censored=False,
            signal_available=True,
            rhythm_known=True
        ))

    metadata = {
        "dataset": "IRIDIA-AF",
        "record_id": record_id,
        "patient_id": patient_id,
        "fs": fs,
        "n_samples": n_samples,
        "duration_s": total_duration_s,
        "duration_h": total_duration_s / 3600.0,
        "af_runs": len(af_runs),
        "total_af_s": sum(r.duration_s for r in af_runs),
        "is_annotated": True
    }
    return intervals, af_runs, metadata
