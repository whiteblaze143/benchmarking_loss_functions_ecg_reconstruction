"""Landmark and cohort extraction for P01-P04 prediction manifests."""
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def extract_p01_landmarks(
    intervals_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    grid_step_s: float = 300.0,
    min_preceding_history_s: float = 1800.0,
    forecast_horizon_s: float = 1800.0,
    primary_sustained_threshold_s: float = 1800.0,
) -> pd.DataFrame:
    """Extracts non-AF monitoring grid landmarks for P01 sustained AF onset forecasting.
    
    A positive event requires an AF episode to begin in (t, t + forecast_horizon_s]
    and reach total duration >= primary_sustained_threshold_s.
    """
    af_onsets = runs_df[runs_df["has_observed_onset"]].copy()

    # Pre-index runs by (dataset, record_id)
    runs_dict = {}
    for (ds, rid), sub in af_onsets.groupby(["dataset", "record_id"]):
        runs_dict[(ds, rid)] = (sub["start_time_s"].values, sub["duration_s"].values)

    rec_bounds = intervals_df.groupby(["dataset", "patient_id", "record_id"]).agg(
        rec_start=("start_time_s", "min"),
        rec_end=("end_time_s", "max")
    ).reset_index()
    rec_bounds_dict = {(r.dataset, r.record_id): (r.rec_start, r.rec_end) for r in rec_bounds.itertuples()}

    non_af = intervals_df[intervals_df["rhythm_label_harmonized"] != "AF"]

    records = []
    for row in non_af.itertuples():
        ds, pid, rid = row.dataset, row.patient_id, row.record_id
        rec_start, rec_end = rec_bounds_dict.get((ds, rid), (0.0, 0.0))
        i_start, i_end = row.start_time_s, row.end_time_s

        grid_start = np.ceil(i_start / grid_step_s) * grid_step_s
        if grid_start >= i_end:
            continue
        grid_points = np.arange(grid_start, i_end, grid_step_s)

        # Ensure preceding history and follow-up
        valid_mask = (grid_points - rec_start >= min_preceding_history_s) & (rec_end - grid_points >= forecast_horizon_s + 1800.0)
        valid_points = grid_points[valid_mask]
        if len(valid_points) == 0:
            continue

        r_starts, r_durs = runs_dict.get((ds, rid), (np.array([]), np.array([])))

        for t in valid_points:
            onset_any = False
            sustained_30m = False
            sustained_6m = False
            sustained_60m = False
            first_onset_time = None
            first_duration = None

            if len(r_starts) > 0:
                idx = (r_starts > t) & (r_starts <= t + forecast_horizon_s)
                if np.any(idx):
                    onset_any = True
                    matching_starts = r_starts[idx]
                    matching_durs = r_durs[idx]
                    first_idx = np.argmin(matching_starts)
                    first_onset_time = matching_starts[first_idx]
                    first_duration = matching_durs[first_idx]

                    sustained_30m = np.any(matching_durs >= 1800.0)
                    sustained_6m = np.any(matching_durs >= 360.0)
                    sustained_60m = np.any(matching_durs >= 3600.0)

            records.append({
                "dataset": ds,
                "patient_id": pid,
                "record_id": rid,
                "landmark_time_s": float(t),
                "historical_start_s": float(max(rec_start, t - 21600.0)),  # up to 6h history
                "current_rhythm": row.rhythm_label_harmonized,
                "target_onset_any_30m": bool(onset_any),
                "target_sustained_30m": bool(sustained_30m),  # Primary Target
                "target_sustained_6m": bool(sustained_6m),
                "target_sustained_60m": bool(sustained_60m),
                "event_time_s": float(first_onset_time) if first_onset_time is not None else float(t + forecast_horizon_s),
                "censoring_time_s": float(rec_end),
                "is_positive": bool(sustained_30m),
                "is_censored": False,
            })

    return pd.DataFrame(records)


def extract_p02_landmarks(
    runs_df: pd.DataFrame,
    landmark_s: float = 60.0,
    primary_persistence_threshold_s: float = 1800.0,
) -> pd.DataFrame:
    """Extracts episode persistence landmarks for P02 at 60s post-onset.
    
    Population: Observed onset AF episodes that survive >= landmark_s.
    Target: Does the episode persist to >= primary_persistence_threshold_s (30m)?
    """
    af_onsets = runs_df[runs_df["has_observed_onset"] & (runs_df["duration_s"] >= landmark_s)].copy()

    records = []
    for row in af_onsets.itertuples():
        total_dur = row.duration_s
        persists_30m = total_dur >= primary_persistence_threshold_s
        persists_6m = total_dur >= 360.0
        persists_60m = total_dur >= 3600.0

        is_censored = row.is_right_censored and (total_dur < primary_persistence_threshold_s)

        records.append({
            "dataset": row.dataset,
            "patient_id": row.patient_id,
            "record_id": row.record_id,
            "run_id": row.run_id,
            "onset_time_s": float(row.start_time_s),
            "landmark_time_s": float(row.start_time_s + landmark_s),
            "elapsed_landmark_s": float(landmark_s),
            "total_duration_s": float(total_dur),
            "target_persists_30m": bool(persists_30m),  # Primary Target
            "target_persists_6m": bool(persists_6m),
            "target_persists_60m": bool(persists_60m),
            "is_positive": bool(persists_30m),
            "is_censored": bool(is_censored),
            "has_observed_termination": bool(row.has_observed_termination),
        })

    return pd.DataFrame(records)


def extract_p03_landmarks(
    runs_df: pd.DataFrame,
    landmarks_s: List[float] = [60.0, 300.0, 900.0],
    termination_horizon_s: float = 900.0,
) -> pd.DataFrame:
    """Extracts dynamic AF termination landmarks for P03 survival prediction.
    
    At landmarks 1m (60s), 5m (300s), 15m (900s):
    Time-to-event: remaining episode duration.
    Primary binary target: Termination within 15 minutes (900s) post-landmark.
    """
    af_onsets = runs_df[runs_df["has_observed_onset"]].copy()

    records = []
    for lm_s in landmarks_s:
        lm_name = f"{int(lm_s // 60)}m" if lm_s >= 60 else f"{int(lm_s)}s"
        eligible = af_onsets[af_onsets["duration_s"] >= lm_s]

        for row in eligible.itertuples():
            time_to_event = row.duration_s - lm_s
            event = row.has_observed_termination
            terminate_15m = (time_to_event <= termination_horizon_s) and event

            records.append({
                "dataset": row.dataset,
                "patient_id": row.patient_id,
                "record_id": row.record_id,
                "run_id": row.run_id,
                "landmark_name": lm_name,
                "landmark_elapsed_s": float(lm_s),
                "landmark_time_s": float(row.start_time_s + lm_s),
                "onset_time_s": float(row.start_time_s),
                "total_duration_s": float(row.duration_s),
                "time_to_event_s": float(time_to_event),
                "event": bool(event),
                "target_terminate_in_15m": bool(terminate_15m),  # Primary Binary Horizon Target
                "is_positive": bool(terminate_15m),
                "is_censored": not bool(event),
            })

    return pd.DataFrame(records)


def parse_noise_holter_headers(dataset_root: Path) -> pd.DataFrame:
    """Parses all 6,130 Noise Holter excerpts from PhysioNet headers."""
    wfdb_dir = dataset_root / "wfdb_files"
    if not wfdb_dir.exists():
        wfdb_dir = dataset_root

    hea_files = sorted(list(wfdb_dir.glob("*.hea")))
    rows = []
    for p in hea_files:
        strip_id = p.stem
        patient_id = strip_id.split("-")[0]
        with open(p, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        fields = lines[0].split()
        channels, fs, samples = int(fields[1]), float(fields[2]), int(fields[3])

        comment_line = next((line for line in lines if line.startswith("# CLASS:")), "# CLASS:Unknown | NOISE:Unknown | EVENTS:None")
        comment = comment_line[1:].strip()
        parts = comment.split("|")
        quality = parts[0].replace("CLASS:", "").strip()
        noise = parts[1].replace("NOISE:", "").strip() if len(parts) > 1 else "None"
        events_str = parts[2].replace("EVENTS:", "").strip() if len(parts) > 2 else "None"

        events = [e.strip() for e in events_str.split("|") if e.strip() and e.strip() != "None"]

        af_present = "A. Fib." in events
        afl_present = "A. Flutter" in events
        ectopy_present = any(e in ["SVPB", "VPB", "SV Couplet", "V Couplet", "TgSV"] for e in events)
        clean_nsr = (quality == "Clean") and (len(events) == 0)

        rows.append({
            "strip_id": strip_id,
            "patient_id": patient_id,
            "recording_prefix": patient_id,
            "dataset": "Noise-Holter",
            "sampling_rate": float(fs),
            "channels": int(channels),
            "sample_count": int(samples),
            "duration_seconds": float(samples / fs),
            "quality_stratum": quality,
            "noise_type": noise,
            "clinical_event_labels": "|".join(sorted(events)) if events else "None",
            "af_present": bool(af_present),
            "afl_present": bool(afl_present),
            "ectopy_present": bool(ectopy_present),
            "clean_nsr": bool(clean_nsr),
            "is_positive": bool(af_present),
            "is_af_ground_truth": bool(af_present),
        })

    return pd.DataFrame(rows)
