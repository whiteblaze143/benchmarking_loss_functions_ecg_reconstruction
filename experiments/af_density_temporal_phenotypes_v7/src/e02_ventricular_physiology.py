"""E02: Fixed Elapsed-Time Ventricular Physiology Extraction.
Extracts ventricular RR intervals and irregularity features in fixed 150-second windows:
- Initial window: 30–180s after onset
- Intermediate window: 5–7.5m after onset
- Primary landmark window: 15–17.5m after onset
- Extended landmark window: 30–32.5m after onset
Calculates CVRR, RMSSD, Mean Rate, Rapid Ventricular Response, and physiological trajectory Delta CVRR.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Import beat extractors
try:
    from src.ventricular_rate import (
        extract_beats_ltafdb,
        extract_beats_shdb,
        extract_beats_iridia
    )
except ImportError:
    try:
        from .ventricular_rate import (
            extract_beats_ltafdb,
            extract_beats_shdb,
            extract_beats_iridia
        )
    except ImportError:
        from ventricular_rate import (
            extract_beats_ltafdb,
            extract_beats_shdb,
            extract_beats_iridia
        )


PHYSIO_WINDOWS = [
    ("win_30_180s", 30.0, 180.0, 180.0, "initial"),
    ("win_5_7_5m", 300.0, 450.0, 450.0, "intermediate"),
    ("win_15_17_5m", 900.0, 1050.0, 1050.0, "primary_landmark"),
    ("win_30_32_5m", 1800.0, 1950.0, 1950.0, "extended_landmark")
]


def extract_window_rr_metrics(beat_times_s: np.ndarray) -> Dict[str, float]:
    """Computes CVRR, RMSSD, rate, and rhythm dynamics from beat timestamps."""
    if len(beat_times_s) < 10:
        return {
            "n_beats": len(beat_times_s),
            "rate_bpm": np.nan,
            "mean_rr_ms": np.nan,
            "median_rr_ms": np.nan,
            "sd_rr_ms": np.nan,
            "cv_rr": np.nan,
            "rmssd_ms": np.nan,
            "median_succ_diff_ms": np.nan,
            "rapid_response_pct": np.nan,
            "bradycardia_pct": np.nan,
            "is_valid_physio": False
        }
        
    # RR intervals in seconds
    rr_s = np.diff(beat_times_s)
    # Physiological filtering: 0.20s <= RR <= 3.0s
    valid_rr = rr_s[(rr_s >= 0.20) & (rr_s <= 3.0)]
    
    if len(valid_rr) < 8:
        return {
            "n_beats": len(beat_times_s),
            "rate_bpm": np.nan,
            "mean_rr_ms": np.nan,
            "median_rr_ms": np.nan,
            "sd_rr_ms": np.nan,
            "cv_rr": np.nan,
            "rmssd_ms": np.nan,
            "median_succ_diff_ms": np.nan,
            "rapid_response_pct": np.nan,
            "bradycardia_pct": np.nan,
            "is_valid_physio": False
        }
        
    mean_rr = float(np.mean(valid_rr))
    sd_rr = float(np.std(valid_rr, ddof=1)) if len(valid_rr) > 1 else 0.0
    cv_rr = sd_rr / mean_rr if mean_rr > 0 else np.nan
    rate_bpm = 60.0 / mean_rr if mean_rr > 0 else np.nan
    
    # Successive differences
    succ_diff = np.abs(np.diff(valid_rr))
    rmssd = float(np.sqrt(np.mean(succ_diff ** 2)) * 1000.0) if len(succ_diff) > 0 else np.nan
    med_diff = float(np.median(succ_diff) * 1000.0) if len(succ_diff) > 0 else np.nan
    
    # Rapid ventricular response (> 120 bpm, RR < 0.500s)
    rapid_pct = float(np.mean(valid_rr < 0.500)) * 100.0
    brady_pct = float(np.mean(valid_rr > 1.000)) * 100.0
    
    return {
        "n_beats": len(beat_times_s),
        "rate_bpm": rate_bpm,
        "mean_rr_ms": mean_rr * 1000.0,
        "median_rr_ms": float(np.median(valid_rr)) * 1000.0,
        "sd_rr_ms": sd_rr * 1000.0,
        "cv_rr": cv_rr,
        "rmssd_ms": rmssd,
        "median_succ_diff_ms": med_diff,
        "rapid_response_pct": rapid_pct,
        "bradycardia_pct": brady_pct,
        "is_valid_physio": True
    }


def extract_features_for_population_b(
    df_manifest: pd.DataFrame,
    max_episodes_per_cohort: int = 500
) -> pd.DataFrame:
    """Extracts fixed elapsed-time RR features for Population B episodes."""
    records = []
    
    # Target episodes with at least 3m duration
    eligible = df_manifest[df_manifest["survives_to_180s"]].copy()
    
    print(f"Extracting fixed elapsed-time physiology across {len(eligible)} candidate episodes...")
    
    for idx, row in eligible.iterrows():
        ds = row["dataset"]
        pid = row["patient_id"]
        rec_id = str(row["record_id"])
        ep_id = row["episode_id"]
        t_on = float(row["onset_time_s"])
        dur_s = float(row["duration_s"])
        
        for w_name, rel_s, rel_e, min_req_dur, w_role in PHYSIO_WINDOWS:
            if dur_s < min_req_dur:
                continue
                
            abs_s = t_on + rel_s
            abs_e = t_on + rel_e
            
            # Extract beats depending on dataset
            try:
                if ds == "IRIDIA-AF":
                    beats_rel, fs = extract_beats_iridia(rec_id, abs_s, abs_e)
                    beat_times = abs_s + (beats_rel / fs)
                elif ds == "LTAFDB":
                    fs = 128.0
                    s_samp = int(round(abs_s * fs))
                    e_samp = int(round(abs_e * fs))
                    beats_samp, _, _ = extract_beats_ltafdb(rec_id, start_sample=s_samp, end_sample=e_samp)
                    beat_times = beats_samp / fs
                elif ds == "SHDB-AF":
                    fs = 200.0
                    s_samp = int(round(abs_s * fs))
                    e_samp = int(round(abs_e * fs))
                    beats_samp, _, _ = extract_beats_shdb(rec_id, start_sample=s_samp, end_sample=e_samp)
                    beat_times = beats_samp / fs
                else:
                    beat_times = np.array([])
            except Exception:
                beat_times = np.array([])
                
            m = extract_window_rr_metrics(beat_times)
            
            records.append({
                "episode_id": ep_id,
                "dataset": ds,
                "patient_id": pid,
                "record_id": rec_id,
                "onset_time_s": t_on,
                "duration_s": dur_s,
                "window_name": w_name,
                "window_role": w_role,
                "window_rel_start_s": rel_s,
                "window_rel_end_s": rel_e,
                "window_abs_start_s": abs_s,
                "window_abs_end_s": abs_e,
                "n_beats": m["n_beats"],
                "rate_bpm": m["rate_bpm"],
                "mean_rr_ms": m["mean_rr_ms"],
                "median_rr_ms": m["median_rr_ms"],
                "sd_rr_ms": m["sd_rr_ms"],
                "cv_rr": m["cv_rr"],
                "rmssd_ms": m["rmssd_ms"],
                "median_succ_diff_ms": m["median_succ_diff_ms"],
                "rapid_response_pct": m["rapid_response_pct"],
                "bradycardia_pct": m["bradycardia_pct"],
                "is_valid_physio": m["is_valid_physio"],
                # Carry forward historical exposures
                "h6_burden": row["h6_burden"],
                "h6_density": row["h6_density"],
                "h6_run_count": row["h6_run_count"],
                "h6_longest_preceding_run_s": row["h6_longest_preceding_run_s"],
                "h6_immediately_preceding_dur_s": row["h6_immediately_preceding_dur_s"],
                "h6_most_recent_non_af_gap_s": row["h6_most_recent_non_af_gap_s"],
                "has_6h_history": row["has_6h_history"]
            })
            
    df_physio = pd.DataFrame(records)
    print(f"Extracted {len(df_physio)} window feature rows across {df_physio['episode_id'].nunique()} episodes.")
    return df_physio
