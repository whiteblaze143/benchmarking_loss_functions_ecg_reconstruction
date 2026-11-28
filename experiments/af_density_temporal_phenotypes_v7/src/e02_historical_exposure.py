"""E02: Historical AF Exposure Manifest and Feasibility Matrix.
Constructs Population A (newly observed AF onsets with complete preceding monitoring)
and Population B (episodes with sufficient duration for fixed elapsed-time physiological measurement).
Extracts historical horizons H1 (1h), H3 (3h), H6 (6h, primary), and H12 (12h).
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Import continuous density solver
try:
    from src.density_continuous import compute_continuous_density
except ImportError:
    try:
        from .density_continuous import compute_continuous_density
    except ImportError:
        from density_continuous import compute_continuous_density


def merge_overlapping_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merges overlapping or touching half-open intervals into disjoint sorted intervals."""
    if not intervals:
        return []
    sorted_int = sorted(intervals, key=lambda x: x[0])
    merged = []
    for s, e in sorted_int:
        if e <= s:
            continue
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(float(s), float(e)) for s, e in merged]


def extract_historical_exposure_features(
    record_runs: pd.DataFrame,
    onset_t: float,
    horizon_s: float
) -> Dict[str, Any]:
    """Calculates historical AF burden, continuous density, run count, recency, and gap to previous AF."""
    hist_start = max(0.0, onset_t - horizon_s)
    hist_dur = onset_t - hist_start
    
    # Preceding runs that overlap [hist_start, onset_t)
    pre_runs = record_runs[
        (record_runs["end_time_s"] > hist_start) &
        (record_runs["start_time_s"] < onset_t)
    ].sort_values("start_time_s").reset_index(drop=True)
    
    if pre_runs.empty or hist_dur <= 0:
        return {
            "hist_burden": 0.0,
            "hist_density": None,
            "hist_run_count": 0,
            "hist_af_duration_s": 0.0,
            "longest_preceding_run_s": 0.0,
            "immediately_preceding_dur_s": 0.0,
            "most_recent_non_af_gap_s": hist_dur,
            "time_since_last_prolonged_s": np.nan,
            "recent_15m_burden": 0.0,
            "recent_30m_burden": 0.0,
            "recent_60m_burden": 0.0
        }
        
    raw_intervals = []
    tot_af_s = 0.0
    for _, r in pre_runs.iterrows():
        s = max(hist_start, r["start_time_s"])
        e = min(onset_t, r["end_time_s"])
        if e > s:
            raw_intervals.append((s - hist_start, e - hist_start))
            tot_af_s += (e - s)
            
    intervals = merge_overlapping_intervals(raw_intervals)
    burden = tot_af_s / hist_dur if hist_dur > 0 else 0.0
    
    # Continuous density (defined when 0 < burden < 1 and multiple runs/gaps)
    try:
        density = compute_continuous_density(intervals, hist_dur) if len(intervals) >= 1 else None
    except Exception:
        density = None
    
    longest_s = max((e - s) for s, e in intervals) if intervals else 0.0
    
    # Immediately preceding AF run and most recent non-AF gap
    last_run = pre_runs.iloc[-1]
    last_end = min(onset_t, last_run["end_time_s"])
    gap_s = onset_t - last_end
    last_dur = last_run["duration_s"]
    
    # Time since last prolonged AF episode (>=30m)
    prol_pre = pre_runs[pre_runs["duration_s"] >= 1800.0]
    if not prol_pre.empty:
        time_since_prol = onset_t - prol_pre["end_time_s"].max()
    else:
        time_since_prol = np.nan
        
    # Recency burdens (last 15m, 30m, 60m)
    def calc_recency(rec_sec: float) -> float:
        r_start = max(hist_start, onset_t - rec_sec)
        r_dur = onset_t - r_start
        if r_dur <= 0:
            return 0.0
        af_in_r = 0.0
        for s, e in intervals:
            # Shift back to absolute time
            abs_s = s + hist_start
            abs_e = e + hist_start
            ov_s = max(r_start, abs_s)
            ov_e = min(onset_t, abs_e)
            if ov_e > ov_s:
                af_in_r += (ov_e - ov_s)
        return af_in_r / r_dur
        
    b15 = calc_recency(900.0)
    b30 = calc_recency(1800.0)
    b60 = calc_recency(3600.0)
    
    return {
        "hist_burden": burden,
        "hist_density": density,
        "hist_run_count": len(intervals),
        "hist_af_duration_s": tot_af_s,
        "longest_preceding_run_s": longest_s,
        "immediately_preceding_dur_s": last_dur,
        "most_recent_non_af_gap_s": gap_s,
        "time_since_last_prolonged_s": time_since_prol,
        "recent_15m_burden": b15,
        "recent_30m_burden": b30,
        "recent_60m_burden": b60
    }


def build_historical_exposure_manifest(
    runs_df: pd.DataFrame,
    primary_horizon_h: float = 6.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Builds Population A & B manifests across IRIDIA-AF (primary), LTAFDB, and SHDB.
    
    Returns:
      (exposure_manifest_df, landmark_feasibility_df, conditional_support_df)
    """
    manifest_rows = []
    
    # Process all newly observed AF onsets with observed onset
    obs_runs = runs_df[runs_df["has_observed_onset"] == True].copy()
    obs_runs = obs_runs.sort_values(["dataset", "patient_id", "start_time_s"]).reset_index(drop=True)
    
    horizons = {
        "H1": 3600.0,
        "H3": 10800.0,
        "H6": 21600.0,
        "H12": 43200.0
    }
    
    for _, r in obs_runs.iterrows():
        ds = r["dataset"]
        pid = r["patient_id"]
        rec_id = str(r["record_id"])
        onset_t = float(r["start_time_s"])
        dur_s = float(r["duration_s"])
        
        # Preceding runs in this record
        rec_runs = runs_df[
            (runs_df["dataset"] == ds) &
            (runs_df["record_id"] == rec_id) &
            (runs_df["start_time_s"] < onset_t)
        ]
        
        # Determine population eligibility
        has_1h = onset_t >= 3600.0
        has_3h = onset_t >= 10800.0
        has_6h = onset_t >= 21600.0
        has_12h = onset_t >= 43200.0
        
        # Landmark durations for Population B
        surv_180s = dur_s >= 180.0
        surv_450s = dur_s >= 450.0
        surv_1050s = dur_s >= 1050.0
        surv_1950s = dur_s >= 1950.0
        
        # Extract features for primary horizon H6 (and H1, H3)
        feat_h6 = extract_historical_exposure_features(rec_runs, onset_t, horizons["H6"])
        feat_h1 = extract_historical_exposure_features(rec_runs, onset_t, horizons["H1"])
        feat_h3 = extract_historical_exposure_features(rec_runs, onset_t, horizons["H3"])
        
        manifest_rows.append({
            "episode_id": f"{ds}_{pid}_{r['run_id']}",
            "dataset": ds,
            "patient_id": pid,
            "record_id": rec_id,
            "onset_time_s": onset_t,
            "onset_time_h": onset_t / 3600.0,
            "duration_s": dur_s,
            "duration_m": dur_s / 60.0,
            "is_complete": r["is_complete"],
            "has_observed_termination": r["has_observed_termination"],
            # Eligibility flags
            "has_1h_history": has_1h,
            "has_3h_history": has_3h,
            "has_6h_history": has_6h,
            "has_12h_history": has_12h,
            "is_population_a": has_6h, # Primary Population A: >=6h preceding monitoring
            "survives_to_180s": surv_180s,
            "survives_to_450s": surv_450s,
            "survives_to_1050s": surv_1050s,
            "survives_to_1950s": surv_1950s,
            "is_population_b_primary": has_6h and surv_1050s, # Primary Population B: >=6h history + survives to 17.5m
            # H6 Features (Primary)
            "h6_burden": feat_h6["hist_burden"],
            "h6_density": feat_h6["hist_density"],
            "h6_run_count": feat_h6["hist_run_count"],
            "h6_af_duration_s": feat_h6["hist_af_duration_s"],
            "h6_longest_preceding_run_s": feat_h6["longest_preceding_run_s"],
            "h6_immediately_preceding_dur_s": feat_h6["immediately_preceding_dur_s"],
            "h6_most_recent_non_af_gap_s": feat_h6["most_recent_non_af_gap_s"],
            "h6_time_since_last_prolonged_s": feat_h6["time_since_last_prolonged_s"],
            "h6_recent_15m_burden": feat_h6["recent_15m_burden"],
            "h6_recent_30m_burden": feat_h6["recent_30m_burden"],
            "h6_recent_60m_burden": feat_h6["recent_60m_burden"],
            # H1 and H3 Sensitivity Features
            "h1_burden": feat_h1["hist_burden"],
            "h1_density": feat_h1["hist_density"],
            "h3_burden": feat_h3["hist_burden"],
            "h3_density": feat_h3["hist_density"],
        })
        
    df_manifest = pd.DataFrame(manifest_rows)
    
    # 2. Build landmark feasibility table
    feas_records = []
    dur_thresholds = [
        ("3m (180s)", 180.0, "survives_to_180s"),
        ("7.5m (450s)", 450.0, "survives_to_450s"),
        ("17.5m (1050s)", 1050.0, "survives_to_1050s"),
        ("32.5m (1950s)", 1950.0, "survives_to_1950s")
    ]
    for h_label, h_flag in [("1h", "has_1h_history"), ("3h", "has_3h_history"), ("6h (Primary)", "has_6h_history"), ("12h", "has_12h_history")]:
        for d_label, d_sec, d_flag in dur_thresholds:
            for ds in ["IRIDIA-AF", "LTAFDB", "SHDB-AF", "ALL"]:
                sub = df_manifest if ds == "ALL" else df_manifest[df_manifest["dataset"] == ds]
                eligible = sub[sub[h_flag] & sub[d_flag]]
                feas_records.append({
                    "dataset": ds,
                    "history_horizon": h_label,
                    "duration_landmark": d_label,
                    "n_eligible_episodes": len(eligible),
                    "n_independent_patients": eligible["patient_id"].nunique(),
                    "observed_terminations": int(eligible["has_observed_termination"].sum()),
                    "mean_history_burden": eligible["h6_burden"].mean() if not eligible.empty else 0.0,
                    "episodes_with_valid_density": eligible["h6_density"].notna().sum()
                })
                
    df_feas = pd.DataFrame(feas_records)
    
    # 3. Conditional support table
    # Evaluates within-patient density variation
    cond_records = []
    for ds in ["IRIDIA-AF", "LTAFDB", "SHDB-AF", "ALL"]:
        sub = df_manifest[df_manifest["is_population_b_primary"]] if ds == "ALL" else df_manifest[(df_manifest["dataset"] == ds) & (df_manifest["is_population_b_primary"])]
        
        # Patients with >= 2 episodes with valid density
        dens_valid = sub[sub["h6_density"].notna()]
        p_counts = dens_valid.groupby("patient_id")["h6_density"].agg(["count", "min", "max", "std"]).reset_index()
        p_counts["range"] = p_counts["max"] - p_counts["min"]
        
        pts_multiple = len(p_counts[p_counts["count"] >= 2])
        pts_with_range_05 = len(p_counts[p_counts["range"] >= 0.05])
        pts_with_range_10 = len(p_counts[p_counts["range"] >= 0.10])
        
        cond_records.append({
            "dataset": ds,
            "population": "Population_B_Primary_H6_1050s",
            "total_episodes": len(sub),
            "total_patients": sub["patient_id"].nunique(),
            "episodes_with_valid_density": len(dens_valid),
            "patients_with_valid_density": dens_valid["patient_id"].nunique(),
            "patients_with_ge_2_episodes": pts_multiple,
            "patients_with_density_range_ge_0_05": pts_with_range_05,
            "patients_with_density_range_ge_0_10": pts_with_range_10,
            "mean_within_patient_density_std": p_counts["std"].mean() if not p_counts.empty else 0.0
        })
        
    df_cond = pd.DataFrame(cond_records)
    
    return df_manifest, df_feas, df_cond
