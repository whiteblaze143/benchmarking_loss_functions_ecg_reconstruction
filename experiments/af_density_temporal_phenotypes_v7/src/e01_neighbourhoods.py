"""E01: Temporal Neighbourhood Construction and Risk Time Extraction.
Defines pre-onset and post-termination windows surrounding prolonged AF episodes (>=30m),
handles overlapping neighbourhoods via nearest-boundary assignment,
extracts non-AF risk time (T_risk), and selects matched remote comparison intervals.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


BAND_CONFIGS = [
    # Individual bands
    ("pre_0_5m", -300.0, 0.0, "pre", "individual"),
    ("pre_5_15m", -900.0, -300.0, "pre", "individual"),
    ("pre_15_30m", -1800.0, -900.0, "pre", "individual"),
    ("pre_30_60m", -3600.0, -1800.0, "pre", "individual"),
    ("post_0_5m", 0.0, 300.0, "post", "individual"),
    ("post_5_15m", 300.0, 900.0, "post", "individual"),
    ("post_15_30m", 900.0, 1800.0, "post", "individual"),
    ("post_30_60m", 1800.0, 3600.0, "post", "individual"),
    # Composite bands
    ("pre_0_15m", -900.0, 0.0, "pre", "composite_primary"),
    ("post_0_15m", 0.0, 900.0, "post", "composite_primary"),
    ("pre_0_60m", -3600.0, 0.0, "pre", "composite_60m"),
    ("post_0_60m", 0.0, 3600.0, "post", "composite_60m"),
]


def build_index_episode_manifest(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts all eligible prolonged AF index episodes (>=30m) from LTAFDB and SHDB."""
    sub = runs_df[
        (runs_df["dataset"].isin(["LTAFDB", "SHDB-AF"])) &
        (runs_df["duration_s"] >= 1800.0)
    ].copy()
    
    sub = sub.sort_values(["dataset", "patient_id", "start_time_s"]).reset_index(drop=True)
    sub["index_episode_id"] = [f"{r.dataset}_{r.patient_id}_idx_{i:04d}" for i, r in sub.iterrows()]
    return sub


def extract_window_rhythm_metrics(
    rhythms_for_record: pd.DataFrame,
    w_start: float,
    w_end: float,
    rec_duration_s: float
) -> Dict[str, float]:
    """Calculates physical duration, known rhythm duration, AF duration, and known non-AF risk time."""
    w_start_clamped = max(0.0, w_start)
    w_end_clamped = min(rec_duration_s, w_end)
    
    nominal_dur = max(0.0, w_end - w_start)
    clamped_dur = max(0.0, w_end_clamped - w_start_clamped)
    
    if clamped_dur <= 0.0:
        return {
            "physical_duration_s": nominal_dur,
            "clamped_duration_s": 0.0,
            "known_rhythm_s": 0.0,
            "af_duration_s": 0.0,
            "known_non_af_s": 0.0,
            "is_censored": True
        }
        
    is_censored = (w_start < 0.0) or (w_end > rec_duration_s)
    
    # Fast overlap filter
    overlap = rhythms_for_record[
        (rhythms_for_record["end_time_s"] > w_start_clamped) &
        (rhythms_for_record["start_time_s"] < w_end_clamped)
    ]
    
    known_s = 0.0
    af_s = 0.0
    
    for _, r in overlap.iterrows():
        s = max(w_start_clamped, r["start_time_s"])
        e = min(w_end_clamped, r["end_time_s"])
        dur = max(0.0, e - s)
        
        if r["rhythm_known"]:
            known_s += dur
            if r["rhythm_label_harmonized"] == "AF":
                af_s += dur
                
    non_af_s = max(0.0, known_s - af_s)
    
    return {
        "physical_duration_s": nominal_dur,
        "clamped_duration_s": clamped_dur,
        "known_rhythm_s": known_s,
        "af_duration_s": af_s,
        "known_non_af_s": non_af_s,
        "is_censored": is_censored
    }


def construct_neighbourhood_intervals(
    index_df: pd.DataFrame,
    rhythms_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    cohort_inv: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constructs all candidate neighbourhood intervals and assigns brief events via nearest-boundary rule.
    
    Returns:
      (neighbourhood_intervals_df, event_assignments_df)
    """
    rec_durs = cohort_inv.set_index(["dataset", "record_id"])["duration_s"].to_dict()
    
    # Pre-filter brief runs (<30s)
    brief_runs = runs_df[
        (runs_df["dataset"].isin(["LTAFDB", "SHDB-AF"])) &
        (runs_df["duration_s"] < 30.0) &
        (runs_df["is_complete"] == True)
    ].copy()
    
    # Pre-filter other runs for counts
    inter_runs = runs_df[
        (runs_df["dataset"].isin(["LTAFDB", "SHDB-AF"])) &
        (runs_df["duration_s"] >= 30.0) &
        (runs_df["duration_s"] < 1800.0)
    ].copy()
    
    intervals_records = []
    candidate_event_links = []
    
    # Group rhythms by record for fast lookup
    rhythms_grouped = {k: v for k, v in rhythms_df.groupby(["dataset", "record_id"])}
    brief_grouped = {k: v for k, v in brief_runs.groupby(["dataset", "record_id"])}
    inter_grouped = {k: v for k, v in inter_runs.groupby(["dataset", "record_id"])}
    prol_grouped = {k: v for k, v in index_df.groupby(["dataset", "record_id"])}
    
    for _, idx_row in index_df.iterrows():
        ds = idx_row["dataset"]
        rec_id = str(idx_row["record_id"])
        pid = idx_row["patient_id"]
        idx_id = idx_row["index_episode_id"]
        t_on = float(idx_row["start_time_s"])
        t_off = float(idx_row["end_time_s"])
        rec_dur = rec_durs.get((ds, rec_id), idx_row["end_time_s"] + 7200.0)
        
        rec_rhythms = rhythms_grouped.get((ds, rec_id), pd.DataFrame())
        rec_brief = brief_grouped.get((ds, rec_id), pd.DataFrame())
        rec_inter = inter_grouped.get((ds, rec_id), pd.DataFrame())
        rec_prol = prol_grouped.get((ds, rec_id), pd.DataFrame())
        
        for band_name, rel_start, rel_end, side, band_type in BAND_CONFIGS:
            if side == "pre":
                w_start = t_on + rel_start
                w_end = t_on + rel_end
                boundary_t = t_on
            else:
                w_start = t_off + rel_start
                w_end = t_off + rel_end
                boundary_t = t_off
                
            metrics = extract_window_rhythm_metrics(rec_rhythms, w_start, w_end, rec_dur)
            
            # Events falling within window
            if not rec_brief.empty:
                b_in_win = rec_brief[
                    (rec_brief["start_time_s"] >= w_start) &
                    (rec_brief["start_time_s"] < w_end)
                ]
                brief_count = len(b_in_win)
                for _, b_row in b_in_win.iterrows():
                    dist_to_boundary = abs(b_row["start_time_s"] - boundary_t)
                    candidate_event_links.append({
                        "run_id": b_row["run_id"],
                        "dataset": ds,
                        "patient_id": pid,
                        "record_id": rec_id,
                        "event_start_time_s": b_row["start_time_s"],
                        "event_duration_s": b_row["duration_s"],
                        "index_episode_id": idx_id,
                        "band_name": band_name,
                        "side": side,
                        "band_type": band_type,
                        "boundary_time_s": boundary_t,
                        "dist_to_boundary_s": dist_to_boundary
                    })
            else:
                brief_count = 0
                
            # Intermediate AF onsets in window
            if not rec_inter.empty:
                inter_count = len(rec_inter[
                    (rec_inter["start_time_s"] >= w_start) &
                    (rec_inter["start_time_s"] < w_end)
                ])
            else:
                inter_count = 0
                
            # Other prolonged AF onsets in window
            if not rec_prol.empty:
                other_prol = rec_prol[
                    (rec_prol["index_episode_id"] != idx_id) &
                    (rec_prol["start_time_s"] >= w_start) &
                    (rec_prol["start_time_s"] < w_end)
                ]
                other_prol_count = len(other_prol)
            else:
                other_prol_count = 0
                
            intervals_records.append({
                "dataset": ds,
                "patient_id": pid,
                "record_id": rec_id,
                "index_episode_id": idx_id,
                "index_start_time_s": t_on,
                "index_end_time_s": t_off,
                "index_duration_s": idx_row["duration_s"],
                "band_name": band_name,
                "side": side,
                "band_type": band_type,
                "window_start_s": w_start,
                "window_end_s": w_end,
                "physical_duration_s": metrics["physical_duration_s"],
                "clamped_duration_s": metrics["clamped_duration_s"],
                "known_rhythm_s": metrics["known_rhythm_s"],
                "af_duration_s": metrics["af_duration_s"],
                "known_non_af_s": metrics["known_non_af_s"],
                "risk_time_h": metrics["known_non_af_s"] / 3600.0,
                "raw_brief_af_count": brief_count,
                "intermediate_af_count": inter_count,
                "other_prolonged_af_count": other_prol_count,
                "is_censored": metrics["is_censored"]
            })
            
    df_intervals = pd.DataFrame(intervals_records)
    df_links = pd.DataFrame(candidate_event_links)
    
    # Apply Deterministic Nearest-Boundary Rule (E01.5)
    # For individual bands and composite bands independently
    assigned_links = []
    if not df_links.empty:
        for b_type in ["individual", "composite_primary", "composite_60m"]:
            sub_links = df_links[df_links["band_type"] == b_type].copy()
            # Sort by run_id, distance to boundary, then index_episode_id (tie breaking)
            sub_links = sub_links.sort_values(
                ["run_id", "dist_to_boundary_s", "index_episode_id"]
            ).reset_index(drop=True)
            # Pick first occurrence per run_id
            dedup = sub_links.drop_duplicates(subset=["run_id"]).copy()
            dedup["assigned_via"] = "nearest_boundary_primary"
            assigned_links.append(dedup)
            
        df_assigned = pd.concat(assigned_links, ignore_index=True)
    else:
        df_assigned = pd.DataFrame()
        
    # Recompute resolved brief_af_onset_count in df_intervals
    if not df_assigned.empty:
        counts = df_assigned.groupby(["index_episode_id", "band_name"]).size().to_dict()
        df_intervals["resolved_brief_af_count"] = [
            counts.get((r["index_episode_id"], r["band_name"]), 0)
            for _, r in df_intervals.iterrows()
        ]
    else:
        df_intervals["resolved_brief_af_count"] = 0
        
    return df_intervals, df_assigned


def construct_remote_comparison_manifest(
    index_df: pd.DataFrame,
    rhythms_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    cohort_inv: pd.DataFrame,
    exclusion_distance_s: float = 3600.0,
    window_duration_s: float = 900.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constructs matched remote comparison intervals from the same recordings.
    
    Requires intervals to be >= exclusion_distance_s from ANY prolonged episode boundary.
    Extracts 15m (900s) windows and counts brief AF onsets.
    """
    rec_durs = cohort_inv.set_index(["dataset", "record_id"])["duration_s"].to_dict()
    rhythms_grouped = {k: v for k, v in rhythms_df.groupby(["dataset", "record_id"])}
    
    brief_runs = runs_df[
        (runs_df["dataset"].isin(["LTAFDB", "SHDB-AF"])) &
        (runs_df["duration_s"] < 30.0) &
        (runs_df["is_complete"] == True)
    ].copy()
    brief_grouped = {k: v for k, v in brief_runs.groupby(["dataset", "record_id"])}
    
    remote_records = []
    
    # Process each record that contains at least one prolonged episode
    for (ds, rec_id), prol_group in index_df.groupby(["dataset", "record_id"]):
        pid = prol_group["patient_id"].iloc[0]
        rec_dur = rec_durs.get((ds, str(rec_id)), prol_group["end_time_s"].max() + 7200.0)
        rec_rhythms = rhythms_grouped.get((ds, str(rec_id)), pd.DataFrame())
        rec_brief = brief_grouped.get((ds, str(rec_id)), pd.DataFrame())
        
        # Build exclusion intervals: [t_on - excl, t_off + excl]
        exclusions = []
        for _, r in prol_group.iterrows():
            exclusions.append((r["start_time_s"] - exclusion_distance_s, r["end_time_s"] + exclusion_distance_s))
            
        # Merge overlapping exclusion intervals
        exclusions.sort()
        merged_excl = []
        for s, e in exclusions:
            if not merged_excl or s > merged_excl[-1][1]:
                merged_excl.append([max(0.0, s), min(rec_dur, e)])
            else:
                merged_excl[-1][1] = max(merged_excl[-1][1], min(rec_dur, e))
                
        # Find eligible remote intervals outside exclusions
        curr_t = 0.0
        rem_windows = []
        for s_ex, e_ex in merged_excl:
            if s_ex > curr_t + window_duration_s:
                # Can tile windows of length window_duration_s
                t_w = curr_t
                while t_w + window_duration_s <= s_ex:
                    rem_windows.append((t_w, t_w + window_duration_s))
                    t_w += window_duration_s
            curr_t = max(curr_t, e_ex)
            
        if curr_t + window_duration_s <= rec_dur:
            t_w = curr_t
            while t_w + window_duration_s <= rec_dur:
                rem_windows.append((t_w, t_w + window_duration_s))
                t_w += window_duration_s
                
        for w_idx, (w_s, w_e) in enumerate(rem_windows):
            metrics = extract_window_rhythm_metrics(rec_rhythms, w_s, w_e, rec_dur)
            
            # Count brief AF runs in remote window
            if not rec_brief.empty:
                b_count = len(rec_brief[
                    (rec_brief["start_time_s"] >= w_s) &
                    (rec_brief["start_time_s"] < w_e)
                ])
            else:
                b_count = 0
                
            remote_records.append({
                "dataset": ds,
                "patient_id": pid,
                "record_id": rec_id,
                "remote_window_id": f"{ds}_{pid}_rem_{w_idx:03d}",
                "window_start_s": w_s,
                "window_end_s": w_e,
                "physical_duration_s": metrics["physical_duration_s"],
                "clamped_duration_s": metrics["clamped_duration_s"],
                "known_rhythm_s": metrics["known_rhythm_s"],
                "af_duration_s": metrics["af_duration_s"],
                "known_non_af_s": metrics["known_non_af_s"],
                "risk_time_h": metrics["known_non_af_s"] / 3600.0,
                "brief_af_onset_count": b_count,
                "is_censored": metrics["is_censored"],
                "min_dist_to_prolonged_boundary_s": exclusion_distance_s
            })
            
    df_remote = pd.DataFrame(remote_records)
    
    # Summary of risk time across conditions
    summary_records = []
    return df_remote


def build_risk_time_summary(
    df_intervals: pd.DataFrame,
    df_remote: pd.DataFrame
) -> pd.DataFrame:
    """Builds E01_risk_time_summary.csv comparing exposure across conditions."""
    records = []
    
    # Filter for primary comparison bands
    bands = ["pre_0_15m", "post_0_15m", "pre_0_60m", "post_0_60m"]
    for b in bands:
        sub = df_intervals[df_intervals["band_name"] == b]
        for ds in ["LTAFDB", "SHDB-AF", "ALL"]:
            csub = sub if ds == "ALL" else sub[sub["dataset"] == ds]
            n_win = len(csub)
            tot_phys_h = csub["clamped_duration_s"].sum() / 3600.0
            tot_known_h = csub["known_rhythm_s"].sum() / 3600.0
            tot_af_h = csub["af_duration_s"].sum() / 3600.0
            tot_risk_h = csub["known_non_af_s"].sum() / 3600.0
            tot_brief = csub["resolved_brief_af_count"].sum()
            
            records.append({
                "dataset": ds,
                "condition": b,
                "n_windows": n_win,
                "total_physical_h": tot_phys_h,
                "total_known_rhythm_h": tot_known_h,
                "total_af_h": tot_af_h,
                "total_risk_time_h": tot_risk_h,
                "risk_time_ratio": tot_risk_h / tot_known_h if tot_known_h > 0 else 0.0,
                "total_brief_onsets": int(tot_brief),
                "physical_rate_per_h": tot_brief / tot_known_h if tot_known_h > 0 else 0.0,
                "risk_time_rate_per_h": tot_brief / tot_risk_h if tot_risk_h > 0 else 0.0
            })
            
    # Add Remote comparison
    for ds in ["LTAFDB", "SHDB-AF", "ALL"]:
        csub = df_remote if ds == "ALL" else df_remote[df_remote["dataset"] == ds]
        n_win = len(csub)
        tot_phys_h = csub["clamped_duration_s"].sum() / 3600.0
        tot_known_h = csub["known_rhythm_s"].sum() / 3600.0
        tot_af_h = csub["af_duration_s"].sum() / 3600.0
        tot_risk_h = csub["known_non_af_s"].sum() / 3600.0
        tot_brief = csub["brief_af_onset_count"].sum()
        
        records.append({
            "dataset": ds,
            "condition": "remote_matched_15m",
            "n_windows": n_win,
            "total_physical_h": tot_phys_h,
            "total_known_rhythm_h": tot_known_h,
            "total_af_h": tot_af_h,
            "total_risk_time_h": tot_risk_h,
            "risk_time_ratio": tot_risk_h / tot_known_h if tot_known_h > 0 else 0.0,
            "total_brief_onsets": int(tot_brief),
            "physical_rate_per_h": tot_brief / tot_known_h if tot_known_h > 0 else 0.0,
            "risk_time_rate_per_h": tot_brief / tot_risk_h if tot_risk_h > 0 else 0.0
        })
        
    return pd.DataFrame(records)
