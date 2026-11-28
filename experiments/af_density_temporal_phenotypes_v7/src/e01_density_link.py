"""E01: Connecting Temporal Neighbourhoods to Continuous Charitos AF Density.
Computes AF burden, exact continuous density, longest-episode share, and brief-run exposure,
and evaluates how temporal neighbourhood clustering interacts with density architecture.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Import verified continuous density solver
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


def analyze_density_architecture_connections(
    runs_df: pd.DataFrame,
    df_intervals: pd.DataFrame,
    df_remote: pd.DataFrame,
    cohort_inv: pd.DataFrame
) -> pd.DataFrame:
    """Produces E01_density_architecture_results.csv linking density to neighbourhood patterns."""
    rec_durs = cohort_inv.set_index(["dataset", "record_id"])["duration_s"].to_dict()
    
    # Calculate patient-level neighbourhood rate ratios
    post_counts = df_intervals[df_intervals["band_name"] == "post_0_15m"].groupby(["dataset", "patient_id"]).agg({
        "resolved_brief_af_count": "sum",
        "risk_time_h": "sum"
    }).reset_index()
    post_counts["post_rate"] = post_counts["resolved_brief_af_count"] / post_counts["risk_time_h"].replace(0, np.nan)
    
    rem_counts = df_remote.groupby(["dataset", "patient_id"]).agg({
        "brief_af_onset_count": "sum",
        "risk_time_h": "sum"
    }).reset_index()
    rem_counts["remote_rate"] = rem_counts["brief_af_onset_count"] / rem_counts["risk_time_h"].replace(0, np.nan)
    
    p_rates = pd.merge(post_counts, rem_counts, on=["dataset", "patient_id"], suffixes=("_post", "_rem"), how="outer")
    p_rates["post_vs_remote_rr"] = p_rates["post_rate"] / p_rates["remote_rate"].replace(0, np.nan)
    
    rate_dict = p_rates.set_index(["dataset", "patient_id"])["post_vs_remote_rr"].to_dict()
    
    records = []
    
    for (ds, pid), group in runs_df.groupby(["dataset", "patient_id"]):
        if ds not in ["LTAFDB", "SHDB-AF"]:
            continue
            
        rec_id = str(group["record_id"].iloc[0])
        rec_dur = rec_durs.get((ds, rec_id), group["end_time_s"].max() + 3600.0)
        
        # Sort and merge overlapping runs
        s_runs = group.sort_values("start_time_s").reset_index(drop=True)
        raw_intervals = [(float(r["start_time_s"]), float(r["end_time_s"])) for _, r in s_runs.iterrows()]
        intervals = merge_overlapping_intervals(raw_intervals)
        
        total_af_s = float(s_runs["duration_s"].sum())
        burden = total_af_s / rec_dur if rec_dur > 0 else 0.0
        
        # Continuous Charitos density
        try:
            density = compute_continuous_density(intervals, rec_dur)
        except Exception:
            density = None
        
        longest_run_s = float(s_runs["duration_s"].max())
        longest_share = longest_run_s / total_af_s if total_af_s > 0 else 0.0
        
        brief_runs = s_runs[(s_runs["duration_s"] < 30.0) & (s_runs["is_complete"] == True)]
        n_brief = len(brief_runs)
        brief_af_s = float(brief_runs["duration_s"].sum())
        brief_share = brief_af_s / total_af_s if total_af_s > 0 else 0.0
        
        prol_runs = s_runs[s_runs["duration_s"] >= 1800.0]
        n_prolonged = len(prol_runs)
        
        rr_post_remote = rate_dict.get((ds, pid), np.nan)
        
        # Architecture categorization:
        if longest_share >= 0.70:
            arch_type = "predominant_prolonged"
        elif brief_share >= 0.30 or longest_share < 0.30:
            arch_type = "recurrent_short_clusters"
        else:
            arch_type = "mixed_prolonged_and_short"
            
        records.append({
            "dataset": ds,
            "patient_id": pid,
            "record_id": rec_id,
            "observation_duration_h": rec_dur / 3600.0,
            "af_burden": burden,
            "continuous_density": density,
            "longest_episode_h": longest_run_s / 3600.0,
            "longest_episode_share": longest_share,
            "n_brief_runs": n_brief,
            "brief_run_exposure_share": brief_share,
            "n_prolonged_runs": n_prolonged,
            "architecture_type": arch_type,
            "post_vs_remote_rate_ratio": rr_post_remote
        })
        
    return pd.DataFrame(records)
