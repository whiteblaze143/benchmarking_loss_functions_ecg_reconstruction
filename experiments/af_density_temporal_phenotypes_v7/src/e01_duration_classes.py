"""E01: Duration class classification and patient feasibility inventory.
Assigns native AF runs into canonical duration classes (S0-S3, M1, L1-L3)
and evaluates patient-level feasibility across LTAFDB and SHDB.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def assign_duration_class(duration_s: float, is_complete: bool = True) -> str:
    """Assigns an AF episode to its canonical duration class.
    
    Classes:
      S0: < 10s
      S1: 10s - < 30s
      S2: 30s - < 2m (120s)
      S3: 2m - < 6m (360s)
      M1: 6m - < 30m (1800s)
      L1: 30m - < 1h (3600s)
      L2: 1h - < 6h (21600s)
      L3: >= 6h (>= 21600s)
      
    If a run is censored and observed duration < 30s, it cannot be certified as brief.
    """
    if duration_s < 10.0:
        return "S0"
    elif duration_s < 30.0:
        return "S1"
    elif duration_s < 120.0:
        return "S2"
    elif duration_s < 360.0:
        return "S3"
    elif duration_s < 1800.0:
        return "M1"
    elif duration_s < 3600.0:
        return "L1"
    elif duration_s < 21600.0:
        return "L2"
    else:
        return "L3"


def process_runs_with_classes(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Augments canonical AF runs dataframe with duration class and brief/prolonged flags."""
    df = runs_df.copy()
    df["duration_class"] = df.apply(
        lambda r: assign_duration_class(r["duration_s"], r["is_complete"]), axis=1
    )
    
    # Primary definitions from E01.2:
    # Brief: 0 < duration < 30s, complete observation
    df["is_brief_run"] = (df["duration_s"] < 30.0) & (df["is_complete"] == True)
    
    # Primary index prolonged episode: duration >= 30m (1800s)
    df["is_prolonged_index"] = df["duration_s"] >= 1800.0
    
    # Secondary prolonged thresholds:
    df["is_prolonged_6m"] = df["duration_s"] >= 360.0
    df["is_prolonged_60m"] = df["duration_s"] >= 3600.0
    df["is_prolonged_6h"] = df["duration_s"] >= 21600.0
    
    return df


def build_duration_class_inventory(df_with_classes: pd.DataFrame) -> pd.DataFrame:
    """Produces the E01_duration_class_inventory.csv summary."""
    records = []
    
    datasets = ["LTAFDB", "SHDB-AF", "IRIDIA-AF", "ALL"]
    order = ["S0", "S1", "S2", "S3", "M1", "L1", "L2", "L3"]
    
    for ds in datasets:
        sub = df_with_classes if ds == "ALL" else df_with_classes[df_with_classes["dataset"] == ds]
        total_runs = len(sub)
        total_dur = sub["duration_s"].sum()
        
        for cls in order:
            csub = sub[sub["duration_class"] == cls]
            c_runs = len(csub)
            c_dur = csub["duration_s"].sum()
            records.append({
                "dataset": ds,
                "duration_class": cls,
                "n_episodes": c_runs,
                "episode_fraction": c_runs / total_runs if total_runs > 0 else 0.0,
                "total_duration_s": c_dur,
                "total_duration_h": c_dur / 3600.0,
                "duration_fraction": c_dur / total_dur if total_dur > 0 else 0.0,
                "mean_duration_s": csub["duration_s"].mean() if c_runs > 0 else 0.0,
                "median_duration_s": csub["duration_s"].median() if c_runs > 0 else 0.0,
                "min_duration_s": csub["duration_s"].min() if c_runs > 0 else 0.0,
                "max_duration_s": csub["duration_s"].max() if c_runs > 0 else 0.0,
                "n_complete": int(csub["is_complete"].sum()),
                "complete_fraction": csub["is_complete"].mean() if c_runs > 0 else 0.0
            })
            
    return pd.DataFrame(records)


def build_patient_feasibility(df_with_classes: pd.DataFrame) -> pd.DataFrame:
    """Produces the E01_patient_feasibility.csv matrix."""
    records = []
    
    for (ds, pid), group in df_with_classes.groupby(["dataset", "patient_id"]):
        n_total = len(group)
        n_brief = int(group["is_brief_run"].sum())
        n_prol = int(group["is_prolonged_index"].sum())
        
        prol_runs = group[group["is_prolonged_index"]]
        has_obs_onset = prol_runs["has_observed_onset"].any() if n_prol > 0 else False
        has_obs_term = prol_runs["has_observed_termination"].any() if n_prol > 0 else False
        
        class_counts = group["duration_class"].value_counts().to_dict()
        
        rec = {
            "dataset": ds,
            "patient_id": pid,
            "total_af_runs": n_total,
            "n_brief_runs": n_brief,
            "n_prolonged_index_runs": n_prol,
            "has_brief": n_brief >= 1,
            "has_prolonged": n_prol >= 1,
            "has_both": (n_brief >= 1) and (n_prol >= 1),
            "has_gte_3_brief": n_brief >= 3,
            "has_gte_5_brief": n_brief >= 5,
            "has_gte_2_prolonged": n_prol >= 2,
            "has_observed_index_onset": has_obs_onset,
            "has_observed_index_termination": has_obs_term,
            "total_af_duration_h": group["duration_s"].sum() / 3600.0,
            "longest_af_run_h": group["duration_s"].max() / 3600.0,
        }
        for cls in ["S0", "S1", "S2", "S3", "M1", "L1", "L2", "L3"]:
            rec[f"n_{cls}"] = class_counts.get(cls, 0)
            
        records.append(rec)
        
    df_p = pd.DataFrame(records)
    
    # Calculate patient concentration in brief runs
    for ds in ["LTAFDB", "SHDB-AF", "IRIDIA-AF"]:
        sub_mask = df_p["dataset"] == ds
        total_ds_brief = df_p.loc[sub_mask, "n_brief_runs"].sum()
        if total_ds_brief > 0:
            df_p.loc[sub_mask, "brief_run_share"] = df_p.loc[sub_mask, "n_brief_runs"] / total_ds_brief
        else:
            df_p.loc[sub_mask, "brief_run_share"] = 0.0
            
    return df_p
