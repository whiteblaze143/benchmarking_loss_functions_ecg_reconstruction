"""E01: Conditional Null Experiments (Null A and Null B).
Implements:
- Null A: Reposition brief runs conditional on fixed prolonged episodes.
- Null B: Permute episode-duration ordering while preserving duration and gap multisets.
Evaluates empirical p-values and z-scores over B=2,000 Monte Carlo realizations.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


def run_null_a_repositioning(
    patient_runs: pd.DataFrame,
    rec_duration_s: float,
    prolonged_runs: pd.DataFrame,
    brief_runs: pd.DataFrame,
    n_permutations: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """Null A: Uniformly repositions brief runs into eligible non-prolonged intervals.
    Preserves: observation duration, prolonged episode positions, brief run durations.
    """
    rng = np.random.RandomState(seed)
    
    # Prolonged intervals to exclude from placement: [s, e]
    prolonged_intervals = [(r["start_time_s"], r["end_time_s"]) for _, r in prolonged_runs.iterrows()]
    prolonged_intervals.sort()
    
    # Pre-onset [t_on - 900, t_on) and Post-termination [t_off, t_off + 900) target zones
    target_zones_pre = [(max(0.0, r["start_time_s"] - 900.0), r["start_time_s"]) for _, r in prolonged_runs.iterrows()]
    target_zones_post = [(r["end_time_s"], min(rec_duration_s, r["end_time_s"] + 900.0)) for _, r in prolonged_runs.iterrows()]
    
    # Eligible non-prolonged regions: complement of prolonged_intervals in [0, rec_duration_s]
    eligible_regions = []
    curr = 0.0
    for s, e in prolonged_intervals:
        if s > curr:
            eligible_regions.append((curr, s))
        curr = max(curr, e)
    if curr < rec_duration_s:
        eligible_regions.append((curr, rec_duration_s))
        
    region_lengths = np.array([max(0.0, e - s) for s, e in eligible_regions])
    total_eligible = region_lengths.sum()
    
    brief_durations = brief_runs["duration_s"].values
    n_brief = len(brief_durations)
    
    if total_eligible <= 0 or n_brief == 0:
        return {
            "null_type": "Null_A",
            "n_permutations": n_permutations,
            "obs_post_count": 0,
            "obs_pre_count": 0,
            "null_post_mean": 0.0,
            "null_post_std": 0.0,
            "null_pre_mean": 0.0,
            "null_pre_std": 0.0,
            "p_value_post": 1.0,
            "z_score_post": 0.0,
            "p_value_pre": 1.0,
            "z_score_pre": 0.0
        }
        
    region_probs = region_lengths / total_eligible
    
    # Observed counts in target zones
    obs_pre = 0
    obs_post = 0
    for _, b in brief_runs.iterrows():
        t = b["start_time_s"]
        if any(zs <= t < ze for zs, ze in target_zones_pre):
            obs_pre += 1
        if any(zs <= t < ze for zs, ze in target_zones_post):
            obs_post += 1
            
    # Simulate Null A
    null_pre_counts = np.zeros(n_permutations, dtype=int)
    null_post_counts = np.zeros(n_permutations, dtype=int)
    
    for b in range(n_permutations):
        # Sample region indices for each brief run
        chosen_reg_idx = rng.choice(len(eligible_regions), size=n_brief, p=region_probs)
        offsets = rng.uniform(0.0, 1.0, size=n_brief)
        
        sim_starts = np.array([
            eligible_regions[idx][0] + offsets[i] * max(0.0, (eligible_regions[idx][1] - eligible_regions[idx][0] - brief_durations[i]))
            for i, idx in enumerate(chosen_reg_idx)
        ])
        
        c_pre = sum(any(zs <= st < ze for zs, ze in target_zones_pre) for st in sim_starts)
        c_post = sum(any(zs <= st < ze for zs, ze in target_zones_post) for st in sim_starts)
        
        null_pre_counts[b] = c_pre
        null_post_counts[b] = c_post
        
    p_val_post = float(np.mean(null_post_counts >= obs_post))
    p_val_pre = float(np.mean(null_pre_counts >= obs_pre))
    
    z_post = (obs_post - null_post_counts.mean()) / (null_post_counts.std() + 1e-6)
    z_pre = (obs_pre - null_pre_counts.mean()) / (null_pre_counts.std() + 1e-6)
    
    return {
        "null_type": "Null_A",
        "n_permutations": n_permutations,
        "obs_post_count": obs_post,
        "obs_pre_count": obs_pre,
        "null_post_mean": float(null_post_counts.mean()),
        "null_post_std": float(null_post_counts.std()),
        "null_pre_mean": float(null_pre_counts.mean()),
        "null_pre_std": float(null_pre_counts.std()),
        "p_value_post": p_val_post,
        "z_score_post": float(z_post),
        "p_value_pre": p_val_pre,
        "z_score_pre": float(z_pre)
    }


def run_null_b_ordering(
    patient_runs: pd.DataFrame,
    rec_duration_s: float,
    n_permutations: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """Null B: Randomly permutes episode ordering while preserving episode durations and gap multiset.
    Preserves: episode durations, inter-episode gaps, total observation duration.
    """
    rng = np.random.RandomState(seed)
    
    sorted_runs = patient_runs.sort_values("start_time_s").reset_index(drop=True)
    n_runs = len(sorted_runs)
    
    if n_runs < 3:
        return {
            "null_type": "Null_B",
            "n_permutations": n_permutations,
            "obs_post_brief": 0,
            "obs_pre_brief": 0,
            "null_post_mean": 0.0,
            "null_post_std": 0.0,
            "p_value_ordering": 1.0,
            "z_score_ordering": 0.0,
            "status": "insufficient_episodes"
        }
        
    durations = sorted_runs["duration_s"].values
    is_prol = durations >= 1800.0
    is_brief = (durations < 30.0) & (sorted_runs["is_complete"].values == True)
    
    # Calculate gaps between episodes
    starts = sorted_runs["start_time_s"].values
    ends = sorted_runs["end_time_s"].values
    
    gaps = [starts[0]] # Leading gap
    for i in range(1, n_runs):
        gaps.append(max(0.0, starts[i] - ends[i-1]))
    gaps.append(max(0.0, rec_duration_s - ends[-1])) # Trailing gap
    
    # Observed brief runs immediately following or preceding prolonged runs
    obs_post = 0
    obs_pre = 0
    for i in range(n_runs):
        if is_prol[i]:
            # Next episode
            if i + 1 < n_runs and is_brief[i + 1] and gaps[i + 1] <= 900.0:
                obs_post += 1
            # Previous episode
            if i - 1 >= 0 and is_brief[i - 1] and gaps[i] <= 900.0:
                obs_pre += 1
                
    null_post_counts = np.zeros(n_permutations, dtype=int)
    null_pre_counts = np.zeros(n_permutations, dtype=int)
    
    # Permute episode order
    indices = np.arange(n_runs)
    for b in range(n_permutations):
        perm_idx = rng.permutation(indices)
        perm_prol = is_prol[perm_idx]
        perm_brief = is_brief[perm_idx]
        
        c_post = 0
        c_pre = 0
        for i in range(n_runs):
            if perm_prol[i]:
                if i + 1 < n_runs and perm_brief[i + 1] and gaps[i + 1] <= 900.0:
                    c_post += 1
                if i - 1 >= 0 and perm_brief[i - 1] and gaps[i] <= 900.0:
                    c_pre += 1
                    
        null_post_counts[b] = c_post
        null_pre_counts[b] = c_pre
        
    p_val_post = float(np.mean(null_post_counts >= obs_post))
    z_post = (obs_post - null_post_counts.mean()) / (null_post_counts.std() + 1e-6)
    
    return {
        "null_type": "Null_B",
        "n_permutations": n_permutations,
        "obs_post_brief": obs_post,
        "obs_pre_brief": obs_pre,
        "null_post_mean": float(null_post_counts.mean()),
        "null_post_std": float(null_post_counts.std()),
        "null_pre_mean": float(null_pre_counts.mean()),
        "null_pre_std": float(null_pre_counts.std()),
        "p_value_ordering": p_val_post,
        "z_score_ordering": float(z_post),
        "status": "converged"
    }


def execute_cohort_null_experiments(
    runs_df: pd.DataFrame,
    cohort_inv: pd.DataFrame,
    n_permutations: int = 2000,
    seed: int = 42
) -> pd.DataFrame:
    """Executes Null A and Null B across all informative patients with both prolonged and brief AF."""
    rec_durs = cohort_inv.set_index(["dataset", "record_id"])["duration_s"].to_dict()
    
    records = []
    
    for (ds, pid), group in runs_df.groupby(["dataset", "patient_id"]):
        if ds not in ["LTAFDB", "SHDB-AF"]:
            continue
            
        prol = group[group["duration_s"] >= 1800.0]
        brief = group[(group["duration_s"] < 30.0) & (group["is_complete"] == True)]
        
        if len(prol) == 0 or len(brief) == 0:
            continue
            
        rec_id = str(group["record_id"].iloc[0])
        rec_dur = rec_durs.get((ds, rec_id), group["end_time_s"].max() + 3600.0)
        
        # Null A
        res_a = run_null_a_repositioning(group, rec_dur, prol, brief, n_permutations=n_permutations, seed=seed)
        res_a["dataset"] = ds
        res_a["patient_id"] = pid
        res_a["record_id"] = rec_id
        res_a["n_prolonged"] = len(prol)
        res_a["n_brief"] = len(brief)
        records.append(res_a)
        
        # Null B
        res_b = run_null_b_ordering(group, rec_dur, n_permutations=n_permutations, seed=seed)
        res_b["dataset"] = ds
        res_b["patient_id"] = pid
        res_b["record_id"] = rec_id
        res_b["n_prolonged"] = len(prol)
        res_b["n_brief"] = len(brief)
        records.append(res_b)
        
    return pd.DataFrame(records)
