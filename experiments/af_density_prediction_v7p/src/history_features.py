"""Extraction of leak-free historical AF architecture, density, and aggregation features (F0-F3)."""
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AF_DENSITY_DIR = PROJECT_ROOT / "experiments" / "af_density"
if str(AF_DENSITY_DIR) not in sys.path:
    sys.path.insert(0, str(AF_DENSITY_DIR))

try:
    from metrics import charitos_density
except ImportError:
    def charitos_density(intervals, observation_seconds):
        t = float(observation_seconds)
        if not math.isfinite(t) or t <= 0:
            return None
        episodes = [(float(a), float(b)) for a, b in intervals if b > a]
        if not episodes:
            return None
        total = math.fsum(b - a for a, b in episodes)
        if total <= 0 or total >= t:
            return None
        durations = [b - a for a, b in episodes]
        prefix = [0.0] + [math.fsum(durations[:i]) for i in range(1, len(durations) + 1)]
        blocks = []
        for i, (start, _) in enumerate(episodes):
            for j in range(i, len(episodes)):
                accumulated = durations[i] if i == j else prefix[j + 1] - prefix[i]
                gap = 0.0 if i == j else episodes[j][1] - start - accumulated
                blocks.append((accumulated, gap))
        blocks.sort()
        suffix_gap = [0.0] * len(blocks)
        smallest = math.inf
        for i in range(len(blocks) - 1, -1, -1):
            smallest = min(smallest, blocks[i][1])
            suffix_gap[i] = smallest
        cuts = sorted({0.0, total, *[b[0] for b in blocks]})
        integral = 0.0
        for i in range(len(cuts) - 1):
            x0, x1 = cuts[i], cuts[i + 1]
            if x1 <= x0:
                continue
            idx = np.searchsorted([b[0] for b in blocks], x0, side='left')
            g = suffix_gap[idx] if idx < len(blocks) else 0.0
            integral += 0.5 * (x1 - x0) * (2 * g + x0 + x1)
        return float(1.0 - (2.0 * integral) / (total * (2.0 * t - total)))


def compute_interval_features_at_time(
    past_runs: List[Tuple[float, float]],
    t: float,
    horizons: Dict[str, float] = {"15m": 900.0, "30m": 1800.0, "1h": 3600.0, "3h": 10800.0, "6h": 21600.0},
) -> Dict[str, float]:
    """Computes leak-free historical AF burden, counts, duration-classes, density and aggregation.
    
    past_runs: List of (start_s, end_s) for episodes occurring before or active at t.
    Episodes ongoing at t are clipped at t: (start_s, min(end_s, t)).
    """
    feats = {}
    clipped_runs = []
    for s, e in past_runs:
        if s < t:
            clipped_runs.append((s, min(e, t)))

    # Sort and merge if any overlap
    clipped_runs.sort()
    merged = []
    for s, e in clipped_runs:
        if e <= s:
            continue
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # F1: Immediately preceding episode and gap
    completed_runs = [r for r in clipped_runs if r[1] < t]
    if completed_runs:
        last_s, last_e = completed_runs[-1]
        feats["last_af_duration_s"] = float(last_e - last_s)
        feats["time_since_last_af_s"] = float(t - last_e)
        feats["has_prior_af"] = 1.0
    else:
        feats["last_af_duration_s"] = 0.0
        feats["time_since_last_af_s"] = np.nan  # Strictly NaN when no prior AF episode exists (never fake 86400s)
        feats["has_prior_af"] = 0.0

    # Loop over horizons
    for h_name, h_sec in horizons.items():
        h_start = max(0.0, t - h_sec)
        h_duration = t - h_start
        if h_duration <= 0:
            continue

        # Intervals inside [h_start, t]
        h_intervals = []
        for s, e in merged:
            ws = max(s, h_start)
            we = min(e, t)
            if we > ws:
                h_intervals.append((ws, we))

        af_time = math.fsum(we - ws for ws, we in h_intervals)
        burden = af_time / h_duration if h_duration > 0 else 0.0
        n_episodes = len(h_intervals)
        longest_run = max([we - ws for ws, we in h_intervals], default=0.0)
        brief_runs = sum(1 for ws, we in h_intervals if (we - ws) < 30.0)

        feats[f"af_burden_{h_name}"] = float(burden)
        feats[f"episode_count_{h_name}"] = float(n_episodes)
        feats[f"longest_af_run_{h_name}"] = float(longest_run)
        feats[f"brief_run_count_{h_name}"] = float(brief_runs)

        # F3: Charitos density & aggregation
        if 0.0 < burden < 1.0 and len(h_intervals) >= 1:
            # Shift intervals to relative window clock [0, h_duration]
            rel_intervals = [(ws - h_start, we - h_start) for ws, we in h_intervals]
            try:
                d = charitos_density(rel_intervals, h_duration)
                feats[f"charitos_density_{h_name}"] = float(d) if d is not None else np.nan
                feats[f"is_density_defined_{h_name}"] = 1.0 if d is not None else 0.0
            except Exception:
                feats[f"charitos_density_{h_name}"] = np.nan
                feats[f"is_density_defined_{h_name}"] = 0.0
            
            # Aggregation: s_max / total_af
            feats[f"af_aggregation_{h_name}"] = float(longest_run / af_time) if af_time > 0 else np.nan
        else:
            feats[f"charitos_density_{h_name}"] = np.nan  # Strictly NaN: mathematically undefined outside 0 < burden < 1
            feats[f"is_density_defined_{h_name}"] = 0.0
            feats[f"af_aggregation_{h_name}"] = np.nan

    return feats


def extract_features_for_landmarks_dataframe(
    manifest_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    runs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Computes F0-F3 features for every landmark row in manifest_df."""
    # Index runs by (dataset, record_id)
    runs_by_rec = {}
    for (ds, rid), sub in runs_df.groupby(["dataset", "record_id"]):
        runs_by_rec[(ds, rid)] = list(zip(sub["start_time_s"].values, sub["end_time_s"].values))

    feature_rows = []
    for row in manifest_df.itertuples():
        ds = row.dataset
        rid = row.record_id
        t = row.landmark_time_s

        rec_runs = runs_by_rec.get((ds, rid), [])
        past_runs = [r for r in rec_runs if r[0] < t]

        feats = compute_interval_features_at_time(past_runs, t)

        # F0: Minimal baseline features
        feats["time_since_rec_start_s"] = float(t)
        if hasattr(row, "elapsed_landmark_s"):
            feats["elapsed_af_duration_s"] = float(row.elapsed_landmark_s)
        elif hasattr(row, "landmark_elapsed_s"):
            feats["elapsed_af_duration_s"] = float(row.landmark_elapsed_s)
        else:
            feats["elapsed_af_duration_s"] = 0.0

        feature_rows.append(feats)

    feats_df = pd.DataFrame(feature_rows)
    return pd.concat([manifest_df.reset_index(drop=True), feats_df], axis=1)
