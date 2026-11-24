"""Conditional randomization models for AF temporal localization (Experiment E03) and episode order (Phase G).
Implements:
  - Null A: Circular shifts on cumulative AF-time clock
  - Null B: Within-episode rate order shuffle / block permutation
  - Episode-order and gap-order permutations
"""

import random
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


def generate_null_a_circular_shifts(
    rate_intervals: List[Dict[str, Any]],
    target_interval_start_s: float,
    target_interval_end_s: float,
    n_permutations: int = 2000,
    seed: int = 42
) -> List[float]:
    """Null A: Shifts rate observations along the cumulative AF clock using random circular shifts.
    Returns: List of captured rapid AF fraction C_rapid(p) across permutations.
    """
    if not rate_intervals:
        return [0.0] * n_permutations

    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rate_intervals).sort_values("start_time_s").reset_index(drop=True)
    n = len(df)
    if n <= 1:
        return [0.0] * n_permutations

    # Mark which 30s bins fall within target interval [target_start, target_end]
    in_target = (df["start_time_s"] >= target_interval_start_s) & (df["end_time_s"] <= target_interval_end_s)
    target_indices = np.where(in_target)[0]

    rapid_indicators = df["is_rapid_120"].values.astype(float)
    total_rapid = float(np.sum(rapid_indicators))
    if total_rapid <= 0:
        return [0.0] * n_permutations

    captured_fractions = []
    # Generate circular shifts
    shifts = rng.randint(1, n, size=n_permutations)
    for shift in shifts:
        shifted_rapid = np.roll(rapid_indicators, shift)
        rapid_in_interval = np.sum(shifted_rapid[target_indices])
        captured_fractions.append(float(rapid_in_interval / total_rapid))

    return captured_fractions


def generate_null_b_within_episode_permutations(
    rate_intervals: List[Dict[str, Any]],
    episode_boundaries: List[Tuple[float, float]],
    target_interval_start_s: float,
    target_interval_end_s: float,
    n_permutations: int = 2000,
    seed: int = 42
) -> List[float]:
    """Null B: Preserves each episode's duration and total rapid AF, randomizing only rate order within episodes.
    Returns: List of captured rapid AF fraction C_rapid(p) across permutations.
    """
    if not rate_intervals or not episode_boundaries:
        return [0.0] * n_permutations

    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rate_intervals).sort_values("start_time_s").reset_index(drop=True)
    total_rapid = float(df["is_rapid_120"].sum())
    if total_rapid <= 0:
        return [0.0] * n_permutations

    # Group bins by episode
    episode_bin_indices = []
    for es, ee in episode_boundaries:
        bin_idx = df[(df["start_time_s"] >= es) & (df["end_time_s"] <= ee)].index.values
        if len(bin_idx) > 0:
            episode_bin_indices.append(bin_idx)

    in_target = (df["start_time_s"] >= target_interval_start_s) & (df["end_time_s"] <= target_interval_end_s)
    target_mask = in_target.values

    captured_fractions = []
    rapid_arr = df["is_rapid_120"].values.astype(float)

    for _ in range(n_permutations):
        perm_rapid = rapid_arr.copy()
        for b_indices in episode_bin_indices:
            if len(b_indices) > 1:
                perm_rapid[b_indices] = rng.permutation(perm_rapid[b_indices])
        rapid_in_interval = np.sum(perm_rapid[target_mask])
        captured_fractions.append(float(rapid_in_interval / total_rapid))

    return captured_fractions


def permute_episode_order(
    episodes: List[Tuple[float, float]],
    total_duration_s: float,
    seed: int = 42
) -> List[Tuple[float, float]]:
    """Phase G: Permutes episode sequence and internal gap sequence while preserving exact durations and bounds."""
    if len(episodes) <= 1:
        return episodes

    rng = random.Random(seed)
    durations = [e - s for s, e in episodes]
    gaps = [episodes[i][0] - episodes[i - 1][1] for i in range(1, len(episodes))]

    # Leading and trailing gaps
    lead_gap = episodes[0][0]
    trail_gap = total_duration_s - episodes[-1][1]

    # Permute internal episodes and internal gaps
    perm_durs = durations.copy()
    perm_gaps = gaps.copy()
    rng.shuffle(perm_durs)
    rng.shuffle(perm_gaps)

    # Reconstruct timeline
    new_episodes = []
    curr_t = lead_gap
    for i in range(len(perm_durs)):
        start = curr_t
        end = start + perm_durs[i]
        new_episodes.append((start, end))
        if i < len(perm_gaps):
            curr_t = end + perm_gaps[i]
        else:
            curr_t = end

    return new_episodes
