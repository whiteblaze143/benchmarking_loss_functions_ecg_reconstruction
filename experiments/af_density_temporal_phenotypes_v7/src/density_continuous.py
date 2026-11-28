"""Exact continuous-time Charitos AF density solver and minimum-interval locator.
Implements the continuous formulation:
  F(p; b) = 1/T * min_{s, e} { e - s : int_s^e A(t) dt >= p*M }
  D = 2/(1 - b) * int_0^1 |F(p; b) - p| dp
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from bisect import bisect_left
import numpy as np


def _absolute_linear_integral(slope: float, intercept: float, lo: float, hi: float) -> float:
    """Exact integral of |slope * x + intercept| from lo to hi."""
    def primitive(x: float) -> float:
        return slope * x * x / 2.0 + intercept * x

    root = -intercept / slope if slope != 0 else math.inf
    if lo < root < hi:
        return abs(primitive(root) - primitive(lo)) + abs(primitive(hi) - primitive(root))
    return abs(primitive(hi) - primitive(lo))


def compute_continuous_density(
    intervals: List[Tuple[float, float]],
    observation_seconds: float
) -> Optional[float]:
    """Exact continuous-time Charitos AF density for sorted, disjoint half-open intervals [start, end).
    Returns None if burden == 0 or burden == 1 (undefined).
    """
    T = float(observation_seconds)
    if not math.isfinite(T) or T <= 0:
        raise ValueError("Observation duration must be finite and positive")

    episodes = [(float(a), float(b)) for a, b in intervals if b > a]
    if not episodes:
        return None

    tol = 1e-6
    prev_end = 0.0
    clean_episodes = []
    for idx, (a, b) in enumerate(episodes):
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(f"Non-finite interval [{a}, {b})")
        if a < -tol or b > T + tol or a >= b:
            raise ValueError(f"Invalid interval [{a}, {b}) outside [0, {T}]")
        if idx > 0 and a < prev_end - tol:
            raise ValueError(f"Overlapping or unsorted intervals: [{a}, {b}) follows {prev_end}")

        a_clamped = max(0.0, a)
        b_clamped = min(float(T), b)
        clean_episodes.append((a_clamped, b_clamped))
        prev_end = b_clamped

    episodes = clean_episodes

    M = math.fsum(b - a for a, b in episodes)
    burden = M / T
    if burden <= 0.0 or burden >= 1.0 or math.isclose(burden, 1.0, abs_tol=1e-9):
        return None

    # Build all episode-block spans
    durations = [b - a for a, b in episodes]
    prefix_m = [0.0] + [math.fsum(durations[:i]) for i in range(1, len(durations) + 1)]

    blocks = []
    for i, (start_i, _) in enumerate(episodes):
        for j in range(i, len(episodes)):
            accumulated = durations[i] if i == j else prefix_m[j + 1] - prefix_m[i]
            gap = 0.0 if i == j else episodes[j][1] - start_i - accumulated
            blocks.append((accumulated, gap))

    blocks.sort()
    amounts = [amt for amt, _ in blocks]
    suffix_gap = [0.0] * len(blocks)
    smallest = math.inf
    for i in range(len(blocks) - 1, -1, -1):
        smallest = min(smallest, blocks[i][1])
        suffix_gap[i] = smallest

    cuts = sorted({0.0, M, *amounts})
    area = 0.0
    for lo, hi in zip(cuts, cuts[1:]):
        mid = (lo + hi) / 2.0
        gap = suffix_gap[bisect_left(amounts, mid)]
        # Slope = 1/T - 1/M, intercept = gap/T
        area += _absolute_linear_integral(1.0 / T - 1.0 / M, gap / T, lo, hi) / M

    return 2.0 * area / (1.0 - burden)


def find_shortest_interval_for_fraction(
    intervals: List[Tuple[float, float]],
    observation_seconds: float,
    p: float
) -> Tuple[float, float, float]:
    """Finds the shortest contiguous physical interval [s, e] containing at least fraction p of total AF.
    Returns: (length, start_s, end_s).
    Deterministic tie-breaking: earliest start_s.
    """
    T = float(observation_seconds)
    if not (0.0 <= p <= 1.0):
        raise ValueError("Coverage fraction p must be in [0, 1]")

    episodes = [(float(a), float(b)) for a, b in intervals if b > a]
    if not episodes:
        return (0.0, 0.0, 0.0)

    M = math.fsum(b - a for a, b in episodes)
    target_mass = p * M
    if target_mass <= 1e-12:
        return (0.0, 0.0, 0.0)

    durations = [b - a for a, b in episodes]
    max_single_dur = max(durations)

    # If target_mass fits entirely inside a single episode, the minimum length is exactly target_mass!
    if target_mass <= max_single_dur:
        # Find the earliest episode that can contain target_mass
        for a, b in episodes:
            if (b - a) >= target_mass:
                return (target_mass, a, a + target_mass)

    # General search across episode block boundaries
    # A candidate minimum interval starts at episode i's start s_i, or ends at episode j's end e_j
    best_len = math.inf
    best_s = 0.0
    best_e = T

    n_eps = len(episodes)
    prefix_m = [0.0] + list(np.cumsum(durations))

    # 1. Candidate intervals starting at an episode start
    for i in range(n_eps):
        s_candidate = episodes[i][0]
        # We need cumulative mass from s_candidate onwards to reach target_mass
        # Find ending point e
        accum = 0.0
        for j in range(i, n_eps):
            dur_j = episodes[j][1] - episodes[j][0]
            if accum + dur_j >= target_mass - 1e-12:
                needed = target_mass - accum
                e_candidate = episodes[j][0] + needed
                cand_len = e_candidate - s_candidate
                if cand_len < best_len - 1e-9:
                    best_len = cand_len
                    best_s = s_candidate
                    best_e = e_candidate
                break
            accum += dur_j

    # 2. Candidate intervals ending at an episode end
    for j in range(n_eps):
        e_candidate = episodes[j][1]
        accum = 0.0
        for i in range(j, -1, -1):
            dur_i = episodes[i][1] - episodes[i][0]
            if accum + dur_i >= target_mass - 1e-12:
                needed = target_mass - accum
                s_candidate = episodes[i][1] - needed
                cand_len = e_candidate - s_candidate
                if cand_len < best_len - 1e-9:
                    best_len = cand_len
                    best_s = s_candidate
                    best_e = e_candidate
                break
            accum += dur_i

    return (best_len, best_s, best_e)
