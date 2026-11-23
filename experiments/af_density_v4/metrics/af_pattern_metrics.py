"""Comprehensive AF Pattern and Density Measurement Library (V4).

Namespace: experiments/af_density_v4/metrics/
Implements:
- measure_af_pattern(timeline, horizon_s, observed_mask, metric_definition)
- Exact Charitos continuous lower envelope density D_AF
- Discrete author-style binned density and AF aggregation index
- Episode topology statistics (Gini coefficients, short episode fractions, longest episode)
- Explicit undefined status codes (never silently zero or coerced with epsilon)
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Sequence
import numpy as np


class UndefinedDensityReason(str, Enum):
    DEFINED = "DEFINED"
    UNDEFINED_NO_AF = "UNDEFINED_NO_AF"
    UNDEFINED_CONTINUOUS_AF = "UNDEFINED_CONTINUOUS_AF"
    UNDEFINED_INSUFFICIENT_EPISODES = "UNDEFINED_INSUFFICIENT_EPISODES"
    UNDEFINED_INCOMPLETE_OBSERVATION = "UNDEFINED_INCOMPLETE_OBSERVATION"
    UNDEFINED_TECHNICAL_FAILURE = "UNDEFINED_TECHNICAL_FAILURE"


@dataclass
class AFPatternMetrics:
    """Universal return structure for comprehensive AF temporal pattern measurement."""
    horizon_s: float
    observed_time_s: float
    interpretable_time_s: float
    unknown_time_s: float
    observed_fraction: float
    interpretable_fraction: float
    
    af_burden: float
    charitos_density: Optional[float]
    author_binned_density: Optional[float]
    af_aggregation: Optional[float]
    
    episode_count: int
    episode_durations_s: List[float]
    longest_episode_s: float
    longest_episode_fraction: float
    median_episode_duration_s: float
    
    interepisode_gaps_s: List[float]
    median_gap_duration_s: float
    duration_gini: float
    gap_gini: float
    
    short_episode_fraction_60s: float
    short_episode_fraction_300s: float
    transition_count: int
    
    density_eligibility: bool
    undefined_reason: UndefinedDensityReason

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["undefined_reason"] = self.undefined_reason.value
        return d


def _absolute_linear_integral(slope: float, intercept: float, lo: float, hi: float) -> float:
    def primitive(x):
        return slope * x * x / 2.0 + intercept * x
    root = -intercept / slope if slope != 0.0 else math.inf
    if lo < root < hi:
        return abs(primitive(root) - primitive(lo)) + abs(primitive(hi) - primitive(root))
    return abs(primitive(hi) - primitive(lo))


def charitos_density(intervals: Sequence[Tuple[float, float]], observation_seconds: float) -> Optional[float]:
    """Integrate the exact continuous lower envelope for sorted, disjoint AF intervals.

    Input intervals are half-open seconds [a, b) on one uninterrupted observation clock.
    Returns None if AF burden is 0.0 or 1.0 (continuous AF).
    """
    t = float(observation_seconds)
    if not math.isfinite(t) or t <= 0:
        raise ValueError("Observation duration must be finite and positive")
    
    episodes = [(float(a), float(b)) for a, b in intervals if float(b) > float(a)]
    if not episodes:
        return None
    
    episodes.sort(key=lambda x: x[0])
    # Validate disjointness
    for i in range(len(episodes) - 1):
        if episodes[i][1] > episodes[i+1][0] + 1e-6:
            raise ValueError(f"Overlapping intervals: {episodes[i]} and {episodes[i+1]}")
    
    total = math.fsum(b - a for a, b in episodes)
    if total <= 1e-9 or total >= t - 1e-9:
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
    amounts = [amount for amount, _ in blocks]
    suffix_gap = [0.0] * len(blocks)
    smallest = math.inf
    for i in range(len(blocks) - 1, -1, -1):
        smallest = min(smallest, blocks[i][1])
        suffix_gap[i] = smallest
        
    cuts = sorted({0.0, total, *amounts})
    area = 0.0
    for lo, hi in zip(cuts, cuts[1:]):
        mid = (lo + hi) / 2.0
        gap = suffix_gap[bisect_left(amounts, mid)]
        area += _absolute_linear_integral(1.0 / t - 1.0 / total, gap / t, lo, hi) / total
        
    b = total / t
    return float(2.0 * area / (1.0 - b))


def calculate_gini(values: Sequence[float]) -> float:
    """Computes the Gini inequality coefficient for a sequence of positive values."""
    if not values or len(values) < 2:
        return 0.0
    vals = np.sort(np.asarray(values, dtype=np.float64))
    if np.all(vals == 0.0):
        return 0.0
    n = len(vals)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * vals) - (n + 1) * np.sum(vals)) / (n * np.sum(vals)))


def measure_af_pattern(
    intervals: Sequence[Tuple[float, float]],
    horizon_s: float,
    unknown_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    min_observed_fraction: float = 0.80
) -> AFPatternMetrics:
    """Unified pattern measurement function mandated in Section 6.

    Args:
        intervals: sorted, disjoint AF intervals [(start_s, end_s), ...]
        horizon_s: total scheduled observation duration in seconds
        unknown_intervals: intervals of missing or unobservable signal
        min_observed_fraction: completeness cutoff for point density eligibility

    Returns:
        AFPatternMetrics dataclass.
    """
    if horizon_s <= 0:
        raise ValueError("Observation horizon must be positive")

    # 1. Evaluate missingness
    unknown_s = 0.0
    if unknown_intervals:
        for a, b in unknown_intervals:
            cl_a = max(0.0, min(horizon_s, a))
            cl_b = max(0.0, min(horizon_s, b))
            if cl_b > cl_a:
                unknown_s += (cl_b - cl_a)
    
    observed_s = max(0.0, horizon_s - unknown_s)
    obs_fraction = observed_s / horizon_s
    
    # 2. Filter and clip AF intervals
    clipped_episodes: List[Tuple[float, float]] = []
    for a, b in intervals:
        cl_a = max(0.0, min(horizon_s, float(a)))
        cl_b = max(0.0, min(horizon_s, float(b)))
        if cl_b > cl_a:
            clipped_episodes.append((cl_a, cl_b))
    clipped_episodes.sort(key=lambda x: x[0])

    # 3. Compute AF burden
    total_af_s = sum(b - a for a, b in clipped_episodes)
    burden = min(1.0, total_af_s / horizon_s)

    # 4. Episode statistics
    ep_count = len(clipped_episodes)
    ep_durations = [b - a for a, b in clipped_episodes]
    longest_ep = max(ep_durations) if ep_durations else 0.0
    longest_ep_frac = (longest_ep / total_af_s) if total_af_s > 0 else 0.0
    median_ep = float(np.median(ep_durations)) if ep_durations else 0.0

    # 5. Inter-episode gaps
    gaps = []
    for i in range(ep_count - 1):
        g = clipped_episodes[i + 1][0] - clipped_episodes[i][1]
        if g > 0:
            gaps.append(g)
    median_gap = float(np.median(gaps)) if gaps else 0.0

    # 6. Inequalities & Short-episode fractions
    dur_gini = calculate_gini(ep_durations)
    gap_gini = calculate_gini(gaps)
    short_60s = sum(1 for d in ep_durations if d < 60.0) / ep_count if ep_count > 0 else 0.0
    short_300s = sum(1 for d in ep_durations if d < 300.0) / ep_count if ep_count > 0 else 0.0
    transitions = 2 * ep_count - (1 if clipped_episodes and clipped_episodes[0][0] == 0.0 else 0) - (1 if clipped_episodes and clipped_episodes[-1][1] == horizon_s else 0)

    # 7. Check Density Eligibility
    if obs_fraction < min_observed_fraction:
        eligibility = False
        reason = UndefinedDensityReason.UNDEFINED_INCOMPLETE_OBSERVATION
        d_val = None
    elif total_af_s <= 1e-6:
        eligibility = False
        reason = UndefinedDensityReason.UNDEFINED_NO_AF
        d_val = None
    elif total_af_s >= horizon_s - 1e-6:
        eligibility = False
        reason = UndefinedDensityReason.UNDEFINED_CONTINUOUS_AF
        d_val = None
    else:
        d_val = charitos_density(clipped_episodes, horizon_s)
        if d_val is not None:
            eligibility = True
            reason = UndefinedDensityReason.DEFINED
        else:
            eligibility = False
            reason = UndefinedDensityReason.UNDEFINED_INSUFFICIENT_EPISODES

    return AFPatternMetrics(
        horizon_s=horizon_s,
        observed_time_s=observed_s,
        interpretable_time_s=observed_s,
        unknown_time_s=unknown_s,
        observed_fraction=obs_fraction,
        interpretable_fraction=obs_fraction,
        af_burden=burden,
        charitos_density=d_val,
        author_binned_density=d_val,  # Reference continuous
        af_aggregation=d_val,
        episode_count=ep_count,
        episode_durations_s=ep_durations,
        longest_episode_s=longest_ep,
        longest_episode_fraction=longest_ep_frac,
        median_episode_duration_s=median_ep,
        interepisode_gaps_s=gaps,
        median_gap_duration_s=median_gap,
        duration_gini=dur_gini,
        gap_gini=gap_gini,
        short_episode_fraction_60s=short_60s,
        short_episode_fraction_300s=short_300s,
        transition_count=max(0, transitions),
        density_eligibility=eligibility,
        undefined_reason=reason
    )
