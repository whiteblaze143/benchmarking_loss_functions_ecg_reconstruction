"""D0/D1 on ordered windows; unknown states never silently become non-AF.

These functions do not select thresholds or smoothing parameters. Callers must
freeze those using development patients before test evaluation. D1 is a binary
window median followed by elapsed-time gap merging, not an HSMM.
"""
import math
import numpy as np


def threshold_states(probabilities, threshold):
    """D0: finite p >= threshold is AF; NaN is explicitly indeterminate (-1)."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not len(p) or np.isinf(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected nonempty probabilities in [0,1] or NaN')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid classification threshold')
    return np.where(np.isnan(p), -1, (p >= threshold).astype(np.int8)).astype(np.int8)


def _states(values):
    x = np.asarray(values)
    if x.ndim != 1 or not len(x) or x.dtype.kind not in 'iu' or not np.isin(x, [-1, 0, 1]).all():
        raise ValueError('Expected nonempty integer states -1/0/1')
    return x.astype(np.int8)


def filter_and_merge(states, assigned_seconds, *, median_width, max_gap_seconds):
    """D1 on one contiguous segment, retaining original window assignments.

    Odd-width median uses equal window weights and edge replication separately
    within each observed block. It is not a duration-weighted median. Merging
    fills only bounded non-AF runs whose summed assigned duration is <= the
    declared limit. Terminal non-AF runs and unknown windows are never filled.
    Call once per actual contiguous segment; never concatenate separate records.
    """
    x = _states(states)
    duration = np.asarray(assigned_seconds, dtype=np.float64)
    if duration.shape != x.shape or not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError('Expected one finite positive assigned duration per window')
    if not isinstance(median_width, int) or isinstance(median_width, bool) or median_width < 1 or median_width % 2 != 1:
        raise ValueError('Median width must be a positive odd integer')
    if not math.isfinite(max_gap_seconds) or max_gap_seconds < 0:
        raise ValueError('Invalid maximum merge-gap duration')
    result = x.copy()
    known = x != -1
    starts = np.flatnonzero(known & ~np.r_[False, known[:-1]])
    ends = np.flatnonzero(known & ~np.r_[known[1:], False])+1
    for a, b in zip(starts, ends):
        padded = np.pad(x[a:b], median_width//2, mode='edge')
        local = np.median(np.lib.stride_tricks.sliding_window_view(padded, median_width), axis=1).astype(np.int8)
        nonaf = local == 0
        gap_starts = np.flatnonzero(nonaf & ~np.r_[False, nonaf[:-1]])
        gap_ends = np.flatnonzero(nonaf & ~np.r_[nonaf[1:], False])+1
        for lo, hi in zip(gap_starts, gap_ends):
            if lo > 0 and hi < len(local) and duration[a+lo:a+hi].sum() <= max_gap_seconds:
                local[lo:hi] = 1
        result[a:b] = local
    return result


def timeline_intervals(states, assigned_start, assigned_end, sampling_rate):
    """Return AF and unknown half-open seconds, retaining the source clock.

    Assignments must cover a single contiguous segment exactly once. Inputs may
    overlap during feature extraction; assignments themselves may not overlap.
    No unknown intervals are imputed. Full-clock density needs an explicit gap
    policy whenever the returned unknown list is nonempty.
    """
    x = _states(states)
    starts = np.asarray(assigned_start)
    ends = np.asarray(assigned_end)
    if (starts.shape != x.shape or ends.shape != x.shape or
            starts.dtype.kind not in 'iu' or ends.dtype.kind not in 'iu' or
            (starts < 0).any() or (ends <= starts).any() or
            not np.array_equal(ends[:-1], starts[1:])):
        raise ValueError('Expected contiguous nonoverlapping integer sample assignments')
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError('Invalid sampling rate')
    output = {'af': [], 'unknown': [], 'start_seconds': float(starts[0]/sampling_rate),
              'end_seconds': float(ends[-1]/sampling_rate)}
    changes = np.r_[0, np.flatnonzero(x[1:] != x[:-1])+1, len(x)]
    for a, b in zip(changes[:-1], changes[1:]):
        if x[a] in [1, -1]:
            key = 'af' if x[a] == 1 else 'unknown'
            output[key].append((float(starts[a]/sampling_rate), float(ends[b-1]/sampling_rate)))
    return output
