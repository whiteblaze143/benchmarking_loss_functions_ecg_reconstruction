"""Discrete pattern endpoints on an explicitly supplied common sample clock."""
import numpy as np
from metrics import discrete_pattern_from_runs
from temporal_baselines import timeline_intervals


def _runs(x):
    return list(zip(np.flatnonzero(x & ~np.r_[False, x[:-1]]).tolist(),
                    (np.flatnonzero(x & ~np.r_[x[1:], False])+1).tolist()))


def compress_sample_clock(clock_samples, clock_map):
    clock = np.asarray(clock_samples, dtype=np.float64)
    if clock.ndim != 1 or not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
        raise ValueError('Expected strictly increasing finite source-clock positions')
    projected = [clock[(clock >= a) & (clock < b)]-a+c for a, b, c in clock_map]
    result = np.concatenate(projected)
    assert (np.diff(result) > 0).all()
    return result


def evaluate_sample_clock(reference_samples, states, starts, ends, sampling_rate, clock_samples):
    timeline_intervals(states, starts, ends, sampling_rate)
    clock = np.asarray(clock_samples, dtype=np.float64)
    if clock.ndim != 1 or not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
        raise ValueError('Expected strictly increasing finite evaluation positions')
    clock = clock[(clock >= starts[0]) & (clock < ends[-1])]
    result = {'clock_observations': len(clock), 'reference': None, 'predicted': None, 'absolute_errors': None,
              'status': 'complete' if len(clock) else 'no_clock_observations'}
    truth = np.zeros(len(clock), bool);previous = -1
    for a, b in reference_samples:
        if not 0 <= a < b or a <= previous: raise ValueError('Expected separated source AF intervals')
        previous = b;truth |= (clock >= a) & (clock < b)
    if not len(clock): return result
    result['reference'] = discrete_pattern_from_runs(_runs(truth), len(clock))
    if (np.asarray(states) == -1).any():
        result['status'] = 'indeterminate_unknown_physical_time';return result
    prediction = np.asarray(states)[np.searchsorted(ends, clock, side='right')] == 1
    result['predicted'] = discrete_pattern_from_runs(_runs(prediction), len(clock))
    result['absolute_errors'] = {k: None if result['reference'][k] is None or result['predicted'][k] is None
                                else abs(result['reference'][k]-result['predicted'][k]) for k in ['density', 'aggregation']}
    return result
