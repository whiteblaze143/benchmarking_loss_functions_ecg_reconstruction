"""Published discrete pattern metrics on one shared released-RR clock.

This is an evaluation projection, never a classifier feature source. Cumulative
released RR timing is approximate relative to the ECG sample clock; labels must
come from the ECG reference, not from the separate released RR label table.
"""
import numpy as np
from metrics import discrete_pattern_from_runs
from temporal_baselines import timeline_intervals


def released_rr_starts(rr_milliseconds, sampling_rate):
    rr = np.asarray(rr_milliseconds, dtype=np.float64)
    if rr.ndim != 1 or not len(rr) or not np.isfinite(rr).all() or (rr < 0).any():
        raise ValueError('Expected finite nonnegative released RR intervals')
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError('Invalid sampling rate')
    starts = np.r_[0., np.cumsum(rr[:-1])]*sampling_rate/1000.
    # A zero-duration RR observation has no distinct elapsed-time support.
    keep = rr > 0
    return starts[keep], int((~keep).sum())


def _runs(binary):
    starts = np.flatnonzero(binary & ~np.r_[False, binary[:-1]])
    stops = np.flatnonzero(binary & ~np.r_[binary[1:], False])+1
    return list(zip(starts.tolist(), stops.tolist()))


def evaluate_shared_beat_clock(reference_samples, states, assigned_start, assigned_end,
                               sampling_rate, rr_milliseconds):
    """Project both timelines onto identical RR-start observations.

    Every physical unknown interval makes the predicted endpoint indeterminate,
    even if it contains no RR observation. No interpolation, ECG/RR alignment
    correction, or compression is performed. No-AF/full-AF remain undefined under
    the published discrete metric's declared domain.
    """
    timeline_intervals(states, assigned_start, assigned_end, sampling_rate)
    starts, ends = np.asarray(assigned_start), np.asarray(assigned_end)
    base, stop = int(starts[0]), int(ends[-1])
    clock, zeros = released_rr_starts(rr_milliseconds, sampling_rate)
    clock = clock[(clock >= base) & (clock < stop)]
    reference = np.zeros(len(clock), bool)
    episode_counts = [];previous = -1
    for a, b in reference_samples:
        if not 0 <= a < b or a <= previous:
            raise ValueError('Expected sorted separated source AF intervals')
        previous = b
        if b <= base or a >= stop: continue
        hit = (clock >= a) & (clock < b)
        reference |= hit;episode_counts.append(int(hit.sum()))
    result = {'clock': 'released_positive_RR_start_observations',
              'clock_observations': len(clock), 'excluded_zero_RR_observations_in_source_file': zeros,
              'reference_episodes_with_no_clock_observation': sum(n == 0 for n in episode_counts),
              'reference_projected_runs': len(_runs(reference)),
              'reference': None, 'predicted': None, 'absolute_errors': None,
              'status': 'no_clock_observations' if not len(clock) else 'complete',
              'limitations': 'Released cumulative RR timing is an approximate ECG clock; positive long intervals are retained without asserting clinical beat validity. These are beat-index endpoints, not elapsed-time endpoints.'}
    if not len(clock): return result
    result['reference'] = discrete_pattern_from_runs(_runs(reference), len(clock))
    if (np.asarray(states) == -1).any():
        result['status'] = 'indeterminate_unknown_physical_time'
        return result
    indices = np.searchsorted(ends, clock, side='right')
    predicted = np.asarray(states)[indices] == 1
    result['predicted'] = discrete_pattern_from_runs(_runs(predicted), len(clock))
    result['absolute_errors'] = {name: None if result['reference'][name] is None or result['predicted'][name] is None
                                else abs(result['reference'][name]-result['predicted'][name])
                                for name in ['density', 'aggregation']}
    return result
