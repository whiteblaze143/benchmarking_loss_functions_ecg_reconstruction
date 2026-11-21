"""Common named elapsed-time endpoints, with explicit unknown denominators.

No threshold selection, imputation, quality exclusion or gap compression occurs
here. Each call is one contiguous segment, never an entire discontinuous record.
"""
import math
import numpy as np
from episode_metrics import episode_errors
from temporal_baselines import timeline_intervals
from metrics import charitos_density, charitos_author_binned_port, discrete_pattern_from_runs


def equal_bin_amounts(intervals, duration, maximum_bin_seconds):
    """AF fractions on equal bins covering the entire duration, including tail.

    The bin count is ceil(duration/maximum_bin_seconds), so actual bins are at
    most the requested size. Edges may fall between source samples; intervals
    are integrated continuously. Roundoff-only clipping is limited to 1e-10.
    """
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(maximum_bin_seconds) or maximum_bin_seconds <= 0:
        raise ValueError('Invalid observation or bin duration')
    count = math.ceil(duration/maximum_bin_seconds)
    edges = np.linspace(0, duration, count+1)
    amount = np.zeros(count)
    previous = -1.
    for a, b in intervals:
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration and a > previous):
            raise ValueError('Expected sorted separated intervals within observation')
        amount += np.maximum(0, np.minimum(edges[1:], b)-np.maximum(edges[:-1], a))
        previous = b
    fractions = amount/np.diff(edges)
    if not ((fractions >= -1e-10) & (fractions <= 1+1e-10)).all():
        raise ValueError('AF amount exceeds bin capacity')
    return np.clip(fractions, 0, 1)


def _runs(binary):
    starts = np.flatnonzero(binary & ~np.r_[False, binary[:-1]])
    ends = np.flatnonzero(binary & ~np.r_[binary[1:], False])+1
    return list(zip(starts.tolist(), ends.tolist()))


def pattern_endpoints(intervals, duration):
    """Keep continuous, author-binned and time-bin analogue metrics distinct."""
    author_amounts = equal_bin_amounts(intervals, duration, 30.)
    count = math.ceil(duration)
    centers = (np.arange(count)+.5)*duration/count
    binary = np.zeros(count, bool)
    for a, b in intervals:
        binary |= (centers >= a) & (centers < b)
    discrete = discrete_pattern_from_runs(_runs(binary), count)
    actual_af = math.fsum(b-a for a, b in intervals)
    return {
        'density_continuous': charitos_density(intervals, duration),
        'density_author_binned_30s_max': charitos_author_binned_port(author_amounts, timeunits=1., minim=.001, limit=True),
        'author_bin_count': len(author_amounts), 'author_bin_seconds': duration/len(author_amounts),
        'density_equal_timebin_analogue': discrete['density'],
        'aggregation_equal_timebin_analogue': discrete['aggregation'],
        'analogue_bin_count': count, 'analogue_bin_seconds': duration/count,
        'analogue_af_seconds_difference': float(binary.sum()*duration/count-actual_af),
        'limitations': 'Equal-time-bin analogues are not the published RR-beat-clock metrics. Author-binned metric retains author no-AF=0/full-AF=1 clamp; continuous and discrete edge values remain undefined.',
    }


def evaluate_elapsed_segment(reference_samples, states, assigned_start, assigned_end, sampling_rate):
    """Evaluate source sample-index reference against declared assigned states.

    Source reference is clipped to the supplied segment; onset/offset at segment
    boundaries is treated as censored for the additional boundary-error measure.
    Unknown positive time is a nondetection in all-reference sensitivity; unknown
    negative time is not a true negative. Complete-case sensitivity/specificity
    and unknown durations are reported separately. Any unknown time makes full
    predicted pattern/episode metrics indeterminate, never implicitly non-AF.
    """
    timeline_intervals(states, assigned_start, assigned_end, sampling_rate)
    base = int(np.asarray(assigned_start)[0]);stop = int(np.asarray(assigned_end)[-1])
    duration = (stop-base)/sampling_rate
    reference_samples_clipped = []
    previous = -1
    for a, b in reference_samples:
        if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)) or not 0 <= a < b or a <= previous:
            raise ValueError('Expected sorted separated integer reference sample intervals')
        previous = b
        if a < stop and b > base:
            reference_samples_clipped.append((int(max(a, base)-base), int(min(b, stop)-base)))
    starts, ends = np.asarray(assigned_start), np.asarray(assigned_end)
    predicted_samples = [(int(starts[a])-base, int(ends[b-1])-base)
                         for a, b in _runs(np.asarray(states) == 1)]
    unknown_samples = [(int(starts[a])-base, int(ends[b-1])-base)
                       for a, b in _runs(np.asarray(states) == -1)]
    reference = [(a/sampling_rate, b/sampling_rate) for a, b in reference_samples_clipped]
    predicted = [(a/sampling_rate, b/sampling_rate) for a, b in predicted_samples]
    unknown = [(a/sampling_rate, b/sampling_rate) for a, b in unknown_samples]
    confusion = episode_errors(reference, predicted, duration)
    # Count confusion on integer source samples before converting to seconds.
    missing = sum(b-a for a, b in unknown_samples)
    missing_af = sum(max(0, min(b, d)-max(a, c)) for a, b in reference_samples_clipped for c, d in unknown_samples)
    af = sum(b-a for a, b in reference_samples_clipped);nonaf = stop-base-af
    tp = sum(max(0, min(b, d)-max(a, c)) for a, b in reference_samples_clipped for c, d in predicted_samples)
    fp = sum(b-a for a, b in predicted_samples)-tp
    tn = nonaf-fp-(missing-missing_af)
    assert min(tp, fp, tn, missing_af) >= 0
    unknown_seconds, unknown_af = missing/sampling_rate, missing_af/sampling_rate
    ref_af = af/sampling_rate
    result = {
        'status': 'indeterminate_unknown_time' if unknown else 'complete',
        'observation_seconds': duration, 'unknown_seconds': unknown_seconds,
        'unknown_reference_af_seconds': unknown_af,
        'represented_seconds': duration-unknown_seconds,
        'reference_af_seconds': ref_af, 'reference_burden': ref_af/duration,
        'sample_confusion': {'true_positive': tp, 'false_positive': fp, 'true_negative': tn,
                             'known_false_negative': af-missing_af-tp,
                             'unknown_af': missing_af, 'unknown_nonaf': missing-missing_af,
                             'reference_af': af, 'reference_nonaf': nonaf},
        'all_reference_time_sensitivity': tp/af if af else None,
        'all_reference_time_specificity': tn/nonaf if nonaf else None,
        'known_time_sensitivity': tp/(af-missing_af) if af > missing_af else None,
        'known_time_specificity': tn/(nonaf-missing+missing_af) if nonaf > missing-missing_af else None,
        'reference_patterns': pattern_endpoints(reference, duration),
        'predicted_patterns': None, 'episode_errors': None,
        'pattern_absolute_errors': None,
    }
    if unknown:
        return result
    patterns = pattern_endpoints(predicted, duration)
    result['predicted_patterns'] = patterns
    result['episode_errors'] = confusion
    errors = {}
    for name in ['density_continuous', 'density_author_binned_30s_max',
                 'density_equal_timebin_analogue', 'aggregation_equal_timebin_analogue']:
        ref_value, pred_value = result['reference_patterns'][name], patterns[name]
        errors[name] = None if ref_value is None or pred_value is None else abs(ref_value-pred_value)
    result['pattern_absolute_errors'] = errors
    uncensored = []
    for pair in confusion['boundary_pairs']:
        a, b = reference[pair['reference_index']]
        if a > 0:
            uncensored.append(abs(pair['onset_error_seconds']))
        if b < duration:
            uncensored.append(abs(pair['offset_error_seconds']))
    result['uncensored_boundary_count'] = len(uncensored)
    result['uncensored_boundary_absolute_error_seconds'] = float(np.mean(uncensored)) if uncensored else None
    return result
