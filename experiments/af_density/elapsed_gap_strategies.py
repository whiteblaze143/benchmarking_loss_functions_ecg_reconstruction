"""Explicit secondary gap conventions on the actual integer sample clock."""
import numpy as np
from temporal_baselines import timeline_intervals


def gap_adjusted_timeline(reference_samples, states, assigned_start, assigned_end, sampling_rate, strategy):
    """Return an imputed or compressed timeline without changing primary inputs.

    Context fill uses the two observed flanks and a midpoint in elapsed samples,
    splitting an assignment when necessary. These are imputed states, not new
    ECG observations. Observed-only compression also transforms the reference;
    its density measures a different observation period and episode pattern.
    """
    timeline_intervals(states, assigned_start, assigned_end, sampling_rate)
    x = np.asarray(states);a = np.asarray(assigned_start);b = np.asarray(assigned_end)
    reference = list(reference_samples);previous = -1
    for lo, hi in reference:
        if not isinstance(lo, (int, np.integer)) or not isinstance(hi, (int, np.integer)) or not 0 <= lo < hi or lo <= previous:
            raise ValueError('Expected sorted separated integer source AF intervals')
        previous = hi
    if strategy not in ['nonaf', 'context', 'observed_only']:
        raise ValueError('Unknown declared gap strategy')
    missing = x == -1;missing_samples = int((b[missing]-a[missing]).sum())
    common = {'strategy': strategy, 'original_missing_samples': missing_samples,
              'original_duration_samples': int(b[-1]-a[0]), 'status': 'complete'}
    if strategy == 'observed_only':
        if missing.all():
            return {**common, 'status': 'indeterminate_no_observed_time', 'states': None,
                    'reference_samples': None, 'clock': 'compressed_observed_samples'}
        known_a, known_b = a[~missing], b[~missing]
        durations = known_b-known_a;new_b = np.cumsum(durations);new_a = np.r_[0, new_b[:-1]]
        projected = []
        for start, stop, target in zip(known_a, known_b, new_a):
            for lo, hi in reference:
                left, right = max(int(start), lo), min(int(stop), hi)
                if left >= right: continue
                piece = (int(target+left-start), int(target+right-start))
                if projected and projected[-1][1] == piece[0]: projected[-1] = (projected[-1][0], piece[1])
                else: projected.append(piece)
        return {**common, 'clock': 'compressed_observed_samples', 'states': x[~missing].copy(),
                'assigned_start': new_a, 'assigned_end': new_b, 'reference_samples': projected,
                'clock_map': list(zip(known_a.tolist(), known_b.tolist(), new_a.tolist())),
                'imputed_samples': 0, 'removed_samples': missing_samples}
    if strategy == 'nonaf':
        filled = x.copy();filled[missing] = 0
        return {**common, 'clock': 'original_samples', 'states': filled, 'assigned_start': a.copy(),
                'assigned_end': b.copy(), 'reference_samples': reference,
                'imputed_samples': missing_samples, 'removed_samples': 0}
    if missing[0] or missing[-1]:
        return {**common, 'status': 'indeterminate_missing_flanks', 'clock': 'original_samples',
                'states': None, 'reference_samples': reference}
    output = [];i = 0
    while i < len(x):
        if not missing[i]:
            output.append((int(a[i]), int(b[i]), int(x[i])));i += 1;continue
        stop = i+1
        while stop < len(x) and missing[stop]: stop += 1
        left, right = int(x[i-1]), int(x[stop]);start, end = int(a[i]), int(b[stop-1])
        if left == right: output.append((start, end, left))
        else:
            middle = start+(end-start)//2
            if middle > start: output.append((start, middle, left))
            output.append((middle, end, right))
        i = stop
    starts, ends, filled = map(np.asarray, zip(*output))
    timeline_intervals(filled, starts, ends, sampling_rate)
    return {**common, 'clock': 'original_samples', 'states': filled.astype(np.int8),
            'assigned_start': starts, 'assigned_end': ends, 'reference_samples': reference,
            'imputed_samples': missing_samples, 'removed_samples': 0}
