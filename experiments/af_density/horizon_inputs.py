"""Contiguous prefix inputs with an explicit prohibition on future ECG context."""
import numpy as np
from temporal_baselines import timeline_intervals


def contiguous_prefix(input_start, assigned_start, assigned_end, sampling_rate, hours):
    if hours not in [6, 12, 24]: raise ValueError('Unsupported declared horizon')
    starts = np.asarray(assigned_start);ends = np.asarray(assigned_end);inputs = np.asarray(input_start)
    timeline_intervals(np.zeros(len(starts), np.int8), starts, ends, sampling_rate)
    if inputs.shape != starts.shape: raise ValueError('Input and assignment rows must match')
    stop = int(starts[0])+hours*3600*sampling_rate
    if ends[-1] < stop: return {'status': 'ineligible_insufficient_contiguous_time', 'hours': hours}
    indices = np.flatnonzero(starts < stop);prefix_ends = np.minimum(ends[indices], stop)
    if (inputs[indices]+30*sampling_rate > stop).any():
        raise ValueError('A selected local input would use ECG beyond the horizon')
    assert prefix_ends[-1] == stop
    assert int((prefix_ends-starts[indices]).sum()) == hours*3600*sampling_rate
    return {'status': 'complete', 'hours': hours, 'indices': indices,
            'assigned_start': starts[indices], 'assigned_end': prefix_ends,
            'source_start_sample': int(starts[0]), 'source_end_sample': int(stop)}
