"""Matched acquisition-condition evaluation on one original sample clock."""
import numpy as np
from decode_timeline import decode_segment
from elapsed_evaluation import evaluate_elapsed_segment
from beat_clock_evaluation import evaluate_shared_beat_clock


def evaluate_conditions(scores, reference_samples, assigned_start, assigned_end, sampling_rate,
                        *, threshold, training_target_prior, duration_model, released_rr_milliseconds):
    """Return all three fixed acquisition subsets crossed with D0/D1/D2.

    This function does no fitting, threshold selection, input repair or gap
    compression. The same reference and assignment clock serve every condition.
    Source identifiers and partition/hash checks belong to the archive reader.
    """
    p = np.asarray(scores, dtype=np.float64)
    if p.shape != (len(assigned_start), 3):
        raise ValueError('Expected one score per assigned window and each of three acquisition subsets')
    configs = [{'mode': 'D0', 'threshold': threshold},
               {'mode': 'D1', 'threshold': threshold, 'median_width': 3, 'max_gap_seconds': 60.},
               {'mode': 'D2', 'posterior_prior': training_target_prior, 'emission_strength': 1.,
                'probability_floor': 1e-6, 'nominal_input_seconds': 30.}]
    rows = []
    for pattern in range(3):
        for config in configs:
            decoded = decode_segment(p[:, pattern], assigned_start, assigned_end, sampling_rate, config,
                duration_model=duration_model if config['mode'] == 'D2' else None)
            result = evaluate_elapsed_segment(reference_samples, decoded['states'], assigned_start, assigned_end, sampling_rate)
            beat = evaluate_shared_beat_clock(reference_samples, decoded['states'], assigned_start, assigned_end,
                sampling_rate, released_rr_milliseconds)
            confusion = result['sample_confusion'];total = int(assigned_end[-1])-int(assigned_start[0])
            predicted_burden = None if result['unknown_seconds'] else (confusion['true_positive']+confusion['false_positive'])/total
            result['predicted_burden'] = predicted_burden
            result['burden_absolute_error'] = None if predicted_burden is None else abs(predicted_burden-result['reference_burden'])
            result['published_shared_RR_clock'] = beat
            rows.append({'pattern': pattern, 'decoder': config['mode'], 'configuration': config,
                         'unknown_windows': decoded['unknown_windows'], 'results': result,
                         'states': decoded['states']})
    return rows
