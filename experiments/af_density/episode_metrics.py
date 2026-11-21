"""Episode errors on one fully observed elapsed-time segment.

Detection means strictly positive overlap, with no minimum episode duration or
boundary tolerance. Split/merge metrics describe the bipartite overlap graph;
they are not a one-to-one event matching score. Boundary errors are reported
only for isolated one-reference/one-prediction pairs. Missing time must be
segmented or handled explicitly before calling; this function never bridges it.
"""
import math


def _intervals(values, duration):
    result = [(float(a), float(b)) for a, b in values]
    previous = -1.0
    for a, b in result:
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration):
            raise ValueError('Invalid episode interval')
        if a <= previous:
            raise ValueError('Episodes must be sorted, separated and nonoverlapping')
        previous = b
    return result


def episode_errors(reference, predicted, observation_seconds):
    """Return explicit counts/denominators and signed boundary errors in seconds.

    Split rate denominator is all reference episodes; merge rate denominator is
    all predicted episodes. An undetected reference or false predicted episode
    remains in its corresponding denominator. Empty denominators return None.
    Burden uses the complete supplied observation clock, including non-AF time.
    This is a measurement convention, not clinical episode adjudication.
    """
    duration = float(observation_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Observation duration must be finite and positive')
    ref = _intervals(reference, duration)
    pred = _intervals(predicted, duration)
    neighbors_ref = [[] for _ in ref]
    neighbors_pred = [[] for _ in pred]
    i = j = 0
    overlap_seconds = 0.0
    while i < len(ref) and j < len(pred):
        a, b = ref[i]
        c, d = pred[j]
        overlap = min(b, d) - max(a, c)
        if overlap > 0:
            neighbors_ref[i].append(j)
            neighbors_pred[j].append(i)
            overlap_seconds += overlap
        if b <= d:
            i += 1
        else:
            j += 1
    detected = sum(bool(x) for x in neighbors_ref)
    true_calls = sum(bool(x) for x in neighbors_pred)
    split = sum(len(x) > 1 for x in neighbors_ref)
    merged = sum(len(x) > 1 for x in neighbors_pred)
    boundary_pairs = []
    for i, neighbors in enumerate(neighbors_ref):
        if len(neighbors) == 1:
            j = neighbors[0]
            if len(neighbors_pred[j]) == 1:
                boundary_pairs.append({
                    'reference_index': i, 'predicted_index': j,
                    'onset_error_seconds': pred[j][0] - ref[i][0],
                    'offset_error_seconds': pred[j][1] - ref[i][1],
                })
    ref_seconds = math.fsum(b-a for a, b in ref)
    pred_seconds = math.fsum(b-a for a, b in pred)
    boundary_mae = (math.fsum(abs(p[k]) for p in boundary_pairs
                             for k in ['onset_error_seconds', 'offset_error_seconds'])
                    / (2 * len(boundary_pairs))) if boundary_pairs else None
    return {
        'observation_seconds': duration,
        'reference_episodes': len(ref), 'predicted_episodes': len(pred),
        'detected_reference_episodes': detected,
        'missed_reference_episodes': len(ref)-detected,
        'overlapping_predicted_episodes': true_calls,
        'false_predicted_episodes': len(pred)-true_calls,
        'episode_sensitivity': detected/len(ref) if ref else None,
        'episode_ppv': true_calls/len(pred) if pred else None,
        'split_reference_episodes': split,
        'split_rate': split/len(ref) if ref else None,
        'merged_predicted_episodes': merged,
        'merge_rate': merged/len(pred) if pred else None,
        'reference_overlap_degrees': [len(x) for x in neighbors_ref],
        'predicted_overlap_degrees': [len(x) for x in neighbors_pred],
        'boundary_pairs': boundary_pairs,
        'boundary_pair_count': len(boundary_pairs),
        'boundary_absolute_error_seconds': boundary_mae,
        'reference_af_seconds': ref_seconds, 'predicted_af_seconds': pred_seconds,
        'true_positive_seconds': overlap_seconds,
        'false_negative_seconds': ref_seconds-overlap_seconds,
        'false_positive_seconds': pred_seconds-overlap_seconds,
        'reference_burden': ref_seconds/duration,
        'predicted_burden': pred_seconds/duration,
        'burden_absolute_error': abs(pred_seconds-ref_seconds)/duration,
    }
