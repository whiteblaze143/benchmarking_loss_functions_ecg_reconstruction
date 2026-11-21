"""One explicit D0/D1/D2 interface on original contiguous sample assignments."""
import math
import numpy as np
from temporal_baselines import threshold_states, filter_and_merge, timeline_intervals
from hsmm import hsmm_viterbi


def posterior_log_emissions(probabilities, *, posterior_prior, emission_strength, probability_floor, assigned_fraction):
    """Training-prior-corrected log scores, with explicit clipping and strength.

    The caller supplies the posterior's fitted population prior, not a test-set
    prevalence or a classification threshold. This conversion assumes the local
    probabilities approximate posteriors under that population; calibration is
    a separate development decision. It does not establish model calibration.
    """
    p = np.asarray(probabilities, dtype=np.float64);fraction = np.asarray(assigned_fraction, dtype=np.float64)
    if p.ndim != 1 or fraction.shape != p.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite available probabilities and assignment fractions')
    if (not math.isfinite(posterior_prior) or not 0 < posterior_prior < 1 or
            not math.isfinite(emission_strength) or emission_strength <= 0 or
            not math.isfinite(probability_floor) or not 0 < probability_floor < .5 or
            not np.isfinite(fraction).all() or (fraction <= 0).any() or (fraction > 1).any()):
        raise ValueError('Invalid emission prior, strength, probability floor or assigned fraction')
    clipped = np.clip(p, probability_floor, 1-probability_floor)
    result = np.column_stack((np.log1p(-clipped)-np.log1p(-posterior_prior),
                              np.log(clipped)-np.log(posterior_prior)))
    return result*(fraction*emission_strength)[:, None], int((p != clipped).sum())


def decode_segment(scores, assigned_start, assigned_end, sampling_rate, config, *, duration_model=None):
    """Preserve unknown windows and source time under every admitted decoder.

    HSMM independently decodes contiguous available blocks with both boundaries
    censored. It does not infer across or fill unknown time. No daily files or
    separate monitoring records may be concatenated before this call.
    """
    fields = {'D0': {'mode', 'threshold'},
              'D1': {'mode', 'threshold', 'median_width', 'max_gap_seconds'},
              'D2': {'mode', 'posterior_prior', 'emission_strength', 'probability_floor', 'nominal_input_seconds'}}
    mode = config.get('mode')
    if mode not in fields or set(config) != fields[mode]:
        raise ValueError('Decoder configuration must contain exactly its declared fields')
    p = np.asarray(scores, dtype=np.float64)
    validation_states = threshold_states(p, .5)
    timeline_intervals(validation_states, assigned_start, assigned_end, sampling_rate)
    starts, ends = np.asarray(assigned_start), np.asarray(assigned_end)
    seconds = (ends.astype(np.int64)-starts.astype(np.int64))/sampling_rate
    available = np.isfinite(p);blocks = [];clipped_count = 0
    if mode != 'D2':
        if duration_model is not None: raise ValueError('D0/D1 do not consume a duration model')
        states = threshold_states(p, config['threshold'])
        if mode == 'D1':
            states = filter_and_merge(states, seconds, median_width=config['median_width'], max_gap_seconds=config['max_gap_seconds'])
    else:
        if duration_model is None or set(duration_model) != {'duration_pmf', 'initial_occupancy'}:
            raise ValueError('D2 requires the explicitly fitted duration model')
        nominal = config['nominal_input_seconds']
        if not math.isfinite(nominal) or nominal <= 0:
            raise ValueError('Invalid nominal local input duration')
        # Validate parameters even for an entirely unavailable recording.
        emissions, clipped_count = posterior_log_emissions(p[available], posterior_prior=config['posterior_prior'],
            emission_strength=config['emission_strength'], probability_floor=config['probability_floor'], assigned_fraction=seconds[available]/nominal)
        states = np.full(len(p), -1, np.int8)
        first = np.flatnonzero(available & ~np.r_[False, available[:-1]])
        stop = np.flatnonzero(available & ~np.r_[available[1:], False])+1
        cursor = 0
        for a, b in zip(first, stop):
            result = hsmm_viterbi(emissions[cursor:cursor+b-a], duration_model['duration_pmf'], duration_model['initial_occupancy'],
                                  left_censored=True, right_censored=True)
            states[a:b] = result['states'];cursor += b-a
            blocks.append({'first_window': int(a), 'end_window_exclusive': int(b),
                           'source_start_sample': int(starts[a]), 'source_end_sample_exclusive': int(ends[b-1]),
                           'log_score': result['log_score']})
        assert cursor == available.sum()
    assert np.array_equal(states != -1, available)
    return {'states': states, 'timeline': timeline_intervals(states, starts, ends, sampling_rate),
            'unknown_windows': int((~available).sum()),
            'unknown_assigned_samples': int((ends[~available].astype(np.int64)-starts[~available]).sum()),
            'clipped_probabilities': clipped_count, 'hsmm_available_blocks': blocks,
            'gap_policy': 'retain_unknown_original_clock'}
