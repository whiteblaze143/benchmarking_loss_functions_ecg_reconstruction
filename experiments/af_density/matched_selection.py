"""Explicit patient weighting and validation-only elapsed-time threshold policy."""
import numpy as np


def patient_sample_weights(patient_ids, assigned_samples, eligible):
    """Equal total weight one per patient; failed representations get weight zero.

    Every declared training patient must have an eligible example. No class
    reweighting or subsequent patient-specific renormalization is performed.
    Multiple availability examples may repeat the same assigned duration; their
    declared row distribution is therefore part of the training protocol.
    """
    patients = np.asarray(patient_ids)
    duration = np.asarray(assigned_samples)
    valid = np.asarray(eligible)
    if (patients.ndim != 1 or not len(patients) or duration.shape != patients.shape or
            duration.dtype.kind not in 'iu' or (duration <= 0).any() or
            valid.shape != patients.shape or valid.dtype != bool):
        raise ValueError('Expected patient IDs, positive integer assignments and boolean eligibility')
    weights = np.zeros(len(patients), dtype=np.float64)
    for patient in np.unique(patients):
        selected = (patients == patient) & valid
        if not selected.any():
            raise ValueError('No eligible training representation for patient '+str(patient))
        weights[selected] = duration[selected]/duration[selected].sum(dtype=np.float64)
        assert abs(weights[patients == patient].sum()-1) < 1e-12
    return weights


def elapsed_balanced_accuracy(scores, af_samples, assigned_samples, patient_ids, threshold):
    """Patient means of AF sensitivity/non-AF specificity on all assigned time.

    NaN is unknown: it contributes neither TP nor TN but retains its true class
    denominator. A patient without a true class is omitted only from that class
    mean; support counts are returned. Time is counted on assigned samples, not
    the 30-second input target or the number of overlapping input windows.
    """
    p = np.asarray(scores, dtype=np.float64)
    af = np.asarray(af_samples);duration = np.asarray(assigned_samples)
    patients = np.asarray(patient_ids)
    if (p.ndim != 1 or not len(p) or af.shape != p.shape or duration.shape != p.shape or
            patients.shape != p.shape or af.dtype.kind not in 'iu' or duration.dtype.kind not in 'iu' or
            (duration <= 0).any() or (af < 0).any() or (af > duration).any() or
            np.isinf(p).any() or ((p < 0) | (p > 1)).any() or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Invalid probabilities or assigned sample references')
    available = np.isfinite(p);positive = available & (p >= threshold);negative = available & ~positive
    sensitivities = [];specificities = []
    for patient in np.unique(patients):
        selected = patients == patient
        pos_total = af[selected].sum(dtype=np.int64)
        neg_total = (duration[selected]-af[selected]).sum(dtype=np.int64)
        if pos_total:
            sensitivities.append(float(af[selected & positive].sum(dtype=np.int64)/pos_total))
        if neg_total:
            specificities.append(float((duration[selected & negative]-af[selected & negative]).sum(dtype=np.int64)/neg_total))
    if not sensitivities or not specificities:
        raise ValueError('Both reference classes required for balanced-accuracy selection')
    sensitivity = float(np.mean(sensitivities));specificity = float(np.mean(specificities))
    return {'balanced_accuracy': (sensitivity+specificity)/2,
            'patient_mean_time_sensitivity': sensitivity, 'patient_mean_time_specificity': specificity,
            'positive_patient_denominator': len(sensitivities), 'negative_patient_denominator': len(specificities),
            'unknown_rows': int((~available).sum()),
            'unknown_assigned_samples': int(duration[~available].sum(dtype=np.int64))}


def select_validation_threshold(scores, af_samples, assigned_samples, patient_ids, *, allowed_validation_patients):
    """Fixed .01--.99 candidates; ties closest to .5, then higher threshold."""
    if set(np.asarray(patient_ids).tolist()) != set(allowed_validation_patients):
        raise ValueError('Rows must cover exactly the declared validation patient set')
    results = []
    for threshold in np.arange(1, 100)/100:
        metric = elapsed_balanced_accuracy(scores, af_samples, assigned_samples, patient_ids, threshold)
        results.append({'threshold': float(threshold), **metric})
    best = max(results, key=lambda r: (r['balanced_accuracy'], -abs(r['threshold']-.5), r['threshold']))
    return {'selected': best, 'candidates': results}
