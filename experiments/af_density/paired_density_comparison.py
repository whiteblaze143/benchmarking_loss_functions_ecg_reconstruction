"""Patient-paired density errors with declared-condition completeness checks."""
import numpy as np


def compare_density_errors(rows, *, expected_keys, replicates=2000, seed=20261120):
    """Each row has patient, segment, pattern and two absolute density errors.

    None means indeterminate; it is never deleted before patient aggregation.
    All segments and patterns receive equal within-patient weight. Complete-pair
    bootstrap is explicitly conditional, and cannot support full-cohort
    superiority if even one declared patient has an indeterminate condition.
    """
    keys = [(r['patient'], r['segment'], r['pattern']) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(expected_keys):
        raise ValueError('Rows must cover each declared paired condition exactly once')
    if not keys or not isinstance(replicates, int) or replicates < 2:
        raise ValueError('Nonempty declared population and at least two bootstrap draws required')
    patients = sorted({key[0] for key in keys})
    differences = [];complete = [];bounds = [];missing = []
    for patient in patients:
        selected = [r for r in rows if r['patient'] == patient]
        segments = {r['segment'] for r in selected}
        # Primary protocol requires all three acquisition patterns for every segment.
        if any({r['pattern'] for r in selected if r['segment'] == s} != {0, 1, 2} for s in segments):
            raise ValueError('Every segment requires exactly the three declared acquisition patterns')
        values = np.array([[np.nan if r[name] is None else r[name] for name in ['set_error', 'baseline_error']]
                           for r in selected], dtype=np.float64)
        if np.isinf(values).any() or ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)).any():
            raise ValueError('Absolute density errors must lie in [0,1] or be indeterminate')
        known = np.isfinite(values)
        low = np.where(known, values, 0.);high = np.where(known, values, 1.)
        bounds.append([float(np.mean(low[:, 0]-high[:, 1])), float(np.mean(high[:, 0]-low[:, 1]))])
        if known.all():
            complete.append(patient);differences.append(float(np.mean(values[:, 0]-values[:, 1])))
        else:
            missing.append({'patient': patient, 'set_missing_conditions': int((~known[:, 0]).sum()),
                            'baseline_missing_conditions': int((~known[:, 1]).sum()), 'declared_conditions': len(selected)})
    ci = None;mean = None
    if differences:
        delta = np.asarray(differences)
        draws = np.random.default_rng(seed).integers(0, len(delta), (replicates, len(delta)))
        ci = np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()
        mean = float(delta.mean())
    return {'declared_patients': len(patients), 'complete_paired_patients': len(complete),
            'complete_patient_ids': complete, 'indeterminate_patients': missing,
            'conditional_mean_error_difference': mean, 'conditional_paired_percentile_95ci': ci,
            'full_cohort_mean_difference_bounds': np.mean(bounds, axis=0).tolist(),
            'full_cohort_superiority_supported': bool(not missing and ci is not None and ci[1] < 0),
            'bootstrap_replicates': replicates, 'bootstrap_seed': seed,
            'direction': 'SetOperator error minus baseline error; negative favors SetOperator',
            'limitation': 'Conditional bootstrap excludes whole incomplete patients; bounds retain all declared patients. Density superiority alone does not establish episode/burden preservation or external replication.'}
