"""Fixed-population paired endpoints with a shared patient bootstrap draw."""
import numpy as np

METRICS = ('density_error', 'burden_error', 'episode_sensitivity', 'episode_ppv')


def compare_endpoints(rows, *, expected_keys, replicates=2000, seed=20261120):
    """Compare named models without deleting failures or declaring preservation.

    Rows contain patient/segment/pattern and models mapping each model name to
    the four METRICS, each in [0,1] or None. An endpoint is patient-evaluable
    only if every declared segment and pattern is defined. All endpoint/model
    pairs reuse draws from the same fixed patient roster. Bootstrap replicates
    with zero complete pairs are undefined and counted explicitly.
    """
    keys = [(r['patient'], r['segment'], r['pattern']) for r in rows]
    expected = list(expected_keys)
    if not keys or len(set(keys)) != len(keys) or len(set(expected)) != len(expected) or set(keys) != set(expected):
        raise ValueError('Require exactly one row for every declared condition')
    if not isinstance(replicates, int) or replicates < 2:
        raise ValueError('At least two bootstrap replicates required')
    models = sorted(rows[0]['models'])
    if len(models) < 2 or any(sorted(r['models']) != models for r in rows):
        raise ValueError('Identical named model roster required on every row')
    patients = sorted({r['patient'] for r in rows})
    values = np.full((len(patients), len(models), len(METRICS)), np.nan)
    ledger = []
    for i, patient in enumerate(patients):
        selected = [r for r in rows if r['patient'] == patient]
        for segment in {r['segment'] for r in selected}:
            if {r['pattern'] for r in selected if r['segment'] == segment} != {0, 1, 2}:
                raise ValueError('Every segment requires the three declared patterns')
        for j, model in enumerate(models):
            for k, metric in enumerate(METRICS):
                raw = [r['models'][model][metric] for r in selected]
                a = np.array([np.nan if x is None else x for x in raw], float)
                if any(x is not None and not np.isfinite(x) for x in raw) or ((a[np.isfinite(a)] < 0) | (a[np.isfinite(a)] > 1)).any():
                    raise ValueError('Endpoints must be finite [0,1] or explicit None')
                count = int(np.isfinite(a).sum())
                ledger.append({'patient': patient, 'model': model, 'metric': metric,
                    'declared_conditions': len(a), 'defined_conditions': count,
                    'patient_complete': count == len(a)})
                if count == len(a):
                    values[i, j, k] = a.mean()
    draws = np.random.default_rng(seed).integers(0, len(patients), (replicates, len(patients)))
    comparisons = []
    for j, model in enumerate(models):
        for b in range(j):
            for k, metric in enumerate(METRICS):
                first, second = values[:, j, k], values[:, b, k]
                complete = np.isfinite(first) & np.isfinite(second)
                delta = first-second
                sampled = delta[draws]
                counts = np.isfinite(sampled).sum(axis=1)
                means = np.divide(np.nansum(sampled, axis=1), counts,
                    out=np.full(replicates, np.nan), where=counts > 0)
                valid_draws = means[np.isfinite(means)]
                low = np.where(np.isfinite(first), first, 0)-np.where(np.isfinite(second), second, 1)
                high = np.where(np.isfinite(first), first, 1)-np.where(np.isfinite(second), second, 0)
                comparisons.append({'model': model, 'comparator': models[b], 'metric': metric,
                    'direction': 'first minus comparator; negative favors first for errors, positive for sensitivity/PPV',
                    'declared_patients': len(patients), 'complete_pair_patients': int(complete.sum()),
                    'complete_pair_ids': [p for p, good in zip(patients, complete) if good],
                    'model_complete_patients': int(np.isfinite(first).sum()),
                    'comparator_complete_patients': int(np.isfinite(second).sum()),
                    'conditional_mean_difference': float(delta[complete].mean()) if complete.any() else None,
                    'conditional_percentile_95ci': np.quantile(valid_draws, [.025, .975]).tolist() if complete.sum() >= 2 and len(valid_draws) else None,
                    'undefined_bootstrap_replicates': int((counts == 0).sum()),
                    'full_population_difference_bounds': [float(low.mean()), float(high.mean())],
                    'full_population_complete': bool(complete.all()),
                    'preservation_claim_authorized': False})
    return {'patients': patients, 'metrics': list(METRICS), 'models': models,
        'coverage_ledger': ledger, 'comparisons': comparisons, 'replicates': replicates, 'seed': seed,
        'shared_resampling': 'Identical draws from declared patient roster across all model pairs and endpoints; incomplete patients remain in coverage/bounds.',
        'inference_scope': 'Descriptive paired intervals, unadjusted for multiplicity. No superiority or noninferiority decision; clinically justified margins and hierarchy require a separate frozen contract.'}
