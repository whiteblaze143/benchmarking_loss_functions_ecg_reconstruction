"""Apply the proposed conjunctive decision; never authorize inference/clinical claims."""
import math

METRICS = ('density_error', 'burden_error', 'episode_sensitivity', 'episode_ppv')


def assess_support(report):
    patients = report['patients']
    if len(patients) < 2 or len(set(patients)) != len(patients):
        raise ValueError('At least two distinct declared patients required')
    if not {'categorical', 'raw', 'rr'}.issubset(report['models']):
        raise ValueError('Require categorical and both declared baselines')
    checks = []
    for baseline in ['raw', 'rr']:
        for metric in METRICS:
            rows = [r for r in report['comparisons'] if r['metric'] == metric
                    and {r['model'], r['comparator']} == {'categorical', baseline}]
            if len(rows) != 1:
                raise ValueError('Exactly one comparison per required endpoint/baseline')
            row = rows[0]
            if row['declared_patients'] != len(patients):
                raise ValueError('Inconsistent fixed patient denominator')
            ci = row['conditional_percentile_95ci']
            if ci is not None:
                if len(ci) != 2 or not all(math.isfinite(x) for x in ci) or not -1 <= ci[0] <= ci[1] <= 1:
                    raise ValueError('Invalid bounded paired interval')
                if row['model'] != 'categorical': ci = [-ci[1], -ci[0]]
            complete = (row['full_population_complete'] and
                        row['complete_pair_patients'] == len(patients) and
                        set(row['complete_pair_ids']) == set(patients))
            favorable = ci is not None and (ci[1] < 0 if metric.endswith('_error') else ci[0] > 0)
            checks.append({'baseline': baseline, 'metric': metric,
                'categorical_minus_baseline_95ci': ci, 'full_population_complete': bool(complete),
                'strict_favorable_interval': bool(favorable), 'support': bool(complete and favorable)})
    density = all(r['support'] for r in checks if r['metric'] == 'density_error')
    joint = all(r['support'] for r in checks)
    return {'density_statistical_support': density, 'joint_computational_improvement_support': joint,
        'clinical_preservation_claim_authorized': False, 'test_inference_authorized': False,
        'decision_rule_status': 'Proposal awaiting independent pre-release approval',
        'checks': checks,
        'interpretation': 'Conjunctive approximate percentile-bootstrap evidence only. No favorable-subset population claim, noninferiority assertion, geometry attribution or clinical utility conclusion.'}
