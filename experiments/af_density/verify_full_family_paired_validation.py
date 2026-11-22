"""Independently reconcile paired validation reports against saved timelines."""
import hashlib
import json
from pathlib import Path
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def main():
    root = W/'pre_release_corrective_v1/paired_math_v2_categorical_hybrid_q_ablated_validation_v1'
    receipt = json.loads((root/'receipt.json').read_text())
    assert receipt['complete'] and receipt['test_access'] is False
    population = json.loads((W/'evaluation_protocol_v1/population.json').read_text())
    checked = 0
    for decoder in ['D0', 'D1', 'D2']:
      for name in ['all_validation', 'primary_informative']:
        report = json.loads((root/f'{decoder}_{name}.json').read_text())
        segments = [s for s in population['segments'] if s['partition'] == 'validation'
                    and (name == 'all_validation' or s['primary_eligible'])]
        patients = sorted({s['patient_id'] for s in segments})
        assert patients == report['patients']
        values = {}
        for model in ['raw', 'rr', 'categorical', 'q_ablated', 'hybrid']:
            for patient in patients:
                records = []
                for s in segments:
                    if s['patient_id'] != patient: continue
                    model_path = model if model in ('raw', 'rr') else 'math_v2_' + model
                    row = json.loads((W/f"evaluation_protocol_v1/validation_{model_path}/file_{s['file_index']:03d}.json").read_text())
                    assert row['patient_id'] == patient
                    conditions = [c for c in row['conditions'] if c['decoder'] == decoder]
                    assert sorted(c['pattern'] for c in conditions) == [0, 1, 2]
                    records.extend(c['results'] for c in conditions)
                for metric in report['metrics']:
                    local = []
                    for r in records:
                        if metric == 'density_error':
                            value = None if r['pattern_absolute_errors'] is None else r['pattern_absolute_errors']['density_author_binned_30s_max']
                        elif metric == 'burden_error': value = r['burden_absolute_error']
                        else: value = None if r['episode_errors'] is None else r['episode_errors'][metric]
                        local.append(value)
                    values[model, patient, metric] = None if any(v is None for v in local) else sum(local)/len(local)
        draws = np.random.default_rng(report['seed']).integers(0, len(patients), (report['replicates'], len(patients)))
        for comparison in report['comparisons']:
            model = comparison['model'];baseline = comparison['comparator'];metric = comparison['metric']
            deltas = [];lo = [];hi = [];complete = []
            for patient in patients:
                a, b = values[model, patient, metric], values[baseline, patient, metric]
                good = a is not None and b is not None
                deltas.append(a-b if good else None)
                if good: complete.append(patient)
                lo.append((0 if a is None else a)-(1 if b is None else b))
                hi.append((1 if a is None else a)-(0 if b is None else b))
            assert complete == comparison['complete_pair_ids']
            mean = None if not complete else sum(d for d in deltas if d is not None)/len(complete)
            if mean is None: assert comparison['conditional_mean_difference'] is None
            else: assert abs(mean-comparison['conditional_mean_difference']) < 1e-14
            sampled = []
            for draw in draws:
                defined = [deltas[int(i)] for i in draw if deltas[int(i)] is not None]
                if defined: sampled.append(sum(defined)/len(defined))
            assert report['replicates']-len(sampled) == comparison['undefined_bootstrap_replicates']
            ci = None if len(complete) < 2 else np.quantile(sampled, [.025, .975]).tolist()
            if ci is None: assert comparison['conditional_percentile_95ci'] is None
            else: assert np.allclose(ci, comparison['conditional_percentile_95ci'], rtol=0, atol=1e-14)
            assert np.allclose([sum(lo)/len(lo), sum(hi)/len(hi)], comparison['full_population_difference_bounds'], rtol=0, atol=1e-14)
            assert comparison['preservation_claim_authorized'] is False
            checked += 1
    result = {'all_pass': True, 'comparisons_checked': checked, 'test_access': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Independent aggregation and identical-roster resampling from original saved five-model validation timelines across three decoders. Confirms arithmetic/coverage, not clinical validity, population identification or a small-sample CI guarantee.'}
    path = root/'independent_aggregation_check.json';assert not path.exists()
    path.write_text(json.dumps(result, indent=2)+'\n');print(json.dumps(result))


if __name__ == '__main__': main()
