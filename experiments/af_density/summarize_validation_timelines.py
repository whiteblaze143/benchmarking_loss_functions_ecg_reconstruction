"""Descriptive fixed-population validation summary; never tune a decoder here."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/evaluation_protocol_v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(result):
    errors = result['pattern_absolute_errors'];episode = result['episode_errors']
    beat = result['published_shared_RR_clock']['absolute_errors']
    return {'density_author_binned_absolute_error': None if errors is None else errors['density_author_binned_30s_max'],
        'density_continuous_absolute_error': None if errors is None else errors['density_continuous'],
        'aggregation_shared_RR_clock_absolute_error': None if beat is None else beat['aggregation'],
        'burden_absolute_error': result['burden_absolute_error'],
        'episode_sensitivity': None if episode is None else episode['episode_sensitivity'],
        'episode_ppv': None if episode is None else episode['episode_ppv'],
        'false_episodes_per_hour': None if episode is None else episode['false_predicted_episodes']/result['observation_seconds']*3600,
        'split_rate': None if episode is None else episode['split_rate'],
        'merge_rate': None if episode is None else episode['merge_rate'],
        'uncensored_boundary_absolute_error_seconds': result.get('uncensored_boundary_absolute_error_seconds'),
        'unknown_seconds': result['unknown_seconds']}


def main(model):
    folder = W/('validation_'+model);receipt = json.loads((folder/'receipt.json').read_text())
    assert receipt['complete'] and receipt['declared_files'] == 60 and receipt['partition'] == 'validation'
    population = json.loads((W/'population.json').read_text())
    eligible = {r['file_index']: r for r in population['segments'] if r['partition'] == 'validation' and r['primary_eligible']}
    assert len(eligible) == 15 and len({r['patient_id'] for r in eligible.values()}) == 11
    by_decoder = {d: [] for d in ['D0', 'D1', 'D2']}
    for saved in receipt['outputs']:
        path = folder/saved['file'];assert sha(path) == saved['sha256']
        data = json.loads(path.read_text());fi = data['file_index']
        if fi not in eligible: continue
        assert data['patient_id'] == eligible[fi]['patient_id']
        assert {(r['decoder'], r['pattern']) for r in data['conditions']} == {(d, p) for d in by_decoder for p in range(3)}
        for condition in data['conditions']:
            by_decoder[condition['decoder']].append({'patient_id': data['patient_id'], 'file_index': fi,
                'pattern': condition['pattern'], **metrics(condition['results'])})
    summary = {}
    for decoder, cases in by_decoder.items():
        assert len(cases) == 45
        patients = sorted({r['patient_id'] for r in cases});patient_values = {};statistics = {}
        for name in metrics(json.loads((folder/receipt['outputs'][0]['file']).read_text())['conditions'][0]['results']):
            values = {}
            for patient in patients:
                selected = [r[name] for r in cases if r['patient_id'] == patient]
                values[patient] = None if any(v is None for v in selected) else float(np.mean(selected))
            finite = [v for v in values.values() if v is not None]
            statistics[name] = {'declared_patients': 11, 'complete_patients': len(finite),
                'declared_segment_conditions': 45, 'defined_segment_conditions': sum(r[name] is not None for r in cases),
                'conditional_patient_mean': float(np.mean(finite)) if finite else None}
            patient_values[name] = values
        summary[decoder] = {'statistics': statistics, 'patient_values': patient_values}
    output = W/('validation_summary_'+model+'.json');assert not output.exists()
    result = {'model': model, 'partition': 'validation', 'informative_patients': 11, 'informative_segments': 15,
        'evaluation_receipt_sha256': sha(folder/'receipt.json'), 'population_sha256': sha(W/'population.json'),
        'source_sha256': sha(__file__), 'decoders': summary,
        'scope': 'Descriptive validation results under frozen settings. Patient mean requires every declared segment/availability condition defined for that metric. Missingness and metric-edge undefined values retained. No test inference, model selection, causal mechanism or superiority claim.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({d: r['statistics']['density_author_binned_absolute_error'] for d, r in summary.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('model', choices=['rr', 'raw'])
    main(parser.parse_args().model)
