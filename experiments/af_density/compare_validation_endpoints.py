"""Execute paired endpoint/coverage accounting on frozen baseline validation."""
import hashlib
import json
from pathlib import Path
from paired_endpoint_comparison import compare_endpoints

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = W/'evaluation_protocol_v1'
    out = W/'pre_release_corrective_v1/paired_validation_endpoints_v2'
    out.mkdir(exist_ok=False)
    population = json.loads((root/'population.json').read_text())
    segments = [r for r in population['segments'] if r['partition'] == 'validation']
    assert len(segments) == 60
    receipts = {};archives = {}
    for model in ['raw', 'rr']:
        receipt = json.loads((root/f'validation_{model}/receipt.json').read_text())
        assert receipt['complete'] and receipt['partition'] == 'validation' and receipt['test_predictions'] is False
        receipts[model] = sha(root/f'validation_{model}/receipt.json')
        archives[model] = {}
        for entry in receipt['outputs']:
            path = root/f'validation_{model}'/entry['file']
            assert sha(path) == entry['sha256']
            row = json.loads(path.read_text())
            archives[model][row['file_index']] = row
    rows = []
    for segment in segments:
        fi = segment['file_index'];patient = segment['patient_id']
        for model in archives:
            assert archives[model][fi]['patient_id'] == patient
        for pattern in range(3):
            models = {}
            for model in archives:
                conditions = [c for c in archives[model][fi]['conditions']
                              if c['pattern'] == pattern and c['decoder'] == 'D2']
                assert len(conditions) == 1
                result = conditions[0]['results'];errors = result['pattern_absolute_errors']
                episode = result['episode_errors']
                models[model] = {'density_error': None if errors is None else errors['density_author_binned_30s_max'],
                    'burden_error': result['burden_absolute_error'],
                    'episode_sensitivity': None if episode is None else episode['episode_sensitivity'],
                    'episode_ppv': None if episode is None else episode['episode_ppv']}
            rows.append({'patient': patient, 'segment': fi, 'pattern': pattern, 'models': models})
    for name, eligible in [('all_validation', {r['file_index'] for r in segments}),
                           ('primary_informative', {r['file_index'] for r in segments if r['primary_eligible']})]:
        selected = [r for r in rows if r['segment'] in eligible]
        result = compare_endpoints(selected, expected_keys=[(r['patient'], r['segment'], r['pattern']) for r in selected])
        (out/f'{name}.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps({'complete': True, 'test_access': False,
        'decoder': 'D2', 'source_sha256': sha(__file__),
        'comparison_source_sha256': sha(Path(__file__).with_name('paired_endpoint_comparison.py')),
        'population_sha256': sha(root/'population.json'), 'validation_receipt_sha256': receipts,
        'scope': 'Executed baseline-only validation accounting; not SetOperator evidence, not a superiority/noninferiority decision.'}, indent=2)+'\n')
    print(out)


if __name__ == '__main__': main()
