"""Execute fixed-population validation comparisons for completed declared models."""
import argparse
import hashlib
import json
from pathlib import Path
from paired_endpoint_comparison import compare_endpoints
from assess_joint_endpoint_support import assess_support

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(modes):
    assert len(set(modes)) == len(modes) and 'categorical' in modes
    root = W/'evaluation_protocol_v1';models = ['raw', 'rr']+sorted(modes)
    population = json.loads((root/'population.json').read_text())
    segments = [s for s in population['segments'] if s['partition'] == 'validation']
    assert len(segments) == 60
    archives = {};receipts = {}
    for model in models:
        folder = root/f'validation_{model}';receipt = json.loads((folder/'receipt.json').read_text())
        assert receipt['complete'] and receipt['partition'] == 'validation' and not receipt['test_predictions']
        assert receipt['evaluation_protocol_sha256'] == sha(root/'protocol.json')
        receipts[model] = sha(folder/'receipt.json');archives[model] = {}
        for entry in receipt['outputs']:
            path = folder/entry['file'];assert sha(path) == entry['sha256']
            row = json.loads(path.read_text());archives[model][row['file_index']] = row
        assert len(archives[model]) == 60
    out = W/'pre_release_corrective_v1'/('paired_'+'_'.join(sorted(modes))+'_validation_v1')
    out.mkdir(exist_ok=False)
    for decoder in ['D0', 'D1', 'D2']:
        rows = []
        for segment in segments:
            fi = segment['file_index'];patient = segment['patient_id']
            assert all(archives[m][fi]['patient_id'] == patient for m in models)
            for pattern in range(3):
                values = {}
                for model in models:
                    matches = [c for c in archives[model][fi]['conditions'] if c['decoder'] == decoder and c['pattern'] == pattern]
                    assert len(matches) == 1
                    r = matches[0]['results'];errors = r['pattern_absolute_errors'];episode = r['episode_errors']
                    values[model] = {'density_error': None if errors is None else errors['density_author_binned_30s_max'],
                        'burden_error': r['burden_absolute_error'],
                        'episode_sensitivity': None if episode is None else episode['episode_sensitivity'],
                        'episode_ppv': None if episode is None else episode['episode_ppv']}
                rows.append({'patient': patient, 'segment': fi, 'pattern': pattern, 'models': values})
        for label, eligible in [('all_validation', {s['file_index'] for s in segments}),
                                 ('primary_informative', {s['file_index'] for s in segments if s['primary_eligible']})]:
            selected = [r for r in rows if r['segment'] in eligible]
            report = compare_endpoints(selected, expected_keys=[(r['patient'], r['segment'], r['pattern']) for r in selected])
            path = out/f'{decoder}_{label}.json';path.write_text(json.dumps(report, indent=2)+'\n')
            if decoder == 'D2' and label == 'primary_informative':
                (out/'proposed_rule_development_assessment.json').write_text(json.dumps(assess_support(report), indent=2)+'\n')
        # Retain exact conditions for independent aggregation/review.
        (out/f'{decoder}_condition_rows.json').write_text(json.dumps(rows)+'\n')
    (out/'receipt.json').write_text(json.dumps({'complete': True, 'partition': 'validation',
        'test_access': False, 'models': models, 'population_sha256': sha(root/'population.json'),
        'source_sha256': sha(__file__), 'comparison_sha256': sha(Path(__file__).with_name('paired_endpoint_comparison.py')),
        'decision_source_sha256': sha(Path(__file__).with_name('assess_joint_endpoint_support.py')),
        'decision_proposal_sha256': sha(root/'decision_hierarchy_proposal_v1.json'),
        'timeline_receipt_sha256': receipts,
        'scope': 'Completed development comparisons under fixed D0/D1/D2. Proposed rule is not approved for test release; validation cannot establish held-out/external superiority.'}, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--set-models', nargs='+',
        choices=['categorical', 'q_ablated', 'hybrid'], required=True)
    main(parser.parse_args().set_models)
