"""Patient-level descriptive gap sensitivity on the fixed validation population."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from summarize_validation_timelines import metrics

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1/evaluation_protocol_v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(model):
    folder = W/('validation_gaps_'+model);receipt = json.loads((folder/'receipt.json').read_text());assert receipt['complete']
    population = json.loads((W/'population.json').read_text())
    eligible = {r['file_index'] for r in population['segments'] if r['partition'] == 'validation' and r['primary_eligible']}
    cases = []
    for saved in receipt['outputs']:
        path = folder/saved['file'];assert sha(path) == saved['sha256'];data = json.loads(path.read_text())
        if data['file_index'] not in eligible: continue
        for case in data['conditions']:
            values = metrics(case['results']) if 'results' in case else None
            cases.append({'patient': data['patient_id'], **case, 'metric_values': values})
    assert len(cases) == 15*3*3*3
    names = list(next(c['metric_values'] for c in cases if c['metric_values'] is not None))
    summary = {}
    for strategy in ['nonaf', 'context', 'observed_only']:
        summary[strategy] = {}
        for decoder in ['D0', 'D1', 'D2']:
            selected = [c for c in cases if c['strategy'] == strategy and c['decoder'] == decoder]
            patients = sorted({c['patient'] for c in selected});assert len(selected) == 45 and len(patients) == 11
            result = {}
            for name in names:
                values = []
                for patient in patients:
                    observations = [None if c['metric_values'] is None else c['metric_values'][name] for c in selected if c['patient'] == patient]
                    if all(v is not None for v in observations): values.append(float(np.mean(observations)))
                result[name] = {'declared_patients': 11, 'complete_patients': len(values),
                    'conditional_patient_mean': float(np.mean(values)) if values else None}
            result['reference_change'] = {'declared_segment_conditions': 45,
                'conditions_with_changed_reference_density': sum(abs(c.get('reference_density_after', 0)-c.get('reference_density_before', 0)) > 1e-9 for c in selected),
                'removed_reference_AF_samples': sum(c.get('removed_reference_AF_samples', 0) for c in selected),
                'original_unknown_seconds_sum_across_conditions': sum(c['original_missing_samples'] for c in selected)/200}
            summary[strategy][decoder] = result
    target = W/('validation_gap_summary_'+model+'.json');assert not target.exists()
    target.write_text(json.dumps({'model': model, 'partition': 'validation', 'results': summary,
        'input_receipt_sha256': sha(folder/'receipt.json'), 'source_sha256': sha(__file__),
        'limitations': 'Fixed secondary conventions, not strategy selection. Filling creates assumed rhythm; observed-only can change the reference and model-specific evaluable clock. These results cannot replace strict primary results or establish full-clock superiority.'}, indent=2)+'\n')
    print(json.dumps({s: r['D2']['density_author_binned_absolute_error'] for s, r in summary.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('model', choices=['rr', 'raw'])
    main(parser.parse_args().model)
