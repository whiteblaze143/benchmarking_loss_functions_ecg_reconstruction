"""Fixed validation population, score availability and extraction-failure ledger."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = W/'pre_release_corrective_v1/validation_coverage_v1';out.mkdir(exist_ok=False)
    population = json.loads((W/'evaluation_protocol_v1/population.json').read_text())
    segments = [r for r in population['segments'] if r['partition'] == 'validation']
    assert len(segments) == 60
    with np.load(W/'windows.npz') as z: data = {k: z[k] for k in z.files}
    fi = data['file_index'];ids = np.flatnonzero(np.isin(fi, [r['file_index'] for r in segments]))
    assert len(ids) == 166047
    roots = {'raw': W/'raw_mask_resnet_v1', 'rr': W/'observed_rr_hgb_v1',
             'categorical': W/'set_models_v1/categorical'}
    rows = [];sources = {}
    pv = np.zeros((len(ids), 2), bool);rv = pv.copy()
    for segment in segments:
        file = segment['file_index'];select = fi[ids] == file
        path = W/'observed_features'/f'file_{file:03d}.npz'
        with np.load(path) as z:
            assert np.array_equal(z['window_index'], ids[select])
            pv[select] = z['phase_valid'];rv[select] = z['rr_valid']
        sources[str(path)] = sha(path)
    for model, root in roots.items():
        lock = json.loads((root/'evaluation_lock.json').read_text())
        assert lock['complete_train_validation'] and not lock['test_inference_performed']
        path = root/'validation_scores.npy';assert sha(path) == lock['validation_scores_sha256']
        scores = np.load(path);assert scores.shape == (len(ids), 3)
        assert not np.isinf(scores).any()
        sources[str(path)] = sha(path);sources[str(root/'evaluation_lock.json')] = sha(root/'evaluation_lock.json')
        for pattern, available in enumerate([[True, False], [False, True], [True, True]]):
            missing = np.isnan(scores[:, pattern])
            expected = np.zeros(len(ids), bool) if model == 'raw' else ~((rv if model == 'rr' else pv) & available).any(axis=1)
            assert np.array_equal(missing, expected), (model, pattern)
            for segment in segments:
                take = fi[ids] == segment['file_index'];chosen = ids[take];unknown = missing[take]
                duration = data['assigned_end'][chosen]-data['assigned_start'][chosen]
                rows.append({'model': model, 'patient_id': segment['patient_id'],
                    'file_index': segment['file_index'], 'pattern': pattern,
                    'primary_eligible': segment['primary_eligible'],
                    'windows': int(take.sum()), 'missing_windows': int(unknown.sum()),
                    'assigned_seconds': float(duration.sum()/200),
                    'unknown_seconds': float(duration[unknown].sum()/200),
                    'unknown_reference_af_seconds': float(data['af_assigned_samples'][chosen][unknown].sum()/200),
                    'timeline_complete': not bool(unknown.any()),
                    'failure_rule': 'none' if model == 'raw' else 'no valid observed RR channel' if model == 'rr' else 'no valid observed phase channel'})
    with (out/'conditions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary = {}
    for model in roots:
        summary[model] = {}
        for label, primary in [('all_validation', False), ('primary_informative', True)]:
            selected = [r for r in rows if r['model'] == model and (not primary or r['primary_eligible'])]
            patients = sorted({r['patient_id'] for r in selected})
            complete = [p for p in patients if all(r['timeline_complete'] for r in selected if r['patient_id'] == p)]
            summary[model][label] = {'declared_patients': len(patients), 'complete_timeline_patients': len(complete),
                'complete_patient_ids': complete, 'indeterminate_patient_ids': [p for p in patients if p not in complete],
                'missing_acquisition_windows': sum(r['missing_windows'] for r in selected),
                'unknown_seconds_across_three_patterns': sum(r['unknown_seconds'] for r in selected),
                'unknown_reference_af_seconds_across_three_patterns': sum(r['unknown_reference_af_seconds'] for r in selected)}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps({'complete': True, 'partition': 'validation', 'test_access': False,
        'files': 60, 'windows': len(ids), 'condition_rows': len(rows), 'source_sha256': sha(__file__),
        'verified_source_hashes': sources, 'unavailable_models': {'q_ablated': 'failed training; no admitted predictions',
        'hybrid': 'training pending; no admitted predictions'},
        'scope': 'Score/extraction coverage on fixed source populations. Complete timeline does not imply every physiological endpoint is defined; endpoint-specific undefined outcomes remain separately reported.'}, indent=2)+'\n')
    print(json.dumps({model: result['primary_informative'] for model, result in summary.items()}))


if __name__ == '__main__': main()
