"""Training-only diagnostic of the locked window-target/assignment resolution."""
import csv
import json
from collections import defaultdict
import numpy as np
from waveform_protocol import W, sha
from episode_metrics import episode_errors
from temporal_baselines import timeline_intervals
from metrics import charitos_density


def main():
    destination = W/'training_target_resolution'
    assert not destination.exists(), 'Preserve the existing diagnostic outputs'
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    reference = defaultdict(list)
    for row in csv.DictReader((W/'reference_intervals.csv').open()):
        reference[int(row['file_index'])].append((int(row['start_sample']), int(row['end_sample_exclusive'])))
    with np.load(W/'windows.npz') as z:
        windows = {key: z[key] for key in z.files}
    results = []
    for row in files:
        if row['partition'] != 'train':
            continue
        fi = int(row['file_index'])
        selected = windows['file_index'] == fi
        starts = windows['assigned_start'][selected]
        ends = windows['assigned_end'][selected]
        timeline = timeline_intervals(windows['target'][selected], starts, ends, 200)
        assert not timeline['unknown']
        base, stop = int(starts[0]), int(ends[-1])
        ref = [((max(a, base)-base)/200, (min(b, stop)-base)/200)
               for a, b in reference[fi] if a < stop and b > base]
        pred = [(a-base/200, b-base/200) for a, b in timeline['af']]
        duration = (stop-base)/200
        result = episode_errors(ref, pred, duration)
        ref_density = charitos_density(ref, duration)
        pred_density = charitos_density(pred, duration)
        result.update({
            'file_index': fi, 'patient_id': row['patient_id'], 'record_id': row['record_id'],
            'daily_file_index': int(row['daily_file_index']), 'partition': 'train',
            'source_rr_annotation_disagreement_flag': row['source_rr_annotation_disagreement_flag'],
            'reference_density_continuous': ref_density,
            'target_density_continuous': pred_density,
            'density_absolute_error': None if ref_density is None or pred_density is None else abs(ref_density-pred_density),
        })
        results.append(result)
    expected = json.loads((W/'input_contract.json').read_text())['partitions']['train']
    assert len(results) == expected['files']
    assert len({r['patient_id'] for r in results}) == expected['patients']
    summary = {}
    for metric in ['density_absolute_error', 'burden_absolute_error', 'episode_sensitivity',
                   'episode_ppv', 'split_rate', 'merge_rate', 'boundary_absolute_error_seconds']:
        by_patient = defaultdict(list)
        for r in results:
            if r[metric] is not None:
                by_patient[r['patient_id']].append(r[metric])
        summary[metric] = {'defined_files': sum(r[metric] is not None for r in results),
                           'defined_patients': len(by_patient),
                           'patient_equal_mean_of_file_means': float(np.mean([np.mean(v) for v in by_patient.values()])) if by_patient else None}
    receipt = {
        'complete': True, 'partition': 'train', 'files': len(results),
        'patients': len({r['patient_id'] for r in results}),
        'reference_episodes': sum(r['reference_episodes'] for r in results),
        'missed_reference_episodes': sum(r['missed_reference_episodes'] for r in results),
        'false_predicted_episodes': sum(r['false_predicted_episodes'] for r in results),
        'split_reference_episodes': sum(r['split_reference_episodes'] for r in results),
        'merged_predicted_episodes': sum(r['merged_predicted_episodes'] for r in results),
        'summary': summary,
        'input_hashes': {name: sha(W/name) for name in ['windows.npz', 'file_manifest.csv', 'reference_intervals.csv', 'input_contract.json']},
        'limitations': [
            'Reference-derived diagnostic, not model predictions or held-out performance.',
            'Not a lower bound on achievable density error: a different imperfect episode timeline can accidentally match density better.',
            'Daily calibration-excluded segments evaluated separately; no gaps compressed or daily files joined.',
            'Continuous-time interval density only; this diagnostic does not evaluate the author-binned convention or AF aggregation.',
            'Undefined densities/episode denominators retained explicitly; reference record_023 is not adjudicated.',
        ],
    }
    destination.mkdir()
    (destination/'file_results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    receipt['file_results_sha256'] = sha(destination/'file_results.json')
    (destination/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
