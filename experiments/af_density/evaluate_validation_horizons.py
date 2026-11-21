"""Rerun fixed temporal decoders on prefix-only validation local scores."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import h5py
import numpy as np
from horizon_inputs import contiguous_prefix
from evaluate_timeline_conditions import evaluate_conditions

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
P = Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')
MODELS = {'rr': 'observed_rr_hgb_v1', 'raw': 'raw_mask_resnet_v1',
          'categorical': 'set_models_v1/categorical', 'q_ablated': 'set_models_v1/q_ablated', 'hybrid': 'set_models_v1/hybrid'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(model):
    root = W/MODELS[model];lock = json.loads((root/'evaluation_lock.json').read_text())
    assert lock['complete_train_validation'] and not lock['test_inference_performed']
    assert sha(root/'validation_scores.npy') == lock['validation_scores_sha256']
    filename = 'development_features.npz' if model == 'rr' else 'development_manifest.npz'
    with np.load(root/filename) as z:
        all_ids = z['window_index'];ids = all_ids[504777:]
        weight = None if model == 'rr' else z['train_weight'].copy()
    with np.load(W/'windows.npz') as z: windows = {k: z[k] for k in z.files}
    if model == 'rr': prior = json.loads((root/'independent_development_verification.json').read_text())['weighted_training_target_prior']
    else:
        labels = windows['target'][all_ids[:504777]]
        assert weight.shape == ((504777,) if model == 'raw' else (504777, 3))
        prior = float(np.sum(weight*(labels if model == 'raw' else labels[:, None]))/weight.sum())
    scores = np.load(root/'validation_scores.npy');assert scores.shape == (166047, 3)
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    expected = np.concatenate([np.flatnonzero(windows['file_index'] == int(r['file_index'])) for r in files if r['partition'] == 'validation'])
    assert np.array_equal(ids, expected)
    reference_rows = list(csv.DictReader((W/'reference_intervals.csv').open()))
    with np.load(W/'duration_model_v1/fitted_law.npz') as z: duration_model = {k: z[k] for k in z.files}
    output = W/'evaluation_protocol_v1'/('validation_horizons_'+model);output.mkdir(exist_ok=False)
    source_paths = [Path(__file__), Path(__file__).with_name('horizon_inputs.py'), Path(__file__).with_name('evaluate_timeline_conditions.py')]
    (output/'input_receipt.json').write_text(json.dumps({'partition': 'validation', 'model': model, 'test_predictions': False,
        'model_evaluation_lock_sha256': sha(root/'evaluation_lock.json'), 'training_target_prior': prior,
        'duration_law_sha256': sha(W/'duration_model_v1/fitted_law.npz'),
        'horizon_inventory_sha256': sha(W/'horizon_inventory_v1/summary.json'),
        'source_hashes': {str(p): sha(p) for p in source_paths},
        'policy': 'Prefixes begin at daily post-calibration known start, not acquisition start. Recompute D0/D1/D2 from local prefix scores; no full-day decoded states or future ECG. Strict unknown time retained.'}, indent=2)+'\n')
    completed = [];started = time.monotonic()
    with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as archive:
        for row in files:
            if row['partition'] != 'validation': continue
            fi = int(row['file_index']);take = windows['file_index'][ids] == fi;chosen = ids[take]
            reference = [(int(r['start_sample']), int(r['end_sample_exclusive'])) for r in reference_rows if int(r['file_index']) == fi]
            member = f"iridia-af-records-v1.0.1/{row['record_id']}/{row['record_id']}_rr_{int(row['daily_file_index']):02d}.h5"
            with h5py.File(io.BytesIO(archive.read(member))) as h: rr = h['rr'][:]
            results = []
            for hours in [6, 12, 24]:
                prefix = contiguous_prefix(windows['input_start'][chosen], windows['assigned_start'][chosen], windows['assigned_end'][chosen], 200, hours)
                result = {'hours': hours, 'status': prefix['status']}
                if prefix['status'] == 'complete':
                    selected = prefix['indices'];start, stop = prefix['source_start_sample'], prefix['source_end_sample']
                    episode_count = sum(a < stop and b > start for a, b in reference)
                    conditions = evaluate_conditions(scores[take][selected], reference, prefix['assigned_start'], prefix['assigned_end'], 200,
                        threshold=lock['threshold'], training_target_prior=prior, duration_model=duration_model, released_rr_milliseconds=rr)
                    for condition in conditions: condition['states'] = condition['states'].tolist()
                    result.update({'source_start_sample': start, 'source_end_sample': stop, 'reference_episode_count': episode_count,
                                   'informative_multi_episode': episode_count >= 2, 'windows': len(selected), 'conditions': conditions})
                results.append(result)
            path = output/f'file_{fi:03d}.json'
            path.write_text(json.dumps({'file_index': fi, 'patient_id': row['patient_id'], 'record_id': row['record_id'], 'horizons': results})+'\n')
            completed.append({'file': path.name, 'sha256': sha(path)})
            print(json.dumps({'completed_files': len(completed), 'declared_files': 60, 'seconds': time.monotonic()-started}), flush=True)
    assert len(completed) == 60
    (output/'receipt.json').write_text(json.dumps({'complete': True, 'partition': 'validation', 'model': model, 'outputs': completed,
        'seconds': time.monotonic()-started, 'test_predictions': False, 'scope': 'Observation-length sensitivity on qualified daily prefixes. Ineligible24h retained, no disease progression or full acquisition-clock claim.'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('model', choices=MODELS)
    main(parser.parse_args().model)
