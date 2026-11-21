"""Execute frozen timeline endpoints on all validation files, without tuning."""
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
from evaluate_timeline_conditions import evaluate_conditions

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
P = Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(model_name):
    protocol = json.loads((W/'evaluation_protocol_v1/protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest
    model_root = W/('observed_rr_hgb_v1' if model_name == 'rr' else 'raw_mask_resnet_v1')
    lock = json.loads((model_root/'evaluation_lock.json').read_text())
    assert lock['complete_train_validation'] and not lock['test_inference_performed']
    assert sha(model_root/'validation_scores.npy') == lock['validation_scores_sha256']
    with np.load(model_root/('development_features.npz' if model_name == 'rr' else 'development_manifest.npz')) as z:
        all_ids = z['window_index'];ids = all_ids[504777:]
        raw_weights = z['train_weight'].copy() if model_name == 'raw' else None
    scores = np.load(model_root/'validation_scores.npy');assert scores.shape == (166047, 3)
    with np.load(W/'windows.npz') as z: windows = {k: z[k] for k in z.files}
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    assert all(manifest[i]['partition'] == 'validation' for i in windows['file_index'][ids])
    expected = np.concatenate([np.flatnonzero(windows['file_index'] == int(r['file_index'])) for r in manifest if r['partition'] == 'validation'])
    assert np.array_equal(ids, expected)
    prior = (json.loads((model_root/'independent_development_verification.json').read_text())['weighted_training_target_prior']
             if model_name == 'rr' else float(np.dot(raw_weights, windows['target'][all_ids[:504777]])/raw_weights.sum()))
    references = list(csv.DictReader((W/'reference_intervals.csv').open()))
    with np.load(W/'duration_model_v1/fitted_law.npz') as z: duration_model = {k: z[k] for k in z.files}
    assert sha(W/'duration_model_v1/fitted_law.npz') == protocol['primary_decoder']['duration_law_sha256']
    output = W/'evaluation_protocol_v1'/('validation_'+model_name)
    output.mkdir(exist_ok=False)
    receipt = {'model': model_name, 'partition': 'validation', 'test_predictions': False,
        'model_evaluation_lock_sha256': sha(model_root/'evaluation_lock.json'),
        'evaluation_protocol_sha256': sha(W/'evaluation_protocol_v1/protocol.json'),
        'source_sha256': sha(__file__), 'condition_evaluator_sha256': sha(Path(__file__).with_name('evaluate_timeline_conditions.py')),
        'training_target_prior': prior, 'windows': len(ids), 'declared_files': 60}
    (output/'input_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    started = time.monotonic();completed = []
    with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as archive:
        for row in manifest:
            if row['partition'] != 'validation': continue
            fi = int(row['file_index']);take = windows['file_index'][ids] == fi;chosen = ids[take]
            reference = [(int(x['start_sample']), int(x['end_sample_exclusive'])) for x in references if int(x['file_index']) == fi]
            member = f"iridia-af-records-v1.0.1/{row['record_id']}/{row['record_id']}_rr_{int(row['daily_file_index']):02d}.h5"
            data = archive.read(member)
            with h5py.File(io.BytesIO(data)) as h: rr = h['rr'][:]
            conditions = evaluate_conditions(scores[take], reference, windows['assigned_start'][chosen],
                windows['assigned_end'][chosen], 200, threshold=lock['threshold'], training_target_prior=prior,
                duration_model=duration_model, released_rr_milliseconds=rr)
            for condition in conditions: condition['states'] = condition['states'].tolist()
            result = {'patient_id': row['patient_id'], 'record_id': row['record_id'], 'file_index': fi,
                'source_RR_member': member, 'source_RR_member_sha256': hashlib.sha256(data).hexdigest(),
                'window_index': chosen.tolist(), 'reference_episodes': len(reference), 'conditions': conditions}
            path = output/f'file_{fi:03d}.json';path.write_text(json.dumps(result)+'\n')
            completed.append({'file': path.name, 'sha256': sha(path)})
            print(json.dumps({'completed_files': len(completed), 'declared_files': 60, 'last_file': fi,
                              'seconds': time.monotonic()-started}), flush=True)
    assert len(completed) == 60
    (output/'receipt.json').write_text(json.dumps({'complete': True, **receipt, 'outputs': completed,
        'seconds': time.monotonic()-started, 'scope': 'All validation daily segments; not held-out evidence and no decoder or threshold tuning'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--model', choices=['rr', 'raw'], required=True)
    main(parser.parse_args().model)
