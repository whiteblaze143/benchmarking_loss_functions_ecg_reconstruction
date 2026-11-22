"""Run frozen timeline endpoints on independently replayed Set validation."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import h5py
import numpy as np
from evaluate_timeline_conditions import evaluate_conditions

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
P = Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(mode):
    root = W/'set_models_v1'/mode;ep = W/'evaluation_protocol_v1'
    lock = json.loads((root/'evaluation_lock.json').read_text())
    replay = json.loads((root/'independent_validation_v1/receipt.json').read_text())
    assert replay['all_pass'] and replay['model_lock_sha256'] == sha(root/'evaluation_lock.json')
    assert not lock['test_inference_performed'] and replay['test_access'] is False
    protocol = json.loads((ep/'protocol.json').read_text())
    for p, digest in protocol['source_hashes'].items(): assert sha(p) == digest
    assert sha(root/'validation_scores.npy') == lock['validation_scores_sha256']
    with np.load(W/'windows.npz') as z: windows = {k: z[k] for k in z.files}
    with np.load(root/'development_manifest.npz') as z:
        ids = z['window_index'][504777:];train_ids = z['window_index'][:504777];weights = z['train_weight']
    prior = float(np.sum(weights*windows['target'][train_ids, None])/weights.sum())
    scores = np.load(root/'validation_scores.npy');assert scores.shape == (166047, 3)
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    val_files = [int(r['file_index']) for r in manifest if r['partition'] == 'validation']
    assert np.array_equal(ids, np.flatnonzero(np.isin(windows['file_index'], val_files)))
    references = list(csv.DictReader((W/'reference_intervals.csv').open()))
    law = W/'duration_model_v1/fitted_law.npz'
    assert sha(law) == protocol['primary_decoder']['duration_law_sha256']
    with np.load(law) as z: duration = {k: z[k] for k in z.files}
    out = ep/f'validation_{mode}';out.mkdir(exist_ok=False);outputs = []
    with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as archive:
        for row in manifest:
            if row['partition'] != 'validation': continue
            fi = int(row['file_index']);take = windows['file_index'][ids] == fi;chosen = ids[take]
            ref = [(int(r['start_sample']), int(r['end_sample_exclusive'])) for r in references if int(r['file_index']) == fi]
            member = f"iridia-af-records-v1.0.1/{row['record_id']}/{row['record_id']}_rr_{int(row['daily_file_index']):02d}.h5"
            data = archive.read(member)
            with h5py.File(io.BytesIO(data)) as h: rr = h['rr'][:]
            conditions = evaluate_conditions(scores[take], ref, windows['assigned_start'][chosen],
                windows['assigned_end'][chosen], 200, threshold=lock['threshold'], training_target_prior=prior,
                duration_model=duration, released_rr_milliseconds=rr)
            for c in conditions: c['states'] = c['states'].tolist()
            result = {'patient_id': row['patient_id'], 'record_id': row['record_id'], 'file_index': fi,
                'window_index': chosen.tolist(), 'reference_episodes': len(ref), 'conditions': conditions,
                'source_RR_member': member, 'source_RR_member_sha256': hashlib.sha256(data).hexdigest()}
            path = out/f'file_{fi:03d}.json';path.write_text(json.dumps(result)+'\n')
            outputs.append({'file': path.name, 'sha256': sha(path)})
            print(json.dumps({'mode': mode, 'completed_validation_files': len(outputs)}), flush=True)
    assert len(outputs) == 60
    receipt = {'complete': True, 'model': mode, 'partition': 'validation', 'test_predictions': False,
        'model_evaluation_lock_sha256': sha(root/'evaluation_lock.json'),
        'independent_validation_receipt_sha256': sha(root/'independent_validation_v1/receipt.json'),
        'evaluation_protocol_sha256': sha(ep/'protocol.json'), 'source_sha256': sha(__file__),
        'training_target_prior': prior, 'windows': len(ids), 'declared_files': 60, 'outputs': outputs}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--mode', choices=['categorical', 'q_ablated', 'hybrid'], required=True)
    main(parser.parse_args().mode)
