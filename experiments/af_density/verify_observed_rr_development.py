"""Fresh-process reconstruction of the fitted observed-RR development inputs."""
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from matched_selection import select_validation_threshold

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
O = W/'observed_rr_hgb_v1'


def digest(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def main():
    receipt = json.loads((O/'fit_input_receipt.json').read_text())
    lock = json.loads((O/'evaluation_lock.json').read_text())
    with np.load(O/'development_features.npz') as z:
        ids, rr, valid = z['window_index'], z['rr_features'], z['rr_valid']
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z:
        fi = z['file_index'][ids];labels = z['target'][ids]
        durations = z['assigned_end'][ids]-z['assigned_start'][ids]
        af = z['af_assigned_samples'][ids]
        expected = np.concatenate([np.concatenate([np.flatnonzero(z['file_index'] == int(r['file_index']))
            for r in manifest if r['partition'] == partition]) for partition in ['train', 'validation']])
    assert np.array_equal(ids, expected)
    patients = np.array([manifest[i]['patient_id'] for i in fi])
    partitions = np.array([manifest[i]['partition'] for i in fi])
    n = 504777
    assert (partitions[:n] == 'train').all() and (partitions[n:] == 'validation').all()
    assert len(set(patients[:n])) == 90 and len(set(patients[n:])) == 30
    assert not set(patients[:n]) & set(patients[n:])
    designs = [];available = []
    for observed in [[True, False], [False, True], [True, True]]:
        active = valid & np.array(observed)
        designs.append(np.column_stack((np.where(active[:, :, None], rr, np.nan).reshape(-1, 32),
            np.tile(observed, (len(rr), 1)), active)).astype(np.float32))
        available.append(active.any(axis=1))
    x = np.stack(designs, axis=1).reshape(-1, 36)
    eligible = np.stack(available, axis=1).reshape(-1)
    train_valid = eligible[:n*3]
    p = np.repeat(patients[:n], 3);d = np.repeat(durations[:n], 3)
    weights = np.zeros(len(p), dtype=np.float64)
    for patient in np.unique(p):
        take = (p == patient) & train_valid
        assert take.any()
        weights[take] = d[take]/d[take].sum(dtype=np.float64)
        assert abs(weights[take].sum()-1) < 1e-12
    assert digest(np.ascontiguousarray(x[:n*3][train_valid])) == receipt['training_input_sha256']
    assert digest(np.repeat(labels[:n], 3)[train_valid]) == receipt['training_target_sha256']
    assert digest(weights[train_valid]) == receipt['training_weight_sha256']
    model = joblib.load(O/'model.joblib')
    val_valid = eligible[n*3:];scores = np.full(len(val_valid), np.nan)
    scores[val_valid] = model.predict_proba(np.ascontiguousarray(x[n*3:][val_valid]))[:, 1]
    saved = np.load(O/'validation_scores.npy').reshape(-1)
    assert np.array_equal(scores, saved, equal_nan=True)
    selected = select_validation_threshold(scores, np.repeat(af[n:], 3), np.repeat(durations[n:], 3),
        np.repeat(patients[n:], 3), allowed_validation_patients=np.unique(patients[n:]))
    assert selected['selected']['threshold'] == lock['threshold']
    result = {'all_pass': True, 'training_windows': n, 'validation_windows': len(ids)-n,
        'fitted_training_examples': int(train_valid.sum()), 'training_patients': 90, 'validation_patients': 30,
        'training_feature_target_weight_hashes_reconstructed': True,
        'fresh_process_validation_bitwise': True, 'threshold_reselected': lock['threshold'],
        'weighted_training_target_prior': float(np.dot(weights[train_valid], np.repeat(labels[:n], 3)[train_valid])/weights.sum()),
        'test_feature_shards_read': False, 'test_predictions': False}
    (O/'independent_development_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
