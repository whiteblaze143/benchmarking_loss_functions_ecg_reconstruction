"""Locked RR inference from observed-voltage shards; no fitting or selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from observed_rr_design import rr_design, PATTERNS

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
R = W/'observed_rr_hgb_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(partition):
    lock = json.loads((R/'evaluation_lock.json').read_text());assert lock['complete_train_validation']
    protocol = json.loads((R/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest
    assert sha(R/'protocol.json') == lock['protocol_sha256'] and sha(R/'model.joblib') == lock['model_sha256']
    if partition == 'test':
        release = json.loads((W/'evaluation_protocol_v1/heldout_release.json').read_text())
        assert release['released'] and 'rr' in release['authorized_models']
        assert sha(W/'evaluation_protocol_v1/protocol.json') == release['evaluation_protocol_sha256']
        assert sha(R/'evaluation_lock.json') == release['evaluation_lock_hashes']['rr']
    output = R/('inference_'+partition+'_v1');output.mkdir(exist_ok=False)
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z: file_index = z['file_index'].copy()
    selected_files = [r for r in manifest if r['partition'] == partition]
    ids = np.concatenate([np.flatnonzero(file_index == int(r['file_index'])) for r in selected_files])
    assert len(ids) == (166047 if partition == 'validation' else 131667)
    expected = np.flatnonzero(np.isin(file_index, [int(r['file_index']) for r in selected_files]))
    assert np.array_equal(ids, expected)
    sources = {};feature_parts = [];valid_parts = []
    for row in selected_files:
        path = W/'observed_features'/f"file_{int(row['file_index']):03d}.npz"
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == receipt['output_sha256'] and receipt['h5_sha256'] == row['h5_sha256']
        assert receipt['preprocessing_freeze_sha256'] == sha(W/'preprocessing_freeze.json')
        with np.load(path) as z:
            assert np.array_equal(z['window_index'], np.flatnonzero(file_index == int(row['file_index'])))
            feature_parts.append(z['rr_features']);valid_parts.append(z['rr_valid'])
        sources[str(path)] = receipt['output_sha256']
    features, valid = np.concatenate(feature_parts), np.concatenate(valid_parts)
    designs = [rr_design(features, valid, pattern) for pattern in PATTERNS]
    x = np.stack([a for a, b in designs], axis=1).reshape(-1, 36)
    available = np.stack([b for a, b in designs], axis=1).reshape(-1)
    x = np.ascontiguousarray(x[available], dtype=np.float32)
    input_receipt = {'partition': partition, 'model_evaluation_lock_sha256': sha(R/'evaluation_lock.json'),
        'source_shard_hashes': sources, 'window_index_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
        'actual_float32_inference_input_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
        'source_sha256': sha(__file__), 'windows': len(ids), 'fit_performed': False, 'threshold': lock['threshold']}
    (output/'input_receipt.json').write_text(json.dumps(input_receipt, indent=2)+'\n')
    model = joblib.load(R/'model.joblib');scores = np.full(len(available), np.nan)
    scores[available] = model.predict_proba(x)[:, 1];scores = scores.reshape(-1, 3)
    if partition == 'validation':
        with np.load(R/'development_features.npz') as z: assert np.array_equal(ids, z['window_index'][504777:])
        assert np.array_equal(scores, np.load(R/'validation_scores.npy'), equal_nan=True), 'Validation inference drift; preserve attempt'
    np.savez_compressed(output/'predictions.npz', window_index=ids, scores=scores)
    result = {'complete': True, 'partition': partition, 'windows': len(ids), 'source_files': len(selected_files),
        'indeterminate_acquisition_conditions': int(np.isnan(scores).sum()), 'threshold': lock['threshold'],
        'prediction_archive_sha256': sha(output/'predictions.npz'), 'input_receipt_sha256': sha(output/'input_receipt.json'),
        'validation_bitwise_replay': True if partition == 'validation' else None,
        'test_inference_performed': partition == 'test', 'no_fit_or_threshold_selection': True}
    (output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n');print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('partition', choices=['validation', 'test'])
    main(parser.parse_args().partition)
