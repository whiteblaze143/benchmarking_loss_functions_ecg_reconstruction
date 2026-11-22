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


def require_test_release(model):
    release = json.loads((W/'evaluation_protocol_v1/heldout_release.json').read_text())
    assert release['released'] and model in release['authorized_models']
    assert release['evaluation_protocol_sha256'] == sha(W/'evaluation_protocol_v1/protocol.json')
    assert release['input_contract_sha256'] == sha(W/'input_contract.json')
    assert release['evaluation_lock_hashes'][model] == sha(R/'evaluation_lock.json')
    assert release['predictor_source_hashes'][model] == sha(__file__)
    reference = release['independent_approval_receipt']
    assert sha(reference['path']) == reference['sha256']
    approval = json.loads(Path(reference['path']).read_text())
    assert approval['authorize_test_inference'] and approval['scope'] == 'locked_internal_benchmark'
    assert approval['reviewed_model_lock_hashes'] == release['evaluation_lock_hashes']
    assert approval['reviewed_predictor_source_hashes'] == release['predictor_source_hashes']
    assert approval['reviewed_input_contract_sha256'] == release['input_contract_sha256']
    parity_path = R/'inference_validation_v3/receipt.json'
    assert sha(parity_path) == release['validation_parity_receipt_hashes'][model]
    parity = json.loads(parity_path.read_text())
    assert parity['complete'] and parity['validation_bitwise_replay']
    assert parity['source_sha256'] == sha(__file__)
    assert parity['model_evaluation_lock_sha256'] == sha(R/'evaluation_lock.json')


def main(partition):
    lock = json.loads((R/'evaluation_lock.json').read_text());assert lock['complete_train_validation']
    protocol = json.loads((R/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest, name
    assert sha(R/'protocol.json') == lock['protocol_sha256'] and sha(R/'model.joblib') == lock['model_sha256']
    if partition == 'test':
        require_test_release('rr')
    output = R/('inference_'+partition+'_v3');output.mkdir(exist_ok=False)
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
    result = {'complete': True, 'partition': partition, 'source_sha256': sha(__file__),
        'model_evaluation_lock_sha256': sha(R/'evaluation_lock.json'), 'windows': len(ids), 'source_files': len(selected_files),
        'indeterminate_acquisition_conditions': int(np.isnan(scores).sum()), 'threshold': lock['threshold'],
        'prediction_archive_sha256': sha(output/'predictions.npz'), 'input_receipt_sha256': sha(output/'input_receipt.json'),
        'validation_bitwise_replay': True if partition == 'validation' else None,
        'test_inference_performed': partition == 'test', 'no_fit_or_threshold_selection': True}
    (output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n');print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('partition', choices=['validation', 'test'])
    main(parser.parse_args().partition)
