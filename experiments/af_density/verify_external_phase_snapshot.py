"""Verify committed external feature shards; certify full scope only if complete."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from extract_external_observed import sha

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
F = W/'external_phase_features_v1'


def main():
    protocol = json.loads((F/'protocol.json').read_text())
    for path, digest in protocol['source_hashes'].items(): assert sha(path) == digest
    for path, digest in protocol['input_hashes'].items(): assert sha(W/path) == digest
    records = [];valid_total = np.zeros(2, np.int64)
    for dataset in ['shdb', 'ltaf']:
        rows = list(csv.DictReader((W/f'external_{dataset}_v1/file_manifest.csv').open()))
        with np.load(W/f'external_{dataset}_v1/windows.npz') as z: file_index = z['file_index']
        for path in sorted((F/dataset).glob('file_*.json')):
            receipt = json.loads(path.read_text());fi = receipt['file_index'];row = rows[fi]
            assert row['input_contract_eligible'] == 'True'
            assert receipt['dataset'] == dataset and receipt['record_id'] == row['record_id']
            assert receipt['feature_protocol_sha256'] == sha(F/'protocol.json')
            assert receipt['kernel_sha256'] == protocol['input_hashes']['phase_kernel_v1/kernel.npz']
            assert receipt['source_sha256'] == row['signal_sha256']
            observed = W/f'external_observed_v1/{dataset}'/path.with_suffix('.npz').name
            assert sha(observed) == receipt['observed_shard_sha256']
            assert sha(path.with_suffix('.npz')) == receipt['output_sha256']
            with np.load(observed) as source, np.load(path.with_suffix('.npz')) as features:
                ids = features['window_index'];n = len(ids)
                assert np.array_equal(ids, np.flatnonzero(file_index == fi))
                for name in ['window_index', 'phase_valid', 'accepted_intervals', 'represented_fraction']:
                    assert np.array_equal(features[name], source[name])
                response, valid = features['response'], features['phase_valid']
                assert response.dtype == np.float32 and response.shape == (n, 2, 16, 128)
                assert valid.dtype == bool and valid.shape == (n, 2)
                assert np.isfinite(response[valid]).all() and np.isnan(response[~valid]).all()
                assert n == receipt['windows']
                assert valid.sum(axis=0).tolist() == receipt['phase_valid_by_channel']
                assert int((~valid.any(axis=1)).sum()) == receipt['both_channels_unavailable_windows']
                for c, error in enumerate(receipt['numpy_max_absolute_error_by_channel']):
                    if valid[:, c].any(): assert error is not None and np.isfinite(error) and 0 <= error <= 1e-7
                    else: assert error is None
                valid_total += valid.sum(axis=0)
            records.append({'dataset': dataset, 'file_index': fi, 'windows': n,
                            'receipt_sha256': sha(path)})
    assert records, 'No committed feature shards available to verify'
    assert len({(r['dataset'], r['file_index']) for r in records}) == len(records)
    count = sum(r['windows'] for r in records)
    status = json.loads((F/'status.json').read_text());complete = status.get('complete', False)
    if complete:
        assert status['status'] == 'complete' and status['files'] == len(records) == 182
        assert status['windows'] == count == 511919
    result = {'all_pass': True, 'complete_scope_verification': complete,
        'checked_files': len(records), 'checked_windows': count,
        'phase_valid_by_channel': valid_total.tolist(), 'files': records,
        'source_sha256': sha(__file__), 'test_inference': False,
        'scope': 'Every committed response shape/value-validity, original window membership, observed validity/coverage, and source/kernel/protocol custody. Original NumPy agreement is the per-channel first-valid-window check. No clinical QRS validation or diagnostic predictions.'}
    (F/f'independent_snapshot_{time.time_ns()}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__': main()
