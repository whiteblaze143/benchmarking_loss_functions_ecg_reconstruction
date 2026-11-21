"""Independent serialized phase-cache checks over currently committed files."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
F = W/'phase_features_v1'


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z: file_index = z['file_index']
    receipts = sorted((F/'shards').glob('file_*.json'));assert receipts
    count = 0;valid_totals = np.zeros(2, np.int64);checked = []
    for path in receipts:
        receipt = json.loads(path.read_text());fi = receipt['file_index'];row = manifest[fi]
        assert receipt['feature_protocol_sha256'] == sha(F/'protocol.json')
        assert receipt['kernel_sha256'] == sha(W/'phase_kernel_v1/kernel.npz')
        assert receipt['h5_sha256'] == row['h5_sha256'] and receipt['partition'] == row['partition']
        observed_path = W/'observed_features'/path.with_suffix('.npz').name
        assert receipt['observed_shard_sha256'] == sha(observed_path)
        assert receipt['output_sha256'] == sha(path.with_suffix('.npz'))
        with np.load(path.with_suffix('.npz')) as z, np.load(observed_path) as original:
            ids = z['window_index'];n = len(ids);valid = z['phase_valid'];response = z['response']
            assert np.array_equal(ids, np.flatnonzero(file_index == fi)) and n == receipt['windows']
            assert response.shape == (n, 2, 16, 128) and response.dtype == np.float32
            assert valid.dtype == bool and np.array_equal(valid, original['phase_valid'])
            assert np.isfinite(response[valid]).all() and np.isnan(response[~valid]).all()
            for key in ['accepted_intervals', 'represented_fraction']:
                assert np.array_equal(z[key], original[key])
            assert valid.sum(axis=0).tolist() == receipt['phase_valid_by_channel']
            valid_totals += valid.sum(axis=0);count += n
        errors = receipt['numpy_max_absolute_error_by_channel']
        assert all(e is None or 0 <= e <= 1e-7 for e in errors)
        checked.append({'file_index': fi, 'receipt_sha256': sha(path)})
    result = {'all_pass': True, 'snapshot_only': True, 'checked_files': len(checked), 'checked_windows': count,
        'phase_valid_by_channel': valid_totals.tolist(), 'files': checked, 'source_sha256': sha(__file__),
        'scope': 'Serialized feature dimensions, finite/NaN validity, source/transform identity, exact window membership and retained coverage. Does not re-run every original NumPy feature or establish clinical validity/full cache completion.'}
    (F/f'independent_snapshot_{time.time_ns()}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__': main()
