"""Independently check all currently committed external observed-peak shards."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
O = W/'external_observed_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    checked = [];phase_total = np.zeros(2, np.int64)
    for dataset in ['shdb', 'ltaf']:
        manifest = list(csv.DictReader((W/('external_'+dataset+'_v1')/'file_manifest.csv').open()))
        with np.load(W/('external_'+dataset+'_v1')/'windows.npz') as z: file_index = z['file_index']
        for path in sorted((O/dataset).glob('file_*.json')):
            if path.name.endswith('.errors.json'): continue
            receipt = json.loads(path.read_text());fi = receipt['file_index'];row = manifest[fi]
            assert receipt['dataset'] == dataset and receipt['record_id'] == row['record_id']
            assert receipt['source_sha256'] == row['signal_sha256'] and receipt['protocol_sha256'] == sha(O/'protocol.json')
            assert sha(path.with_suffix('.npz')) == receipt['output_sha256']
            assert sha(path.with_suffix('.errors.json')) == receipt['errors_sha256']
            with np.load(path.with_suffix('.npz')) as z:
                ids = z['window_index'];n = len(ids)
                assert np.array_equal(ids, np.flatnonzero(file_index == fi)) and n == receipt['windows']
                assert z['rr_features'].shape == (n, 2, 16) and z['rr_features'].dtype == np.float32
                assert np.isnan(z['rr_features'][~z['rr_valid']]).all()
                assert np.isin(z['detector_status'], [0, 1, 2, 3]).all()
                assert (z['peak_counts'][~z['source_channel_available']] == 0).all()
                for c in range(2):
                    offsets = z[f'peak_offsets_{c}'];peaks = z[f'peak_samples_{c}']
                    assert offsets.shape == (n+1,) and offsets[0] == 0 and offsets[-1] == len(peaks)
                    assert np.array_equal(np.diff(offsets), z['peak_counts'][:, c])
                    assert ((peaks >= 0) & (peaks < 6000)).all()
                    counts = [];fractions = []
                    for a, b in zip(offsets[:-1], offsets[1:]):
                        rr = np.diff(peaks[a:b].astype(np.int32));assert (rr > 0).all()
                        valid = (rr >= 60) & (rr <= 400)
                        counts.append(valid.sum());fractions.append(rr[valid].sum()/6000)
                    counts = np.asarray(counts)
                    assert np.array_equal(counts, z['accepted_intervals'][:, c])
                    assert np.array_equal(counts >= 2, z['phase_valid'][:, c])
                    assert np.array_equal(np.asarray(fractions, np.float32), z['represented_fraction'][:, c])
                assert z['phase_valid'].sum(axis=0).tolist() == receipt['phase_valid_by_channel']
                assert z['rr_valid'].sum(axis=0).tolist() == receipt['rr_valid_by_channel']
                phase_total += z['phase_valid'].sum(axis=0)
            checked.append({'dataset': dataset, 'file_index': fi, 'windows': n, 'receipt_sha256': sha(path)})
    assert checked
    result = {'all_pass': True, 'snapshot_only': True, 'checked_files': len(checked),
        'checked_windows': sum(r['windows'] for r in checked), 'files': checked,
        'phase_valid_by_channel': phase_total.tolist(), 'source_sha256': sha(__file__),
        'scope': 'Committed shard hashes, exact window membership, serialized peak geometry, validity and phase coverage. Does not prove clinical QRS accuracy or full extraction completion.'}
    (O/f'verification_snapshot_{time.time_ns()}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__': main()
