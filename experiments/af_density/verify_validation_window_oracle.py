"""Independently reconstruct oracle targets, runs and overlap counts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def main():
    root = W/'pre_release_corrective_v1/window_oracle_validation'
    receipt = json.loads((root/'receipt.json').read_text())
    for name, digest in receipt['input_sha256'].items():
        assert hashlib.sha256((W/name).read_bytes()).hexdigest() == digest
    with np.load(W/'windows.npz') as z:
        windows = {k: z[k] for k in z.files}
    references = list(csv.DictReader((W/'reference_intervals.csv').open()))
    files = [r for r in csv.DictReader((W/'file_manifest.csv').open()) if r['partition'] == 'validation']
    checked = 0
    for row in files:
        fi = int(row['file_index']);take = windows['file_index'] == fi
        starts = windows['input_start'][take];ends = starts+6000
        refs = [(int(r['start_sample']), int(r['end_sample_exclusive'])) for r in references if int(r['file_index']) == fi]
        amounts = np.zeros(len(starts), np.int64)
        for a, b in refs:
            amounts += np.maximum(0, np.minimum(ends, b)-np.maximum(starts, a))
        target = amounts >= 3000
        assert np.array_equal(target, windows['target'][take])
        astart = windows['assigned_start'][take];aend = windows['assigned_end'][take]
        assert np.array_equal(astart[1:], aend[:-1])
        base, stop = int(astart[0]), int(aend[-1])
        clipped = [(max(a, base), min(b, stop)) for a, b in refs if a < stop and b > base]
        rise = np.flatnonzero(np.diff(np.r_[False, target, False].astype(int)) == 1)
        fall = np.flatnonzero(np.diff(np.r_[False, target, False].astype(int)) == -1)
        predicted = [(int(astart[a]), int(aend[b-1])) for a, b in zip(rise, fall)]
        overlaps = np.array([[max(0, min(b, d)-max(a, c)) for c, d in predicted] for a, b in clipped], dtype=np.int64).reshape(len(clipped), len(predicted))
        ref_degree = (overlaps > 0).sum(axis=1);pred_degree = (overlaps > 0).sum(axis=0)
        result = json.loads((root/f'file_{fi:03d}.json').read_text())['results'];episode = result['episode_errors']
        expected = {'reference_episodes': len(clipped), 'predicted_episodes': len(predicted),
            'missed_reference_episodes': int((ref_degree == 0).sum()),
            'false_predicted_episodes': int((pred_degree == 0).sum()),
            'split_reference_episodes': int((ref_degree > 1).sum()),
            'merged_predicted_episodes': int((pred_degree > 1).sum())}
        for key, value in expected.items(): assert episode[key] == value, (fi, key)
        assert result['sample_confusion']['true_positive'] == int(overlaps.sum())
        assert result['sample_confusion']['reference_af'] == sum(b-a for a, b in clipped)
        assert result['unknown_seconds'] == 0
        for name, error in result['pattern_absolute_errors'].items():
            x, y = result['reference_patterns'][name], result['predicted_patterns'][name]
            assert error is None if x is None or y is None else error == abs(x-y)
        checked += len(starts)
    assert len(files) == 60 and checked == 166047
    report = {'all_pass': True, 'files': len(files), 'windows': checked, 'test_access': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Independent annotation-overlap targets, assigned-clock runs, episode counts, sample confusion and error arithmetic. Published metric fidelity relies on separately preserved author-R verification; this verifier does not independently reimplement that metric.'}
    path = root/'independent_verification.json';assert not path.exists()
    path.write_text(json.dumps(report, indent=2)+'\n');print(json.dumps(report))


if __name__ == '__main__': main()
