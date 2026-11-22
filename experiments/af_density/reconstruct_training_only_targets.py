"""Reconstruct source labels from training-record CSVs only; never read mixed target arrays."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = list(csv.DictReader((W/'file_manifest.csv').open()))
    assert all(int(row['file_index']) == i for i, row in enumerate(rows))
    source_hashes = json.loads((W/'input_contract.json').read_text())['label_csv_sha256']
    train_records = {row['record_id'] for row in rows if row['partition'] == 'train'}
    other_records = {row['record_id'] for row in rows if row['partition'] != 'train'}
    assert train_records.isdisjoint(other_records)
    record_events = {}
    for rid in sorted(train_records):
        path = W/'reference_ecg_labels'/f'{rid}.csv'
        assert sha(path) == source_hashes[rid]
        with path.open(newline='') as f: record_events[rid] = list(csv.DictReader(f))
    indices = [];targets = [];offset = 0
    for row in rows:
        count = int(row['windows'])
        if row['partition'] == 'train':
            length = int(row['source_samples']);excluded = int(row['excluded_prefix_samples'])
            starts = np.arange(excluded, length-6000+1, 6000, dtype=np.int64)
            if starts[-1]+6000 < length: starts = np.r_[starts, length-6000]
            assert len(starts) == count
            overlap = np.zeros(count, dtype=np.int64);day = int(row['daily_file_index'])
            for event in record_events[row['record_id']]:
                start_file, end_file = int(event['start_file_index']), int(event['end_file_index'])
                if start_file <= day <= end_file:
                    a = int(event['start_qrs_index']) if start_file == day else 0
                    b = int(event['end_qrs_index']) if end_file == day else length
                    assert 0 <= a < b <= length
                    overlap += np.maximum(0, np.minimum(starts+6000, b)-np.maximum(starts, a))
            indices.append(np.arange(offset, offset+count, dtype=np.int64))
            targets.append((overlap >= 3000).astype(np.int8))
        offset += count
    ids = np.concatenate(indices);labels = np.concatenate(targets)
    assert len(ids) == 504777 and len(train_records) == 99
    for model in ['categorical', 'q_ablated', 'hybrid']:
        with np.load(W/f'set_models_math_sdp_v2/{model}/development_manifest.npz') as z:
            assert np.array_equal(ids, z['window_index'][:len(ids)])
    root = W/'pre_release_corrective_v1/training_only_targets_v1';root.mkdir(exist_ok=False)
    np.savez_compressed(root/'targets.npz', window_index=ids, target=labels)
    receipt = {'complete': True, 'partition': 'train', 'windows': len(ids), 'source_records': len(train_records),
        'heldout_label_arrays_or_csv_opened': False, 'source_label_csv_sha256': {rid: source_hashes[rid] for rid in sorted(train_records)},
        'file_manifest_sha256': sha(W/'file_manifest.csv'), 'input_contract_sha256': sha(W/'input_contract.json'),
        'source_sha256': sha(Path(__file__)), 'output_sha256': sha(root/'targets.npz'),
        'scope': 'Reconstructed only training labels from source record CSVs and frozen per-file window geometry; matches all three training index manifests. Not independent clinical adjudication.'}
    (root/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'complete': True, 'training_windows': len(ids), 'training_records': len(train_records),
                      'af_positive_windows': int(labels.sum()), 'heldout_label_arrays_or_csv_opened': False}))


if __name__ == '__main__': main()
