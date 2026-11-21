"""Independent source-patient/weight and optional full normalization replay."""
import argparse
import csv
import json
import time
import h5py
import numpy as np
from waveform_protocol import W, sha

R = W/'raw_mask_resnet_v1'


def verify_manifest():
    protocol = json.loads((R/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest, name
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest, name
    rows = list(csv.DictReader((W/'file_manifest.csv').open()))
    partitions = np.array([r['partition'] for r in rows]);patients = np.array([r['patient_id'] for r in rows])
    with np.load(W/'windows.npz') as z:
        file_index = z['file_index'];assigned = z['assigned_end'].astype(np.int64)-z['assigned_start']
        tr = np.flatnonzero(partitions[file_index] == 'train');va = np.flatnonzero(partitions[file_index] == 'validation')
    with np.load(R/'development_manifest.npz') as z:
        ids = z['window_index'];pid = z['patient_id'];tr_weight = z['train_weight'];va_weight = z['validation_weight']
    assert len(tr) == 504777 and len(va) == 166047 and np.array_equal(ids, np.r_[tr, va])
    assert np.array_equal(pid, patients[file_index[ids]])
    assert not set(pid[:len(tr)]) & set(pid[len(tr):])
    for selected, actual in [(tr, tr_weight), (va, va_weight)]:
        expected = np.zeros(len(selected));patient = patients[file_index[selected]]
        duration = assigned[selected]
        for p in np.unique(patient):
            mask = patient == p
            expected[mask] = duration[mask]/sum(int(v) for v in duration[mask])
            assert abs(float(actual[mask].sum())-1) < 1e-12
        np.testing.assert_array_equal(actual, expected)
    normal = json.loads((R/'normalization_and_loading.json').read_text())
    expected_sources = {r['h5_path']: r['h5_sha256'] for r in rows if r['partition'] != 'test'}
    assert normal['source_hashes'] == expected_sources and not normal['test_waveforms_loaded']
    expected_samples = sum(int(r['source_samples'])-int(r['excluded_prefix_samples']) for r in rows if r['partition'] == 'train')
    assert expected_samples == normal['training_source_samples'] == int(assigned[tr].sum(dtype=np.int64))
    result = {'all_pass': True, 'training_windows': len(tr), 'validation_windows': len(va),
              'training_patients': len(set(pid[:len(tr)])), 'validation_patients': len(set(pid[len(tr):])),
              'train_validation_patient_overlap': 0, 'exact_weight_replay': True, 'patient_weight_totals': '1 within1e-12',
              'training_source_samples_counted_once': expected_samples, 'test_waveform_sources_in_loading_receipt': 0,
              'hashes': {name: sha(R/name) for name in ['protocol.json', 'development_manifest.npz', 'normalization_and_loading.json']},
              'scope': 'Manifest, source membership and exact assigned-duration weights; full numerical normalization replay is separate'}
    (R/'development_manifest_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return rows, normal, result


def verify_normalization(rows, normal):
    count = 0;mean = np.zeros(2);m2 = np.zeros(2);files = 0;started = time.monotonic()
    for row in rows:
        if row['partition'] != 'train': continue
        assert sha(row['h5_path']) == row['h5_sha256']
        with h5py.File(row['h5_path']) as h:
            signal = h['ecg'].astype('float32')
            for start in range(int(row['excluded_prefix_samples']), int(row['source_samples']), 1000000):
                x = signal[start:min(start+1000000, int(row['source_samples']))].astype(np.float64)
                n = len(x);chunk_mean = x.mean(axis=0);chunk_m2 = np.square(x-chunk_mean).sum(axis=0)
                delta = chunk_mean-mean;total = count+n
                m2 += chunk_m2+delta**2*(count*n/total)
                mean += delta*(n/total);count = total
        files += 1
        print(json.dumps({'normalization_files_checked': files, 'source_samples': count}), flush=True)
    sd = np.sqrt(m2/count)
    expected_mean = np.asarray(normal['mean_mv']);expected_sd = np.asarray(normal['population_sd_mv'])
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(sd, expected_sd, rtol=1e-9, atol=1e-10)
    assert files == 181 and count == normal['training_source_samples']
    result = {'all_pass': True, 'training_files': files, 'training_source_samples': count,
              'method': 'Independent chunked float64 parallel-variance merge of source-float32 values; no validation/test samples',
              'maximum_mean_difference_mv': float(abs(mean-expected_mean).max()),
              'maximum_population_sd_difference_mv': float(abs(sd-expected_sd).max()),
              'normalization_receipt_sha256': sha(R/'normalization_and_loading.json'), 'seconds': time.monotonic()-started}
    (R/'normalization_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--normalization', action='store_true');args = parser.parse_args()
    rows, normal, result = verify_manifest();print(json.dumps(result, indent=2), flush=True)
    if args.normalization: print(json.dumps(verify_normalization(rows, normal), indent=2), flush=True)
