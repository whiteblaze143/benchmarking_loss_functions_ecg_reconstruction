"""Independent full native-sample reconstruction of external window targets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
BASE = Path('/data/mithunmanivannan/datasets')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(dataset):
    folder = W/('external_'+dataset+'_v1')
    contract = json.loads((folder/'input_contract.json').read_text())
    for name, digest in contract['input_hashes'].items(): assert sha(folder/name) == digest
    labels_path = (BASE/'iridia_af_8405941_v1.0.1/shdb_reference_intervals.csv' if dataset == 'shdb'
                   else BASE/'ltafdb_1.0.0_reference/reference_rhythm_intervals.csv')
    assert sha(labels_path) == contract['source_label_table_sha256']
    labels = list(csv.DictReader(labels_path.open()))
    manifest = list(csv.DictReader((folder/'file_manifest.csv').open()))
    with np.load(folder/'windows.npz') as z: windows = {k: z[k] for k in z.files}
    covered_windows = 0;known_samples = 0;af_samples = 0;eligible_records = 0
    for row in manifest:
        fi, length, fs, known = (int(row[k]) for k in ['file_index', 'source_samples', 'sampling_rate', 'initial_unknown_samples'])
        spans = [r for r in labels if r['record_id'] == row['record_id']]
        assert known == (int(spans[0]['start_sample']) if spans else length)
        native = np.zeros(length, np.uint8)
        for span in spans:
            if span['rhythm'] == '(AFIB':
                native[int(span['start_sample']):int(span['end_sample_exclusive'])] = 1
        take = windows['file_index'] == fi
        assert take.sum() == int(row['windows'])
        if not take.any():
            assert row['input_contract_eligible'] == 'False'
            continue
        eligible_records += 1
        inputs, starts, ends = (windows[k][take] for k in ['input_start', 'assigned_start', 'assigned_end'])
        assert starts[0] == known and ends[-1] == length and np.array_equal(starts[1:], ends[:-1])
        assert (ends > starts).all() and (inputs >= known).all() and (inputs+30*fs <= length).all()
        assert (starts >= inputs).all() and (ends <= inputs+30*fs).all()
        assert np.array_equal(inputs[:-1], starts[:-1])
        if inputs[-1] != starts[-1]: assert inputs[-1] == length-30*fs
        prefix = np.r_[0, np.cumsum(native, dtype=np.int64)]
        assigned_af = prefix[ends]-prefix[starts]
        target = ((prefix[inputs+30*fs]-prefix[inputs])*2 >= 30*fs).astype(np.uint8)
        assert np.array_equal(assigned_af, windows['af_assigned_samples'][take])
        assert np.array_equal(target, windows['target'][take])
        assert assigned_af.sum() == native[known:].sum(dtype=np.int64)
        covered_windows += len(inputs);known_samples += length-known;af_samples += int(assigned_af.sum())
        print(dataset, row['record_id'], len(inputs), 'verified', flush=True)
    assert covered_windows == contract['declared_windows'] and eligible_records == contract['eligible_records']
    result = {'all_pass': True, 'dataset': dataset, 'checked_records': len(manifest),
        'eligible_records': eligible_records, 'checked_windows': covered_windows,
        'native_known_samples': known_samples, 'native_assigned_af_samples': af_samples,
        'method': 'Independent dense native-sample AF mask and prefix sum from source rhythm CSV; every input target/assigned count and coverage boundary checked, no reuse of preparation overlap routine.',
        'input_contract_sha256': sha(folder/'input_contract.json'), 'source_sha256': sha(__file__),
        'scope': 'Annotation/assignment integrity; no clinical adjudication or model performance claim.'}
    (folder/'independent_input_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('dataset', choices=['shdb', 'ltaf'])
    main(parser.parse_args().dataset)
