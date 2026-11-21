"""Freeze external native-clock cohorts/windows before diagnostic inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import wfdb
from external_waveform_inputs import native_window_layout
from metrics import charitos_density

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
BASE = Path('/data/mithunmanivannan/datasets')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(dataset):
    root = BASE/('shdb_af' if dataset == 'shdb' else 'ltafdb_1.0.0_reference')
    label_path = (BASE/'iridia_af_8405941_v1.0.1/shdb_reference_intervals.csv' if dataset == 'shdb'
                  else root/'reference_rhythm_intervals.csv')
    integrity = json.loads((root/'waveform_integrity_audit.json').read_text());assert integrity['complete'] and integrity['publisher_hashes_match']
    labels = list(csv.DictReader(label_path.open()))
    metadata = {r['Data_ID']: r for r in csv.DictReader((root/'AdditionalData.csv').open())} if dataset == 'shdb' else {}
    output = W/('external_'+dataset+'_v1');output.mkdir(exist_ok=False)
    files = [];arrays = [];references = []
    for index, checked in enumerate(integrity['files']):
        rid = checked['record'];h = wfdb.rdheader(str(root/rid));fs = int(h.fs)
        spans = [r for r in labels if r['record_id'] == rid]
        known = int(spans[0]['start_sample']) if spans else h.sig_len
        previous = known;af = [];af_flutter = []
        for span in spans:
            a, b = int(span['start_sample']), int(span['end_sample_exclusive'])
            assert a == previous and a < b <= h.sig_len and int(span['sampling_rate']) == fs
            previous = b
            references.append({'file_index': index, 'rhythm': span['rhythm'], 'start_sample': a, 'end_sample_exclusive': b})
            for chosen, rhythms in [(af, {'(AFIB'}), (af_flutter, {'(AFIB', '(AFL'})]:
                if span['rhythm'] in rhythms:
                    if chosen and chosen[-1][1] == a: chosen[-1] = (chosen[-1][0], b)
                    else: chosen.append((a, b))
        assert previous == h.sig_len
        represented = h.sig_len-known >= 30*fs
        if dataset == 'shdb': assert (metadata[rid]['Annotated'] == 'True') == bool(spans)
        duration = (h.sig_len-known)/fs;af_samples = sum(b-a for a, b in af)
        row = {'file_index': index, 'dataset': dataset, 'record_id': rid,
               'subject_id': metadata[rid]['Subject_ID'] if dataset == 'shdb' else '',
               'inference_group': metadata[rid]['Subject_ID'] if dataset == 'shdb' else 'record_'+rid,
               'patient_linkage_verified': dataset == 'shdb', 'sampling_rate': fs,
               'signal_path': str(root/(rid+'.dat')), 'signal_sha256': checked['signal_sha256'],
               'source_samples': h.sig_len, 'initial_unknown_samples': known,
               'known_reference_seconds': duration, 'input_contract_eligible': represented,
               'af_intervals': len(af), 'af_seconds': af_samples/fs,
               'af_burden_known_suffix': af_samples/(h.sig_len-known) if duration else None,
               'longest_af_seconds': max((b-a)/fs for a, b in af) if af else 0.,
               'reference_density_continuous_known_suffix': charitos_density([(max(a, known)/fs-known/fs, b/fs-known/fs) for a, b in af], duration) if duration else None,
               'af_flutter_intervals': len(af_flutter), 'geometry_verified': False,
               'external_categorical_ids': '-1,-1', 'clinical_signal_quality_adjudicated': False,
               'released_header_discrepancy': not all(checked['header_checksum_matches']) or not checked['initial_value_matches']}
        if represented:
            inputs, starts, ends = native_window_layout(h.sig_len, known, fs)
            count = np.zeros(len(inputs), np.int64);assigned = np.zeros(len(inputs), np.int64)
            for a, b in af:
                count += np.maximum(0, np.minimum(inputs+30*fs, b)-np.maximum(inputs, a))
                assigned += np.maximum(0, np.minimum(ends, b)-np.maximum(starts, a))
            assert assigned.sum() == af_samples
            arrays.append({'file_index': np.full(len(inputs), index, np.int16), 'input_start': inputs,
                'assigned_start': starts, 'assigned_end': ends, 'af_assigned_samples': assigned,
                'target': (count*2 >= 30*fs).astype(np.uint8)})
            row['windows'] = len(inputs)
        else: row['windows'] = 0
        files.append(row)
    for name, rows in [('file_manifest.csv', files), ('reference_intervals.csv', references)]:
        with (output/name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'windows.npz', **{k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    summary = {'dataset': dataset, 'complete_input_manifest': True, 'source_records': len(files),
        'eligible_records': sum(r['input_contract_eligible'] for r in files),
        'declared_windows': sum(r['windows'] for r in files), 'multi_episode_records': sum(r['af_intervals'] >= 2 for r in files),
        'multi_episode_inference_groups': len({r['inference_group'] for r in files if r['af_intervals'] >= 2}),
        'inference_unit': 'subject' if dataset == 'shdb' else 'record; patient linkage absent',
        'input_hashes': {name: sha(output/name) for name in ['file_manifest.csv', 'reference_intervals.csv', 'windows.npz']},
        'source_label_table_sha256': sha(label_path), 'source_integrity_audit_sha256': sha(root/'waveform_integrity_audit.json'),
        'source_hashes': {str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('external_waveform_inputs.py')]},
        'input_policy': 'Actual 30-second native-clock inputs; model200Hz float32 mV. LTAF polyphase25/16 Kaiser5 constant-zero FIR edge extension; no new observed time. Native annotation/assignment coordinates unchanged. Initial unknown prefix retained in record ledger and excluded from known-suffix estimand; whole-record density remains indeterminate.',
        'missing_policy': 'Any nonfinite native value makes that channel unavailable for the local input. No repair/imputation. Every declared condition remains in evaluation denominator.',
        'normalization': 'Use internal training-only raw normalization, Phase-KME kernel and hybrid RR scaling unchanged; no external fitting.',
        'diagnostic_predictions_generated': False}
    (output/'input_contract.json').write_text(json.dumps(summary, indent=2)+'\n');print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('dataset', choices=['shdb', 'ltaf'])
    main(parser.parse_args().dataset)
