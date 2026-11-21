"""Observed-voltage external QRS extraction; reference beats never enter here."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import wfdb
from external_waveform_inputs import model_window
from observed_waveform import extract_channel

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
O = W/'external_observed_v1'
DATA = {}


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def initialize():
    for dataset in ['shdb', 'ltaf']:
        folder = W/('external_'+dataset+'_v1')
        with np.load(folder/'windows.npz') as z:
            DATA[dataset] = {k: z[k] for k in ['file_index', 'input_start']}


def source_signal(row):
    path = Path(row['signal_path']);h = wfdb.rdheader(str(path.with_suffix('')))
    assert h.fmt == ['16', '16'] and h.n_sig == 2 and h.units == ['mV', 'mV']
    assert h.sig_len == int(row['source_samples']) and h.fs == int(row['sampling_rate'])
    assert all(not v for v in h.byte_offset) and all(not v for v in h.skew)
    assert np.isfinite(h.adc_gain).all() and min(h.adc_gain) > 0
    return np.memmap(path, mode='r', dtype='<i2', shape=(h.sig_len, 2)), h


def observe(digital, header, start):
    raw = digital[start:start+30*int(header.fs)]
    voltage = (raw.astype(np.float64)-np.asarray(header.baseline))/np.asarray(header.adc_gain)
    voltage[raw == -32768] = np.nan
    x, available = model_window(voltage, int(header.fs))
    return [extract_channel(x[:, c]) for c in range(2)], available


def process_file(item):
    dataset, row = item;fi = int(row['file_index']);folder = O/dataset
    prefix = folder/f'file_{fi:03d}';destination = prefix.with_suffix('.npz');receipt_path = prefix.with_suffix('.json')
    assert not destination.exists() and not receipt_path.exists(), 'Preserve prior attempt; no automatic overwrite'
    assert sha(row['signal_path']) == row['signal_sha256']
    protocol = json.loads((O/'protocol.json').read_text())
    header_path = Path(row['signal_path']).with_suffix('.hea')
    assert sha(header_path) == protocol['header_hashes'][str(header_path)]
    digital, header = source_signal(row);data = DATA[dataset]
    ids = np.flatnonzero(data['file_index'] == fi);starts = data['input_start'][ids];n = len(ids)
    out = {'window_index': ids.astype(np.int32), 'rr_features': np.full((n, 2, 16), np.nan, np.float32),
           'rr_valid': np.zeros((n, 2), bool), 'phase_valid': np.zeros((n, 2), bool),
           'source_channel_available': np.zeros((n, 2), bool), 'accepted_intervals': np.zeros((n, 2), np.uint16),
           'represented_fraction': np.zeros((n, 2), np.float32), 'detector_status': np.zeros((n, 2), np.int8),
           'peak_counts': np.zeros((n, 2), np.uint16)}
    peaks = [[], []];offsets = [[0], [0]];errors = [];started = time.monotonic()
    for j, start in enumerate(starts):
        channels, available = observe(digital, header, int(start));out['source_channel_available'][j] = available
        for c, r in enumerate(channels):
            for key in ['rr_features', 'rr_valid', 'phase_valid', 'accepted_intervals', 'represented_fraction']:
                out[key][j, c] = r[key]
            out['detector_status'][j, c] = r['status'];out['peak_counts'][j, c] = len(r['peaks'])
            peaks[c].append(r['peaks'].astype(np.int16));offsets[c].append(offsets[c][-1]+len(r['peaks']))
            if r['error']: errors.append({'window_index': int(ids[j]), 'channel': c, 'error': r['error']})
    for c in range(2):
        out[f'peak_samples_{c}'] = np.concatenate(peaks[c]);out[f'peak_offsets_{c}'] = np.asarray(offsets[c], np.int64)
    temporary = destination.with_suffix('.npz.partial')
    with temporary.open('xb') as f: np.savez_compressed(f, **out)
    temporary.rename(destination)
    error_path = prefix.with_suffix('.errors.json');error_path.write_text(json.dumps(errors)+'\n')
    receipt = {'dataset': dataset, 'record_id': row['record_id'], 'file_index': fi, 'windows': n,
        'phase_valid_by_channel': out['phase_valid'].sum(axis=0).tolist(),
        'rr_valid_by_channel': out['rr_valid'].sum(axis=0).tolist(),
        'both_phase_unavailable_windows': int((~out['phase_valid'].any(axis=1)).sum()),
        'source_sha256': row['signal_sha256'], 'output_sha256': sha(destination), 'errors_sha256': sha(error_path),
        'protocol_sha256': sha(O/'protocol.json'), 'seconds': time.monotonic()-started}
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n');return receipt


def main(mode):
    items = []
    for dataset in ['shdb', 'ltaf']:
        folder = W/('external_'+dataset+'_v1')
        assert json.loads((folder/'independent_input_verification.json').read_text())['all_pass']
        items.extend((dataset, r) for r in csv.DictReader((folder/'file_manifest.csv').open()) if r['input_contract_eligible'] == 'True')
    if mode == 'freeze':
        O.mkdir(exist_ok=False)
        sources = [Path(__file__), Path(__file__).with_name('external_waveform_inputs.py'), Path(__file__).with_name('observed_waveform.py'), Path(__file__).with_name('rr_features.py')]
        protocol = {'workers': 2, 'records': 182, 'windows': 511919,
            'source_hashes': {str(p): sha(p) for p in sources},
            'input_hashes': {str(W/('external_'+d+'_v1')/name): sha(W/('external_'+d+'_v1')/name) for d in ['shdb', 'ltaf'] for name in ['input_contract.json', 'windows.npz', 'file_manifest.csv']},
            'header_hashes': {str(Path(r['signal_path']).with_suffix('.hea')): sha(Path(r['signal_path']).with_suffix('.hea')) for d, r in items},
            'clock': 'Peak coordinates are local200Hz model-input samples; native source assignments remain untouched in separate frozen manifests.',
            'detector': 'Unchanged internal extract_channel: observed channel only, XQRSlearnTrue, no invented peaks, unchanged RR and phase validity/coverage rules.',
            'missing': 'Any encoded missing sample makes its native input channel unavailable. No interpolation. Every declared window retained.',
            'forbidden_inputs': 'No reference QRS, rhythm labels, AF targets, patient outcomes or external fitting enter observe/process_file.'}
        (O/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n');return
    protocol = json.loads((O/'protocol.json').read_text())
    for group in ['source_hashes', 'input_hashes', 'header_hashes']:
        for name, digest in protocol[group].items(): assert sha(name) == digest
    if mode == 'smoke':
        initialize();checks = []
        for dataset in ['shdb', 'ltaf']:
            row = next(r for d, r in items if d == dataset);digital, header = source_signal(row)
            ids = np.flatnonzero(DATA[dataset]['file_index'] == int(row['file_index']))[:8]
            for wi in ids:
                start = int(DATA[dataset]['input_start'][wi]);a, available = observe(digital, header, start);b, _ = observe(digital, header, start)
                for c in range(2):
                    assert np.array_equal(a[c]['peaks'], b[c]['peaks']) and np.array_equal(a[c]['rr_features'], b[c]['rr_features'], equal_nan=True)
                checks.append({'dataset': dataset, 'window_index': int(wi), 'peaks': [len(r['peaks']) for r in a], 'available': available.tolist()})
        (O/'smoke.json').write_text(json.dumps({'all_pass': True, 'actual_windows': len(checks), 'repeat_peak_RR_bitwise': True, 'checks': checks}, indent=2)+'\n')
        print('EXTERNAL_OBSERVED_SMOKE_PASS', len(checks));return
    assert json.loads((O/'smoke.json').read_text())['all_pass']
    assert not (O/'status.json').exists(), 'Preserve prior run; no automatic retry'
    for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: assert os.environ[variable] == '1'
    for dataset in ['shdb', 'ltaf']: (O/dataset).mkdir()
    (O/'status.json').write_text(json.dumps({'status': 'running', 'pid': os.getpid()})+'\n')
    completed = [];started = time.monotonic()
    with ProcessPoolExecutor(max_workers=2, initializer=initialize) as pool:
        for future in as_completed([pool.submit(process_file, item) for item in items]):
            completed.append(future.result())
            progress = {'files': len(completed), 'declared_files': 182, 'windows': sum(r['windows'] for r in completed), 'seconds': time.monotonic()-started}
            (O/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress), flush=True)
    assert len(completed) == 182 and sum(r['windows'] for r in completed) == 511919
    (O/'status.json').write_text(json.dumps({'status': 'complete', 'complete': True, 'files': 182, 'windows': 511919, 'seconds': time.monotonic()-started})+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode', choices=['freeze', 'smoke', 'run'])
    main(parser.parse_args().mode)
