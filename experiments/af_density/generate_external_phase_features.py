"""Fixed internal-kernel external features; no labels, fitting or predictions."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from extract_external_observed import source_signal, sha
from external_waveform_inputs import model_window
from phase_atom_workers import prepare_chunk
from generate_phase_features import commit
from phase_representation import phase_response
from torch_phase_response import TorchPhaseResponse
from batched_phase_response import batched_phase_response
from repecg.common.kernels import WhiteningTransform, NystromMap

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
F = W/'external_phase_features_v1'


def items():
    return [(d, r) for d in ['shdb', 'ltaf'] for r in
            csv.DictReader((W/f'external_{d}_v1/file_manifest.csv').open())
            if r['input_contract_eligible'] == 'True']


def freeze():
    assert not F.exists()
    names = ['generate_external_phase_features.py', 'extract_external_observed.py',
             'external_waveform_inputs.py', 'phase_atom_workers.py', 'generate_phase_features.py',
             'phase_representation.py', 'torch_phase_response.py', 'batched_phase_response.py']
    paths = [Path(__file__).with_name(n) for n in names]
    paths += [Path('experiments/ptbxl_distributional_repecg/src/repecg/common')/n
              for n in ['phase.py', 'kernels.py']]
    inputs = ['phase_kernel_v1/kernel.npz', 'phase_kernel_v1/receipt.json',
              'phase_kernel_v1/kernel_audit.json', 'external_observed_v1/protocol.json',
              'external_observed_v1/status.json']
    inputs += [f'external_{d}_v1/{n}' for d in ['shdb', 'ltaf'] for n in
               ['input_contract.json', 'windows.npz', 'file_manifest.csv']]
    protocol = {'records': 182, 'windows': 511919,
        'source_hashes': {str(p.resolve()): sha(p) for p in paths},
        'input_hashes': {p: sha(W/p) for p in inputs},
        'kernel': 'Frozen internal training-only kernel; no external fit/scaling/selection.',
        'waveforms': 'Unchanged native-mV conversion and model_window; real30s inputs, local200Hz detected peaks; native reference clock untouched.',
        'features': 'Unchanged phase_atoms/Phase-KME:16cells x128 features; float64 GPU transform, float32 serialization, unavailable channels NaN.',
        'numerical_check': 'First valid window per file/channel against original NumPy, maximum absolute float32 difference<=1e-7.',
        'runtime': 'Four spawned preparation workers,1 numerical thread each;8 queued chunks of8windows. Release unused CUDA cache after each file to bound allocator fragmentation alongside training.',
        'failure_policy': 'Abort on mismatch or existing output; no retry, imputation, fabricated beats or automatic resume.',
        'scope': 'Fixed preprocessing only; no rhythm labels/reference QRS used in feature computation, diagnostic predictions or test release.'}
    F.mkdir();(F/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for d in ['shdb', 'ltaf']: (F/d).mkdir()


def chunks(digital, header, starts, observed, pool):
    def tasks():
        for first in range(0, len(starts), 8):
            last = min(first+8, len(starts));waveforms = [];peaks = []
            for j in range(first, last):
                raw = digital[int(starts[j]):int(starts[j])+30*int(header.fs)]
                mv = (raw.astype(np.float64)-np.asarray(header.baseline))/np.asarray(header.adc_gain)
                mv[raw == -32768] = np.nan
                x, available = model_window(mv, int(header.fs))
                assert np.array_equal(available, observed['source_channel_available'][j])
                waveforms.append(x);row = []
                for c in [0, 1]:
                    offsets = observed[f'peak_offsets_{c}']
                    row.append(observed[f'peak_samples_{c}'][offsets[j]:offsets[j+1]])
                peaks.append(row)
            yield first, np.asarray(waveforms), peaks, observed['phase_valid'][first:last], observed['accepted_intervals'][first:last]
    if pool is None:
        for task in tasks(): yield prepare_chunk(task)
        return
    pending = deque();iterator = iter(tasks())
    for _ in range(8):
        task = next(iterator, None)
        if task is None: break
        pending.append(pool.submit(prepare_chunk, task))
    while pending:
        result = pending.popleft().result();task = next(iterator, None)
        if task is not None: pending.append(pool.submit(prepare_chunk, task))
        yield result


def transform_file(dataset, row, transform, white, mapping, pool, limit=None):
    fi = int(row['file_index']);source = W/f'external_observed_v1/{dataset}/file_{fi:03d}.npz'
    receipt = json.loads(source.with_suffix('.json').read_text())
    assert sha(source) == receipt['output_sha256'] and receipt['source_sha256'] == row['signal_sha256']
    assert sha(row['signal_path']) == row['signal_sha256']
    header_path = str(Path(row['signal_path']).with_suffix('.hea'))
    observed_protocol = json.loads((W/'external_observed_v1/protocol.json').read_text())
    assert sha(header_path) == observed_protocol['header_hashes'][header_path]
    with np.load(source) as z: observed = {k: z[k] for k in z.files if k != 'rr_features'}
    with np.load(W/f'external_{dataset}_v1/windows.npz') as z:
        ids = np.flatnonzero(z['file_index'] == fi);starts = z['input_start'][ids]
    assert np.array_equal(ids, observed['window_index'])
    if limit is not None:
        ids = ids[:limit];starts = starts[:limit]
        for k in ['window_index', 'phase_valid', 'accepted_intervals', 'represented_fraction', 'source_channel_available']:
            observed[k] = observed[k][:limit]
    digital, header = source_signal(row);valid = observed['phase_valid'];errors = [None, None]
    out = {k: observed[k] for k in ['window_index', 'phase_valid', 'accepted_intervals', 'represented_fraction']}
    out['response'] = np.full((len(ids), 2, 16, 128), np.nan, np.float32)
    for atoms, positions in chunks(digital, header, starts, observed, pool):
        if not atoms: continue
        responses = batched_phase_response(transform, atoms)
        for a, response, (j, c) in zip(atoms, responses, positions):
            if errors[c] is None:
                errors[c] = float(abs(phase_response(a, white, mapping)-response).max())
                assert errors[c] <= 1e-7
            out['response'][j, c] = response
    assert np.isfinite(out['response'][valid]).all() and np.isnan(out['response'][~valid]).all()
    return out, errors, receipt['output_sha256']


def main(mode):
    if mode == 'freeze': return freeze()
    protocol = json.loads((F/'protocol.json').read_text())
    for p, digest in protocol['source_hashes'].items(): assert sha(p) == digest
    for p, digest in protocol['input_hashes'].items(): assert sha(W/p) == digest
    assert json.loads((W/'external_observed_v1/status.json').read_text())['complete']
    assert json.loads((W/'phase_kernel_v1/kernel_audit.json').read_text())['passed']
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']: assert os.environ[key] == '1'
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.cuda.mem_get_info()[0] >= 6*2**30
    with np.load(W/'phase_kernel_v1/kernel.npz') as z:
        white = WhiteningTransform(z['mean'], z['components'], z['scales'])
        mapping = NystromMap(z['landmarks'], z['inverse_root'], float(z['c2']))
    transform = TorchPhaseResponse(white, mapping, device='cuda')
    if mode == 'smoke':
        assert not (F/'smoke.json').exists();checks = []
        for dataset in ['shdb', 'ltaf']:
            row = next(r for d, r in items() if d == dataset)
            a, errors, _ = transform_file(dataset, row, transform, white, mapping, None, limit=8)
            b, _, _ = transform_file(dataset, row, transform, white, mapping, None, limit=8)
            assert np.array_equal(a['response'], b['response'], equal_nan=True)
            checks.append({'dataset': dataset, 'windows': len(a['response']), 'numpy_max_absolute_error_by_channel': errors})
        (F/'smoke.json').write_text(json.dumps({'all_pass': True, 'repeat_bitwise': True, 'checks': checks,
            'protocol_sha256': sha(F/'protocol.json')}, indent=2)+'\n')
        print(json.dumps(checks), flush=True);return
    smoke = json.loads((F/'smoke.json').read_text());assert smoke['all_pass'] and smoke['protocol_sha256'] == sha(F/'protocol.json')
    assert not (F/'status.json').exists() and shutil.disk_usage(F).free >= 12*2**30
    (F/'status.json').write_text(json.dumps({'status': 'running', 'pid': os.getpid()})+'\n')
    started = time.monotonic();completed = []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        for dataset, row in items():
            out, errors, observed_sha = transform_file(dataset, row, transform, white, mapping, pool)
            receipt = commit(F/dataset/f"file_{int(row['file_index']):03d}", out,
                {'dataset': dataset, 'file_index': int(row['file_index']), 'record_id': row['record_id'],
                 'feature_protocol_sha256': sha(F/'protocol.json'), 'kernel_sha256': sha(W/'phase_kernel_v1/kernel.npz'),
                 'source_sha256': row['signal_sha256'], 'observed_shard_sha256': observed_sha,
                 'numpy_max_absolute_error_by_channel': errors})
            completed.append(receipt);torch.cuda.empty_cache()
            progress = {'files': len(completed), 'windows': sum(r['windows'] for r in completed), 'seconds': time.monotonic()-started}
            (F/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress), flush=True)
    assert len(completed) == 182 and sum(r['windows'] for r in completed) == 511919
    (F/'status.json').write_text(json.dumps({'status': 'complete', 'complete': True, **progress})+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode', choices=['freeze', 'smoke', 'run'])
    main(parser.parse_args().mode)
