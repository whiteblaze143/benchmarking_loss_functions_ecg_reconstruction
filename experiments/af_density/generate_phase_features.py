"""Frozen, resumable full-cohort Phase-KME cache; no diagnostic predictions."""
import argparse
import csv
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import h5py
import numpy as np
import torch
from waveform_protocol import W, sha
from phase_representation import phase_atoms, phase_response
from torch_phase_response import TorchPhaseResponse
from batched_phase_response import batched_phase_response
from phase_atom_workers import prepared_chunks
from repecg.common.kernels import WhiteningTransform, NystromMap

F = W/'phase_features_v1'
K = W/'phase_kernel_v1'


def commit(prefix, arrays, provenance):
    response, valid = arrays['response'], arrays['phase_valid']
    assert response.dtype == np.float32 and response.shape == (len(valid), 2, 16, 128)
    assert valid.dtype == bool and valid.shape == (len(response), 2)
    assert np.isfinite(response[valid]).all() and np.isnan(response[~valid]).all()
    target = prefix.with_suffix('.npz');receipt_path = prefix.with_suffix('.json')
    assert not target.exists() and not receipt_path.exists(), 'Preserve committed/orphan output'
    temporary = prefix.with_suffix('.npz.partial')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **arrays)
    os.replace(temporary, target)
    receipt = {**provenance, 'windows': len(response), 'phase_valid_by_channel': valid.sum(axis=0).astype(int).tolist(),
               'both_channels_unavailable_windows': int((~valid.any(axis=1)).sum()), 'output_sha256': sha(target)}
    temporary = receipt_path.with_suffix('.json.partial');temporary.write_text(json.dumps(receipt, indent=2)+'\n')
    os.replace(temporary, receipt_path)
    return receipt


def verified_commit(prefix, provenance):
    path = prefix.with_suffix('.json')
    if not path.exists():
        assert not prefix.with_suffix('.npz').exists(), 'Orphan output requires inspection, not overwrite'
        return None
    receipt = json.loads(path.read_text())
    for name, value in provenance.items(): assert receipt[name] == value, name
    assert sha(prefix.with_suffix('.npz')) == receipt['output_sha256']
    return receipt


def freeze():
    assert not F.exists(), 'Preserve frozen feature-generation protocol'
    sources = [Path(__file__), Path(__file__).with_name('phase_representation.py'),
               Path(__file__).with_name('torch_phase_response.py'),
               Path(__file__).with_name('batched_phase_response.py'),
               Path(__file__).with_name('phase_atom_workers.py'),
               Path('experiments/ptbxl_distributional_repecg/src/repecg/common/phase.py'),
               Path('experiments/ptbxl_distributional_repecg/src/repecg/common/kernels.py')]
    protocol = {
        'stage': 'Fixed-transform all-window feature generation only; no classifier or threshold selection',
        'declared_files': 289, 'declared_windows': 802491, 'stored_channels': [0, 1],
        'source_hashes': {str(p.resolve()): sha(p) for p in sources},
        'input_hashes': {name: sha(W/name) for name in ['input_contract.json', 'windows.npz', 'file_manifest.csv', 'preprocessing_freeze.json', 'phase_kernel_v1/protocol.json']},
        'dependency': 'Complete observed extraction and completed training-only phase kernel with passing approximation audit; kernel/shard hashes bound before generation',
        'features': 'Original PCHIP256; phase-index derivative;16cells;128 IMQ Nyström features. Original float64 whitening/map, torch float64 mean-before-inverse-root evaluation, final float32 cache.',
        'failure': 'Failed/insufficient phase extraction retains row, source counts/coverage and phase_valid=False; all corresponding feature cells NaN. No fabricated beats or zero-valued successful representation.',
        'numeric_audit': 'First successful window per file/channel checked against original NumPy mean-of-features, maximum float32 absolute discrepancy<=1e-7; failure aborts before commit',
        'resume': 'Reuse only hash-verified committed shards with identical protocol/kernel/HDF5/observed-peak provenance; no silent orphan overwrite',
        'test_policy': 'Test waveform feature preprocessing permitted with frozen training-only transforms; no held-out labels used for fitting and no diagnostic prediction generated.',
        'runtime': 'Four spawned CPU preparation workers plus one CPU-thread/CUDA controller, each numerical thread1; bounded8 queued chunks of8 waveform windows/at most16 channel representations. Compute padding has zero feature weight, no added beats/coverage. No CPU fallback/optimizer.',
    }
    F.mkdir();(F/'shards').mkdir();(F/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'frozen': True, 'declared_windows': 802491}))


def make_arrays(voltage, starts, observed, transform, whitening, mapping, pool=None):
    n = len(starts);valid = observed['phase_valid']
    out = {'window_index': observed['window_index'], 'phase_valid': valid,
           'accepted_intervals': observed['accepted_intervals'], 'represented_fraction': observed['represented_fraction'],
           'response': np.full((n, 2, 16, 128), np.nan, np.float32)}
    errors = [None, None]
    for atoms_list, positions in prepared_chunks(voltage, starts, observed, pool):
        if not atoms_list: continue
        responses = batched_phase_response(transform, atoms_list)
        for atoms, response, (j, channel) in zip(atoms_list, responses, positions):
            if errors[channel] is None:
                errors[channel] = float(abs(phase_response(atoms, whitening, mapping)-response).max())
                assert errors[channel] <= 1e-7, 'Original NumPy feature agreement failed'
            out['response'][j, channel] = response
    return out, errors


def run():
    protocol = json.loads((F/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest, name
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest, name
    status = json.loads((W/'extraction_status.json').read_text())
    assert status.get('complete') and status['files'] == 289 and status['windows'] == 802491, 'Extraction incomplete'
    kernel_receipt = json.loads((K/'receipt.json').read_text())
    assert kernel_receipt['complete'] and sha(K/'kernel.npz') == kernel_receipt['kernel_sha256']
    assert sha(K/'protocol.json') == kernel_receipt['protocol_sha256']
    assert sha(K/'kernel_audit.json') == kernel_receipt['kernel_audit_sha256']
    assert json.loads((K/'kernel_audit.json').read_text())['passed']
    lock = {'protocol_sha256': sha(F/'protocol.json'), 'kernel_sha256': sha(K/'kernel.npz'),
            'kernel_receipt_sha256': sha(K/'receipt.json'), 'extraction_status_sha256': sha(W/'extraction_status.json')}
    if (F/'input_lock.json').exists(): assert json.loads((F/'input_lock.json').read_text()) == lock
    else: (F/'input_lock.json').write_text(json.dumps(lock, indent=2)+'\n')
    old = F/'status.json'
    if old.exists():
        previous = json.loads(old.read_text());pid = previous.get('pid')
        command = Path(f'/proc/{pid}/cmdline')
        if previous['status'] == 'running' and command.exists():
            assert 'generate_phase_features.py' not in command.read_bytes().replace(b'\0', b' ').decode(), 'Feature process still live'
    for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert os.environ.get(variable) == '1', variable
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with np.load(K/'kernel.npz') as z:
        white = WhiteningTransform(z['mean'], z['components'], z['scales'])
        mapping = NystromMap(z['landmarks'], z['inverse_root'], float(z['c2']))
    transform = TorchPhaseResponse(white, mapping, device='cuda')
    rows = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z: file_index, input_start = z['file_index'], z['input_start']
    old.write_text(json.dumps({'status': 'running', 'pid': os.getpid()})+'\n')
    (F/f'run_{os.getpid()}.json').write_text(json.dumps({'pid': os.getpid(), 'input_lock_sha256': sha(F/'input_lock.json'),
         'cuda_device': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
         'numerical_threads_per_process': 1, 'cpu_preparation_workers': 4,
         'scope': 'Fixed-transform feature generation; no diagnostic predictions'}, indent=2)+'\n')
    receipts = [];started = time.monotonic()
    keys = ['window_index', 'phase_valid', 'accepted_intervals', 'represented_fraction',
            'peak_offsets_0', 'peak_samples_0', 'peak_offsets_1', 'peak_samples_1']
    pool = ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn'))
    try:
        for row in rows:
            file_started = time.monotonic()
            fi = int(row['file_index']);source = W/'observed_features'/f'file_{fi:03d}.npz'
            observed_receipt = json.loads(source.with_suffix('.json').read_text())
            assert sha(source) == observed_receipt['output_sha256']
            assert observed_receipt['h5_sha256'] == row['h5_sha256']
            assert observed_receipt['preprocessing_freeze_sha256'] == sha(W/'preprocessing_freeze.json')
            provenance = {'feature_protocol_sha256': lock['protocol_sha256'], 'kernel_sha256': lock['kernel_sha256'],
                          'h5_sha256': row['h5_sha256'], 'observed_shard_sha256': observed_receipt['output_sha256'],
                          'file_index': fi, 'record_id': row['record_id'], 'partition': row['partition']}
            prefix = F/'shards'/f'file_{fi:03d}'
            receipt = verified_commit(prefix, provenance)
            if receipt is None:
                assert sha(row['h5_path']) == row['h5_sha256']
                with h5py.File(row['h5_path']) as h: voltage = h['ecg'].astype('float32')[:]
                assert voltage.shape == (int(row['source_samples']), 2)
                with np.load(source) as z: observed = {name: z[name] for name in keys}
                ids = observed['window_index'];assert np.array_equal(ids, np.flatnonzero(file_index == fi))
                arrays, errors = make_arrays(voltage, input_start[ids], observed, transform, white, mapping, pool)
                assert arrays['phase_valid'].sum(axis=0).astype(int).tolist() == observed_receipt['phase_valid_by_channel']
                receipt = commit(prefix, arrays, {**provenance, 'numpy_max_absolute_error_by_channel': errors,
                                 'compute_seconds_before_commit': time.monotonic()-file_started})
            receipts.append(receipt)
            progress = {'completed_files': len(receipts), 'completed_windows': sum(r['windows'] for r in receipts),
                        'declared_windows': 802491, 'seconds': time.monotonic()-started}
            temporary = F/'progress.json.partial';temporary.write_text(json.dumps(progress)+'\n');os.replace(temporary, F/'progress.json')
            print(json.dumps(progress), flush=True)
        totals = np.sum([r['phase_valid_by_channel'] for r in receipts], axis=0).astype(int).tolist()
        assert len(receipts) == 289 and sum(r['windows'] for r in receipts) == 802491
        assert totals == status['phase_valid_by_channel']
        old.write_text(json.dumps({'status': 'complete', 'complete': True, 'files': 289, 'windows': 802491,
                                 'phase_valid_by_channel': totals, 'seconds': time.monotonic()-started,
                                 'peak_gpu_allocated_gib_this_process': torch.cuda.max_memory_allocated()/2**30}, indent=2)+'\n')
    except Exception as e:
        old.write_text(json.dumps({'status': 'failed', 'error': type(e).__name__+': '+str(e), 'completed_files': len(receipts)})+'\n')
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def smoke():
    # An explicit disposable fixture: production kernel and cache remain untouched.
    from verify_set_gpu import main as numerical_witness
    numerical_witness()
    row = next(r for r in csv.DictReader((W/'file_manifest.csv').open()) if r['partition'] == 'train')
    with np.load(W/'windows.npz') as z: starts = z['input_start'].copy()
    keys = ['window_index', 'phase_valid', 'accepted_intervals', 'represented_fraction',
            'peak_offsets_0', 'peak_samples_0', 'peak_offsets_1', 'peak_samples_1']
    with np.load(W/'observed_features'/f"file_{int(row['file_index']):03d}.npz") as z:
        observed = {name: z[name] if name.startswith('peak_') else z[name][:256] for name in keys}
    with h5py.File(row['h5_path']) as h: voltage = h['ecg'].astype('float32')[:]
    offsets = observed['peak_offsets_0'];peaks = observed['peak_samples_0'][offsets[0]:offsets[1]]
    first = int(starts[observed['window_index'][0]])
    atoms = phase_atoms(voltage[first:first+6000, 0], peaks).reshape(-1, 2)
    rng = np.random.default_rng(42);pool = atoms[rng.choice(len(atoms), 2048, replace=False)]
    white = WhiteningTransform.fit(pool);mapping = NystromMap.fit(white.transform(pool), landmarks=128, c2=1., seed=42)
    transform = TorchPhaseResponse(white, mapping, device='cuda')
    tick = time.monotonic()
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        actual, errors = make_arrays(voltage, starts[observed['window_index']], observed, transform, white, mapping, pool)
        first_seconds = time.monotonic()-tick;tick = time.monotonic()
        replay, _ = make_arrays(voltage, starts[observed['window_index']], observed, transform, white, mapping, pool)
        warm_seconds = time.monotonic()-tick
        np.testing.assert_array_equal(actual['response'], replay['response'])
    with tempfile.TemporaryDirectory() as directory:
        real = Path(directory)/'actual_training_fixture'
        real_receipt = commit(real, actual, {'scope': 'temporary actual-training256-window fixture'})
        assert verified_commit(real, {'scope': 'temporary actual-training256-window fixture'}) == real_receipt
        with np.load(real.with_suffix('.npz')) as z:
            np.testing.assert_array_equal(z['response'], actual['response'])
        prefix = Path(directory)/'feature_fixture'
        valid = np.array([[True, False], [False, True]])
        response = np.full((2, 2, 16, 128), np.nan, np.float32);response[valid] = 1.
        arrays = {'window_index': np.array([1, 2], np.int32), 'phase_valid': valid, 'response': response}
        receipt = commit(prefix, arrays, {'scope': 'synthetic serialization fixture'})
        assert verified_commit(prefix, {'scope': 'synthetic serialization fixture'}) == receipt
        with np.load(prefix.with_suffix('.npz')) as z:
            np.testing.assert_array_equal(z['response'], response);np.testing.assert_array_equal(z['phase_valid'], valid)
    print('WITNESS_AF_PHASE_CACHE', json.dumps({'numerical_witness_passed': True, 'roundtrip_and_commit_verification': True,
          'actual_training_windows': 256, 'numpy_max_absolute_error_by_channel': errors,
          'first_pass_seconds_including_worker_startup_and_numpy_audits': first_seconds,
          'warm_pass_seconds_including_numpy_audits': warm_seconds, 'exact_cached_response_replay': True,
          'production_kernel_or_cache_written': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();group = parser.add_mutually_exclusive_group(required=True)
    for mode in ['freeze', 'run', 'smoke']: group.add_argument('--'+mode, action='store_true')
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.run: run()
    else: smoke()
