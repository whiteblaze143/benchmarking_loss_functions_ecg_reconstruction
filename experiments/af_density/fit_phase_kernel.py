"""Freeze training-only Phase-KME kernel fitting, then fit after extraction."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np
from waveform_protocol import W, sha
from phase_representation import phase_atoms, phase_response
from repecg.common.kernels import WhiteningTransform, NystromMap, imq_kernel, biased_mmd2, audit_nystrom

K = W/'phase_kernel_v1'


def linear_mean_response(atoms, whitening, mapping):
    """Same linear feature mean as the authentic map, in a cheaper order.

    mean(K(X,L) @ G) = mean(K(X,L)) @ G. All operations before the final
    float32 cache conversion remain float64. This is not bitwise equivalence;
    the numerical equivalence witness is required before production use.
    """
    values = np.asarray(atoms)
    if values.ndim != 3 or values.shape[0] != 16 or values.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite 16-cell two-coordinate atoms')
    if mapping.landmarks.shape != (128, 2):
        raise ValueError('Expected original 128-landmark two-coordinate map')
    response = np.stack([imq_kernel(whitening.transform(cell), mapping.landmarks, c2=mapping.c2).mean(axis=0)
                         @ mapping.inverse_root for cell in values])
    assert response.shape == (16, 128) and np.isfinite(response).all()
    return response.astype(np.float32)


def freeze():
    assert not K.exists(), 'Preserve frozen phase-kernel protocol'
    sources = [Path(__file__), Path(__file__).with_name('phase_representation.py'),
               Path('experiments/ptbxl_distributional_repecg/src/repecg/common/phase.py'),
               Path('experiments/ptbxl_distributional_repecg/src/repecg/common/kernels.py')]
    protocol = {
        'stage': 'Training-only unsupervised kernel fitting; no classifier fitting or held-out prediction',
        'patients': 90, 'stored_channels': [0, 1], 'windows_per_patient_channel': 32,
        'atoms_per_selected_window': 128,
        'selection': 'For each training patient and stored channel, sample 32 distinct phase-valid windows uniformly without replacement. Seed SHA256(af-phase-v1:patient_id:channel) first 8 bytes little endian. Draw one int64 atom_seed from that RNG per selected window; initialize a new RNG with atom_seed to sample128 distinct flattened phase atoms. Fewer32 valid windows raises.',
        'expected_channel_windows': 5760, 'expected_atoms': 737280,
        'classification_training_scope': 'Kernel sampling is not a cap on classifier windows. Full declared training cohort remains required.',
        'representation': 'Source mV float32, original PCHIP256, phase-index derivative,16 cells, original float64 WhiteningTransform and IMQ NystromMap128 c2=1 seed42.',
        'whitening_and_landmarks': 'Pooled equal-count atoms from every training patient/channel; no labels or validation/test waveforms used.',
        'feature_average': 'Authentic mean of Nyström features; linear mean-before-inverse-root equivalent implementation requires <=1e-7 float32 absolute discrepancy witness.',
        'kernel_audit': '64 seeded training sample-window pairs, seed20261121; original biased_mmd2 versus squared distance of original feature means; original audit_nystrom requires Spearman>=.90 and median relative error<=.15. Failed audit blocks readiness, without held-out tuning.',
        'continuous_geometry': 'Not admitted; no numeric q inferred from stored-channel identity.',
        'source_hashes': {str(p.resolve()): sha(p) for p in sources},
        'input_hashes': {n: sha(W/n) for n in ['input_contract.json', 'windows.npz', 'file_manifest.csv', 'patient_splits.csv', 'preprocessing_freeze.json', 'author_voltage_units_evidence.json']},
        'dependency': 'Full extraction complete=True with289files/802491windows; verify every selected shard receipt and source hash before fit.',
    }
    K.mkdir();(K/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'frozen': True, 'expected_channel_windows': 5760, 'expected_atoms': 737280}))


def fit():
    assert not (K/'kernel.npz').exists(), 'Kernel already fitted; do not overwrite'
    assert not (K/'fit_input_receipt.json').exists(), 'Preserve prior fit attempt; inspect before retry'
    for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert os.environ.get(variable) == '1', variable
    started = time.monotonic()
    protocol = json.loads((K/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest, name
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest, name
    status = json.loads((W/'extraction_status.json').read_text())
    assert status.get('complete') and status['files'] == 289 and status['windows'] == 802491, 'Extraction not complete'
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z: windows = {name: z[name] for name in z.files}
    valid = np.zeros((len(windows['target']), 2), bool)
    shard_hashes = {}
    for row in manifest:
        if row['partition'] != 'train': continue
        fi = int(row['file_index']);source = W/'observed_features'/f'file_{fi:03d}.npz'
        receipt = json.loads(source.with_suffix('.json').read_text())
        assert sha(source) == receipt['output_sha256'] and receipt['h5_sha256'] == row['h5_sha256']
        assert receipt['preprocessing_freeze_sha256'] == sha(W/'preprocessing_freeze.json')
        shard_hashes[str(source)] = receipt['output_sha256']
        with np.load(source) as z:
            ids = z['window_index'];assert np.array_equal(ids, np.flatnonzero(windows['file_index'] == fi))
            valid[ids] = z['phase_valid']
    patients = sorted({r['patient_id'] for r in manifest if r['partition'] == 'train'})
    assert len(patients) == protocol['patients']
    selections = []
    for patient in patients:
        files = [int(r['file_index']) for r in manifest if r['patient_id'] == patient]
        patient_mask = np.isin(windows['file_index'], files)
        for channel in [0, 1]:
            seed = int.from_bytes(hashlib.sha256(f'af-phase-v1:{patient}:{channel}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            eligible = np.flatnonzero(patient_mask & valid[:, channel])
            assert len(eligible) >= 32, (patient, channel, len(eligible))
            for wi in rng.choice(eligible, 32, replace=False):
                selections.append({'patient_id': patient, 'channel': channel, 'window_index': int(wi),
                                   'atom_seed': int(rng.integers(0, 2**63))})
    # Group source reads without changing declared sample order.
    samples = [None]*len(selections)
    for row in manifest:
        fi = int(row['file_index']);selected = [(j, s) for j, s in enumerate(selections) if windows['file_index'][s['window_index']] == fi]
        if not selected: continue
        assert row['partition'] == 'train' and sha(row['h5_path']) == row['h5_sha256']
        source = W/'observed_features'/f'file_{fi:03d}.npz'
        with h5py.File(row['h5_path']) as h, np.load(source) as z:
            voltage = h['ecg'].astype('float32')[:];lookup = {int(wi): i for i, wi in enumerate(z['window_index'])}
            for j, s in selected:
                wi, channel = s['window_index'], s['channel'];local = lookup[wi];start = windows['input_start'][wi]
                offsets = z[f'peak_offsets_{channel}'];peaks = z[f'peak_samples_{channel}'][offsets[local]:offsets[local+1]]
                atoms = phase_atoms(voltage[start:start+6000, channel], peaks)
                assert atoms is not None
                flat = atoms.reshape(-1, 2);rng = np.random.default_rng(s['atom_seed'])
                samples[j] = flat[rng.choice(len(flat), 128, replace=False)]
    atoms = np.concatenate(samples).astype(np.float64)
    assert atoms.shape == (737280, 2) and np.isfinite(atoms).all()
    (K/'selection.json').write_text(json.dumps(selections)+'\n')
    fit_input = {'protocol_sha256': sha(K/'protocol.json'), 'selection_sha256': sha(K/'selection.json'),
                 'atom_array_float64_sha256': hashlib.sha256(atoms.tobytes()).hexdigest(),
                 'observed_shard_hashes': shard_hashes,
                 'extraction_status_sha256': sha(W/'extraction_status.json'),
                 'runtime_threads': {n: os.environ[n] for n in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}}
    (K/'fit_input_receipt.json').write_text(json.dumps(fit_input, indent=2)+'\n')
    whitening = WhiteningTransform.fit(atoms)
    mapping = NystromMap.fit(whitening.transform(atoms), landmarks=128, c2=1., seed=42)
    rng = np.random.default_rng(20261121);exact = [];approximate = []
    for i, j in rng.choice(len(samples), (64, 2), replace=False):
        a, b = whitening.transform(samples[i]), whitening.transform(samples[j])
        exact.append(biased_mmd2(a, b, c2=1.))
        approximate.append(float(np.square(mapping.mean(a)-mapping.mean(b)).sum()))
    audit = audit_nystrom(exact, approximate)
    (K/'kernel_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    assert audit['passed'], 'Training-only Nyström approximation audit failed'
    np.savez_compressed(K/'kernel.npz', mean=whitening.mean, components=whitening.components, scales=whitening.scales,
                        landmarks=mapping.landmarks, inverse_root=mapping.inverse_root, c2=mapping.c2)
    receipt = {'complete': True, 'training_patients': len(patients), 'channel_windows': len(selections), 'atoms': len(atoms),
               'protocol_sha256': sha(K/'protocol.json'), 'kernel_sha256': sha(K/'kernel.npz'),
               'selection_sha256': sha(K/'selection.json'), 'observed_shard_hashes': shard_hashes,
               'fit_input_receipt_sha256': sha(K/'fit_input_receipt.json'), 'kernel_audit_sha256': sha(K/'kernel_audit.json'),
               'seconds_including_source_loading': time.monotonic()-started}
    (K/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n');print(json.dumps({k: v for k, v in receipt.items() if k != 'observed_shard_hashes'}))


def smoke():
    row = next(r for r in csv.DictReader((W/'file_manifest.csv').open()) if r['partition'] == 'train')
    fi = int(row['file_index']);responses = []
    with np.load(W/'windows.npz') as z: starts = z['input_start'].copy()
    source = W/'observed_features'/f'file_{fi:03d}.npz'
    with h5py.File(row['h5_path']) as h, np.load(source) as z:
        for i, wi in enumerate(z['window_index'][:2]):
            voltage = h['ecg'][starts[wi]:starts[wi]+6000].astype(np.float32)
            for channel in [0, 1]:
                offsets = z[f'peak_offsets_{channel}'];peaks = z[f'peak_samples_{channel}'][offsets[i]:offsets[i+1]]
                atoms = phase_atoms(voltage[:, channel], peaks);assert atoms is not None
                responses.append(atoms)
    rng = np.random.default_rng(42);pool = np.concatenate([a.reshape(-1, 2) for a in responses])
    pool = pool[rng.choice(len(pool), 2048, replace=False)]
    whitening = WhiteningTransform.fit(pool);mapping = NystromMap.fit(whitening.transform(pool), landmarks=128, c2=1., seed=42)
    errors = [float(abs(phase_response(a, whitening, mapping)-linear_mean_response(a, whitening, mapping)).max()) for a in responses]
    assert max(errors) <= 1e-7
    print('WITNESS_AF_PHASE_KERNEL', json.dumps({'partition': 'train', 'channel_windows': len(responses),
          'shape': [16, 128], 'maximum_float32_absolute_difference': max(errors),
          'scope': 'Temporary training-only equivalence fixture; not production kernel fit'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();group = parser.add_mutually_exclusive_group(required=True)
    for mode in ['freeze', 'fit', 'smoke']: group.add_argument('--'+mode, action='store_true')
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.fit: fit()
    else: smoke()
