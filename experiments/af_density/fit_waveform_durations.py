"""Freeze and fit training-only censored duration laws; no ECG classifier fit."""
import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
from duration_fit import fit_duration_law
from waveform_protocol import W, sha

D = W/'duration_model_v1'
SUPPORT = 86400  # 30 days of 30-second steps; final bin contains longer durations.


def freeze():
    assert not D.exists(), 'Preserve the frozen duration protocol'
    files = [r for r in csv.DictReader((W/'file_manifest.csv').open()) if r['partition'] == 'train']
    counts = Counter(r['patient_id'] for r in files)
    rows = []
    by_patient = {patient: [0, 0] for patient in counts}
    with np.load(W/'windows.npz') as z:
        labels, file_index = z['target'], z['file_index']
        lengths = z['assigned_end'].astype(np.int64)-z['assigned_start']
        for file in files:
            selected = file_index == int(file['file_index'])
            states, assigned = labels[selected], lengths[selected]
            bounds = np.r_[0, np.flatnonzero(states[1:] != states[:-1])+1, len(states)]
            for a, b in zip(bounds[:-1], bounds[1:]):
                rows.append({'patient_id': file['patient_id'], 'record_id': file['record_id'],
                             'file_index': file['file_index'], 'state': int(states[a]),
                             'length_steps': int(b-a), 'left_censored': bool(a == 0),
                             'right_censored': bool(b == len(states)),
                             'file_likelihood_weight': 1/counts[file['patient_id']]})
            by_patient[file['patient_id']][0] += int(assigned[states == 1].sum())
            by_patient[file['patient_id']][1] += int(assigned.sum())
    assert len(files) == 181 and len(counts) == 90
    positive = float(np.mean([a/b for a, b in by_patient.values()]))
    assert 0 < positive < 1
    D.mkdir()
    with (D/'training_runs.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sources = [Path(__file__), Path(__file__).with_name('duration_fit.py'),
               Path(__file__).with_name('hsmm.py')]
    contract = {
        'stage': 'Training-annotation duration fit only; no classifier, calibration or test prediction',
        'files': len(files), 'patients': len(counts), 'run_observations': len(rows),
        'support_steps': SUPPORT, 'duration_unit': 'Ordered 30-second input windows; partial final assignment counts as one step',
        'family': 'Discrete rounded lognormal; first bin (0,1.5), last bin [support-.5,infinity) lumped at support',
        'censoring': 'Every segment initial/final run censored. Initial residual-life, terminal survival, both residual-survival; stationary observation assumption explicit.',
        'weights': 'Each file likelihood receives inverse number of training files for its patient. Sum run log-likelihood within each file; average file likelihood per patient. Not equal weight per episode.',
        'initial_occupancy': [1-positive, positive],
        'initial_occupancy_fit': 'Mean across training patients of target-assigned AF samples / assigned samples, no class balancing',
        'optimizer': 'L-BFGS-B, maxiter300 ftol1e-12 gtol1e-7; mu bounds log(.1)..log(2*support), sigma .1..5; failure raises, near-bound parameters reported',
        'source_hashes': {str(p.resolve()): sha(p) for p in sources},
        'input_hashes': {n: sha(W/n) for n in ['input_contract.json', 'windows.npz', 'file_manifest.csv', 'patient_splits.csv']},
        'training_runs_sha256': sha(D/'training_runs.csv'),
        'threads': 1, 'gpu': 0,
        'limitations': ['Parametric duration assumption is not established clinical physiology.',
                       'No shared true episode continuation is inferred across excluded calibration time.',
                       'No validation/test label is used for this fit; development decoder/emission selection remains pending.'],
    }
    (D/'protocol.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(json.dumps({'frozen': True, 'files': len(files), 'patients': len(counts), 'runs': len(rows)}))


def fit():
    assert not (D/'fitted_law.npz').exists() and not (D/'receipt.json').exists()
    protocol = json.loads((D/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items():
        assert sha(name) == digest, name
    for name, digest in protocol['input_hashes'].items():
        assert sha(W/name) == digest, name
    assert sha(D/'training_runs.csv') == protocol['training_runs_sha256']
    for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert os.environ.get(variable) == '1', variable
    rows = list(csv.DictReader((D/'training_runs.csv').open()))
    models, laws = [], []
    started = time.monotonic()
    for state in [0, 1]:
        selected = [r for r in rows if int(r['state']) == state]
        result = fit_duration_law(np.array([int(r['length_steps']) for r in selected]),
                                  np.array([r['left_censored'] == 'True' for r in selected]),
                                  np.array([r['right_censored'] == 'True' for r in selected]),
                                  [float(r['file_likelihood_weight']) for r in selected], support=protocol['support_steps'])
        laws.append(result.pop('pmf'));models.append({'state': state, **result})
    np.savez_compressed(D/'fitted_law.npz', duration_pmf=np.stack(laws), initial_occupancy=protocol['initial_occupancy'])
    receipt = {'complete': True, 'seconds': time.monotonic()-started, 'models': models,
               'protocol_sha256': sha(D/'protocol.json'), 'fitted_law_sha256': sha(D/'fitted_law.npz'),
               'runtime_threads': {n: os.environ[n] for n in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}}
    (D/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n');print(json.dumps(receipt, indent=2))


def smoke():
    rng = np.random.default_rng(61)
    lengths = np.clip(np.floor(rng.lognormal(2., .6, 1000)+.5), 1, 128).astype(int)
    result = fit_duration_law(lengths, np.zeros(1000, bool), np.zeros(1000, bool), np.ones(1000), support=128)
    assert abs(result['mu_log_steps']-2) < .1 and abs(result['sigma_log_steps']-.6) < .1
    print('WITNESS_AF_DURATION_FIT', json.dumps({k: v for k, v in result.items() if k != 'pmf'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();group = parser.add_mutually_exclusive_group(required=True)
    for mode in ['freeze', 'fit', 'smoke']:
        group.add_argument('--'+mode, action='store_true')
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.fit: fit()
    else: smoke()
