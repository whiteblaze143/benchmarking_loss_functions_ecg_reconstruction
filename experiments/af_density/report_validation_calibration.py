"""Descriptive calibration of frozen local AF probabilities, without refitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_validation_horizons import MODELS

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(model):
    root = W/MODELS[model];lock = json.loads((root/'evaluation_lock.json').read_text())
    assert lock['complete_train_validation'] and sha(root/'validation_scores.npy') == lock['validation_scores_sha256']
    evaluation = W/'evaluation_protocol_v1';protocol = json.loads((evaluation/'protocol.json').read_text())
    assert sha(evaluation/'protocol.json') == json.loads((evaluation/'freeze.json').read_text())['protocol_sha256']
    assert sha(W/'input_contract.json') == protocol['input_contract_sha256']
    contract = json.loads((W/'input_contract.json').read_text())
    for name, digest in contract['input_hashes'].items(): assert sha(W/name) == digest
    with np.load(root/('development_features.npz' if model == 'rr' else 'development_manifest.npz')) as z: ids = z['window_index'][504777:]
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z:
        expected = np.concatenate([np.flatnonzero(z['file_index'] == int(r['file_index'])) for r in files if r['partition'] == 'validation'])
        assert np.array_equal(ids, expected) and len(ids) == 166047
        y = np.repeat(z['target'][ids], 3).astype(np.float64)
        duration = np.repeat(z['assigned_end'][ids]-z['assigned_start'][ids], 3).astype(np.float64)
        patient = np.repeat(np.array([r['patient_id'] for r in files])[z['file_index'][ids]], 3)
    p = np.load(root/'validation_scores.npy').reshape(-1);available = np.isfinite(p)
    assert not np.isinf(p).any() and ((p[available] >= 0) & (p[available] <= 1)).all()
    weights = np.zeros(len(y));patients = sorted(set(patient));assert len(patients) == 30
    for name in patients:
        take = patient == name;weights[take] = duration[take]/duration[take].sum()
    assert abs(weights.sum()-30) < 1e-10
    squared = np.zeros(len(y));squared[available] = (p[available]-y[available])**2
    valid_weight = float(weights[available].sum());missing_weight = float(weights[~available].sum())
    bins = [];ece = 0.
    for index in range(10):
        take = available & (p >= index/10) & ((p < (index+1)/10) if index < 9 else (p <= 1))
        mass = float(weights[take].sum())
        mean_score = float(np.average(p[take], weights=weights[take])) if mass else None
        event_fraction = float(np.average(y[take], weights=weights[take])) if mass else None
        if mass: ece += mass/valid_weight*abs(mean_score-event_fraction)
        bins.append({'lower': index/10, 'upper': (index+1)/10, 'conditions': int(take.sum()),
                     'patient_weight_mass': mass, 'mean_probability': mean_score, 'observed_binary_AF_fraction': event_fraction})
    by_patient = {}
    for name in patients:
        take = patient == name;known = take & available;unknown = float(weights[take & ~available].sum())
        loss = float(np.dot(weights[take], squared[take]))
        by_patient[name] = {'conditional_available_Brier': float(np.average(squared[known], weights=weights[known])) if known.any() else None,
                            'missing_weight_fraction': unknown, 'all_reference_Brier_bounds': [loss, loss+unknown]}
    known_loss = float(np.dot(weights, squared))
    result = {'model': model, 'partition': 'validation', 'patients': 30, 'windows': 166047,
        'acquisition_conditions': len(p), 'indeterminate_conditions': int((~available).sum()),
        'conditional_available_patient_weighted_Brier': known_loss/valid_weight,
        'all_reference_patient_weighted_Brier_bounds': [known_loss/30, (known_loss+missing_weight)/30],
        'missing_patient_weight_fraction': missing_weight/30,
        'conditional_expected_calibration_error_10_fixed_bins': ece,
        'reliability_bins': bins, 'patient_results': by_patient,
        'model_lock_sha256': sha(root/'evaluation_lock.json'), 'source_sha256': sha(__file__),
        'scope': 'Binary local30s majority-AF target, assigned-duration/patient weights and equally expanded3 acquisition conditions. Not continuous-time rhythm or burden calibration. Missing predictions are not replaced by0.5; full-reference Brier bounds use the mathematical loss range[0,1]. No calibrator/threshold fitted, no test prediction and no proof of HSMM posterior calibration.'}
    path = evaluation/('validation_calibration_'+model+'.json');assert not path.exists()
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['reliability_bins', 'patient_results']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('model', choices=MODELS)
    main(parser.parse_args().model)
