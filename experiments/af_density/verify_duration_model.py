"""Independently rebuild training runs and replay the serialized duration laws."""
import csv
import json
from collections import Counter
import numpy as np
from scipy.stats import lognorm
from waveform_protocol import W, sha


def main():
    directory = W/'duration_model_v1'
    protocol = json.loads((directory/'protocol.json').read_text())
    receipt = json.loads((directory/'receipt.json').read_text())
    launch = json.loads((directory/'launch_receipt.json').read_text())
    for name, digest in launch['hashes'].items():
        assert sha(name) == digest, name
    for name, digest in protocol['source_hashes'].items():
        assert sha(name) == digest, name
    for name, digest in protocol['input_hashes'].items():
        assert sha(W/name) == digest, name
    assert sha(directory/'training_runs.csv') == protocol['training_runs_sha256']
    assert sha(directory/'protocol.json') == receipt['protocol_sha256']
    assert sha(directory/'fitted_law.npz') == receipt['fitted_law_sha256']
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    train = [r for r in manifest if r['partition'] == 'train']
    file_counts = Counter(r['patient_id'] for r in train)
    expected = []
    occupancy = {patient: [0, 0] for patient in file_counts}
    with np.load(W/'windows.npz') as z:
        for file in train:
            ids = np.flatnonzero(z['file_index'] == int(file['file_index']))
            states = z['target'][ids]
            current, start = int(states[0]), 0
            for stop in range(1, len(states)+1):
                if stop == len(states) or states[stop] != current:
                    expected.append((file['patient_id'], file['record_id'], int(file['file_index']),
                                     current, stop-start, start == 0, stop == len(states),
                                     1/file_counts[file['patient_id']]))
                    if stop < len(states):
                        current, start = int(states[stop]), stop
            assigned = z['assigned_end'][ids].astype(np.int64)-z['assigned_start'][ids]
            occupancy[file['patient_id']][0] += int(assigned[states == 1].sum())
            occupancy[file['patient_id']][1] += int(assigned.sum())
    rows = list(csv.DictReader((directory/'training_runs.csv').open()))
    observed = [(r['patient_id'], r['record_id'], int(r['file_index']), int(r['state']),
                 int(r['length_steps']), r['left_censored'] == 'True', r['right_censored'] == 'True',
                 float(r['file_likelihood_weight'])) for r in rows]
    assert observed == expected and len(expected) == 599
    patient_file_weight = {patient: sum(1/file_counts[patient] for file in train if file['patient_id'] == patient)
                           for patient in file_counts}
    assert np.allclose(list(patient_file_weight.values()), 1, rtol=0, atol=1e-12)
    with np.load(directory/'fitted_law.npz') as z:
        pmfs, prior = z['duration_pmf'], z['initial_occupancy']
    expected_af = np.mean([a/b for a, b in occupancy.values()])
    np.testing.assert_array_equal(prior, [1-expected_af, expected_af])
    assert pmfs.shape == (2, protocol['support_steps'])
    assert (pmfs >= 0).all() and np.isfinite(pmfs).all()
    np.testing.assert_allclose(pmfs.sum(axis=1), 1, rtol=0, atol=1e-12)
    max_difference = 0.
    nll_differences = []
    for model in receipt['models']:
        state = model['state'];support = protocol['support_steps']
        edge = np.r_[0., np.arange(1.5, support, 1), np.inf]
        law = lognorm(s=model['sigma_log_steps'], scale=np.exp(model['mu_log_steps']))
        cdf, sf = law.cdf(edge), law.sf(edge)
        reconstructed = np.where(edge[:-1] >= np.exp(model['mu_log_steps']), sf[:-1]-sf[1:], cdf[1:]-cdf[:-1])
        np.testing.assert_allclose(pmfs[state], reconstructed, rtol=1e-10, atol=1e-15)
        max_difference = max(max_difference, float(abs(pmfs[state]-reconstructed).max()))
        survival = np.cumsum(pmfs[state, ::-1])[::-1]
        residual = survival/np.dot(pmfs[state], np.arange(1, support+1))
        residual_survival = np.cumsum(residual[::-1])[::-1]
        log_likelihood, weight_total = 0., 0.
        selected = [r for r in observed if r[3] == state]
        for patient, record, fi, s, length, left, right, weight in selected:
            probability = (residual_survival if left and right else residual if left else
                           survival if right else pmfs[state])[length-1]
            assert probability > 0
            log_likelihood += weight*np.log(probability)
            weight_total += weight
        difference = abs(-log_likelihood/weight_total-model['weighted_mean_negative_log_likelihood'])
        assert difference < 1e-11
        nll_differences.append(difference)
        assert model['runs'] == len(selected)
    result = {'all_pass': True, 'rebuilt_training_files': len(train), 'training_patients': len(file_counts),
              'exactly_reconstructed_runs': len(expected), 'source_and_launch_hashes_match': True,
              'patient_total_file_likelihood_weight': '1 within 1e-12',
              'independent_pmf_max_absolute_difference': max_difference,
              'serialized_likelihood_absolute_differences': nll_differences,
              'scope': 'Reconstructs training-only observations and serialized parameter likelihood; does not establish clinical model validity or measured decoder performance'}
    (directory/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
