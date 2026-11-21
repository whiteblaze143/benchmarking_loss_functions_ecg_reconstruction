"""Full observed-channel RR baseline, train/validation only; no native RR input."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from observed_rr_design import rr_design, PATTERNS
from matched_selection import patient_sample_weights, select_validation_threshold
from waveform_protocol import W, sha

O = W/'observed_rr_hgb_v1'


def verify_protocol():
    protocol = json.loads((O/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest, name
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest, name
    return protocol


def freeze():
    assert not O.exists(), 'Preserve frozen RR baseline'
    sources = [Path(__file__), Path(__file__).with_name('observed_rr_design.py'),
               Path(__file__).with_name('matched_selection.py'), Path(__file__).with_name('rr_features.py')]
    protocol = {
        'stage': 'Observed-channel RR train/validation only; no held-out inference in runner',
        'features': 'Actual16 named local RR statistics from waveform_v1 observed_features;32 ordered channel feature columns plus2 acquisition masks and2 RR-valid flags. Missing/unobserved channel features remain NaN.',
        'dtype': 'Source and every fit/predict design matrix float32; never infer from float64 then cache float32.',
        'eligibility': 'At least one requested channel RR-valid. Otherwise score remains NaN/indeterminate and keeps validation/evaluation denominator.',
        'availability_patterns': PATTERNS,
        'availability_training': 'Each declared training window expanded into channel0/channel1/both equally; all eligible rows fitted. Same acquisition subsets as raw cycling schedule, different estimator optimization budget explicitly retained.',
        'weights': 'Assigned-sample weights renormalized across eligible acquisition examples to total1 per declared training patient. No class reweighting; no represented examples for a patient raises.',
        'estimator': {'max_iter': 200, 'learning_rate': .1, 'max_leaf_nodes': 31, 'min_samples_leaf': 20,
                      'l2_regularization': 0., 'max_bins': 255, 'early_stopping': False, 'random_state': 42, 'categorical_features': None},
        'checkpoint': 'Fixed200 iterations, no validation-selected early stopping; save model before validation predictions; exact fresh joblib reload replay required.',
        'threshold': 'Pool all3 validation acquisition patterns, preserve every assigned sample/class denominator including unavailable rows; fixed .01-.99 balanced-accuracy policy in matched_selection.',
        'test_policy': 'Do not read test feature shards, generate test predictions or evaluate external data in this runner.',
        'dependency': 'Complete289-file/802491-window waveform extraction before preparation.',
        'source_hashes': {str(p.resolve()): sha(p) for p in sources},
        'input_hashes': {name: sha(W/name) for name in ['input_contract.json', 'windows.npz', 'file_manifest.csv', 'patient_splits.csv', 'preprocessing_freeze.json']},
    }
    O.mkdir();(O/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'frozen': True, 'train_windows_declared': 504777, 'native_source_rr_used': False}))


def prepare():
    verify_protocol();assert not (O/'development_features.npz').exists()
    status = json.loads((W/'extraction_status.json').read_text())
    assert status.get('complete') and status['files'] == 289 and status['windows'] == 802491, 'Extraction incomplete'
    rows = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z: windows = {name: z[name] for name in z.files}
    partitions = {p: np.flatnonzero(np.isin(windows['file_index'], [int(r['file_index']) for r in rows if r['partition'] == p]))
                  for p in ['train', 'validation']}
    ids = np.r_[partitions['train'], partitions['validation']]
    assert len(partitions['train']) == 504777 and len(partitions['validation']) == 166047
    lookup = np.full(len(windows['target']), -1, np.int64);lookup[ids] = np.arange(len(ids))
    features = np.full((len(ids), 2, 16), np.nan, np.float32);valid = np.zeros((len(ids), 2), bool)
    filled = np.zeros(len(ids), bool);shards = {}
    for row in rows:
        if row['partition'] == 'test': continue
        fi = int(row['file_index']);source = W/'observed_features'/f'file_{fi:03d}.npz'
        receipt = json.loads(source.with_suffix('.json').read_text())
        assert sha(source) == receipt['output_sha256'] and row['h5_sha256'] == receipt['h5_sha256']
        assert receipt['preprocessing_freeze_sha256'] == sha(W/'preprocessing_freeze.json')
        with np.load(source) as z:
            expected = np.flatnonzero(windows['file_index'] == fi)
            assert np.array_equal(z['window_index'], expected)
            local = lookup[expected];assert not filled[local].any() and (local >= 0).all()
            assert z['rr_features'].dtype == np.float32 and z['rr_valid'].dtype == bool
            features[local] = z['rr_features'];valid[local] = z['rr_valid'];filled[local] = True
        shards[str(source)] = receipt['output_sha256']
    assert filled.all()
    np.savez_compressed(O/'development_features.npz', window_index=ids, rr_features=features, rr_valid=valid)
    receipt = {'complete': True, 'protocol_sha256': sha(O/'protocol.json'), 'cache_sha256': sha(O/'development_features.npz'),
               'extraction_status_sha256': sha(W/'extraction_status.json'), 'source_shard_hashes': shards,
               'development_windows': len(ids), 'test_feature_shards_read': False}
    (O/'cache_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n');print(json.dumps({k: v for k, v in receipt.items() if k != 'source_shard_hashes'}))


def train():
    protocol = verify_protocol();assert not (O/'fit_input_receipt.json').exists(), 'Preserve prior fit attempt'
    for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert os.environ.get(variable) == '1', variable
    cache = json.loads((O/'cache_receipt.json').read_text())
    assert sha(O/'development_features.npz') == cache['cache_sha256'] and cache['protocol_sha256'] == sha(O/'protocol.json')
    with np.load(O/'development_features.npz') as z: ids, rr, valid = z['window_index'], z['rr_features'], z['rr_valid']
    views = [rr_design(rr, valid, pattern) for pattern in PATTERNS]
    x = np.stack([v[0] for v in views], axis=1).reshape(-1, 36)
    eligible = np.stack([v[1] for v in views], axis=1).reshape(-1)
    assert x.dtype == np.float32
    with np.load(W/'windows.npz') as z:
        labels = z['target'][ids];duration = (z['assigned_end'][ids]-z['assigned_start'][ids]).astype(np.int64)
        af = z['af_assigned_samples'][ids].astype(np.int64);file_index = z['file_index'][ids]
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    patients = np.array([r['patient_id'] for r in manifest])[file_index]
    partitions = np.array([r['partition'] for r in manifest])[file_index]
    ntrain = int((partitions == 'train').sum());assert ntrain == 504777
    assert (partitions[:ntrain] == 'train').all() and (partitions[ntrain:] == 'validation').all()
    train_eligible = eligible[:ntrain*3];val_eligible = eligible[ntrain*3:]
    train_patients = np.repeat(patients[:ntrain], 3)
    weight = patient_sample_weights(train_patients, np.repeat(duration[:ntrain], 3), train_eligible)
    assert len(np.unique(train_patients)) == 90
    training_x = np.ascontiguousarray(x[:ntrain*3][train_eligible], dtype=np.float32)
    training_y = np.repeat(labels[:ntrain], 3)[train_eligible]
    training_w = weight[train_eligible]
    receipt = {'protocol_sha256': sha(O/'protocol.json'), 'cache_receipt_sha256': sha(O/'cache_receipt.json'),
               'training_input_sha256': hashlib.sha256(training_x.tobytes()).hexdigest(),
               'training_target_sha256': hashlib.sha256(training_y.tobytes()).hexdigest(),
               'training_weight_sha256': hashlib.sha256(training_w.tobytes()).hexdigest(),
               'declared_training_windows': ntrain, 'declared_training_acquisition_examples': ntrain*3,
               'fitted_training_examples': len(training_y), 'indeterminate_training_examples': int((~train_eligible).sum()),
               'patient_weight_totals': {p: float(weight[train_patients == p].sum()) for p in np.unique(train_patients)},
               'fit_dtype': str(training_x.dtype), 'test_predictions': False}
    (O/'fit_input_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    started = time.monotonic();model = HistGradientBoostingClassifier(**protocol['estimator'])
    model.fit(training_x, training_y, sample_weight=training_w)
    assert model.n_iter_ == 200
    joblib.dump(model, O/'model.joblib')
    model_sha = sha(O/'model.joblib')
    validation_x = np.ascontiguousarray(x[ntrain*3:][val_eligible], dtype=np.float32)
    scores = np.full(len(val_eligible), np.nan)
    scores[val_eligible] = model.predict_proba(validation_x)[:, 1]
    replay = joblib.load(O/'model.joblib').predict_proba(validation_x)[:, 1]
    assert np.array_equal(replay, scores[val_eligible])
    np.save(O/'validation_scores.npy', scores.reshape(-1, 3))
    threshold = select_validation_threshold(scores, np.repeat(af[ntrain:], 3), np.repeat(duration[ntrain:], 3),
                 np.repeat(patients[ntrain:], 3), allowed_validation_patients=np.unique(patients[ntrain:]))
    (O/'threshold_selection.json').write_text(json.dumps(threshold, indent=2)+'\n')
    lock = {'complete_train_validation': True, 'test_inference_performed': False, 'model_sha256': model_sha,
            'protocol_sha256': sha(O/'protocol.json'), 'fit_input_receipt_sha256': sha(O/'fit_input_receipt.json'),
            'validation_scores_sha256': sha(O/'validation_scores.npy'), 'threshold': threshold['selected']['threshold'],
            'training_iterations': model.n_iter_, 'fresh_reload_validation_scores_bitwise': True,
            'validation_windows': len(ids)-ntrain, 'validation_indeterminate_acquisition_examples': int((~val_eligible).sum()),
            'seconds_including_fit_validation_selection': time.monotonic()-started}
    (O/'evaluation_lock.json').write_text(json.dumps(lock, indent=2)+'\n');print(json.dumps(lock, indent=2))


def smoke():
    row = next(r for r in csv.DictReader((W/'file_manifest.csv').open()) if r['partition'] == 'train')
    source = W/'observed_features'/f"file_{int(row['file_index']):03d}.npz"
    with np.load(source) as z: rr, valid = z['rr_features'][:32], z['rr_valid'][:32]
    designs = [rr_design(rr, valid, p) for p in PATTERNS]
    x = np.concatenate([a for a, b in designs]);eligible = np.concatenate([b for a, b in designs])
    y = np.arange(len(x)) % 2
    model = HistGradientBoostingClassifier(max_iter=3, min_samples_leaf=2, early_stopping=False, random_state=42)
    model.fit(x[eligible], y[eligible]);scores = model.predict_proba(x[eligible])[:, 1]
    assert np.isfinite(scores).all() and x.dtype == np.float32
    print('WITNESS_AF_OBSERVED_RR', json.dumps({'training_source_windows': len(rr), 'acquisition_examples': len(x),
          'shape': list(x.shape), 'dtype': str(x.dtype), 'source_native_rr_used': False,
          'scope': 'Temporary actual-training feature fixture with synthetic labels; not production clinical classifier fit'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();group = parser.add_mutually_exclusive_group(required=True)
    for mode in ['freeze', 'prepare', 'train', 'smoke']: group.add_argument('--'+mode, action='store_true')
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.prepare: prepare()
    elif args.train: train()
    else: smoke()
