"""Target-independent author API witnesses and immutable model contracts."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def checkpoint_receipt(directory):
    files = {str(path.relative_to(directory)): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
             for path in sorted(directory.rglob('*')) if path.is_file()}
    return {'files': files, 'sha256': hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def fixed_arguments(x, waveform):
    n = len(x)
    args = {'x': tf.constant(x), 'prec_windows': tf.zeros(n, tf.int32),
            'glob_lab': tf.ones(n), 'ids': tf.constant([str(i) for i in range(n)])}
    if waveform:
        args['succ_windows'] = tf.zeros(n, tf.int32)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # A completed cohort build is an enforced prerequisite to loading models.
    integrity = json.loads((args.output / 'manifests/release_integrity.json').read_text())
    assert integrity['physical_strips'] == 6130 and not integrity['checksum_failures']
    tf.random.set_seed(42)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    detection = args.repository / 'external/AFtoolkit/detection'
    spec = importlib.util.spec_from_file_location('author_runner', detection / 'run_detection.py')
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    source_commit = subprocess.check_output(['git', '-C', str(detection), 'rev-parse', 'HEAD'], text=True).strip()
    raw = np.load(args.output / 'development/target_independent_examples.npz')['raw'].astype(np.float32)
    # Multiple physically plausible gains test the same source-native raw input
    # contract; these are development witnesses, not diagnostic test samples.
    raw = np.concatenate([raw, raw * .8, raw * 1.2], axis=0)
    rr_frame = pd.read_csv(detection / 'data/df_test.csv').iloc[:8 * 60]
    rr = rr_frame.iloc[:, 0].to_numpy(np.float32).reshape(8, 60)
    results = []
    for name, model_dir, x, waveform in [('RawECGNet', 'ArNetECG_exported_model', raw, True),
                                        ('ArNet2', 'exported_model', rr, False)]:
        directory = detection / model_dir
        checkpoint = checkpoint_receipt(directory)
        model = tf.saved_model.load(str(directory))
        signatures = {key: {'inputs': str(value.structured_input_signature), 'outputs': str(value.structured_outputs)}
                      for key, value in model.signatures.items()}
        fixed = model.signatures['predict_fixed']
        graph = fixed.graph.as_graph_def()
        constants = []
        for function in graph.library.function:
            if 'predict_fixed' not in function.signature.name:
                continue
            for node in function.node_def:
                if node.op == 'Const':
                    value = tf.make_ndarray(node.attr['value'].tensor)
                    if value.size == 1 and value.dtype.kind == 'f' and 0 < float(value.reshape(-1)[0]) < 1:
                        constants.append({'function': function.signature.name, 'name': node.name,
                                          'value': float(value.reshape(-1)[0])})
        if len(constants) != 1:
            raise ValueError(f'Cannot uniquely establish fixed source threshold: {constants}')
        threshold = constants[0]['value']
        started = time.perf_counter()
        metadata = fixed_arguments(x, waveform)
        native = fixed(**metadata)
        probabilities = native['probs'].numpy()
        predictions = native['pred'].numpy()
        assert len(probabilities) == len(x) and np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        # Recover the frozen threshold from the model graph, never from target
        # predictions. ArNet2 uses0.432074189, not the assumed0.5.
        explicit = model.signatures['serving_default'](**metadata, threshold=tf.constant(threshold))
        np.testing.assert_allclose(probabilities, explicit['probs'].numpy(), atol=1e-6)
        np.testing.assert_array_equal(predictions, explicit['pred'].numpy())
        window = model.signatures['predict_windows'](x=tf.constant(x), threshold=tf.constant(threshold))
        window_scores = window['probs'].numpy().reshape(-1)
        window_calls = window['pred'].numpy().reshape(-1)
        assert len(window_scores) == len(x) and np.isfinite(window_scores).all()
        np.testing.assert_array_equal(window_calls, (window_scores > threshold).astype(np.int32))
        # Actual upstream runner parity, not merely successful deserialization.
        if waveform:
            formatted = np.column_stack([x, np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x)), np.arange(len(x))])
            namespace = argparse.Namespace(saved_model=str(directory))
            author_result = author._run_arnet_ecg(formatted, namespace)
        else:
            namespace = argparse.Namespace(saved_model=str(directory), input_file=str(detection / 'data/df_test.csv'))
            author_result = author._run_arnet2(rr_frame, namespace)
            # Author groups this development example as one contiguous record.
            source_meta = fixed_arguments(x, False)
            source_meta.update(prec_windows=tf.range(len(x)), ids=tf.constant([str(rr_frame.iloc[0, 2])] * len(x)))
            native = fixed(**source_meta)
        np.testing.assert_allclose(author_result.proba.to_numpy(), native['probs'].numpy(), atol=1e-6)
        np.testing.assert_array_equal(author_result.pred.to_numpy(), native['pred'].numpy())
        # Changing an unused ground-truth-like metadata placeholder must not
        # change the released fixed prediction for isolated source windows.
        zeros = dict(metadata, glob_lab=tf.zeros(len(x)))
        np.testing.assert_allclose(fixed(**metadata)['probs'].numpy(), fixed(**zeros)['probs'].numpy(), atol=1e-6)
        contract = {'model_name': name, 'architecture_version': 'author TensorFlow SavedModel; full contextual and window-only signatures distinguished',
                    'source_repository': 'https://github.com/aim-lab/AFtoolkit', 'source_repository_commit': source_commit,
                    'checkpoint_path': str(directory), 'checkpoint_sha256': checkpoint['sha256'], 'checkpoint_files': checkpoint['files'],
                    'input': {'modality': 'raw_ecg' if waveform else 'RR_seconds', 'channels': 1 if waveform else None,
                              'sampling_rate_hz': 200 if waveform else None, 'duration_seconds': 30 if waveform else None,
                              'rr_count': None if waveform else 60, 'required_observed_peaks': None if waveform else 61,
                              'normalization': 'physical mV; no external normalization in pinned author notebook' if waveform else 'RR intervals in seconds, no interval padding or clipping',
                              'filtering': 'no external filtering in pinned author waveform notebook' if waveform else 'original paper uses epltd after bandpass; full waveform extraction admission remains separate',
                              'allow_anatomical_channel_alias': False},
                    'output': {'positive_class': 'AF_or_AFL', 'score_semantics': 'probability', 'threshold': threshold,
                               'threshold_provenance': 'released fixed/default API on target-independent examples, plus graph receipt',
                               'abstention_supported': False},
                    'signatures': signatures, 'fixed_threshold_graph_constants': constants,
                    'window_signature': 'predict_windows; exported fixed threshold; first-stage native model, not full contextual model',
                    'witness': {'rows': len(x), 'seconds': time.perf_counter() - started,
                                'author_runner_parity': True, 'fixed_vs_explicit_threshold_parity': True,
                                'window_scores': window_scores.tolist(), 'window_decisions': window_calls.tolist(),
                                'global_label_placeholder_invariance': True,
                                'devices': [device.name for device in tf.config.list_physical_devices()]},
                    'admission_scope': 'executable frozen score model; target waveform pipeline and original extraction contract must also pass before benchmark admission'}
        (args.output / f'configs/model_contracts/{name}.json').write_text(json.dumps(contract, indent=2))
        results.append({'model': name, 'checkpoint_sha256': checkpoint['sha256'], 'verified_output_rows': len(x), 'author_runner_parity': True})
        print('FROZEN_MODEL_WITNESS', json.dumps(results[-1]), flush=True)
    (args.output / 'reports/model_execution_witness.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
