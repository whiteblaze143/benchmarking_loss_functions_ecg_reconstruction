"""Reproduce the saved window-export defect without any Noise Holter input."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def native_cnn_probability(model, inputs, required_length):
    """Diagnostic repair candidate: native frozen head, no extra sigmoid.

    This does not assert that a contextual model's threshold is calibrated for
    its standalone CNN head. No clinical decisions are generated here.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] != required_length or not np.isfinite(inputs).all():
        raise ValueError('Invalid native input; no padding, resizing or silent replacement')
    result = tf.nest.flatten(model.fe(tf.constant(inputs[..., None]), training=False))
    if len(result) != 1 or result[0].shape[-1] != 1:
        raise ValueError('Unverified learned output structure')
    return result[0].numpy().reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tf.random.set_seed(42)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    root = args.output
    raw = np.load(root / 'development/target_independent_examples.npz')['raw']
    raw = np.concatenate([raw, raw * .8, raw * 1.2])
    rr = pd.read_csv(args.repository / 'external/AFtoolkit/detection/data/df_test.csv').iloc[:480, 0].to_numpy(np.float32).reshape(8, 60)
    report = {}
    for name, directory, inputs, length in [('RawECGNet', 'ArNetECG_exported_model', raw, 6000),
                                           ('ArNet2', 'exported_model', rr, 60)]:
        model = tf.saved_model.load(str(args.repository / 'external/AFtoolkit/detection' / directory))
        native = native_cnn_probability(model, inputs, length)
        exported = model.signatures['predict_windows'](x=tf.constant(inputs), threshold=tf.constant(.5))['probs'].numpy().reshape(-1)
        expected = 1 / (1 + np.exp(-native))
        np.testing.assert_allclose(exported, expected, rtol=1e-6, atol=1e-7)
        assert ((native >= 0) & (native <= 1)).all()
        contract_path = root / f'configs/model_contracts/{name}.json'
        contract = json.loads(contract_path.read_text())
        threshold = contract['output']['threshold']
        assert threshold < .5
        # This is a mathematical range proof, not a target-derived threshold.
        assert (exported > threshold).all()
        item = {'native_head_probability': native.tolist(), 'window_export_score': exported.tolist(),
                'maximum_error_export_vs_sigmoid_of_head': float(np.max(np.abs(exported - expected))),
                'source_fixed_threshold': threshold,
                'export_score_range_for_any_native_probability': [.5, float(1 / (1 + np.exp(-1)))],
                'native_head_calls_at_source_threshold_DIAGNOSTIC_ONLY': (native > threshold).astype(int).tolist(),
                'window_export_calls_at_source_threshold': (exported > threshold).astype(int).tolist(),
                'proof': 'Saved learned sigmoid probability is passed through a second sigmoid by predict_windows; no target ECG or labels used.',
                'repair_status': 'native_cnn_probability is executable and returns the original learned probability; its standalone-head operating point is not independently established',
                'scientific_admission': 'rejected_as_shipped_window_export'}
        report[name] = item
        contract['scientific_admission'] = item['scientific_admission']
        contract['output']['threshold_scope'] = 'recovered contextual predict_fixed threshold; not independently verified for standalone CNN probability'
        contract['window_export_semantics'] = 'sigmoid(already_sigmoid_probability), range[0.5,0.731059]; not native calibrated probability'
        contract['window_signature'] = 'predict_windows rejected as shipped; native CNN repair candidate kept separate'
        contract_path.write_text(json.dumps(contract, indent=2))
        print('SEMANTIC_GATE', name, item['scientific_admission'], flush=True)
    (root / 'reports/window_export_semantic_failure.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
