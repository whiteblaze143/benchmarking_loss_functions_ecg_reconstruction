"""Compare saved development training histories with a training-only constant predictor."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with np.load(W/'windows.npz') as z:
        labels = z['target']
    result = {'test_outcomes_accessed': False,
              'scope': ('Descriptive training-only target baseline using all declared training examples and '
                        'eligibility-weighted masks. Logged training losses use stochastic channel masks and dropout; '
                        'differences are not paired evaluations and do not establish a causal mechanism.'),
              'models': {}}
    for model in ['categorical', 'q_ablated', 'hybrid']:
        root = W/'set_models_math_sdp_v2'/model
        manifest = root/'development_manifest.npz';history = root/'history.json';lock = root/'evaluation_lock.json'
        with np.load(manifest) as z:
            indices = z['window_index'][:504777]
            weights = z['train_weight']*z['eligibility'][:504777]
        target = labels[indices]
        assert set(np.unique(target)).issubset({0, 1}) and abs(weights.sum()-90)<1e-8
        rate = float((weights*target[:, None]).sum()/weights.sum())
        constant_bce = -rate*math.log(rate)-(1-rate)*math.log(1-rate)
        hist = json.loads(history.read_text());selected = json.loads(lock.read_text())['best_epoch']
        assert selected == min(hist, key=lambda row: row['validation_bce_all_rows'])['epoch']
        selected_row = next(row for row in hist if row['epoch'] == selected)
        result['models'][model] = {'training_window_count': len(indices), 'patient_weight_sum': float(weights.sum()),
            'weighted_training_target_prevalence': rate, 'optimal_constant_training_bce': constant_bce,
            'first_epoch_logged_training_bce': hist[0]['training_weighted_bce'],
            'last_epoch_logged_training_bce': hist[-1]['training_weighted_bce'],
            'selected_epoch': selected, 'selected_epoch_logged_training_bce': selected_row['training_weighted_bce'],
            'selected_epoch_validation_bce': selected_row['validation_bce_all_rows'],
            'manifest_sha256': sha(manifest), 'history_sha256': sha(history), 'lock_sha256': sha(lock)}
    result['source_sha256'] = sha(Path(__file__))
    out = W/'pre_release_corrective_v1/constant_score_training_diagnostic_v1.json'
    assert not out.exists();out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
