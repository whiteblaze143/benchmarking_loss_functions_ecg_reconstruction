"""Frozen-model, validation-only representation-variation diagnostic, no labels."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from set_waveform_model import AFSetOperator
from train_raw_waveform import setup

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def stats(x):
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    sd = x.std(axis=0)
    return {'mean_coordinate_std': float(sd.mean()), 'max_coordinate_std': float(sd.max()),
            'min': float(x.min()), 'max': float(x.max())}


def main():
    rows = list(csv.DictReader((W/'file_manifest.csv').open()))
    selected = [];patients = set()
    for row in rows:
        if row['partition'] == 'validation' and row['patient_id'] not in patients:
            selected.append(row);patients.add(row['patient_id'])
            if len(selected) == 4: break
    assert len(selected) == 4
    tensors = [];phase_masks = [];all_ids = [];files = []
    for row in selected:
        path = W/'phase_features_v1/shards'/f"file_{int(row['file_index']):03d}.npz"
        meta = json.loads(path.with_suffix('.json').read_text());assert sha(path) == meta['output_sha256']
        with np.load(path) as z:
            candidates = np.flatnonzero(z['phase_valid'][:, 0]);assert len(candidates) >= 64
            chosen = candidates[np.linspace(0, len(candidates)-1, 64, dtype=int)]
            tensors.append(z['response'][chosen]);phase_masks.append(z['phase_valid'][chosen]);all_ids.extend(z['window_index'][chosen].tolist())
        files.append({'file_index': int(row['file_index']), 'patient_id': row['patient_id'], 'source_sha256': sha(path)})
    response = np.concatenate(tensors);valid = np.concatenate(phase_masks);ids = np.array(all_ids)
    assert response.shape == (256, 2, 16, 128) and valid[:, 0].all()
    setup();device = 'cuda';x = torch.from_numpy(response).to(device);v = torch.from_numpy(valid).to(device)
    availability = torch.tensor([True, False], device=device).expand(len(x), -1)
    operators = torch.tensor([0, 1], device=device).expand(len(x), -1)
    result = {'partition': 'validation', 'test_access': False, 'labels_read': False,
        'sampling': '64 evenly spaced channel-0-valid windows from each of the first four distinct validation patients; not a random or representative cohort sample.',
        'files': files, 'windows': len(ids), 'phase_input': stats(response[:, 0]), 'models': {},
        'scope': 'Observational intermediate-activation variability for frozen models; does not establish root cause, generalization, training dynamics or a clinical mechanism.'}
    for mode in ['categorical', 'q_ablated']:
        root = W/'set_models_math_sdp_v2'/mode;lock = json.loads((root/'evaluation_lock.json').read_text())
        assert sha(root/'best.pt') == lock['best_checkpoint_sha256']
        ckpt = torch.load(root/'best.pt', weights_only=True);model = AFSetOperator(mode).to(device).eval();model.load_state_dict(ckpt['state_dict'])
        activations = [];encoded = [];pooled = [];layer_output = [[], []]
        hook = model.original.response.register_forward_hook(lambda module, args, out: activations.append(out.detach().cpu().numpy()))
        encoder_hook = model.original.set_encoder.register_forward_hook(lambda module, args, out: encoded.append(out.detach().cpu().numpy()))
        pool_hook = model.original.pool.register_forward_hook(lambda module, args, out: pooled.append(out[0].detach().cpu().numpy()))
        def hook_layer(i):
            def collect(module, args, out):
                if out.is_nested: out = out.to_padded_tensor(0.)
                layer_output[i].append(out.detach().cpu().numpy())
            return collect
        layer_hooks = [layer.register_forward_hook(hook_layer(i)) for i, layer in enumerate(model.original.set_encoder.layers)]
        with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
            latent = model.encode(x, availability, v, operators);logits = model.original.head(latent).squeeze(-1)
        hook.remove();encoder_hook.remove();pool_hook.remove()
        for handle in layer_hooks: handle.remove()
        raw = activations[0].reshape(len(ids), 2, -1)[:, 0]
        with np.load(root/'development_manifest.npz') as z: reference_ids = z['window_index'][504777:]
        idx = np.searchsorted(reference_ids, ids);assert np.array_equal(reference_ids[idx], ids)
        saved = np.load(root/'best_validation_logits.npy')[idx, 0]
        computed = logits.cpu().numpy();maximum = float(np.max(np.abs(computed-saved)))
        assert maximum < 1e-4
        result['models'][mode] = {'response_encoder': stats(raw), 'set_encoder_layer_1_active_token': stats(layer_output[0][0][:, 0]), 'set_encoder_layer_2_active_token': stats(layer_output[1][0][:, 0]), 'set_encoder_active_token': stats(encoded[0][:, 0]), 'pool': stats(pooled[0][:, 0]), 'latent': stats(latent.cpu().numpy()),
            'logit': stats(computed), 'head_weight_l2': float(model.original.head.weight.detach().norm().cpu()),
            'max_absolute_logit_difference_from_full_validation_replay': maximum,
            'checkpoint_sha256': lock['best_checkpoint_sha256']}
    result['source_sha256'] = sha(Path(__file__))
    out = W/'pre_release_corrective_v1/validation_activation_variation_v3.json';assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n');print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
