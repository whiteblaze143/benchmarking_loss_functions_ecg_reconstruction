"""Small actual-training feature and authentic SetOperator GPU witness only."""
import csv
import json
import h5py
import numpy as np
import torch
from torch import nn
from waveform_protocol import W
from phase_representation import phase_atoms, phase_response
from repecg.common.kernels import WhiteningTransform, NystromMap
from torch_phase_response import TorchPhaseResponse
from set_waveform_model import AFSetOperator
from train_raw_waveform import setup


def main():
    setup()
    row = next(r for r in csv.DictReader((W/'file_manifest.csv').open()) if r['partition'] == 'train')
    with np.load(W/'windows.npz') as z: starts = z['input_start'].copy()
    fi = int(row['file_index']);atoms = []
    with h5py.File(row['h5_path']) as h, np.load(W/'observed_features'/f'file_{fi:03d}.npz') as z:
        for i, wi in enumerate(z['window_index'][:2]):
            voltage = h['ecg'][starts[wi]:starts[wi]+6000].astype(np.float32)
            for channel in [0, 1]:
                offsets = z[f'peak_offsets_{channel}'];peaks = z[f'peak_samples_{channel}'][offsets[i]:offsets[i+1]]
                a = phase_atoms(voltage[:, channel], peaks);assert a is not None;atoms.append(a)
    pool = np.concatenate([a.reshape(-1, 2) for a in atoms]);rng = np.random.default_rng(42)
    pool = pool[rng.choice(len(pool), 2048, replace=False)]
    white = WhiteningTransform.fit(pool);mapping = NystromMap.fit(white.transform(pool), landmarks=128, c2=1., seed=42)
    transform = TorchPhaseResponse(white, mapping, device='cuda')
    responses = [transform(a) for a in atoms]
    error = max(float(abs(phase_response(a, white, mapping)-r).max()) for a, r in zip(atoms, responses))
    assert error <= 1e-7
    x = torch.tensor(np.stack(responses).reshape(2, 2, 16, 128), device='cuda')
    available = torch.ones((2, 2), dtype=torch.bool, device='cuda');valid = available.clone()
    ids = torch.tensor([[0, 1], [0, 1]], device='cuda');results = {}
    for mode in ['categorical', 'q_ablated']:
        torch.manual_seed(42);model = AFSetOperator(mode).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, available, valid, ids)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.], device='cuda'))
        loss.backward();assert torch.isfinite(loss)
        for name in ['response.input.weight', 'set_encoder.layers.0.linear1.weight', 'pool.in_proj_weight', 'head.weight']:
            gradient = dict(model.original.named_parameters())[name].grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        assert all(p.grad is None for p in model.original.decoder.parameters())
        model.eval()
        with torch.inference_mode():
            scores = model(x, available, valid, ids)
            permuted = model(x.flip(1), available.flip(1), valid.flip(1), ids.flip(1))
            permutation_error = float(abs(scores-permuted).max());assert permutation_error <= 1e-6
            one = available.clone();one[:, 1] = False
            changed = x.clone();changed[:, 1] = torch.inf
            assert torch.equal(model(x, one, valid, ids), model(changed, one, valid, ids))
            latent = model.encode(x, available, valid, ids);assert latent.shape == (2, 256)
        results[mode] = {'finite_bf16_gradient_loss': float(loss.detach()), 'permutation_max_absolute_error': permutation_error,
                         'unobserved_response_isolation': True, 'latent_shape': list(latent.shape),
                         'total_parameters': sum(p.numel() for p in model.parameters()),
                         'parameters_with_gradients': sum(p.numel() for p in model.parameters() if p.grad is not None),
                         'auxiliary_decoder_gradients': False}
    print('WITNESS_AF_SET_GPU', json.dumps({'device': torch.cuda.get_device_name(), 'partition': 'train',
          'channel_windows': 4, 'phase_response_shape': [16, 128], 'phase_float32_max_absolute_error': error,
          'models': results, 'peak_gpu_allocated_gib_this_process': torch.cuda.max_memory_allocated()/2**30,
          'scope': 'Temporary training-only kernel fixture and synthetic binary gradient loss; no production kernel or classifier fit, no held-out prediction'}))


if __name__ == '__main__':
    main()
