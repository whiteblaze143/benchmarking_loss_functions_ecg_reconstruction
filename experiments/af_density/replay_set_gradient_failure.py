"""Diagnose a captured finite-loss/nonfinite-gradient batch; no model fitting."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from set_waveform_model import AFSetOperator
from train_raw_waveform import setup

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def main():
    source = W/'q_ablated_nonfinite_diagnostic_v1/first_nonfinite.pt'
    output = W/'pre_release_corrective_v1/gradient_backend_diagnostic_v1.json'
    assert not output.exists()
    captured = torch.load(source, weights_only=True)
    assert all(torch.isfinite(v).all() for v in captured['state_dict'].values())
    setup();inputs = [x.cuda() for x in captured['inputs']]
    response, available, phase_valid, ids = inputs[:4]
    assert torch.isfinite(response[available & phase_valid]).all()
    results = []
    for precision in ['bf16', 'fp32']:
        for backend in ['default', 'math']:
            for label in ['zeros', 'ones', 'alternating']:
                for seed in [0, 42]:
                    model = AFSetOperator('q_ablated').cuda().train()
                    model.load_state_dict(captured['state_dict'])
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    target = (torch.zeros(len(response), device='cuda') if label == 'zeros' else
                              torch.ones(len(response), device='cuda') if label == 'ones' else
                              torch.arange(len(response), device='cuda').remainder(2).float())
                    context = nullcontext() if backend == 'default' else sdpa_kernel(SDPBackend.MATH)
                    with context, torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                        logits = model(response, available, phase_valid, ids)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                    loss.backward()
                    bad = [name for name, p in model.named_parameters()
                           if p.grad is not None and not torch.isfinite(p.grad).all()]
                    results.append({'precision': precision, 'backend': backend, 'synthetic_target': label,
                        'seed': seed, 'loss_finite': bool(torch.isfinite(loss)),
                        'gradients_finite': not bool(bad), 'nonfinite_gradient_parameters': bad})
    result = {'captured_state_all_finite': True, 'active_phase_inputs_all_finite': True,
        'batch_rows': len(response), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'captured_batch_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'results': results,
        'test_access': False, 'optimizer_updates': 0,
        'scope': 'Saved failing state and actual input batch; synthetic BCE targets and newly declared dropout seeds. This does not replay the original labels, patient weights or original dropout mask. It tests numerical behavior, not clinical performance or a registered training retry.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    for precision in ['bf16', 'fp32']:
        for backend in ['default', 'math']:
            selected = [r for r in results if r['precision'] == precision and r['backend'] == backend]
            print(json.dumps({'precision': precision, 'backend': backend,
                'finite_gradient_cases': sum(r['gradients_finite'] for r in selected), 'cases': len(selected)}), flush=True)


if __name__ == '__main__': main()
