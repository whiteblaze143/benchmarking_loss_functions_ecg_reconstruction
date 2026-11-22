"""Instrument an isolated reproduction; never repair or promote its checkpoint."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
import train_set_waveform as runner


def main():
    original = runner.S
    output = original.parent/'q_ablated_nonfinite_diagnostic_v1'
    output.mkdir(exist_ok=False)
    shutil.copyfile(original/'protocol.json', output/'protocol.json')
    (output/'diagnostic_contract.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Isolated exact-seed development reproduction with finite-gradient instrumentation. Not an admitted training retry; no checkpoint may be used for evaluation.',
        'original_failed_attempt': str(original/'q_ablated'),
        'test_inference': False}, indent=2)+'\n')
    runner.S = output
    original_forward = runner.forward
    original_step = torch.optim.AdamW.step
    current = {};steps = 0

    def record_forward(model, mode, response, available, phase_valid, ids, rr, rv, coverage, mean, sd):
        current.clear()
        current.update(model=model, inputs=[response, available, phase_valid, ids, rr, rv, coverage, mean, sd])
        result = original_forward(model, mode, response, available, phase_valid, ids, rr, rv, coverage, mean, sd)
        if model.training and not torch.isfinite(result).all():
            capture('nonfinite_training_logits')
        return result

    def capture(reason):
        model = current['model']
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        torch.save({'state_dict': {n: p.detach().cpu() for n, p in model.state_dict().items()},
                    'inputs': [x.detach().cpu() for x in current['inputs']],
                    'gradients': {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None},
                    'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()}, output/'first_nonfinite.pt')
        (output/'finding.json').write_text(json.dumps({'reason': reason, 'completed_optimizer_steps': steps,
            'nonfinite_gradient_parameters': bad, 'test_access': False}, indent=2)+'\n')
        raise RuntimeError(reason+'; diagnostic captured, no retry or repair')

    def checked_step(optimizer, *args, **kwargs):
        nonlocal steps
        checks = [torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if not torch.stack(checks).all():
            capture('nonfinite_gradient_before_optimizer_step')
        result = original_step(optimizer, *args, **kwargs)
        steps += 1
        return result

    runner.forward = record_forward
    torch.optim.AdamW.step = checked_step
    runner.train('q_ablated')
    (output/'finding.json').write_text(json.dumps({'reason': 'original_failure_not_reproduced',
        'completed_optimizer_steps': steps, 'admitted_for_evaluation': False}, indent=2)+'\n')


if __name__ == '__main__':
    main()
