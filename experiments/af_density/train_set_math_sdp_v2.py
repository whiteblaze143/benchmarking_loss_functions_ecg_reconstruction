"""Registered backend correction; authentic Set architecture and full data unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import train_set_waveform as original
from set_waveform_model import AFSetOperator
from train_raw_waveform import setup

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
ROOT = W/'set_models_math_sdp_v2'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(mode):
    if mode == 'freeze':
        assert not ROOT.exists()
        evidence = W/'pre_release_corrective_v1/gradient_backend_diagnostic_v1.json'
        finding = json.loads(evidence.read_text())
        for r in finding['results']:
            assert r['gradients_finite'] == (r['precision'] != 'bf16' or r['backend'] == 'math')
        protocol = json.loads((W/'set_models_v1/protocol.json').read_text())
        protocol.update({'version': 'math_sdp_v2',
            'original_protocol_sha256': sha(W/'set_models_v1/protocol.json'),
            'backend_correction': 'Use torch.nn.attention.sdpa_kernel(SDPBackend.MATH) throughout training and replay; all categorical/q_ablated/hybrid variants receive the same correction. BF16 autocast training, FP32 validation and architecture remain unchanged.',
            'reason': 'Captured finite state/input yields nonfinite BF16 default-backend gradients; BF16 MATH and FP32 alternatives finite in saved-batch diagnostic. Technical correction, not a learning-rate, model, seed, threshold or cohort search.',
            'diagnostic_evidence_sha256': sha(evidence),
            'comparison_policy': 'Do not pool or select between v1 and v2 results by performance. v1 failure and unfavorable categorical results remain preserved. Corrected three-variant family requires its own complete locks/replay and independent pre-release review.',
            'test_authorization': False})
        protocol['source_hashes'][str(Path(__file__).resolve())] = sha(__file__)
        ROOT.mkdir();(ROOT/'witnesses').mkdir()
        (ROOT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n');return
    protocol = json.loads((ROOT/'protocol.json').read_text())
    for p, digest in protocol['source_hashes'].items(): assert sha(p) == digest
    for p, digest in protocol['input_hashes'].items(): assert sha(W/p) == digest
    if mode == 'smoke':
        setup();capture = W/'q_ablated_nonfinite_diagnostic_v1/first_nonfinite.pt'
        saved = torch.load(capture, weights_only=True);x, available, valid, ids = [a.cuda() for a in saved['inputs'][:4]]
        results = []
        for seed in [0, 42]:
            for label in [0, 1]:
                model = AFSetOperator('q_ablated').cuda().train();model.load_state_dict(saved['state_dict'])
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x, available, valid, ids)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.full_like(logits, label))
                loss.backward();assert torch.isfinite(loss)
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                results.append({'seed': seed, 'synthetic_target': label, 'finite_gradients': True})
        result = {'all_pass': True, 'protocol_sha256': sha(ROOT/'protocol.json'),
            'captured_batch_sha256': sha(capture), 'source_sha256': sha(__file__),
            'cases': results, 'optimizer_updates': 0, 'test_inference': False}
        path = ROOT/'witnesses'/f'witness_{time.time_ns()}.json';path.write_text(json.dumps(result, indent=2)+'\n')
        print('WITNESS_SET_MATH_SDP_V2', str(path), flush=True);return
    acceptance = json.loads((ROOT/'witness_acceptance.json').read_text())
    assert acceptance['all_pass'] and acceptance['protocol_sha256'] == sha(ROOT/'protocol.json')
    assert not (W/'evaluation_protocol_v1/heldout_release.json').exists()
    original.S = ROOT
    with sdpa_kernel(SDPBackend.MATH): original.train(mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode', choices=['freeze', 'smoke', 'categorical', 'q_ablated', 'hybrid'])
    main(parser.parse_args().mode)
