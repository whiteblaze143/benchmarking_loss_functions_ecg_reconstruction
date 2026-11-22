"""Locked corrected-family predictions; test access requires source-bound approval."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from set_waveform_model import AFSetOperator
from hybrid_set_model import AFSetRR, rr_side_input
from train_raw_waveform import setup

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p):
    with Path(p).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_test_release(mode):
    key = 'math_v2_'+mode;root = W/'set_models_math_sdp_v2'/mode
    release = json.loads((W/'evaluation_protocol_v1/heldout_release.json').read_text())
    assert release['released'] and key in release['authorized_models']
    assert release['evaluation_protocol_sha256'] == sha(W/'evaluation_protocol_v1/protocol.json')
    assert release['input_contract_sha256'] == sha(W/'input_contract.json')
    assert release['corrected_family_protocol_sha256'] == sha(W/'set_models_math_sdp_v2/protocol.json')
    assert release['evaluation_lock_hashes'][key] == sha(root/'evaluation_lock.json')
    assert release['predictor_source_hashes'][key] == sha(__file__)
    approval_ref = release['independent_approval_receipt']
    assert sha(approval_ref['path']) == approval_ref['sha256']
    approval = json.loads(Path(approval_ref['path']).read_text())
    assert approval['authorize_test_inference'] and approval['scope'] == 'locked_internal_benchmark'
    assert approval['reviewed_model_lock_hashes'] == release['evaluation_lock_hashes']
    assert approval['reviewed_predictor_source_hashes'] == release['predictor_source_hashes']
    assert approval['reviewed_input_contract_sha256'] == release['input_contract_sha256']
    parity_path = root/'inference_validation_v1/receipt.json'
    assert sha(parity_path) == release['validation_parity_receipt_hashes'][key]
    parity = json.loads(parity_path.read_text())
    assert parity['complete'] and parity['validation_bitwise_replay']
    assert parity['source_sha256'] == sha(__file__)
    assert parity['model_evaluation_lock_sha256'] == sha(root/'evaluation_lock.json')


def main(mode, partition):
    # This gate precedes input/feature reads, CUDA setup and output creation.
    if partition == 'test': require_test_release(mode)
    family = W/'set_models_math_sdp_v2';root = family/mode
    contract = json.loads((family/'prediction_reader_contract_v1.json').read_text())
    assert contract['source_sha256'] == sha(__file__)
    for name, digest in contract['input_hashes'].items(): assert sha(W/name) == digest
    lock = json.loads((root/'evaluation_lock.json').read_text())
    assert lock['complete_train_validation'] and not lock['test_inference_performed']
    assert sha(family/'protocol.json') == lock['protocol_sha256']
    protocol = json.loads((family/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest
    assert sha(root/'best.pt') == lock['best_checkpoint_sha256']
    replay = json.loads((root/'independent_validation_v1/receipt.json').read_text())
    assert replay['all_pass'] and replay['model_lock_sha256'] == sha(root/'evaluation_lock.json')
    assert replay['validation_reader_contract_sha256'] == sha(family/'validation_reader_contract.json')
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    selected = [r for r in files if r['partition'] == partition]
    with np.load(W/'windows.npz') as z: file_index = z['file_index']
    ids = np.flatnonzero(np.isin(file_index, [int(r['file_index']) for r in selected]))
    assert len(ids) == (166047 if partition == 'validation' else 131667)
    assert len(selected) == (60 if partition == 'validation' else 48)
    if partition == 'validation':
        with np.load(root/'development_manifest.npz') as z: assert np.array_equal(ids, z['window_index'][504777:])
    setup();assert torch.cuda.mem_get_info()[0] >= 6*2**30
    output = root/f'inference_{partition}_v1';output.mkdir(exist_ok=False)
    started = time.monotonic()
    (output/'input_receipt.json').write_text(json.dumps({'partition': partition, 'mode': mode,
        'source_sha256': sha(__file__), 'model_evaluation_lock_sha256': sha(root/'evaluation_lock.json'),
        'input_contract_sha256': sha(W/'input_contract.json'), 'windows': len(ids),
        'window_index_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
        'threshold': lock['threshold'], 'fit_or_threshold_selection': False}, indent=2)+'\n')
    lookup = np.full(len(file_index), -1, np.int64);lookup[ids] = np.arange(len(ids))
    response = torch.empty((len(ids), 2, 16, 128), device='cuda')
    phase = np.zeros((len(ids), 2), bool);rr_valid = phase.copy()
    rr = np.full((len(ids), 2, 16), np.nan, np.float32);coverage = np.zeros((len(ids), 2), np.float32)
    written = np.zeros(len(ids), bool);sources = {}
    feature_lock = json.loads((W/'phase_features_v1/input_lock.json').read_text())
    for row in selected:
        fi = int(row['file_index']);path = W/'phase_features_v1/shards'/f'file_{fi:03d}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == meta['output_sha256'] and meta['h5_sha256'] == row['h5_sha256']
        assert meta['kernel_sha256'] == feature_lock['kernel_sha256']
        assert meta['feature_protocol_sha256'] == sha(W/'phase_features_v1/protocol.json')
        observed = W/'observed_features'/path.name;assert sha(observed) == meta['observed_shard_sha256']
        expected = np.flatnonzero(file_index == fi);local = lookup[expected]
        assert (local >= 0).all() and not written[local].any()
        with np.load(path) as z:
            assert np.array_equal(z['window_index'], expected)
            values, valid = z['response'], z['phase_valid']
            assert values.dtype == np.float32 and values.shape == (len(expected), 2, 16, 128)
            assert np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all()
            response[torch.tensor(local, device='cuda')] = torch.from_numpy(values).cuda()
            phase[local] = valid;coverage[local] = z['represented_fraction']
        with np.load(observed) as z:
            assert np.array_equal(z['window_index'], expected) and np.array_equal(z['phase_valid'], phase[local])
            rr[local] = z['rr_features'];rr_valid[local] = z['rr_valid']
        written[local] = True;sources[str(path)] = meta['output_sha256']
    assert written.all()
    checkpoint = torch.load(root/'best.pt', weights_only=True)
    model = (AFSetRR() if mode == 'hybrid' else AFSetOperator(mode)).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'])
    phase_gpu = torch.tensor(phase, device='cuda');rv_gpu = torch.tensor(rr_valid, device='cuda')
    rr_gpu = torch.tensor(rr, device='cuda');coverage_gpu = torch.tensor(coverage, device='cuda')
    mean, sd = checkpoint['rr_mean'].cuda(), checkpoint['rr_sd'].cuda()
    logits = np.full((len(ids), 3), np.nan, np.float32)
    with sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
        for pattern_index, pattern in enumerate([[True, False], [False, True], [True, True]]):
            eligible = ((phase | rr_valid if mode == 'hybrid' else phase) & pattern).any(axis=1)
            for start in range(0, len(ids), 256):
                ix = np.arange(start, min(start+256, len(ids)));ix = ix[eligible[ix]]
                if not len(ix): continue
                index = torch.tensor(ix, device='cuda');available = torch.tensor(pattern, device='cuda').expand(len(ix), -1)
                operators = torch.tensor([0, 1], device='cuda').expand(len(ix), -1)
                if mode == 'hybrid':
                    side, present = rr_side_input(rr_gpu[index], rv_gpu[index], available, mean, sd)
                    value = model(response[index], available, phase_gpu[index], operators, side, present, coverage_gpu[index])
                else: value = model(response[index], available, phase_gpu[index], operators)
                assert torch.isfinite(value).all();logits[ix, pattern_index] = value.cpu().numpy()
    scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
    np.savez_compressed(output/'predictions.npz', window_index=ids, logits=logits, scores=scores)
    exact = None
    if partition == 'validation':
        exact = bool(np.array_equal(logits, np.load(root/'best_validation_logits.npy'), equal_nan=True)
                     and np.array_equal(scores, np.load(root/'validation_scores.npy'), equal_nan=True))
    receipt = {'complete': exact if partition == 'validation' else True, 'mode': mode, 'partition': partition,
        'source_sha256': sha(__file__), 'model_evaluation_lock_sha256': sha(root/'evaluation_lock.json'),
        'family_protocol_sha256': sha(family/'protocol.json'), 'windows': len(ids), 'source_files': len(selected),
        'prediction_reader_contract_sha256': sha(family/'prediction_reader_contract_v1.json'),
        'indeterminate_conditions': int(np.isnan(scores).sum()), 'validation_bitwise_replay': exact,
        'prediction_archive_sha256': sha(output/'predictions.npz'), 'source_feature_hashes': sources,
        'threshold': lock['threshold'], 'no_fit_or_threshold_selection': True,
        'test_inference_performed': partition == 'test', 'seconds': time.monotonic()-started}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    assert partition != 'validation' or exact, 'Preserve failed validation reader attempt; no test release'
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_feature_hashes'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('mode', choices=['categorical', 'q_ablated', 'hybrid'])
    parser.add_argument('partition', choices=['validation', 'test']);args = parser.parse_args()
    main(args.mode, args.partition)
