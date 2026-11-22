"""Fresh-process, validation-only Set inference with an independent shard loader."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from set_waveform_model import AFSetOperator
from hybrid_set_model import AFSetRR, rr_side_input
from train_raw_waveform import setup

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(mode):
    root = W/'set_models_v1';directory = root/mode
    lock = json.loads((directory/'evaluation_lock.json').read_text())
    assert lock['complete_train_validation'] and not lock['test_inference_performed']
    protocol = json.loads((root/'protocol.json').read_text())
    assert sha(root/'protocol.json') == lock['protocol_sha256']
    for p, digest in protocol['source_hashes'].items(): assert sha(p) == digest
    for p, digest in protocol['input_hashes'].items(): assert sha(W/p) == digest
    assert sha(directory/'best.pt') == lock['best_checkpoint_sha256']
    assert sha(directory/'validation_scores.npy') == lock['validation_scores_sha256']
    output = directory/'independent_validation_v1';output.mkdir(exist_ok=False)
    setup()
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z: file_index = z['file_index']
    chosen_files = [int(r['file_index']) for r in manifest if r['partition'] == 'validation']
    ids = np.flatnonzero(np.isin(file_index, chosen_files));assert len(ids) == 166047
    with np.load(directory/'development_manifest.npz') as z:
        assert np.array_equal(z['window_index'][504777:], ids)
    lookup = np.full(len(file_index), -1, np.int64);lookup[ids] = np.arange(len(ids))
    response = torch.empty((len(ids), 2, 16, 128), device='cuda')
    pv = np.zeros((len(ids), 2), bool);rv = pv.copy()
    rr = np.full((len(ids), 2, 16), np.nan, np.float32);coverage = np.zeros((len(ids), 2), np.float32)
    written = np.zeros(len(ids), bool)
    feature_lock = json.loads((W/'phase_features_v1/input_lock.json').read_text())
    sources = {}
    for fi in chosen_files:
        path = W/'phase_features_v1/shards'/f'file_{fi:03d}.npz'
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == receipt['output_sha256']
        assert receipt['kernel_sha256'] == feature_lock['kernel_sha256']
        assert receipt['feature_protocol_sha256'] == sha(W/'phase_features_v1/protocol.json')
        observed = W/'observed_features'/path.name
        assert sha(observed) == receipt['observed_shard_sha256']
        expected = np.flatnonzero(file_index == fi);local = lookup[expected]
        assert not written[local].any() and (local >= 0).all()
        with np.load(path) as z:
            assert np.array_equal(z['window_index'], expected)
            response[torch.tensor(local, device='cuda')] = torch.from_numpy(z['response']).cuda()
            pv[local] = z['phase_valid'];coverage[local] = z['represented_fraction']
        with np.load(observed) as z:
            assert np.array_equal(z['window_index'], expected)
            rr[local] = z['rr_features'];rv[local] = z['rr_valid']
        written[local] = True;sources[str(path)] = receipt['output_sha256']
    assert written.all()
    checkpoint = torch.load(directory/'best.pt', weights_only=True)
    model = (AFSetRR() if mode == 'hybrid' else AFSetOperator(mode)).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'])
    pv_gpu = torch.tensor(pv, device='cuda');rv_gpu = torch.tensor(rv, device='cuda')
    rr_gpu = torch.tensor(rr, device='cuda');coverage_gpu = torch.tensor(coverage, device='cuda')
    result = np.full((len(ids), 3), np.nan, np.float32)
    with torch.inference_mode():
        for p, pattern in enumerate([[True, False], [False, True], [True, True]]):
            eligible = ((pv | rv if mode == 'hybrid' else pv) & pattern).any(axis=1)
            for start in range(0, len(ids), 256):
                ix = np.arange(start, min(start+256, len(ids)));ix = ix[eligible[ix]]
                if not len(ix): continue
                index = torch.tensor(ix, device='cuda')
                available = torch.tensor(pattern, device='cuda').expand(len(ix), -1)
                op = torch.tensor([0, 1], device='cuda').expand(len(ix), -1)
                if mode == 'hybrid':
                    side, present = rr_side_input(rr_gpu[index], rv_gpu[index], available,
                        checkpoint['rr_mean'].cuda(), checkpoint['rr_sd'].cuda())
                    logits = model(response[index], available, pv_gpu[index], op, side, present, coverage_gpu[index])
                else: logits = model(response[index], available, pv_gpu[index], op)
                assert torch.isfinite(logits).all()
                result[ix, p] = logits.cpu().numpy()
    saved = np.load(directory/'best_validation_logits.npy')
    assert np.array_equal(saved, result, equal_nan=True)
    scores = torch.sigmoid(torch.from_numpy(result)).numpy()
    assert np.array_equal(scores, np.load(directory/'validation_scores.npy'), equal_nan=True)
    receipt = {'all_pass': True, 'mode': mode, 'files': len(chosen_files), 'windows': len(ids),
        'conditions': int(result.size), 'indeterminate_conditions': int(np.isnan(result).sum()),
        'logits_bitwise_equal': True, 'scores_bitwise_equal': True, 'test_access': False,
        'source_sha256': sha(__file__), 'model_lock_sha256': sha(directory/'evaluation_lock.json'),
        'verified_validation_source_hashes': sources,
        'scope': 'Separate-process complete validation replay; independent loader/batching, authentic frozen model implementation. No fitting or test inference.'}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'verified_validation_source_hashes'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--mode', choices=['categorical', 'q_ablated', 'hybrid'], required=True)
    main(parser.parse_args().mode)
