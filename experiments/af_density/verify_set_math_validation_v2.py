"""Corrected MATH-family validation replay; v1 verifier remains unchanged."""
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
from torch.nn.attention import sdpa_kernel, SDPBackend

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reselect_threshold_independently(scores, af_samples, assigned_samples, patients):
    """Reconstruct fixed99-candidate policy; failed scores retain class time."""
    candidates = np.arange(1, 100)/100
    sensitivity = [];specificity = []
    for patient in np.unique(patients):
        take = patients == patient
        p = scores[take].astype(np.float64).reshape(-1)
        af = np.repeat(af_samples[take].astype(np.int64), 3)
        nonaf = np.repeat(assigned_samples[take], 3)-af
        positive = np.isfinite(p[:, None]) & (p[:, None] >= candidates)
        negative = np.isfinite(p[:, None]) & ~positive
        if af.sum(): sensitivity.append((positive.T@af)/af.sum())
        if nonaf.sum(): specificity.append((negative.T@nonaf)/nonaf.sum())
    if not sensitivity or not specificity:
        raise ValueError('Both reference classes required for threshold selection')
    balanced = [float((np.mean([r[i] for r in sensitivity])+np.mean([r[i] for r in specificity]))/2)
                for i in range(len(candidates))]
    selected = max(range(99), key=lambda i: (balanced[i], -abs(candidates[i]-.5), candidates[i]))
    return float(candidates[selected]), balanced


def main(mode):
    root = W/'set_models_math_sdp_v2';directory = root/mode
    reader_contract = json.loads((root/'validation_reader_contract.json').read_text())
    assert reader_contract['reader_source_sha256'] == sha(__file__)
    for p, digest in reader_contract['input_hashes'].items(): assert sha(W/p) == digest
    lock = json.loads((directory/'evaluation_lock.json').read_text())
    assert lock['complete_train_validation'] and not lock['test_inference_performed']
    protocol = json.loads((root/'protocol.json').read_text())
    assert sha(root/'protocol.json') == lock['protocol_sha256']
    for p, digest in protocol['source_hashes'].items(): assert sha(p) == digest
    for p, digest in protocol['input_hashes'].items(): assert sha(W/p) == digest
    assert sha(directory/'best.pt') == lock['best_checkpoint_sha256']
    assert sha(directory/'final.pt') == lock['final_checkpoint_sha256']
    assert sha(directory/'validation_scores.npy') == lock['validation_scores_sha256']
    history = json.loads((directory/'history.json').read_text())
    assert len(history) == lock['epochs_run'] and 1 <= len(history) <= 40
    assert [r['epoch'] for r in history] == list(range(len(history)))
    assert all(r['training_declared_rows'] == 504777 for r in history)
    losses = [r['validation_bce_all_rows'] for r in history]
    assert np.isfinite(losses).all() and int(np.argmin(losses)) == lock['best_epoch']
    if len(history) < 40: assert len(history)-1-lock['best_epoch'] == 8
    output = directory/'independent_validation_v1';output.mkdir(exist_ok=False)
    setup()
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z:
        file_index = z['file_index']
        labels = z['target'];af_samples = z['af_assigned_samples']
        assigned_samples = z['assigned_end']-z['assigned_start']
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
    final = torch.load(directory/'final.pt', weights_only=True)
    assert checkpoint['mode'] == final['mode'] == mode
    assert checkpoint['epoch'] == lock['best_epoch'] and final['epoch'] == len(history)-1
    assert checkpoint['validation_loss'] == losses[lock['best_epoch']]
    assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
    assert all(torch.isfinite(v).all() for v in final['state_dict'].values())
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
    patients = np.array([r['patient_id'] for r in manifest])[file_index[ids]]
    unique_patients = np.unique(patients);assert len(unique_patients) == 30
    val_weight = np.zeros(len(ids), np.float64)
    for patient in unique_patients:
        take = patients == patient
        val_weight[take] = assigned_samples[ids][take]/assigned_samples[ids][take].sum(dtype=np.float64)
    safe = np.where(np.isfinite(result), result, 0.).astype(np.float64)
    terms = np.logaddexp(0, safe)-labels[ids, None].astype(np.float64)*safe
    rebuilt_loss = float(np.sum(terms*val_weight[:, None])/30/3)
    assert abs(rebuilt_loss-checkpoint['validation_loss']) <= 1e-12
    # Independent threshold reconstruction; unknown predictions retain class time.
    threshold, balanced = reselect_threshold_independently(scores, af_samples[ids], assigned_samples[ids], patients)
    threshold_archive = json.loads((directory/'threshold_selection.json').read_text())
    assert threshold == lock['threshold'] == threshold_archive['selected']['threshold']
    assert np.allclose(balanced, [r['balanced_accuracy'] for r in threshold_archive['candidates']], rtol=0, atol=1e-14)
    receipt = {'all_pass': True, 'mode': mode, 'files': len(chosen_files), 'windows': len(ids),
        'conditions': int(result.size), 'indeterminate_conditions': int(np.isnan(result).sum()),
        'logits_bitwise_equal': True, 'scores_bitwise_equal': True, 'test_access': False,
        'strict_history_minimum_verified': True, 'best_final_identity_verified': True,
        'selected_validation_bce_recomputed': rebuilt_loss,
        'threshold_independently_reselected': threshold, 'attention_backend': 'MATH',
        'source_sha256': sha(__file__), 'model_lock_sha256': sha(directory/'evaluation_lock.json'),
        'validation_reader_contract_sha256': sha(root/'validation_reader_contract.json'),
        'verified_validation_source_hashes': sources,
        'scope': 'Separate-process complete validation replay; independent loader/batching, authentic frozen model implementation. No fitting or test inference.'}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'verified_validation_source_hashes'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--mode', choices=['categorical', 'q_ablated', 'hybrid'], required=True)
    with sdpa_kernel(SDPBackend.MATH):
        main(parser.parse_args().mode)
