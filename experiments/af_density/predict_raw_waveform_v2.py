"""Locked raw inference, with validation parity required before test release."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from raw_waveform_model import RawMaskResNet
from train_raw_waveform import setup, validation_logits

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
R = W/'raw_mask_resnet_v1'
PADDING_ROWS = 9


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def require_test_release(model):
    release = json.loads((W/'evaluation_protocol_v1/heldout_release.json').read_text())
    assert release['released'] and model in release['authorized_models']
    assert release['evaluation_protocol_sha256'] == sha(W/'evaluation_protocol_v1/protocol.json')
    assert release['input_contract_sha256'] == sha(W/'input_contract.json')
    assert release['evaluation_lock_hashes'][model] == sha(R/'evaluation_lock.json')
    assert release['predictor_source_hashes'][model] == sha(__file__)
    reference = release['independent_approval_receipt']
    assert sha(reference['path']) == reference['sha256']
    approval = json.loads(Path(reference['path']).read_text())
    assert approval['authorize_test_inference'] and approval['scope'] == 'locked_internal_benchmark'
    assert approval['reviewed_model_lock_hashes'] == release['evaluation_lock_hashes']
    assert approval['reviewed_predictor_source_hashes'] == release['predictor_source_hashes']
    assert approval['reviewed_input_contract_sha256'] == release['input_contract_sha256']
    parity_path = R/'inference_validation_v2/receipt.json'
    assert sha(parity_path) == release['validation_parity_receipt_hashes'][model]
    parity = json.loads(parity_path.read_text())
    assert parity['complete'] and parity['validation_bitwise_replay']
    assert parity['source_sha256'] == sha(__file__)
    assert parity['model_evaluation_lock_sha256'] == sha(R/'evaluation_lock.json')


def main(partition):
    lock = json.loads((R/'evaluation_lock.json').read_text());assert lock['complete_train_validation']
    protocol = json.loads((R/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest
    for filename, key in [('protocol.json', 'protocol_sha256'), ('best.pt', 'best_checkpoint_sha256'),
                          ('normalization_and_loading.json', 'normalization_sha256'), ('validation_scores.npy', 'validation_scores_sha256')]:
        assert sha(R/filename) == lock[key]
    if partition == 'test':
        require_test_release('raw')
    output = R/('inference_'+partition+'_v2');output.mkdir(exist_ok=False)
    setup();assert torch.cuda.mem_get_info()[0] >= 11*2**30, 'Insufficient GPU memory; no cap or retry'
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z: file_index, input_start = z['file_index'], z['input_start']
    selected = [r for r in files if r['partition'] == partition]
    ids = np.flatnonzero(np.isin(file_index, [int(r['file_index']) for r in selected]))
    assert len(ids) == (166047 if partition == 'validation' else 131667)
    if partition == 'validation':
        with np.load(R/'development_manifest.npz') as z: assert np.array_equal(ids, z['window_index'][504777:])
    wave = torch.empty((PADDING_ROWS+len(ids), 2, 6000), dtype=torch.float32, device='cuda')
    lookup = np.full(len(file_index), -1, np.int64);lookup[ids] = PADDING_ROWS+np.arange(len(ids))
    filled = np.zeros(len(ids), bool);sources = {};started = time.monotonic()
    (output/'input_receipt.json').write_text(json.dumps({'partition': partition, 'source_sha256': sha(__file__),
        'model_evaluation_lock_sha256': sha(R/'evaluation_lock.json'), 'window_index_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
        'padding_rows_never_read': PADDING_ROWS, 'windows': len(ids), 'threshold': lock['threshold'],
        'layout_policy': 'Nine uninitialized prefix rows preserve original training-prefix row offset modulo64, reducing reserved memory. No prefix is read. Full validation logits/scores must replay bitwise under this inference layout before any test release.',
        'normalization_fitted': False}, indent=2)+'\n')
    for row in selected:
        assert sha(row['h5_path']) == row['h5_sha256']
        with h5py.File(row['h5_path']) as h: voltage = h['ecg'].astype('float32')[:]
        assert voltage.shape == (int(row['source_samples']), 2) and np.isfinite(voltage).all()
        global_ids = np.flatnonzero(file_index == int(row['file_index']));local = lookup[global_ids]
        assert (local >= PADDING_ROWS).all() and not filled[local-PADDING_ROWS].any()
        for start in range(0, len(global_ids), 256):
            chunk = global_ids[start:start+256]
            batch = np.ascontiguousarray(voltage[input_start[chunk, None]+np.arange(6000)].transpose(0, 2, 1))
            wave[torch.tensor(lookup[chunk], device='cuda')] = torch.from_numpy(batch).cuda()
        filled[local-PADDING_ROWS] = True;sources[row['h5_path']] = row['h5_sha256']
        print(json.dumps({'loaded_windows': int(filled.sum()), 'declared_windows': len(ids)}), flush=True)
    assert filled.all()
    checkpoint = torch.load(R/'best.pt', weights_only=True)
    model = RawMaskResNet().cuda();model.load_state_dict(checkpoint['state_dict'])
    logits = validation_logits(model, wave, checkpoint['mean_mv'].cuda(), checkpoint['sd_mv'].cuda(), PADDING_ROWS)
    scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
    np.savez_compressed(output/'predictions.npz', window_index=ids, scores=scores, logits=logits)
    exact = None;maximum = None
    if partition == 'validation':
        original = np.load(R/'best_validation_logits.npy')
        maximum = float(abs(logits-original).max())
        exact = bool(np.array_equal(logits, original) and np.array_equal(scores, np.load(R/'validation_scores.npy')))
    result = {'complete': partition != 'validation' or exact, 'partition': partition, 'source_sha256': sha(__file__),
        'windows': len(ids), 'source_files': len(sources), 'source_waveform_hashes': sources, 'threshold': lock['threshold'],
        'prediction_archive_sha256': sha(output/'predictions.npz'), 'input_receipt_sha256': sha(output/'input_receipt.json'),
        'validation_bitwise_replay': exact, 'maximum_validation_logit_difference': maximum,
        'peak_gpu_GiB': torch.cuda.max_memory_allocated()/2**30, 'seconds': time.monotonic()-started,
        'test_inference_performed': partition == 'test', 'no_fit_or_threshold_selection': True}
    (output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_waveform_hashes'}), flush=True)
    assert partition != 'validation' or exact, 'Inference layout failed validation parity; preserve attempt and do not release test'


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('partition', choices=['validation', 'test'])
    main(parser.parse_args().partition)
