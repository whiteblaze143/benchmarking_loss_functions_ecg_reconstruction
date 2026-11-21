"""Fresh-process source-waveform replay of all selected raw validation logits."""
import csv
import json
import time
import h5py
import numpy as np
import torch
from waveform_protocol import W, sha
from raw_waveform_model import RawMaskResNet
from train_raw_waveform import setup, validation_logits
from matched_selection import select_validation_threshold

R = W/'raw_mask_resnet_v1'


def main():
    assert not (R/'fresh_process_validation_replay.json').exists(), 'Preserve previous replay result'
    lock = json.loads((R/'evaluation_lock.json').read_text());assert lock['complete_train_validation'] and not lock['test_inference_performed']
    protocol = json.loads((R/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest, name
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest, name
    for name, key in [('protocol.json', 'protocol_sha256'), ('best.pt', 'best_checkpoint_sha256'),
                      ('final.pt', 'final_checkpoint_sha256'), ('normalization_and_loading.json', 'normalization_sha256'),
                      ('validation_scores.npy', 'validation_scores_sha256')]: assert sha(R/name) == lock[key], name
    setup();assert torch.cuda.mem_get_info()[0] >= 33*2**30, 'Full-layout replay requires33GiB free GPU'
    started = time.monotonic();manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z:
        file_index, starts = z['file_index'], z['input_start']
        assigned = (z['assigned_end'].astype(np.int64)-z['assigned_start']);af = z['af_assigned_samples'].astype(np.int64)
    val_files = [int(r['file_index']) for r in manifest if r['partition'] == 'validation']
    ids = np.flatnonzero(np.isin(file_index, val_files));assert len(ids) == 166047
    with np.load(R/'development_manifest.npz') as z: assert np.array_equal(ids, z['window_index'][504777:])
    # Empty prefix is never read: retain original validation tensor layout/address offset.
    ntrain = 504777;wave = torch.empty((ntrain+len(ids), 2, 6000), dtype=torch.float32, device='cuda')
    lookup = np.full(len(file_index), -1, np.int64);lookup[ids] = ntrain+np.arange(len(ids))
    filled = np.zeros(len(ids), bool);sources = {}
    for row in manifest:
        if row['partition'] != 'validation': continue
        assert sha(row['h5_path']) == row['h5_sha256']
        with h5py.File(row['h5_path']) as h: voltage = h['ecg'].astype('float32')[:]
        global_ids = np.flatnonzero(file_index == int(row['file_index']));local = lookup[global_ids]
        assert (local >= ntrain).all() and not filled[local-ntrain].any()
        for start in range(0, len(global_ids), 256):
            chunk = global_ids[start:start+256]
            x = np.ascontiguousarray(voltage[starts[chunk, None]+np.arange(6000)].transpose(0, 2, 1))
            wave[torch.tensor(lookup[chunk], device='cuda')] = torch.from_numpy(x).cuda()
        filled[local-ntrain] = True;sources[row['h5_path']] = row['h5_sha256']
        print(json.dumps({'validation_windows_loaded': int(filled.sum()), 'declared_validation_windows': len(ids)}), flush=True)
    assert filled.all() and len(sources) == 60
    best = torch.load(R/'best.pt', weights_only=True);final = torch.load(R/'final.pt', weights_only=True)
    history = json.loads((R/'history.json').read_text())
    expected_best = min(history, key=lambda r: r['validation_patient_weighted_bce_all_patterns'])
    assert best['epoch'] == lock['best_epoch'] == expected_best['epoch']
    assert best['validation_loss'] == expected_best['validation_patient_weighted_bce_all_patterns']
    assert final['epoch'] == history[-1]['epoch'] == len(history)-1
    model = RawMaskResNet().cuda();model.load_state_dict(best['state_dict'])
    logits = validation_logits(model, wave, best['mean_mv'].cuda(), best['sd_mv'].cuda(), ntrain)
    stored_logits = np.load(R/'best_validation_logits.npy');stored_scores = np.load(R/'validation_scores.npy')
    scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
    patients = np.array([r['patient_id'] for r in manifest])[file_index[ids]]
    threshold = select_validation_threshold(scores.reshape(-1), np.repeat(af[ids], 3), np.repeat(assigned[ids], 3),
                 np.repeat(patients, 3), allowed_validation_patients=np.unique(patients))
    exact_logits = np.array_equal(logits, stored_logits);exact_scores = np.array_equal(scores, stored_scores)
    threshold_matches = threshold['selected']['threshold'] == lock['threshold']
    result = {'all_pass': bool(exact_logits and exact_scores and threshold_matches),
              'validation_windows': len(ids), 'acquisition_predictions': int(scores.size), 'validation_patients': len(np.unique(patients)),
              'source_waveform_files': len(sources), 'test_waveforms_read': False,
              'best_checkpoint_is_strict_history_minimum': True, 'best_epoch': best['epoch'], 'final_epoch': final['epoch'],
              'fresh_process_logits_bitwise': bool(exact_logits), 'fresh_process_scores_bitwise': bool(exact_scores),
              'maximum_logit_difference': float(abs(logits-stored_logits).max()),
              'maximum_score_difference': float(abs(scores-stored_scores).max()),
              'decision_changes_at_locked_threshold': int(((scores >= lock['threshold']) != (stored_scores >= lock['threshold'])).sum()),
              'validation_threshold_reselection_matches': bool(threshold_matches),
              'source_h5_sha256': sources, 'evaluation_lock_sha256': sha(R/'evaluation_lock.json'),
              'replay_source_sha256': sha(__file__), 'seconds_including_source_loading': time.monotonic()-started,
              'layout_note': 'Uninitialized training prefix reserved only for matching tensor layout; never read or used as model input. All predictions derive from reconstructed validation waveforms.'}
    (R/'fresh_process_validation_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_h5_sha256'}, indent=2))
    assert result['all_pass'], 'Fresh-process replay failure retained in receipt'


if __name__ == '__main__':
    main()
