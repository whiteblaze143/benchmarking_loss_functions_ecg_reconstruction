"""Train/validation only: full-cohort matched variable-channel raw ResNet."""
import argparse
import csv
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from raw_waveform_model import RawMaskResNet, availability_ids
from matched_selection import patient_sample_weights, select_validation_threshold
from waveform_protocol import W, sha

R = W/'raw_mask_resnet_v1'
PATTERNS = [[True, False], [False, True], [True, True]]


def freeze():
    assert not R.exists(), 'Preserve the frozen raw-model protocol'
    sources = [Path(__file__), Path(__file__).with_name('raw_waveform_model.py'),
               Path(__file__).with_name('matched_selection.py'),
               Path('experiments/zzu_avblock/baseline.py'), Path('experiments/zzu_avblock/protocol.py')]
    protocol = {
        'stage': 'Train/validation only; held-out inference not implemented in this runner',
        'model': 'Existing eight-block ResNet32/64/128/256; four stem channels: two masked voltages and two explicit availability indicators; binary head',
        'seed': 42, 'epochs_max': 40, 'patience': 8, 'batch_size': 256,
        'optimizer': 'AdamW lr=.001 weight_decay=.0001; no scheduler',
        'input': 'All declared actual30-second float32 source-mV windows; no padding/downsampling/dataset cap',
        'normalization': 'Per-channel population mean/SD of all training source samples after measured calibration exclusions, counted once; no validation/test fitting; batch-local float32 normalization',
        'availability_patterns': PATTERNS,
        'availability_schedule': 'pattern=(global_window_index+epoch)%3; every training window once per epoch, all three patterns over each three epochs. Matching Set/RR models must use this acquisition distribution.',
        'training_weights': 'assigned sample count divided by patient total assigned samples, each patient sums1. For minibatch mean loss, multiply all weights by common Ntrain/Npatients; no class reweighting.',
        'loss': 'BCEWithLogits against locked >=50% source-AF input target; BF16 autocast training, FP32 model parameters; FP32 validation',
        'selection': 'Strict minimum validation patient-duration-weighted binary-target BCE, averaged equally over all3 availability patterns; ties keep earlier epoch; patience8',
        'matched_selection_note': 'Future representation failures remain indeterminate. Their fixed missing development rows must be reported; no held-out redesign or synthetic beats.',
        'threshold': 'Pool all3 validation availability patterns with repeated assigned sample references and patient IDs; fixed .01-.99 candidates maximize patient-mean elapsed-time balanced accuracy; ties closest .5 then higher.',
        'replay': 'Fixed inference batch256, FP32, deterministic CUDA/cuDNN, TF32 off; exact fresh model reload of validation logits required before readiness.',
        'checkpoint': 'Detached CPU clones for every tensor; best and final saved separately; no optimizer resume or overwrite',
        'test_policy': 'No test waveforms loaded or model predictions generated. Test endpoint, decoder policy and all comparison model protocols must be frozen before a separate held-out evaluation.',
        'prior_exposure': 'Source-RR pilot/test and reference patterns previously inspected; waveform test is not claimed entirely untouched at cohort level.',
        'source_hashes': {str(p.resolve()): sha(p) for p in sources},
        'input_hashes': {n: sha(W/n) for n in ['input_contract.json', 'windows.npz', 'file_manifest.csv', 'patient_splits.csv', 'input_verification.json', 'cohort_ledger_v1/verification.json']},
    }
    R.mkdir();(R/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'frozen': True, 'training_windows': 504777, 'validation_windows': 166047, 'test_inference': False}))


def setup():
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    assert torch.cuda.is_available() and torch.backends.cudnn.version() == 90300
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.backends.cudnn.benchmark = False;torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_development():
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z: data = {name: z[name] for name in z.files}
    train_files = [int(r['file_index']) for r in files if r['partition'] == 'train']
    val_files = [int(r['file_index']) for r in files if r['partition'] == 'validation']
    tr = np.flatnonzero(np.isin(data['file_index'], train_files));va = np.flatnonzero(np.isin(data['file_index'], val_files))
    assert len(tr) == 504777 and len(va) == 166047
    ids = np.r_[tr, va];lookup = np.full(len(data['target']), -1, np.int64);lookup[ids] = np.arange(len(ids))
    waveform = torch.empty((len(ids), 2, 6000), dtype=torch.float32, device='cuda')
    written = np.zeros(len(ids), bool);sums = np.zeros(2);squares = np.zeros(2);samples = 0
    source_hashes = {}
    for row in files:
        if row['partition'] == 'test': continue
        assert sha(row['h5_path']) == row['h5_sha256']
        source_hashes[row['h5_path']] = row['h5_sha256']
        with h5py.File(row['h5_path']) as h: voltage = h['ecg'].astype('float32')[:]
        assert np.isfinite(voltage).all()
        if row['partition'] == 'train':
            observed = voltage[int(row['excluded_prefix_samples']):]
            sums += observed.sum(axis=0, dtype=np.float64)
            squares += np.square(observed, dtype=np.float64).sum(axis=0)
            samples += len(observed)
        global_ids = np.flatnonzero(data['file_index'] == int(row['file_index']));local_ids = lookup[global_ids]
        assert (local_ids >= 0).all() and not written[local_ids].any()
        for i in range(0, len(global_ids), 256):
            chunk = global_ids[i:i+256];starts = data['input_start'][chunk]
            batch = np.ascontiguousarray(voltage[starts[:, None]+np.arange(6000)].transpose(0, 2, 1))
            waveform[torch.tensor(lookup[chunk], device='cuda')] = torch.from_numpy(batch).to('cuda')
        written[local_ids] = True
        print(json.dumps({'stage': 'loading_development', 'file_index': int(row['file_index']),
                          'loaded_windows': int(written.sum()), 'declared_windows': len(ids)}), flush=True)
    assert written.all()
    mean = sums/samples;sd = np.sqrt(squares/samples-mean**2)
    assert np.isfinite(mean).all() and np.isfinite(sd).all() and (sd > 0).all()
    patient_by_file = np.array([r['patient_id'] for r in files])
    patient = patient_by_file[data['file_index'][ids]]
    duration = (data['assigned_end'][ids]-data['assigned_start'][ids]).astype(np.int64)
    weight = patient_sample_weights(patient[:len(tr)], duration[:len(tr)], np.ones(len(tr), bool))
    val_weight = patient_sample_weights(patient[len(tr):], duration[len(tr):], np.ones(len(va), bool))
    normal = {'mean_mv': mean.tolist(), 'population_sd_mv': sd.tolist(), 'training_source_samples': samples,
              'source_hashes': source_hashes, 'test_waveforms_loaded': False,
              'training_patient_weight_totals': {p: float(weight[patient[:len(tr)] == p].sum()) for p in np.unique(patient[:len(tr)])}}
    (R/'normalization_and_loading.json').write_text(json.dumps(normal, indent=2)+'\n')
    np.savez_compressed(R/'development_manifest.npz', window_index=ids, patient_id=patient,
                        train_weight=weight, validation_weight=val_weight)
    return waveform, data, ids, patient, weight, val_weight, mean, sd, len(tr)


@torch.inference_mode()
def validation_logits(model, waveform, mean, sd, ntrain):
    model.eval();chunks = []
    for pattern in PATTERNS:
        scores = []
        for start in range(ntrain, len(waveform), 256):
            x = (waveform[start:start+256]-mean)/sd
            mask = torch.tensor(pattern, dtype=torch.bool, device='cuda').expand(len(x), -1)
            scores.append(model(x, mask).cpu().numpy())
        chunks.append(np.concatenate(scores))
    return np.stack(chunks, axis=1)


def train():
    assert not (R/'training_launch.json').exists(), 'Preserve prior training attempt'
    protocol = json.loads((R/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest, name
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest, name
    setup();started = time.monotonic()
    (R/'training_launch.json').write_text(json.dumps({'pid': os.getpid(), 'protocol_sha256': sha(R/'protocol.json'),
             'cuda_device': torch.cuda.get_device_name(), 'cudnn_version': torch.backends.cudnn.version(),
             'torch_version': torch.__version__, 'test_inference': False}, indent=2)+'\n')
    wave, data, ids, patient, weights, val_weights, mean, sd, ntrain = load_development()
    mean = torch.tensor(mean, dtype=torch.float32, device='cuda').reshape(1, 2, 1)
    sd = torch.tensor(sd, dtype=torch.float32, device='cuda').reshape(1, 2, 1)
    y = torch.tensor(data['target'][ids], dtype=torch.float32, device='cuda')
    weight = torch.tensor(weights*ntrain/90, dtype=torch.float32, device='cuda')
    global_ids = torch.tensor(ids[:ntrain], device='cuda');patterns = torch.tensor(PATTERNS, device='cuda')
    model = RawMaskResNet().cuda();optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history = [];best = float('inf');best_epoch = -1
    for epoch in range(40):
        model.train();tick = time.monotonic();permutation = torch.randperm(ntrain, device='cuda')
        weighted_loss = torch.zeros((), device='cuda')
        for start in range(0, ntrain, 256):
            idx = permutation[start:start+256];mask = patterns[availability_ids(global_ids[idx], epoch)]
            x = (wave[idx]-mean)/sd;optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                individual = nn.functional.binary_cross_entropy_with_logits(model(x, mask), y[idx], reduction='none')
                loss = (individual*weight[idx]).mean()
            assert torch.isfinite(loss)
            loss.backward();optimizer.step();weighted_loss += (individual.detach()*weight[idx]).sum()
            if start % (256*200) == 0:
                print(json.dumps({'epoch': epoch, 'training_rows_processed': min(start+256, ntrain), 'seconds': time.monotonic()-tick}), flush=True)
        logits = validation_logits(model, wave, mean, sd, ntrain)
        assert np.isfinite(logits).all(), 'Nonfinite validation logits'
        labels = data['target'][ids[ntrain:]].astype(np.float64)[:, None]
        val_loss = float(np.sum((np.logaddexp(0, logits.astype(np.float64))-labels*logits)*val_weights[:, None])/30/3)
        row = {'epoch': epoch, 'validation_patient_weighted_bce_all_patterns': val_loss,
               'training_weighted_bce': float(weighted_loss.cpu())/ntrain, 'seconds': time.monotonic()-tick}
        history.append(row);print(json.dumps(row), flush=True)
        state = {'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                 'epoch': epoch, 'validation_loss': val_loss, 'mean_mv': mean.cpu(), 'sd_mv': sd.cpu()}
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            torch.save(state, R/'best.pt');np.save(R/'best_validation_logits.npy', logits)
        (R/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        if epoch-best_epoch >= 8: break
    torch.save(state, R/'final.pt')
    checkpoint = torch.load(R/'best.pt', weights_only=True)
    assert torch.equal(checkpoint['mean_mv'], mean.cpu()) and torch.equal(checkpoint['sd_mv'], sd.cpu())
    replay_model = RawMaskResNet().cuda();replay_model.load_state_dict(checkpoint['state_dict'])
    replay = validation_logits(replay_model, wave, checkpoint['mean_mv'].cuda(), checkpoint['sd_mv'].cuda(), ntrain)
    saved = np.load(R/'best_validation_logits.npy')
    assert np.array_equal(replay, saved), 'Saved best checkpoint does not replay validation logits exactly'
    scores = torch.sigmoid(torch.from_numpy(saved)).numpy()
    np.save(R/'validation_scores.npy', scores)
    val_ids = ids[ntrain:];duration = (data['assigned_end'][val_ids]-data['assigned_start'][val_ids]).astype(np.int64)
    threshold = select_validation_threshold(scores.reshape(-1), np.repeat(data['af_assigned_samples'][val_ids].astype(np.int64), 3),
                 np.repeat(duration, 3), np.repeat(patient[ntrain:], 3), allowed_validation_patients=np.unique(patient[ntrain:]))
    (R/'threshold_selection.json').write_text(json.dumps(threshold, indent=2)+'\n')
    lock = {'complete_train_validation': True, 'test_inference_performed': False, 'best_epoch': best_epoch,
            'epochs_run': len(history), 'threshold': threshold['selected']['threshold'],
            'best_checkpoint_sha256': sha(R/'best.pt'), 'final_checkpoint_sha256': sha(R/'final.pt'),
            'protocol_sha256': sha(R/'protocol.json'), 'normalization_sha256': sha(R/'normalization_and_loading.json'),
            'validation_scores_sha256': sha(R/'validation_scores.npy'), 'validation_replay_bitwise': True,
            'training_windows': ntrain, 'validation_windows': len(ids)-ntrain, 'training_patients': 90,
            'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30, 'seconds_including_loading': time.monotonic()-started}
    (R/'evaluation_lock.json').write_text(json.dumps(lock, indent=2)+'\n');print(json.dumps(lock), flush=True)


def smoke():
    setup();model = RawMaskResNet().cuda();x = torch.randn(8, 2, 6000, device='cuda')
    patterns = torch.tensor(PATTERNS, dtype=torch.bool, device='cuda');mask = patterns[torch.arange(8, device='cuda')%3]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x, mask), torch.tensor([0, 1]*4, dtype=torch.float32, device='cuda'))
    loss.backward();assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval();mask = patterns[0].expand(len(x), -1);changed = x.clone();changed[:, 1] += 10000
    with torch.inference_mode():
        original = model(x, mask);perturbed = model(changed, mask)
    assert torch.equal(original, perturbed)
    # Reserve the full development cache before exercising the real batch size.
    reserve = torch.empty((670824, 2, 6000), dtype=torch.float32, device='cuda')
    model.train();large = torch.randn(256, 2, 6000, device='cuda')
    mask = patterns[torch.arange(256, device='cuda')%3]
    target = torch.arange(256, device='cuda').remainder(2).float()
    torch.cuda.synchronize();tick = time.monotonic()
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            large_loss = nn.functional.binary_cross_entropy_with_logits(model(large, mask), target)
        large_loss.backward();assert torch.isfinite(large_loss)
    torch.cuda.synchronize();seconds = time.monotonic()-tick
    assert reserve.shape[0] == 670824
    print('WITNESS_AF_RAW_MASK', json.dumps({'shape': list(original.shape), 'device': torch.cuda.get_device_name(),
          'loss': float(loss.detach()), 'unobserved_channel_isolation': True, 'cudnn': torch.backends.cudnn.version(),
          'batch_size': 256, 'development_cache_reserved_gib': reserve.numel()*4/2**30,
          'synthetic_training_windows_per_second': 768/seconds,
          'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();group = parser.add_mutually_exclusive_group(required=True)
    for mode in ['freeze', 'train', 'smoke']: group.add_argument('--'+mode, action='store_true')
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.train: train()
    else: smoke()
