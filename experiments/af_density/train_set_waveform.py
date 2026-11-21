"""Matched categorical/q-ablated/hybrid SetOperator train/validation runners."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from waveform_protocol import W, sha
from set_waveform_model import AFSetOperator
from hybrid_set_model import AFSetRR, fit_rr_scaling, rr_side_input
from matched_selection import patient_sample_weights, select_validation_threshold
from raw_waveform_model import availability_ids
from train_raw_waveform import setup, PATTERNS

S = W/'set_models_v1'
MODES = ['categorical', 'q_ablated', 'hybrid']


def model_for(mode):
    return AFSetRR() if mode == 'hybrid' else AFSetOperator(mode)


def freeze():
    assert not S.exists(), 'Preserve frozen SetOperator training protocol'
    original = Path('experiments/ptbxl_distributional_repecg/src/repecg')
    sources = [Path(__file__)]+[Path(__file__).with_name(name) for name in
               ['set_waveform_model.py', 'hybrid_set_model.py', 'matched_selection.py', 'raw_waveform_model.py', 'train_raw_waveform.py']]
    sources += [original/'paper07_operator/model.py', original/'common/models.py', original/'common/variants.py']
    protocol = {
        'stage': 'Train/validation only, no held-out inference path', 'modes': MODES,
        'backbone': 'Unchanged authentic OperatorSetModel, binary AF head; no auxiliary reconstruction/orientation loss in any admitted mode',
        'q_ablated': 'Also serves the channel-agnostic set control; do not train a duplicate identical no-q model',
        'hybrid': 'Original categorical256-D morphology latent + standardized32 RR statistics +2 acquisition/2 RR-valid flags +2 modality-presence flags +2 phase coverage fractions; single296-to1 linear head',
        'hybrid_missingness': 'Available actual RR may support a score with morphology missing; zero morphology latent is an explicitly flagged sentinel, never invented beats. If both modalities absent, indeterminate. Report RR-only, morphology-only and joint input counts; RR-only recovery is not evidence of geometry advantage.',
        'rr_scaling': 'Fit per-channel/per-feature population mean/SD on finite RR-valid training rows only. Individual missing statistics mean-imputed after standardization; exact constants use declared SD1. No channel with no supported feature silently admitted.',
        'seed': 42, 'epochs_max': 40, 'patience': 8, 'batch_size': 256,
        'optimizer': 'AdamW lr=.001 weight_decay=.0001, no scheduler',
        'availability': 'Same raw schedule: pattern=(global_window_index+epoch)%3, every declared training window once each epoch; skip only unavailable scheduled representation and report counts',
        'training_weights': 'Across all3 acquisition examples, assigned-duration weights total1 per training patient after representation eligibility, no class balancing. Multiply by common3*Ntrain/90 for mean minibatch loss; divide batch sum by declared batch size including unavailable rows. Objective is patient-equal over each3-epoch acquisition cycle.',
        'checkpoint_selection': 'Strict minimum validation patient-duration-weighted BCE, averaged over all3 patterns and all declared rows. Fixed unavailable rows contribute log(2) only to this development selection loss, not invented prediction scores. Ties earlier, patience8.',
        'precision': 'BF16 training autocast, FP32 parameters/validation, deterministic CUDA, TF32 disabled; fresh-model exact validation logit replay required',
        'threshold': 'Same fixed pooled3-pattern validation elapsed-time balanced-accuracy selection as raw; unknown scores keep denominator',
        'source_hashes': {str(p.resolve()): sha(p) for p in sources},
        'input_hashes': {n: sha(W/n) for n in ['input_contract.json', 'windows.npz', 'file_manifest.csv', 'patient_splits.csv', 'phase_features_v1/protocol.json', 'raw_mask_resnet_v1/protocol.json']},
        'dependencies': 'Complete all-window phase cache and observed extraction. Verify feature/kernel/source hashes at loading. At least14GiB free GPU before allocating development cache; no fallback/data cap.',
        'test_policy': 'Read only training/validation feature shards; no test labels used for selection or test predictions. Final endpoints/temporal policies must be frozen separately before test inference.',
    }
    S.mkdir();(S/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'frozen': True, 'modes': MODES, 'held_out_inference': False}))


def load_development(directory):
    feature_root = W/'phase_features_v1'
    status = json.loads((feature_root/'status.json').read_text())
    assert status.get('complete') and status['files'] == 289 and status['windows'] == 802491
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    feature_lock = json.loads((feature_root/'input_lock.json').read_text())
    assert feature_lock['protocol_sha256'] == sha(feature_root/'protocol.json')
    with np.load(W/'windows.npz') as z: data = {name: z[name] for name in z.files}
    partition_by_file = np.array([r['partition'] for r in manifest]);patient_by_file = np.array([r['patient_id'] for r in manifest])
    tr = np.flatnonzero(partition_by_file[data['file_index']] == 'train')
    va = np.flatnonzero(partition_by_file[data['file_index']] == 'validation')
    assert len(tr) == 504777 and len(va) == 166047
    ids = np.r_[tr, va];lookup = np.full(len(data['target']), -1, np.int64);lookup[ids] = np.arange(len(ids))
    response = torch.empty((len(ids), 2, 16, 128), dtype=torch.float32, device='cuda')
    phase_valid = np.zeros((len(ids), 2), bool);rr_valid = phase_valid.copy()
    rr = np.full((len(ids), 2, 16), np.nan, np.float32);coverage = np.zeros((len(ids), 2), np.float32)
    filled = np.zeros(len(ids), bool);source_hashes = {}
    for row in manifest:
        if row['partition'] == 'test': continue
        fi = int(row['file_index']);feature = feature_root/'shards'/f'file_{fi:03d}.npz'
        receipt = json.loads(feature.with_suffix('.json').read_text())
        assert sha(feature) == receipt['output_sha256'] and receipt['feature_protocol_sha256'] == sha(feature_root/'protocol.json')
        assert receipt['kernel_sha256'] == feature_lock['kernel_sha256']
        assert receipt['h5_sha256'] == row['h5_sha256']
        observed = W/'observed_features'/f'file_{fi:03d}.npz'
        assert sha(observed) == receipt['observed_shard_sha256']
        expected = np.flatnonzero(data['file_index'] == fi);local = lookup[expected]
        assert (local >= 0).all() and not filled[local].any()
        with np.load(feature) as z:
            assert np.array_equal(z['window_index'], expected) and z['response'].dtype == np.float32
            values = z['response'];valid = z['phase_valid']
            assert valid.dtype == bool
            assert np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all()
            response[torch.tensor(local, device='cuda')] = torch.from_numpy(values).cuda()
            phase_valid[local] = valid;coverage[local] = z['represented_fraction']
        with np.load(observed) as z:
            assert np.array_equal(z['window_index'], expected) and np.array_equal(z['phase_valid'], phase_valid[local])
            local_rr, local_valid = z['rr_features'], z['rr_valid']
            assert local_rr.dtype == np.float32 and local_valid.dtype == bool
            rr[local] = local_rr;rr_valid[local] = local_valid
        filled[local] = True;source_hashes[str(feature)] = receipt['output_sha256']
    assert filled.all()
    patients = patient_by_file[data['file_index'][ids]]
    (directory/'loaded_sources.json').write_text(json.dumps({'feature_status_sha256': sha(feature_root/'status.json'),
           'feature_input_lock_sha256': sha(feature_root/'input_lock.json'),
           'source_feature_hashes': source_hashes, 'test_feature_shards_read': False}, indent=2)+'\n')
    return response, phase_valid, rr, rr_valid, coverage, ids, patients, data


def forward(model, mode, response, available, phase_valid, ids, rr, rr_valid, coverage, mean, sd):
    if mode != 'hybrid': return model(response, available, phase_valid, ids)
    side, present = rr_side_input(rr, rr_valid, available, mean, sd)
    return model(response, available, phase_valid, ids, side, present, coverage)


@torch.inference_mode()
def validation_logits(model, mode, response, phase_valid, rr, rr_valid, coverage, mean, sd, ntrain, eligibility):
    model.eval();result = np.full((len(response)-ntrain, 3), np.nan, np.float32)
    for pattern_index, pattern in enumerate(PATTERNS):
        for start in range(ntrain, len(response), 256):
            indices = np.arange(start, min(start+256, len(response)))
            valid = eligibility[indices, pattern_index];indices = indices[valid]
            if not len(indices): continue
            idx = torch.tensor(indices, device='cuda');available = torch.tensor(pattern, device='cuda').expand(len(idx), -1)
            operator_ids = torch.tensor([0, 1], device='cuda').expand(len(idx), -1)
            logits = forward(model, mode, response[idx], available, phase_valid[idx], operator_ids,
                             rr[idx], rr_valid[idx], coverage[idx], mean, sd)
            assert torch.isfinite(logits).all()
            result[indices-ntrain, pattern_index] = logits.cpu().numpy()
    return result


def train(mode):
    protocol = json.loads((S/'protocol.json').read_text())
    for name, digest in protocol['source_hashes'].items(): assert sha(name) == digest, name
    for name, digest in protocol['input_hashes'].items(): assert sha(W/name) == digest, name
    directory = S/mode;assert not directory.exists(), 'Preserve prior model attempt'
    setup();assert torch.cuda.mem_get_info()[0] >= 14*2**30, 'Insufficient free GPU; do not cap cohort or silently retry'
    directory.mkdir();started = time.monotonic()
    response, pv, rr_cpu, rv, coverage_cpu, ids, patients, data = load_development(directory)
    ntrain = 504777
    if mode == 'hybrid': scaling = fit_rr_scaling(rr_cpu[:ntrain], rv[:ntrain])
    else: scaling = {'mean': np.zeros((2, 16)), 'sd': np.ones((2, 16)), 'unused_for_plain_set': True}
    np.savez_compressed(directory/'rr_scaling.npz', **scaling)
    mean = torch.tensor(scaling['mean'], dtype=torch.float32, device='cuda');sd = torch.tensor(scaling['sd'], dtype=torch.float32, device='cuda')
    base_valid = (pv | rv) if mode == 'hybrid' else pv
    eligibility = np.stack([(base_valid & np.asarray(pattern)).any(axis=1) for pattern in PATTERNS], axis=1)
    duration = (data['assigned_end'][ids]-data['assigned_start'][ids]).astype(np.int64)
    expanded_patient = np.repeat(patients[:ntrain], 3)
    weights = patient_sample_weights(expanded_patient, np.repeat(duration[:ntrain], 3), eligibility[:ntrain].reshape(-1)).reshape(ntrain, 3)
    val_weights = patient_sample_weights(patients[ntrain:], duration[ntrain:], np.ones(len(ids)-ntrain, bool))
    assert len(np.unique(patients[:ntrain])) == 90 and len(np.unique(patients[ntrain:])) == 30
    np.savez_compressed(directory/'development_manifest.npz', window_index=ids, eligibility=eligibility, train_weight=weights, validation_weight=val_weights)
    modality_counts = {}
    for name, selection in [('train', slice(0, ntrain)), ('validation', slice(ntrain, None))]:
        modality_counts[name] = {}
        for pattern in PATTERNS:
            has_phase = (pv[selection] & np.asarray(pattern)).any(axis=1)
            has_rr = (rv[selection] & np.asarray(pattern)).any(axis=1)
            modality_counts[name][str(pattern)] = {'both': int((has_phase & has_rr).sum()),
                'phase_only': int((has_phase & ~has_rr).sum()), 'rr_only': int((~has_phase & has_rr).sum()),
                'neither': int((~has_phase & ~has_rr).sum())}
    (directory/'fit_input_receipt.json').write_text(json.dumps({'protocol_sha256': sha(S/'protocol.json'),
        'development_manifest_sha256': sha(directory/'development_manifest.npz'), 'loaded_sources_sha256': sha(directory/'loaded_sources.json'),
        'rr_scaling_sha256': sha(directory/'rr_scaling.npz'), 'training_windows_declared': ntrain,
        'eligible_training_acquisition_examples': int(eligibility[:ntrain].sum()),
        'modality_counts_by_acquisition_pattern': modality_counts,
        'patient_weight_totals': {p: float(weights.reshape(-1)[expanded_patient == p].sum()) for p in np.unique(expanded_patient)},
        'test_feature_shards_read': False}, indent=2)+'\n')
    pv_gpu = torch.tensor(pv, device='cuda');rr = torch.tensor(rr_cpu, device='cuda');rv_gpu = torch.tensor(rv, device='cuda')
    coverage = torch.tensor(coverage_cpu, device='cuda');eligible_gpu = torch.tensor(eligibility, device='cuda')
    weight = torch.tensor(weights*3*ntrain/90, dtype=torch.float32, device='cuda')
    target = torch.tensor(data['target'][ids], dtype=torch.float32, device='cuda');global_ids = torch.tensor(ids[:ntrain], device='cuda')
    patterns = torch.tensor(PATTERNS, device='cuda');model = model_for(mode).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history = [];best = float('inf');best_epoch = -1
    for epoch in range(40):
        model.train();tick = time.monotonic();permutation = torch.randperm(ntrain, device='cuda');seen = 0
        total_loss = torch.zeros((), device='cuda')
        for start in range(0, ntrain, 256):
            declared = permutation[start:start+256];pattern_ids = availability_ids(global_ids[declared], epoch)
            keep = eligible_gpu[declared, pattern_ids];idx = declared[keep];pattern_ids = pattern_ids[keep]
            if not len(idx): continue
            available = patterns[pattern_ids];operator_ids = torch.tensor([0, 1], device='cuda').expand(len(idx), -1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = forward(model, mode, response[idx], available, pv_gpu[idx], operator_ids,
                                 rr[idx], rv_gpu[idx], coverage[idx], mean, sd)
                individual = nn.functional.binary_cross_entropy_with_logits(logits, target[idx], reduction='none')
                weighted = (individual*weight[idx, pattern_ids]).sum();loss = weighted/len(declared)
            assert torch.isfinite(loss);loss.backward();optimizer.step();total_loss += weighted.detach();seen += len(idx)
            if start % (256*200) == 0:
                print(json.dumps({'epoch': epoch, 'declared_rows_processed': min(start+256, ntrain),
                                  'eligible_rows_seen': seen, 'seconds': time.monotonic()-tick}), flush=True)
        logits = validation_logits(model, mode, response, pv_gpu, rr, rv_gpu, coverage, mean, sd, ntrain, eligibility)
        labels = data['target'][ids[ntrain:]].astype(np.float64)[:, None]
        safe = np.where(np.isfinite(logits), logits, 0.).astype(np.float64)
        terms = np.logaddexp(0, safe)-labels*safe  # Missing rows contribute log2 only to selection loss.
        val_loss = float(np.sum(terms*val_weights[:, None])/30/3)
        row = {'epoch': epoch, 'training_declared_rows': ntrain, 'training_eligible_rows_seen': seen,
               'training_weighted_bce': float(total_loss.cpu())/ntrain, 'validation_bce_all_rows': val_loss,
               'validation_indeterminate_acquisition_examples': int((~eligibility[ntrain:]).sum()), 'seconds': time.monotonic()-tick}
        history.append(row);print(json.dumps(row), flush=True)
        state = {'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'mode': mode,
                 'epoch': epoch, 'validation_loss': val_loss, 'rr_mean': mean.cpu(), 'rr_sd': sd.cpu()}
        if val_loss < best:
            best, best_epoch = val_loss, epoch;torch.save(state, directory/'best.pt');np.save(directory/'best_validation_logits.npy', logits)
        (directory/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        if epoch-best_epoch >= 8: break
    torch.save(state, directory/'final.pt')
    checkpoint = torch.load(directory/'best.pt', weights_only=True);replay_model = model_for(mode).cuda();replay_model.load_state_dict(checkpoint['state_dict'])
    replay = validation_logits(replay_model, mode, response, pv_gpu, rr, rv_gpu, coverage,
                               checkpoint['rr_mean'].cuda(), checkpoint['rr_sd'].cuda(), ntrain, eligibility)
    saved = np.load(directory/'best_validation_logits.npy');assert np.array_equal(replay, saved, equal_nan=True)
    scores = torch.sigmoid(torch.from_numpy(saved)).numpy();np.save(directory/'validation_scores.npy', scores)
    threshold = select_validation_threshold(scores.reshape(-1), np.repeat(data['af_assigned_samples'][ids[ntrain:]].astype(np.int64), 3),
                 np.repeat(duration[ntrain:], 3), np.repeat(patients[ntrain:], 3), allowed_validation_patients=np.unique(patients[ntrain:]))
    (directory/'threshold_selection.json').write_text(json.dumps(threshold, indent=2)+'\n')
    lock = {'complete_train_validation': True, 'mode': mode, 'test_inference_performed': False,
            'best_epoch': best_epoch, 'epochs_run': len(history), 'threshold': threshold['selected']['threshold'],
            'best_checkpoint_sha256': sha(directory/'best.pt'), 'final_checkpoint_sha256': sha(directory/'final.pt'),
            'protocol_sha256': sha(S/'protocol.json'), 'fit_input_receipt_sha256': sha(directory/'fit_input_receipt.json'),
            'validation_scores_sha256': sha(directory/'validation_scores.npy'), 'fresh_reload_validation_logits_bitwise': True,
            'parameters_total': sum(p.numel() for p in model.parameters()),
            'parameters_with_gradients': sum(p.numel() for p in model.parameters() if p.grad is not None),
            'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30, 'seconds_including_loading': time.monotonic()-started}
    (directory/'evaluation_lock.json').write_text(json.dumps(lock, indent=2)+'\n');print(json.dumps(lock), flush=True)


def smoke():
    setup();row = next(r for r in csv.DictReader((W/'file_manifest.csv').open()) if r['partition'] == 'train')
    with np.load(W/'observed_features'/f"file_{int(row['file_index']):03d}.npz") as z:
        rr_cpu, valid_cpu = z['rr_features'][:32], z['rr_valid'][:32]
    scaling = fit_rr_scaling(rr_cpu, valid_cpu)
    mean = torch.tensor(scaling['mean'], dtype=torch.float32, device='cuda');sd = torch.tensor(scaling['sd'], dtype=torch.float32, device='cuda')
    rr = torch.tensor(rr_cpu[:8], device='cuda');rv = torch.tensor(valid_cpu[:8], device='cuda');rv[0] = False
    response = torch.randn(8, 2, 16, 128, device='cuda');pv = torch.ones(8, 2, dtype=torch.bool, device='cuda')
    available = torch.tensor(PATTERNS, device='cuda')[torch.arange(8, device='cuda')%3]
    operator_ids = torch.tensor([0, 1], device='cuda').expand(8, -1);coverage = pv.float()*.8
    results = {}
    for mode in MODES:
        torch.manual_seed(42);model = model_for(mode).cuda();phase_valid = pv.clone()
        if mode == 'hybrid': phase_valid[1] = False
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = forward(model, mode, response, available, phase_valid, operator_ids, rr, rv, coverage, mean, sd)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0, 1]*4, device='cuda', dtype=torch.float32))
        loss.backward();assert torch.isfinite(loss)
        active = [p.grad for p in model.parameters() if p.grad is not None]
        assert active and all(torch.isfinite(g).all() for g in active)
        model.eval()
        with torch.inference_mode():
            a = forward(model, mode, response, available, phase_valid, operator_ids, rr, rv, coverage, mean, sd)
            b = forward(model, mode, response, available, phase_valid, operator_ids, rr, rv, coverage, mean, sd)
        assert torch.equal(a, b)
        results[mode] = {'finite_bf16_gradient_loss': float(loss.detach()), 'repeated_fp32_logits_bitwise': True,
                         'shape': list(a.shape), 'parameters_with_gradients': sum(p.numel() for p in model.parameters() if p.grad is not None)}
    print('WITNESS_AF_SET_TRAINER', json.dumps({'models': results, 'peak_gpu_gib_this_process': torch.cuda.max_memory_allocated()/2**30,
          'scope': 'Synthetic phase responses/labels, temporary scaling of actual training RR statistics; no production fit or test prediction'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true');group.add_argument('--train', choices=MODES);group.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.freeze: freeze()
    elif args.smoke: smoke()
    else: train(args.train)
