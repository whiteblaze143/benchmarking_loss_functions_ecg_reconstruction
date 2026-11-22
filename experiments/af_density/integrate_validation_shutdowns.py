"""Execute the18 frozen shutdown cases on the first raw validation segment."""
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import h5py
import numpy as np
from channel_shutdowns import select_available_scores
from decode_timeline import decode_segment
from elapsed_evaluation import evaluate_elapsed_segment
from elapsed_gap_strategies import gap_adjusted_timeline
from beat_clock_evaluation import released_rr_starts
from sample_clock_patterns import evaluate_sample_clock, compress_sample_clock

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
P = Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evaluate(reference, states, starts, ends, clock):
    result = evaluate_elapsed_segment(reference, states, starts, ends, 200)
    result['published_shared_RR_clock'] = evaluate_sample_clock(reference, states, starts, ends, 200, clock)
    c = result['sample_confusion'];total = int(ends[-1]-starts[0])
    result['predicted_burden'] = None if result['unknown_seconds'] else (c['true_positive']+c['false_positive'])/total
    result['burden_absolute_error'] = None if result['predicted_burden'] is None else abs(result['predicted_burden']-result['reference_burden'])
    return result


def main():
    output = W/'pre_release_corrective_v1/shutdown_integration_raw_first_validation_v1.json'
    assert not output.exists()
    raw = W/'raw_mask_resnet_v1';lock = json.loads((raw/'evaluation_lock.json').read_text())
    assert sha(raw/'validation_scores.npy') == lock['validation_scores_sha256']
    contract = json.loads((W/'controlled_shutdowns_v1/protocol.json').read_text())
    assert contract['complete'] and sha(W/'controlled_shutdowns_v1/schedules.npz') == contract['schedule_sha256']
    for p, digest in contract['source_hashes'].items(): assert sha(p) == digest
    protocol = json.loads((W/'evaluation_protocol_v1/protocol.json').read_text())
    for p, digest in protocol['source_hashes'].items(): assert sha(p) == digest
    assert sha(W/'duration_model_v1/fitted_law.npz') == protocol['primary_decoder']['duration_law_sha256']
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    row = next(r for r in files if r['partition'] == 'validation');fi = int(row['file_index'])
    with np.load(raw/'development_manifest.npz') as z:
        ids = z['window_index'][504777:];train_ids = z['window_index'][:504777];weights = z['train_weight']
    with np.load(W/'windows.npz') as z:
        selected = z['file_index'][ids] == fi;chosen = ids[selected]
        a, b = z['assigned_start'][chosen], z['assigned_end'][chosen]
        prior = float(np.dot(weights, z['target'][train_ids])/weights.sum())
    scores = np.load(raw/'validation_scores.npy')[selected]
    assert len(scores) == len(a) and np.isfinite(scores).all()
    refs = [(int(r['start_sample']), int(r['end_sample_exclusive'])) for r in
            csv.DictReader((W/'reference_intervals.csv').open()) if int(r['file_index']) == fi]
    with np.load(W/'controlled_shutdowns_v1/schedules.npz') as z: masks = z['missing_windows'][:, chosen]
    with np.load(W/'duration_model_v1/fitted_law.npz') as z: law = {k: z[k] for k in z.files}
    member = f"iridia-af-records-v1.0.1/{row['record_id']}/{row['record_id']}_rr_{int(row['daily_file_index']):02d}.h5"
    with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as archive:
        with h5py.File(io.BytesIO(archive.read(member))) as h: clock, clock_meta = released_rr_starts(h['rr'][:], 200)
    conditions = [];started = time.monotonic()
    for index, config in enumerate(contract['configurations']):
        missing = masks[index]
        for mode in contract['channel_modes']:
            available = np.ones((len(a), 2), bool)
            if mode in ['channel0_shutdown', 'both_shutdown']: available[missing, 0] = False
            if mode in ['channel1_shutdown', 'both_shutdown']: available[missing, 1] = False
            selected_scores = select_available_scores(scores, available)
            inside = scores[:, 1] if mode == 'channel0_shutdown' else scores[:, 0] if mode == 'channel1_shutdown' else np.full(len(a), np.nan)
            assert np.array_equal(selected_scores, np.where(missing, inside, scores[:, 2]), equal_nan=True)
            decoded = decode_segment(selected_scores, a, b, 200, {'mode': 'D2', 'posterior_prior': prior,
                'emission_strength': 1., 'probability_floor': 1e-6, 'nominal_input_seconds': 30.}, duration_model=law)
            states = decoded['states'];expected_missing = int(missing.sum()) if mode == 'both_shutdown' else 0
            assert decoded['unknown_windows'] == expected_missing
            strict = evaluate(refs, states, a, b, clock);secondary = []
            for strategy in ['nonaf', 'context', 'observed_only']:
                adjusted = gap_adjusted_timeline(refs, states, a, b, 200, strategy)
                entry = {k: v for k, v in adjusted.items() if k not in ['states', 'assigned_start', 'assigned_end', 'clock_map', 'reference_samples']}
                if adjusted['status'] == 'complete':
                    projected = compress_sample_clock(clock, adjusted['clock_map']) if strategy == 'observed_only' else clock
                    result = evaluate(adjusted['reference_samples'], adjusted['states'], adjusted['assigned_start'], adjusted['assigned_end'], projected)
                    removed = strict['sample_confusion']['reference_af']-result['sample_confusion']['reference_af']
                    assert removed == (strict['sample_confusion']['unknown_af'] if strategy == 'observed_only' else 0)
                    entry['results'] = result;entry['removed_reference_AF_samples'] = removed
                secondary.append(entry)
            conditions.append({'configuration': config, 'channel_mode': mode,
                'controlled_missing_samples': int((b-a)[missing].sum()),
                'strict_states': states.tolist(), 'strict_results': strict, 'secondary_gap_strategies': secondary})
            print(json.dumps({'completed_conditions': len(conditions), 'expected_conditions': 18}), flush=True)
    assert len(conditions) == 18
    result = {'complete': True, 'partition': 'validation', 'model': 'raw', 'file_index': fi,
        'patient_id': row['patient_id'], 'windows': len(a), 'conditions': conditions,
        'source_sha256': sha(__file__), 'model_lock_sha256': sha(raw/'evaluation_lock.json'),
        'shutdown_protocol_sha256': sha(W/'controlled_shutdowns_v1/protocol.json'),
        'evaluation_protocol_sha256': sha(W/'evaluation_protocol_v1/protocol.json'),
        'training_target_prior': prior, 'reference_RR_clock_metadata': clock_meta,
        'seconds': time.monotonic()-started, 'test_inference': False,
        'scope': 'One prespecified first validation segment,18 shutdown configurations, strictD2 and declared secondary gap conventions. Integration evidence only; not full-cohort robustness, measured SQI/noise validation or model superiority.'}
    output.write_text(json.dumps(result)+'\n')


if __name__ == '__main__': main()
