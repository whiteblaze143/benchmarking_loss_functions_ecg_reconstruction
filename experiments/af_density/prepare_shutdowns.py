"""Freeze all internal controlled-shutdown schedules before held-out inference."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from channel_shutdowns import shutdown_schedule

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    output = W/'controlled_shutdowns_v1';output.mkdir(exist_ok=False)
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    with np.load(W/'windows.npz') as z:
        fi = z['file_index'];assigned = z['assigned_end']-z['assigned_start'];af = z['af_assigned_samples']
    configs = [(fraction, block) for fraction in [.01, .05, .10] for block in [60, 300]]
    masks = np.zeros((len(configs), len(fi)), bool);receipts = []
    for row in manifest:
        indices = np.flatnonzero(fi == int(row['file_index']))
        for index, (fraction, block) in enumerate(configs):
            missing, receipt = shutdown_schedule(len(indices), dataset='iridia', record=row['record_id'],
                segment=int(row['daily_file_index']), fraction=fraction, block_seconds=block)
            masks[index, indices] = missing
            # Labels are used only after schedule selection for descriptive counts.
            receipt.update({'file_index': int(row['file_index']), 'configuration_index': index,
                'partition': row['partition'], 'missing_seconds': int(assigned[indices][missing].sum())/200,
                'missing_reference_af_seconds': int(af[indices][missing].sum())/200,
                'observation_seconds': int(assigned[indices].sum())/200})
            receipts.append(receipt)
    np.savez_compressed(output/'schedules.npz', missing_windows=masks)
    (output/'schedule_receipts.json').write_text(json.dumps(receipts)+'\n')
    contract = {'complete': True, 'files': len(manifest), 'windows': len(fi), 'configuration_count': len(configs),
        'configurations': [{'fraction': f, 'block_seconds': b} for f, b in configs],
        'channel_modes': ['channel0_shutdown', 'channel1_shutdown', 'both_shutdown'],
        'same_schedule_for_all_models_and_channel_modes': True, 'diagnostic_predictions_generated': False,
        'window_manifest_sha256': sha(W/'windows.npz'), 'file_manifest_sha256': sha(W/'file_manifest.csv'),
        'schedule_sha256': sha(output/'schedules.npz'), 'receipts_sha256': sha(output/'schedule_receipts.json'),
        'source_hashes': {str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('channel_shutdowns.py')]},
        'seed_serialization': 'SHA256 compact JSON list [dataset,record,segment,fraction,block_seconds,20261120], ASCII, first8 bytes little-endian. NumPy default_rng permutes disjoint candidate slots. Closest cumulative window count, at least one slot, earlier prefix on ties.',
        'limitations': 'Controlled availability after voltage-derived feature/prediction computation, not a measured SQI detector or physiological noise simulation. Missing duration and reference AF time are descriptive counts after label-independent placement.'}
    (output/'protocol.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(json.dumps({k: contract[k] for k in ['complete', 'files', 'windows', 'configuration_count']}))


if __name__ == '__main__': main()
