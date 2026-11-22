"""Measure frozen window-target discretization on validation only; no fitting."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from elapsed_evaluation import evaluate_elapsed_segment

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def main():
    output = W/'pre_release_corrective_v1/window_oracle_validation'
    output.mkdir(parents=True, exist_ok=False)
    with np.load(W/'windows.npz') as z:
        windows = {k: z[k] for k in z.files}
    manifest = list(csv.DictReader((W/'file_manifest.csv').open()))
    references = list(csv.DictReader((W/'reference_intervals.csv').open()))
    rows = []
    for row in manifest:
        if row['partition'] != 'validation':
            continue
        fi = int(row['file_index'])
        take = windows['file_index'] == fi
        ref = [(int(r['start_sample']), int(r['end_sample_exclusive']))
               for r in references if int(r['file_index']) == fi]
        result = evaluate_elapsed_segment(ref, windows['target'][take],
            windows['assigned_start'][take], windows['assigned_end'][take], 200)
        rows.append({'file_index': fi, 'patient_id': row['patient_id'], 'results': result})
        (output/f'file_{fi:03d}.json').write_text(json.dumps(rows[-1])+'\n')
    assert len(rows) == 60
    inputs = ['windows.npz', 'file_manifest.csv', 'reference_intervals.csv',
              'evaluation_protocol_v1/protocol.json']
    receipt = {'complete': True, 'partition': 'validation', 'files': len(rows),
        'patients': len({r['patient_id'] for r in rows}), 'test_access': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256': {p: hashlib.sha256((W/p).read_bytes()).hexdigest() for p in inputs},
        'scope': 'Unmodified majority-window targets reconstructed on original assigned time. Diagnostic discretization error, not a universal lower bound; no decoder, fitting, or label changes.'}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
