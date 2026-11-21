"""Fixed secondary gap conventions applied to saved validation timelines."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import h5py
import numpy as np
from beat_clock_evaluation import released_rr_starts
from elapsed_gap_strategies import gap_adjusted_timeline
from elapsed_evaluation import evaluate_elapsed_segment
from sample_clock_patterns import evaluate_sample_clock, compress_sample_clock

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
P = Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(model):
    parent = W/'evaluation_protocol_v1';source = parent/('validation_'+model)
    receipt = json.loads((source/'receipt.json').read_text());assert receipt['complete']
    output = parent/('validation_gaps_'+model);output.mkdir(exist_ok=False)
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    references = list(csv.DictReader((W/'reference_intervals.csv').open()))
    with np.load(W/'windows.npz') as z: starts, ends = z['assigned_start'], z['assigned_end']
    sources = [Path(__file__), Path(__file__).with_name('elapsed_gap_strategies.py'), Path(__file__).with_name('sample_clock_patterns.py')]
    (output/'input_receipt.json').write_text(json.dumps({'parent_receipt_sha256': sha(source/'receipt.json'),
        'robustness_policy_sha256': sha(parent/'robustness_addendum.json'),
        'source_hashes': {str(p): sha(p) for p in sources}, 'test_predictions': False}, indent=2)+'\n')
    completed = []
    with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as archive:
        for saved in receipt['outputs']:
            path = source/saved['file'];assert sha(path) == saved['sha256']
            original = json.loads(path.read_text());fi = original['file_index'];row = files[fi]
            ids = original['window_index'];a, b = starts[ids], ends[ids]
            reference = [(int(r['start_sample']), int(r['end_sample_exclusive'])) for r in references if int(r['file_index']) == fi]
            clock = None;results = []
            for condition in original['conditions']:
                for strategy in ['nonaf', 'context', 'observed_only']:
                    header = {'decoder': condition['decoder'], 'pattern': condition['pattern'], 'strategy': strategy}
                    if not condition['unknown_windows']:
                        results.append({**header, 'status': 'complete', 'original_missing_samples': 0,
                            'unchanged_strict_result_reused': True, 'results': condition['results']});continue
                    adjusted = gap_adjusted_timeline(reference, condition['states'], a, b, 200, strategy)
                    result = {**header, **{k: v for k, v in adjusted.items()
                        if k not in ['states', 'assigned_start', 'assigned_end', 'clock_map', 'reference_samples']}}
                    if adjusted['status'] == 'complete':
                        elapsed = evaluate_elapsed_segment(adjusted['reference_samples'], adjusted['states'],
                            adjusted['assigned_start'], adjusted['assigned_end'], 200)
                        confusion = elapsed['sample_confusion']
                        duration_samples = int(adjusted['assigned_end'][-1]-adjusted['assigned_start'][0])
                        elapsed['predicted_burden'] = (confusion['true_positive']+confusion['false_positive'])/duration_samples
                        elapsed['burden_absolute_error'] = abs(elapsed['predicted_burden']-elapsed['reference_burden'])
                        if clock is None:
                            member = f"iridia-af-records-v1.0.1/{row['record_id']}/{row['record_id']}_rr_{int(row['daily_file_index']):02d}.h5"
                            with h5py.File(io.BytesIO(archive.read(member))) as h: rr = h['rr'][:]
                            clock, _ = released_rr_starts(rr, 200)
                        projected = compress_sample_clock(clock, adjusted['clock_map']) if strategy == 'observed_only' else clock
                        elapsed['published_shared_RR_clock'] = evaluate_sample_clock(adjusted['reference_samples'], adjusted['states'],
                            adjusted['assigned_start'], adjusted['assigned_end'], 200, projected)
                        result['results'] = elapsed
                        result['removed_reference_AF_samples'] = condition['results']['sample_confusion']['reference_af']-confusion['reference_af']
                        result['reference_density_before'] = condition['results']['reference_patterns']['density_author_binned_30s_max']
                        result['reference_density_after'] = elapsed['reference_patterns']['density_author_binned_30s_max']
                    results.append(result)
            target = output/path.name
            target.write_text(json.dumps({'file_index': fi, 'patient_id': original['patient_id'], 'conditions': results})+'\n')
            completed.append({'file': target.name, 'sha256': sha(target)})
            print(json.dumps({'completed_files': len(completed), 'declared_files': 60}), flush=True)
    assert len(completed) == 60
    (output/'receipt.json').write_text(json.dumps({'complete': True, 'model': model, 'partition': 'validation',
        'outputs': completed, 'test_predictions': False,
        'limitations': 'Secondary sensitivity only; primary strict metrics untouched. Imputed time is not observed rhythm. Observed-only compression can change the reference differently across models and therefore cannot establish a common full-clock advantage.'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('model', choices=['rr', 'raw'])
    main(parser.parse_args().model)
