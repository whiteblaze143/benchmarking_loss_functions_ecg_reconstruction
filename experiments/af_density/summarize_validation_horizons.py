"""Describe horizon-specific and matched-source validation populations separately."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from summarize_validation_timelines import metrics

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')
E = W/'evaluation_protocol_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize(cases):
    result = {}
    for decoder in ['D0', 'D1', 'D2']:
        rows = [r for r in cases if r['decoder'] == decoder]
        patients = sorted({r['patient'] for r in rows})
        files = {r['file_index'] for r in rows};assert len(rows) == len(files)*3
        values = {}
        names = list(rows[0]['metrics']) if rows else []
        for name in names:
            by_patient = {}
            for patient in patients:
                selected = [r['metrics'][name] for r in rows if r['patient'] == patient]
                by_patient[patient] = None if any(v is None for v in selected) else float(np.mean(selected))
            defined = [v for v in by_patient.values() if v is not None]
            values[name] = {'declared_patients': len(patients), 'complete_patients': len(defined),
                'defined_segment_conditions': sum(r['metrics'][name] is not None for r in rows),
                'conditional_patient_mean': float(np.mean(defined)) if defined else None,
                'patient_values': by_patient}
        result[decoder] = {'files': len(files), 'declared_patients': len(patients), 'declared_conditions': len(rows), 'metrics': values}
    return result


def main(model):
    folder = E/('validation_horizons_'+model);receipt = json.loads((folder/'receipt.json').read_text());assert receipt['complete']
    inventory = [r for r in csv.DictReader((W/'horizon_inventory_v1/daily_prefix_horizons.csv').open()) if r['partition'] == 'validation']
    informative = {h: {int(r['file_index']) for r in inventory if r['hours'] == str(h) and r['informative_multi_episode'] == 'True'} for h in [6, 12, 24]}
    common = informative[6] & informative[12];all_cases = {h: [] for h in [6, 12, 24]}
    eligible_counts = {h: 0 for h in all_cases}
    for saved in receipt['outputs']:
        path = folder/saved['file'];assert sha(path) == saved['sha256'];data = json.loads(path.read_text());fi = data['file_index']
        for horizon in data['horizons']:
            hours = horizon['hours'];expected = next(r for r in inventory if int(r['file_index']) == fi and r['hours'] == str(hours))
            assert (horizon['status'] == 'complete') == (expected['eligible_contiguous_daily_prefix'] == 'True')
            if horizon['status'] != 'complete': continue
            eligible_counts[hours] += 1
            assert horizon['reference_episode_count'] == int(expected['reference_AF_intervals'])
            assert {(c['decoder'], c['pattern']) for c in horizon['conditions']} == {(d, p) for d in ['D0', 'D1', 'D2'] for p in range(3)}
            for condition in horizon['conditions']:
                all_cases[hours].append({'file_index': fi, 'patient': data['patient_id'], 'decoder': condition['decoder'],
                    'pattern': condition['pattern'], 'metrics': metrics(condition['results'])})
    results = {h: {'eligible_files': eligible_counts[h], 'all_eligible_reference_strata': summarize(cases),
        'informative_multi_episode': summarize([r for r in cases if r['file_index'] in informative[h]])} for h, cases in all_cases.items()}
    matched = {str(h): summarize([r for r in all_cases[h] if r['file_index'] in common]) for h in [6, 12]}
    full = [];parent = E/('validation_'+model);parent_receipt = json.loads((parent/'receipt.json').read_text())
    for saved in parent_receipt['outputs']:
        path = parent/saved['file'];assert sha(path) == saved['sha256'];data = json.loads(path.read_text())
        if data['file_index'] not in common: continue
        for condition in data['conditions']:
            full.append({'file_index': data['file_index'], 'patient': data['patient_id'], 'decoder': condition['decoder'],
                         'pattern': condition['pattern'], 'metrics': metrics(condition['results'])})
    matched['full_daily_segment'] = summarize(full)
    target = E/('validation_horizon_summary_'+model+'.json');assert not target.exists()
    target.write_text(json.dumps({'model': model, 'partition': 'validation', 'horizon_specific': results,
        'matched_multi_episode_6_12_full': matched, 'matched_file_indices': sorted(common),
        'input_receipt_sha256': sha(folder/'receipt.json'), 'source_sha256': sha(__file__),
        'limitations': 'Post-calibration daily prefixes, not acquisition-anchored monitoring. All-eligible summaries retain no-AF/single/continuous strata; informative summaries restrict by reference episodes at that horizon. Matched subset is source-defined as multi-episode at both6h and12h, with identical files across6h/12h/full. Metric-specific indeterminate patients remain explicit; no progression or general horizon-superiority claim.'}, indent=2)+'\n')
    print(json.dumps({h: r['informative_multi_episode']['D2'] for h, r in results.items()}, default=lambda v: v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('model', choices=['rr', 'raw'])
    main(parser.parse_args().model)
