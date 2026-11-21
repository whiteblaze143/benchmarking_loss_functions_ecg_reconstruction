"""Inventory physical acquisition horizons separately from known daily prefixes."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

W = Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/waveform_v1')


def main():
    output = W/'horizon_inventory_v1';output.mkdir(exist_ok=False)
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    references = list(csv.DictReader((W/'reference_intervals.csv').open()))
    records = defaultdict(list)
    for row in files: records[row['record_id']].append(row)
    physical = [];daily = []
    for rid, rows in records.items():
        rows.sort(key=lambda r: int(r['daily_file_index']));total = 0;gaps = []
        for row in rows:
            offset = int(row['record_file_offset_samples']);length = int(row['source_samples'])
            assert offset == total;total += length
            gaps.append((offset, offset+int(row['excluded_prefix_samples'])))
        candidates = [('acquisition_prefix', h, 0, h*3600*200) for h in [6, 12, 24]]
        candidates += [('fixed_nonoverlap_24h', 24, i*86400*200, (i+1)*86400*200) for i in range(total//(86400*200))]
        for kind, hours, start, stop in candidates:
            available = stop <= total
            unknown = sum(max(0, min(stop, b)-max(start, a)) for a, b in gaps) if available else None
            physical.append({'record_id': rid, 'patient_id': rows[0]['patient_id'], 'partition': rows[0]['partition'],
                'kind': kind, 'hours': hours, 'source_start_sample': start, 'source_end_sample': stop,
                'sufficient_source_duration': available, 'excluded_or_unknown_samples': unknown,
                'strict_full_clock_eligible': available and unknown == 0})
    for row in files:
        start = int(row['excluded_prefix_samples']);length = int(row['source_samples']);fi = int(row['file_index'])
        for hours in [6, 12, 24]:
            stop = start+hours*3600*200;available = stop <= length
            events = [(max(start, int(r['start_sample'])), min(stop, int(r['end_sample_exclusive']))) for r in references
                if int(r['file_index']) == fi and int(r['start_sample']) < stop and int(r['end_sample_exclusive']) > start] if available else []
            daily.append({'file_index': fi, 'record_id': row['record_id'], 'patient_id': row['patient_id'],
                'partition': row['partition'], 'hours': hours, 'source_start_sample': start, 'source_end_sample': stop,
                'eligible_contiguous_daily_prefix': available, 'reference_AF_intervals': len(events) if available else None,
                'reference_AF_samples': sum(b-a for a, b in events) if available else None,
                'informative_multi_episode': available and len(events) >= 2})
    for name, rows in [('physical_horizons.csv', physical), ('daily_prefix_horizons.csv', daily)]:
        with (output/name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary = {'source_records': len(records), 'source_patients': len({r['patient_id'] for r in files}),
        'physical_fixed24h_blocks_with_enough_source': sum(r['kind'] == 'fixed_nonoverlap_24h' for r in physical),
        'physical_fixed24h_blocks_strictly_eligible': sum(r['kind'] == 'fixed_nonoverlap_24h' and r['strict_full_clock_eligible'] for r in physical),
        'daily_prefix_counts': {h: {'eligible_files': sum(r['hours'] == h and r['eligible_contiguous_daily_prefix'] for r in daily),
            'eligible_patients': len({r['patient_id'] for r in daily if r['hours'] == h and r['eligible_contiguous_daily_prefix']}),
            'informative_patients': len({r['patient_id'] for r in daily if r['hours'] == h and r['informative_multi_episode']})} for h in [6, 12, 24]},
        'input_contract_sha256': hashlib.sha256((W/'input_contract.json').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'interpretation': 'Acquisition-anchored physical horizons retain all measured calibration exclusions as unknown, so strict density may be infeasible. Uninterrupted daily prefixes start after each measured calibration exclusion and are a different explicitly qualified estimand. Decoder must be rerun using prefix-only local scores and no ECG beyond the prefix. Horizon dependence is not disease progression.'}
    assert summary['physical_fixed24h_blocks_strictly_eligible'] == 0
    assert summary['daily_prefix_counts'][24]['eligible_files'] == 0
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n');print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
