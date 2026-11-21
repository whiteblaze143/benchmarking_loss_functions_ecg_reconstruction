"""Subject-level waveform feasibility, preserving disjoint monitoring clocks."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from waveform_protocol import W, sha
from metrics import charitos_density, discrete_pattern_from_runs


def main():
    destination = W/'cohort_ledger_v1'
    assert not destination.exists(), 'Preserve the existing ledger'
    contract = json.loads((W/'input_contract.json').read_text())
    for name, digest in contract['input_hashes'].items():
        assert sha(W/name) == digest, name
    raw_counts = {}
    for record, digest in contract['label_csv_sha256'].items():
        source = W/'reference_ecg_labels'/(record+'.csv')
        assert sha(source) == digest
        raw_counts[record] = len(list(csv.DictReader(source.open())))
    references = defaultdict(list)
    for row in csv.DictReader((W/'reference_intervals.csv').open()):
        references[int(row['file_index'])].append((int(row['start_sample']), int(row['end_sample_exclusive'])))
    files = list(csv.DictReader((W/'file_manifest.csv').open()))
    patients = defaultdict(list)
    for row in files:
        fi = int(row['file_index']);start = int(row['excluded_prefix_samples']);stop = int(row['source_samples'])
        offset = int(row['record_file_offset_samples'])
        ref = [(max(a, start), min(b, stop)) for a, b in references[fi] if a < stop and b > start]
        complement = []
        cursor = start
        for a, b in ref:
            if a > cursor:
                complement.append((cursor, a))
            cursor = b
        if cursor < stop:
            complement.append((cursor, stop))
        af_samples = sum(b-a for a, b in ref)
        assert af_samples+sum(b-a for a, b in complement) == stop-start
        duration = (stop-start)/200
        intervals = [((a-start)/200, (b-start)/200) for a, b in ref]
        count = math.ceil(duration)
        centers = start+(np.arange(count)+.5)*(stop-start)/count
        labels = np.zeros(count, bool)
        for a, b in ref:
            labels |= (centers >= a) & (centers < b)
        run_starts = np.flatnonzero(labels & ~np.r_[False, labels[:-1]])
        run_ends = np.flatnonzero(labels & ~np.r_[labels[1:], False])+1
        discrete = discrete_pattern_from_runs(list(zip(run_starts.tolist(), run_ends.tolist())), count)
        patients[row['patient_id']].append({
            'file_index': fi, 'record_id': row['record_id'], 'partition': row['partition'],
            'daily_file_index': int(row['daily_file_index']),
            'record_offset_samples': offset, 'segment_start_sample_in_file': start,
            'segment_end_sample_in_file': stop, 'observed_seconds': duration,
            'source_file_seconds': stop/200, 'excluded_prefix_seconds': start/200,
            'af_seconds': af_samples/200, 'af_burden': af_samples/(stop-start),
            'af_interval_count': len(ref),
            'longest_observed_interval_seconds': max((b-a for a, b in ref), default=0)/200,
            'af_intervals_record_samples': [(a+offset, b+offset) for a, b in ref],
            'nonaf_complement_intervals_record_samples': [(a+offset, b+offset) for a, b in complement],
            'excluded_intervals_record_samples': [(offset, offset+start)],
            'density_continuous': charitos_density(intervals, duration),
            'density_equal_timebin_analogue': discrete['density'],
            'aggregation_equal_timebin_analogue': discrete['aggregation'],
            'analogue_bin_seconds': duration/count,
            'analogue_af_seconds_difference': float(labels.sum()*duration/count-af_samples/200),
            'left_censored_af': bool(ref and ref[0][0] == start),
            'right_censored_af': bool(ref and ref[-1][1] == stop),
            'source_rr_annotation_disagreement_flag': row['source_rr_annotation_disagreement_flag'] == 'True',
        })
    rows = []
    for patient, segments in sorted(patients.items()):
        partitions = {s['partition'] for s in segments};assert len(partitions) == 1
        records = sorted({s['record_id'] for s in segments})
        density = [s['density_continuous'] for s in segments if s['density_continuous'] is not None]
        aggregation = [s['aggregation_equal_timebin_analogue'] for s in segments if s['aggregation_equal_timebin_analogue'] is not None]
        observed = sum(s['observed_seconds'] for s in segments)
        af = sum(s['af_seconds'] for s in segments)
        rows.append({
            'patient_id': patient, 'partition': next(iter(partitions)), 'recordings': len(records),
            'record_ids': ';'.join(records), 'contiguous_segments': len(segments),
            'source_waveform_seconds': sum(s['source_file_seconds'] for s in segments),
            'calibration_excluded_seconds': sum(s['excluded_prefix_seconds'] for s in segments),
            'observed_seconds_before_clinical_quality': observed,
            'af_seconds_on_observed_time': af, 'af_burden_on_observed_time': af/observed,
            'released_episode_count_raw': sum(raw_counts[r] for r in records),
            'reference_intervals_within_segments': sum(s['af_interval_count'] for s in segments),
            'multi_episode_contiguous_segments': sum(s['af_interval_count'] > 1 for s in segments),
            'has_multi_episode_contiguous_segment': any(s['af_interval_count'] > 1 for s in segments),
            'longest_observed_interval_within_segment_seconds': max(s['longest_observed_interval_seconds'] for s in segments),
            'density_continuous_defined_segments': len(density),
            'mean_segment_density_continuous': float(np.mean(density)) if density else None,
            'aggregation_timebin_analogue_defined_segments': len(aggregation),
            'mean_segment_aggregation_timebin_analogue': float(np.mean(aggregation)) if aggregation else None,
            'uninterrupted_24h_segments': sum(s['observed_seconds'] >= 86400 for s in segments),
            'stored_channel_ids': '0;1', 'voltage_units': 'author-loader documented mV',
            'canonical_operator_geometry_known': False, 'permitted_set_modes': 'categorical;q_ablated',
            'clinical_quality_adjudicated': False, 'observed_peak_extraction_complete': False,
            'source_rr_annotation_disagreement_flag': any(s['source_rr_annotation_disagreement_flag'] for s in segments),
        })
    assert len(rows) == 152 and len(files) == 289 and len(raw_counts) == 167
    assert sum(r['released_episode_count_raw'] for r in rows) == 388
    summary = {partition: {
        'patients': sum(r['partition'] == partition for r in rows),
        'patients_with_multi_episode_contiguous_segment': sum(r['partition'] == partition and r['has_multi_episode_contiguous_segment'] for r in rows),
        'multi_episode_contiguous_segments': sum(r['multi_episode_contiguous_segments'] for r in rows if r['partition'] == partition),
    } for partition in ['train', 'validation', 'test']}
    destination.mkdir()
    with (destination/'subjects.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (destination/'subject_timelines.jsonl').open('w') as f:
        for patient, segments in sorted(patients.items()):
            f.write(json.dumps({'patient_id': patient, 'segments': segments}, allow_nan=False)+'\n')
    receipt = {
        'complete': True, 'patients': len(rows), 'records': len(raw_counts), 'files': len(files),
        'released_episodes': 388, 'partitions': summary,
        'observed_hours_before_clinical_quality': sum(r['observed_seconds_before_clinical_quality'] for r in rows)/3600,
        'calibration_excluded_seconds': sum(r['calibration_excluded_seconds'] for r in rows),
        'uninterrupted_24h_segments': sum(r['uninterrupted_24h_segments'] for r in rows),
        'input_contract_sha256': sha(W/'input_contract.json'), 'source_sha256': sha(Path(__file__)),
        'output_hashes': {name: sha(destination/name) for name in ['subjects.csv', 'subject_timelines.jsonl']},
        'limitations': ['Descriptive gate, not model predictions or clinically adjudicated waveform quality.',
                       'Mean segment density/aggregation is not density/aggregation of a concatenated patient timeline.',
                       'Non-AF complement is not a clinically adjudicated normal sinus rhythm annotation.',
                       'Time-bin aggregation is an explicitly labelled analogue, not the published RR-beat-clock endpoint.',
                       'Peak extraction completion and representation coverage must be joined only after its live run completes.'],
    }
    (destination/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
