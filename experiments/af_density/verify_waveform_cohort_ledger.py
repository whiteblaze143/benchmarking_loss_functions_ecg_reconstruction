"""Verify the subject ledger against an earlier independent timeline audit."""
import csv
import json
from pathlib import Path
import numpy as np
from waveform_protocol import W, P, sha


def main():
    d = W/'cohort_ledger_v1'
    receipt = json.loads((d/'receipt.json').read_text())
    for name, digest in receipt['output_hashes'].items():
        assert sha(d/name) == digest, name
    assert sha(W/'input_contract.json') == receipt['input_contract_sha256']
    original = {(r['record_id'], int(r['daily_file_index'])): r
                for r in csv.DictReader((P/'contiguous_segment_feasibility.csv').open())}
    exclusions = {(r['record_id'], int(r['daily_file_index'])): r
                  for r in csv.DictReader((P/'calibration_exclusion_intervals.csv').open())}
    source_subjects = {r['patient_id']: r for r in csv.DictReader((P/'subject_feasibility_ledger.csv').open())}
    subjects = {r['patient_id']: r for r in csv.DictReader((d/'subjects.csv').open())}
    splits = {r['patient_id']: r['split'] for r in csv.DictReader((W/'patient_splits.csv').open())}
    checked = 0;density_delta = 0.;informative = set()
    for line in (d/'subject_timelines.jsonl').read_text().splitlines():
        patient = json.loads(line);pid = patient['patient_id'];subject = subjects[pid]
        assert subject['partition'] == splits[pid]
        assert int(subject['released_episode_count_raw']) == int(source_subjects[pid]['released_af_intervals'])
        total_af = total_observed = total_excluded = 0.
        for s in patient['segments']:
            key = (s['record_id'], s['daily_file_index']);old = original[key];excluded = exclusions[key]
            assert old['patient_id'] == pid and s['partition'] == splits[pid]
            start = s['record_offset_samples']+s['segment_start_sample_in_file']
            stop = s['record_offset_samples']+s['segment_end_sample_in_file']
            assert start == int(old['start_sample']) and stop == int(old['end_sample_exclusive'])
            assert s['af_interval_count'] == int(old['af_intervals'])
            np.testing.assert_allclose(s['af_seconds'], float(old['af_seconds']), rtol=0, atol=1e-8)
            np.testing.assert_allclose(s['observed_seconds'], float(old['observation_seconds']), rtol=0, atol=1e-8)
            if old['density_contiguous_segment']:
                delta = abs(s['density_continuous']-float(old['density_contiguous_segment']))
                assert delta < 1e-12;density_delta = max(density_delta, delta)
            else:
                assert s['density_continuous'] is None
            assert s['excluded_intervals_record_samples'] == [[int(excluded['excluded_start_sample']), int(excluded['excluded_end_sample_exclusive'])]]
            intervals = sorted(s['af_intervals_record_samples']+s['nonaf_complement_intervals_record_samples'])
            assert intervals[0][0] == start and intervals[-1][1] == stop
            assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
            assert all(a < b for a, b in intervals)
            total_af += s['af_seconds'];total_observed += s['observed_seconds'];total_excluded += s['excluded_prefix_seconds']
            if s['af_interval_count'] > 1:
                informative.add(pid)
            checked += 1
        np.testing.assert_allclose(total_af, float(subject['af_seconds_on_observed_time']), rtol=0, atol=1e-8)
        np.testing.assert_allclose(total_observed, float(subject['observed_seconds_before_clinical_quality']), rtol=0, atol=1e-8)
        assert total_excluded == float(subject['calibration_excluded_seconds'])
    assert checked == 289 and len(subjects) == 152 and len(informative) == 57
    result = {'all_pass': True, 'patients': 152, 'segments': checked, 'informative_multi_episode_patients': 57,
              'max_density_difference_from_prior_independent_audit': density_delta,
              'independent_audit_input_hashes': {name: sha(P/name) for name in ['contiguous_segment_feasibility.csv', 'calibration_exclusion_intervals.csv', 'subject_feasibility_ledger.csv']},
              'scope': 'Checks source-patient partitions, source episode counts, all segment clocks/AF counts, exact AF/non-AF coverage, exclusions and continuous density; not clinical quality or peak accuracy'}
    (d/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
