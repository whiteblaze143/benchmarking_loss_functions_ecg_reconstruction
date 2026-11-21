"""Produce the Phase1/2 gate decision; never launches a target comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    integrity = json.loads((root / 'manifests/release_integrity.json').read_text())
    proof = json.loads((root / 'reports/window_export_semantic_failure.json').read_text())
    frame = pd.read_parquet(root / 'manifests/segment_manifest.parquet')
    eligibility = pd.read_csv(root / 'cohorts/eligibility_by_model.csv', dtype={'recording_prefix': str}).fillna('')
    admission = []
    contracts = {}
    for name in ['RawECGNet', 'ArNet2']:
        contract = json.loads((root / f'configs/model_contracts/{name}.json').read_text())
        assert contract['scientific_admission'] == 'rejected_as_shipped_window_export'
        contracts[name] = contract
        admission.append({'model_name': name, 'source_commit': contract['source_repository_commit'],
                          'checkpoint_sha256': contract['checkpoint_sha256'],
                          'development_execution_passed': True, 'original_runner_parity_passed': True,
                          'fixed_contextual_threshold': contract['output']['threshold'],
                          'native_input': '6000 physical-mV samples at200Hz' if name == 'RawECGNet' else '60 measured RR intervals in seconds',
                          'scientific_admission': 'rejected_as_shipped_window_export',
                          'reason': 'predict_windows applies sigmoid to an already sigmoid probability; contextual operating point not verified for a recovered standalone head',
                          'native_CNN_repair_candidate': 'executable diagnostics only; no standalone operating-point admission',
                          'full_contextual_model': 'not admitted for isolated-strip comparison; cannot stitch disjoint records or claim unavailable history'})
    other = [
        ('Single-lead ECG-FM AF adaptation', 'reference implementation explicitly excludes pretrained/fine-tuned checkpoints and evaluation scripts'),
        ('Mishra2026 native10s', '12-lead slot mapping requires verified lead identity; A/B cannot be assumed LeadII; no compatible frozen execution demonstrated'),
        ('SAFER', 'published source describes code and on-request additional materials; frozen compatible checkpoint not verified'),
        ('OpenECG', 'shipped boundary/codec tools are not an admitted published AF binary classifier with a verified source operating point'),
        ('FOUND-AF', 'released code extracts embeddings and trains XGBoost readouts; no frozen task-specific head admitted without new training'),
        ('Archived SetOperator', 'previously trained on Noise Holter; historical description only, not a frozen external comparator'),
    ]
    for name, reason in other:
        admission.append({'model_name': name, 'scientific_admission': 'not_admitted', 'reason': reason})
    pd.DataFrame(admission).to_csv(root / 'manifests/model_admission.csv', index=False)
    eligibility['scientific_model_admission'] = 'rejected_as_shipped_window_export'
    eligibility.to_csv(root / 'cohorts/eligibility_by_model.csv', index=False)
    support = eligibility[eligibility.technical_input_available == True].groupby(['model_candidate', 'cohort']).agg(
        strips=('strip_id', 'size'), recording_prefixes_NOT_PATIENTS=('recording_prefix', 'nunique')).reset_index()
    support.to_csv(root / 'cohorts/model_technical_support_summary.csv', index=False)
    common_ids = set.intersection(*(set(eligibility.loc[(eligibility.model_candidate == model) & eligibility.technical_input_available, 'strip_id'])
                                   for model in ['RawECGNet_30s', 'ArNet2_60RR']))
    common = frame[frame.strip_id.isin(common_ids)]
    common[['strip_id', 'recording_prefix', 'primary_clinical_cohort', 'duration_seconds', 'channel_identity']].to_csv(root / 'cohorts/common_technical_support.csv', index=False)
    locked = {'study': 'Noise Holter AF reliability, retrospective existing-label study',
              'release_sha256': integrity['archive_sha256'], 'manifest_sha256': integrity['manifest_sha256'],
              'model_contracts': {name: {'checkpoint_sha256': contract['checkpoint_sha256'],
                                       'source_commit': contract['source_repository_commit'],
                                       'fixed_contextual_threshold': contract['output']['threshold'],
                                       'admission': contract['scientific_admission']} for name, contract in contracts.items()},
              'phase2_passed': False, 'large_comparison_enabled': False, 'admitted_primary_models': [],
              'cohort_policy': {'AF_precedence_over_ectopy': True, 'AFL_separate': True,
                                'primary_ectopy': ['SVPB', 'VPB'], 'extended_ectopy_sensitivity': ['SV Couplet', 'V Couplet', 'TgSV'],
                                'EVENTS_None_is_negative': False, 'N2_role': 'separate unsafe-output and abstention only',
                                'duplicate_conflicts': 'exclude_all', 'duplicates_nonconflicting': 'lexicographically_first',
                                'channel_selection': 'first genuinely observed channel in original header, never quality-selected',
                                'anatomical_channel_mapping': 'unknown', 'patient_mapping': 'unknown'},
              'synthetic_plan_NOT_EXECUTED': {'noise_database': 'NSTDB bw,ma,em', 'snr_db': [24, 12, 6, 0],
                                              'realizations': 3, 'seed': 42, 'noise_resampling': 'polyphase to actual ECG sampling rate',
                                              'power_definition': 'mean-removed whole-input signal/noise power; report achieved SNR',
                                              'clipping': 'none allowed; no hidden waveform clipping or noise tuning'},
              'analysis_policy': {'missed_AF_denominator': 'all prespecified eligible AF inputs including failed/abstained',
                                  'competing_rhythm_denominator': 'explicit competing annotations without AF/AFL, provisional negatives',
                                  'weighting': ['strip_descriptive', 'equal_recording_sensitivity'],
                                  'uncertainty': 'exploratory recording-prefix clusters, never patient confidence intervals',
                                  'AF_leave_one_recording_out': True, 'two_models': 'ordering only, no predictive correlation'},
              'outcome_scope': 'No comparative target predictions inspected or generated in this implementation. Data were exposed in previous work; this is not prospective preregistration.',
              'stop_reason': 'No scientifically admitted short-input baseline at this gate; do not generate a noise ranking from incorrect score semantics.'}
    (root / 'configs/experiment_locked.yaml').write_text(yaml.safe_dump(locked, sort_keys=False))
    (root / 'manifests/label_semantics.md').write_text('''# Source label contract

AF means explicit A. Fib. header annotation. AF overrides co-occurring ectopy.
AFL is retained separately and never used as an AF-negative ectopy control.
Primary competing labels are SVPB/VPB without AF/AFL; couplets/trigeminy form a
separate sensitivity cohort. Absence of AF/AFL is not independently adjudicated
AF absence: these controls remain provisional. EVENTS:None is not NSR.
N2 is a safety/abstention set, not a rhythm-negative set. Contextual strip labels
do not establish the rhythm throughout a cropped model window or isolated-strip
human diagnostic observability. No reader labels were created.

Recording prefixes are imperfect clusters, not verified patients. Device,
protocol, patient ID and A/B lead geometry remain unknown. No reference .atr
or .qrs peaks enter the model-input/extractor eligibility pipeline. Those files
are hashed for provenance only. Original header text and per-file hashes are
preserved in the master manifest. Duplicate label conflicts are excluded in
full; nonconflicting physical duplicates retain the first ID.
''')
    selected = support[support.cohort.str.contains('N1_AF|N1_ectopy', regex=True)]
    (root / 'reports/feasibility_report.md').write_text(f'''# Phase1/2 feasibility

Frozen release: {len(frame)} strips, {frame.recording_prefix.nunique()} recording
prefixes, zero verified patient IDs. Every supplied checksum matched.
{integrity['duplicate_noncanonical_strips']} noncanonical duplicate strips and
{integrity['duplicate_conflict_strips']} conflict strips were excluded. All raw
rows, header text and annotation hashes remain available in the master manifest.

Clinical groups are exclusive for primary analysis. Strict SVPB/VPB controls
have 116 N1 strips at 10 seconds, compared with 125 in the earlier extended ectopy
union; both have 29 at 30 seconds. These are different cohort definitions, not
missing records. AF remains 110 N1 strips/nine prefixes at 10 seconds and 17/five
at 30 seconds before further model-input exclusions.

Source-native technical support after duplicate/label/input exclusions:

```
{selected.to_string(index=False)}
```

Common technical support contains {len(common)} strips across all groups; this
is not an admitted model-comparison cohort. Exact membership is in
`cohorts/common_technical_support.csv`. The RR availability receipt documents
the source epltd executable, local WFDB dependency, observed warm-up policy and
all failures. It does not silently substitute an oracle detector or certify
publication-level filtering equivalence.

The as-shipped standalone window exports failed the semantic gate. Successful
model execution and author-runner parity do not override this failure.
''')
    (root / 'reports/benchmark_results.md').write_text('''# Benchmark status

NOT EXECUTED. Phase2 scientific admission failed before target inference.
There are no Clean/N1/N2 or synthetic model predictions in this run and no
clinical miss/call rates, model ordering, correlation, bootstrap interval or
publication result to report. Synthetic generation and target comparison were
not launched. Empty prediction files are not presented as completed experiments.

Only target-independent development predictions were used for API/parity and
export-semantic checks. The original released checkpoints were not modified.
The saved native CNN-head diagnostic repair is not silently substituted for
the contextual model or given an unverified standalone operating point.
''')
    (root / 'reports/hrs_decision.md').write_text('''# NO-GO at the frozen-model admission gate

Phase1 is complete. Phase2 execution and source-input auditing are complete,
but scientific admission did not pass. Both local window exports apply an
additional sigmoid to learned sigmoid probabilities. Returned scores therefore
lie between 0.5 and 0.731059. Their recovered contextual fixed thresholds are
RawECGNet 0.425999999 and ArNet2 0.432074189, so naively using those thresholds
on the window exports makes every input positive. This is an inference-export
problem, not evidence that clinical AF models necessarily fail under noise.

The underlying frozen CNN head can be called directly and its original
probabilities recovered. That repair candidate is implemented and witnessed,
but a full contextual model's operating point is not automatically a validated
standalone-head operating point. Do not rescale or tune thresholds on N1 to
conceal the mismatch. No verified compatible native 10-second frozen AF model
was admitted in the audited candidates; code-only encoders/readout-training
frameworks do not supply such a classifier.

Under the attachment's stop rule, a comparative robustness ranking must stop
when fewer than two valid models are admitted. Thus Phases 3–5 are gated off;
no new training, target benchmark, synthetic result or HRS abstract is claimed.
The small 30-second N1 AF support, provisional event-negative semantics and
unknown patient linkage also remain limits even after an export repair.

To reopen the gate: establish the correct published standalone score/decision
contract or obtain a verified repaired export with its source-locked operating
point, reproduce it on target-independent examples, and re-freeze the contract.
The manifest, source hashes, development fixtures, failure proof and executable
repair probe make that next step reviewable. No outside investigator messages
were sent and no clinical-reader labels were invented.
''')
    index_paths = [root / 'configs/experiment_locked.yaml', root / 'manifests/model_admission.csv',
                   root / 'manifests/segment_manifest.parquet', root / 'cohorts/eligibility_by_model.csv',
                   root / 'reports/window_export_semantic_failure.json', root / 'reports/hrs_decision.md']
    receipt = {'phase1_complete': True, 'phase2_execution_verified': True,
               'phase2_scientific_admission_passed': False, 'admitted_primary_models': 0,
               'benchmark_launched': False,
               'files': {str(path.relative_to(root)): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() for path in index_paths}}
    (root / 'reports/admission_completion_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
