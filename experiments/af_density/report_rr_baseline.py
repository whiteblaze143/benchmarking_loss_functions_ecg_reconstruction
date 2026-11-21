import csv,json
from collections import defaultdict
import numpy as np
from rr_protocol import O
E=O/'external_shdb_v3'
internal=json.loads((O/'summary.json').read_text());external=json.loads((E/'summary.json').read_text())
assert json.loads((O/'verification.json').read_text())['all_pass'] and json.loads((E/'verification.json').read_text())['all_pass']
patients=list(csv.DictReader((O/'patient_splits.csv').open()));split=np.array([p['split'] for p in patients])
with np.load(O/'windows.npz') as z:a={k:z[k] for k in z.files}
scores=np.load(O/'all_window_scores.npy');threshold=internal['threshold'];rng=np.random.default_rng(42)
by={}
for i,p in enumerate(patients):
    if p['split']!='test':continue
    m=a['patient_index']==i;pos=a['assigned_positive'][m].astype(np.int64);neg=a['assigned_count'][m].astype(np.int64)-pos;prediction=scores[m]>=threshold
    by[p['patient_id']]=(int(pos.sum()),int(neg.sum()),int(pos[prediction].sum()),int(neg[~prediction].sum()))
external_by=defaultdict(lambda:[0,0,0,0])
for r in csv.DictReader((E/'record_metrics.csv').open()):
    v=external_by[r['subject_id']]
    for j,key in enumerate(['af_rr','nonaf_rr','true_positive_rr','true_negative_rr']):v[j]+=int(r[key])
intervals={}
for dataset,data in [('internal',by),('external_shdb',external_by)]:
    intervals[dataset]={}
    for name,denom,numer in [('sensitivity',0,2),('specificity',1,3)]:
        values=np.array([v[numer]/v[denom] for v in data.values() if v[denom]])
        bootstrap=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
        intervals[dataset][name]={'contributing_patients':len(values),'patient_mean':float(values.mean()),'patient_bootstrap_95_low':float(np.quantile(bootstrap,.025)),'patient_bootstrap_95_high':float(np.quantile(bootstrap,.975))}
(O/'patient_detection_intervals.json').write_text(json.dumps(intervals,indent=2)+'\n')
patt=list(csv.DictReader((O/'patient_pattern_summary.csv').open()));test_patterns={r['metric']:r for r in patt if r['split']=='test'}
def number(value):return f'{float(value):.4f}'
t=internal['partitions']['test'];lines=['# Source-RR baseline: internal and external results','', '[INTEGRITY CONCERN] Initial experiment audit: FAIL (same-family, provisional). See EXPERIMENT_AUDIT.md, PROTOCOL_ERRATA.json and AS_EXECUTED_CONTRACT.json. Original plans and failed versions remain preserved; later corrections do not establish prospective sealing.', '',
'RR classification is strong, but this does not establish accurate elapsed-time AF density or a SetOperator contribution. This is a classical source-RR baseline trained from scratch, with a source-patient split, all eligible training windows, and a validation-only threshold.', '',
'## Frozen design','',
'IRIDIA: 152 source patient IDs split into 90 train, 30 validation and 32 test. All 324,367 training windows were fitted; 519,083 windows were prepared across all 167 records. No convenience cap, early stopping or pretrained classifier was used. The HistGradientBoosting model has 200 fixed iterations and 16 RR features. Feature statistics retain original adjacency; implausible RR values are flagged, not replaced by manufactured beats. The actual final 60-beat input covers the remaining tail without padding.', '',
'Training weights start as inverse patient window count and are then rebalanced by class. This does not preserve exactly equal final training weight per patient; the frozen protocol wording should be read with this qualification. Evaluation summaries, separately, give patients equal weight. See training_weight_audit.json.', '',
'Window targets are positive at at least 30/60 AF labels, including ties. The original majority wording was imprecise; there are 10 internal and 11 external tie windows.', '',
'Threshold 0.48 was selected on validation patient-mean beat balanced accuracy. Model, threshold and existing source hashes were frozen before internal test predictions. External SHDB-AF used the identical checkpoint and threshold, without fitting or tuning.', '',
'Calibration exclusion is a conservative native-RR proxy: first 31 intervals in the first daily file and 61 in later files. Pattern metrics are per daily piece on the observed native beat clock. They are not 24-hour elapsed-time Charitos density, and AF-beat proportion is not time burden.', '',
'## Results','',
'| Metric | Internal test | External SHDB-AF |','|---|---:|---:|',
f"| Source patients | 32 | 93 |",f"| Declared daily files / records | 48 | 98 |",f"| Windows | 87,593 | 172,542 |",f"| Window AUROC | {number(t['window_auroc'])} | {number(external['window_auroc'])} |",f"| Window AUPRC | {number(t['window_auprc'])} | {number(external['window_auprc'])} |",f"| Patient-mean beat sensitivity | {100*t['sensitivity_patient_mean_worst_case_indeterminate']:.2f}% | {100*external['sensitivity_patient_mean']:.2f}% |",f"| Patient-mean beat specificity | {100*t['specificity_patient_mean_worst_case_indeterminate']:.2f}% | {100*external['specificity_patient_mean']:.2f}% |",f"| Window Brier | {number(t['window_brier'])} | {number(external['window_brier'])} |"]
for metric,label in [('density_absolute_error','Beat-clock density MAE'),('aggregation_absolute_error','Beat-clock aggregation MAE'),('burden_absolute_error','AF-beat proportion MAE')]:
    x=test_patterns[metric];y=external['pattern_metrics'][metric]
    lines.append(f"| {label} | {number(x['patient_equal_weight_mean'])} | {number(y['patient_equal_weight_mean'])} |")
lines+=['',
'Patient bootstrap intervals are in patient_detection_intervals.json and patient_pattern_summary.csv, using 2000 draws. External sensitivity has 72 contributing subjects with AF, whereas specificity has 93. Window ranking metrics use windows as observations; no independent-window confidence intervals are claimed.', '',
'Density and aggregation errors are defined for 41/48 internal daily pieces (all 32 test patients have a defined piece) and 77/98 external records (72 subjects). All declared records remain in the tables. No true-AF internal piece was lost because the model produced an undefined pattern metric; the seven undefined reference pieces have no-AF/edge-case references. Across the prepared native-RR windows, representation coverage is 100%; this does not mean 100% ECG signal quality or beat-detection coverage.', '',
'## External competing rhythms','',
'| Reference rhythm | Positive AF calls / reference RR intervals | Fraction | Subjects |','|---|---:|---:|---:|']
groups={}
for r in csv.DictReader((E/'rhythm_subgroup_calls.csv').open()):
    v=groups.setdefault(r['rhythm'],[0,0,set()]);v[0]+=int(r['positive_calls_rr']);v[1]+=int(r['reference_rr']);v[2].add(r['subject_id'])
for rhythm in ['(N','(AFL','(AT','(NOD','(PAT','(AB']:
    positive,total,ids=groups[rhythm];lines.append(f'| {rhythm} | {positive:,}/{total:,} | {100*positive/total:.2f}% | {len(ids)} |')
lines+=['',
'These are pooled RR-level positive-call fractions, not independent patients or episode PPV. Small subgroups, particularly AT/PAT/NOD, have few subjects. Do not infer a physiological error mechanism from these counts alone.', '',
'## Integrity qualifications','',
'All 87,593 internal and all 172,542 final external scores replay bit-for-bit. Patient splits, source hash preservation, validation-only threshold recovery, full assignment coverage and actual RR feature values were checked independently.', '',
'The first external attempt stopped at an exactly duplicated QRS timestamp. Eleven redundant timestamps across seven source records were subsequently reduced to one entry each, with every removal logged. No new beat was inserted; decreasing or out-of-range timestamps remained hard failures. Version 1 and its failed source snapshot are preserved.', '',
'External version 2 used float64 feature values for inference but saved float32 features, unlike the frozen training preparation. Archive replay changed 1,629 scores and two decisions (maximum score difference 0.07274). Version 2 and its failed verification are preserved. Version 3 enforces training float32, reuses the same source features/checkpoint/threshold, reconstructs only affected binary records and explicitly verifies unchanged record vectors. All version-3 scores replay exactly. This was a technical consistency correction, not a performance-based model choice.', '',
'One native-RR/ECG reference endpoint in IRIDIA training record record_023 disagrees by 3,294.095 seconds (54.90 minutes). No offset has been fitted and neither annotation was silently declared correct. The baseline remains an as-released RR-label result; this prevents an elapsed-time measurement-fidelity claim until joint reference alignment is resolved.', '',
'The original evaluation loop repeatedly decompressed NPZ arrays. It was stopped after the checkpoint and scores were saved; finalization reused unchanged outputs and the frozen metric code with arrays cached once. Checkpoint, threshold and predictions were preserved. Exact model fit time was not persisted before this stop; the 3.24-second finalization time is not training time. The recorded launch commands requested two CPU threads. runtime_versions.json is a post-hoc environment snapshot, not contemporaneous fit telemetry; exact fit-time resource telemetry was not persisted.', '',
'## Decision','',
'Admit this as a source-RR baseline, not as the primary measurement-matched comparison. Source QRS detector channel availability is unknown. Waveform-only experiments must detect beats from their actually available channels and retain failures. No SetOperator model has been trained for this endpoint and no advantage has been established. Next requirements are elapsed-time reference reconciliation, observed-channel beat extraction, matched raw/set/RR inputs and the predeclared temporal-decoder comparison.']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Report saved',O/'REPORT.md')
