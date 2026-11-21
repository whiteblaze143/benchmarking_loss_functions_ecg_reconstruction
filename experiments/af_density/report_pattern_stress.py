"""Report controlled stress outcomes without converting them into model claims."""
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/controlled_pattern_stress')
r=list(csv.DictReader((P/'case_results.csv').open()));q=list(csv.DictReader((P/'second_grid_quantization.csv').open()))
rng=np.random.default_rng(20261120);paired=[]
for scenario in dict.fromkeys(x['scenario'] for x in r):
    rows=[x for x in r if x['scenario']==scenario];by_patient=defaultdict(list)
    for x in rows:
        if x['density_absolute_error'] and x['aggregation_absolute_error']:
            by_patient[x['patient_id']].append(float(x['density_absolute_error'])-float(x['aggregation_absolute_error']))
    vals=np.array([np.mean(v) for _,v in sorted(by_patient.items())])
    if len(vals):
        boot=vals[rng.integers(0,len(vals),(2000,len(vals)))].mean(axis=1)
        paired.append({'scenario':scenario,'declared_segments':len(rows),'paired_defined_segments':sum(bool(x['density_absolute_error']) and bool(x['aggregation_absolute_error']) for x in rows),'paired_patient_ids':len(vals),'mean_density_minus_aggregation_absolute_error':float(vals.mean()),'patient_bootstrap_95_low':float(np.quantile(boot,.025)),'patient_bootstrap_95_high':float(np.quantile(boot,.975))})
with (P/'paired_metric_error_comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
quantization_max=max(abs(float(x['source_continuous_density'])-float(x['second_grid_continuous_density'])) for x in q if x['source_continuous_density'] and x['second_grid_continuous_density'])
s=list(csv.DictReader((P/'patient_weighted_summary.csv').open()))
lookup={(x['scenario'],x['metric']):x for x in s}
lines=['# Controlled reference-pattern stress test','',
'No detector was trained or evaluated. This is an annotation-perturbation experiment on 71 calibration-excluded contiguous daily segments from 57 IRIDIA patient IDs. All segments were declared before metric computation. Results are conditional on existing labels and have no clinical waveform adjudication.',
'', 'Both metrics use the published discrete formulas on one-second time bins. These are explicitly time-grid analogues, not RR-beat metrics or reproduction of the original detector/SQI experiment. Bin-center labeling changed no episode counts in these segments; maximum continuous-density difference due to one-second quantization was '+f'{quantization_max:.6f}.', '',
'| Alteration | Mean density absolute error | Mean aggregation absolute error | Defined / declared segments |',
'|---|---:|---:|---:|']
for scenario in ['miss_shortest','miss_longest','split_longest_up_to_30s','merge_closest','false_episode_30s','midpoint_shutdown_60s_nonaf','midpoint_shutdown_60s_context','midpoint_shutdown_60s_observed_only']:
    a=lookup[(scenario,'density_absolute_error')];b=lookup[(scenario,'aggregation_absolute_error')]
    lines.append(f"| {scenario} | {float(a['patient_equal_weight_mean']):.6f} | {float(b['patient_equal_weight_mean']):.6f} | {a['defined_metric_segments']}/{a['declared_segments']} |")
lines+=['','Means give each source patient equal weight after averaging that patient\'s segments. Patient bootstrap intervals (2000 draws, seed 20261120) and paired error differences are in the CSVs. The three undefined merge cases are retained in the declared 71; their metrics are not assigned zero error. Sensitivity, specificity, burden error and episode split/merge counts are measured directly from each altered binary timeline.', '',
'The outcome is mixed. Density is more affected by merging, whereas aggregation is more affected by missing the longest episode. This pilot does not establish that either metric is universally more reliable, reproduce the published ICCs, or show a SetOperator benefit.', '',
'Observed-only handling compares two identically compressed timelines, explaining its zero error. It changes the target clock; original-versus-compressed reference metric changes are logged per case. This cannot be interpreted as recovery of the full original monitoring period.', '',
'Gap context filling uses only observed flanking predictions. Equal flanks fill the gap with that state; different flanks split at the midpoint. A terminal gap without both flanks remains indeterminate. The controlled midpoint shutdown is fixed at 60 seconds at the center of each segment; it is not chosen using rhythm labels.', '',
'Next: replicate with measured detector predictions and actual excluded-time masks, including quality and beat-extraction failures, before comparing representation models. No model sensitivity is assumed or hard-coded.']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('paired comparisons',len(paired),'quantization max',quantization_max)
