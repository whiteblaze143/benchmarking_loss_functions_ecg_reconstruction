"""Controlled reference-pattern stress test; no detector or clinical validation."""
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from metrics import discrete_pattern_from_runs,charitos_density
from gaps import apply_gap_strategy
P=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')
D=P/'controlled_pattern_stress';D.mkdir(exist_ok=True)

def runs(x):
    return list(zip(np.flatnonzero(x & ~np.r_[False,x[:-1]]).tolist(),(np.flatnonzero(x & ~np.r_[x[1:],False])+1).tolist()))

def evaluate(x,y):
    a=runs(x.astype(bool));b=runs(y.astype(bool));n=len(x)
    ref=discrete_pattern_from_runs(a,n);pred=discrete_pattern_from_runs(b,n)
    result={'reference_density':ref['density'],'predicted_density':pred['density'],'reference_aggregation':ref['aggregation'],'predicted_aggregation':pred['aggregation'],'density_absolute_error':None if ref['density'] is None or pred['density'] is None else abs(ref['density']-pred['density']),'aggregation_absolute_error':None if ref['aggregation'] is None or pred['aggregation'] is None else abs(ref['aggregation']-pred['aggregation']),'reference_episodes':len(a),'predicted_episodes':len(b),'burden_absolute_error':abs(float(x.mean())-float(y.mean())),'time_sensitivity':float(y[x==1].mean()) if x.any() else None,'time_specificity':float(1-y[x==0].mean()) if (x==0).any() else None,'split_source_episodes':sum(sum(c<e and d>s for c,d in b)>1 for s,e in a),'merged_prediction_episodes':sum(sum(s<d and e>c for s,e in a)>1 for c,d in b),'false_prediction_episodes_no_reference_overlap':sum(not any(s<d and e>c for s,e in a) for c,d in b)}
    return result

events=defaultdict(list)
for r in csv.DictReader((P/'timeline_intervals.csv').open()):events[r['record_id']].append((int(r['start_sample']),int(r['end_sample_exclusive'])))
segments=[r for r in csv.DictReader((P/'contiguous_segment_feasibility.csv').open()) if int(r['af_intervals'])>1]
rows=[];quantization=[]
for s in segments:
    start=int(s['start_sample']);end=int(s['end_sample_exclusive']);n=(end-start)//200
    x=np.zeros(n,dtype=np.int8)
    for a,b in events[s['record_id']]:
        lo=max(0,math.ceil((a-start-100)/200));hi=min(n,math.ceil((b-start-100)/200))
        if lo<hi:x[lo:hi]=1
    af=runs(x.astype(bool));off=runs((1-x).astype(bool))
    quantization.append({'patient_id':s['patient_id'],'record_id':s['record_id'],'daily_file_index':s['daily_file_index'],'source_intervals_in_segment':int(s['af_intervals']),'second_grid_intervals':len(af),'discarded_trailing_fraction_seconds':(end-start)/200-n,'source_continuous_density':float(s['density_contiguous_segment']) if s['density_contiguous_segment'] else None,'second_grid_continuous_density':charitos_density(af,n)})
    variants=[]
    for name,index in [('miss_shortest',min(range(len(af)),key=lambda i:af[i][1]-af[i][0])),('miss_longest',max(range(len(af)),key=lambda i:af[i][1]-af[i][0]))]:
        y=x.copy();a,b=af[index];y[a:b]=0;variants.append((name,y,'original_seconds','ok',b-a))
    index=max(range(len(af)),key=lambda i:af[i][1]-af[i][0]);a,b=af[index]
    if b-a>=3:
        length=min(30,(b-a)//3);lo=(a+b-length)//2;y=x.copy();y[lo:lo+length]=0
        variants.append(('split_longest_up_to_30s',y,'original_seconds','ok',length))
    else:variants.append(('split_longest_up_to_30s',None,'original_seconds','not_applicable_episode_too_short',0))
    if len(af)>1:
        i=min(range(len(af)-1),key=lambda i:af[i+1][0]-af[i][1]);a=af[i][1];b=af[i+1][0];y=x.copy();y[a:b]=1
        variants.append(('merge_closest',y,'original_seconds','ok',b-a))
    else:variants.append(('merge_closest',None,'original_seconds','not_applicable_fewer_than_two_grid_episodes',0))
    a,b=max(off,key=lambda ab:ab[1]-ab[0]) if off else (0,0)
    if b-a>=30:
        lo=(a+b-30)//2;y=x.copy();y[lo:lo+30]=1;variants.append(('false_episode_30s',y,'original_seconds','ok',30))
    else:variants.append(('false_episode_30s',None,'original_seconds','not_applicable_nonaf_gap_too_short',0))
    # A fixed mid-observation shutdown avoids choosing a location based on labels.
    missing=np.zeros(n,dtype=bool);lo=n//2-30;missing[lo:lo+60]=True
    for strategy in ['observed_only','nonaf','context']:
        out=apply_gap_strategy(x,x,missing,strategy)
        variants.append(('midpoint_shutdown_60s_'+strategy,out['predicted'],out['clock'],out['status'],60))
    for name,y,clock,status,changed in variants:
        r={'patient_id':s['patient_id'],'record_id':s['record_id'],'daily_file_index':s['daily_file_index'],'scenario':name,'status':status,'clock':clock,'altered_or_excluded_seconds':changed,'source_grid_seconds':n}
        if y is not None:
            original_metrics=discrete_pattern_from_runs(af,n)
            reference=x[~missing] if name=='midpoint_shutdown_60s_observed_only' else x
            r.update(evaluate(reference,y))
            r['reference_target_density_change_from_original_clock']=None if r['reference_density'] is None else abs(r['reference_density']-original_metrics['density'])
            r['reference_target_aggregation_change_from_original_clock']=None if r['reference_aggregation'] is None else abs(r['reference_aggregation']-original_metrics['aggregation'])
        rows.append(r)
fields=list(dict.fromkeys(k for r in rows for k in r))
with (D/'case_results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
with (D/'second_grid_quantization.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(quantization[0]));w.writeheader();w.writerows(quantization)
rng=np.random.default_rng(20261120);summary=[]
for scenario in dict.fromkeys(r['scenario'] for r in rows):
    rs=[r for r in rows if r['scenario']==scenario]
    for metric in ['density_absolute_error','aggregation_absolute_error','burden_absolute_error','time_sensitivity','time_specificity']:
        by_patient=defaultdict(list)
        for r in rs:
            if r.get(metric) is not None:by_patient[r['patient_id']].append(r[metric])
        values=np.array([np.mean(v) for _,v in sorted(by_patient.items())])
        if len(values):
            boots=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
            summary.append({'scenario':scenario,'metric':metric,'declared_segments':len(rs),'successful_scenarios':sum(r['status']=='ok' for r in rs),'defined_metric_segments':sum(r.get(metric) is not None for r in rs),'undefined_metric_successful_segments':sum(r['status']=='ok' and r.get(metric) is None for r in rs),'unavailable_scenarios':sum(r['status']!='ok' for r in rs),'patient_ids_with_defined_metric':len(values),'patient_equal_weight_mean':float(values.mean()),'patient_bootstrap_95_low':float(np.quantile(boots,.025)),'patient_bootstrap_95_high':float(np.quantile(boots,.975))})
with (D/'patient_weighted_summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
receipt={'complete':True,'declared_segments':len(segments),'patient_ids':len({s['patient_id'] for s in segments}),'case_scenarios':len(rows),'seed':20261120,'bootstrap_draws':2000,'clock':'1-second bin centers, with trailing fractional second dropped; density and aggregation use the published discrete equations on time bins, not RR beats.','limitations':['Controlled annotation perturbations, not measured model predictions or replication of the original detector/SQI study.','Reference morphology/quality not clinically adjudicated.','Observed-only strategy changes both reference and prediction clocks and targets; zero error does not mean the full original timeline is recovered.','Undefined predicted metrics remain in declared denominators and are counted separately, not assigned zero error.']}
(D/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
