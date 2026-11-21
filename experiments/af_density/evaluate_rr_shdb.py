"""External source-QRS RR evaluation, using frozen IRIDIA model and threshold."""
import csv,hashlib,json,time
from collections import defaultdict
from pathlib import Path
import joblib,numpy as np,wfdb
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from rr_features import features,layout
from rr_protocol import P,O
from train_rr_baseline import runs
from metrics import discrete_pattern_from_runs
S=Path('/data/mithunmanivannan/datasets/shdb_af');E=O/'external_shdb_v2';E.mkdir(exist_ok=True)
assert not (E/'frozen_before_external.json').exists(),'External protocol already frozen'
frozen=json.loads((O/'frozen_before_test.json').read_text());threshold=frozen['threshold']['threshold']
assert hashlib.sha256((O/'model.joblib').read_bytes()).hexdigest()==frozen['checkpoint_sha256']
assert hashlib.sha256(Path('experiments/af_density/rr_features.py').read_bytes()).hexdigest()==frozen['sources']['experiments/af_density/rr_features.py']
expected={}
for line in (P/'source_docs/shdb_1.0.1_SHA256SUMS.txt').read_text().splitlines():
    digest,name=line.split(maxsplit=1);expected[name.lstrip('*')]=digest
meta=[r for r in csv.DictReader((S/'AdditionalData.csv').open()) if r['Annotated']=='True']
checks=[]
for name in ['AdditionalData.csv']+[r['Data_ID']+'.'+ext for r in meta for ext in ['qrs','atr','hea']]:
    digest=hashlib.sha256((S/name).read_bytes()).hexdigest();assert digest==expected[name],name;checks.append({'file':name,'sha256':digest})
(E/'frozen_before_external.json').write_text(json.dumps({'threshold':threshold,'checkpoint_sha256':frozen['checkpoint_sha256'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':checks,'protocol':'Adjacent exactly duplicated QRS timestamps are one physical event and reduced to one entry, with counts retained per record. Decreasing or out-of-range timestamps remain hard failures. No fit or retuning. Source automatic QRS RR differences in milliseconds; only RR intervals starting at or after first explicit reference rhythm marker; 60 actual RR features; overlapping tail input; RR endpoint determines source rhythm.','limitations':['Native QRS channel provenance does not establish observed-channel matching.','Beat-clock density and aggregation only; no elapsed-time or waveform-quality validation.']},indent=2)+'\n')
source_intervals=defaultdict(list)
for r in csv.DictReader((P/'shdb_reference_intervals.csv').open()):source_intervals[r['record_id']].append((int(r['start_sample']),int(r['end_sample_exclusive']),r['rhythm']))
model=joblib.load(O/'model.joblib');records=[];parts=[];subgroups=[];start=time.time()
for index,m in enumerate(meta):
    uid=m['Data_ID'];h=wfdb.rdheader(str(S/uid));q=wfdb.rdann(str(S/uid),'qrs').sample
    assert np.all(np.diff(q)>=0) and q[0]>=0 and q[-1]<h.sig_len,uid
    duplicate_events=int((np.diff(q)==0).sum())
    q=q[np.r_[True,np.diff(q)>0]]
    first=source_intervals[uid][0][0];keep=q[:-1]>=first
    rr=np.diff(q)[keep].astype(float)*1000/h.fs;endpoints=q[1:][keep]
    truth=np.zeros(len(rr),dtype=np.int8);strata=np.full(len(rr),'unknown',dtype='<U8')
    for lo,hi,rhythm in source_intervals[uid]:
        select=(endpoints>=lo)&(endpoints<hi);strata[select]=rhythm
        if rhythm=='(AFIB':truth[select]=1
    assert not (strata=='unknown').any()
    starts,assigned,ends=layout(len(rr),0);x,valid=features(rr[starts[:,None]+np.arange(60)])
    score=np.full(len(starts),np.nan);score[valid]=model.predict_proba(x[valid])[:,1]
    pred=np.full(len(rr),-1,dtype=np.int8)
    for i,(lo,hi) in enumerate(zip(assigned,ends)):pred[lo:hi]=int(score[i]>=threshold) if valid[i] else -1
    prefix=np.r_[0,np.cumsum(truth)];target=((prefix[starts+60]-prefix[starts])>=30).astype(np.int8)
    pos=int(truth.sum());neg=len(rr)-pos;tp=int(((pred==1)&(truth==1)).sum());tn=int(((pred==0)&(truth==0)).sum())
    ref=discrete_pattern_from_runs(runs(truth),len(rr));pr=discrete_pattern_from_runs(runs(pred==1),len(rr)) if (pred>=0).all() else {'density':None,'aggregation':None}
    records.append({'record_id':uid,'subject_id':m['Subject_ID'],'qrs_exact_duplicate_events_removed':duplicate_events,'declared_rr':len(rr),'af_rr':pos,'nonaf_rr':neg,'true_positive_rr':tp,'true_negative_rr':tn,'indeterminate_rr':int((pred<0).sum()),'windows':len(starts),'source_af_intervals':sum(rhythm=='(AFIB' for _,_,rhythm in source_intervals[uid]),'rr_grid_af_intervals':len(runs(truth)),'reference_density':ref['density'],'predicted_density':pr['density'],'density_absolute_error':None if ref['density'] is None or pr['density'] is None else abs(ref['density']-pr['density']),'reference_aggregation':ref['aggregation'],'predicted_aggregation':pr['aggregation'],'aggregation_absolute_error':None if ref['aggregation'] is None or pr['aggregation'] is None else abs(ref['aggregation']-pr['aggregation']),'burden_absolute_error':abs(float(truth.mean())-float(pred.mean())) if (pred>=0).all() else None})
    for rhythm in np.unique(strata):
        mask=strata==rhythm
        subgroups.append({'record_id':uid,'subject_id':m['Subject_ID'],'rhythm':str(rhythm),'reference_rr':int(mask.sum()),'positive_calls_rr':int(((pred==1)&mask).sum()),'indeterminate_rr':int(((pred<0)&mask).sum())})
    parts.append({'X':x.astype(np.float32),'scores':score,'target':target,'valid':valid,'record_index':np.full(len(starts),index,dtype=np.int16),'input_start':starts.astype(np.int32),'assigned_start':assigned.astype(np.int32),'assigned_end':ends.astype(np.int32)})
    print(uid,'complete',index+1,'of',len(meta),flush=True)
for name,rs in [('record_metrics.csv',records),('rhythm_subgroup_calls.csv',subgroups)]:
    with (E/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
a={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};np.savez_compressed(E/'window_predictions.npz',**a)
patient=defaultdict(list)
for r in records:patient[r['subject_id']].append(r)
sens=[];spec=[]
for rs in patient.values():
    positive=sum(r['af_rr'] for r in rs);negative=sum(r['nonaf_rr'] for r in rs)
    if positive:sens.append(sum(r['true_positive_rr'] for r in rs)/positive)
    if negative:spec.append(sum(r['true_negative_rr'] for r in rs)/negative)
valid=a['valid'];summary={'complete':True,'records':len(records),'source_subject_ids':len(patient),'windows':len(a['scores']),'threshold_unchanged':threshold,'indeterminate_rr':sum(r['indeterminate_rr'] for r in records),'sensitivity_patient_mean':float(np.mean(sens)),'patients_contributing_sensitivity':len(sens),'specificity_patient_mean':float(np.mean(spec)),'patients_contributing_specificity':len(spec),'window_auroc':float(roc_auc_score(a['target'][valid],a['scores'][valid])),'window_auprc':float(average_precision_score(a['target'][valid],a['scores'][valid])),'window_brier':float(brier_score_loss(a['target'][valid],a['scores'][valid])),'elapsed_seconds':time.time()-start,'pattern_metrics':{}}
rng=np.random.default_rng(42)
for metric in ['density_absolute_error','aggregation_absolute_error','burden_absolute_error']:
    values=np.array([np.mean([r[metric] for r in rs if r[metric] is not None]) for rs in patient.values() if any(r[metric] is not None for r in rs)])
    bootstrap=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
    summary['pattern_metrics'][metric]={'defined_records':sum(r[metric] is not None for r in records),'declared_records':len(records),'subject_ids_with_defined_metric':len(values),'patient_equal_weight_mean':float(values.mean()),'patient_bootstrap_95_low':float(np.quantile(bootstrap,.025)),'patient_bootstrap_95_high':float(np.quantile(bootstrap,.975))}
(E/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
