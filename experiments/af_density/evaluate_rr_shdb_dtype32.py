"""Enforce the frozen training dtype; preserve the original external run.

Reuse cached float32 source features. Only records with changed class decisions
need their binary timeline rebuilt; unchanged vectors have identical metrics.
"""
import csv,hashlib,json,time
from collections import defaultdict
from pathlib import Path
import joblib,numpy as np,wfdb
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from rr_protocol import P,O
from train_rr_baseline import runs
from metrics import discrete_pattern_from_runs
V=O/'external_shdb_v2';E=O/'external_shdb_v3';E.mkdir(exist_ok=True)
assert not (E/'frozen_before_external.json').exists(),'Refusing to replace completed dtype protocol'
f=json.loads((V/'frozen_before_external.json').read_text());threshold=f['threshold']
assert hashlib.sha256((O/'model.joblib').read_bytes()).hexdigest()==f['checkpoint_sha256']
source=Path('experiments/af_density/evaluate_rr_shdb.py').read_text();assert hashlib.sha256(source.encode()).hexdigest()==f['source_sha256']
for r in f['inputs']:assert hashlib.sha256((Path('/data/mithunmanivannan/datasets/shdb_af')/r['file']).read_bytes()).hexdigest()==r['sha256']
with np.load(V/'window_predictions.npz') as z:a={k:z[k] for k in z.files}
assert a['X'].dtype==np.float32
with np.load(O/'windows.npz') as train_data:assert train_data['X'].dtype==a['X'].dtype
freeze={'threshold':threshold,'checkpoint_sha256':f['checkpoint_sha256'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'feature_dtype':'float32, identical to frozen training preparation','cached_source_archive_sha256':hashlib.sha256((V/'window_predictions.npz').read_bytes()).hexdigest(),'original_evaluation_source_sha256':f['source_sha256'],'inputs':f['inputs'],'change':'Correct float64 external inference mismatch; no retraining, threshold tuning, feature redesign or new reference labels.'}
(E/'frozen_before_external.json').write_text(json.dumps(freeze,indent=2)+'\n')
start=time.time();model=joblib.load(O/'model.joblib');valid=a['valid'];old=a['scores'].copy();a['scores'][valid]=model.predict_proba(a['X'][valid])[:,1]
changed=(old>=threshold)!=(a['scores']>=threshold);changed_records=set(a['record_index'][changed].tolist())
records=list(csv.DictReader((V/'record_metrics.csv').open()))
# Restore numeric fields so the exact original summary block can be reused.
float_fields={'reference_density','predicted_density','density_absolute_error','reference_aggregation','predicted_aggregation','aggregation_absolute_error','burden_absolute_error'}
for r in records:
    for key in list(r):
        if key in ['record_id','subject_id']:continue
        r[key]=None if r[key]=='' else float(r[key]) if key in float_fields else int(r[key])
intervals=defaultdict(list)
for r in csv.DictReader((P/'shdb_reference_intervals.csv').open()):intervals[r['record_id']].append((int(r['start_sample']),int(r['end_sample_exclusive']),r['rhythm']))
subgroups=list(csv.DictReader((V/'rhythm_subgroup_calls.csv').open()))
for index,r in enumerate(records):
    if index not in changed_records:
        m=a['record_index']==index;assert np.array_equal(old[m]>=threshold,a['scores'][m]>=threshold);continue
    uid=r['record_id'];base=Path('/data/mithunmanivannan/datasets/shdb_af')/uid;h=wfdb.rdheader(str(base));q=wfdb.rdann(str(base),'qrs').sample
    assert (np.diff(q)>=0).all() and q[-1]<h.sig_len;q=q[np.r_[True,np.diff(q)>0]]
    keep=q[:-1]>=intervals[uid][0][0];ends=q[1:][keep];truth=np.zeros(len(ends),dtype=np.int8);strata=np.full(len(ends),'unknown',dtype='<U8')
    for lo,hi,rhythm in intervals[uid]:
        m=(ends>=lo)&(ends<hi);strata[m]=rhythm
        if rhythm=='(AFIB':truth[m]=1
    assert not (strata=='unknown').any();pred=np.full(len(truth),-1,dtype=np.int8)
    for i in np.flatnonzero(a['record_index']==index):pred[a['assigned_start'][i]:a['assigned_end'][i]]=int(a['scores'][i]>=threshold) if valid[i] else -1
    assert (pred>=0).all();assert len(pred)==r['declared_rr']
    r['true_positive_rr']=int(((truth==1)&(pred==1)).sum());r['true_negative_rr']=int(((truth==0)&(pred==0)).sum())
    ref=discrete_pattern_from_runs(runs(truth),len(truth));pr=discrete_pattern_from_runs(runs(pred==1),len(pred))
    for name in ['density','aggregation']:
        r['predicted_'+name]=pr[name];r[name+'_absolute_error']=None if ref[name] is None or pr[name] is None else abs(ref[name]-pr[name])
    r['burden_absolute_error']=abs(float(truth.mean())-float(pred.mean()))
    for row in subgroups:
        if row['record_id']==uid:row['positive_calls_rr']=int(((pred==1)&(strata==row['rhythm'])).sum())
for name,rs in [('record_metrics.csv',records),('rhythm_subgroup_calls.csv',subgroups)]:
    with (E/name).open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
np.savez_compressed(E/'window_predictions.npz',**a)
exec(compile(source[source.index('patient=defaultdict(list)'):],'unchanged_external_summary_block','exec'))
change={'original_run_preserved':True,'scores_changed':int((old!=a['scores']).sum()),'max_absolute_score_change':float(np.max(abs(old-a['scores']))),'decisions_changed':int(changed.sum()),'records_reconstructed':[records[i]['record_id'] for i in sorted(changed_records)],'unchanged_record_binary_vectors_explicitly_verified':len(records)-len(changed_records)}
(E/'dtype_correction_audit.json').write_text(json.dumps(change,indent=2)+'\n');print(json.dumps(change,indent=2))
