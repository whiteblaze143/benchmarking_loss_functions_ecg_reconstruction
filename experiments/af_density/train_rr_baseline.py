import csv,hashlib,io,json,time,zipfile
from collections import defaultdict
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from rr_protocol import P,O
from metrics import discrete_pattern_from_runs

def runs(x):
    x=np.asarray(x,dtype=bool)
    return list(zip(np.flatnonzero(x&~np.r_[False,x[:-1]]).tolist(),(np.flatnonzero(x&~np.r_[x[1:],False])+1).tolist()))

def main():
    assert not (O/'model.joblib').exists(),'Refusing to replace fitted checkpoint'
    protocol=json.loads((O/'protocol.json').read_text());receipt=json.loads((O/'preparation_receipt.json').read_text())
    assert hashlib.sha256((O/'patient_splits.csv').read_bytes()).hexdigest()==protocol['split_manifest_sha256']
    assert hashlib.sha256((O/'windows.npz').read_bytes()).hexdigest()==receipt['data_sha256']
    patients=list(csv.DictReader((O/'patient_splits.csv').open()));split=np.array([r['split'] for r in patients])
    a=np.load(O/'windows.npz');pi=a['patient_index'];valid=a['valid'];x=a['X'];y=a['y'];positive=a['assigned_positive'].astype(float);covered=a['assigned_count'].astype(float);negative=covered-positive
    masks={s:split[pi]==s for s in ['train','validation','test']}
    train=masks['train']&valid
    counts=np.bincount(pi[train],minlength=len(patients));weights=1/counts[pi[train]]
    for cls in [0,1]:
        m=y[train]==cls;weights[m]*=.5/weights[m].sum()
    # Normalize mean weight to one so l2_regularization has its stated scale.
    weights*=len(weights)/weights.sum()
    started=time.time();model=HistGradientBoostingClassifier(**protocol['parameters']).fit(x[train],y[train],sample_weight=weights)
    train_seconds=time.time()-started;joblib.dump(model,O/'model.joblib')
    scores=np.full(len(y),np.nan)
    for s in ['train','validation']:
        m=masks[s]&valid;scores[m]=model.predict_proba(x[m])[:,1]
    def rates(mask,threshold):
        pred=scores>=threshold;finite=np.isfinite(scores);ids=pi[mask]
        pos=np.bincount(ids,weights=positive[mask],minlength=len(patients));neg=np.bincount(ids,weights=negative[mask],minlength=len(patients))
        tp=np.bincount(ids,weights=(positive*pred*finite)[mask],minlength=len(patients));tn=np.bincount(ids,weights=(negative*(~pred)*finite)[mask],minlength=len(patients))
        sens=float((tp[pos>0]/pos[pos>0]).mean());spec=float((tn[neg>0]/neg[neg>0]).mean())
        return sens,spec
    candidates=[]
    for t in np.arange(1,100)/100:
        sens,spec=rates(masks['validation'],t)
        candidates.append({'threshold':float(t),'sensitivity_patient_mean':sens,'specificity_patient_mean':spec,'balanced_accuracy':(sens+spec)/2})
    selected=max(candidates,key=lambda r:(r['balanced_accuracy'],-abs(r['threshold']-.5),r['threshold']));threshold=selected['threshold']
    with (O/'validation_thresholds.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(candidates[0]));w.writeheader();w.writerows(candidates)
    source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path('experiments/af_density').glob('*.py'))}
    frozen={'threshold':selected,'checkpoint_sha256':hashlib.sha256((O/'model.joblib').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((O/'protocol.json').read_bytes()).hexdigest(),'sources':source_hashes,'test_predictions_not_yet_generated':True}
    (O/'frozen_before_test.json').write_text(json.dumps(frozen,indent=2)+'\n');print('MODEL_AND_THRESHOLD_FROZEN',selected,flush=True)
    # First held-out model predictions are generated only after the source and
    # threshold receipt above is committed to disk.
    test=masks['test']&valid;scores[test]=model.predict_proba(x[test])[:,1]
    np.save(O/'all_window_scores.npy',scores,allow_pickle=False)
    summary={'threshold':threshold,'train_seconds':train_seconds,'iterations':int(model.n_iter_),'train_declared_windows':int(masks['train'].sum()),'train_fitted_windows':int(train.sum()),'partitions':{}}
    for s,mask in masks.items():
        m=mask&valid;sens,spec=rates(mask,threshold)
        summary['partitions'][s]={'patients':int(sum(split==s)),'declared_windows':int(mask.sum()),'usable_windows':int(m.sum()),'indeterminate_windows':int((mask&~valid).sum()),'beat_coverage':float(covered[m].sum()/covered[mask].sum()),'sensitivity_patient_mean_worst_case_indeterminate':sens,'specificity_patient_mean_worst_case_indeterminate':spec,'window_auroc':float(roc_auc_score(y[m],scores[m])),'window_auprc':float(average_precision_score(y[m],scores[m])),'window_brier':float(brier_score_loss(y[m],scores[m]))}
    # Reconstruct each native-RR daily piece with all assigned intervals retained.
    files=list(csv.DictReader((O/'file_manifest.csv').open()));file_rows=[]
    with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as z:
        tables={}
        for m in files:
            rid=m['record_id'];ri=int(m['record_index']);day=int(m['day']);excluded=int(m['excluded_calibration_proxy_rr']);length=int(m['source_rr_count'])
            if rid not in tables:
                base=f'iridia-af-records-v1.0.1/{rid}/{rid}'
                tables[rid]=list(csv.DictReader(io.StringIO(z.read(base+'_rr_labels.csv').decode())))
            truth=np.zeros(length,dtype=np.int8)
            for e in tables[rid]:
                sf=int(e['start_file_index']);ef=int(e['end_file_index'])
                if sf<=day<=ef:truth[int(e['start_rr_index']) if sf==day else 0:int(e['end_rr_index']) if ef==day else length]=1
            truth=truth[excluded:];pred=np.full(len(truth),-1,dtype=np.int8)
            indices=np.flatnonzero((a['record_index']==ri)&(a['day']==day))
            for i in indices:
                lo=int(a['assigned_start'][i])-excluded;hi=int(a['assigned_end'][i])-excluded
                pred[lo:hi]=int(scores[i]>=threshold) if np.isfinite(scores[i]) else -1
            ref_runs=runs(truth);ref=discrete_pattern_from_runs(ref_runs,len(truth));indeterminate=int((pred<0).sum())
            pr=discrete_pattern_from_runs(runs(pred==1),len(pred)) if indeterminate==0 else {'density':None,'aggregation':None}
            file_rows.append({'patient_id':m['patient_id'],'split':patients[int(pi[indices[0]])]['split'],'record_id':rid,'day':day,'declared_rr':len(truth),'indeterminate_rr':indeterminate,'reference_episodes':len(ref_runs),'predicted_episodes':len(runs(pred==1)) if indeterminate==0 else None,'reference_density':ref['density'],'predicted_density':pr['density'],'density_absolute_error':None if ref['density'] is None or pr['density'] is None else abs(ref['density']-pr['density']),'reference_aggregation':ref['aggregation'],'predicted_aggregation':pr['aggregation'],'aggregation_absolute_error':None if ref['aggregation'] is None or pr['aggregation'] is None else abs(ref['aggregation']-pr['aggregation']),'burden_absolute_error':abs(float(truth.mean())-float(pred.mean())) if indeterminate==0 else None})
    with (O/'file_pattern_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(file_rows[0]));w.writeheader();w.writerows(file_rows)
    pattern=[];rng=np.random.default_rng(42)
    for s in masks:
        rs=[r for r in file_rows if r['split']==s]
        for metric in ['density_absolute_error','aggregation_absolute_error','burden_absolute_error']:
            by=defaultdict(list)
            for r in rs:
                if r[metric] is not None:by[r['patient_id']].append(r[metric])
            values=np.array([np.mean(v) for _,v in sorted(by.items())]);bootstrap=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
            pattern.append({'split':s,'metric':metric,'declared_files':len(rs),'defined_files':sum(r[metric] is not None for r in rs),'patient_ids_with_defined_metric':len(values),'patient_equal_weight_mean':float(values.mean()),'patient_bootstrap_95_low':float(np.quantile(bootstrap,.025)),'patient_bootstrap_95_high':float(np.quantile(bootstrap,.975))})
    with (O/'patient_pattern_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(pattern[0]));w.writeheader();w.writerows(pattern)
    summary['total_seconds']=time.time()-started;summary['complete']=True
    (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
