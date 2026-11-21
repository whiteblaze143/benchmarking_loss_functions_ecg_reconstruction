"""Prediction-only evaluation, paired group resampling and metadata shortcut control."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib
from protocol import metrics,locked_threshold

KEYS=['sensitivity_all_eligible','specificity_completed','auroc','auprc','brier']

def paired_group_intervals(frame,score_b=None,replicates=2000):
    """Resample identical record groups for both scores; frozen thresholds/models."""
    if score_b is not None:
        for key in ['record_uid','record_group','label']:
            if not np.array_equal(frame[key].to_numpy(),score_b[key].to_numpy()):raise ValueError('Unaligned paired predictions: '+key)
    groups=[g.index.to_numpy() for _,g in frame.reset_index(drop=True).groupby('record_group')]
    frame=frame.reset_index(drop=True);rng=np.random.default_rng(42);values={k:[] for k in KEYS}
    for _ in range(replicates):
        index=np.concatenate([groups[j] for j in rng.integers(0,len(groups),size=len(groups))]);a=frame.iloc[index]
        m=metrics(a.label,a.score,a.decision_status,float(a.threshold.iloc[0]))
        if score_b is not None:
            b=score_b.iloc[index];other=metrics(b.label,b.score,b.decision_status,float(b.threshold.iloc[0]))
            m={k:m.get(k)-other.get(k) if m.get(k) is not None and other.get(k) is not None else None for k in KEYS}
        for k in KEYS:
            if m.get(k) is not None:values[k].append(m[k])
    return {k:{'low':float(np.quantile(v,.025)),'high':float(np.quantile(v,.975)),'valid_replicates':len(v),'first1000_low':float(np.quantile(v[:1000],.025)),'first1000_high':float(np.quantile(v[:1000],.975)),'max_endpoint_change_vs_first1000':float(max(abs(np.quantile(v,.025)-np.quantile(v[:1000],.025)),abs(np.quantile(v,.975)-np.quantile(v[:1000],.975))))} for k,v in values.items() if v}

def evaluate(out):
    frame=pd.read_parquet(out/'prediction_manifest.parquet');test=frame[frame.partition=='test'].reset_index(drop=True)
    point=metrics(test.label,test.score,test.decision_status,float(test.threshold.iloc[0]));ci=paired_group_intervals(test)
    pd.DataFrame([{'model':'M0_raw_ResNet18_1d','lead_configuration':'full12','duration_seconds':10,'analysis':'first_baseline_exploratory_reference','threshold':test.threshold.iloc[0],**point,**{f'{k}_ci_{bound}':v[bound] for k,v in ci.items() for bound in ['low','high']}}]).to_csv(out/'primary_results.csv',index=False)
    subgroup=[]
    for field in ['negative_stratum','sex','source_cohort','multilabel']:
        for name,g in test.groupby(field,dropna=False):subgroup.append({'variable':field,'group':name,**metrics(g.label,g.score,g.decision_status,float(test.threshold.iloc[0]))})
    age=pd.cut(test.age_years,[-np.inf,18,65,np.inf],right=False,labels=['under18','18to64','65plus'])
    for name,g in test.groupby(age,observed=True):subgroup.append({'variable':'age','group':name,**metrics(g.label,g.score,g.decision_status,float(test.threshold.iloc[0]))})
    pd.DataFrame(subgroup).to_csv(out/'subgroup_results.csv',index=False)
    cal=[]
    bins=np.linspace(0,1,11)
    for j in range(10):
        g=test[(test.score>=bins[j]) & ((test.score<bins[j+1]) if j<9 else (test.score<=1))]
        cal.append({'bin_low':bins[j],'bin_high':bins[j+1],'n':len(g),'mean_probability':g.score.mean(),'observed_fraction':g.label.mean()})
    pd.DataFrame(cal).to_csv(out/'calibration_results.csv',index=False)
    # Metadata-only control shares source records, partitions and validation rules.
    f=pd.read_parquet(out/'split_manifest.parquet');f=f[f.primary_eligible].copy()
    columns=['age_years_approx','duration_seconds','sex','cohort']
    prep=ColumnTransformer([('numeric',StandardScaler(),columns[:2]),('categorical',OneHotEncoder(handle_unknown='ignore'),columns[2:])])
    meta=make_pipeline(prep,LogisticRegression(max_iter=1000,random_state=42));tr=f[f.partition=='train'];va=f[f.partition=='validation'];te=f[f.partition=='test']
    meta.fit(tr[columns],tr.label);threshold=locked_threshold(va.label,meta.predict_proba(va[columns])[:,1]);prob=meta.predict_proba(te[columns])[:,1]
    joblib.dump(meta,out/'metadata_control.joblib')
    metaledger=te[['record_uid','record_group','label','negative_stratum']].copy();metaledger['score']=prob;metaledger['threshold']=threshold;metaledger['decision_status']='completed';metaledger.to_parquet(out/'metadata_predictions.parquet',index=False)
    m=metrics(te.label,prob,np.repeat('completed',len(te)),threshold);pd.DataFrame([{'model':'metadata_only','inputs':','.join(columns),'threshold':threshold,**m}]).to_csv(out/'metadata_results.csv',index=False)
    # Align before any paired comparison; never silently inner-join away failures.
    metaledger=metaledger.set_index('record_uid').loc[test.record_uid].reset_index();assert np.array_equal(test.label,metaledger.label)
    stats={'replicates':2000,'unit':'conservative record group; not verified patient','threshold_uncertainty':'conditional on fixed validation-selected threshold and trained checkpoint','M0_intervals':ci,'M1_vs_M0':{'status':'not_run','reason':'first baseline returned before SetOperator expansion'},'test_record_groups':test.record_group.nunique(),'bootstrap_stability':'first1000 and full2000 endpoints with absolute change saved per metric; no inference of independent patients','formal_p_values':None}
    (out/'paired_statistics.json').write_text(json.dumps(stats,indent=2))
    print('BASELINE_TEST',json.dumps(point),flush=True);print('METADATA_TEST',json.dumps(m),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);evaluate(p.parse_args().output)
