"""Check flatness in the actual nested prefixes, not merely the complete record."""
import argparse,hashlib,io,json,shutil,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.io import loadmat
from protocol import LEADS,CONFIGS

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    if (out/'prediction_manifest.parquet').exists():raise RuntimeError('Cannot change analysis population after held-out inference')
    if (out/'aborted_before_test_crop_qc').exists():raise RuntimeError('Crop amendment already applied; preserve its audit history')
    source=Path('/data/mithunmanivannan/datasets/figshare_31229326_v3')
    f=pd.read_parquet(out/'split_manifest.parquet');before=f.copy();pack=np.load(out/'grouping_fingerprints.npz')
    features=pack['features'].reshape(-1,12,50);owners=pack['record_indices'];starts=pack['start_samples']
    zero=(features==0).all(axis=-1);candidates={}
    # Flat raw prefixes necessarily have zero normalized PAA in every constituent
    # block. All candidates are confirmed on original raw values, avoiding any
    # inference from a lossy fingerprint or from changing constant block offsets.
    for seconds in [10,20,30]:
        possible=np.zeros((len(f),12),dtype=bool)
        for ri in np.unique(owners):
            ix=np.flatnonzero((owners==ri)&(starts<seconds*500))
            if f.at[int(ri),'duration_seconds']>=seconds:possible[ri]=zero[ix].all(axis=0)
        candidates[seconds]=possible
    flagged=np.flatnonzero(np.logical_or.reduce([v.any(axis=1) for v in candidates.values()]))
    flat={s:np.zeros((len(f),12),dtype=bool) for s in [10,20,30]}
    confirmation=[]
    for cohort in ['critical','normal']:
        archive=source/('Normal ECG signal.zip' if cohort=='normal' else 'Database # A publicly available 12-lead electrocardiogram criti-cal value dataset covering 13126 records.zip')
        with zipfile.ZipFile(archive) as z:
            for ri in flagged:
                row=f.iloc[ri]
                if row.cohort!=cohort:continue
                d=loadmat(io.BytesIO(z.read(row.mat_archive_member)));x=np.stack([d[k].reshape(-1) for k in LEADS])
                for seconds in [10,20,30]:
                    if row.duration_seconds>=seconds:
                        flat[seconds][ri]=np.ptp(x[:,:seconds*500],axis=1)==0
                        if flat[seconds][ri].any():confirmation.append({'record_uid':row.record_uid,'duration_seconds':seconds,'flat_leads':json.dumps([k for k,b in zip(LEADS,flat[seconds][ri]) if b])})
    pd.DataFrame(confirmation).to_csv(out/'crop_flatness_confirmed.csv',index=False)
    f['primary_eligible_before_crop_audit']=f.primary_eligible
    f['primary_eligible']=(f.duration_seconds>=30)&f.finite&~flat[10].any(axis=1)
    assert not ((f.cache_index<0)&f.primary_eligible).any()
    for s in [10,20,30]:f[f'prefix_{s}s_flat_leads']=[json.dumps([k for k,b in zip(LEADS,row) if b]) for row in flat[s]]
    counts=[];elig=[]
    for config,leads in CONFIGS.items():
        ix=[LEADS.index(k) for k in leads]
        for seconds in [10,20,30]:
            bad=flat[seconds][:,ix].any(axis=1);ok=(f.duration_seconds>=seconds)&f.finite&~bad
            for stratum,g in f.groupby('negative_stratum'):counts.append({'lead_configuration':config,'duration_seconds':seconds,'stratum':stratum,'records_before_technical':int((g.duration_seconds>=seconds).sum()),'eligible_records':int(ok.loc[g.index].sum())})
            elig.extend({'record_uid':row.record_uid,'lead_configuration':config,'duration_seconds':seconds,'eligible':bool(ok[i]),'exclusion_reason':'too_short' if row.duration_seconds<seconds else 'selected_prefix_lead_flat' if bad[i] else ''} for i,row in f.iterrows())
    # Preserve the old lock, trained development-only checkpoint, log and manifest.
    archive=out/'aborted_before_test_crop_qc';archive.mkdir(exist_ok=True)
    for name in ['clinical_cohort_contract.json','split_manifest.parquet','endpoint_counts.csv','configuration_eligibility.parquet','pretraining_lock.json']:
        shutil.copy2(out/name,archive/name)
    for name in ['M0_best.pt','baseline.log']:
        if (out/name).exists():shutil.move(out/name,archive/name)
    f.to_parquet(out/'split_manifest.parquet',index=False);pd.DataFrame(counts).to_csv(out/'endpoint_counts.csv',index=False);pd.DataFrame(elig).to_parquet(out/'configuration_eligibility.parquet',index=False)
    f[f.primary_eligible].groupby(['partition','negative_stratum']).size().rename('records').reset_index().to_csv(out/'primary_split_counts.csv',index=False)
    c=json.loads((out/'clinical_cohort_contract.json').read_text());c['primary_population']='>=30 seconds, finite, all 12 observed leads nonflat in actual first10s; hence common technical support for nested first10/20/30s'
    c['primary_n']=int(f.primary_eligible.sum());c['primary_positives']=int(f.loc[f.primary_eligible,'label'].sum());c['crop_QC_amendment']={'timing':'before any held-out prediction ledger existed','source_candidates_confirmed_on_raw_MAT':len(flagged),'removed_from_provisional_primary':int((before.primary_eligible&~f.primary_eligible).sum()),'partitions_reassigned':False,'aborted_run':'interrupted at zero-based epoch26, no held-out inference; not used for model selection in replacement run','policy':'same fixed architecture/optimizer/seed/budget; restart from fresh weights, not provisional checkpoint'}
    (out/'clinical_cohort_contract.json').write_text(json.dumps(c,indent=2));(out/'pretraining_lock.json').write_text(json.dumps({'contract_sha256':hashlib.file_digest((out/'clinical_cohort_contract.json').open('rb'),'sha256').hexdigest(),'split_sha256':hashlib.file_digest((out/'split_manifest.parquet').open('rb'),'sha256').hexdigest(),'no_test_predictions_present':True},indent=2))
    print(json.dumps({'primary_n':c['primary_n'],'primary_positives':c['primary_positives'],'removed':c['crop_QC_amendment']['removed_from_provisional_primary'],'confirmed_candidates':len(flagged)},indent=2),flush=True)
if __name__=='__main__':main()
