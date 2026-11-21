"""Freeze all source records, observed-lead eligibility, conservative groups and split."""
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold
from protocol import LEADS,CONFIGS,ENDPOINT,RHYTHM,unit_normalize_windows,candidates

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'split_manifest.parquet').exists():raise RuntimeError('Frozen split already exists; use a new output root')
    source=Path('/data/mithunmanivannan/datasets/figshare_31229326_v3');manifest=source/'analysis/record_manifest.parquet'
    f=pd.read_parquet(manifest).sort_values('record_uid').reset_index(drop=True)
    assert len(f)==20126
    labs=f.labels_json.map(json.loads);f['label']=labs.map(lambda x:int(ENDPOINT in x))
    f['negative_stratum']=[ 'positive_AVB' if y else 'normal' if c=='normal' else 'other_rhythm_critical' if set(l)&RHYTHM else 'other_critical' for y,c,l in zip(f.label,f.cohort,labs)]
    f['primary_eligible']=(f.duration_seconds>=30)&(f.flat_lead_count==0)&f.finite
    primary=f[f.primary_eligible].copy();f['cache_index']=-1;f.loc[primary.index,'cache_index']=np.arange(len(primary))
    # Cache is unnormalized physical mV. Every entry contains a real 30-second prefix.
    cache=np.lib.format.open_memmap(out/'primary_30s_mV.npy',mode='w+',dtype='float32',shape=(len(primary),12,15000))
    nwin=int((f.sample_count_per_lead//2500).sum());windows=np.lib.format.open_memmap(out/'grouping_windows.npy',mode='w+',dtype='float32',shape=(nwin,6000))
    small=np.empty((nwin,600),np.float32);owners=np.empty(nwin,np.int32);starts=np.empty(nwin,np.int32);at=0;start=time.time()
    for cohort,g in f.groupby('cohort',sort=False):
        archive=source/('Normal ECG signal.zip' if cohort=='normal' else 'Database # A publicly available 12-lead electrocardiogram criti-cal value dataset covering 13126 records.zip')
        with zipfile.ZipFile(archive) as z:
            for index,row in g.iterrows():
                data=loadmat(io.BytesIO(z.read(row.mat_archive_member)));x=np.stack([data[k].reshape(-1) for k in LEADS]).astype(np.float32)
                if row.primary_eligible:cache[int(row.cache_index)]=x[:,:15000]*.001
                n=x.shape[1]//2500
                # Five-sample boxcar average for grouping only, not model input.
                w=x[:,:n*2500].reshape(12,n,500,5).mean(axis=-1).transpose(1,0,2)
                norm=unit_normalize_windows(w);windows[at:at+n]=norm
                paa=norm.reshape(n,12,50,10).mean(axis=-1).reshape(n,600);den=np.linalg.norm(paa,axis=1,keepdims=True)
                small[at:at+n]=np.divide(paa,den,out=np.zeros_like(paa),where=den>0)
                owners[at:at+n]=index;starts[at:at+n]=np.arange(n)*2500;at+=n
                if (index+1)%2000==0:print('CACHE',index+1,'seconds',round(time.time()-start,1),flush=True)
    assert at==nwin;cache.flush();windows.flush()
    np.savez(out/'grouping_fingerprints.npz',features=small,record_indices=owners,start_samples=starts)
    pairs=candidates(small);print('LSH candidates',len(pairs),flush=True)
    parent=np.arange(len(f))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        i,j=find(i),find(j)
        if i!=j:parent[max(i,j)]=min(i,j)
    # Even conflicting source normal ID strings remain together conservatively.
    for _,g in f[f.cohort=='normal'].groupby('patient_id_from_explicit_merge_only'):
        for index in g.index[1:]:union(int(g.index[0]),int(index))
    edges=[];coarse_candidates=0
    for begin in range(0,len(pairs),4096):
        block=pairs[begin:begin+4096];sim=np.einsum('ij,ij->i',small[block[:,0]],small[block[:,1]])
        for (i,j),coarse in zip(block[sim>=.98],sim[sim>=.98]):
            if owners[i]==owners[j]:continue
            coarse_candidates+=1;score=float(np.dot(windows[i],windows[j]))
            if score>=.995:
                ri,rj=int(owners[i]),int(owners[j]);union(ri,rj)
                edges.append({'record_a':f.at[ri,'record_uid'],'record_b':f.at[rj,'record_uid'],'start_a':int(starts[i]),'start_b':int(starts[j]),'correlation':score})
    pd.DataFrame(edges,columns=['record_a','record_b','start_a','start_b','correlation']).to_csv(out/'near_duplicate_edges.csv',index=False)
    f['record_group']=[f'rg{find(i):05d}' for i in range(len(f))]
    f['fold']=-1
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    for fold,(_,idx) in enumerate(splitter.split(f,f.negative_stratum,f.record_group)):f.loc[idx,'fold']=fold
    f['partition']=f.fold.map({0:'test',1:'validation',2:'train',3:'train',4:'train'})
    assert f.groupby('record_group').partition.nunique().max()==1
    f.to_parquet(out/'split_manifest.parquet',index=False)
    counts=[];elig=[]
    for conf,leads in CONFIGS.items():
        flat=f.flat_leads.map(lambda x:bool(set(json.loads(x))&set(leads)))
        for seconds in [10,20,30]:
            ok=(f.duration_seconds>=seconds)&f.finite&~flat
            for stratum,g in f.groupby('negative_stratum'):
                counts.append({'lead_configuration':conf,'duration_seconds':seconds,'stratum':stratum,'records_before_technical':int((g.duration_seconds>=seconds).sum()),'eligible_records':int(ok.loc[g.index].sum())})
            for index,row in f.iterrows():
                elig.append({'record_uid':row.record_uid,'lead_configuration':conf,'duration_seconds':seconds,'eligible':bool(ok[index]),'exclusion_reason':'too_short' if row.duration_seconds<seconds else 'selected_lead_flat' if flat[index] else ''})
    pd.DataFrame(counts).to_csv(out/'endpoint_counts.csv',index=False);pd.DataFrame(elig).to_parquet(out/'configuration_eligibility.parquet',index=False)
    split=f[f.primary_eligible].groupby(['partition','negative_stratum']).size().rename('records').reset_index();split.to_csv(out/'primary_split_counts.csv',index=False)
    contract={'endpoint':ENDPOINT,'source_label_scope':'record-level source diagnosis; no verified crop-level event times','source_manifest_sha256':hashlib.file_digest(manifest.open('rb'),'sha256').hexdigest(),'all_records':len(f),'all_positives':int(f.label.sum()),'primary_population':'>=30 seconds, finite, all 12 observed leads nonflat; fixed common cohort for lead/duration comparisons','primary_n':int(f.primary_eligible.sum()),'primary_positives':int(f.loc[f.primary_eligible,'label'].sum()),'primary_input_first_run':{'model':'M0 raw ResNet','lead_configuration':'full12','duration_seconds':10,'crop':'first contiguous observed prefix, no padding/stitching/aggregation','units':'physical mV, source uV /1000','sampling_rate_hz':500},'secondary_populations':'configuration-specific observed-lead eligibility at >=10/20/30s, with primary common cohort for paired contrasts','specificity_target':.95,'threshold_selection':'development validation negatives only; ties conservative; never fit threshold to test','split':{'seed':42,'method':'StratifiedGroupKFold 5; fold0 test, fold1 validation, fold2-4 train','unit':'conservative record group, not patient','grouping':'union source normal ID strings and full 12-lead normalized 5-second waveform correlations >=0.995 after approximate retrieval','patient_independence_verified':False},'grouping_search':{'windows':nwin,'lsh_candidates':len(pairs),'coarse_candidates':coarse_candidates,'verified_edges':len(edges),'record_groups':f.record_group.nunique(),'coverage':'nonoverlapping 5s blocks across complete records; 100Hz grouping waveform, 50 samples/lead PAA, 128-bit SimHash with 8x16-bit bands, coarse cosine>=0.98 then full waveform cosine>=0.995','limitation':'approximate retrieval and fixed grid miss some offsets/near-duplicates; groups do not establish patient identities'},'comparison_policy':'same ECG-only inputs, groups, crop policy, train-only preprocessing; R peaks restricted to supplied channels; no synthetic intervals on segmentation failure','initial_training':{'epochs_max':40,'patience':8,'selection':'maximum validation AUROC; strict improvement checkpoint copy','seed':42,'batch_size':128,'optimizer':'AdamW lr0.001 weight_decay0.0001','loss':'unweighted BCEWithLogits; preserve observed training class balance'},'bootstrap':{'replicates':2000,'resampling_unit':'record_group','interpretation':'conditional uncertainty with residual patient overlap; paired resampling for future model contrasts'},'formal_primary_contrast':'M1 minus M0 sensitivity at validation-selected >=95% specificity for Lead II, 10s, on primary common cohort; full12 reference; all other contrasts exploratory initially','multiplicity':'one formal primary contrast; extensions require pre-test lock amendment, not post-test selection','status':'data framework and first M0 baseline only; no SetOperator superiority claim'}
    (out/'clinical_cohort_contract.json').write_text(json.dumps(contract,indent=2));print(json.dumps(contract,indent=2),flush=True)
if __name__=='__main__':main()
