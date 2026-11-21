"""Audit frozen provenance, source/cache identity and checkpoint prediction replay."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.io import loadmat
from baseline import ResNet
from protocol import LEADS,locked_threshold,metrics

def digest(path):
    with Path(path).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.output;torch.set_num_threads(2)
    lock=json.loads((r/'pretraining_lock.json').read_text());receipt=json.loads((r/'baseline_receipt.json').read_text());checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    check('contract unchanged after training',digest(r/'clinical_cohort_contract.json')==lock['contract_sha256'])
    check('split unchanged after training',digest(r/'split_manifest.parquet')==lock['split_sha256'])
    check('checkpoint digest',digest(r/'M0_best.pt')==receipt['checkpoint_sha256'])
    check('executed baseline source unchanged',digest(Path(__file__).with_name('baseline.py'))==receipt['source_sha256'])
    cp=torch.load(r/'M0_best.pt',map_location='cpu',weights_only=True);check('checkpoint froze correct contract',cp['contract_sha256']==lock['contract_sha256'])
    f=pd.read_parquet(r/'split_manifest.parquet');pframe=pd.read_parquet(r/'prediction_manifest.parquet');primary=f[f.primary_eligible].sort_values('cache_index')
    check('full source preserved',len(f)==20126 and f.record_uid.nunique()==20126 and f.label.sum()==1481)
    check('primary source counts',len(primary)==9426 and primary.label.sum()==967)
    check('no cross-partition record groups',f.groupby('record_group').partition.nunique().max()==1)
    check('actual prefixes have no flat selected lead',primary.prefix_10s_flat_leads.eq('[]').all())
    check('all final validation/test units retained',len(pframe)==len(primary[primary.partition!='train']) and pframe.record_uid.nunique()==len(pframe))
    joined=pframe.merge(primary[['record_uid','label','partition']],on='record_uid',suffixes=('_pred','_source'),validate='one_to_one')
    check('labels and partitions agree with source',joined.label_pred.eq(joined.label_source).all() and joined.partition_pred.eq(joined.partition_source).all())
    check('checkpoint/preprocessing hashes in every prediction',pframe.checkpoint_sha256.eq(receipt['checkpoint_sha256']).all() and pframe.preprocessing_sha256.eq(digest(r/'preprocessing.json')).all())
    hist=pd.read_csv(r/'training_history.csv');check('best checkpoint selection',int(hist.loc[hist.validation_auroc.idxmax(),'epoch'])==receipt['best_epoch']==cp['best_epoch'])
    prep=json.loads((r/'preprocessing.json').read_text());train_ids='\n'.join(primary[primary.partition=='train'].record_uid)
    check('preprocessing fit IDs training-only',hashlib.sha256(train_ids.encode()).hexdigest()==prep['fit_record_ids_sha256'])
    val=pframe[pframe.partition=='validation'];threshold=locked_threshold(val.label.to_numpy(),val.score.to_numpy(dtype=np.float32))
    check('source-native precision validation threshold reproduced',threshold==receipt['threshold'] and val.threshold.eq(threshold).all())
    check('validation specificity target met',((val.loc[val.label==0,'score']<threshold).mean()>=.95))
    mm=np.load(r/'primary_30s_mV.npy',mmap_mode='r');source=Path('/data/mithunmanivannan/datasets/figshare_31229326_v3')
    samples=primary.groupby('partition',group_keys=False).head(2)
    for cohort,g in samples.groupby('cohort'):
        archive=source/('Normal ECG signal.zip' if cohort=='normal' else 'Database # A publicly available 12-lead electrocardiogram criti-cal value dataset covering 13126 records.zip')
        with zipfile.ZipFile(archive) as z:
            for row in g.itertuples():
                raw=z.read(row.mat_archive_member);check('original MAT hash '+row.record_uid,hashlib.sha256(raw).hexdigest()==row.mat_sha256)
                d=loadmat(io.BytesIO(raw));x=np.stack([d[k].reshape(-1) for k in LEADS]).astype('float32')[:,:15000]*.001
                check('physical mV cache identity '+row.record_uid,np.array_equal(x,mm[int(row.cache_index)]))
    model=ResNet().cuda();model.load_state_dict(cp['state_dict']);model.eval();mean=cp['mean'].cuda();std=cp['std'].cuda();replay=[]
    with torch.inference_mode():
        for partition in ['validation','test']:
            rows=primary[primary.partition==partition];scores=[]
            for start in range(0,len(rows),256):
                part=rows.iloc[start:start+256];x=torch.from_numpy(mm[part.cache_index.to_numpy(),:,:5000]).cuda();x=((x-mean)/std).to(torch.bfloat16)
                with torch.autocast('cuda',dtype=torch.bfloat16):score=torch.sigmoid(model(x).float()).cpu().numpy()
                scores.extend(score.tolist())
            stored=pframe.set_index('record_uid').loc[rows.record_uid];scores=np.asarray(scores)
            maxdiff=float(np.abs(stored.score.to_numpy()-scores).max());diffdec=int(np.count_nonzero((scores>=threshold)!=stored.decision.to_numpy()))
            check(partition+' inference replay tolerance and decisions',maxdiff<=.003 and diffdec==0)
            replay.append({'partition':partition,'records':len(rows),'maximum_absolute_score_difference':maxdiff,'different_decisions':diffdec})
    te=pframe[pframe.partition=='test'];point=metrics(te.label,te.score,te.decision_status,threshold);csv=pd.read_csv(r/'primary_results.csv').iloc[0]
    check('saved metrics reproduce from ledger',all(abs(point[k]-csv[k])<1e-12 for k in ['sensitivity_all_eligible','specificity_completed','auroc','auprc','brier']))
    files={str(p.relative_to(r)):digest(p) for p in r.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.md','.pdf','.parquet','.txt'] and p.name!='baseline_verification.json'}
    (r/'baseline_verification.json').write_text(json.dumps({'checks':checks,'passed':True,'replay':replay,'files':files},indent=2));print(json.dumps({'passed':True,'checks':len(checks),'replay':replay},indent=2),flush=True)
if __name__=='__main__':main()
