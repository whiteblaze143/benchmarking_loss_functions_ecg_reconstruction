"""Locked Lead-II comparison; imports authentic morphology and operator modules."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import roc_auc_score
from baseline import ResNet, predict
from protocol import locked_threshold,metrics
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'experiments/ptbxl_distributional_repecg/src'))
from repecg.common.beats import valid_rr_pairs
from repecg.common.phase import phase_normalize
from repecg.common.kernels import WhiteningTransform,NystromMap,biased_mmd2,audit_nystrom
from repecg.paper07_operator.model import OperatorSetModel
from repecg.paper07_operator.measurements import canonical_operators
from wfdb.processing import xqrs_detect
BASE=Path('/data/mithunmanivannan/codex_artifacts/zzu_avblock_20261120')
OUT=Path('/data/mithunmanivannan/codex_artifacts/zzu_leadii_20261120')
Q=canonical_operators()[1]

def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def cohort():
    f=pd.read_parquet(BASE/'split_manifest.parquet');f=f[f.primary_eligible].sort_values('cache_index').reset_index(drop=True)
    assert len(f)==9426 and f.label.sum()==967
    return f

def extract(signal):
    # Same XQRS and original RR limits, with only the observed physical Lead II.
    try:peaks=np.asarray(xqrs_detect(signal,fs=500,verbose=False),dtype=np.int64);error=''
    except (ValueError,IndexError,RuntimeError) as e:peaks=np.empty(0,dtype=np.int64);error=type(e).__name__+': '+str(e)
    intervals=valid_rr_pairs(peaks,500,300,2000)
    beats=phase_normalize(signal[:,None],intervals,samples=256).astype(np.float32)
    rr=np.diff(peaks)/500
    return beats,{'rpeaks':peaks.tolist(),'valid_intervals':intervals.tolist(),'detected_peaks':len(peaks),'valid_beats':len(beats),'represented_fraction':float(np.diff(intervals,axis=1).sum()/5000),'rr_seconds':rr.tolist(),'median_rate_bpm':float(60/np.median(rr)) if len(rr) else None,'extraction_error':error,'decision_status':'completed' if len(beats)>=2 else 'indeterminate_insufficient_valid_RR'}

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():raise RuntimeError('Preparation already frozen')
    sources=[Path(__file__),REPO/'experiments/zzu_avblock/baseline.py',REPO/'experiments/zzu_avblock/protocol.py',REPO/'experiments/ptbxl_distributional_repecg/src/repecg/paper07_operator/model.py',REPO/'experiments/ptbxl_distributional_repecg/src/repecg/common/models.py',REPO/'experiments/ptbxl_distributional_repecg/src/repecg/common/kernels.py',REPO/'experiments/ptbxl_distributional_repecg/src/repecg/common/phase.py']
    protocol={'models':['M0_RESNET_LEADII_10S','M1_SETOPERATOR_LEADII_10S'],'source_hashes':{str(p):sha(p) for p in sources},'reference_hashes':{name:sha(BASE/name) for name in ['clinical_cohort_contract.json','split_manifest.parquet','M0_best.pt','prediction_manifest.parquet']},'native_sampling_rate':500,'prefix_samples':5000,'observed_leads':['II'],'operator_q':Q.tolist(),'epochs_max':40,'patience':8,'seed':42,'batch_size':128,'optimizer':'AdamW lr=.001 weight_decay=.0001','loss':'unweighted BCEWithLogits','selection':'strict maximum validation AUROC among representation-completed rows; minimum two valid RR intervals for M1','threshold':'lowest native float32 threshold with >=95% validation specificity among completed controls; also report all-control coverage','failure_policy':'No score and indeterminate; positive failures count as nondetections in all-eligible sensitivity; negative failures are not true negatives. AUROC/AUPRC/Brier completed-only; paired ranking contrasts additionally use common-completed subset','peak_detection':'XQRS observed Lead II only; original inclusive 300-2000 ms RR; minimum two valid intervals; no fabricated peaks','representation':'original 256 phase samples, 16 cells, 128 IMQ Nystrom features; c2=1; original whitening and NystromMap; physical mV','fit':'voltage population SD from all training Lead-II prefixes; at most128 seeded atoms per successful training record for whitening/landmarks; all successful training records eligible; no validation/test fitting','temporal_extension':'not authorized to launch; assess only','preprocessing_reference':'Matched prior M0 protocol; NOT paper denoising replication; raw M0 train-only per-lead standardization, M1 authentic train-fit kernel transform'}
    save(OUT/'protocol.json',protocol)
    f=cohort();f.to_parquet(OUT/'cohort.parquet',index=False)
    mm=np.load(BASE/'primary_30s_mV.npy',mmap_mode='r');wave=np.asarray(mm[f.cache_index.to_numpy(),1,:5000]).copy();del mm
    np.save(OUT/'leadII_mV.npy',wave)
    start=time.time();rows=[];parts=[];offset=0
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,(beats,row) in enumerate(pool.map(extract,wave,chunksize=16)):
            row.update(record_uid=f.record_uid.iloc[i],beat_offset=offset);offset+=len(beats);parts.append(beats);rows.append(row)
            if i%500==0:print('EXTRACT',i,'seconds',round(time.time()-start,1),flush=True)
    pd.DataFrame(rows).to_parquet(OUT/'extraction.parquet',index=False)
    packed=np.concatenate(parts);np.save(OUT/'phase_beats_mV.npy',packed)
    tr=np.flatnonzero(f.partition.to_numpy()=='train');scale=float(wave[tr].std(dtype=np.float64));rng=np.random.default_rng(42);atoms=[]
    for i in tr:
        row=rows[i]
        if row['valid_beats']<2:continue
        y=packed[row['beat_offset']:row['beat_offset']+row['valid_beats'],:,0]/scale
        a=np.stack((y,np.gradient(y,axis=-1)),axis=-1).reshape(-1,2)
        atoms.append(a[rng.choice(len(a),size=min(128,len(a)),replace=False)])
    atoms=np.concatenate(atoms);white=WhiteningTransform.fit(atoms);mapping=NystromMap.fit(white.transform(atoms),landmarks=128,c2=1,seed=42)
    np.savez(OUT/'phase_fit.npz',voltage_scale=scale,whitening_mean=white.mean,whitening_components=white.components,whitening_scales=white.scales,landmarks=mapping.landmarks,inverse_root=mapping.inverse_root,c2=1,sampling_rate=500)
    # Exact versus approximate training-only kernel distances on independently drawn cells.
    cells=[white.transform(atoms[rng.choice(len(atoms),16,replace=False)]) for _ in range(128)]
    exact=[];approx=[]
    for _ in range(500):
        a,b=rng.choice(128,2,replace=False);exact.append(biased_mmd2(cells[a],cells[b]));approx.append(np.square(mapping.mean(cells[a])-mapping.mean(cells[b])).sum())
    audit=audit_nystrom(np.array(exact),np.array(approx));save(OUT/'kernel_audit.json',audit)
    assert audit['passed'],audit
    # Import original feature computation without its invalid fallback extractor.
    sys.path.insert(0,str(REPO/'experiments/echonext_operator_benchmark'))
    from phase_kme_representation import EchoNextPhaseKMEExtractor
    extractor=EchoNextPhaseKMEExtractor(OUT/'phase_fit.npz',device='cuda');q=torch.tensor(Q[None],dtype=torch.float32,device='cuda')
    response=np.zeros((len(f),1,16,128),dtype=np.float32)
    checks={}
    for i,row in enumerate(rows):
        if row['valid_beats']<2:continue
        b=packed[row['beat_offset']:row['beat_offset']+row['valid_beats']]
        # Unobserved coordinates remain zero only in this algebra adapter; q selects II exactly.
        basis=np.zeros((len(b),256,8),np.float32);basis[:,:,1]=b[:,:,0]
        with torch.inference_mode():r=extractor.compute_phase_kme(basis,q)
        response[i]=r.cpu().numpy()
        if not checks:
            y=b[:,:,0]/scale;a=np.stack((y,np.gradient(y,axis=-1)),axis=-1).reshape(len(b),16,16,2).transpose(1,0,2,3).reshape(16,-1,2)
            direct=np.stack([mapping.mean(white.transform(c)) for c in a])
            checks['numpy_torch_max_error']=float(np.abs(direct-response[i,0]).max());assert np.allclose(direct,response[i,0],atol=2e-4)
            checks['unit_identity']=bool(np.array_equal(basis@Q,b[:,:,0]));assert checks['unit_identity']
            with torch.inference_mode():negative=extractor.compute_phase_kme(-basis,-q)
            checks['joint_polarity_error']=float(torch.max(torch.abs(r-negative)));assert checks['joint_polarity_error']<1e-5
        if i%1000==0:print('FEATURE',i,flush=True)
    assert np.isfinite(response).all();np.save(OUT/'responses.npy',response)
    checks.update(shape=list(response.shape),preparation_seconds=time.time()-start,fit_training_uid_sha256=hashlib.sha256('\n'.join(f.iloc[tr].record_uid).encode()).hexdigest(),fit_atoms=len(atoms),successful_training_records=len(atoms)//128)
    save(OUT/'representation_checks.json',checks)

class SingleSet(nn.Module):
    def __init__(self):
        super().__init__();self.original=OperatorSetModel(response_dim=128,classes=1,operator_mode='continuous',operator_dim=8,patient_context_dim=0)
        self.register_buffer('q',torch.tensor(Q,dtype=torch.float32).reshape(1,1,8))
    def forward(self,x):return self.original(self.q.expand(len(x),-1,-1),x).squeeze(-1)

def model_for(name):
    if name=='M1':return SingleSet()
    model=ResNet();model.stem[0]=nn.Conv1d(1,32,15,2,padding=7,bias=False);return model

def train(name,smoke=False):
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);assert torch.backends.cudnn.version()==90300;torch.backends.cudnn.benchmark=True
    model=model_for(name).cuda()
    if smoke:
        x=torch.randn((8,1,16,128) if name=='M1' else (8,1,5000),device='cuda')
        loss=nn.functional.binary_cross_entropy_with_logits(model(x),torch.tensor([0,1]*4,dtype=torch.float32,device='cuda'));loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model(x).shape==(8,)
        print('LEADII_KERNEL_GRADIENT_WITNESS',name,float(loss.detach()),flush=True);return
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert sha(path)==digest,path
    directory=OUT/name;directory.mkdir(exist_ok=True)
    if (directory/'prediction_manifest.parquet').exists():raise RuntimeError('Frozen predictions exist')
    start=time.time();f=cohort();yy=f.label.to_numpy();ex=pd.read_parquet(OUT/'extraction.parquet');valid=ex.decision_status.eq('completed').to_numpy() if name=='M1' else np.ones(len(f),bool)
    x=torch.tensor(np.load(OUT/('responses.npy' if name=='M1' else 'leadII_mV.npy')),device='cuda')
    indices={p:np.flatnonzero((f.partition.to_numpy()==p)&valid) for p in ['train','validation','test']}
    tr=torch.tensor(indices['train'],device='cuda');va=torch.tensor(indices['validation'],device='cuda');prep={}
    if name=='M0':
        x=x[:,None,:];mean=x[tr].mean(dim=(0,2),keepdim=True);std=x[tr].std(dim=(0,2),unbiased=False,keepdim=True);x=(x-mean)/std;prep={'mean':mean.cpu(),'std':std.cpu()}
    x=x.to(torch.bfloat16);y=torch.tensor(yy,dtype=torch.float32,device='cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[];best=-1;best_epoch=-1
    for epoch in range(40):
        tick=time.time();model.train();perm=tr[torch.randperm(len(tr),device='cuda')];total=0.
        for j in range(0,len(perm),128):
            idx=perm[j:j+128];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=nn.functional.binary_cross_entropy_with_logits(model(x[idx]),y[idx])
            assert torch.isfinite(loss);loss.backward();optimizer.step();total+=float(loss.detach())*len(idx)
        scores=predict(model,x,va);auc=roc_auc_score(yy[indices['validation']],scores)
        row={'epoch':epoch,'loss':total/len(tr),'validation_auroc_completed':auc,'seconds':time.time()-tick};history.append(row);print(name,json.dumps(row),flush=True)
        state={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,'validation_auroc':float(auc),**prep}
        if auc>best:best=auc;best_epoch=epoch;torch.save(state,directory/'best.pt')
        if epoch-best_epoch>=8:break
    torch.save(state,directory/'final.pt');pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
    cp=torch.load(directory/'best.pt',weights_only=True);model.load_state_dict(cp['state_dict']);val=predict(model,x,va);threshold=locked_threshold(yy[indices['validation']],val)
    # Freeze before any held-out model inference.
    lock={'checkpoint_sha256':sha(directory/'best.pt'),'source_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'threshold':threshold,'best_epoch':best_epoch,'validation_auroc_completed':best,'training_records_completed':len(tr),'validation_records_completed':len(va)}
    save(directory/'evaluation_lock.json',lock)
    frames=[]
    for partition in ['validation','test']:
        idx=np.flatnonzero(f.partition.to_numpy()==partition);completed=indices[partition];scores=val if partition=='validation' else predict(model,x,torch.tensor(completed,device='cuda'))
        allscores=np.full(len(f),np.nan);allscores[completed]=scores
        frame=f.iloc[idx].copy();frame['score']=allscores[idx];frame['decision_status']=np.where(valid[idx],'completed','indeterminate_insufficient_valid_RR');frame['threshold']=threshold;frame['decision']=np.where(valid[idx],allscores[idx]>=threshold,False);frame['model']=name;frames.append(frame)
    ledger=pd.concat(frames);ledger.to_parquet(directory/'prediction_manifest.parquet',index=False)
    # Independent checkpoint reload and second inference: prove saved outputs correspond to saved weights.
    replay=model_for(name).cuda();replay.load_state_dict(torch.load(directory/'best.pt',weights_only=True)['state_dict']);delta=0.
    for part in ['validation','test']:
        ids=indices[part];s=predict(replay,x,torch.tensor(ids,device='cuda'));saved=ledger.set_index('record_uid').loc[f.iloc[ids].record_uid].score.to_numpy();delta=max(delta,float(np.max(np.abs(s-saved))))
    assert delta==0
    test=ledger[ledger.partition=='test'];result=metrics(test.label,test.score,test.decision_status,threshold)
    result.update(correct_negative_fraction_all_controls=float(((test.label==0)&test.decision_status.eq('completed')&~test.decision).sum()/(test.label==0).sum()))
    save(directory/'results.json',result);save(directory/'receipt.json',dict(**lock,epochs_run=len(history),seconds=time.time()-start,parameters_total=sum(p.numel() for p in model.parameters()),parameters_with_gradient=sum(p.numel() for p in model.parameters() if p.grad is not None),peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,replay_max_score_difference=delta,final_checkpoint_sha256=sha(directory/'final.pt')))
    print('RESULT',name,json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','M0','M1','smoke']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='smoke':train('M0',True);train('M1',True)
    else:train(a.stage)
