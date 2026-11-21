"""First broad ZZU-ZUA 15-label comparison; immutable source record partitions."""
import argparse,hashlib,io,json,sys,time,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd,torch
from torch import nn
from scipy.io import loadmat
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'experiments/zzu_avblock'))
sys.path.insert(0,str(REPO/'experiments/ptbxl_distributional_repecg/src'))
from baseline import ResNet
from protocol import LEADS,locked_threshold
from leadii import extract as extract_II
from repecg.common.beats import detect_rpeaks
from repecg.common.phase import phase_normalize
from repecg.common.kernels import WhiteningTransform,NystromMap,biased_mmd2,audit_nystrom
from repecg.paper07_operator.model import OperatorSetModel
BASE=Path('/data/mithunmanivannan/codex_artifacts/zzu_avblock_20261120')
SOURCE=Path('/data/mithunmanivannan/datasets/figshare_31229326_v3')
OUT=Path('/data/mithunmanivannan/codex_artifacts/zzu_multilabel_20261120')
VOCAB=['Acute Myocardial Infarction','Acute Myocardial Ischemia','High-Grade/Third-Degree Atrioventricular Block','Severe Hypokalemia','R-on-T Ventricular Premature Contractions','Pulmonary Embolism','Ventricular Fibrillation','Overt T-Wave Alternans','Ventricular Flutter','Torsade de Pointes','Polymorphic Ventricular Tachycardia','Severe Hyperkalemia','Bidirectional Ventricular Tachycardia','Long RR Interval (>=3 seconds)','Atrial Fibrillation with Pre-excitation']
TACHY=[6,8,9,10,12,14]
Q=np.zeros((12,8),np.float64);Q[0,0]=1;Q[1,1]=1;Q[2,:2]=[-1,1];Q[3,:2]=[-.5,-.5];Q[4,:2]=[1,-.5];Q[5,:2]=[-.5,1];Q[6:,2:]=np.eye(6)
QN=np.linalg.norm(Q,axis=1);Q=Q/QN[:,None]

def sha(p):
 with Path(p).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def frame():return pd.read_parquet(OUT/'cohort.parquet')
def targets(f):
 y=np.zeros((len(f),15),np.float32)
 for i,labs in enumerate(f.labels_json.map(json.loads)):
  for label in labs:
   if label!='Normal':y[i,VOCAB.index(label)]=1
 return y

def prepare():
 OUT.mkdir(exist_ok=True)
 if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
 original=pd.read_parquet(BASE/'split_manifest.parquet').sort_values('record_uid').reset_index(drop=True)
 f=original[(original.duration_seconds>=10)&original.finite].copy().reset_index(drop=True);assert len(f)==20113;f['cache_row']=np.arange(len(f));y=targets(f)
 assert targets(original).sum(axis=0).astype(int).tolist()==[6289,3589,1481,724,616,365,96,96,60,49,15,10,9,5,2]
 assert (original.label_count>1).sum()==274
 support=[]
 for j,label in enumerate(VOCAB):
  row={'label_index':j,'label':label,'source_all_n':int(targets(original)[:,j].sum()),'eligible_10s_n':int(y[:,j].sum())}
  for p in ['train','validation','test']:row[p]=int(y[f.partition.eq(p).to_numpy(),j].sum())
  row['primary_macro_included']=row['train']>=20 and row['validation']>=5
  row['evidence']='individually_evaluable' if row['primary_macro_included'] else 'exploratory_or_undefined'
  support.append(row)
 evaluable=[r['label_index'] for r in support if r['primary_macro_included']]
 exposed=pd.read_parquet(BASE/'prediction_manifest.parquet');previous_test=set(exposed[exposed.partition=='test'].record_uid)
 frozen_files=[Path(__file__),REPO/'experiments/zzu_avblock/baseline.py',REPO/'experiments/zzu_avblock/protocol.py',REPO/'experiments/zzu_avblock/leadii.py']+list((REPO/'experiments/ptbxl_distributional_repecg/src/repecg/common').glob('*.py'))+[REPO/'experiments/ptbxl_distributional_repecg/src/repecg/paper07_operator/model.py']
 proto={'source_hashes':{str(p):sha(p) for p in frozen_files},'source_manifest_sha256':sha(SOURCE/'analysis/record_manifest.parquet'),'old_split_sha256':sha(BASE/'split_manifest.parquet'),'old_prediction_sha256':sha(BASE/'prediction_manifest.parquet'),'vocabulary':VOCAB,'severe_tachyarrhythmia_indices':TACHY,'severe_source_all_n':int((targets(original)[:,TACHY].sum(axis=1)>0).sum()),'cohort_n':len(f),'cohort_policy':'>=10 actual seconds and finite; first observed5000 samples at500Hz, uV/1000=mV; no duration30 selection; retain flatness flags; all-observed-flat configurations indeterminate, partial-flat records retained','split':'reuse existing record-group assignments for all source records, no reassignment or new tuning; groups not verified patients','prior_test_overlap':len(previous_test&set(f[f.partition=='test'].record_uid)),'prior_test_exposure':'binary full12 and LeadII models evaluated overlapping test records; new study is exploratory, not untouched external validation','target':'15 independent binary labels; Normal allzero; preserve multilabel source diagnoses; no crop event labels invented','macro_rule':'fixed labels with training>=20 and validation>=5 positives, selected before model fitting; no heldout-based label selection; undefined metrics remain null','macro_indices':evaluable,'primary_contrasts':'LeadII SetOperator minus natively trained LeadII ResNet: macroAUPRC on common-completed heldout records and severe-group all-eligible sensitivity at validation95% specificity; report coverage and full12 reference; no causal architecture superiority from failed fit','clinical_group_score':'maximum predicted sigmoid among the six source severe-tachyarrhythmia labels, threshold selected on validation only','individual_thresholds':'validation95% specificity only for labels with >=5 validation positives and >=1 training positive; otherwise no fitted threshold, raw cases/probabilities only','train':{'seed':42,'epochs_max':40,'patience':10,'minimum_epochs':20,'batch_size':128,'loss':'unweighted multilabel BCEWithLogits','optimizer':'AdamW wd=.0001','ResNet_lr':.001,'SetOperator_lr':.0001,'selection':'maximum fixed-label macroAUPRC on completed validation; strict checkpoint copy','precision':'ResNet BF16, SetOperator FP32 due training-only quantization evidence from prior binary experiment; scores stored FP32'},'representation':'original 16x128 PhaseKME,256phase samples,IMQ c2=1,128Nystrom; train-only kernel/voltage fit; no new architecture or reconstruction loss','detector':'LeadII: observed II only XQRS. Full12: original detector using actual I,II,V1..V6 with II primary and I,V2,V5 fallbacks. AcceptedRR300-2000ms; >=2 intervals; no invented beats','operator_convention':'[I,II,V1..V6]; normalized nominal clinical coefficients; measured voltage divided by coefficient norm for consistent normalized-q response. Use actual measured channels, not synthesized leads. Derived limb coefficient relation approximate on this release; source residuals retained','normalization':'ResNet train-only perlead mean/populationSD; SetOperator shared train-only physical voltage SD plus authentic atom-whitening and Nystrom. No metadata. No unspecified paper denoising invented','bootstrap':{'replicates':2000,'unit':'record_group, not patient','conditional':'fixed fitted models/validation thresholds; prior exposure, residual patient overlap and one seed limit inference'},'stage_gate':'four native models only:raw/set full12 and II. Variablelead,balancedloss,temporal and phaseablation not launched'}
 assert proto['severe_source_all_n']==223
 save(OUT/'protocol.json',proto);f.to_parquet(OUT/'cohort.parquet',index=False);np.save(OUT/'targets.npy',y);pd.DataFrame(support).to_csv(OUT/'label_support.csv',index=False)
 overlap=y.astype(int).T@y.astype(int);pd.DataFrame(overlap,index=VOCAB,columns=VOCAB).to_csv(OUT/'label_overlap.csv')
 original.assign(eligible_10s=original.record_uid.isin(f.record_uid)).to_parquet(OUT/'source_selection.parquet',index=False)
 combinations=f[f.label_count>1].groupby('labels_json').size().rename('n').reset_index();combinations.to_csv(OUT/'eligible_multilabel_combinations.csv',index=False)
 mm=np.lib.format.open_memmap(OUT/'waveforms_mV.npy',mode='w+',dtype='float32',shape=(len(f),12,5000));t=time.time();qc=[]
 for cohort,g in f.groupby('cohort',sort=False):
  archive=SOURCE/('Normal ECG signal.zip' if cohort=='normal' else 'Database # A publicly available 12-lead electrocardiogram criti-cal value dataset covering 13126 records.zip')
  with zipfile.ZipFile(archive) as z:
   for i,row in g.iterrows():
    data=loadmat(io.BytesIO(z.read(row.mat_archive_member)));x=np.stack([data[k].reshape(-1)[:5000] for k in LEADS]).astype(np.float32)*.001
    assert x.shape==(12,5000) and np.isfinite(x).all();mm[i]=x;qc.append({'record_uid':row.record_uid,**{lead+'_flat':bool(np.ptp(x[j])==0) for j,lead in enumerate(LEADS)}})
    if i%2000==0:print('SOURCE_CACHE',i,'seconds',round(time.time()-t,1),flush=True)
 mm.flush();pd.DataFrame(qc).set_index('record_uid').loc[f.record_uid].reset_index().to_parquet(OUT/'prefix_quality.parquet',index=False)
 save(OUT/'cache_receipt.json',{'records':len(f),'seconds':time.time()-t,'shape':list(mm.shape),'units':'mV','cache_sha256':sha(OUT/'waveforms_mV.npy'),'multilabel_source':274,'multilabel_eligible':int((f.label_count>1).sum()),'multilabel_combinations_eligible':len(combinations)})
 print('PREPARED',len(f),evaluable,flush=True)

def detect(item):
 signal,config=item
 if config=='II':
  _,info=extract_II(signal[1]);intervals=np.array(info['valid_intervals'],dtype=np.int64).reshape(-1,2);peaks=info['rpeaks'];lead='II'
 else:
  detection=detect_rpeaks(signal[[0,1,6,7,8,9,10,11]].T,fs=500);intervals=detection.valid_intervals;peaks=detection.rpeaks.tolist();lead=detection.lead
 return {'rpeaks':peaks,'valid_intervals':intervals.tolist(),'detector_lead':lead,'detected_peaks':len(peaks),'valid_beats':len(intervals),'represented_fraction':float(np.diff(intervals,axis=1).sum()/5000),'decision_status':'completed' if len(intervals)>=2 else 'indeterminate_insufficient_RR'}

def atoms_for(signal,intervals,leads,scale):
 b=phase_normalize(signal[leads].T,intervals,samples=256).astype(np.float32)/QN[leads][None,None,:]/scale
 y=b.transpose(2,0,1);return np.stack((y,np.gradient(y,axis=-1)),axis=-1).reshape(len(leads),len(b),16,16,2).transpose(0,2,1,3,4).reshape(len(leads),16,-1,2)

def represent():
 if (OUT/'representation_receipt.json').exists():raise RuntimeError('Representations already frozen')
 torch.set_num_threads(2);f=frame();wave=np.load(OUT/'waveforms_mV.npy',mmap_mode='r');tr=np.flatnonzero(f.partition.eq('train').to_numpy());started=time.time()
 for config in ['II','full12']:
  path=OUT/f'extraction_{config}.parquet'
  if path.exists():continue
  infos=[]
  with ProcessPoolExecutor(max_workers=4) as pool:
   items=((np.asarray(wave[i]),config) for i in range(len(f)))
   for i,info in enumerate(pool.map(detect,items,chunksize=16)):
    info['record_uid']=f.record_uid.iloc[i];infos.append(info)
    if i%2000==0:print('DETECT',config,i,'seconds',round(time.time()-started,1),flush=True)
  pd.DataFrame(infos).to_parquet(path,index=False)
 receipts={}
 # Each native model's kernel fit sees only its own observed training channels.
 for config,leads in [('II',[1]),('full12',list(range(12)))]:
  fit_start=time.time();scale_sum=0.;scale_sq=0.;n=0
  for i in tr:
   x=np.asarray(wave[i,leads],dtype=np.float64)/QN[leads,None];scale_sum+=x.sum();scale_sq+=np.square(x).sum();n+=x.size
  scale=float(np.sqrt(scale_sq/n-(scale_sum/n)**2));rng=np.random.default_rng(42);ex=pd.read_parquet(OUT/f'extraction_{config}.parquet');samples=[]
  for i in tr:
   row=ex.iloc[i]
   if row.valid_beats<2:continue
   intervals=np.array(list(row.valid_intervals),dtype=np.int64).reshape(-1,2);a=atoms_for(wave[i],intervals,leads,scale).reshape(-1,2);samples.append(a[rng.choice(len(a),128,replace=False)])
  samples=np.concatenate(samples);wh=WhiteningTransform.fit(samples);mapping=NystromMap.fit(wh.transform(samples),landmarks=128,c2=1,seed=42)
  np.savez(OUT/f'kernel_fit_{config}.npz',voltage_scale=scale,whitening_mean=wh.mean,whitening_components=wh.components,whitening_scales=wh.scales,landmarks=mapping.landmarks,inverse_root=mapping.inverse_root,c2=1)
  cells=[wh.transform(samples[rng.choice(len(samples),16,replace=False)]) for _ in range(128)];exact=[];approx=[]
  for _ in range(500):
   a,b=rng.choice(128,2,replace=False);exact.append(biased_mmd2(cells[a],cells[b]));approx.append(np.square(mapping.mean(cells[a])-mapping.mean(cells[b])).sum())
  audit=audit_nystrom(np.array(exact),np.array(approx));save(OUT/f'kernel_audit_{config}.json',audit);assert audit['passed']
  mean=torch.tensor(wh.mean,dtype=torch.float32,device='cuda');comp=torch.tensor(wh.components,dtype=torch.float32,device='cuda');scales=torch.tensor(wh.scales,dtype=torch.float32,device='cuda');landmarks=torch.tensor(mapping.landmarks,dtype=torch.float32,device='cuda');inv=torch.tensor(mapping.inverse_root,dtype=torch.float32,device='cuda');maxerror=0.
  response=np.lib.format.open_memmap(OUT/f'responses_{config}.npy',mode='w+',dtype='float32',shape=(len(f),len(leads),16,128));response[:]=0
  for i,row in ex.iterrows():
   if row.valid_beats<2:continue
   intervals=np.array(list(row.valid_intervals),dtype=np.int64).reshape(-1,2);a=atoms_for(wave[i],intervals,leads,scale);tensor=torch.tensor(a,dtype=torch.float32,device='cuda');white=(tensor-mean)@comp*scales
   with torch.inference_mode():r=(torch.rsqrt(torch.cdist(white,landmarks).square()+1)@inv).mean(dim=-2)
   response[i]=r.cpu().numpy()
   if i%2000==0:print('FEATURE',config,i,'seconds',round(time.time()-started,1),flush=True)
   if i<10:
    direct=np.stack([np.stack([mapping.mean(wh.transform(c)) for c in lead]) for lead in a]);err=float(np.abs(response[i]-direct).max());maxerror=max(maxerror,err);assert np.allclose(response[i],direct,atol=3e-4)
  response.flush();assert np.isfinite(response).all()
  receipts[config]={'seconds_kernel_fit_and_features':time.time()-fit_start,'training_fit_record_n':len(samples)//128,'fit_atom_n':len(samples),'max_numpy_torch_error':maxerror,'train_uid_sha256':hashlib.sha256('\n'.join(f.iloc[tr].record_uid).encode()).hexdigest(),'voltage_scale':scale,'fit_available_leads':[LEADS[j] for j in leads]}
 save(OUT/'representation_receipt.json',{'seconds':time.time()-started,'configurations':receipts,'original_kernel_source_sha256':sha(REPO/'experiments/ptbxl_distributional_repecg/src/repecg/common/kernels.py')})

class SetModel(nn.Module):
 def __init__(self,config):
  super().__init__();self.original=OperatorSetModel(response_dim=128,classes=15,operator_mode='continuous',operator_dim=8);self.register_buffer('q',torch.tensor(Q[[1]] if config=='II' else Q,dtype=torch.float32)[None])
 def forward(self,x):return self.original(self.q.expand(len(x),-1,-1),x)
def model_for(family,config):
 if family=='set':return SetModel(config)
 model=ResNet()
 if config=='II':model.stem[0]=nn.Conv1d(1,32,15,2,padding=7,bias=False)
 model.head=nn.Linear(256,15);return model
@torch.inference_mode()
def predict(model,x,idx,family):
 model.eval();chunks=[]
 for j in range(0,len(idx),256):
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=family=='raw'):logit=model(x[idx[j:j+256]])
  chunks.append(logit.float().cpu().numpy())
 return np.concatenate(chunks)
def macro(y,s,classes):
 if any(y[:,j].sum()==0 for j in classes):raise RuntimeError('Fixed macro label unsupported on completed validation; do not silently change labels')
 return float(np.mean([average_precision_score(y[:,j],s[:,j]) for j in classes]))
def train(family,config,smoke=False):
 torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);assert torch.backends.cudnn.version()==90300;torch.backends.cudnn.benchmark=True
 model=model_for(family,config).cuda()
 if smoke:
  x=torch.randn((4,1 if config=='II' else 12,16,128) if family=='set' else (4,1 if config=='II' else 12,5000),device='cuda');y=torch.zeros(4,15,device='cuda');y[0,0]=1
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=family=='raw'):loss=nn.functional.binary_cross_entropy_with_logits(model(x),y)
  loss.backward();assert torch.isfinite(loss) and model(x).shape==(4,15) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);print('MULTILABEL_GPU_WITNESS',family,config,float(loss.detach()),flush=True);return
 proto=json.loads((OUT/'protocol.json').read_text());assert all(sha(p)==h for p,h in proto['source_hashes'].items())
 out=OUT/(family+'_'+config);out.mkdir(exist_ok=True)
 if (out/'predictions.npz').exists():raise RuntimeError('Prediction archive already frozen')
 start=time.time();f=frame();yy=np.load(OUT/'targets.npy');classes=proto['macro_indices'];qc=pd.read_parquet(OUT/'prefix_quality.parquet');leads=[1] if config=='II' else list(range(12));valid=~qc[[LEADS[j]+'_flat' for j in leads]].all(axis=1).to_numpy()
 if family=='set':valid &= pd.read_parquet(OUT/f'extraction_{config}.parquet').decision_status.eq('completed').to_numpy();values=np.load(OUT/f'responses_{config}.npy')
 else:values=np.load(OUT/'waveforms_mV.npy',mmap_mode='r')[:,leads,:].copy()
 idx={p:np.flatnonzero(f.partition.eq(p).to_numpy()&valid) for p in ['train','validation','test']};tr=torch.tensor(idx['train'],device='cuda');va=torch.tensor(idx['validation'],device='cuda');x=torch.tensor(values,device='cuda');prep={};del values
 if family=='raw':
  mean=x[tr].mean(dim=(0,2),keepdim=True);std=x[tr].std(dim=(0,2),unbiased=False,keepdim=True);assert (std>0).all();x=(x-mean)/std;prep={'mean':mean.cpu(),'std':std.cpu()};x=x.to(torch.bfloat16)
 y=torch.tensor(yy,device='cuda');opt=torch.optim.AdamW(model.parameters(),lr=.001 if family=='raw' else .0001,weight_decay=.0001);best=-1.;best_epoch=-1;history=[]
 for epoch in range(40):
  tick=time.time();model.train();perm=tr[torch.randperm(len(tr),device='cuda')];total=0.
  for j in range(0,len(perm),128):
   ids=perm[j:j+128];opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=family=='raw'):loss=nn.functional.binary_cross_entropy_with_logits(model(x[ids]),y[ids])
   assert torch.isfinite(loss);loss.backward();opt.step();total+=float(loss.detach())*len(ids)
  logits=predict(model,x,va,family);score=1/(1+np.exp(-logits));value=macro(yy[idx['validation']],score,classes);row={'epoch':epoch,'loss':total/len(tr),'validation_macro_auprc':value,'seconds':time.time()-tick};history.append(row);print(family,config,json.dumps(row),flush=True)
  cp={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,'validation_macro_auprc':value,**prep}
  if value>best:best=value;best_epoch=epoch;torch.save(cp,out/'best.pt')
  if epoch>=19 and epoch-best_epoch>=10:break
 torch.save(cp,out/'final.pt');pd.DataFrame(history).to_csv(out/'history.csv',index=False);cp=torch.load(out/'best.pt',weights_only=True);model.load_state_dict(cp['state_dict']);val_logits=predict(model,x,va,family);val_score=1/(1+np.exp(-val_logits))
 thresholds=[]
 for j in range(15):thresholds.append(locked_threshold(yy[idx['validation'],j],val_score[:,j]) if yy[idx['validation'],j].sum()>=5 and yy[idx['train'],j].sum()>0 else None)
 group_y=(yy[:,TACHY].sum(axis=1)>0).astype(int);group_threshold=locked_threshold(group_y[idx['validation']],val_score[:,TACHY].max(axis=1))
 lock={'family':family,'config':config,'source_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'checkpoint_sha256':sha(out/'best.pt'),'label_thresholds':thresholds,'severe_group_threshold':group_threshold,'selection':'validation macroAUPRC only','best_epoch':best_epoch,'best_validation_macro_auprc':best};save(out/'evaluation_lock.json',lock)
 # Test inference only after selected checkpoint and endpoint thresholds are frozen.
 all_logits=np.full((len(f),15),np.nan,np.float32)
 for part in ['validation','test']:all_logits[idx[part]]=val_logits if part=='validation' else predict(model,x,torch.tensor(idx[part],device='cuda'),family)
 probability=1/(1+np.exp(-all_logits));np.savez_compressed(out/'predictions.npz',record_uid=f.record_uid.to_numpy(dtype=str),partition=f.partition.to_numpy(dtype=str),target=yy,logits=all_logits,probabilities=probability,completed=valid)
 # Check reload on a fixed completed heldout batch, no changes to scores or selection.
 replay=model_for(family,config).cuda();replay.load_state_dict(torch.load(out/'best.pt',weights_only=True)['state_dict']);ids=idx['test'][:256];again=predict(replay,x,torch.tensor(ids,device='cuda'),family);delta=float(abs(again-all_logits[ids]).max());assert delta==0
 save(out/'receipt.json',{**lock,'epochs':len(history),'seconds':time.time()-start,'peak_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'parameters':sum(p.numel() for p in model.parameters()),'gradient_parameters':sum(p.numel() for p in model.parameters() if p.grad is not None),'test_replay_max_logit_difference':delta,'completed_counts':{p:len(v) for p,v in idx.items()},'final_checkpoint_sha256':sha(out/'final.pt')});print('FINISHED',family,config,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','represent','train','smoke']);p.add_argument('--family',choices=['raw','set']);p.add_argument('--config',choices=['full12','II']);a=p.parse_args()
 if a.stage=='prepare':prepare()
 elif a.stage=='represent':represent()
 elif a.stage=='smoke':
  for family in ['raw','set']:
   for config in ['full12','II']:train(family,config,True)
 else:train(a.family,a.config)
