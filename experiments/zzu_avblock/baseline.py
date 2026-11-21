"""One development-selected raw ResNet baseline; held-out scores are emitted once."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from protocol import metrics,locked_threshold

class Block(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.main=nn.Sequential(nn.Conv1d(cin,cout,7,stride,padding=3,bias=False),nn.BatchNorm1d(cout),nn.ReLU(),nn.Conv1d(cout,cout,7,padding=3,bias=False),nn.BatchNorm1d(cout))
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(nn.Conv1d(cin,cout,1,stride,bias=False),nn.BatchNorm1d(cout))
    def forward(self,x):return torch.relu(self.main(x)+self.skip(x))

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(12,32,15,2,padding=7,bias=False),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(3,2,padding=1))
        self.layers=nn.Sequential(Block(32,32),Block(32,32),Block(32,64,2),Block(64,64),Block(64,128,2),Block(128,128),Block(128,256,2),Block(256,256))
        self.head=nn.Linear(256,1)
    def forward(self,x):return self.head(self.layers(self.stem(x)).mean(dim=-1)).squeeze(-1)

@torch.inference_mode()
def predict(model,x,indices,batch=256):
    model.eval();chunks=[]
    for i in range(0,len(indices),batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):logit=model(x[indices[i:i+batch]])
        chunks.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(chunks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=a.output
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    assert torch.cuda.is_available() and torch.backends.cudnn.version()==90300
    torch.backends.cudnn.benchmark=True
    model=ResNet().cuda()
    if a.smoke:
        x=torch.randn(8,12,5000,device='cuda');y=torch.tensor([0,1]*4,device='cuda',dtype=torch.float32)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=nn.functional.binary_cross_entropy_with_logits(model(x),y)
        loss.backward();assert torch.isfinite(loss)
        print('ZZU_RESNET_KERNEL_WITNESS',tuple(model(x).shape),torch.cuda.get_device_name(),'loss',loss.item(),flush=True);return
    if (out/'prediction_manifest.parquet').exists():raise RuntimeError('Held-out predictions already exist; do not rerun/tune this baseline')
    contract=json.loads((out/'clinical_cohort_contract.json').read_text());cfg=contract['initial_training']
    f=pd.read_parquet(out/'split_manifest.parquet');f=f[f.primary_eligible].sort_values('cache_index').reset_index(drop=True)
    assert len(f)==contract['primary_n'] and f.cache_index.is_unique and (f.cache_index>=0).all()
    indices={p:np.flatnonzero(f.partition.to_numpy()==p) for p in ['train','validation','test']}
    for p,idx in indices.items():assert len(np.unique(f.label.to_numpy()[idx]))==2,p
    mm=np.load(out/'primary_30s_mV.npy',mmap_mode='r')
    started=time.time();x=torch.from_numpy(mm[f.cache_index.to_numpy(),:,:5000]).cuda();del mm
    yy=f.label.to_numpy();y=torch.tensor(yy,dtype=torch.float32,device='cuda')
    train_idx=torch.tensor(indices['train'],device='cuda');val_idx=torch.tensor(indices['validation'],device='cuda')
    mean=x[train_idx].mean(dim=(0,2),keepdim=True);std=x[train_idx].std(dim=(0,2),keepdim=True,unbiased=False)
    assert bool((std>0).all());x=((x-mean)/std).to(torch.bfloat16)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best_auc=-1;best_epoch=-1;history=[];train_start=time.time()
    for epoch in range(cfg['epochs_max']):
        before=time.time();model.train();perm=train_idx[torch.randperm(len(train_idx),device='cuda')];loss_sum=0.
        for start in range(0,len(perm),cfg['batch_size']):
            idx=perm[start:start+cfg['batch_size']];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=nn.functional.binary_cross_entropy_with_logits(model(x[idx]),y[idx])
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        scores=predict(model,x,val_idx);auc=roc_auc_score(yy[indices['validation']],scores)
        item={'epoch':epoch,'train_loss':loss_sum/len(train_idx),'validation_auroc':auc,'seconds':time.time()-before,'train_examples_per_second':len(train_idx)/(time.time()-before)}
        history.append(item);print(json.dumps(item),flush=True)
        if auc>best_auc:
            best_auc=auc;best_epoch=epoch
            payload={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'mean':mean.cpu(),'std':std.cpu(),'best_epoch':epoch,'validation_auroc':float(auc),'contract_sha256':hashlib.file_digest((out/'clinical_cohort_contract.json').open('rb'),'sha256').hexdigest()}
            torch.save(payload,out/'M0_best.pt')
        if epoch-best_epoch>=cfg['patience']:break
    training_seconds=time.time()-train_start;pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    checkpoint=torch.load(out/'M0_best.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict'],strict=True)
    val_scores=predict(model,x,val_idx);threshold=locked_threshold(yy[indices['validation']],val_scores,.95)
    # Test access starts only after checkpoint and threshold are frozen.
    cp_hash=hashlib.file_digest((out/'M0_best.pt').open('rb'),'sha256').hexdigest()
    prep={'version':'training_partition_only_per_lead_mean_population_std_mV','mean':checkpoint['mean'].numpy().tolist(),'std':checkpoint['std'].numpy().tolist(),'fit_record_ids_sha256':hashlib.sha256('\n'.join(f.iloc[indices['train']].record_uid).encode()).hexdigest()}
    (out/'preprocessing.json').write_text(json.dumps(prep,indent=2));prep_hash=hashlib.file_digest((out/'preprocessing.json').open('rb'),'sha256').hexdigest()
    ledger=[]
    for partition in ['validation','test']:
        idx=indices[partition];scores=val_scores if partition=='validation' else predict(model,x,torch.tensor(idx,device='cuda'))
        for j,score in zip(idx,scores):
            row=f.iloc[j]
            ledger.append({'record_uid':row.record_uid,'record_group':row.record_group,'partition':partition,'label':int(row.label),'negative_stratum':row.negative_stratum,'model':'M0_raw_ResNet18_1d','checkpoint_sha256':cp_hash,'preprocessing_sha256':prep_hash,'lead_configuration':'full12','duration_seconds':10,'crop_start_sample':0,'sampling_rate_hz':500,'score':float(score),'decision_status':'completed','threshold':threshold,'decision':bool(score>=threshold),'age_years':row.age_years_approx,'sex':row.sex,'source_cohort':row.cohort,'source_duration_seconds':row.duration_seconds,'multilabel':row.label_count>1})
    ledger=pd.DataFrame(ledger);ledger.to_parquet(out/'prediction_manifest.parquet',index=False)
    from evaluate import evaluate
    evaluate(out)
    # Real forward latency after warmup; excludes disk/preprocessing, explicitly batch 128.
    model.eval();batch=x[:128]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(5):model(batch)
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(30):model(batch)
        torch.cuda.synchronize();lat=(time.perf_counter()-t)/30
    receipt={'active_parameters':sum(p.numel() for p in model.parameters()),'epochs_run':len(history),'best_epoch':best_epoch,'best_validation_auroc':best_auc,'training_seconds':training_seconds,'total_load_train_predict_seconds':time.time()-started,'inference_latency_ms_per_record_amortized_batch128':1000*lat/128,'gpu_peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'device':torch.cuda.get_device_name(),'threshold':threshold,'checkpoint_sha256':cp_hash,'source_sha256':hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),'test_was_not_used_for_selection':True,'cudnn_benchmark':True,'mixed_precision':'bfloat16','reproducibility':'seeded, cuDNN autotuning; exact bitwise retraining not asserted'}
    (out/'baseline_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()
