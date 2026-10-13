#!/usr/bin/env python3
"""Stage 2C-2: functional generalization to frozen unseen operators."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from repecg.paper07_operator import normalize_operator
from repecg.paper07_operator.model import OperatorSetModel
from run_stagewise_linear_probe_audit import CONFIGURATIONS, _extract, _macro_auc, _patient_split

def load_map(path, device):
    with np.load(path) as x:
        keys=('physical_mean','physical_std','voltage_scale','whitening_mean','whitening_components','whitening_scales','landmarks','inverse_root','c2')
        if any(k not in x for k in keys): raise ValueError('incomplete response fit')
        return {k:torch.as_tensor(x[k],device=device,dtype=torch.float32) for k in keys}

def phase_map(beats, q, fit):
    # beats [B,256,8] physical; q [Q,8] normalized; output [Q,16,128]
    scale=fit['voltage_scale']; wave=torch.einsum('bti,qi->bqt',beats,q)/scale; deriv=torch.gradient(wave,dim=2)[0]
    atoms=torch.stack((wave,deriv),dim=-1).reshape(beats.shape[0],q.shape[0],16,16,2).permute(2,0,1,3,4)
    cells=atoms.reshape(16,-1,2); white=(cells-fit['whitening_mean'])@fit['whitening_components']*fit['whitening_scales']
    dist=torch.cdist(white,fit['landmarks'][None,:,:]); phi=torch.rsqrt(dist.square()+fit['c2'])@fit['inverse_root']
    return phi.reshape(16,beats.shape[0],q.shape[0],16,128).mean(dim=(1,3)).permute(1,0,2)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--contexts',type=Path,required=True); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--response-fit',type=Path,required=True); ap.add_argument('--bank',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--q-batch',type=int,default=16); ap.add_argument('--record-batch',type=int,default=32); ap.add_argument('--seed',type=int,default=2718); args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    torch.backends.cudnn.enabled=False; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with np.load(args.bank) as x: q=np.asarray(x['q'],dtype=np.float32); families=np.asarray(x['family'])
    train_ops=np.load('/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/development_representations/representation_train.npz',mmap_mode='r')['operators'].reshape(-1,8)
    train_ops=np.unique(np.asarray(train_ops),axis=0); exact=[]; mins=[]
    for row in q:
        exact.append(bool(np.any(np.linalg.norm(train_ops-row,axis=1)<1e-8) or np.any(np.linalg.norm(train_ops+row,axis=1)<1e-8)))
        mins.append(float(np.arccos(np.clip(np.max(np.abs(train_ops@row)),0,1))))
    if any(exact): raise ValueError('audit bank overlaps training operators')
    ctx={s:{k:np.load(args.contexts/f'{s}_{k}.npy',mmap_mode='r') for k in ('labels','patient_ids','ecg_ids')} for s in ('train','validation')}
    qc=pd.read_csv(args.cache/'qc.csv').query('eligible').reset_index(drop=True)
    if not np.array_equal(qc.ecg_id.to_numpy(np.int64),ctx['validation']['ecg_ids']): raise ValueError('cache/context record order mismatch')
    fit=load_map(args.response_fit,device); physical=[]
    for artifact in qc.artifact:
        with np.load(args.cache/artifact) as x: physical.append(np.asarray(x['beats'],dtype=np.float32)*fit['physical_std'].cpu().numpy()+fit['physical_mean'].cpu().numpy())
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False); variant=str(payload.get('variant','')); operator_mode='raw_q' if variant in {'linear_q_encoder','raw_q_from_set_operator_robust'} else 'continuous'; model=OperatorSetModel(response_dim=128,classes=5,operator_mode=operator_mode); model.load_state_dict(payload['state_dict'],strict=True); model=model.to(device).eval()
    # Fit the exact Stage-1 Q8 E6 probe on folds 1-7, with patient-disjoint selection.
    train_resp=np.load('/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_full12_aux_mmd/frozen_full12_contexts/train_responses.npy',mmap_mode='r'); ops=np.load(args.contexts/'operators.npy'); q8=list(CONFIGURATIONS['Q8']); q8feat=_extract(model,train_resp[:,q8],ops[q8],tuple(range(len(q8))),device,256)['E6_latent']; fitmask,selmask=_patient_split(ctx['train']['patient_ids'],42); sc=StandardScaler().fit(q8feat[fitmask]); probe=Ridge(alpha=1e-3).fit(sc.transform(q8feat[fitmask]),ctx['train']['labels'][fitmask])
    labels=ctx['validation']['labels']; native=np.zeros((len(q),len(labels),5),np.float32); aligned=np.zeros_like(native)
    with torch.inference_mode():
        for start in range(0,len(q),args.q_batch):
            qb=torch.as_tensor(q[start:start+args.q_batch],device=device); qn=len(qb)
            # explicit per-record path preserves beat grouping and is the reference implementation
            native_rows=[]; aligned_rows=[]
            for beats_np in physical:
                r=phase_map(torch.as_tensor(beats_np,device=device),qb,fit).unsqueeze(1); o=qb.unsqueeze(1)
                logits=model(o,r); z=model.encode_context(o,r); native_rows.append(torch.sigmoid(logits).float().cpu().numpy()); aligned_rows.append(probe.predict(sc.transform(z.float().cpu().numpy())).astype(np.float32))
            native[start:start+qn]=np.asarray(native_rows).transpose(1,0,2); aligned[start:start+qn]=np.asarray(aligned_rows).transpose(1,0,2)
    out=[]
    for i in range(len(q)):
        out.append({'index':i,'family':str(families[i]),'min_train_angle_rad':mins[i],'native_macro_auroc':_macro_auc(labels,native[i]),'q8_probe_macro_auroc':_macro_auc(labels,aligned[i])})
    args.output.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output/'predictions.npz',q=q,native=native,q8_probe=aligned,labels=labels)
    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result={'schema_version':'p07_unseen_operator_functional_audit_v1','status':'complete','checkpoint_path':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_sha256,'training_overlap_exact':exact,'records':out,'bank_families':{f:int(np.sum(families==f)) for f in np.unique(families)}}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':'complete','output':str(args.output),'checkpoint_path':str(args.checkpoint.resolve()),'checkpoint_sha256':checkpoint_sha256}))
if __name__=='__main__': main()
