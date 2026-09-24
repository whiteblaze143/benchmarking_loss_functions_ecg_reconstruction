#!/usr/bin/env python3
"""Frozen SJLIFE watch-to-clinical transport matching under an explicit uV-to-mV assumption."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from repecg.common.beats import detect_rpeaks
from repecg.common.phase import phase_normalize
from repecg.common.preprocess import bandpass_ecg, independent_basis
from repecg.paper07_operator import OperatorSetModel, canonical_operators, response_atoms, time_indexed_response_atoms
from scripts.paper07.build_frozen_full12_context_responses import TorchResponseMap
from scripts.paper07.run_frozen_ptbxl_conditional_probe import _checkpoint_state

def _basis(path: Path, watch: bool) -> np.ndarray:
    values=np.load(path).astype(np.float64)/1000.0 # Explicit transport-assumption: stored amplitudes are uV.
    if watch:
        full=np.zeros((len(values),12)); full[:,0]=values; return independent_basis(bandpass_ecg(full,fs=512))
    return independent_basis(bandpass_ecg(values[0].T,fs=500))

def _latent(basis: np.ndarray, fs: int, model: OperatorSetModel, response_map: TorchResponseMap, time_indexed: bool, device: torch.device) -> np.ndarray|None:
    detection=detect_rpeaks(
        basis, fs=fs, primary_lead="I" if fs == 512 else "II",
        fallback_leads=() if fs == 512 else ("I", "V2", "V5"),
    )
    beats=phase_normalize(basis,detection.valid_intervals)
    if len(beats)<2: return None
    atoms=(time_indexed_response_atoms if time_indexed else response_atoms)(beats,canonical_operators()[0],response_map.voltage_scale)
    response=response_map.phase_means(atoms)[None,None]
    q=torch.as_tensor(canonical_operators()[None,:1],device=device,dtype=torch.float32)
    with torch.inference_mode(): return model.encode_context(q,torch.as_tensor(response,device=device)).cpu().numpy()[0]

def _model(checkpoint: Path, device: torch.device) -> OperatorSetModel:
    state,_=_checkpoint_state(checkpoint); model=OperatorSetModel(response_dim=128,classes=5,operator_mode="continuous").to(device); missing,unexpected=model.load_state_dict(state,strict=False)
    if set(missing)!={key for key in missing if key.startswith('decoder.')} or unexpected: raise ValueError('checkpoint mismatch')
    return model.eval()

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True);p.add_argument('--original-response-fit',type=Path,required=True);p.add_argument('--time-response-fit',type=Path,required=True);p.add_argument('--original-checkpoint',type=Path,required=True);p.add_argument('--time-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists(): raise FileExistsError(a.output)
 torch.backends.cudnn.enabled=False; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); table=pd.read_csv(a.data/'shared_paired_data_243.csv')
 variants={'original_kme':(a.original_response_fit,a.original_checkpoint,False),'time_indexed_kme':(a.time_response_fit,a.time_checkpoint,True)}; out={}
 for name,(fit,checkpoint,time) in variants.items():
  rm=TorchResponseMap(fit,device); model=_model(checkpoint,device); watch=[];clinical=[];ids=[]
  for _,row in table.iterrows():
   ap=a.data/row.apple_loc_ECG_243; cp=a.data/'ClinicalECGs_full_243'/Path(row.clinical_loc_ECG_243).name.replace('apple_','clinical_')
   w=_latent(_basis(ap,True),512,model,rm,time,device); c=_latent(_basis(cp,False),500,model,rm,time,device)
   if w is not None and c is not None: watch.append(w);clinical.append(c);ids.append(str(row.sudoID))
  w=np.asarray(watch);c=np.asarray(clinical);d=np.linalg.norm(w[:,None]-c[None,:],axis=-1);same=np.diag(d);different=d[~np.eye(len(d),dtype=bool)]
  order=np.argsort(d,axis=1); ranks=np.asarray([np.flatnonzero(order[i]==i)[0]+1 for i in range(len(d))])
  out[name]={'admitted_pairs':len(ids),'same_distance':{'mean':float(same.mean()),'median':float(np.median(same))},'different_distance':{'mean':float(different.mean()),'median':float(np.median(different))},'retrieval':{'top1':float(np.mean(ranks<=1)),'top5':float(np.mean(ranks<=5)),'median_rank':float(np.median(ranks))},'ids':ids}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'schema_version':'sjlife_frozen_transport_matching_v1','status':'complete','assumption':'waveform arrays are microvolts and are converted to mV; transport-only, not sample-aligned reconstruction','results':out},indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'complete','results':{k:{x:v for x,v in q.items() if x!='ids'} for k,q in out.items()}}))
if __name__=='__main__':main()
