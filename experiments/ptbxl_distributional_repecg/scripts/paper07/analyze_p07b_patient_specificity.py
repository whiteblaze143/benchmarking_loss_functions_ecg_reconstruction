#!/usr/bin/env python3
"""Add raw-ridge control and patient-level morphology table to frozen P07-B outputs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy import signal
from sklearn.linear_model import Ridge

INDEPENDENT=(0,1,6,7,8,9,10,11); NAMES=("I","II","V1","V2","V3","V4","V5","V6")
CONTEXTS={"lead_I":0,"lead_II":1}

def raw(path, split, samples):
    x=np.load(path/f"{split}_waveforms.npy",mmap_mode="r")
    return signal.resample(np.asarray(x[:,INDEPENDENT],np.float32),samples,axis=-1).astype(np.float32)

def corr(a,b):
    a=a-a.mean(); b=b-b.mean(); d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d>1e-12 else 0.0

def fit(train_x,train_y,val_x,val_y):
    best=None
    for alpha in (1e-3,1e-2,1e-1,1.,10.):
        p=Ridge(alpha=alpha).fit(train_x,train_y).predict(val_x)
        key=(float(np.mean((p-val_y)**2)),alpha)
        if best is None or key<best[0]: best=(key,p)
    return best[1]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--targets',type=Path,required=True); p.add_argument('--original',type=Path,required=True); p.add_argument('--time-indexed',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--samples',type=int,default=256); a=p.parse_args()
 if a.output.exists() and any(a.output.iterdir()): raise FileExistsError(a.output)
 manifest=json.loads((a.targets/'manifest.json').read_text()); scale=np.asarray(manifest['scaler']['std'])[list(INDEPENDENT)]; mean=np.asarray(manifest['scaler']['mean'])[list(INDEPENDENT)]
 tr=raw(a.targets,'train',a.samples); va=raw(a.targets,'validation',a.samples)
 ids=np.load(a.targets/'validation_patient_ids.npy'); ecg=np.load(a.targets/'validation_ecg_ids.npy')
 pred={}
 for context,index in CONTEXTS.items(): pred[f'raw_waveform_ridge__{context}']=fit(tr[:,index],tr.reshape(len(tr),-1),va[:,index],va.reshape(len(va),-1)).reshape(len(va),8,a.samples)
 for label,path in [('original_kme',a.original),('time_indexed_kme',a.time_indexed)]:
  with np.load(path/'fold8_predictions.npz',allow_pickle=False) as z:
   for key in z.files:
    if key.endswith(('__ridge','__imq_nystrom')):
     context='lead_I' if '__lead_I__' in key else 'lead_II'
     readout='latent_ridge' if key.endswith('__ridge') else 'latent_imq_nystrom'
     pred[f'{label}__{context}__{readout}']=z[key][:,list(INDEPENDENT)]
 a.output.mkdir(parents=True); rows=[]; summary={}
 target=va*scale[None,:,None]+mean[None,:,None]
 for key,value in pred.items():
  if key.startswith('raw_'): encoder='raw_waveform'; _,context=key.split('__'); readout='ridge'
  else: encoder,context,readout=key.split('__')
  # Frozen-probe archives store standard-lead predictions in physical units;
  # only the newly fitted raw waveform ridge remains standardized here.
  if encoder == 'raw_waveform': value=value*scale[None,:,None]+mean[None,:,None]
  vals=[]
  for patient in np.unique(ids):
   ix=np.flatnonzero(ids==patient)
   for lead,name in enumerate(NAMES):
    true=target[ix,lead]; guess=value[ix,lead]; tv=float(np.var(true)); pv=float(np.var(guess)); row={'patient_id':int(patient),'context_lead':context,'target_lead':name,'encoder_variant':encoder,'readout':readout,'mae_mv':float(np.mean(np.abs(guess-true))),'rmse_mv':float(np.sqrt(np.mean((guess-true)**2))),'pearson_r':float(np.mean([corr(x,y) for x,y in zip(true,guess)])),'true_variance':tv,'pred_variance':pv,'variance_ratio':pv/max(tv,1e-12)}; rows.append(row); vals.append(row)
  for name in NAMES:
   x=[r for r in vals if r['target_lead']==name]; tv=np.asarray([r['true_variance'] for r in x]); pv=np.asarray([r['pred_variance'] for r in x]); summary.setdefault(key,{})[name]={'patients':len(x),'mae_mv':float(np.mean([r['mae_mv'] for r in x])),'rmse_mv':float(np.mean([r['rmse_mv'] for r in x])),'pearson_r':float(np.mean([r['pearson_r'] for r in x])),'variance_ratio':float(np.mean([r['variance_ratio'] for r in x])),'patient_variance_correlation':corr(tv,pv)}
 with (a.output/'patient_level_metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 np.savez_compressed(a.output/'raw_ridge_predictions.npz',ecg_ids=ecg,patient_ids=ids,**{k:v.astype(np.float32) for k,v in pred.items()})
 (a.output/'results.json').write_text(json.dumps({'status':'complete','fold':8,'patients':int(len(np.unique(ids))),'records':int(len(ids)),'metrics':summary,'patient_table':'patient_level_metrics.csv'},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
