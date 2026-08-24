#!/usr/bin/env python3
"""Score frozen finalists under five AF conditions into compact SQLite."""
import argparse, json, sqlite3, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.af_probe import load_checkpoint, preprocess
from scripts.af_protocol import sha256_file

CONDITIONS=('A_real12','B_source','C_real11','D_hybrid12','E_synthetic11')
def build(real,recon,source,condition):
 mask=np.ones(12,dtype=np.float32)
 if condition=='A_real12': x=real
 elif condition=='B_source': x=real.copy(); mask[:]=0; mask[source]=1
 elif condition=='C_real11': x=real.copy(); mask[source]=0
 elif condition=='D_hybrid12': x=recon.copy(); x[:,source]=real[:,source]
 elif condition=='E_synthetic11': x=recon.copy(); mask[source]=0
 else: raise ValueError(condition)
 return x,mask
def assert_identity(real,recon,x,mask,source,condition):
 if condition=='B_source' and (mask.sum()!=1 or mask[source]!=1): raise AssertionError('B source leakage')
 if condition in {'C_real11','E_synthetic11'} and (mask.sum()!=11 or mask[source]!=0): raise AssertionError('11-lead mask violation')
 if condition=='D_hybrid12' and not np.array_equal(x[:,source],real[:,source]): raise AssertionError('D source changed')
 if condition=='E_synthetic11' and not np.array_equal(x[:,np.arange(12)!=source],recon[:,np.arange(12)!=source]): raise AssertionError('E not synthetic')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--bundle',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--calibration',type=Path,required=True); p.add_argument('--output-db',type=Path,required=True); p.add_argument('--cohort',choices=['rdb_test','ptbxl_fold10'],required=True); p.add_argument('--confirm-frozen-final',action='store_true'); p.add_argument('--device',default='cuda'); a=p.parse_args()
 if not a.confirm_frozen_final: raise SystemExit('BLOCKED: requires frozen finalist confirmation')
 cal=json.loads(a.calibration.read_text())
 if cal['status']!='frozen' or cal['calibration_folds']!=[9] or cal['checkpoint_sha256']!=sha256_file(a.checkpoint): raise RuntimeError('calibration mismatch')
 z=np.load(a.bundle,allow_pickle=False); req={'real','recon','label','record_id','subgroup','source_index'}
 if not req.issubset(z.files): raise RuntimeError(f'missing {req-set(z.files)}')
 real,recon=z['real'],z['recon']
 if real.shape!=recon.shape or real.shape[1:]!=(5000,12): raise RuntimeError('expected matched [N,5000,12]')
 model,_=load_checkpoint(a.checkpoint,a.device); rows=[]
 with torch.inference_mode():
  for i in range(len(real)):
   source=int(z['source_index'][i])
   if source not in (0,1): raise RuntimeError('source must be I or II')
   for condition in CONDITIONS:
    x,mask=build(real[i],recon[i],source,condition); assert_identity(real[i],recon[i],x,mask,source,condition)
    prob=float(torch.sigmoid(model(preprocess(x,mask)[None].to(a.device))).cpu())
    threshold_key={'A_real12':'real12','D_hybrid12':'real12','B_source':f'source_{"I" if source==0 else "II"}','C_real11':f'real11_{"I" if source==0 else "II"}','E_synthetic11':f'real11_{"I" if source==0 else "II"}'}[condition]
    threshold=float(cal['conditions'][threshold_key]['threshold'])
    rows.append((a.cohort,str(z['record_id'][i]),str(z['subgroup'][i]),int(z['label'][i]),source,condition,prob,threshold,int(prob>=threshold)))
 a.output_db.parent.mkdir(parents=True,exist_ok=True); con=sqlite3.connect(a.output_db)
 con.executescript('CREATE TABLE IF NOT EXISTS predictions(cohort TEXT,record_id TEXT,subgroup TEXT,label INTEGER,source_index INTEGER,condition TEXT,probability REAL,threshold REAL,prediction INTEGER,PRIMARY KEY(cohort,record_id,source_index,condition)); CREATE TABLE IF NOT EXISTS provenance(key TEXT PRIMARY KEY,value TEXT);')
 con.executemany('INSERT OR REPLACE INTO predictions VALUES(?,?,?,?,?,?,?,?,?)',rows); con.executemany('INSERT OR REPLACE INTO provenance VALUES(?,?)',[('checkpoint_sha256',sha256_file(a.checkpoint)),('calibration_sha256',sha256_file(a.calibration)),('bundle_sha256',sha256_file(a.bundle))]); con.commit()
 summary={}
 for c in CONDITIONS:
  q=[r for r in rows if r[5]==c]; y=np.array([r[3] for r in q]); prob=np.array([r[6] for r in q]); pred=np.array([r[8] for r in q]); pos=y==1; neg=~pos
  subgroup_fpr={g:float(np.mean([r[8] for r in q if r[2]==g])) for g in sorted({r[2] for r in q if r[3]==0})}
  summary[c]={'n':len(q),'auroc':float(roc_auc_score(y,prob)),'auprc':float(average_precision_score(y,prob)),'sensitivity':float(pred[pos].mean()),'specificity':float(1-pred[neg].mean()),'subgroup_fpr':subgroup_fpr}
 print(json.dumps(summary,indent=2)); con.close()
if __name__=='__main__': main()
