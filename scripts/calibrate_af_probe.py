#!/usr/bin/env python3
"""Freeze AFProbe-v1 thresholds on real PTB-XL fold 9 only."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.af_probe import PTBAFDataset, TRAIN_CONDITIONS, load_checkpoint, load_rows
from scripts.af_protocol import stable_hash, sha256_file

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--workers',type=int,default=6); p.add_argument('--device',default='cuda'); a=p.parse_args()
 model,payload=load_checkpoint(a.checkpoint,a.device); rows=load_rows({9}); result={}
 for condition in TRAIN_CONDITIONS:
  loader=DataLoader(PTBAFDataset(rows,(condition,),False),batch_size=a.batch_size,num_workers=a.workers)
  ys=[]; ps=[]
  with torch.inference_mode():
   for x,y,_,_ in loader: ys.extend(y.numpy()); ps.extend(torch.sigmoid(model(x.to(a.device))).cpu().numpy())
  fpr,tpr,thr=roc_curve(ys,ps); ix=int(np.argmax(tpr-fpr)); result[condition]={'threshold':float(thr[ix]),'youden_j':float(tpr[ix]-fpr[ix]),'auroc':float(roc_auc_score(ys,ps)),'auprc':float(average_precision_score(ys,ps)),'n':len(ys)}
 out={'status':'frozen','calibration_dataset':'PTB-XL','calibration_folds':[9],'prohibited_folds':[10],'reconstruction_seen':False,'checkpoint_sha256':sha256_file(a.checkpoint),'checkpoint_contract':payload['contract'],'conditions':result}
 out['calibration_hash']=stable_hash(out); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
