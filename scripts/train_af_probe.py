#!/usr/bin/env python3
"""Train AFProbe-v1 exclusively on real PTB-XL folds 1--8."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.af_probe import AFProbeV1, PTBAFDataset, TRAIN_CONDITIONS, checkpoint_contract, load_rows
from scripts.af_protocol import DEFAULT_LABEL_MAP, sha256_file

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--workers',type=int,default=6); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='cuda'); a=p.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
 if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
 rows=load_rows(set(range(1,9))); ds=PTBAFDataset(rows,TRAIN_CONDITIONS)
 y=np.repeat(rows.af_label.to_numpy(dtype=int),len(TRAIN_CONDITIONS)); counts=np.bincount(y,minlength=2); weights=np.where(y==1,1/counts[1],1/counts[0])
 gen=torch.Generator().manual_seed(a.seed); loader=DataLoader(ds,batch_size=a.batch_size,sampler=WeightedRandomSampler(weights,len(weights),generator=gen),num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
 model=AFProbeV1().to(a.device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); lossfn=torch.nn.BCEWithLogitsLoss()
 history=[]
 for epoch in range(a.epochs):
  model.train(); total=n=0
  for x,y,_,_ in loader:
   x,y=x.to(a.device,non_blocking=True),y.to(a.device,non_blocking=True); opt.zero_grad(set_to_none=True); loss=lossfn(model(x),y); loss.backward(); opt.step(); total+=float(loss)*len(y); n+=len(y)
  history.append({'epoch':epoch+1,'train_loss':total/n}); print(json.dumps(history[-1]),flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 torch.save({'model':model.state_dict(),'width':64,'contract':checkpoint_contract(),'label_map_sha256':sha256_file(DEFAULT_LABEL_MAP),'seed':a.seed,'history':history},a.output)
if __name__=='__main__': main()
