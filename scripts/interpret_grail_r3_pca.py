#!/usr/bin/env python3
"""Reconcile and interpret the frozen R3 fixed-readout PCA experiment."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/grail_v2/r3_linear_probe_audit"
KEYS=["model","tier","concept","k"]
PAIRS=[("model_101","b1"),("model_001","b1"),("model_m","b1"),("model_101","model_m")]

def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def main():
 p=OUT/"pca_clinical_dimensionality.parquet";regp=OUT/"probe_regularization_registry.parquet"
 d=pd.read_parquet(p);reg=pd.read_parquet(regp).query("converged")[['model','tier','concept','auroc']].rename(columns={'auroc':'full_auroc'})
 expected={"ub":429,"b0":390,"b1":390,"model_001":390,"model_101":390,"model_m":390}
 counts=d.groupby('model').size().to_dict();dups=int(d.duplicated(KEYS).sum())
 if len(d)!=2379 or dups or counts!=expected:raise RuntimeError(f"PCA reconciliation failed: rows={len(d)}, duplicates={dups}, counts={counts}")
 if digest(regp)!="be20a947fe1d8532a7650fdf0fb1096852ffc3516be63ac2cccf6390a50e39ce":raise RuntimeError("frozen registry hash mismatch")
 d=d.merge(reg,on=['model','tier','concept'],how='left',validate='many_to_one')
 d['denominator_stable']=(d.full_auroc-.5).abs()>=.05
 d['chance_adjusted_retention']=np.where(d.denominator_stable,(d.auroc-.5)/(d.full_auroc-.5),np.nan)
 rows=[]
 for key,g in d.groupby(['model','tier','concept'],sort=False):
  g=g.sort_values('k'); stable=bool(g.denominator_stable.iloc[0]); hit=g.loc[g.chance_adjusted_retention>=.95,'k']
  d95=float(hit.min()) if len(hit) else np.nan; sustained=np.nan
  if stable:
   vals=g.chance_adjusted_retention.to_numpy();ks=g.k.to_numpy()
   good=np.isfinite(vals)&(vals>=.95)
   for i in range(len(g)):
    if good[i:].all():sustained=float(ks[i]);break
  rows.append(dict(model=key[0],tier=key[1],concept=key[2],full_auroc=float(g.full_auroc.iloc[0]),denominator_stable=stable,d95_clinical=d95,d95_sustained=sustained,reached_d95=bool(np.isfinite(d95)),reached_sustained=bool(np.isfinite(sustained))))
 pc=pd.DataFrame(rows);pc.to_parquet(OUT/"r3_pca_d95_per_concept.parquet",index=False)
 wide=d.pivot_table(index=['model','tier','concept'],columns='k',values='auroc').reset_index()
 pc=pc.merge(wide,on=['model','tier','concept'],how='left')
 sums=[]
 for (m,t),g in pc.groupby(['model','tier']):
  valid=g[g.denominator_stable]
  rec=dict(model=m,tier=t,n_concepts=len(g),n_stable=len(valid),n_unstable=len(g)-len(valid),full_auroc_mean=g.full_auroc.mean())
  for col,label in [('d95_clinical','d95'),('d95_sustained','d95_sustained')]:
   x=valid[col].dropna();rec.update({f'{label}_median':x.median(),f'{label}_q1':x.quantile(.25),f'{label}_q3':x.quantile(.75),f'{label}_mean':x.mean(),f'{label}_n_reached':len(x)})
   for k in (8,16,24,32):rec[f'{label}_fraction_le_{k}']=(valid[col]<=k).mean()
  for k in (8,16,24,32):rec[f'auroc_k{k}_mean']=g[k].mean() if k in g else np.nan
  sums.append(rec)
 summary=pd.DataFrame(sums);summary.to_csv(OUT/"r3_pca_model_tier_summary.csv",index=False)
 paired=[]
 for a,b in PAIRS:
  ga=pc[pc.model==a].set_index(['tier','concept']);gb=pc[pc.model==b].set_index(['tier','concept'])
  for key in ga.index.intersection(gb.index):
   rec=dict(comparison=f'{a}-{b}',model_a=a,model_b=b,tier=key[0],concept=key[1])
   for col in ['full_auroc','d95_clinical','d95_sustained',8,16,24,32]:rec[f'{col}_delta']=ga.loc[key,col]-gb.loc[key,col]
   paired.append(rec)
 pd.DataFrame(paired).to_parquet(OUT/"r3_pca_paired_model_deltas.parquet",index=False)
 # Compact, data-derived interpretation table.
 lead=summary[summary.model.isin(['b1','model_001','model_101','model_m'])][['model','tier','n_concepts','n_stable','full_auroc_mean','d95_median','d95_sustained_median','auroc_k8_mean','auroc_k16_mean','auroc_k24_mean','auroc_k32_mean']]
 text=["# R3 PCA Interpretation","","Estimand: **fixed-readout PCA clinical accessibility**. Sustained d95 is the conservative complexity measure. P3 is exploratory (n=2).","","```text",lead.to_string(index=False),"```",""]
 q=pd.DataFrame(paired)
 for comp in q.comparison.unique():
  text += [f"## {comp}","","```text",q[q.comparison==comp].groupby('tier')[['full_auroc_delta','d95_clinical_delta','d95_sustained_delta','8_delta','16_delta','24_delta','32_delta']].mean().to_string(),"```",""]
 (OUT/"R3_PCA_INTERPRETATION.md").write_text("\n".join(text))
 manifest=dict(status='complete',cells=len(d),unique_cells=len(d.drop_duplicates(KEYS)),duplicate_cells=dups,missing_expected_cells=0,models=counts,registry_sha256=digest(regp),estimand='fixed-readout PCA clinical accessibility',unstable_denominator_threshold_abs_full_minus_chance=.05)
 (OUT/"r3_pca_reconciliation_final.json").write_text(json.dumps(manifest,indent=2)+"\n")
 print(lead.to_string(index=False));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
