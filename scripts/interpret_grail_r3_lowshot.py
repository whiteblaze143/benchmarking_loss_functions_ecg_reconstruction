#!/usr/bin/env python3
"""Reconcile and interpret the fixed-design R3 low-shot experiment."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/grail_v2/r3_linear_probe_audit"
KEYS=['model','tier','concept','fraction','repeat'];PAIRS=[('model_101','b1'),('model_001','b1'),('model_m','b1'),('model_101','model_m')]
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=OUT/'r3_lowshot_all_cells.parquet';d=pd.read_parquet(p);v=json.loads((OUT/'scheduler_validation_manifest.json').read_text())
 duplicates=int(d.duplicated(KEYS).sum());observed=set(map(tuple,d[KEYS].itertuples(index=False,name=None)))
 design=pd.read_parquet(OUT/'lowshot_design_cells.parquet');expected=set(map(tuple,design[KEYS].itertuples(index=False,name=None)))
 missing=len(expected-observed);extra=len(observed-expected)
 if len(d)!=28080 or duplicates or missing or extra:raise RuntimeError('low-shot reconciliation failed')
 if digest(OUT/'lowshot_patient_subsets.parquet')!=v['patient_design_sha256'] or digest(OUT/'probe_regularization_registry.parquet')!=v['registry_sha256']:raise RuntimeError('frozen-design hash mismatch')
 valid=d[d.status=='complete'].copy()
 summary=valid.groupby(['model','tier','fraction'],as_index=False).agg(auroc_mean=('auroc','mean'),auroc_std=('auroc','std'),auprc_mean=('auprc','mean'),n_valid=('auroc','size'),n_concepts=('concept','nunique'))
 inel=d[d.status=='ineligible'].groupby(['model','tier','fraction']).size().rename('n_ineligible').reset_index();summary=summary.merge(inel,on=['model','tier','fraction'],how='outer').fillna({'n_ineligible':0})
 summary['auroc_ci95_half_width']=1.96*summary.auroc_std/np.sqrt(summary.n_valid)
 summary.to_csv(OUT/'r3_lowshot_model_tier_summary.csv',index=False)
 paired=[]
 for a,b in PAIRS:
  aa=d[d.model==a];bb=d[d.model==b]
  z=aa.merge(bb,on=['tier','concept','fraction','repeat'],suffixes=('_a','_b'),validate='one_to_one')
  z['comparison']=f'{a}-{b}';z['auroc_delta']=z.auroc_a-z.auroc_b;z['auprc_delta']=z.auprc_a-z.auprc_b
  paired.append(z[['comparison','tier','concept','fraction','repeat','status_a','status_b','auroc_delta','auprc_delta']])
 q=pd.concat(paired,ignore_index=True);q.to_parquet(OUT/'r3_lowshot_paired_deltas.parquet',index=False)
 table=q[(q.status_a=='complete')&(q.status_b=='complete')].groupby(['comparison','tier','fraction'])[['auroc_delta','auprc_delta']].agg(['mean','median','count'])
 text=['# R3 Low-shot Interpretation','', 'Estimand: **fixed-hyperparameter low-shot accessibility**. Patient subsets are nested within repeats and identical across representations. Fold 8 is evaluation-only.','', 'The 100% condition is deterministic under the fixed design; its 20 duplicate-design evaluations are not sampling variability.','', '```text',table.to_string(),'```','']
 (OUT/'R3_LOWSHOT_INTERPRETATION.md').write_text('\n'.join(text))
 rec=dict(status='complete',planned_cells=28080,observed_cells=len(d),unique_cells=len(observed),duplicate_cells=duplicates,missing_cells=missing,extra_cells=extra,ineligible_cells=int((d.status=='ineligible').sum()),invalid_nonconvergence_cells=int((d.status=='invalid_nonconvergence').sum()),registry_sha256=v['registry_sha256'],patient_design_sha256=v['patient_design_sha256'],estimand='fixed-hyperparameter low-shot accessibility',fold8_selection=False,full_fraction_deterministic=True)
 (OUT/'r3_lowshot_reconciliation.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
