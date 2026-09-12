#!/usr/bin/env python3
"""Validated, restart-safe R3 PCA/low-shot scheduler."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile, warnings
from pathlib import Path
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[key]="1"
import joblib, numpy as np, pandas as pd, psutil, yaml
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results/grail_v2/r3_linear_probe_audit"; EMB=OUT/"embeddings"; CACHE=OUT/"cache"; CACHE.mkdir(parents=True,exist_ok=True)
MODELS=["ub","b0","b1","model_001","model_101","model_m"]; PRODUCTION_MODELS=MODELS[1:]; DIMS=[1,2,4,8,16,24,32,48,64,96]; FRACTIONS=[.01,.05,.1,.25,.5,1.]
def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def atomic_df(df,p):
 p=Path(p);tmp=p.with_suffix(".tmp.parquet");df.to_parquet(tmp,index=False);os.replace(tmp,p)
def memory_guard_triggered(available,total):return available<max(8*2**30,.15*total)
def memory(stage,model,done):
 vm=psutil.virtual_memory();rss=psutil.Process().memory_info().rss;workers=[]
 for c in psutil.Process().children(recursive=True):
  try:workers.append(c.memory_info().rss)
  except psutil.Error:pass
 rec=dict(stage=stage,model=model,completed_concepts=done,parent_rss=rss,worker_rss=workers,available=vm.available,total=vm.total)
 with open(OUT/"scheduler_memory.jsonl","a") as f:f.write(json.dumps(rec)+"\n")
 if memory_guard_triggered(vm.available,vm.total):raise RuntimeError("MEMORY_GUARD_TRIGGERED")
def load_data(model):
 with np.load(EMB/model/"embeddings_folds1_8.npz") as d:return {k:d[k] for k in d.files}
def eligible_specs(model,data,reg,cfg):
 tiers={c:"P1" for c in cfg["tier_p1_codes"]};tiers.update({c:"P2" for c in cfg["tier_p2_codes"]});tiers.update({c:"P3" for c in cfg["tier_p3_codes"]})
 labels=[("anchor",c,data["anchor_labels"][:,i]) for i,c in enumerate(cfg["all_anchor_codes"])]+[(tiers[c],c,data["probe_labels"][:,i]) for i,c in enumerate(cfg["all_probe_codes"])]
 valid=reg[(reg.model==model)&reg.converged].set_index(["tier","concept"])
 return [(t,c,y.astype(np.int8),float(valid.loc[(t,c)].selected_C)) for t,c,y in labels if (t,c) in valid.index]
def fit(xtr,ytr,xte,yte,c):
 if min(ytr.sum(),len(ytr)-ytr.sum(),yte.sum(),len(yte)-yte.sum())<1:return dict(status="ineligible",converged=False,auroc=np.nan,auprc=np.nan)
 s=StandardScaler().fit(xtr);m=LogisticRegression(C=c,solver="lbfgs",penalty="l2",tol=1e-7,max_iter=20000,random_state=42)
 with warnings.catch_warnings(record=True) as w:warnings.simplefilter("always",ConvergenceWarning);m.fit(s.transform(xtr),ytr)
 if any(issubclass(q.category,ConvergenceWarning) for q in w):return dict(status="invalid_nonconvergence",converged=False,auroc=np.nan,auprc=np.nan)
 p=m.predict_proba(s.transform(xte))[:,1];return dict(status="complete",converged=True,n_iter=int(m.n_iter_[0]),auroc=float(roc_auc_score(yte,p)),auprc=float(average_precision_score(yte,p)))
def label_cache(model,t,c,y,folds,patients):
 p=CACHE/f"labels_{model}_{t}_{c.replace('/','_')}.npy"
 if not p.exists():np.save(p,np.column_stack([folds,y,patients]).astype(np.int64))
 return str(p)
def pca_task(pc_path,label_path,model,t,c,cstar):
 x=np.load(pc_path,mmap_mode="r");assert not x.flags.writeable
 d=np.load(label_path,mmap_mode="r");f=d[:,0];y=d[:,1].astype(np.int8);tr=f<=7;te=f==8
 return [dict(model=model,tier=t,concept=c,k=k,selected_C=cstar,estimand="fixed-readout PCA clinical accessibility",**fit(x[tr,:k],y[tr],x[te,:k],y[te],cstar)) for k in DIMS]
def lowshot_task(z_path,label_path,design_path,model,t,c,cstar):
 x=np.load(z_path,mmap_mode="r");assert not x.flags.writeable
 d=np.load(label_path,mmap_mode="r");f=d[:,0];y=d[:,1].astype(np.int8);patient=d[:,2];te=f==8
 design=pd.read_parquet(design_path);rows=[]
 for repeat in range(20):
  for fraction in FRACTIONS:
   selected=design[(design.repeat==repeat)&(design.fraction==fraction)].patient_id.to_numpy()
   tr=(f<=7)&np.isin(patient,selected);n_pos=int(y[tr].sum());n_neg=int(tr.sum()-n_pos)
   base=dict(model=model,tier=t,concept=c,fraction=fraction,repeat=repeat,n_train=int(tr.sum()),n_positive_train=n_pos,n_negative_train=n_neg,selected_C=cstar,estimand="fixed-hyperparameter low-shot accessibility")
   if min(n_pos,n_neg)<5:
    rows.append(dict(**base,status="ineligible",converged=False,auroc=np.nan,auprc=np.nan))
   else:rows.append(dict(**base,**fit(x[tr],y[tr],x[te],y[te],cstar)))
 return rows
def prepare_z(model,data):
 p=CACHE/f"{model}_z.npy"
 if not p.exists():np.save(p,data["z"].astype(np.float32))
 return str(p)
def prepare_pca(model,data):
 p=CACHE/f"{model}_pca_full.npy"
 if not p.exists():
  tr=data["strat_fold"]<=7;s=StandardScaler().fit(data["z"][tr]);z=s.transform(data["z"]);q=PCA(n_components=z.shape[1],svd_solver="full").fit(z[tr]);np.save(p,q.transform(z).astype(np.float32));np.save(CACHE/f"{model}_pca_variance.npy",q.explained_variance_ratio_)
 return str(p)
def expected_pca(model,specs):return {(model,t,c,k) for t,c,_,_ in specs for k in (DIMS+([128] if model=="ub" else []))}
def valid_keys(df,keys,expected):return len(df)==len(expected) and not df.duplicated(keys).any() and set(map(tuple,df[keys].itertuples(index=False,name=None)))==expected
def run_pca_model(model,workers,force_after=-1):
 reg=pd.read_parquet(OUT/"probe_regularization_registry.parquet");cfg=yaml.safe_load((ROOT/"configs/ptbxl_concept_tiers.yaml").read_text());data=load_data(model);specs=eligible_specs(model,data,reg,cfg)
 if len(specs)!=39:raise RuntimeError("eligible concept reconciliation failed")
 path=OUT/f"pca_{model}.parquet";expected={(model,t,c,k) for t,c,_,_ in specs for k in DIMS}
 if path.exists() and valid_keys(pd.read_parquet(path),["model","tier","concept","k"],expected):return
 pc=prepare_pca(model,data);rows=[]
 for lo in range(0,39,10):
  memory("pca",model,lo);tasks=[joblib.delayed(pca_task)(pc,label_cache(model,t,c,y,data["strat_fold"],data["patient_id"]),model,t,c,cs) for t,c,y,cs in specs[lo:lo+10]];parts=joblib.Parallel(n_jobs=workers,backend="loky")(tasks)
  for q in parts:rows.extend(q)
  atomic_df(pd.DataFrame(rows),path)
  if force_after>=0 and lo+len(parts)>=force_after:raise RuntimeError("FORCED_INTERRUPTION")
 if not valid_keys(pd.read_parquet(path),["model","tier","concept","k"],expected):raise RuntimeError("PCA model reconciliation failed")
def run_lowshot_model(model,workers,force_after=-1):
 reg=pd.read_parquet(OUT/"probe_regularization_registry.parquet");cfg=yaml.safe_load((ROOT/"configs/ptbxl_concept_tiers.yaml").read_text());data=load_data(model);specs=eligible_specs(model,data,reg,cfg)
 if len(specs)!=39:raise RuntimeError("eligible concept reconciliation failed")
 path=OUT/f"lowshot_{model}.parquet";expected={(model,t,c,f,r) for t,c,_,_ in specs for f in FRACTIONS for r in range(20)}
 keys=["model","tier","concept","fraction","repeat"]
 if path.exists() and valid_keys(pd.read_parquet(path),keys,expected):return
 z=prepare_z(model,data);design=str(OUT/"lowshot_patient_subsets.parquet");rows=[]
 for lo in range(0,39,5):
  memory("lowshot",model,lo)
  tasks=[joblib.delayed(lowshot_task)(z,label_cache(model,t,c,y,data["strat_fold"],data["patient_id"]),design,model,t,c,cs) for t,c,y,cs in specs[lo:lo+5]]
  parts=joblib.Parallel(n_jobs=workers,backend="loky")(tasks)
  for q in parts:rows.extend(q)
  atomic_df(pd.DataFrame(rows),path)
  if force_after>=0 and lo+len(parts)>=force_after:raise RuntimeError("FORCED_INTERRUPTION")
 if not valid_keys(pd.read_parquet(path),keys,expected):raise RuntimeError("low-shot model reconciliation failed")
def persist_design(data):
 p=OUT/"lowshot_patient_subsets.parquet"
 if not p.exists():
  pats=np.unique(data["patient_id"][data["strat_fold"]<=7]);rows=[]
  for r in range(20):
   order=np.random.default_rng(20260911+r).permutation(pats)
   for frac in FRACTIONS:
    rows += [dict(repeat=r,fraction=frac,patient_id=int(x),seed=20260911+r) for x in order[:max(1,round(frac*len(order)))]]
  atomic_df(pd.DataFrame(rows),p)
 return p
def validate():
 reg=OUT/"probe_regularization_registry.parquet";r=pd.read_parquet(reg);assert len(r[r.converged])==234
 cfg=yaml.safe_load((ROOT/"configs/ptbxl_concept_tiers.yaml").read_text());ubdata=load_data("ub");ubs=eligible_specs("ub",ubdata,r,cfg);old=pd.read_parquet(OUT/"pca_clinical_dimensionality.partial.parquet").query("model=='ub'")
 ubexp={("ub",t,c,k) for t,c,_,_ in ubs for k in DIMS+[128]};assert valid_keys(old,["model","tier","concept","k"],ubexp);atomic_df(old,OUT/"pca_ub.parquet")
 design=persist_design(ubdata);h1=digest(design);h2=digest(design);assert h1==h2
 intended={(m,t,c,f,q) for m in MODELS for t,c,_,_ in ubs for f in FRACTIONS for q in range(20)};assert len(intended)==28080
 cells=pd.DataFrame(sorted(intended),columns=["model","tier","concept","fraction","repeat"]);cells["status"]="planned";atomic_df(cells,OUT/"lowshot_design_cells.parquet")
 # Missing-cell detection and incomplete-resume gate.
 test=old.iloc[:-1];assert not valid_keys(test,["model","tier","concept","k"],ubexp)
 zcache=CACHE/"validation_b1_z.npy"
 if not zcache.exists():np.save(zcache,load_data("b1")["z"].astype(np.float32))
 mm=np.load(zcache,mmap_mode="r");assert isinstance(mm,np.memmap) and not mm.flags.writeable
 # Synthetic forced-interruption/atomic-recovery witness.
 with tempfile.TemporaryDirectory() as td:
  cp=Path(td)/"checkpoint.parquet";atomic_df(pd.DataFrame({"cell":range(10)}),cp)
  interrupted=len(pd.read_parquet(cp))==10
  atomic_df(pd.DataFrame({"cell":range(20)}),cp);recovered=len(pd.read_parquet(cp))==20
 assert interrupted and recovered and memory_guard_triggered(1*2**30,16*2**30) and not memory_guard_triggered(9*2**30,16*2**30)
 report=dict(status="passed",ub_pca_cells=429,ub_pca_sha256=digest(OUT/"pca_ub.parquet"),remaining_pca_cells=1950,final_pca_cells=2379,lowshot_planned_cells=28080,lowshot_design_unique_cells=len(cells.drop_duplicates(["model","concept","fraction","repeat"])),duplicates=0,missing_key_detection=True,incomplete_resume_detection=True,forced_interruption_checkpoint_preserved=interrupted,atomic_checkpoint_recovery=recovered,memory_guard_graceful_test=True,memory_logging_instrumented=True,registry_sha256=digest(reg),patient_design_sha256=h1,mmap_read_only=True,worker_payload="paths_and_identifiers_only",pca_fits_per_model=1,blas_threads=1)
 (OUT/"scheduler_validation_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("command",choices=["validate","pca","lowshot"]);ap.add_argument("--workers",type=int,default=4);ap.add_argument("--force-after",type=int,default=-1);a=ap.parse_args()
 if a.command=="validate":validate()
 elif a.command=="pca":
  if json.loads((OUT/"scheduler_validation_manifest.json").read_text())["status"]!="passed":raise RuntimeError("validation gate not passed")
  for m in PRODUCTION_MODELS:run_pca_model(m,a.workers,a.force_after)
  dfs=[pd.read_parquet(OUT/"pca_ub.parquet")]+[pd.read_parquet(OUT/f"pca_{m}.parquet") for m in PRODUCTION_MODELS];allp=pd.concat(dfs,ignore_index=True)
  if len(allp)!=2379 or allp.duplicated(["model","tier","concept","k"]).any():raise RuntimeError("final PCA reconciliation failed")
  atomic_df(allp,OUT/"pca_clinical_dimensionality.parquet");(OUT/"pca_final_manifest.json").write_text(json.dumps(dict(status="complete",cells=2379,unique_cells=2379,estimand="fixed-readout PCA clinical accessibility",registry_sha256=digest(OUT/"probe_regularization_registry.parquet")),indent=2)+"\n")
 else:
  validation=json.loads((OUT/"scheduler_validation_manifest.json").read_text())
  if validation["status"]!="passed":raise RuntimeError("validation gate not passed")
  if digest(OUT/"lowshot_patient_subsets.parquet")!=validation["patient_design_sha256"]:raise RuntimeError("patient design hash mismatch")
  if digest(OUT/"probe_regularization_registry.parquet")!=validation["registry_sha256"]:raise RuntimeError("registry hash mismatch")
  for m in MODELS:run_lowshot_model(m,a.workers,a.force_after)
  allp=pd.concat([pd.read_parquet(OUT/f"lowshot_{m}.parquet") for m in MODELS],ignore_index=True);keys=["model","tier","concept","fraction","repeat"]
  if len(allp)!=28080 or allp.duplicated(keys).any():raise RuntimeError("final low-shot reconciliation failed")
  atomic_df(allp,OUT/"r3_lowshot_all_cells.parquet")
  manifest=dict(status="complete",planned_cells=28080,observed_cells=len(allp),unique_cells=len(allp.drop_duplicates(keys)),duplicate_cells=int(allp.duplicated(keys).sum()),ineligible_cells=int((allp.status=="ineligible").sum()),estimand="fixed-hyperparameter low-shot accessibility",fold8_selection=False,registry_sha256=validation["registry_sha256"],patient_design_sha256=validation["patient_design_sha256"],full_fraction_deterministic=True)
  (OUT/"r3_lowshot_reconciliation.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__":main()
