#!/usr/bin/env python3
"""R3 Passes B/C: one-PCA-per-model and genuine frozen-C low-shot probes."""
from __future__ import annotations
import argparse, json, os, warnings
from pathlib import Path
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="1"
import joblib, numpy as np, pandas as pd, yaml
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]; MODELS=["ub","b0","b1","model_001","model_101","model_m"]
FRACTIONS=[.01,.05,.10,.25,.50,1.0]; DIMS=[1,2,4,8,16,24,32,48,64,96]

def fit_fixed(xtr,ytr,xte,yte,c,meta):
    if min(ytr.sum(),len(ytr)-ytr.sum(),yte.sum(),len(yte)-yte.sum())<1: return None
    s=StandardScaler().fit(xtr); clf=LogisticRegression(C=c,solver="lbfgs",penalty="l2",tol=1e-7,max_iter=20000,random_state=42)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always",ConvergenceWarning); clf.fit(s.transform(xtr),ytr)
    if any(issubclass(w.category,ConvergenceWarning) for w in ws): return {**meta,"converged":False}
    p=clf.predict_proba(s.transform(xte))[:,1]
    return {**meta,"converged":True,"n_iter":int(clf.n_iter_[0]),"auroc":float(roc_auc_score(yte,p)),"auprc":float(average_precision_score(yte,p))}

def nested_patient_sets(patient,train_mask,repeats=20):
    pats=np.unique(patient[train_mask]); out={}
    for r in range(repeats):
        order=np.random.default_rng(20260911+r).permutation(pats)
        for f in FRACTIONS: out[(r,f)]=order[:max(1,round(f*len(order)))]
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=4); a=ap.parse_args()
    out=ROOT/"results/grail_v2/r3_linear_probe_audit"; emb=out/"embeddings"
    reg=pd.read_parquet(out/"probe_regularization_registry.parquet"); cfg=yaml.safe_load((ROOT/"configs/ptbxl_concept_tiers.yaml").read_text())
    tier={c:"P1" for c in cfg["tier_p1_codes"]}; tier.update({c:"P2" for c in cfg["tier_p2_codes"]}); tier.update({c:"P3" for c in cfg["tier_p3_codes"]})
    pca_all=[]; low_all=[]
    for model in MODELS:
        with np.load(emb/model/"embeddings_folds1_8.npz") as d: data={k:d[k] for k in d.files}
        z=data["z"].astype(np.float64); folds=data["strat_fold"]; patient=data["patient_id"]; tr=folds<=7; te=folds==8
        specs=[("anchor",c,data["anchor_labels"][:,i].astype(np.int8)) for i,c in enumerate(cfg["all_anchor_codes"])]
        specs += [(tier[c],c,data["probe_labels"][:,i].astype(np.int8)) for i,c in enumerate(cfg["all_probe_codes"])]
        cstar={(r.concept,r.tier):float(r.selected_C) for _,r in reg[(reg.model==model)&(reg.converged==True)].iterrows()}
        # One standardized maximum-rank PCA per model; all k conditions are slices.
        base_scaler=StandardScaler().fit(z[tr]); zstd=base_scaler.transform(z)
        maxdim=z.shape[1]
        # Transform every split with this single fitted decomposition.
        pca_obj=PCA(n_components=maxdim,svd_solver="full").fit(zstd[tr]); pcs=pca_obj.transform(zstd)
        jobs=[]
        for t,c,y in specs:
            if (c,t) not in cstar: continue
            for k in DIMS+([128] if model=="ub" else []):
                if k<=maxdim: jobs.append(joblib.delayed(fit_fixed)(pcs[tr,:k],y[tr],pcs[te,:k],y[te],cstar[(c,t)],dict(model=model,concept=c,tier=t,k=k,selected_C=cstar[(c,t)])))
        rows=joblib.Parallel(n_jobs=a.workers,backend="loky",verbose=5)(jobs); pca_all += [r for r in rows if r]
        pd.DataFrame(pca_all).to_parquet(out/"pca_clinical_dimensionality.partial.parquet",index=False)
        # Shared nested patient subsets; no Fold-7 injection and no C retuning.
        subsets=nested_patient_sets(patient,tr); jobs=[]
        for t,c,y in specs:
            if (c,t) not in cstar: continue
            for (repeat,fraction),chosen in subsets.items():
                sel=tr & np.isin(patient,chosen); pos=int(y[sel].sum()); neg=int(len(y[sel])-pos)
                if min(pos,neg)<5: continue
                jobs.append(joblib.delayed(fit_fixed)(z[sel],y[sel],z[te],y[te],cstar[(c,t)],dict(model=model,concept=c,tier=t,fraction=fraction,repeat=repeat,n_train_patients=len(chosen),n_train=len(y[sel]),n_positive_train=pos,n_negative_train=neg,selected_C=cstar[(c,t)],estimand="fixed-hyperparameter low-shot accessibility")))
        rows=joblib.Parallel(n_jobs=a.workers,backend="loky",verbose=5)(jobs); low_all += [r for r in rows if r]
        pd.DataFrame(low_all).to_parquet(out/"low_shot_shared_nested_patients.partial.parquet",index=False)
    pca=pd.DataFrame(pca_all); full=reg[["model","concept","tier","auroc"]].rename(columns={"auroc":"full_auroc"}); pca=pca.merge(full,on=["model","concept","tier"],how="left"); pca["chance_adjusted_retention"]=(pca.auroc-.5)/(pca.full_auroc-.5); pca.to_parquet(out/"pca_clinical_dimensionality.parquet",index=False)
    pca[pca.chance_adjusted_retention>=.95].groupby(["model","concept","tier"],as_index=False).k.min().to_parquet(out/"clinical_d95.parquet",index=False)
    low=pd.DataFrame(low_all); low.to_parquet(out/"low_shot_shared_nested_patients.parquet",index=False)
    summ=low[low.converged==True].groupby(["model","tier","fraction"],as_index=False).auroc.agg(["mean","std","count"]).reset_index(); summ["ci95_half_width"]=1.96*summ["std"]/np.sqrt(summ["count"]); summ.to_csv(out/"low_shot_summary.csv",index=False)
    (out/"frozen_c_followups_manifest.json").write_text(json.dumps({"status":"complete","C_source":"probe_regularization_registry.parquet","PCA_fits_total":6,"PCA_fit_folds":[1,2,3,4,5,6,7],"low_shot_fractions":FRACTIONS,"low_shot_repeats":20,"nested_patient_subsets":True,"fold7_injected":False,"estimand":"fixed-hyperparameter low-shot accessibility","evaluation_fold":8},indent=2)+"\n")
if __name__=="__main__": main()
