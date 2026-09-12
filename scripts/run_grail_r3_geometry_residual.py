#!/usr/bin/env python3
"""R3 Pass D1: cross-representation geometry and B1-residual probes."""
from __future__ import annotations
import json, os, warnings
from pathlib import Path
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="1"
import numpy as np, pandas as pd, yaml
from scipy.linalg import orthogonal_procrustes, subspace_angles
from sklearn.cross_decomposition import CCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results/grail_v2/r3_linear_probe_audit"; EMB=OUT/"embeddings"
MODELS=["b1","model_001","model_101","model_m"]

def load(n):
    with np.load(EMB/n/"embeddings_folds1_8.npz") as d: return {k:d[k] for k in d.files}
def cka(x,y):
    x=x-x.mean(0); y=y-y.mean(0); return float(np.linalg.norm(x.T@y,"fro")**2/(np.linalg.norm(x.T@x,"fro")*np.linalg.norm(y.T@y,"fro")+1e-12))
def nbr_overlap(x,y,k=10):
    ix=NearestNeighbors(n_neighbors=k+1).fit(x).kneighbors(return_distance=False)[:,1:]
    iy=NearestNeighbors(n_neighbors=k+1).fit(y).kneighbors(return_distance=False)[:,1:]
    return float(np.mean([len(set(a)&set(b))/k for a,b in zip(ix,iy)]))
def probe(xtr,ytr,xte,yte,c):
    s=StandardScaler().fit(xtr); clf=LogisticRegression(C=c,solver="lbfgs",penalty="l2",tol=1e-7,max_iter=20000,random_state=42)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always",ConvergenceWarning); clf.fit(s.transform(xtr),ytr)
    if any(issubclass(w.category,ConvergenceWarning) for w in ws): return None
    p=clf.predict_proba(s.transform(xte))[:,1]
    return float(roc_auc_score(yte,p)),float(average_precision_score(yte,p)),int(clf.n_iter_[0])
def main():
    registry=pd.read_parquet(OUT/"probe_regularization_registry.parquet")
    cfg=yaml.safe_load((ROOT/"configs/ptbxl_concept_tiers.yaml").read_text())
    data={m:load(m) for m in MODELS}; ref=data["b1"]
    for m in MODELS[1:]:
        if not np.array_equal(ref["ecg_id"],data[m]["ecg_id"]): raise RuntimeError("record mismatch")
    folds=ref["strat_fold"]; tr=folds<=7; te=folds==8
    geom=[]; residuals={}
    for target in MODELS[1:]:
        a,b=data["b1"]["z"],data[target]["z"]
        sa,sb=StandardScaler().fit(a[tr]),StandardScaler().fit(b[tr]); A,B=sa.transform(a),sb.transform(b)
        ridge=Ridge(alpha=1.0).fit(A[tr],B[tr]); pred=ridge.predict(A[te]); resid=B-ridge.predict(A)
        residuals[target]=resid
        r2=1-np.sum((B[te]-pred)**2)/np.sum((B[te]-B[te].mean(0))**2)
        cos=np.mean(np.sum(B[te]*pred,1)/(np.linalg.norm(B[te],axis=1)*np.linalg.norm(pred,axis=1)+1e-12))
        q=min(32,A.shape[1],B.shape[1]); qa=np.linalg.svd(A[tr],full_matrices=False)[2][:q].T; qb=np.linalg.svd(B[tr],full_matrices=False)[2][:q].T
        R,_=orthogonal_procrustes(A[tr],B[tr]); proc=np.linalg.norm(A[te]@R-B[te])/np.linalg.norm(B[te])
        geom.append(dict(source="b1",target=target,fold8_linear_r2=float(r2),fold8_cosine=float(cos),fold8_normalized_error=float(np.linalg.norm(B[te]-pred)/np.linalg.norm(B[te])),centered_linear_cka=cka(A[te],B[te]),procrustes_error=float(proc),mean_principal_angle_rad=float(subspace_angles(qa,qb).mean()),neighborhood_overlap_k10=nbr_overlap(A[te],B[te],10)))
    pd.DataFrame(geom).to_parquet(OUT/"cross_representation_geometry.parquet",index=False)
    tiers={c:"P1" for c in cfg["tier_p1_codes"]}; tiers.update({c:"P2" for c in cfg["tier_p2_codes"]}); tiers.update({c:"P3" for c in cfg["tier_p3_codes"]})
    rows=[]
    for target,resid in residuals.items():
        specs=[("anchor",c,ref["anchor_labels"][:,i]) for i,c in enumerate(cfg["all_anchor_codes"])]
        specs += [(tiers[c],c,ref["probe_labels"][:,i]) for i,c in enumerate(cfg["all_probe_codes"])]
        for tier,c,y in specs:
            rr=registry[(registry.model==target)&(registry.concept==c)&(registry.converged==True)]
            if rr.empty or len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
            met=probe(resid[tr],y[tr],resid[te],y[te],float(rr.iloc[0].selected_C))
            if met: rows.append(dict(target=target,source="b1",concept=c,tier=tier,selected_C=float(rr.iloc[0].selected_C),auroc=met[0],auprc=met[1],n_iter=met[2],converged=True))
    pd.DataFrame(rows).to_parquet(OUT/"b1_residual_subspace_probes.parquet",index=False)
    (OUT/"geometry_residual_manifest.json").write_text(json.dumps({"status":"complete","alignment_fit_folds":[1,2,3,4,5,6,7],"evaluation_fold":8,"C_source":"probe_regularization_registry.parquet"},indent=2)+"\n")
if __name__=="__main__": main()
