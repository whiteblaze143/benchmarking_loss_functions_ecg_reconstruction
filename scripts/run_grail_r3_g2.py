#!/usr/bin/env python3
"""R3-G2 incremental accessibility from training-derived B1 residuals."""
from __future__ import annotations
import hashlib, json, os, warnings
from pathlib import Path
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[k]="1"
import joblib, numpy as np, pandas as pd, yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results/grail_v2/r3_linear_probe_audit"; EMB=OUT/"embeddings"
TARGETS=["model_001","model_101","model_m"]; CS=[1e-4,1e-3,1e-2,1e-1,1.,10.,100.]
def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def atomic(df,path):
    tmp=path.with_suffix(".tmp"+path.suffix); df.to_parquet(tmp,index=False); tmp.replace(path)
def load(name):
    with np.load(EMB/name/"embeddings_folds1_8.npz") as d:return {k:d[k] for k in d.files}
def metrics(y,p):
    fpr,tpr,_=roc_curve(y,p); spec=1-fpr
    return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),sens_at_95spec=float(np.max(tpr[fpr<=.05])),spec_at_95sens=float(np.max(spec[tpr>=.95])) if np.any(tpr>=.95) else 0.)
def fit_g2(target,concept,tier,x,y,folds):
    m6=folds<=6;m7=folds==7;m17=folds<=7;m8=folds==8
    s6=StandardScaler().fit(x[m6]); a,b=s6.transform(x[m6]),s6.transform(x[m7]); candidates=[]
    clf=LogisticRegression(solver="lbfgs",penalty="l2",warm_start=True,max_iter=5000,tol=1e-7,random_state=42)
    for c in CS:
        clf.set_params(C=c)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always",ConvergenceWarning);clf.fit(a,y[m6])
        if not any(issubclass(w.category,ConvergenceWarning) for w in ws):candidates.append((roc_auc_score(y[m7],clf.predict_proba(b)[:,1]),-CS.index(c),c,int(clf.n_iter_[0])))
    if not candidates:return dict(target=target,concept=concept,tier=tier,status="invalid_C_search",converged=False)
    va,_,c,nsel=max(candidates);s=StandardScaler().fit(x[m17]);final=LogisticRegression(C=c,solver="lbfgs",penalty="l2",max_iter=20000,tol=1e-7,random_state=42)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always",ConvergenceWarning);final.fit(s.transform(x[m17]),y[m17])
    if any(issubclass(w.category,ConvergenceWarning) for w in ws):return dict(target=target,concept=concept,tier=tier,status="invalid_final",selected_C=c,converged=False)
    return dict(target=target,concept=concept,tier=tier,status="complete",selected_C=c,fold7_auroc=float(va),selection_n_iter=nsel,final_n_iter=int(final.n_iter_[0]),converged=True,**metrics(y[m8],final.predict_proba(s.transform(x[m8]))[:,1]))
def main():
    reg=pd.read_parquet(OUT/"probe_regularization_registry.parquet"); old=pd.read_parquet(OUT/"b1_residual_subspace_probes.parquet");cfg=yaml.safe_load((ROOT/"configs/ptbxl_concept_tiers.yaml").read_text())
    b1=load("b1");folds=b1["strat_fold"];tr=folds<=7;te=folds==8;valid=reg[(reg.model=="b1")&reg.converged].set_index(["tier","concept"])
    tiers={c:"P1" for c in cfg["tier_p1_codes"]};tiers.update({c:"P2" for c in cfg["tier_p2_codes"]});tiers.update({c:"P3" for c in cfg["tier_p3_codes"]})
    specs=[("anchor",c,b1["anchor_labels"][:,i].astype(int)) for i,c in enumerate(cfg["all_anchor_codes"])];specs += [(tiers[c],c,b1["probe_labels"][:,i].astype(int)) for i,c in enumerate(cfg["all_probe_codes"])];specs=[q for q in specs if (q[0],q[1]) in valid.index]
    if len(specs)!=39:raise RuntimeError(f"eligible concepts {len(specs)} != 39")
    jobs=[];residual_by_target={}
    for target in TARGETS:
        d=load(target)
        if not np.array_equal(d["ecg_id"],b1["ecg_id"]):raise RuntimeError("record mismatch")
        sa=StandardScaler().fit(b1["z"][tr]);sb=StandardScaler().fit(d["z"][tr]);A=sa.transform(b1["z"]);B=sb.transform(d["z"]);ridge=Ridge(alpha=1.).fit(A[tr],B[tr]);R=B-ridge.predict(A);residual_by_target[target]=R
        X=np.concatenate([A,R],axis=1)
        for tier,concept,y in specs:jobs.append(joblib.delayed(fit_g2)(target,concept,tier,X,y,folds))
    rows=joblib.Parallel(n_jobs=4,backend="loky",verbose=10)(jobs);inc=pd.DataFrame(rows)
    base=reg[reg.converged][["model","concept","tier","auroc"]]
    b=base[base.model=="b1"][["concept","tier","auroc"]].rename(columns={"auroc":"b1_auroc"});comp=[]
    for target in TARGETS:
        t=base[base.model==target][["concept","tier","auroc"]].rename(columns={"auroc":"target_auroc"});r=old[old.target==target][["concept","tier","auroc"]].rename(columns={"auroc":"residual_auroc"});q=inc[inc.target==target].merge(b,on=["concept","tier"]).merge(t,on=["concept","tier"]).merge(r,on=["concept","tier"]);q["concat_b1_residual_auroc"]=q.auroc;q["target_minus_b1"]=q.target_auroc-q.b1_auroc;q["incremental_delta"]=q.concat_b1_residual_auroc-q.b1_auroc;q["residual_retention"]=(q.residual_auroc-.5)/(q.target_auroc-.5);q["recovery_denominator_stable"]=q.target_minus_b1.abs()>=.01;q["recovery_fraction_Q"]=np.where(q.recovery_denominator_stable,q.incremental_delta/q.target_minus_b1,np.nan);q["residual_denominator_stable"]=q.target_auroc>.55;comp.append(q)
    final=pd.concat(comp,ignore_index=True);atomic(final,OUT/"r3_geometry_incremental_per_concept.parquet");atomic(final[["target","concept","tier","residual_auroc","target_auroc","residual_retention","residual_denominator_stable"]],OUT/"r3_residual_retention_per_concept.parquet")
    summary=final.groupby(["target","tier"],as_index=False).agg(n=("concept","size"),b1_auroc=("b1_auroc","mean"),target_auroc=("target_auroc","mean"),residual_auroc=("residual_auroc","mean"),concat_auroc=("concat_b1_residual_auroc","mean"),incremental_delta=("incremental_delta","mean"),residual_retention=("residual_retention","mean"));summary.to_csv(OUT/"r3_geometry_incremental_tier_summary.csv",index=False)
    manifest=dict(status="complete",eligible_concepts=39,targets=TARGETS,expected_rows=117,observed_rows=len(final),fold8_used_for_alignment=False,fold8_used_for_hyperparameter_selection=False,encoders_frozen=True,all_probes_converged=bool(inc.converged.all()),explicit_invalid_rows=int((~inc.converged).sum()),registry_sha256=sha(OUT/"probe_regularization_registry.parquet"));(OUT/"r3_geometry_extension_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if len(final)!=117:raise RuntimeError("R3-G2 row reconciliation failed")
if __name__=="__main__":main()
