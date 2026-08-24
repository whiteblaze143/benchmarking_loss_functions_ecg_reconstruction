#!/usr/bin/env python3
"""Rigorous, resumable post-analysis for the compact RDB embedding study.

This stage never re-extracts activations. It reads completed, hash-bound feature
jobs, fits all preprocessing on train, freezes choices on validation, and uses
test only for final paired estimates. UMAP remains descriptive.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd


def connect(path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(path, timeout=120)
    c.row_factory = sqlite3.Row
    c.execute("pragma journal_mode=WAL")
    c.executescript("""
    CREATE TABLE IF NOT EXISTS analysis_runs(
      analysis_version TEXT PRIMARY KEY,status TEXT NOT NULL,started_at TEXT DEFAULT CURRENT_TIMESTAMP,
      completed_at TEXT,details_json TEXT);
    CREATE TABLE IF NOT EXISTS extended_results(
      job_id TEXT NOT NULL,outcome TEXT NOT NULL,predictor_set TEXT NOT NULL,metric TEXT NOT NULL,
      value REAL,ci_low REAL,ci_high REAL,n INTEGER,details_json TEXT,
      PRIMARY KEY(job_id,outcome,predictor_set,metric));
    CREATE TABLE IF NOT EXISTS paired_comparisons(
      job_a TEXT NOT NULL,job_b TEXT NOT NULL,outcome TEXT NOT NULL,metric TEXT NOT NULL,
      delta_b_minus_a REAL,ci_low REAL,ci_high REAL,p_value REAL,details_json TEXT,
      PRIMARY KEY(job_a,job_b,outcome,metric));
    CREATE TABLE IF NOT EXISTS checkpoint_similarity(
      job_a TEXT NOT NULL,job_b TEXT NOT NULL,split TEXT NOT NULL,metric TEXT NOT NULL,
      value REAL,details_json TEXT,PRIMARY KEY(job_a,job_b,split,metric));
    CREATE TABLE IF NOT EXISTS umap_stability(
      job_id TEXT NOT NULL,split TEXT NOT NULL,seed_a INTEGER NOT NULL,neighbors_a INTEGER NOT NULL,
      seed_b INTEGER NOT NULL,neighbors_b INTEGER NOT NULL,metric TEXT NOT NULL,value REAL,
      PRIMARY KEY(job_id,split,seed_a,neighbors_a,seed_b,neighbors_b,metric));
    CREATE TABLE IF NOT EXISTS multiplicity_results(
      family TEXT NOT NULL,hypothesis TEXT NOT NULL,raw_p REAL,adjusted_p REAL,method TEXT NOT NULL,
      details_json TEXT,PRIMARY KEY(family,hypothesis,method));
    """)
    return c


def unpack(blob, dim):
    x = np.frombuffer(zlib.decompress(blob), dtype=np.float16).astype(np.float32)
    if x.size != dim: raise ValueError("feature dimension mismatch")
    return x


def load_job(c, jid):
    rows = c.execute("""select f.record_id,f.dim,f.feature_blob,r.* from features f
      join records r using(record_id) where f.job_id=? order by r.split,f.record_id""", (jid,)).fetchall()
    m = pd.DataFrame([{k:r[k] for k in r.keys() if k not in ("feature_blob","dim")} for r in rows])
    return m, np.stack([unpack(r["feature_blob"], r["dim"]) for r in rows])


def metric_values(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    return {"auroc":roc_auc_score(y,p), "auprc":average_precision_score(y,p),
            "brier":brier_score_loss(y,p), "log_loss":log_loss(y,p,labels=[0,1])}


def stratified_indices(y, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == k), np.sum(y == k), True) for k in np.unique(y)])


def ci(values):
    a=np.asarray(values,float); return float(np.quantile(a,.025)),float(np.quantile(a,.975))


def calibrate(logit_val, y_val, logit_all):
    from sklearn.linear_model import LogisticRegression
    z=np.asarray(logit_val).reshape(-1,1); za=np.asarray(logit_all).reshape(-1,1)
    model=LogisticRegression(C=1e6,solver="lbfgs",max_iter=3000).fit(z,y_val)
    return model.predict_proba(za)[:,1],float(model.intercept_[0]),float(model.coef_[0,0])


def fit_binary(meta, x):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr=meta.split.eq("train").to_numpy(); va=meta.split.eq("val").to_numpy(); te=meta.split.eq("test").to_numpy()
    y=np.isin(meta.released_rhythm.to_numpy(),["AF","AFIB"]).astype(int)
    sx=StandardScaler().fit(x[tr]); xs=sx.transform(x)
    pca=PCA(n_components=8,random_state=42).fit(xs[tr]); xp=pca.transform(xs)
    raw=meta[["heart_rate_bpm","signal_rms","spectral_entropy"]].to_numpy(float)
    imp=SimpleImputer(strategy="median").fit(raw[tr]); raw=imp.transform(raw)
    raw=StandardScaler().fit(raw[tr]).transform(raw)
    arrays={"waveform":[(3,raw)], "latent":[(d,xp[:,:d]) for d in (1,2,4,8)],
            "waveform_plus_latent":[(d,np.c_[raw,xp[:,:d]]) for d in (1,2,4,8)]}
    out={}
    for name, choices in arrays.items():
        candidates=[]
        for d,a in choices:
            for C in (.01,.1,1,10):
                q=LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=3000).fit(a[tr],y[tr])
                loss=metric_values(y[va],q.predict_proba(a[va])[:,1])["log_loss"]
                candidates.append((loss,d,C,a))
        _,d,C,a=min(candidates,key=lambda v:(v[0],v[1],v[2]))
        q=LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=3000).fit(a[tr],y[tr])
        lv=q.decision_function(a[va]); la=q.decision_function(a)
        p,cal_i,cal_s=calibrate(lv,y[va],la)
        out[name]={"p":p,"y":y,"test":te,"dims":d,"C":C,"calibration_intercept":cal_i,
                   "calibration_slope":cal_s}
    return out


def calibration_on_test(y,p):
    import statsmodels.api as sm
    z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))
    fit=sm.Logit(y,sm.add_constant(z)).fit(disp=0)
    return float(fit.params[0]),float(fit.params[1])


def store_binary(c,jid,fits,nboot,seed):
    rng=np.random.default_rng(seed)
    for name,o in fits.items():
        y=o["y"][o["test"]]; p=o["p"][o["test"]]; vals=metric_values(y,p)
        boots={k:[] for k in vals}
        for _ in range(nboot):
            ix=stratified_indices(y,rng)
            for k,v in metric_values(y[ix],p[ix]).items(): boots[k].append(v)
        details={"dims":o["dims"],"C":o["C"],"validation_calibration_intercept":o["calibration_intercept"],
                 "validation_calibration_slope":o["calibration_slope"],"bootstrap":"stratified_cached_record_id"}
        for k,v in vals.items():
            lo,hi=ci(boots[k]); c.execute("insert or replace into extended_results values(?,?,?,?,?,?,?,?,?)",
              (jid,"AF_AFIB_code_membership",name,k,float(v),lo,hi,len(y),json.dumps(details)))
        try:
            a,b=calibration_on_test(y,p)
            for k,v in (("calibration_intercept",a),("calibration_slope",b)):
                c.execute("insert or replace into extended_results values(?,?,?,?,?,?,?,?,?)",
                  (jid,"AF_AFIB_code_membership",name,k,v,None,None,len(y),json.dumps(details)))
        except Exception as e:
            details["test_calibration_error"]=str(e)
        q25,q75=np.quantile(p,[.25,.75])
        c.execute("insert or replace into extended_results values(?,?,?,?,?,?,?,?,?)",
          (jid,"AF_AFIB_code_membership",name,"predicted_risk_IQR_contrast",float(q75-q25),None,None,len(y),json.dumps(details)))


def regression_metrics(y,p):
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y=np.asarray(y,float);p=np.asarray(p,float);d=p-y
    sy=np.std(y);sp=np.std(p);rho=np.corrcoef(y,p)[0,1] if sy>0 and sp>0 else np.nan
    ccc=2*rho*sy*sp/(sy**2+sp**2+(np.mean(y)-np.mean(p))**2) if np.isfinite(rho) else np.nan
    return {"mae_ms":mean_absolute_error(y,p),"median_absolute_error_ms":np.median(np.abs(d)),
            "rmse_ms":math.sqrt(mean_squared_error(y,p)),"r2":r2_score(y,p),
            "pearson":rho,"spearman":spearmanr(y,p).statistic if sp>0 else np.nan,"lin_ccc":ccc,
            "calibration_intercept_ms":float(np.polyfit(p,y,1)[1]) if sp>0 else np.nan,
            "calibration_slope":float(np.polyfit(p,y,1)[0]) if sp>0 else np.nan,
            "bland_altman_bias_ms":float(np.mean(d)),
            "bland_altman_loa_low_ms":float(np.mean(d)-1.96*np.std(d,ddof=1)),
            "bland_altman_loa_high_ms":float(np.mean(d)+1.96*np.std(d,ddof=1))}


def fit_and_store_regression(c,jid,meta,x,nboot,seed):
    """Annotation-derived QRS-onset-to-T-offset probe; explicitly not QT/QTc."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    tr=meta.split.eq("train").to_numpy();va=meta.split.eq("val").to_numpy();te=meta.split.eq("test").to_numpy()
    y=meta.qrson_toff_ms.to_numpy(float);ok=np.isfinite(y)
    sx=StandardScaler().fit(x[tr]);xs=sx.transform(x)
    pca=PCA(n_components=8,random_state=42).fit(xs[tr]);xp=pca.transform(xs)
    raw=meta[["heart_rate_bpm","signal_rms","spectral_entropy"]].to_numpy(float)
    imp=SimpleImputer(strategy="median").fit(raw[tr]);raw=imp.transform(raw);raw=StandardScaler().fit(raw[tr]).transform(raw)
    arrays={"mean_only":[(0,np.ones((len(meta),1)))],"waveform":[(3,raw)],
            "latent":[(d,xp[:,:d]) for d in (1,2,4,8)],
            "waveform_plus_latent":[(d,np.c_[raw,xp[:,:d]]) for d in (1,2,4,8)]}
    outputs={};rng=np.random.default_rng(seed)
    for name,choices in arrays.items():
        cand=[]
        for d,a in choices:
          for alpha in ((1e6,) if name=="mean_only" else (.01,.1,1,10,100)):
            q=Ridge(alpha=alpha).fit(a[tr&ok],y[tr&ok]);err=np.mean(np.abs(y[va&ok]-q.predict(a[va&ok])))
            cand.append((err,d,alpha,a))
        _,d,alpha,a=min(cand,key=lambda z:(z[0],z[1],z[2]))
        q=Ridge(alpha=alpha).fit(a[(tr|va)&ok],y[(tr|va)&ok]);pred=q.predict(a[te&ok]);truth=y[te&ok]
        vals=regression_metrics(truth,pred);boots={k:[] for k in vals}
        for _ in range(nboot):
            ix=rng.choice(len(truth),len(truth),True);v=regression_metrics(truth[ix],pred[ix])
            for k,z in v.items():boots[k].append(z)
        details={"dims":d,"alpha":alpha,"n":len(truth),"endpoint":"annotation-derived QRSon-Toff",
                 "not_clinical_QT_or_QTc":True,"bootstrap":"cached_record_id"}
        for k,v in vals.items():
            finite=np.asarray(boots[k]);finite=finite[np.isfinite(finite)];lo,hi=ci(finite) if len(finite) else (None,None)
            c.execute("insert or replace into extended_results values(?,?,?,?,?,?,?,?,?)",
              (jid,"annotation_QRSon_Toff_ms",name,k,float(v) if np.isfinite(v) else None,lo,hi,len(truth),json.dumps(details)))
        outputs[name]={"record_id":meta.loc[te&ok,"record_id"].to_numpy(),"truth":truth,"pred":pred}
    return outputs


def quality_control_or(c,jid,meta,x,oracle_path):
    """OR for measured reconstruction failure per SD latent outlier score."""
    import statsmodels.api as sm
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    if not oracle_path.exists():return None
    job=c.execute("select model_id,checkpoint_sha256 from jobs where job_id=?",(jid,)).fetchone()
    oc=sqlite3.connect(oracle_path);oc.row_factory=sqlite3.Row
    ev=oc.execute("select evaluation_id from evaluations where model_id=? and checkpoint_sha256=? and status='complete'",
                  (job["model_id"],job["checkpoint_sha256"])).fetchone()
    if not ev:oc.close();return None
    rows=oc.execute("""select record_id,pearson_mean,mse_mean from record_role_signal_metrics
      where evaluation_id=? and lead_role='primary_missing_precordial'""",(ev[0],)).fetchall();oc.close()
    q=meta.copy();q["ix"]=np.arange(len(q));q=q.merge(pd.DataFrame([dict(r) for r in rows]),on="record_id",how="inner")
    tr=q.split.eq("train").to_numpy();va=q.split.eq("val").to_numpy();te=q.split.eq("test").to_numpy()
    pearson_cut=float(np.quantile(q.loc[va,"pearson_mean"],.10));mse_cut=float(np.quantile(q.loc[va,"mse_mean"],.90))
    failure=((q.pearson_mean<pearson_cut)|(q.mse_mean>mse_cut)).astype(int).to_numpy()
    sx=StandardScaler().fit(x[q.loc[tr,"ix"]]);z=sx.transform(x[q.ix])
    pca=PCA(n_components=8,random_state=42).fit(z[tr]);zp=pca.transform(z)
    cov=LedoitWolf().fit(zp[tr]);score=np.sqrt(np.maximum(cov.mahalanobis(zp),0))
    score=(score-np.mean(score[tr]))/np.std(score[tr]);fit=sm.Logit(failure[te],sm.add_constant(score[te])).fit(disp=0)
    beta,se,pv=fit.params[1],fit.bse[1],fit.pvalues[1];orr=math.exp(beta);lo=math.exp(beta-1.96*se);hi=math.exp(beta+1.96*se)
    details={"oracle_evaluation_id":ev[0],"checkpoint_sha256_matched":True,"lead_role":"primary_missing_precordial",
             "validation_pearson_p10":pearson_cut,"validation_mse_p90":mse_cut,"failure_rule":"pearson<p10 OR mse>p90",
             "PCA_dims":8,"covariance":"LedoitWolf","not_patient_prognosis":True}
    c.execute("insert or replace into odds_ratios values(?,?,?,?,?,?,?,?,?,?,?)",
      (jid,"reconstruction_failure","latent_outlier","score_per_sd",orr,lo,hi,float(pv),int(failure[te].sum()),int(te.sum()),json.dumps(details)))
    q25,q75=np.quantile(score[te],[.25,.75]);risk=lambda s:1/(1+np.exp(-(fit.params[0]+fit.params[1]*s)))
    c.execute("insert or replace into extended_results values(?,?,?,?,?,?,?,?,?)",
      (jid,"reconstruction_failure","latent_outlier","predicted_risk_IQR_contrast",float(risk(q75)-risk(q25)),None,None,int(te.sum()),json.dumps(details)))
    return {"p_value":float(pv),"or":orr,"ci_low":lo,"ci_high":hi}


def paired_delta(c,ja,jb,fa,fb,nboot,seed):
    oa,ob=fa["latent"],fb["latent"]
    ma,mb=load_job(c,ja)[0],load_job(c,jb)[0]
    ia=pd.DataFrame({"record_id":ma.record_id,"split":ma.split,"y":oa["y"],"pa":oa["p"]})
    ib=pd.DataFrame({"record_id":mb.record_id,"pb":ob["p"]})
    z=ia.merge(ib,on="record_id"); z=z[z.split.eq("test")]
    y=z.y.to_numpy(int);pa=z.pa.to_numpy(float);pb=z.pb.to_numpy(float)
    va=metric_values(y,pa);vb=metric_values(y,pb);rng=np.random.default_rng(seed)
    boots={k:[] for k in va}
    for _ in range(nboot):
        ix=stratified_indices(y,rng);aa=metric_values(y[ix],pa[ix]);bb=metric_values(y[ix],pb[ix])
        for k in va:boots[k].append(bb[k]-aa[k])
    for k in va:
        d=vb[k]-va[k];lo,hi=ci(boots[k]);arr=np.asarray(boots[k]);p=min(1.,2*min(np.mean(arr<=0),np.mean(arr>=0)))
        c.execute("insert or replace into paired_comparisons values(?,?,?,?,?,?,?,?,?)",
          (ja,jb,"AF_AFIB_code_membership",k,float(d),lo,hi,float(p),json.dumps({"paired_record_ids":len(y),"predictor_set":"latent"})))


def linear_cka(a,b):
    a=a-a.mean(0);b=b-b.mean(0)
    cross=np.linalg.norm(a.T@b,"fro")**2
    return float(cross/(np.linalg.norm(a.T@a,"fro")*np.linalg.norm(b.T@b,"fro")+1e-12))


def rsa(a,b):
    from scipy.stats import spearmanr
    from sklearn.metrics import pairwise_distances
    da=pairwise_distances(a,metric="cosine");db=pairwise_distances(b,metric="cosine")
    ix=np.triu_indices_from(da,1);return float(spearmanr(da[ix],db[ix]).statistic)


def neighbor_overlap(a,b,k=10):
    from sklearn.neighbors import NearestNeighbors
    na=NearestNeighbors(n_neighbors=k+1,metric="cosine").fit(a).kneighbors(return_distance=False)[:,1:]
    nb=NearestNeighbors(n_neighbors=k+1,metric="cosine").fit(b).kneighbors(return_distance=False)[:,1:]
    return float(np.mean([len(set(x)&set(y))/len(set(x)|set(y)) for x,y in zip(na,nb)]))


def checkpoint_pairs(c,jobs):
    from sklearn.preprocessing import StandardScaler
    loaded={j:load_job(c,j) for j in jobs}
    for ai,ja in enumerate(jobs):
      for jb in jobs[ai+1:]:
        ma,xa=loaded[ja];mb,xb=loaded[jb]
        da=pd.DataFrame({"record_id":ma.record_id,"split":ma.split,"i":np.arange(len(ma))})
        db=pd.DataFrame({"record_id":mb.record_id,"j":np.arange(len(mb))})
        q=da.merge(db,on="record_id");q=q[q.split.eq("test")]
        a=StandardScaler().fit_transform(xa[q.i]);b=StandardScaler().fit_transform(xb[q.j])
        for metric,value in (("linear_CKA",linear_cka(a,b)),("distance_RSA_spearman",rsa(a,b)),("knn10_jaccard",neighbor_overlap(a,b))):
            c.execute("insert or replace into checkpoint_similarity values(?,?,?,?,?,?)",
              (ja,jb,"test",metric,value,json.dumps({"n":len(q),"same_record_ids":True})))


def umap_stability(c,jid):
    from scipy.linalg import orthogonal_procrustes
    rows=c.execute("select * from projections where job_id=? and method='umap_train_transform'",(jid,)).fetchall()
    if not rows:return
    d=pd.DataFrame([dict(r) for r in rows]);meta=pd.read_sql_query("select record_id,split from records",c)
    d=d.merge(meta,on="record_id")
    configs=sorted(set(zip(d.seed,d.neighbors)))
    for split in ("train","val","test"):
      maps={(s,n):g.sort_values("record_id")[["x","y"]].to_numpy() for (s,n),g in d[d.split.eq(split)].groupby(["seed","neighbors"])}
      for i,a in enumerate(configs):
       for b in configs[i+1:]:
        if a not in maps or b not in maps:continue
        x=maps[a]-maps[a].mean(0);y=maps[b]-maps[b].mean(0);r,_=orthogonal_procrustes(y,x);ya=y@r
        score=float(1-np.linalg.norm(x-ya)/(np.linalg.norm(x)+1e-12))
        c.execute("insert or replace into umap_stability values(?,?,?,?,?,?,?,?)",(jid,split,a[0],a[1],b[0],b[1],"procrustes_similarity",score))


def adjust_p(c, qc_results):
    # Prespecified family: AF/AFIB-coded delta AUROC and Brier, annotation-derived
    # QRSon-Toff delta MAE, and reconstruction-failure outlier-score OR.
    rows=c.execute("""select job_a,job_b,outcome,metric,p_value from paired_comparisons
      where p_value is not null and ((outcome='AF_AFIB_code_membership' and metric in ('auroc','brier'))
      or (outcome='annotation_QRSon_Toff_ms' and metric='mae_ms'))""").fetchall()
    entries=[(f'{r["outcome"]}:{r["metric"]}:{r["job_a"]}:{r["job_b"]}',float(r["p_value"])) for r in rows]
    entries += [(f'reconstruction_failure:latent_outlier:{jid}',float(v["p_value"])) for jid,v in qc_results.items() if v]
    if not entries:return
    from statsmodels.stats.multitest import multipletests
    ps=np.array([v for _,v in entries]);adj=multipletests(ps,method="holm")[1]
    for (h,raw),p in zip(entries,adj):
        c.execute("insert or replace into multiplicity_results values(?,?,?,?,?,?)",
          ("prespecified_confirmatory",h,raw,float(p),"Holm",json.dumps({"family_size":len(entries)})))


def paired_regression_delta(c,ja,jb,ra,rb,nboot,seed):
    a=ra["latent"];b=rb["latent"]
    da=pd.DataFrame({"record_id":a["record_id"],"truth_a":a["truth"],"pa":a["pred"]})
    db=pd.DataFrame({"record_id":b["record_id"],"truth_b":b["truth"],"pb":b["pred"]})
    z=da.merge(db,on="record_id");
    if not np.allclose(z.truth_a,z.truth_b):raise RuntimeError("paired regression target mismatch")
    y=z.truth_a.to_numpy(float);pa=z.pa.to_numpy(float);pb=z.pb.to_numpy(float)
    va=regression_metrics(y,pa);vb=regression_metrics(y,pb);rng=np.random.default_rng(seed)
    for metric in ("mae_ms","rmse_ms","spearman","lin_ccc"):
        boot=[]
        for _ in range(nboot):
            ix=rng.choice(len(y),len(y),True);aa=regression_metrics(y[ix],pa[ix]);bb=regression_metrics(y[ix],pb[ix]);boot.append(bb[metric]-aa[metric])
        arr=np.asarray(boot);arr=arr[np.isfinite(arr)];lo,hi=ci(arr);p=min(1.,2*min(np.mean(arr<=0),np.mean(arr>=0)))
        c.execute("insert or replace into paired_comparisons values(?,?,?,?,?,?,?,?,?)",
          (ja,jb,"annotation_QRSon_Toff_ms",metric,float(vb[metric]-va[metric]),lo,hi,float(p),
           json.dumps({"paired_record_ids":len(y),"predictor_set":"latent","not_clinical_QT_or_QTc":True})))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--db",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=2000);p.add_argument("--seed",type=int,default=240824)
    p.add_argument("--oracle-db",type=Path,default=ROOT/"results/ecgaim_rdb_oracle/ecgaim_rdb_oracle.sqlite")
    a=p.parse_args();c=connect(a.db);version="rigorous_v1"
    c.execute("insert or replace into analysis_runs(analysis_version,status,details_json) values(?,'running',?)",
              (version,json.dumps({"bootstrap":a.bootstrap,"seed":a.seed,"umap_is_descriptive":True})));c.commit()
    try:
        jobs=[r["job_id"] for r in c.execute("select job_id from jobs where status='complete' order by rowid")]
        primary=[r["job_id"] for r in c.execute("select job_id from jobs where status='complete' and split_scope='all' order by rowid")]
        fits={};regression={};qc={}
        for j in primary:
            m,x=load_job(c,j);fits[j]=fit_binary(m,x);store_binary(c,j,fits[j],a.bootstrap,a.seed)
            regression[j]=fit_and_store_regression(c,j,m,x,a.bootstrap,a.seed+2)
            qc[j]=quality_control_or(c,j,m,x,a.oracle_db);umap_stability(c,j);c.commit()
        if len(primary)>=2:
            paired_delta(c,primary[0],primary[1],fits[primary[0]],fits[primary[1]],a.bootstrap,a.seed+1)
            paired_regression_delta(c,primary[0],primary[1],regression[primary[0]],regression[primary[1]],a.bootstrap,a.seed+3)
        checkpoint_pairs(c,jobs);adjust_p(c,qc)
        c.execute("update analysis_runs set status='complete',completed_at=CURRENT_TIMESTAMP where analysis_version=?",(version,));c.commit()
        print(json.dumps({"event":"rigorous_analysis_complete","jobs":len(jobs),"primary_jobs":len(primary)}))
    except Exception as e:
        c.execute("update analysis_runs set status='error',details_json=? where analysis_version=?",(json.dumps({"error":repr(e)}),version));c.commit();raise
    finally:c.close()


if __name__ == "__main__": main()
