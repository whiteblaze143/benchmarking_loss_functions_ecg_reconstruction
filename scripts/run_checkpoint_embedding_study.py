#!/usr/bin/env python3
"""Resumable, compact, one-worker RDB checkpoint representation study.

The runner stores compressed pooled activations and derived summaries in one
SQLite database. It never writes wide CSVs and never uses UMAP coordinates for
inferential models.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import signal
import sqlite3
import sys
import time
import traceback
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(ROOT / "book") not in sys.path: sys.path.insert(0, str(ROOT / "book"))

import numpy as np
import pandas as pd
import torch

from book.checkpoint_embeddings import (
    _forward, _pool_activation, _tensor_from_output, load_verified_model,
)

PRIMARY_MODELS = ["factorial_ecg_aim_1000000_s42", "factorial_ecg_aim_1011011_s42"]
PANEL_MODELS = [
    "factorial_ecg_aim_1000013_s42", "factorial_ecg_aim_1001004_s42",
    "factorial_ecg_aim_1001014_s42", "factorial_ecg_aim_1010003_s42",
    "factorial_ecg_aim_1010010_s42", "factorial_ecg_aim_1010013_s42",
    "factorial_ecg_aim_1010014_s42", "factorial_ecg_aim_1011002_s42",
    "factorial_ecg_aim_1011003_s42",
]
POOLING_VERSION = "feature_mean_std_max_v1"
STOP = False


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def manifest_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        s = p.stat(); h.update(f"{p.name}\0{s.st_size}\0{s.st_mtime_ns}\n".encode())
    return h.hexdigest()


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 1024**2
    return 0.0


def free_disk_gib(path: Path) -> float:
    s = os.statvfs(path); return s.f_bavail * s.f_frsize / 1024**3


def wait_resources(path: Path, min_ram: float, min_disk: float, stop_file: Path):
    global STOP
    while available_gib() < min_ram or free_disk_gib(path) < min_disk:
        print(json.dumps({"event":"resource_wait","ram_gib":available_gib(),
                          "disk_gib":free_disk_gib(path),"at":now()}), flush=True)
        if STOP or stop_file.exists(): raise InterruptedError("STOP requested during resource wait")
        time.sleep(60)


@contextlib.contextmanager
def single_worker(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError(f"another worker holds {lock_path}") from e
        yield


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=60)
    c.row_factory = sqlite3.Row
    c.execute("pragma journal_mode=WAL"); c.execute("pragma synchronous=NORMAL")
    c.executescript("""
    CREATE TABLE IF NOT EXISTS study_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS jobs(
      job_id TEXT PRIMARY KEY,model_id TEXT NOT NULL,checkpoint_sha256 TEXT NOT NULL,
      architecture TEXT NOT NULL,layer TEXT NOT NULL,pooling_version TEXT NOT NULL,
      split_scope TEXT NOT NULL,record_manifest_sha256 TEXT NOT NULL,
      expected_records INTEGER NOT NULL,completed_records INTEGER NOT NULL DEFAULT 0,
      feature_dim INTEGER,status TEXT NOT NULL,started_at TEXT,completed_at TEXT,error TEXT);
    CREATE TABLE IF NOT EXISTS records(
      record_id TEXT PRIMARY KEY,patient_id TEXT,released_rhythm TEXT,canonical_rhythm TEXT,
      source_dataset TEXT,split TEXT NOT NULL,heart_rate_bpm REAL,signal_rms REAL,
      spectral_entropy REAL,rr_cv REAL,qrson_toff_ms REAL);
    CREATE TABLE IF NOT EXISTS features(
      job_id TEXT NOT NULL,record_id TEXT NOT NULL,dim INTEGER NOT NULL,
      dtype TEXT NOT NULL,compression TEXT NOT NULL,feature_blob BLOB NOT NULL,
      PRIMARY KEY(job_id,record_id),FOREIGN KEY(job_id) REFERENCES jobs(job_id));
    CREATE TABLE IF NOT EXISTS projections(
      job_id TEXT NOT NULL,method TEXT NOT NULL,seed INTEGER NOT NULL,neighbors INTEGER NOT NULL,
      record_id TEXT NOT NULL,x REAL NOT NULL,y REAL NOT NULL,
      PRIMARY KEY(job_id,method,seed,neighbors,record_id));
    CREATE TABLE IF NOT EXISTS diagnostics(
      job_id TEXT NOT NULL,analysis TEXT NOT NULL,split TEXT NOT NULL,seed INTEGER,
      neighbors INTEGER,metric TEXT NOT NULL,value REAL,details_json TEXT,
      PRIMARY KEY(job_id,analysis,split,seed,neighbors,metric));
    CREATE TABLE IF NOT EXISTS probe_results(
      job_id TEXT NOT NULL,outcome TEXT NOT NULL,predictor_set TEXT NOT NULL,
      split TEXT NOT NULL,metric TEXT NOT NULL,value REAL,ci_low REAL,ci_high REAL,
      details_json TEXT,PRIMARY KEY(job_id,outcome,predictor_set,split,metric));
    CREATE TABLE IF NOT EXISTS odds_ratios(
      job_id TEXT NOT NULL,outcome TEXT NOT NULL,predictor_set TEXT NOT NULL,
      term TEXT NOT NULL,odds_ratio REAL,ci_low REAL,ci_high REAL,p_value REAL,
      events INTEGER,total INTEGER,details_json TEXT,
      PRIMARY KEY(job_id,outcome,predictor_set,term));
    """)
    return c


def intervals(seg: np.ndarray, cls: int) -> list[tuple[int,int]]:
    z = np.asarray(seg == cls, dtype=np.int8)
    edge = np.diff(np.pad(z, (1,1)))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def record_metadata(example: dict) -> dict:
    y = example["waveform"].float().numpy(); seg = example["segmentation"][1].numpy()
    qrs = intervals(seg, 2); tw = intervals(seg, 3)
    rr = np.diff([a for a,_ in qrs]) / 500 if len(qrs) > 1 else np.array([])
    qrston = []
    for i,(qa,_) in enumerate(qrs):
        next_q = qrs[i+1][0] if i+1 < len(qrs) else len(seg)
        candidate = next((tb for ta,tb in tw if ta >= qa and tb <= next_q), None)
        if candidate is not None: qrston.append((candidate-qa)*2.0)
    observed = y[[0,1,7]]
    power = np.abs(np.fft.rfft(observed, axis=-1))**2
    prob = power / np.maximum(power.sum(axis=-1,keepdims=True), 1e-12)
    entropy = -(prob*np.log(np.maximum(prob,1e-12))).sum(axis=-1)/np.log(prob.shape[-1])
    return {
      "record_id":str(example["record_id"]),"patient_id":str(example.get("patient_id","unknown")),
      "released_rhythm":str(example.get("released_rhythm","unknown")),
      "canonical_rhythm":str(example.get("canonical_rhythm","unknown")),
      "source_dataset":str(example.get("source_dataset","unknown")),"split":str(example.get("split","unknown")),
      "heart_rate_bpm":float(60/np.median(rr)) if len(rr) and np.median(rr)>0 else None,
      "signal_rms":float(np.sqrt(np.mean(observed**2))),"spectral_entropy":float(np.mean(entropy)),
      "rr_cv":float(np.std(rr)/np.mean(rr)) if len(rr)>1 and np.mean(rr)>0 else None,
      "qrson_toff_ms":float(np.median(qrston)) if qrston else None,
    }


def pack_feature(x: np.ndarray) -> tuple[int,bytes]:
    a = np.asarray(x,dtype=np.float16).reshape(-1)
    if not np.isfinite(a).all(): raise ValueError("non-finite pooled feature")
    return len(a), zlib.compress(a.tobytes(), level=6)


def unpack_feature(blob: bytes, dim: int) -> np.ndarray:
    a=np.frombuffer(zlib.decompress(blob),dtype=np.float16).astype(np.float32)
    if len(a)!=dim: raise ValueError("feature BLOB dimension mismatch")
    return a


def files_for_scope(data_dir: Path, scope: str) -> list[Path]:
    splits = ["train","val","test"] if scope == "all" else ["test"]
    return [p for s in splits for p in sorted((data_dir/s).glob("*.pt"))]


def extract_job(c, model_id: str, scope: str, data_dir: Path, layer: str,
                device: str, min_ram: float, min_disk: float, stop_file: Path,
                max_records: int = 0):
    global STOP
    files=files_for_scope(data_dir,scope)
    if max_records and len(files)>max_records:
      idx=np.linspace(0,len(files)-1,max_records,dtype=int);files=[files[i] for i in idx]
    mh=manifest_hash(files)
    wait_resources(c.execute("pragma database_list").fetchone()[2] and Path(c.execute("pragma database_list").fetchone()[2]).parent or ROOT,
                   min_ram,min_disk,stop_file)
    model,row,identity=load_verified_model(model_id,device)
    jid=hashlib.sha256(f"{identity['sha256']}|{mh}|{layer}|{POOLING_VERSION}".encode()).hexdigest()[:24]
    existing=c.execute("select * from jobs where job_id=?",(jid,)).fetchone()
    if existing and existing["status"]=="complete":
        print(json.dumps({"event":"job_skip_complete","job_id":jid,"model_id":model_id}),flush=True); return jid
    c.execute("""insert into jobs(job_id,model_id,checkpoint_sha256,architecture,layer,pooling_version,
      split_scope,record_manifest_sha256,expected_records,status,started_at)
      values(?,?,?,?,?,?,?,?,?,'running',?) on conflict(job_id) do update set status='running',error=null""",
      (jid,model_id,identity["sha256"],row["architecture"],layer,POOLING_VERSION,scope,mh,len(files),now()));c.commit()
    modules=dict(model.named_modules())
    if layer not in modules: raise KeyError(f"missing layer {layer}; candidates={[n for n in modules if 'encoder' in n][:50]}")
    captured=[]
    def hook(_m,_i,o):
        t=_tensor_from_output(o)
        if t is None: raise TypeError("hook emitted no tensor")
        captured.append(_pool_activation(t,row["architecture"]))
    handle=modules[layer].register_forward_hook(hook)
    try:
      for ix,p in enumerate(files,1):
        if STOP or stop_file.exists(): raise InterruptedError("STOP requested")
        rid=p.stem.replace("rdb_","")
        if c.execute("select 1 from features where job_id=? and record_id=?",(jid,rid)).fetchone(): continue
        if ix % 25 == 0: wait_resources(Path(c.execute("pragma database_list").fetchone()[2]).parent,min_ram,min_disk,stop_file)
        e=torch.load(p,map_location="cpu",weights_only=False); rid=str(e.get("record_id",rid)); y=e["waveform"].float().unsqueeze(0)
        captured.clear()
        with torch.inference_mode(): _forward(model,row["architecture"],y,device)
        if len(captured)!=1 or captured[0].shape[0]!=1: raise RuntimeError("unexpected hook capture count")
        dim,blob=pack_feature(captured[0][0]); meta=record_metadata(e)
        c.execute("""insert into records values(:record_id,:patient_id,:released_rhythm,:canonical_rhythm,
          :source_dataset,:split,:heart_rate_bpm,:signal_rms,:spectral_entropy,:rr_cv,:qrson_toff_ms)
          on conflict(record_id) do update set patient_id=excluded.patient_id""",meta)
        c.execute("insert into features values(?,?,?,'float16','zlib',?)",(jid,rid,dim,blob))
        count=c.execute("select count(*) from features where job_id=?",(jid,)).fetchone()[0]
        c.execute("update jobs set completed_records=?,feature_dim=? where job_id=?",(count,dim,jid));c.commit()
        if ix==1 or ix%25==0: print(json.dumps({"event":"extract_progress","model_id":model_id,"job_id":jid,"records":count,"expected":len(files),"ram_gib":available_gib(),"at":now()}),flush=True)
      count=c.execute("select count(*) from features where job_id=?",(jid,)).fetchone()[0]
      dims=c.execute("select count(distinct dim) from features where job_id=?",(jid,)).fetchone()[0]
      if count!=len(files) or dims!=1: raise RuntimeError(f"completion gate failed count={count}/{len(files)} dims={dims}")
      c.execute("update jobs set status='complete',completed_at=?,completed_records=? where job_id=?",(now(),count,jid));c.commit()
      print(json.dumps({"event":"job_complete","model_id":model_id,"job_id":jid,"records":count}),flush=True)
      return jid
    except Exception as e:
      status="interrupted" if isinstance(e,InterruptedError) else "error"
      c.execute("update jobs set status=?,error=? where job_id=?",(status,str(e),jid));c.commit();raise
    finally:
      handle.remove();del model


def load_matrix(c, jid: str):
    rows=c.execute("""select f.record_id,f.dim,f.feature_blob,r.* from features f
      join records r on r.record_id=f.record_id where f.job_id=? order by r.split,f.record_id""",(jid,)).fetchall()
    meta=pd.DataFrame([{k:r[k] for k in r.keys() if k not in ("feature_blob","dim")} for r in rows])
    x=np.stack([unpack_feature(r["feature_blob"],r["dim"]) for r in rows]);return meta,x


def metric_values(y,p):
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
    return {"auroc":roc_auc_score(y,p),"auprc":average_precision_score(y,p),
            "brier":brier_score_loss(y,p),"log_loss":log_loss(y,p,labels=[0,1])}


def bootstrap_metrics(y,p,n=2000,seed=42):
    rng=np.random.default_rng(seed); vals={k:[] for k in metric_values(y,p)}
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    for _ in range(n):
      idx=np.r_[rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]
      for k,v in metric_values(y[idx],p[idx]).items(): vals[k].append(v)
    return {k:(float(np.quantile(v,.025)),float(np.quantile(v,.975))) for k,v in vals.items()}


def analyze_job(c,jid:str,bootstrap:int,umap_seeds:list[int],neighbors:list[int]):
    meta,x=load_matrix(c,jid); job=c.execute("select * from jobs where job_id=?",(jid,)).fetchone()
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression,LinearRegression
    from sklearn.metrics import balanced_accuracy_score,f1_score,roc_auc_score,mean_absolute_error,mean_squared_error,r2_score
    train=meta.split.eq("train").to_numpy();val=meta.split.eq("val").to_numpy();test=meta.split.eq("test").to_numpy()
    fitmask=train if train.any() else test
    scaler=StandardScaler().fit(x[fitmask]);xs=scaler.transform(x)
    pca=PCA(n_components=min(50,fitmask.sum()-1,x.shape[1]),random_state=42).fit(xs[fitmask]);xp=pca.transform(xs)
    # Descriptive UMAP: fit on training when available, then transform all splits.
    import umap
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score,pairwise_distances
    labels=meta.canonical_rhythm.to_numpy()
    for seed in umap_seeds:
      for nn in neighbors:
        nfit=int(fitmask.sum()); actual=min(nn,nfit-1)
        reducer=umap.UMAP(n_neighbors=actual,min_dist=.15,metric="cosine",random_state=seed,n_components=2)
        reducer.fit(xp[fitmask]);xy=reducer.transform(xp)
        c.executemany("insert or replace into projections values(?,?,?,?,?,?,?)",
          [(jid,"umap_train_transform",seed,nn,rid,float(a),float(b)) for rid,a,b in zip(meta.record_id,xy[:,0],xy[:,1])])
        for split,mask in [("train",train),("val",val),("test",test)]:
          if mask.sum()<5: continue
          k=min(10,max(1,(mask.sum()-1)//2));tw=float(trustworthiness(xp[mask],xy[mask],n_neighbors=k,metric="cosine"))
          n_labels=len(np.unique(labels[mask]))
          sil=float(silhouette_score(xp[mask],labels[mask],metric="cosine")) if 1<n_labels<mask.sum() else None
          d=pairwise_distances(xp[mask],metric="cosine");np.fill_diagonal(d,np.inf);nearest=d.argmin(1);agree=float(np.mean(labels[mask]==labels[mask][nearest]))
          for metric,value in [("trustworthiness",tw),("rhythm_silhouette",sil),("nearest_code_agreement",agree)]:
            c.execute("insert or replace into diagnostics values(?,?,?,?,?,?,?,?)",(jid,"umap",split,seed,nn,metric,value,None))
        c.commit()
    if min(train.sum(),val.sum(),test.sum()) < 20:
      c.execute("insert or replace into diagnostics values(?,?,?,?,?,?,?,?)",
                (jid,"probe_gate","all",None,None,"status",None,json.dumps({"reason":"fewer than 20 records in a required split"})));c.commit()
      print(json.dumps({"event":"analysis_smoke_complete","job_id":jid,"probe_gate":"skipped_small_split"}),flush=True);return
    y=np.isin(meta.released_rhythm.to_numpy(),["AF","AFIB"]).astype(int)
    if any(len(np.unique(y[m]))<2 for m in (train,val,test)):
      raise RuntimeError("binary outcome lacks both classes in a required split")
    if not set(labels[test]).issubset(set(labels[train|val])):
      raise RuntimeError("test rhythm code absent from train/validation")
    base_cols=["heart_rate_bpm","signal_rms","spectral_entropy"]
    xb=meta[base_cols].to_numpy(float);imp=SimpleImputer(strategy="median").fit(xb[train]);xb=imp.transform(xb);bs=StandardScaler().fit(xb[train]);xb=bs.transform(xb)
    candidates=[]
    for name,arr,dims in [("waveform",xb,[3]),("latent",xp,[1,2,4,8]),("waveform_plus_latent",None,[1,2,4,8])]:
      for d in dims:
        xx=xb if name=="waveform" else xp[:,:d] if name=="latent" else np.c_[xb,xp[:,:d]]
        for C in [.01,.1,1,10]:
          m=LogisticRegression(C=C,max_iter=3000,class_weight="balanced",solver="liblinear").fit(xx[train],y[train])
          pv=m.predict_proba(xx[val])[:,1];loss=metric_values(y[val],pv)["log_loss"]
          candidates.append((name,d,C,loss,xx))
    chosen={}
    for name in ("waveform","latent","waveform_plus_latent"):
      cand=min((v for v in candidates if v[0]==name),key=lambda z:(z[3],z[1],z[2]));chosen[name]=cand
      _,d,C,_,xx=cand;m=LogisticRegression(C=C,max_iter=3000,class_weight="balanced",solver="liblinear").fit(xx[train|val],y[train|val]);p=m.predict_proba(xx[test])[:,1]
      vals=metric_values(y[test],p);cis=bootstrap_metrics(y[test],p,bootstrap)
      for metric,value in vals.items(): c.execute("insert or replace into probe_results values(?,?,?,?,?,?,?,?,?)",(jid,"AF_AFIB_code_membership",name,"test",metric,float(value),cis[metric][0],cis[metric][1],json.dumps({"dims":d,"C":C})))
      # Exploratory diagnostic-score OR per SD on test.
      score=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1));score=(score-score.mean())/score.std()
      try:
        import statsmodels.api as sm
        fit=sm.Logit(y[test],sm.add_constant(score)).fit(disp=0);beta,se,pv=fit.params[1],fit.bse[1],fit.pvalues[1]
        c.execute("insert or replace into odds_ratios values(?,?,?,?,?,?,?,?,?,?,?)",(jid,"AF_AFIB_code_membership",name,"score_per_sd",math.exp(beta),math.exp(beta-1.96*se),math.exp(beta+1.96*se),float(pv),int(y[test].sum()),int(test.sum()),json.dumps({"cross_sectional_code_association":True})))
      except Exception as e: print(json.dumps({"event":"or_unavailable","job_id":jid,"predictor":name,"error":str(e)}),flush=True)
    # Multiclass secondary probe on the selected latent dimension.
    _,d,C,_,_=chosen["latent"];mc=LogisticRegression(C=C,max_iter=3000,class_weight="balanced",solver="lbfgs").fit(xp[train|val,:d],labels[train|val]);pred=mc.predict(xp[test,:d]);proba=mc.predict_proba(xp[test,:d])
    for metric,value in [("balanced_accuracy",balanced_accuracy_score(labels[test],pred)),("macro_f1",f1_score(labels[test],pred,average="macro")),("macro_ovr_auroc",roc_auc_score(labels[test],proba,multi_class="ovr",average="macro",labels=mc.classes_))]:
      c.execute("insert or replace into probe_results values(?,?,?,?,?,?,?,?,?)",(jid,"eight_code_rhythm","latent","test",metric,float(value),None,None,json.dumps({"dims":d,"C":C,"classes":mc.classes_.tolist()})))
    # Annotation-derived QRSon-Toff regression, explicitly not clinical QT.
    target=meta.qrson_toff_ms.to_numpy(float);ok=np.isfinite(target)
    if (train&ok).sum()>20 and (test&ok).sum()>10:
      best=None
      for d in [1,2,4,8]:
        for alpha in [.01,.1,1,10,100]:
          from sklearn.linear_model import Ridge
          m=Ridge(alpha=alpha).fit(xp[train&ok,:d],target[train&ok]);mae=mean_absolute_error(target[val&ok],m.predict(xp[val&ok,:d]))
          if best is None or (mae,d)<(best[0],best[1]):best=(mae,d,alpha)
      _,d,alpha=best;from sklearn.linear_model import Ridge;m=Ridge(alpha=alpha).fit(xp[(train|val)&ok,:d],target[(train|val)&ok]);pred=m.predict(xp[test&ok,:d]);truth=target[test&ok]
      vals={"mae_ms":mean_absolute_error(truth,pred),"rmse_ms":math.sqrt(mean_squared_error(truth,pred)),"r2":r2_score(truth,pred),"pearson":np.corrcoef(truth,pred)[0,1]}
      for metric,value in vals.items():c.execute("insert or replace into probe_results values(?,?,?,?,?,?,?,?,?)",(jid,"annotation_QRSon_Toff_ms","latent","test",metric,float(value),None,None,json.dumps({"dims":d,"alpha":alpha,"n":len(truth),"not_clinical_QT":True})))
    c.commit();print(json.dumps({"event":"analysis_complete","job_id":jid,"model_id":job["model_id"]}),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--db",type=Path,default=ROOT/"results/checkpoint_embeddings/compact.sqlite")
    p.add_argument("--data-dir",type=Path,default=ROOT/"data/rdb_wavelet_delineation_cache");p.add_argument("--layer",default="encoder")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu");p.add_argument("--torch-threads",type=int,default=1)
    p.add_argument("--min-free-ram-gib",type=float,default=3);p.add_argument("--min-free-disk-gib",type=float,default=8)
    p.add_argument("--bootstrap",type=int,default=2000);p.add_argument("--max-records",type=int,default=0)
    p.add_argument("--primary-model",action="append",default=[]);p.add_argument("--panel-model",action="append",default=[])
    p.add_argument("--no-panel",action="store_true");p.add_argument("--stop-file",type=Path,default=ROOT/"results/checkpoint_embeddings/STOP")
    return p


def main():
    global STOP
    a=parser().parse_args();torch.set_num_threads(a.torch_threads);torch.set_num_interop_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"]="" if a.device=="cpu" else os.environ.get("CUDA_VISIBLE_DEVICES","")
    def stopped(_sig,_frame):
      global STOP;STOP=True
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    models_primary=a.primary_model or PRIMARY_MODELS;models_panel=[] if a.no_panel else (a.panel_model or PANEL_MODELS)
    with single_worker(a.db.with_suffix(a.db.suffix+".lock")):
      c=connect(a.db)
      config={"primary_models":models_primary,"panel_models":models_panel,"layer":a.layer,"pooling":POOLING_VERSION,
              "data_dir":str(a.data_dir.resolve()),"torch_threads":a.torch_threads,"bootstrap":a.bootstrap,
              "umap_seeds":[17,42,73,101,211],"neighbors":[10,30,60],"created_at":now(),"code_sha256":sha256(Path(__file__))}
      c.execute("insert or replace into study_metadata values('frozen_config_json',?)",(json.dumps(config,sort_keys=True),));c.commit()
      jobs=[]
      try:
        for model in models_primary: jobs.append(extract_job(c,model,"all",a.data_dir,a.layer,a.device,a.min_free_ram_gib,a.min_free_disk_gib,a.stop_file,a.max_records))
        for model in models_panel: jobs.append(extract_job(c,model,"test",a.data_dir,a.layer,a.device,a.min_free_ram_gib,a.min_free_disk_gib,a.stop_file,a.max_records))
        for jid in jobs: analyze_job(c,jid,a.bootstrap,[17,42,73,101,211],[10,30,60])
        c.execute("insert or replace into study_metadata values('study_status','complete')");c.execute("insert or replace into study_metadata values('completed_at',?)",(now(),));c.commit()
      except InterruptedError as e:
        c.execute("insert or replace into study_metadata values('study_status','interrupted')");c.commit();print(json.dumps({"event":"interrupted","error":str(e)}),flush=True);return 130
      except Exception:
        c.execute("insert or replace into study_metadata values('study_status','error')");c.execute("insert or replace into study_metadata values('error_traceback',?)",(traceback.format_exc(),));c.commit();raise
      finally:c.close()
    return 0


if __name__=="__main__": raise SystemExit(main())
