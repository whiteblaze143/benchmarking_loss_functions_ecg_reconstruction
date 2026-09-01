#!/usr/bin/env python3
"""
Train / smoke-test Wavelet-SSL + delineation ECG-AIM.

Drop into:
  scripts/train_1lead_wavelet_ssl_mtl.py

Features:
- current PTB-XL 500 Hz / 5000 sample reconstruction protocol
- current CombinatorialCompositeLoss
- wavelet BYOL SSL
- two-head reconstruction + delineation training
- strict delineation tensor cache schema
- 3-epoch curated broad sweep generation
- resumable queue runner
- automatic CSV summary
- audit + quick verification modes

The initial architecture screen defaults to pure MSE mask 1000000 because the
current spatial results show that this exposes architecture effects more clearly.
Do not use test data to select the winning architecture.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, random, shutil, sqlite3, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any, Iterator, Optional

_ROOT=Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0,str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths(include_fairseq=True)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

from scripts.train_mcma_3lead import PTBXLDataset
from scripts.common_loss import CombinatorialCompositeLoss
from unified_latents.engineering.utils.common import mask_unobserved_leads
from unified_latents.engineering.utils.regimes import make_lead_indices
from unified_latents.engineering.experimental.wavelet_ssl_ecg_aim import build_wavelet_ecg_aim

LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
_MODEL_PATH=_ROOT/"unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py"

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def atomic_save(obj,dst):
    dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    def compact(x):
        if isinstance(x,torch.Tensor) and x.is_floating_point(): return x.detach().half().cpu()
        if isinstance(x,dict): return {k:compact(v) for k,v in x.items()}
        if isinstance(x,list): return [compact(v) for v in x]
        return x
    with tempfile.NamedTemporaryFile(dir=dst.parent,suffix=".tmp",delete=False) as f: tmp=Path(f.name)
    try:
        torch.save(compact(obj),tmp)
        with tmp.open("rb") as f:os.fsync(f.fileno())
        os.replace(tmp,dst)
        directory_fd=os.open(dst.parent,os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    finally: tmp.unlink(missing_ok=True)

def atomic_torch_save(obj,dst):
    """Atomic full-precision save for optimizer-complete resume state."""
    dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent,suffix=".tmp",delete=False) as f:tmp=Path(f.name)
    try:
        torch.save(obj,tmp)
        with tmp.open("rb") as f:os.fsync(f.fileno())
        os.replace(tmp,dst)
        directory_fd=os.open(dst.parent,os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    finally:tmp.unlink(missing_ok=True)

def atomic_write_text(dst,text):
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=dst.parent,suffix=".tmp",delete=False) as f:
        tmp=Path(f.name);f.write(text);f.flush();os.fsync(f.fileno())
    try:
        os.replace(tmp,dst)
        directory_fd=os.open(dst.parent,os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    finally:tmp.unlink(missing_ok=True)

_NON_TRAINING_ARGS={
    "audit_delineation_dir","audit_output","emit_sweep_manifest","run_sweep_manifest",
    "summarize_sweep","summary_csv","sweep_output_root","sweep_epochs","sweep_leads",
    "sweep_masks","quick_verify","retry_failed","queue_max_attempts",
    "queue_min_free_gib","queue_min_available_ram_gib","queue_continue_on_error",
    "rolling_resume","resume_min_free_gib",
}

def training_config(a):
    return {k:v for k,v in vars(a).items() if k not in _NON_TRAINING_ARGS}

def input_fingerprints(a):
    data_manifest=Path(a.data_manifest)
    if not data_manifest.is_absolute():data_manifest=_ROOT/data_manifest
    if not data_manifest.is_file():raise FileNotFoundError(f"PTB-XL content manifest is missing: {data_manifest}")
    data_contract=json.loads(data_manifest.read_text())
    declared_root=Path(data_contract.get("tensor_root",""))
    if not declared_root.is_absolute():declared_root=_ROOT/declared_root
    actual_root=Path(a.data_dir)
    if not actual_root.is_absolute():actual_root=_ROOT/actual_root
    if declared_root.resolve()!=actual_root.resolve():
        raise ValueError(f"data manifest covers {declared_root}, not --data-dir {actual_root}")
    for split in ("train","val"):
        if int(data_contract.get("splits",{}).get(split,{}).get("records",0))<1:
            raise ValueError(f"data manifest has no {split} records")
    result={
        "trainer_sha256":sha256_file(Path(__file__).resolve()),
        "model_sha256":sha256_file(_MODEL_PATH),
        "data_manifest_sha256":sha256_file(data_manifest),
    }
    if a.delineation_dir:
        cache_manifest=Path(a.delineation_dir)/"manifest.json"
        if not cache_manifest.is_file():raise FileNotFoundError(f"delineation manifest is missing: {cache_manifest}")
        result["delineation_manifest_sha256"]=sha256_file(cache_manifest)
    assets = {
        "custom_wavelet_asset": a.custom_wavelet_asset,
        "view_a_custom_wavelet_asset": a.view_a_custom_wavelet_asset,
        "view_b_custom_wavelet_asset": a.view_b_custom_wavelet_asset,
    }
    required = {
        "custom_wavelet_asset": a.wavelet_bank == "custom_asset",
        "view_a_custom_wavelet_asset": a.view_a_bank == "custom_asset",
        "view_b_custom_wavelet_asset": a.view_b_bank == "custom_asset",
    }
    for key, value in assets.items():
        if required[key] and not value:raise ValueError(f"{key} is required for custom_asset")
        if value:
            custom=Path(value)
            if not custom.is_file():raise FileNotFoundError(f"custom wavelet asset is missing: {custom}")
            result[f"{key}_sha256"]=sha256_file(custom)
    if a.init_checkpoint:
        initial=Path(a.init_checkpoint)
        if not initial.is_file():raise FileNotFoundError(f"initial checkpoint is missing: {initial}")
        result["init_checkpoint_sha256"]=sha256_file(initial)
    return result

def training_config_sha256(a):
    raw=json.dumps(
        {"config":training_config(a),"input_fingerprints":input_fingerprints(a)},
        sort_keys=True,separators=(",",":"),allow_nan=False
    )
    return hashlib.sha256(raw.encode()).hexdigest()

def rng_state():
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    if torch.cuda.is_available():state["cuda"]=torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:torch.cuda.set_rng_state_all(state["cuda"])

def model_checkpoint_payload(model,a,protocol_sha,best_metrics=None):
    inputs=input_fingerprints(a)
    return {
        "checkpoint_version":1,"architecture":str(model.architecture),
        "model_state_dict":model.state_dict(),"config":training_config(a),
        "training_config_sha256":protocol_sha,"best_metrics":best_metrics or {},
        "provenance":{
            "run_name":a.run_name,"factorial_mask":a.factorial_mask,"seed":a.seed,
            "preprocessing":{"observed_leads":list(a.observed_leads),"target_len":5000,"sample_rate_hz":500},
            "inputs":inputs,
        },
    }

def waveform_from_batch(batch):
    if isinstance(batch,torch.Tensor): return batch
    if isinstance(batch,dict):
        for k in ("waveform","y","signal","ecg","x"):
            if k in batch and isinstance(batch[k],torch.Tensor): return batch[k]
    if isinstance(batch,(tuple,list)):
        ts=[x for x in batch if isinstance(x,torch.Tensor)]
        for x in reversed(ts):
            if x.ndim>=3 and x.shape[-2]==12: return x
        if ts: return ts[-1]
    raise TypeError(f"Cannot identify ECG waveform in {type(batch)}")

def missing_pearson(pred,y,observed):
    m=torch.ones(12,dtype=torch.bool,device=pred.device); m[observed]=False
    p=pred[:,m]; t=y[:,m]
    p=p-p.mean(-1,keepdim=True); t=t-t.mean(-1,keepdim=True)
    return F.cosine_similarity(p.flatten(1),t.flatten(1),dim=1)


class DelineationTensorDataset(Dataset):
    """Strict .pt schema.

    Required:
      waveform      float [12,5000], already 500 Hz, mV
      segmentation  long  [12,5000], classes 0 bg / 1 P / 2 QRS / 3 T / -1 invalid

    Optional:
      seg_valid             bool [12,5000]
      fiducial_heatmaps     float [12,6,5000]
      fiducial_valid        bool [12,6]
      annotation_weight     scalar
      annotation_type       lead_specific / integrated
      record_id, patient_id, source_dataset
    """
    def __init__(self,root):
        self.root=Path(root); self.files=sorted(self.root.glob("*.pt"))
        if not self.files: raise FileNotFoundError(f"No .pt files in {self.root}")
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        p=self.files[i]; d=torch.load(p,map_location="cpu",weights_only=False)
        if not isinstance(d,dict): raise TypeError(f"{p}: expected dict")
        x=torch.as_tensor(d["waveform"],dtype=torch.float32)
        s=torch.as_tensor(d["segmentation"],dtype=torch.long)
        if x.shape!=(12,5000) or s.shape!=(12,5000):
            raise ValueError(f"{p}: waveform={tuple(x.shape)} seg={tuple(s.shape)}")
        legal=((s>=0)&(s<=3))|(s==-1)
        if not bool(legal.all()): raise ValueError(f"{p}: illegal segmentation classes")
        v=torch.as_tensor(d.get("seg_valid",s!=-1),dtype=torch.bool)
        if v.shape!=s.shape: raise ValueError(f"{p}: seg_valid mismatch")
        if not torch.equal(v,s!=-1):raise ValueError(f"{p}: seg_valid and -1 labels disagree")
        fh=d.get("fiducial_heatmaps")
        fv=d.get("fiducial_valid")
        if fh is None:
            # Fixed shape permits batches mixing records with and without
            # optional fiducials; validity keeps the synthetic zeros inert.
            fh=torch.zeros(12,6,5000,dtype=torch.float32); fv=torch.zeros(12,6,dtype=torch.bool)
        else:
            fh=torch.as_tensor(fh,dtype=torch.float32)
            if fh.shape!=(12,6,5000): raise ValueError(f"{p}: fiducial shape")
            fv=torch.as_tensor(fv if fv is not None else torch.ones(12,6),dtype=torch.bool)
            if fv.shape!=(12,6): raise ValueError(f"{p}: fiducial_valid shape")
        return dict(
            waveform=x,segmentation=s,seg_valid=v,fiducial_heatmaps=fh,fiducial_valid=fv,
            annotation_weight=torch.tensor(float(d.get("annotation_weight",1.0))),
            annotation_type=str(d.get("annotation_type","unknown")),
            record_id=str(d.get("record_id",p.stem)),patient_id=str(d.get("patient_id","")),
            source_dataset=str(d.get("source_dataset","")),path=str(p)
        )

def audit_cache(root):
    ds=DelineationTensorDataset(root); counts=[0]*5; nonfinite=0; ids=[]; files=[]; types={}; sources={}
    missing_ids=0;records_with_fiducials=0;valid_fiducials=0
    for i,p in enumerate(tqdm(ds.files,desc="audit")):
        d=ds[i]; s=d["segmentation"]; x=d["waveform"]
        nonfinite+=int((~torch.isfinite(x)).sum())
        for c in range(4): counts[c]+=int((s==c).sum())
        counts[4]+=int((s==-1).sum())
        if d["patient_id"]: ids.append(d["patient_id"])
        else:missing_ids+=1
        valid_count=int(d["fiducial_valid"].sum());valid_fiducials+=valid_count
        records_with_fiducials+=int(valid_count>0)
        types[d["annotation_type"]]=types.get(d["annotation_type"],0)+1
        sources[d["source_dataset"]]=sources.get(d["source_dataset"],0)+1
        files.append({"file":str(p),"sha256":sha256_file(p)})
    return {
        "root":str(Path(root).resolve()),"records":len(ds),"nonfinite":nonfinite,
        "class_counts":{"bg":counts[0],"P":counts[1],"QRS":counts[2],"T":counts[3],"invalid":counts[4]},
        "annotation_types":types,"sources":sources,"unique_patient_ids":len(set(ids)),
        "missing_patient_ids":missing_ids,"records_with_fiducials":records_with_fiducials,
        "valid_fiducial_channels":valid_fiducials,"files":files
    }


def lead_mask(B,observed,device,missing_only=True):
    m=torch.ones(B,12,dtype=torch.bool,device=device)
    if missing_only: m[:,observed]=False
    return m

def ce_loss(logits,target,valid,lmask,aw,class_weights=None):
    eff=valid&lmask[:,:,None]; t=target.clone(); t[~eff]=-1
    per=F.cross_entropy(
        logits.permute(0,1,3,2).reshape(-1,4),t.reshape(-1),
        weight=class_weights,ignore_index=-1,reduction="none"
    ).reshape_as(t)
    w=eff.float()*aw[:,None,None]
    return (per*w).sum()/w.sum().clamp_min(1)

def dice_loss(logits,target,valid,lmask,aw):
    prob=logits.softmax(2); eff=(valid&lmask[:,:,None])[:,:,None].float()
    oh=F.one_hot(target.clamp(0,3),4).permute(0,1,3,2).float()
    p=prob[:,:,1:]*eff; y=oh[:,:,1:]*eff; w=aw[:,None,None,None]
    inter=(p*y*w).sum((0,1,3)); den=((p+y)*w).sum((0,1,3))
    return 1-((2*inter+1e-5)/(den+1e-5)).mean()

def boundary_loss(logits,target,valid,lmask,aw):
    prob=logits.softmax(2); dp=(prob[...,1:]-prob[...,:-1]).abs().sum(2).clamp(1e-5,1-1e-5)
    gt=(target[...,1:]!=target[...,:-1])&(target[...,1:]>=0)&(target[...,:-1]>=0)
    eff=valid[...,1:]&valid[...,:-1]&lmask[:,:,None]
    # Probability-form BCE is intentionally rejected by torch autocast.  The
    # clamped analytic form is identical and computes safely in float32.
    dp=dp.float(); y=gt.float()
    per=-(y*dp.log()+(1-y)*torch.log1p(-dp)); w=eff.float()*aw[:,None,None]
    return (per*w).sum()/w.sum().clamp_min(1)

def fid_loss(logits,target,valid,lmask,aw):
    if logits is None or target.numel()==0: return target.new_zeros(())
    target=target.float()
    per=F.binary_cross_entropy_with_logits(logits.float(),target,reduction="none")
    channel_mask=(valid[:,:,:,None]&lmask[:,:,None,None]).float()*aw[:,None,None,None]
    # Sparse temporal landmarks otherwise admit an all-background solution.
    # Normalize by the actual weighted time points, not merely by channels.
    w=channel_mask*(1.0+20.0*target)
    return (per*w).sum()/w.sum().clamp_min(1)


def confusion_update(C,logits,target,valid,lmask):
    p=logits.argmax(2); eff=valid&lmask[:,:,None]&(target>=0)
    y=target[eff].long(); q=p[eff].long()
    if y.numel(): C+=torch.bincount(y*4+q,minlength=16).reshape(4,4).cpu()

def confusion_metrics(C):
    tp=C.diag().float(); fp=C.sum(0).float()-tp; fn=C.sum(1).float()-tp
    iou=tp/(tp+fp+fn).clamp_min(1); rec=tp/(tp+fn).clamp_min(1); pre=tp/(tp+fp).clamp_min(1)
    f1=2*pre*rec/(pre+rec).clamp_min(1e-8)
    return {"miou_wave":float(iou[1:].mean()),"macro_f1_wave":float(f1[1:].mean()),
            "P_iou":float(iou[1]),"QRS_iou":float(iou[2]),"T_iou":float(iou[3])}

def boundary_counts(logits,target,valid,lmask,tol):
    """Smoke-only transition matching. Do NOT label this AAMI sensitivity."""
    p=logits.argmax(2); TP=FP=FN=0
    for b in range(target.shape[0]):
        for l in range(12):
            if not bool(lmask[b,l]): continue
            v=valid[b,l]; y=target[b,l]; q=p[b,l]
            g=torch.where((y[1:]!=y[:-1])&v[1:]&v[:-1]&(y[1:]>=0)&(y[:-1]>=0))[0]+1
            r=torch.where((q[1:]!=q[:-1])&v[1:]&v[:-1])[0]+1
            used=torch.zeros(r.numel(),dtype=torch.bool,device=r.device); hit=0
            for z in g:
                if not r.numel(): continue
                d=(r-z).abs(); d[used]=tol+1; j=int(d.argmin())
                if int(d[j])<=tol: used[j]=True; hit+=1
            TP+=hit; FN+=int(g.numel())-hit; FP+=int(r.numel())-hit
    return TP,FP,FN


def build_model(a):
    return build_wavelet_ecg_aim(
        target_len=5000,patch_size=a.patch_size,width=a.width,encoder_depth=a.encoder_depth,
        decoder_depth=a.decoder_depth,heads=a.heads,random_mask_ratio=a.random_mask_ratio,
        temporal_mask_ratio=a.temporal_mask_ratio,consistency_weight=a.consistency_weight,
        lead_conditioning_mode=a.lead_conditioning_mode,use_relative_geometry=a.use_relative_geometry,
        use_spatial_film=a.use_spatial_film,spatial_gain_init=a.spatial_gain_init,
        geometry_control=a.geometry_control,use_wavelet_branch=a.use_wavelet_branch,
        wavelet_bank=a.wavelet_bank,custom_wavelet_asset=a.custom_wavelet_asset,
        view_a_bank=a.view_a_bank,view_b_bank=a.view_b_bank,
        view_a_custom_wavelet_asset=a.view_a_custom_wavelet_asset,
        view_b_custom_wavelet_asset=a.view_b_custom_wavelet_asset,
        n_scales=a.n_scales,min_freq_hz=a.min_freq_hz,max_freq_hz=a.max_freq_hz,
        morlet_cycles=a.morlet_cycles,view_a=a.view_a,view_b=a.view_b,
        wavelet_encoder=a.wavelet_encoder,wavelet_dim=a.wavelet_dim,
        wavelet_depth=a.wavelet_depth,wavelet_heads=a.wavelet_heads,
        wavelet_conv_hidden=a.wavelet_conv_hidden,wavelet_fusion=a.wavelet_fusion,
        fusion_heads=a.fusion_heads,inference_view=a.inference_view,ssl_mode=a.ssl_mode,
        ssl_projector_hidden=a.ssl_projector_hidden,ssl_projector_dim=a.ssl_projector_dim,
        ssl_predictor_hidden=a.ssl_predictor_hidden,byol_tau=a.byol_tau,
        use_delineation_head=not a.no_delineation_head,delineation_hidden=a.delineation_hidden,
        delineation_kernel=a.delineation_kernel,predict_fiducials=not a.no_fiducial_head,
        mask_type_mode=a.mask_type_mode
    )

def load_init(model,p,strict=True):
    if not p:return
    d=torch.load(p,map_location="cpu",weights_only=False)
    s=d.get("model_state_dict",d) if isinstance(d,dict) else d
    m,u=model.load_state_dict(s,strict=strict)
    print(f"init missing={len(m)} unexpected={len(u)} strict={strict}")

def forward_model(model,y,obs,compute_delineation=True,compute_ssl=True):
    masked=mask_unobserved_leads(y,obs).contiguous()
    idx=make_lead_indices(obs,y.shape[0],y.device)
    return model(
        masked,y_full=y,lead_indices=idx,compute_delineation=compute_delineation,
        compute_ssl=compute_ssl
    )

def compose(res,y,criterion,a,db=None):
    recon=res["y_pred"].new_zeros(())
    if a.reconstruction_weight:
        recon,*_=criterion(res["y_pred"][...,:y.shape[-1]],y)
    consistency=res.get("limb_consistency_loss")
    if not isinstance(consistency,torch.Tensor):consistency=recon.new_zeros(())
    total=a.reconstruction_weight*recon+a.consistency_weight*consistency
    terms={"recon":float(recon.detach()),"consistency":float(consistency.detach()),
           "ssl":0.,"ce":0.,"dice":0.,"boundary":0.,"fid":0.}
    ssl=res.get("wavelet_ssl_loss")
    if isinstance(ssl,torch.Tensor) and a.ssl_weight:
        total=total+a.ssl_weight*ssl; terms["ssl"]=float(ssl.detach())
    if db is not None and res.get("seg_logits") is not None:
        s=db["segmentation"].to(y.device); v=db["seg_valid"].to(y.device)
        aw=db["annotation_weight"].to(y.device).float(); lm=lead_mask(y.shape[0],a.observed_leads,y.device,a.seg_missing_only)
        cw=None if a.seg_class_weights is None else torch.tensor(a.seg_class_weights,device=y.device)
        ce=ce_loss(res["seg_logits"],s,v,lm,aw,cw); total+=a.seg_ce_weight*ce; terms["ce"]=float(ce.detach())
        if a.dice_weight:
            d=dice_loss(res["seg_logits"],s,v,lm,aw); total+=a.dice_weight*d; terms["dice"]=float(d.detach())
        if a.boundary_weight:
            b=boundary_loss(res["seg_logits"],s,v,lm,aw); total+=a.boundary_weight*b; terms["boundary"]=float(b.detach())
        if a.fiducial_weight:
            fh=db["fiducial_heatmaps"].to(y.device); fv=db["fiducial_valid"].to(y.device)
            f=fid_loss(res.get("fiducial_logits"),fh,fv,lm,aw); total+=a.fiducial_weight*f; terms["fid"]=float(f.detach())
    return total,terms

@torch.no_grad()
def validate_recon(model,loader,criterion,a,device):
    model.eval(); losses=[]; rs=[]
    for i,b in enumerate(loader):
        if a.max_val_batches is not None and i>=a.max_val_batches:break
        y=waveform_from_batch(b)[...,:5000].to(device)
        with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.bfloat16):
            r=forward_model(
                model,y,a.observed_leads,compute_delineation=False,compute_ssl=False
            )
            loss,*_=criterion(r["y_pred"],y)
        losses.append(float(loss)); rs+=missing_pearson(r["y_pred"].float(),y.float(),a.observed_leads).cpu().tolist()
    if not losses or not rs:raise RuntimeError("reconstruction validation produced no samples")
    result={"val_recon_loss":float(np.mean(losses)),"val_missing_pearson":float(np.mean(rs)),
            "val_missing_pearson_p05":float(np.quantile(rs,.05))}
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError(f"non-finite reconstruction metrics: {result}")
    return result

@torch.no_grad()
def validate_del(model,loader,a,device):
    if loader is None:return {}
    model.eval(); C=torch.zeros(4,4,dtype=torch.long); TP=FP=FN=0
    for i,b in enumerate(loader):
        if a.max_val_batches is not None and i>=a.max_val_batches:break
        y=b["waveform"].to(device); s=b["segmentation"].to(device); v=b["seg_valid"].to(device)
        with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.bfloat16):
            r=forward_model(
                model,y,a.observed_leads,compute_delineation=True,compute_ssl=False
            )
        if r.get("seg_logits") is None:continue
        lm=lead_mask(y.shape[0],a.observed_leads,device,a.seg_missing_only)
        confusion_update(C,r["seg_logits"].float(),s,v,lm)
        x,z,q=boundary_counts(r["seg_logits"].float(),s,v,lm,a.boundary_tolerance_samples)
        TP+=x;FP+=z;FN+=q
    m=confusion_metrics(C); sens=TP/max(TP+FN,1); ppv=TP/max(TP+FP,1)
    m.update(boundary_sensitivity_smoke=sens,boundary_ppv_smoke=ppv,
             boundary_f1_smoke=2*sens*ppv/max(sens+ppv,1e-12))
    if not all(math.isfinite(v) for v in m.values()):
        raise FloatingPointError(f"non-finite delineation metrics: {m}")
    return m

def forever(loader):
    while True:
        for x in loader:yield x

def train(a):
    seed_all(a.seed)
    if a.require_cuda and not torch.cuda.is_available():raise RuntimeError("CUDA is required")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda":
        # The installed cuDNN 9/A100 stack aborts Conv1d forward or backward
        # with ptrDesc->finalize() for the delineation-head shapes.  Backward
        # executes after a module-local context exits, so this must remain
        # disabled for the complete training step. ATen still executes on CUDA.
        torch.backends.cudnn.enabled=False
        if hasattr(torch,"set_float32_matmul_precision"):torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)
    pin=device.type=="cuda"
    root=Path(a.data_dir)
    tr_generator=torch.Generator()
    tr=DataLoader(PTBXLDataset(str(root/"train")),batch_size=a.batch_size,shuffle=True,
        num_workers=a.num_workers,pin_memory=pin,generator=tr_generator)
    va=DataLoader(PTBXLDataset(str(root/"val")),batch_size=a.batch_size,shuffle=False,
        num_workers=a.num_workers,pin_memory=pin)
    dt=dv=None;dt_generator=None
    if a.delineation_dir:
        dr=Path(a.delineation_dir)
        dt_generator=torch.Generator()
        dt=DataLoader(DelineationTensorDataset(dr/"train"),batch_size=a.delineation_batch_size,
            shuffle=True,num_workers=a.num_workers,pin_memory=pin,generator=dt_generator)
        dv=DataLoader(DelineationTensorDataset(dr/"val"),batch_size=a.delineation_batch_size,
            shuffle=False,num_workers=a.num_workers,pin_memory=pin)
    if a.train_head_only and dt is None:raise ValueError("head-only requires delineation data")
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    resume_path=out/"resume.pt"
    if (out/"_SUCCESS.json").exists():
        raise FileExistsError(f"completed output already exists: {out}")
    protocol_sha=training_config_sha256(a)
    inputs=input_fingerprints(a)
    config_path=out/"config.json"
    resume=torch.load(resume_path,map_location="cpu",weights_only=False) if a.rolling_resume and resume_path.is_file() else None
    if resume is not None:
        if resume.get("version")!=1 or resume.get("run_name")!=a.run_name:
            raise RuntimeError("resume.pt version/run identity does not match this invocation")
        if resume.get("training_config_sha256")!=protocol_sha:
            raise RuntimeError("resume.pt training-config hash does not match this invocation")
        if resume.get("input_fingerprints")!=inputs:
            raise RuntimeError("resume.pt code/data fingerprints do not match current inputs")
        if config_path.is_file():
            existing=json.loads(config_path.read_text())
            if training_config(argparse.Namespace(**existing))!=training_config(a):
                raise RuntimeError("existing config.json does not match the resume invocation")
    atomic_write_text(config_path,json.dumps(vars(a),indent=2,sort_keys=True,allow_nan=False)+"\n")
    model=build_model(a).to(device)
    if resume is not None:
        model.load_state_dict(resume["model_state_dict"],strict=True)
    else:
        load_init(model,a.init_checkpoint,a.init_strict)
    if a.train_head_only:
        for p in model.parameters():p.requires_grad=False
        for p in model.delineation_head.parameters():p.requires_grad=True
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=a.lr,betas=(.9,.95),weight_decay=a.weight_decay)
    crit=CombinatorialCompositeLoss(a.factorial_mask)
    base_steps=min(len(tr),a.max_train_batches) if a.max_train_batches else len(tr)
    if a.train_head_only: steps=min(len(dt),a.max_train_batches) if a.max_train_batches else len(dt)
    else: steps=base_steps+(base_steps//a.delineation_every if dt else 0)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=a.max_lr,total_steps=max(1,a.epochs*steps),pct_start=a.pct_start)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best=-1e9;bestm={};beste=0;start_epoch=1;history=[]
    prior_peak=0
    if resume is not None:
        if int(resume.get("steps_per_epoch",-1))!=steps:
            raise RuntimeError("resume.pt optimizer-step schedule does not match current loaders")
        opt.load_state_dict(resume["optimizer_state_dict"])
        sch.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        best=float(resume["best_score"]);bestm=dict(resume["best_metrics"])
        beste=int(resume["best_epoch"]);start_epoch=int(resume["next_epoch"])
        history=list(resume.get("history",[]));restore_rng_state(resume["rng_state"])
        prior_peak=int(resume.get("peak_gpu_memory_bytes",0))
        if not (1<=start_epoch<=a.epochs+1):raise RuntimeError("resume.pt next_epoch is out of range")
        if not math.isfinite(best) or not all(math.isfinite(float(v)) for row in history for v in row.values()):
            raise RuntimeError("resume.pt contains non-finite score/history values")
        if history and (beste<1 or beste>len(history)):
            raise RuntimeError("resume.pt best_epoch is inconsistent with completed history")
        if int(sch.state_dict().get("last_epoch",-1))!=(start_epoch-1)*steps:
            raise RuntimeError("resume.pt scheduler position is inconsistent with completed epochs")
        for state in opt.state.values():
            for value in state.values():
                if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):
                    raise RuntimeError("resume.pt optimizer state contains non-finite values")
        print(f"Resuming {a.run_name} at epoch {start_epoch}/{a.epochs}",flush=True)
    if [int(row.get("epoch",-1)) for row in history] != list(range(1,start_epoch)):
        raise RuntimeError("resume history is not a consecutive completed-epoch prefix")
    atomic_write_text(out/"metrics.jsonl","".join(json.dumps(row,sort_keys=True,allow_nan=False)+"\n" for row in history))

    def update(y,db):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.bfloat16):
            r=forward_model(
                model,y,a.observed_leads,compute_delineation=db is not None,
                compute_ssl=bool(a.ssl_weight)
            )
            total,terms=compose(r,y,crit,a,db)
        if not bool(torch.isfinite(total)):raise FloatingPointError(f"non-finite loss: {terms}")
        scaler.scale(total).backward();scaler.unscale_(opt)
        grad_norm=torch.nn.utils.clip_grad_norm_(params,a.grad_clip,error_if_nonfinite=True)
        old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
        if scaler.get_scale()<old_scale:
            raise FloatingPointError("GradScaler skipped an optimizer step")
        sch.step();model.update_byol_target()
        terms["grad_norm"]=float(grad_norm.detach())
        if not all(math.isfinite(v) for v in terms.values()):
            raise FloatingPointError(f"non-finite training terms: {terms}")
        terms["total"]=float(total.detach());return terms

    for ep in range(start_epoch,a.epochs+1):
        tr_generator.manual_seed(a.seed*1_000_003+ep)
        if dt_generator is not None:dt_generator.manual_seed(a.seed*1_000_003+100_000+ep)
        di=iter(dt) if dt is not None else None
        model.train(); terms=[]
        if a.train_head_only:
            for i,db in enumerate(tqdm(dt,desc=f"head {ep}")):
                if a.max_train_batches is not None and i>=a.max_train_batches:break
                terms.append(update(db["waveform"].to(device),db))
        else:
            for i,b in enumerate(tqdm(tr,desc=f"epoch {ep}")):
                if a.max_train_batches is not None and i>=a.max_train_batches:break
                y=waveform_from_batch(b)[...,:5000].to(device);terms.append(update(y,None))
                if di is not None and (i+1)%max(a.delineation_every,1)==0:
                    try:db=next(di)
                    except StopIteration:di=iter(dt);db=next(di)
                    terms.append(update(db["waveform"].to(device),db))
        if not terms:raise RuntimeError("training epoch produced no optimizer steps")
        m={"epoch":ep}
        if terms:
            for k in terms[0]:m["train_"+k]=float(np.mean([x[k] for x in terms]))
        m.update(validate_recon(model,va,crit,a,device));m.update(validate_del(model,dv,a,device))
        score=m.get("miou_wave",0)+.25*m.get("boundary_f1_smoke",0)+.1*m["val_missing_pearson"] if dv else m["val_missing_pearson"]
        if not math.isfinite(score) or not all(math.isfinite(v) for v in m.values()):
            raise FloatingPointError(f"non-finite epoch metrics: {m}")
        print(json.dumps(m,sort_keys=True,allow_nan=False));history.append(m)
        atomic_write_text(out/"metrics.jsonl","".join(json.dumps(row,sort_keys=True,allow_nan=False)+"\n" for row in history))
        if score>best:
            best,beste,bestm=score,ep,dict(m)
            if a.checkpoint_policy in {"best","all"}:
                atomic_save(model_checkpoint_payload(model,a,protocol_sha,bestm),out/"best.pt")
        if a.rolling_resume:
            free=shutil.disk_usage(out).free
            if free<a.resume_min_free_gib*1024**3:
                raise RuntimeError(f"resume disk gate: only {free/1024**3:.2f} GiB free")
            atomic_torch_save({
                "version":1,"run_name":a.run_name,"training_config_sha256":protocol_sha,
                "input_fingerprints":inputs,
                "model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),
                "scheduler_state_dict":sch.state_dict(),"scaler_state_dict":scaler.state_dict(),
                "next_epoch":ep+1,"best_score":best,"best_epoch":beste,
                "steps_per_epoch":steps,
                "best_metrics":bestm,"history":history,"rng_state":rng_state(),
                "peak_gpu_memory_bytes":max(
                    prior_peak,torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0
                ),
                "train_generator_state":tr_generator.get_state(),
                "delineation_generator_state":None if dt_generator is None else dt_generator.get_state(),
            },resume_path)
    if not (1<=beste<=a.epochs):raise RuntimeError("no valid best epoch was selected")
    for name,param in model.named_parameters():
        if not bool(torch.isfinite(param).all()):raise FloatingPointError(f"non-finite model parameter: {name}")
    if a.checkpoint_policy in {"last","all"}:
        atomic_save(model_checkpoint_payload(model,a,protocol_sha,bestm),out/"last.pt")
    summary={"run_name":a.run_name,"epochs_completed":a.epochs,"best_score":best,
             "best_epoch":beste,"training_config_sha256":protocol_sha,
             "input_fingerprints":inputs,
             "peak_gpu_memory_bytes":max(
                 prior_peak,torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0
             ),
             **bestm}
    summary_text=json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n"
    atomic_write_text(out/"summary.json",summary_text)
    checkpoint_artifacts={}
    for checkpoint_name in ("best.pt","last.pt"):
        checkpoint_path=out/checkpoint_name
        if checkpoint_path.is_file():checkpoint_artifacts[checkpoint_name]=sha256_file(checkpoint_path)
    success={
        "version":1,"run_name":a.run_name,"completed_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "epochs_completed":a.epochs,"training_config_sha256":protocol_sha,
        "training_config":training_config(a),"input_fingerprints":inputs,
        "checkpoint_artifacts":checkpoint_artifacts,
        "config_sha256":sha256_file(config_path),"metrics_sha256":sha256_file(out/"metrics.jsonl"),
        "summary_sha256":sha256_file(out/"summary.json"),
    }
    atomic_write_text(out/"_SUCCESS.json",json.dumps(success,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if a.rolling_resume:resume_path.unlink(missing_ok=True)
    return summary


# ------------------------- Broad smoke matrix -------------------------------

def cell(name,**kw):
    d=dict(name=name,lead_conditioning_mode="learned",use_relative_geometry=False,use_spatial_film=False,
           use_wavelet_branch=False,wavelet_bank="ecg_admissible_morlet",n_scales=32,min_freq_hz=.5,max_freq_hz=45.,
           morlet_cycles=6.,view_a="magnitude",view_b="phase_sin",wavelet_encoder="timesformer",
           wavelet_dim=192,wavelet_depth=2,wavelet_heads=6,wavelet_fusion="gated_add",inference_view="a",
           ssl_mode="none",ssl_weight=0.,seg_ce_weight=1.,dice_weight=.5,boundary_weight=0.,
           fiducial_weight=0.,delineation_hidden=96,delineation_kernel=15,mask_type_mode="legacy",
           no_fiducial_head=True)
    d.update(kw);return d

def broad_cells(include_fiducials=False):
    C=[cell("A0_raw"),cell("E1_raw",lead_conditioning_mode="panorama_hybrid",use_relative_geometry=True,use_spatial_film=True)]
    for g,gk in [("A0",{}),("E1",dict(lead_conditioning_mode="panorama_hybrid",use_relative_geometry=True,use_spatial_film=True))]:
        for f in ("gated_add","cross_attn"):C.append(cell(f"{g}_wave_noSSL_{f}",**gk,use_wavelet_branch=True,wavelet_fusion=f))
    pure_ueg="refine-assets/repolarization_ueg_wavelets_v1.pt"
    wyatt={
        .1:"refine-assets/repolarization_ueg_wavelets_wyatt_010.pt",
        .25:"refine-assets/repolarization_ueg_wavelets_wyatt_025.pt",
        .5:"refine-assets/repolarization_ueg_wavelets_wyatt_050.pt",
    }
    physiology=[
        # Matched parent for the principal A0 -> wavelet -> wavelet+SSL chain.
        # Unlike the older engineering-bank controls, this differs from R0
        # only by disabling SSL.
        cell("P1_A0_morlet_mag_phase_noSSL",use_wavelet_branch=True,wavelet_bank="morlet",view_a="magnitude",view_b="phase",ssl_mode="none",ssl_weight=0.),
        cell("R0_morlet_mag_morlet_phase",use_wavelet_branch=True,wavelet_bank="morlet",view_a="magnitude",view_b="phase",ssl_mode="both",ssl_weight=.05),
        cell("R1_morlet_mag_ueg_phase",use_wavelet_branch=True,wavelet_bank="morlet",view_a_bank="morlet",view_b_bank="custom_asset",view_b_custom_wavelet_asset=pure_ueg,view_a="magnitude",view_b="phase",ssl_mode="both",ssl_weight=.05),
        # E1 is a complementarity test, not the presumed improved base.
        cell("C1_E1_morlet_mag_morlet_phase",lead_conditioning_mode="panorama_hybrid",use_relative_geometry=True,use_spatial_film=True,use_wavelet_branch=True,wavelet_bank="morlet",view_a="magnitude",view_b="phase",ssl_mode="both",ssl_weight=.05),
        cell("R2_morlet_mag_ueg_phase_sin",use_wavelet_branch=True,wavelet_bank="morlet",view_a_bank="morlet",view_b_bank="custom_asset",view_b_custom_wavelet_asset=pure_ueg,view_a="magnitude",view_b="phase_sin",ssl_mode="both",ssl_weight=.05),
        cell("R3_morlet_logmag_ueg_phase_sin",use_wavelet_branch=True,wavelet_bank="morlet",view_a_bank="morlet",view_b_bank="custom_asset",view_b_custom_wavelet_asset=pure_ueg,view_a="log_magnitude",view_b="phase_sin",ssl_mode="both",ssl_weight=.05),
        cell("R7_morlet_mag_ueg_real",use_wavelet_branch=True,wavelet_bank="morlet",view_a_bank="morlet",view_b_bank="custom_asset",view_b_custom_wavelet_asset=pure_ueg,view_a="magnitude",view_b="real",ssl_mode="both",ssl_weight=.05),
        cell("R8_morlet_mag_ueg_mag",use_wavelet_branch=True,wavelet_bank="morlet",view_a_bank="morlet",view_b_bank="custom_asset",view_b_custom_wavelet_asset=pure_ueg,view_a="magnitude",view_b="magnitude",ssl_mode="both",ssl_weight=.05),
        cell("R9_ueg_mag_ueg_phase",use_wavelet_branch=True,wavelet_bank="morlet",view_a_bank="custom_asset",view_b_bank="custom_asset",view_a_custom_wavelet_asset=pure_ueg,view_b_custom_wavelet_asset=pure_ueg,view_a="magnitude",view_b="phase",ssl_mode="both",ssl_weight=.05),
    ]
    for label,mix in (("R4",.1),("R5",.25),("R6",.5)):
        physiology.append(cell(f"{label}_morlet_mag_ueg_phase_wyatt",use_wavelet_branch=True,wavelet_bank="morlet",view_a_bank="morlet",view_b_bank="custom_asset",view_b_custom_wavelet_asset=wyatt[mix],view_a="magnitude",view_b="phase",ssl_mode="both",ssl_weight=.05))
    C += physiology
    for va,vb in [("magnitude","phase"),("magnitude","phase_sin"),("log_magnitude","phase_sin"),("log_magnitude","real")]:
        for s in ("local","both"):
            for f in ("gated_add","cross_attn"):
                C.append(cell(f"ssl_{va}_{vb}_{s}_{f}",use_wavelet_branch=True,view_a=va,view_b=vb,ssl_mode=s,ssl_weight=.05,wavelet_fusion=f))
    C += [
        cell("ssl_global",use_wavelet_branch=True,ssl_mode="global",ssl_weight=.05),
        cell("ssl_both_infer_b",use_wavelet_branch=True,ssl_mode="both",ssl_weight=.05,inference_view="b"),
        cell("ssl_both_infer_mean",use_wavelet_branch=True,ssl_mode="both",ssl_weight=.05,inference_view="mean")]
    for sc,cy in [(16,4.),(16,8.),(32,4.),(48,6.)]:
        C.append(cell(f"tf_sc{sc}_cy{cy:g}",use_wavelet_branch=True,n_scales=sc,morlet_cycles=cy,ssl_mode="both",ssl_weight=.05))
    C += [
        cell("tf_small96",use_wavelet_branch=True,wavelet_dim=96,wavelet_heads=4,ssl_mode="both",ssl_weight=.05),
        cell("tf_deep4",use_wavelet_branch=True,wavelet_depth=4,ssl_mode="both",ssl_weight=.05),
        cell("conv_control",use_wavelet_branch=True,wavelet_encoder="conv",ssl_mode="both",ssl_weight=.05)]
    for w in (.01,.1):C.append(cell(f"ssl_w{w:g}",use_wavelet_branch=True,ssl_mode="both",ssl_weight=w))
    for t in (.99,.999):C.append(cell(f"tau{t:g}",use_wavelet_branch=True,ssl_mode="both",ssl_weight=.05,byol_tau=t))
    C += [
        cell("del_ce",dice_weight=0.),cell("del_ce_dice"),
        cell("del_boundary",boundary_weight=.1),
        cell("del_wave_ce",use_wavelet_branch=True,ssl_mode="both",ssl_weight=.05,dice_weight=0.),
        cell("del_wave_boundary",use_wavelet_branch=True,ssl_mode="both",ssl_weight=.05,boundary_weight=.1),
        cell("del_head64k9",delineation_hidden=64,delineation_kernel=9),
        cell("del_head128k25",delineation_hidden=128,delineation_kernel=25)]
    if include_fiducials:
        C.append(cell("del_fid",use_wavelet_branch=True,ssl_mode="both",ssl_weight=.05,
                      fiducial_weight=.1,no_fiducial_head=False))
    C += [
        cell("E1_ssl_local",lead_conditioning_mode="panorama_hybrid",use_relative_geometry=True,use_spatial_film=True,use_wavelet_branch=True,ssl_mode="local",ssl_weight=.05),
        cell("E1_ssl_both",lead_conditioning_mode="panorama_hybrid",use_relative_geometry=True,use_spatial_film=True,use_wavelet_branch=True,ssl_mode="both",ssl_weight=.05),
        cell("E1_ssl_cross",lead_conditioning_mode="panorama_hybrid",use_relative_geometry=True,use_spatial_film=True,use_wavelet_branch=True,ssl_mode="both",ssl_weight=.05,wavelet_fusion="cross_attn")]
    C += [cell("mask_fixed_raw",mask_type_mode="all_masked"),
          cell("mask_fixed_wave",use_wavelet_branch=True,ssl_mode="both",ssl_weight=.05,mask_type_mode="all_masked")]
    # De-duplicate actual configurations, not just labels.  In particular,
    # the original A0_raw and del_ce_dice entries were the same experiment.
    unique=[];seen=set()
    for candidate in C:
        key=json.dumps({k:v for k,v in candidate.items() if k!="name"},sort_keys=True,separators=(",",":"))
        if key not in seen:seen.add(key);unique.append(candidate)
    return unique

def cmd_for(c,a,lead,mask):
    name=f"{c['name']}_{mask}_s{a.seed}_l{lead}"
    cmd=[sys.executable,str(Path(__file__).resolve()),"--run-name",name,"--output-dir",str(Path(a.sweep_output_root)/name),
         "--data-dir",a.data_dir,"--data-manifest",a.data_manifest,
         "--delineation-dir",a.delineation_dir,"--factorial-mask",mask,
         "--observed-leads",str(lead),"--epochs",str(a.sweep_epochs),"--seed",str(a.seed),
         "--batch-size",str(a.batch_size),"--delineation-batch-size",str(a.delineation_batch_size),
         "--num-workers",str(a.num_workers),"--checkpoint-policy","none","--rolling-resume","--require-cuda"]
    bools={"use_relative_geometry","use_spatial_film","use_wavelet_branch"}
    store_true={"no_fiducial_head"}
    for k,v in c.items():
        if k=="name":continue
        flag="--"+k.replace("_","-")
        if k in bools:cmd.append(flag if v else "--no-"+k.replace("_","-"))
        elif k in store_true:
            if v:cmd.append(flag)
        else:cmd += [flag,str(v)]
    return name,cmd

def emit_manifest(a):
    if not a.delineation_dir:raise ValueError("sweep requires --delineation-dir")
    cache_root=Path(a.delineation_dir)
    train_audit=audit_cache(cache_root/"train")
    val_audit=audit_cache(cache_root/"val")
    train_ids={torch.load(p,map_location="cpu",weights_only=False).get("patient_id","") for p in sorted((cache_root/"train").glob("*.pt"))}
    val_ids={torch.load(p,map_location="cpu",weights_only=False).get("patient_id","") for p in sorted((cache_root/"val").glob("*.pt"))}
    train_ids.discard("");val_ids.discard("")
    overlap=sorted(train_ids&val_ids)
    if train_audit["nonfinite"] or val_audit["nonfinite"]:raise ValueError("non-finite delineation cache")
    if train_audit["missing_patient_ids"] or val_audit["missing_patient_ids"]:raise ValueError("delineation cache lacks patient IDs")
    if overlap:raise ValueError(f"delineation train/val patient overlap: {overlap[:5]}")
    include_fiducials=(train_audit["records_with_fiducials"]>0 and val_audit["records_with_fiducials"]>0)
    C=broad_cells(include_fiducials=include_fiducials); jobs=[]
    for mask in a.sweep_masks:
        for lead in a.sweep_leads:
            for c in C:
                n,cmd=cmd_for(c,a,lead,mask);jobs.append({"id":n,"status":"pending","command":cmd,"cell":c})
    p=Path(a.emit_sweep_manifest);p.parent.mkdir(parents=True,exist_ok=True)
    inputs=input_fingerprints(a)
    asset_inventory={}
    for job in jobs:
        c=job["cell"]
        for bank_key,asset_key in (("wavelet_bank","custom_wavelet_asset"),
                                   ("view_a_bank","view_a_custom_wavelet_asset"),
                                   ("view_b_bank","view_b_custom_wavelet_asset")):
            if c.get(bank_key, "inherit") != "custom_asset":continue
            asset=Path(c[asset_key])
            if not asset.is_absolute():asset=_ROOT/asset
            if not asset.is_file():raise FileNotFoundError(f"custom wavelet asset is missing: {asset}")
            asset_inventory[str(asset.resolve())]=sha256_file(asset)
    payload={
        "version":2,"created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "cells":len(C),"jobs":jobs,"checkpoint_retention":"rolling_resume_only; delete on success",
        **inputs,
        "custom_wavelet_assets":dict(sorted(asset_inventory.items())),
        "delineation_audit":{"train":train_audit,"val":val_audit,"patient_overlap":0,
                              "fiducial_cells_enabled":include_fiducials},
    }
    atomic_write_text(p,json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(f"{len(C)} cells; {len(jobs)} jobs -> {p}")

def run_manifest(p,a):
    from scripts.wavelet_ssl_queue import run_queue
    return run_queue(
        p,project_root=_ROOT,max_attempts=a.queue_max_attempts,
        retry_failed=a.retry_failed,min_free_gib=a.queue_min_free_gib,
        min_available_ram_gib=a.queue_min_available_ram_gib,
        continue_on_error=a.queue_continue_on_error,
    )

def summarize(root,out):
    from scripts.wavelet_ssl_queue import validate_success
    queue_dir=Path(root);manifest_path=queue_dir/"manifest.json" if queue_dir.is_dir() else queue_dir
    queue_dir=manifest_path.parent;manifest=json.loads(manifest_path.read_text());expected=len(manifest["jobs"])
    connection=sqlite3.connect(queue_dir/"queue.sqlite");connection.row_factory=sqlite3.Row
    try:db_rows=connection.execute("SELECT id,command_json,cell_json,status FROM jobs ORDER BY ordinal").fetchall()
    finally:connection.close()
    if len(db_rows)!=expected or any(row["status"]!="completed" for row in db_rows):
        raise RuntimeError("refusing to summarize an incomplete queue")
    rows=[]
    for row in db_rows:
        summary=validate_success(json.loads(row["command_json"]),_ROOT)
        rows.append({"job_id":row["id"],"cell":json.loads(row["cell_json"]).get("name"),**summary})
    rows.sort(key=lambda x:(x.get("miou_wave",-1),x.get("boundary_f1_smoke",-1),x.get("val_missing_pearson",-1)),reverse=True)
    keys=sorted({k for r in rows for k in r});buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    atomic_write_text(Path(out),buffer.getvalue())
    print(f"{len(rows)} rows -> {out}")


def add_bool(p,name,default=False):
    p.add_argument("--"+name.replace("_","-"),action=argparse.BooleanOptionalAction,default=default)

def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--run-name",default="wavelet_smoke");p.add_argument("--output-dir",default="refine-logs/wavelet_single")
    p.add_argument("--data-dir",default="data/ptb_xl/tensors")
    p.add_argument("--data-manifest",default="refine-logs/ptbxl_tensor_content_manifest.json")
    p.add_argument("--delineation-dir")
    p.add_argument("--factorial-mask",default="1000000");p.add_argument("--observed-leads",type=int,nargs="+",default=[0])
    p.add_argument("--epochs",type=int,default=3);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--delineation-batch-size",type=int,default=32)
    p.add_argument("--num-workers",type=int,default=2);p.add_argument("--max-train-batches",type=int)
    p.add_argument("--max-val-batches",type=int);p.add_argument("--delineation-every",type=int,default=2)
    p.add_argument("--init-checkpoint");add_bool(p,"init_strict",True)
    p.add_argument("--checkpoint-policy",choices=["none","best","last","all"],default="none")
    add_bool(p,"rolling_resume");p.add_argument("--resume-min-free-gib",type=float,default=3.)
    add_bool(p,"require_cuda")
    p.add_argument("--patch-size",type=int,default=25);p.add_argument("--width",type=int,default=768)
    p.add_argument("--encoder-depth",type=int,default=8);p.add_argument("--decoder-depth",type=int,default=4);p.add_argument("--heads",type=int,default=12)
    p.add_argument("--random-mask-ratio",type=float,default=.5);p.add_argument("--temporal-mask-ratio",type=float,default=.25)
    p.add_argument("--consistency-weight",type=float,default=.05)
    p.add_argument("--lead-conditioning-mode",choices=["learned","panorama","panorama_hybrid"],default="learned")
    add_bool(p,"use_relative_geometry");add_bool(p,"use_spatial_film");p.add_argument("--spatial-gain-init",type=float,default=.1)
    p.add_argument("--geometry-control",choices=["standard","fixed_random","permuted"],default="standard")
    add_bool(p,"use_wavelet_branch");p.add_argument(
        "--wavelet-bank",choices=["morlet","ecg_admissible_morlet","custom_asset"],
        default="ecg_admissible_morlet"
    )
    p.add_argument("--custom-wavelet-asset")
    bank_choices=["inherit","morlet","ecg_admissible_morlet","custom_asset"]
    p.add_argument("--view-a-bank",choices=bank_choices,default="inherit")
    p.add_argument("--view-b-bank",choices=bank_choices,default="inherit")
    p.add_argument("--view-a-custom-wavelet-asset")
    p.add_argument("--view-b-custom-wavelet-asset")
    p.add_argument("--n-scales",type=int,default=32)
    p.add_argument("--min-freq-hz",type=float,default=.5);p.add_argument("--max-freq-hz",type=float,default=45.)
    p.add_argument("--morlet-cycles",type=float,default=6.)
    p.add_argument("--view-a",choices=sorted({"magnitude","log_magnitude","power","phase","phase_sin","phase_cos","real","imag"}),default="magnitude")
    p.add_argument("--view-b",choices=sorted({"magnitude","log_magnitude","power","phase","phase_sin","phase_cos","real","imag"}),default="phase_sin")
    p.add_argument("--wavelet-encoder",choices=["timesformer","conv"],default="timesformer")
    p.add_argument("--wavelet-dim",type=int,default=192);p.add_argument("--wavelet-depth",type=int,default=2)
    p.add_argument("--wavelet-heads",type=int,default=6);p.add_argument("--wavelet-conv-hidden",type=int,default=96)
    p.add_argument("--wavelet-fusion",choices=["none","gated_add","concat_mlp","cross_attn"],default="gated_add")
    p.add_argument("--fusion-heads",type=int,default=8);p.add_argument("--inference-view",choices=["a","b","mean"],default="a")
    p.add_argument("--ssl-mode",choices=["none","global","local","both"],default="none");p.add_argument("--ssl-weight",type=float,default=0.)
    p.add_argument("--ssl-projector-hidden",type=int,default=512);p.add_argument("--ssl-projector-dim",type=int,default=256)
    p.add_argument("--ssl-predictor-hidden",type=int,default=512);p.add_argument("--byol-tau",type=float,default=.996)
    p.add_argument("--delineation-hidden",type=int,default=96);p.add_argument("--delineation-kernel",type=int,default=15)
    add_bool(p,"seg_missing_only",True);p.add_argument("--seg-ce-weight",type=float,default=1.);p.add_argument("--dice-weight",type=float,default=.5)
    p.add_argument("--boundary-weight",type=float,default=0.);p.add_argument("--fiducial-weight",type=float,default=0.)
    p.add_argument("--seg-class-weights",type=float,nargs=4);p.add_argument("--boundary-tolerance-samples",type=int,default=10)
    p.add_argument("--no-delineation-head",action="store_true");p.add_argument("--no-fiducial-head",action="store_true")
    p.add_argument("--mask-type-mode",choices=["legacy","all_masked"],default="legacy")
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--max-lr",type=float,default=5e-4);p.add_argument("--pct-start",type=float,default=.2)
    p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--grad-clip",type=float,default=1.)
    p.add_argument("--reconstruction-weight",type=float,default=1.);p.add_argument("--train-head-only",action="store_true")
    modes=p.add_mutually_exclusive_group()
    modes.add_argument("--audit-delineation-dir");modes.add_argument("--emit-sweep-manifest")
    modes.add_argument("--run-sweep-manifest");modes.add_argument("--summarize-sweep")
    p.add_argument("--audit-output");p.add_argument("--quick-verify",action="store_true")
    p.add_argument("--sweep-output-root",default="refine-logs/wavelet_ssl_smokes/runs")
    p.add_argument("--sweep-epochs",type=int,default=3);p.add_argument("--sweep-leads",type=int,nargs="+",default=[0,1])
    p.add_argument("--sweep-masks",nargs="+",default=["1000000"]);p.add_argument("--summary-csv",default="refine-logs/wavelet_ssl_smokes/summary.csv")
    p.add_argument("--queue-max-attempts",type=int,default=2)
    p.add_argument("--queue-min-free-gib",type=float,default=8.)
    p.add_argument("--queue-min-available-ram-gib",type=float,default=5.)
    add_bool(p,"queue_continue_on_error");add_bool(p,"retry_failed")
    return p

def validate_train_args(a):
    if a.epochs<1:raise ValueError("--epochs must be positive")
    if a.batch_size<1 or a.delineation_batch_size<1:raise ValueError("batch sizes must be positive")
    if a.num_workers<0:raise ValueError("--num-workers cannot be negative")
    if a.max_train_batches is not None and a.max_train_batches<1:raise ValueError("--max-train-batches must be positive")
    if a.max_val_batches is not None and a.max_val_batches<1:raise ValueError("--max-val-batches must be positive")
    if a.delineation_every<1:raise ValueError("--delineation-every must be positive")
    if len(a.observed_leads)!=1 or not (0<=a.observed_leads[0]<12):
        raise ValueError("this program requires exactly one observed lead in [0,11]")
    if len(a.factorial_mask)!=7 or set(a.factorial_mask[:6])-{"0","1"} or a.factorial_mask[6] not in "01234":
        raise ValueError("--factorial-mask must match [01]{6}[0-4]")
    if a.ssl_weight and (not a.use_wavelet_branch or a.ssl_mode=="none"):
        raise ValueError("positive SSL weight requires the wavelet branch and a non-none SSL mode")
    if a.fiducial_weight and a.no_fiducial_head:
        raise ValueError("positive fiducial weight requires the fiducial head")
    if a.train_head_only and a.no_delineation_head:raise ValueError("head-only mode requires the delineation head")
    if a.resume_min_free_gib<=0:raise ValueError("--resume-min-free-gib must be positive")

def main():
    a=parser().parse_args()
    if a.audit_delineation_dir:
        r=audit_cache(a.audit_delineation_dir);txt=json.dumps(r,indent=2);print(txt)
        if a.audit_output:atomic_write_text(Path(a.audit_output),txt+"\n")
        if r["nonfinite"]:raise SystemExit(2)
        return
    if a.emit_sweep_manifest:return emit_manifest(a)
    if a.run_sweep_manifest:raise SystemExit(run_manifest(a.run_sweep_manifest,a))
    if a.summarize_sweep:return summarize(a.summarize_sweep,a.summary_csv)
    if a.quick_verify:
        a.epochs=1;a.max_train_batches=2;a.max_val_batches=2;a.checkpoint_policy="none"
        a.rolling_resume=False
    if a.train_head_only and (a.reconstruction_weight!=0 or a.ssl_weight!=0):
        raise ValueError("Head-only requires --reconstruction-weight 0 --ssl-weight 0")
    validate_train_args(a)
    print(json.dumps(train(a),indent=2,allow_nan=False))

if __name__=="__main__":main()
