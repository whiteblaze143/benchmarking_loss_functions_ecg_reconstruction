#!/usr/bin/env python3
"""SAP-Compliant Clinical Evaluator v2 (evaluate_sap_v2.py).

Vectorized, ultra-fast, SAP-compliant evaluation:
  1. Pearson correlations: Fisher-z (atanh → mean → tanh) not direct mean
  2. p05/p50/p95: patient-level summaries, not ECG-level
  3. Patient is the independent bootstrap unit
  4. Fully vectorized array operations via bincount and rankdata
  5. Lean memory footprint: num_workers=0, clean VRAM caching

Modes:
  --mode gpu-infer     GPU only: reconstruct + ECGFounder/SemiSeg/EchoNext, save .npy
  --mode cpu-aggregate CPU only: load .npy, compute all SAP metrics, write DB
  --mode full          Sequential both (testing / single model)
"""
from __future__ import annotations
import argparse, copy, datetime as dt, gc, json, logging, math, os, sqlite3
import sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch, torch.nn.functional as F
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
for _dep in (
    _ROOT / "external/semiseg/runtime_deps",
    _ROOT / "external/semiseg/semi-seg-ecg/src",
    _ROOT,
):
    if str(_dep) not in sys.path:
        sys.path.insert(0, str(_dep))

import models.backbones as vendor_backbones
import models.decode_heads as vendor_heads
from models.encoder_decoder import EncoderDecoder
from scripts.bootstrap_paths import setup_import_paths; setup_import_paths()
from scripts.ecgfounder_classifier import (
    load_ecgfounder, load_ptbxl_labels, load_task_names, preprocess_ecgfounder,
)
from scripts.echonext_classifier import EchoNextMiniModel, load_echonext_test_metadata
from scripts.evaluate_echonext import load_and_validate as load_echonext_waveforms
from unified_latents.engineering.models.three_d_theta_reconstruction import ThreeDThetaECGAIM
from unified_latents.engineering.experimental.aim_1_lead import build_alitok_vae_1d, mask_unobserved_leads
from unified_latents.engineering.utils.regimes import make_lead_indices
from unified_latents.engineering.experimental.wavelet_ssl_ecg_aim import build_wavelet_ecg_aim
from scripts.train_3dtheta_ablation import CELL_CONFIGS
torch.backends.cudnn.enabled = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
OBSERVED_LEAD_IDX = 0
MISSING_LEAD_INDICES = list(range(1, 12))
REGION_GROUPS = {
    "chest":         [6,7,8,9,10,11],
    "limb":          [1,2,3,4,5],
    "septal":        [6,7],
    "anterior":      [7,8,9],
    "lateral_chest": [9,10,11],
    "high_lateral":  [4,5],
    "inferior":      [1,2,5],
}
EVALUATION_VERSION = "sap_v2"
N_BOOT_PRIMARY   = 10_000
N_BOOT_COMPONENT = 1_000
N_BOOT_EXPL      = 1_000
FLOOR_VAR        = 1e-4

# ============================================================
# SAP Vectorized Statistical Utilities
# ============================================================

def fz(r, clip=0.9999): return np.arctanh(np.clip(r, -clip, clip))

def pt_groupby_mean(vals, patient_ids):
    """Vectorized patient-level mean via np.bincount."""
    pts, inv = np.unique(patient_ids, return_inverse=True)
    counts = np.bincount(inv)
    if vals.ndim == 1:
        return np.bincount(inv, weights=vals) / counts
    out = np.zeros((len(pts), vals.shape[1]), dtype=np.float32)
    for col in range(vals.shape[1]):
        out[:, col] = np.bincount(inv, weights=vals[:, col]) / counts
    return out

def patient_r_array(r_ecg_lead, patient_ids, lmask=None):
    """[N,L] → [P] Fisher-z patient means."""
    z = fz(r_ecg_lead)
    if lmask is not None: z = z[:, lmask]
    pts, inv = np.unique(patient_ids, return_inverse=True)
    counts = np.bincount(inv)
    if z.ndim == 1 or (z.ndim == 2 and z.shape[1] == 1):
        z_1d = z.reshape(-1)
        return np.tanh(np.bincount(inv, weights=z_1d) / counts)
    z_ecg = z.mean(axis=1)
    return np.tanh(np.bincount(inv, weights=z_ecg) / counts)

def pt_mean_r(r_ecg_lead, patient_ids, lmask=None):
    return float(patient_r_array(r_ecg_lead, patient_ids, lmask).mean())

def pt_quantile_r(r_ecg_lead, patient_ids, q, lmask=None):
    return float(np.quantile(patient_r_array(r_ecg_lead, patient_ids, lmask), q))

def pw_mae(abs_err, patient_ids):
    pt_err = pt_groupby_mean(abs_err, patient_ids)
    return float(pt_err.mean())

def pw_rmse(sq_err, patient_ids):
    pt_sq = pt_groupby_mean(sq_err, patient_ids)
    return float(np.sqrt(pt_sq.mean()))

def pw_snr(yt, yr, patient_ids):
    sig = (yt**2).mean(-1); err = ((yt-yr)**2).mean(-1)
    snr_ecg = 10.0*np.log10(np.clip(sig/(err+1e-12), 1e-6, None))
    pt_s = pt_groupby_mean(snr_ecg, patient_ids)
    return float(pt_s.mean())

def pw_ba(yt, yr, patient_ids):
    err = (yr - yt) if yr.ndim == 1 else (yr - yt).mean(-1)
    ep = pt_groupby_mean(err, patient_ids)
    bias = float(ep.mean())
    sd = float(ep.std(ddof=1)) if len(ep) > 1 else 0.0
    return {"bias": bias, "sd": sd, "loa_low": bias - 1.96*sd, "loa_high": bias + 1.96*sd}

def boot_ci_mean(pt_vals, n=N_BOOT_PRIMARY, seed=42):
    P = len(pt_vals)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P, size=(n, P))
    sample_means = pt_vals[idx].mean(axis=1)
    return float(np.percentile(sample_means, 2.5)), float(np.percentile(sample_means, 97.5))

def boot_ci_quantile(pt_vals, q, n=N_BOOT_PRIMARY, seed=42):
    P = len(pt_vals)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P, size=(n, P))
    sample_qs = np.quantile(pt_vals[idx], q, axis=1)
    return float(np.percentile(sample_qs, 2.5)), float(np.percentile(sample_qs, 97.5))

def fast_macro_auroc(y_binary, pred, elig_cols=None):
    P, C = pred.shape
    ranks = rankdata(pred, axis=0)
    n_pos = (y_binary == 1).sum(axis=0)
    n_neg = P - n_pos
    r_pos = (ranks * (y_binary == 1)).sum(axis=0)
    U = r_pos - n_pos * (n_pos + 1) / 2.0
    denom = n_pos * n_neg
    valid = (n_pos >= 2) & (n_neg >= 2)
    if elig_cols is not None:
        mask = np.zeros(C, dtype=bool)
        mask[elig_cols] = True
        valid = valid & mask
    auc = np.full(C, np.nan)
    auc[valid] = U[valid] / denom[valid]
    return float(np.nanmean(auc)) if np.any(valid) else np.nan

def boot_ci_auroc_and_delta(y_binary, pred1, pred2, elig_cols=None, n=1000, seed=42):
    P = len(y_binary)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P, size=(n, P))
    ests_auc = []
    ests_delta = []
    for b in range(n):
        bidx = idx[b]
        yb = y_binary[bidx]
        a1 = fast_macro_auroc(yb, pred1[bidx], elig_cols)
        a2 = fast_macro_auroc(yb, pred2[bidx], elig_cols)
        ests_auc.append(a1)
        ests_delta.append(a1 - a2)
    a_auc = np.array(ests_auc)
    a_auc = a_auc[np.isfinite(a_auc)]
    ci_auc = (float(np.percentile(a_auc, 2.5)), float(np.percentile(a_auc, 97.5))) if len(a_auc) > 0 else (np.nan, np.nan)
    
    a_del = np.array(ests_delta)
    a_del = a_del[np.isfinite(a_del)]
    ci_del = (float(np.percentile(a_del, 2.5)), float(np.percentile(a_del, 97.5))) if len(a_del) > 0 else (np.nan, np.nan)
    return ci_auc, ci_del

# ============================================================
# SemiSeg + QRS
# ============================================================

def load_semiseg(device):
    ckpt = _ROOT/"results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42/best-MeanIoU.pth"
    pl = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(pl["config"])
    bn, bk = next(iter(cfg["backbone"].items()))
    if bn=="vit_seg_tiny": bn="vit_tiny"
    hn, hk = next(iter(cfg["decode_head"].items()))
    m = EncoderDecoder(
        backbone=getattr(vendor_backbones, bn)(**bk),
        decode_head=getattr(vendor_heads, hn)(**hk),
        decode_head_loss=torch.nn.CrossEntropyLoss(),
        use_latent_projection=bool(cfg.get("use_latent_projection",False)),
        projection_in_dim=cfg.get("projection_in_dim"),
        projection_out_dim=cfg.get("projection_out_dim"),
    )
    m.load_state_dict(pl["model_ema"], strict=True)
    return m.float().to(device).eval()

def qrs_ms(mask):
    active=(mask==2); ch=np.diff(np.pad(active.astype(np.int8),(1,1)))
    st=np.flatnonzero(ch==1); sp=(np.flatnonzero(ch==-1)-1)
    if not len(st) or not len(sp): return np.nan
    v=[]
    for on in st:
        c=sp[sp>on]
        if len(c):
            d=(c[0]-on+1)/500*1000
            if 40<=d<=300: v.append(d)
    return float(np.median(v)) if v else np.nan

# ============================================================
# Model Loader (verbatim from v1)
# ============================================================

class Reconstructor:
    def __init__(self, mid, ckpt, device):
        self.mid=mid; self.ckpt=Path(ckpt); self.device=device; self.is_zscore=False
        if mid in {"reference", "__original_l0__"}:
            self.model=None; self.atype="identity"
            return
        self.model, self.atype = self._load()
        self.model=self.model.to(device).eval()

    def _load(self):
        if not self.ckpt.exists(): raise FileNotFoundError(self.ckpt)
        pl=torch.load(self.ckpt, map_location="cpu", weights_only=False)
        sd=pl.get("model_state_dict",pl) if isinstance(pl,dict) else pl
        sd={k.replace("_orig_mod.",""):v for k,v in sd.items()}
        bid=self.mid
        for s in ("_s42_l0","_s200_l0","_s1337_l0","_s43_l0","_s44_l0"): bid=bid.replace(s,"")
        if bid in CELL_CONFIGS:
            c=CELL_CONFIGS[bid]; m=ThreeDThetaECGAIM(code_mode=c["code_mode"],fusion=c["fusion"],width=768,encoder_depth=8,decoder_depth=4,heads=12)
            m.load_state_dict(sd,strict=True); return m,"3dtheta"
        if self.mid.startswith(("conv10e_", "conv15e_", "lean2_")):
            c=pl.get("config",{}) if isinstance(pl,dict) else {}
            kw=dict(target_len=5000,patch_size=c.get("patch_size",25),width=c.get("width",768),
                    encoder_depth=c.get("encoder_depth",8),decoder_depth=c.get("decoder_depth",4),
                    heads=c.get("heads",12),lead_conditioning_mode=c.get("lead_conditioning_mode","learned"),
                    use_relative_geometry=c.get("use_relative_geometry",False),
                    use_spatial_film=c.get("use_spatial_film",False),
                    spatial_gain_init=c.get("spatial_gain_init",0.1),
                    geometry_control=c.get("geometry_control","standard"),
                    use_wavelet_branch=c.get("use_wavelet_branch",False),
                    ssl_mode=c.get("ssl_mode","none"),
                    use_delineation_head=not c.get("no_delineation_head",False),
                    predict_fiducials=c.get("predict_fiducials",False),
                    wavelet_encoder=c.get("wavelet_encoder","timesformer"),
                    wavelet_dim=c.get("wavelet_dim",192),wavelet_depth=c.get("wavelet_depth",2),
                    wavelet_heads=c.get("wavelet_heads",6),wavelet_conv_hidden=c.get("wavelet_conv_hidden",96),
                    wavelet_fusion=c.get("wavelet_fusion","gated_add"),fusion_heads=c.get("fusion_heads",8),
                    view_a=c.get("view_a","magnitude"),view_b=c.get("view_b","phase"),
                    view_a_bank=c.get("view_a_bank","morlet"),view_b_bank=c.get("view_b_bank","morlet"),
                    view_b_custom_wavelet_asset=c.get("view_b_custom_wavelet_asset"),
                    n_scales=c.get("n_scales",32),min_freq_hz=c.get("min_freq_hz",0.5),
                    max_freq_hz=c.get("max_freq_hz",45.0),morlet_cycles=c.get("morlet_cycles",6.0))
            m=build_wavelet_ecg_aim(**kw); m.load_state_dict(sd,strict=False)
            self.is_zscore=bool(c.get("zscore_norm",False)); return m,"wavelet_mtl"
        tl=pl.get("target_len",5000) if isinstance(pl,dict) else 5000
        ps=pl.get("alitok_patch_size",25) if isinstance(pl,dict) else 25
        ed=pl.get("alitok_encoder_depth",8) if isinstance(pl,dict) else 8
        dd=pl.get("alitok_decoder_depth",4) if isinstance(pl,dict) else 4
        w=pl.get("alitok_width",768) if isinstance(pl,dict) else 768
        h=pl.get("alitok_heads",12) if isinstance(pl,dict) else 12
        lc=pl.get("lead_conditioning_mode","learned") if isinstance(pl,dict) else "learned"
        rg=pl.get("use_relative_geometry",False) if isinstance(pl,dict) else False
        sf=pl.get("use_spatial_film",False) if isinstance(pl,dict) else False
        sg=pl.get("spatial_gain_init",0.1) if isinstance(pl,dict) else 0.1
        gc_=pl.get("geometry_control","standard") if isinstance(pl,dict) else "standard"
        ar=pl.get("architecture","ecg_aim_v1") if isinstance(pl,dict) else "ecg_aim_v1"
        an={"ecg_aim":"ecg_aim_v1","ecg_aim_spatial":"ecg_aim_spatial_v1","ecg_aim_panorama_author":"ecg_aim_panorama_author_v1","ecg_aim_exact_theta":"ecg_aim_exact_theta_factorial_v1"}.get(ar,ar)
        m=build_alitok_vae_1d(architecture=an,target_len=tl,patch_size=ps,encoder_depth=ed,decoder_depth=dd,encoder_width=w,decoder_width=w,encoder_heads=h,decoder_heads=h,lead_conditioning_mode=lc,use_relative_geometry=rg,use_spatial_film=sf,spatial_gain_init=sg,geometry_control=gc_)
        m.load_state_dict(sd,strict=False); return m,"alitok"

    @torch.inference_mode()
    def reconstruct(self, wav):
        wav=wav.float().to(self.device); B=wav.shape[0]
        if self.atype=="identity":
            return wav.clone()
        if self.is_zscore:
            m_=wav.mean(dim=(-2,-1),keepdim=True); s_=wav.std(dim=(-2,-1),keepdim=True).clamp_min(1e-6)
            inp=(wav-m_)/s_
        else: inp=wav
        if self.atype=="3dtheta": rc=self.model(inp[:,0:1,:],obs_lead_idx=0)["y_pred"]
        else:
            mk=mask_unobserved_leads(inp,[OBSERVED_LEAD_IDX]); li=make_lead_indices([OBSERVED_LEAD_IDX],B,self.device)
            if hasattr(self.model,"impute_from_regressor"): rc=self.model.impute_from_regressor(mk,lead_indices=li)["y_pred"]
            else: rc=self.model(mk,y_full=inp,lead_indices=li,mode="stage1")["y_pred"]
        if self.is_zscore: rc=rc*s_+m_
        ml=min(rc.shape[-1],wav.shape[-1]); rc=rc[...,:ml]; rc[:,OBSERVED_LEAD_IDX,:]=wav[:,OBSERVED_LEAD_IDX,:ml]
        return rc

class PTBDataset(Dataset):
    def __init__(self, d): self.f=sorted(list(d.glob("*.pt")))
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        p=self.f[i]; return torch.load(p,map_location="cpu",weights_only=True).float(), int(p.stem)

# ============================================================
# Database
# ============================================================

DDL = """
CREATE TABLE IF NOT EXISTS sap_model_metrics (
    model_id TEXT NOT NULL, metric_name TEXT NOT NULL,
    dataset TEXT NOT NULL, block TEXT NOT NULL,
    point_estimate REAL, ci_low REAL, ci_high REAL,
    n_patients INTEGER, n_ecgs INTEGER, n_bootstraps INTEGER,
    fisher_z_corrected INTEGER DEFAULT 0,
    patient_weighted INTEGER DEFAULT 0,
    legacy_only INTEGER DEFAULT 0,
    evaluation_version TEXT, created_at TIMESTAMP,
    PRIMARY KEY(model_id, metric_name, dataset, evaluation_version)
);"""

def init_db(db_path):
    db_path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db_path,timeout=60) as c: c.executescript(DDL)

def wdb(db_path, mid, mn, ds, blk, v,
        cl=None, ch=None, np_=None, ne=None, nb=None,
        fz_=False, pw_=False, lg_=False):
    def safe(x): return float(x) if x is not None and np.isfinite(x) else None
    now_str = dt.datetime.now(dt.timezone.utc).isoformat()
    with sqlite3.connect(db_path,timeout=60) as c:
        c.execute("""INSERT OR REPLACE INTO sap_model_metrics
            (model_id,metric_name,dataset,block,point_estimate,ci_low,ci_high,
             n_patients,n_ecgs,n_bootstraps,fisher_z_corrected,patient_weighted,
             legacy_only,evaluation_version,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (mid,mn,ds,blk,safe(v),safe(cl),safe(ch),np_,ne,nb,
             int(fz_),int(pw_),int(lg_),EVALUATION_VERSION,now_str))

# ============================================================
# GPU Inference Phase
# ============================================================

def gpu_infer(mid, ckpt, raw_dir, device, batch_size=32, smoke=False):
    raw_dir.mkdir(parents=True,exist_ok=True)
    done=raw_dir/"_DONE"
    if done.exists(): log.info("[%s] GPU arrays exist, skipping.", mid); return
    log.info("[%s] GPU inference → %s", mid, raw_dir)
    rec=Reconstructor(mid, ckpt, device)
    seg=load_semiseg(device)
    ptb_dir=_ROOT/"data/ptb_xl/tensors/test"
    import pandas as pd
    df=pd.read_csv(_ROOT/"data/ptb_xl/ptbxl_database.csv",index_col="ecg_id")
    er=_ROOT/"ecg_fm_integration/ecgfounder_repo"
    tasks=load_task_names(er/"tasks.txt")
    ecgf=load_ecgfounder(er, er/"checkpoint/12_lead_ECGFounder.pth", device, len(tasks))
    ldf=load_ptbxl_labels(er/"csv/ptbxl_label.csv")
    lmap=dict(zip(ldf["filename_hr"],ldf["ecgfounder_labels"]))
    ds=PTBDataset(ptb_dir)
    ld=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    yt_l,yr_l,pid_l,eid_l=[],[],[],[]
    flab_l,fgt_l,frec_l=[],[],[]
    sgt_l,srec_l=[],[]
    with torch.inference_mode():
        for bi,(batch,ids) in enumerate(tqdm(ld,desc=f"PTB-XL {mid}")):
            t=batch.to(device); r=rec.reconstruct(t)
            yt_l.append(t.cpu().numpy()); yr_l.append(r.cpu().numpy())
            tnn=preprocess_ecgfounder(t); rnn=preprocess_ecgfounder(r)
            pt=torch.sigmoid(ecgf(tnn)).cpu().numpy()
            pr=torch.sigmoid(ecgf(rnn)).cpu().numpy()
            tr=F.interpolate(t,size=2500,mode="linear",align_corners=False)
            rr=F.interpolate(r,size=2500,mode="linear",align_corners=False)
            ts=(tr-tr.mean(-1,keepdim=True))/(tr.std(-1,keepdim=True)+1e-6)
            rs=(rr-rr.mean(-1,keepdim=True))/(rr.std(-1,keepdim=True)+1e-6)
            mst=seg(ts[:,1:2,:])["seg_logits"].argmax(1).repeat_interleave(2,-1)[:,:5000].cpu().numpy()
            msr=seg(rs[:,1:2,:])["seg_logits"].argmax(1).repeat_interleave(2,-1)[:,:5000].cpu().numpy()
            for i,eid in enumerate(ids.numpy()):
                if eid not in df.index: continue
                row=df.loc[eid]; pid_l.append(int(row["patient_id"])); eid_l.append(int(eid))
                fn=row.get("filename_hr","")
                if fn in lmap: flab_l.append(lmap[fn]); frec_l.append(pr[i]); fgt_l.append(pt[i])
                sgt_l.append(mst[i]); srec_l.append(msr[i])
            if smoke and bi>=3: break
    Y=np.concatenate(yt_l,0); R=np.concatenate(yr_l,0)
    np.save(raw_dir/"ptbxl_y_true.npy",Y.astype(np.float32))
    np.save(raw_dir/"ptbxl_y_recon.npy",R.astype(np.float32))
    np.save(raw_dir/"ptbxl_patient_ids.npy",np.array(pid_l))
    np.save(raw_dir/"ptbxl_ecg_ids.npy",np.array(eid_l))
    if flab_l:
        np.save(raw_dir/"founder_labels.npy",np.array(flab_l,dtype=np.float32))
        np.save(raw_dir/"founder_gt_probs.npy",np.array(fgt_l,dtype=np.float32))
        np.save(raw_dir/"founder_recon_probs.npy",np.array(frec_l,dtype=np.float32))
        with open(raw_dir/"founder_task_names.json","w") as fh: json.dump(list(tasks),fh)
    if sgt_l:
        np.save(raw_dir/"semiseg_masks_gt.npy",np.array(sgt_l,dtype=np.uint8))
        np.save(raw_dir/"semiseg_masks_recon.npy",np.array(srec_l,dtype=np.uint8))
    del rec,seg,ecgf,yt_l,yr_l; gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache()
    # EchoNext
    echo_f=_ROOT/"data/echonext/EchoNext_test_waveforms.npy"
    if echo_f.exists():
        log.info("[%s] EchoNext inference...",mid)
        rec2=Reconstructor(mid,ckpt,device)
        ed,_=load_echonext_waveforms(_ROOT/"data/echonext")
        ne=min(1000,len(ed)) if not smoke else 32
        er2=_ROOT/"ecg_fm_integration/echonext_minimodel_repo/7-EchoNext Minimodel"
        shd=EchoNextMiniModel(er2,device)
        smeta,stab,slabs=load_echonext_test_metadata(_ROOT/"data/echonext/echonext_metadata_100k.csv",shd.transformer_path)
        ref_c,rec_c=[],[]
        with torch.inference_mode():
            for s in range(0,ne,batch_size):
                e=min(s+batch_size,ne); bt=torch.from_numpy(ed.batch(s,e)).float().to(device)
                rt=rec2.reconstruct(bt)
                ref_c.append(shd.predict_official_waveforms(ed.official_normalized_batch(s,e),stab[s:e]))
                rec_c.append(shd.predict_reconstruction_500hz(rt,stab[s:e]))
        ref_shd=np.concatenate(ref_c,0); rec_shd=np.concatenate(rec_c,0)
        np.save(raw_dir/"echo_ref_probs.npy",ref_shd[:ne].astype(np.float32))
        np.save(raw_dir/"echo_rec_probs.npy",rec_shd[:ne].astype(np.float32))
        np.save(raw_dir/"echo_labels.npy",slabs[:ne].astype(np.float32))
        ekeys=smeta["patient_key"].values[:ne]; _,epids=np.unique(ekeys,return_inverse=True)
        np.save(raw_dir/"echo_patient_ids.npy",epids.astype(np.int32))
        del rec2,shd; gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()
    done.touch()
    log.info("[%s] GPU done.",mid)

# ============================================================
# CPU Aggregation Phase
# ============================================================

def cpu_aggregate(mid, raw_dir, db_path, smoke=False):
    log.info("[%s] CPU SAP aggregation → %s",mid,db_path)
    init_db(db_path)
    Y=np.load(raw_dir/"ptbxl_y_true.npy")
    R=np.load(raw_dir/"ptbxl_y_recon.npy")
    pids=np.load(raw_dir/"ptbxl_patient_ids.npy")
    N=len(pids); P=len(np.unique(pids))
    log.info("[%s] N=%d ECGs, P=%d patients",mid,N,P)

    def W(mn,v,cl=None,ch=None,nb=None,fz_=False,pw_=False,lg_=False,blk="reconstruction",ds="ptbxl"):
        wdb(db_path,mid,mn,ds,blk,v,cl,ch,P,N,nb,fz_,pw_,lg_)

    nb_pri  = max(100, N_BOOT_PRIMARY  // (10 if smoke else 1))
    nb_comp = max(100, N_BOOT_COMPONENT// (10 if smoke else 1))
    nb_exp  = max(100, N_BOOT_EXPL     // (10 if smoke else 1))
    missing_mask=np.zeros(12,dtype=bool); missing_mask[MISSING_LEAD_INDICES]=True

    # --- Block 8: Per-lead metrics ---
    log.info("[%s] Block 8: per-lead...",mid)
    r_el=np.zeros((N,12),dtype=np.float32)
    mae_el=np.zeros((N,12),dtype=np.float32)
    mse_el=np.zeros((N,12),dtype=np.float32)
    for li in range(12):
        yt=Y[:,li,:]; yr=R[:,li,:]
        ytc=yt-yt.mean(-1,keepdims=True); yrc=yr-yr.mean(-1,keepdims=True)
        denom=np.sqrt((ytc**2).sum(-1)*(yrc**2).sum(-1))+1e-8
        r_el[:,li]=(ytc*yrc).sum(-1)/denom
        mae_el[:,li]=np.abs(yt-yr).mean(-1)
        mse_el[:,li]=((yt-yr)**2).mean(-1)

    for li,ln in enumerate(LEAD_NAMES):
        is_m=(li!=OBSERVED_LEAD_IDX)
        blk="lead_missing" if is_m else "lead_observed"
        pfx=f"lead_{ln}" if is_m else f"source_consistency_lead_{ln}"
        pt_r_lead = patient_r_array(r_el[:,li:li+1], pids)
        rv = float(pt_r_lead.mean())
        cl, ch = boot_ci_mean(pt_r_lead, n=nb_exp)
        W(f"{pfx}_pearson_r",rv,cl,ch,nb_exp,fz_=True,pw_=True,blk=blk)
        W(f"pearson_lead_{ln}",rv,fz_=True,pw_=True,blk=blk)
        W(f"p05_lead_{ln}",pt_quantile_r(r_el[:,li:li+1],pids,0.05),fz_=True,pw_=True,blk=blk)
        W(f"p50_lead_{ln}",pt_quantile_r(r_el[:,li:li+1],pids,0.50),fz_=True,pw_=True,blk=blk)
        W(f"p95_lead_{ln}",pt_quantile_r(r_el[:,li:li+1],pids,0.95),fz_=True,pw_=True,blk=blk)
        W(f"mae_lead_{ln}",pw_mae(mae_el[:,li],pids),pw_=True,blk=blk)
        W(f"rmse_lead_{ln}",pw_rmse(mse_el[:,li],pids),pw_=True,blk=blk)
        W(f"snr_lead_{ln}",pw_snr(Y[:,li],R[:,li],pids),pw_=True,blk=blk)
        ba=pw_ba(Y[:,li],R[:,li],pids)
        W(f"lead_{ln}_bland_bias_mv",ba["bias"],pw_=True,blk=blk)
        W(f"lead_{ln}_mae_mv",pw_mae(mae_el[:,li],pids),pw_=True,blk=blk)
        W(f"lead_{ln}_pearson_r",rv,fz_=True,pw_=True,blk=blk)
        if ln=="V3":
            W("lead_V3_loa_low_mv",ba["loa_low"],pw_=True,blk=blk)
            W("lead_V3_loa_high_mv",ba["loa_high"],pw_=True,blk=blk)

    # --- Block 9: Regional (Lead I excluded) ---
    log.info("[%s] Block 9: regional...",mid)
    pt_r_missing = patient_r_array(r_el, pids, missing_mask)
    r_pri = float(pt_r_missing.mean())
    cl, ch = boot_ci_mean(pt_r_missing, n=N_BOOT_PRIMARY)
    W("mean_all_missing_r",r_pri,cl,ch,N_BOOT_PRIMARY,fz_=True,pw_=True,blk="regional")
    cl, ch = boot_ci_quantile(pt_r_missing, 0.05, n=N_BOOT_PRIMARY)
    W("p05_all_missing_r",float(np.quantile(pt_r_missing,0.05)),cl,ch,N_BOOT_PRIMARY,fz_=True,pw_=True,blk="regional")
    W("p50_all_missing_r",float(np.quantile(pt_r_missing,0.50)),fz_=True,pw_=True,blk="regional")
    W("p95_all_missing_r",float(np.quantile(pt_r_missing,0.95)),fz_=True,pw_=True,blk="regional")
    for rn,ri in REGION_GROUPS.items():
        rm=np.zeros(12,dtype=bool); rm[ri]=True
        pt_r_reg = patient_r_array(r_el, pids, rm)
        W(f"mean_{rn}_r",float(pt_r_reg.mean()),fz_=True,pw_=True,blk="regional")
        W(f"p05_{rn}_r",float(np.quantile(pt_r_reg,0.05)),fz_=True,pw_=True,blk="regional")

    # --- Block 5: QRS / Conduction ---
    if (raw_dir/"semiseg_masks_gt.npy").exists():
        log.info("[%s] Block 5: QRS/SemiSeg...",mid)
        mgt=np.load(raw_dir/"semiseg_masks_gt.npy")
        mre=np.load(raw_dir/"semiseg_masks_recon.npy")
        qgt=np.array([qrs_ms(m) for m in mgt])
        qre=np.array([qrs_ms(m) for m in mre])
        ok=np.isfinite(qgt)&np.isfinite(qre)
        pok=pids[ok]
        W("qrs_duration_mae_ms",pw_mae(np.abs(qgt[ok]-qre[ok]),pok),pw_=True,blk="qrs")
        ba_q=pw_ba(qgt[ok],qre[ok],pok)
        W("qrs_duration_bland_bias_ms",ba_q["bias"],pw_=True,blk="qrs")
        W("qrs_duration_loa_low_ms",ba_q["loa_low"],pw_=True,blk="qrs")
        W("qrs_duration_loa_high_ms",ba_q["loa_high"],pw_=True,blk="qrs")
        # SemiSeg IoU
        iou_map={"semiseg_qrs_wave_iou":2,"semiseg_p_wave_iou":1,"semiseg_t_wave_iou":3}
        iou_all={}
        for nm,ci in iou_map.items():
            ie=np.array([((mt==ci)&(mr==ci)).sum()/max(((mt==ci)|(mr==ci)).sum(),1) for mt,mr in zip(mgt,mre)],dtype=np.float32)
            iou_all[nm]=ie
            ipt = pt_groupby_mean(ie, pids)
            W(nm,float(ipt.mean()),pw_=True,blk="semiseg")
        miou=np.stack(list(iou_all.values()),1).mean(1)
        mpt=pt_groupby_mean(miou, pids)
        W("semiseg_miou",float(mpt.mean()),pw_=True,blk="semiseg")
        # Conduction >120ms
        yd=(qgt[ok]>120).astype(int); rd=(qre[ok]>120).astype(int)
        if yd.sum()>=5 and (1-yd).sum()>=5:
            try:
                W("conduction_delay_120ms_auroc",roc_auc_score(yd,qre[ok]),blk="qrs")
                W("conduction_delay_120ms_auprc",average_precision_score(yd,qre[ok]),blk="qrs")
                from sklearn.metrics import confusion_matrix,f1_score as _f1
                cm=confusion_matrix(yd,rd); tn,fp,fn,tp=(cm.ravel() if cm.size==4 else (cm[0,0],0,0,0))
                W("conduction_delay_120ms_sens",tp/(tp+fn+1e-9),blk="qrs")
                W("conduction_delay_120ms_spec",tn/(tn+fp+1e-9),blk="qrs")
                W("conduction_delay_120ms_ppv",tp/(tp+fp+1e-9),blk="qrs")
                W("conduction_delay_120ms_npv",tn/(tn+fn+1e-9),blk="qrs")
                W("conduction_delay_120ms_f1",_f1(yd,rd,zero_division=0),blk="qrs")
            except Exception as e: log.warning("[%s] conduction: %s",mid,e)
            try:
                from scipy.stats import fisher_exact
                cm2=np.array([[(yd==0)&(rd==0),(yd==0)&(rd==1)],[(yd==1)&(rd==0),(yd==1)&(rd==1)]]).reshape(2,2)
                _,fp2=fisher_exact(cm2); W("conduction_delay_120ms_fisher_p",fp2,lg_=True,blk="qrs")
            except: pass

    # --- Block 6: LVH ---
    log.info("[%s] Block 6: LVH...",mid)
    sv1=np.abs(Y[:,6,1000:4000].min(-1)); rv5=Y[:,10,1000:4000].max(-1)
    sv1r=np.abs(R[:,6,1000:4000].min(-1)); rv5r=R[:,10,1000:4000].max(-1)
    slvg=sv1+rv5; slvr=sv1r+rv5r
    W("lvh_sokolowlyon_mae_mv",pw_mae(np.abs(slvg-slvr),pids),pw_=True,blk="lvh")
    ba_l=pw_ba(slvg,slvr,pids)
    W("lvh_sokolowlyon_bland_bias_mv",ba_l["bias"],pw_=True,blk="lvh")
    W("lvh_sokolowlyon_loa_low_mv",ba_l["loa_low"],pw_=True,blk="lvh")
    W("lvh_sokolowlyon_loa_high_mv",ba_l["loa_high"],pw_=True,blk="lvh")
    ylvh=(slvg>3.5).astype(int); rlvh=(slvr>3.5).astype(int)
    if ylvh.sum()>=5 and (1-ylvh).sum()>=5:
        try:
            W("lvh_sokolowlyon_auroc",roc_auc_score(ylvh,slvr),blk="lvh")
            W("lvh_sokolowlyon_auprc",average_precision_score(ylvh,slvr),blk="lvh")
            from sklearn.metrics import confusion_matrix,f1_score as _f1
            cm=confusion_matrix(ylvh,rlvh); tn,fp,fn,tp=(cm.ravel() if cm.size==4 else (cm[0,0],0,0,0))
            W("lvh_sokolowlyon_sens",tp/(tp+fn+1e-9),blk="lvh")
            W("lvh_sokolowlyon_spec",tn/(tn+fp+1e-9),blk="lvh")
            W("lvh_sokolowlyon_ppv",tp/(tp+fp+1e-9),blk="lvh")
            W("lvh_sokolowlyon_npv",tn/(tn+fn+1e-9),blk="lvh")
            W("lvh_sokolowlyon_f1",_f1(ylvh,rlvh,zero_division=0),blk="lvh")
        except Exception as e: log.warning("[%s] LVH: %s",mid,e)

    # --- Block 7: PreSACAN ---
    log.info("[%s] Block 7: PreSACAN...",mid)
    for pn,li in [("V3",8),("V6",11)]:
        ytr=Y[:,li,1000:4000].max(-1); yrr=R[:,li,1000:4000].max(-1)
        upts=np.unique(pids)
        vgt=np.array([ytr[pids==p].var() for p in upts])
        vre=np.array([yrr[pids==p].var() for p in upts])
        ok2=vgt>=FLOOR_VAR
        if ok2.sum()>2:
            lr=np.log(vre[ok2]/(vgt[ok2]+1e-9))
            W(f"v{pn.lower()}_r_var_ret_pct",float(np.exp(lr.mean())*100),blk="presacan")
        if len(ytr)>2:
            sl,*_=stats.linregress(ytr,yrr); W(f"v{pn.lower()}_r_direct_slope",float(sl),blk="presacan")
            sl2,_,rps,*_=stats.linregress(ytr,yrr-ytr)
            W(f"v{pn.lower()}_r_presacan_slope",float(sl2),blk="presacan")
            W(f"v{pn.lower()}_r_presacan_r2",float(rps**2),blk="presacan")
    # Spurious coupling
    yi=Y[:,0,1000:4000].max(-1); yv3=Y[:,8,1000:4000].max(-1); rv3=R[:,8,1000:4000].max(-1)
    if len(yi)>2:
        _,_,rr_,*_=stats.linregress(yi,yv3); _,_,rrr,*_=stats.linregress(yi,rv3)
        r2g=float(rr_**2); r2r=float(rrr**2)
        W("spurious_coupling_r2_gt_I_V3",r2g,blk="presacan")
        W("spurious_coupling_r2_recon_I_V3",r2r,blk="presacan")
        if r2g>=0.01: W("spurious_coupling_ratio_v3",float(math.exp(math.log(r2r/(r2g+1e-9)))),blk="presacan")
        else: W("spurious_coupling_ratio_v3_diff",r2r-r2g,blk="presacan")
    # V3 T-wave
    ytv3t=Y[:,8,3000:4500].max(-1); yrv3t=R[:,8,3000:4500].max(-1)
    vgt_t=float(np.var(ytv3t)); vre_t=float(np.var(yrv3t))
    if vgt_t>FLOOR_VAR: W("v3_t_var_ret_pct",vre_t/vgt_t*100,blk="presacan")
    slt,*_=stats.linregress(ytv3t,yrv3t-ytv3t); W("v3_t_presacan_slope",float(slt),blk="presacan")
    W("avg_precordial_var_ret_pct",float(np.nanmean([np.var(R[:,li,1000:4000].max(-1))/(np.var(Y[:,li,1000:4000].max(-1))+1e-9)*100 for li in [6,7,8,9,10,11]])),blk="presacan")

    # --- Block 2: ECGFounder ---
    if (raw_dir/"founder_labels.npy").exists():
        log.info("[%s] Block 2: ECGFounder...",mid)
        fl=np.load(raw_dir/"founder_labels.npy")
        fg=np.load(raw_dir/"founder_gt_probs.npy")
        fr=np.load(raw_dir/"founder_recon_probs.npy")
        with open(raw_dir/"founder_task_names.json") as fh: tnames=json.load(fh)
        elig=np.where((fl>0.5).astype(int).sum(0)>=5)[0]
        lpt = (pt_groupby_mean((fl > 0.5).astype(np.float32), pids) > 0.5).astype(int)
        rpt = pt_groupby_mean(fr, pids)
        gpt = pt_groupby_mean(fg, pids)
        mac_r = fast_macro_auroc(lpt, rpt, elig)
        mac_g = fast_macro_auroc(lpt, gpt, elig)
        (cl_auc, ch_auc), (cl_del, ch_del) = boot_ci_auroc_and_delta(lpt, rpt, gpt, elig, n=nb_comp)
        W("ecgfounder_macro_150_auroc",mac_r,cl_auc,ch_auc,nb_comp,pw_=True,blk="ecgfounder")
        W("ecgfounder_delta_auroc_vs_gt",mac_r-mac_g,cl_del,ch_del,nb_comp,pw_=True,blk="ecgfounder")
        W("ecgfounder_pval_vs_gt",np.nan,lg_=True,blk="ecgfounder")
        # Superclass
        tu=[t.upper() for t in tnames]
        for scn,sct in [("arrhythmia",["ATRIAL_FIBRILLATION","SINUS_TACHYCARDIA","SUPRAVENTRICULAR"]),
                         ("conduction",["BUNDLE_BRANCH","AV_BLOCK","FASCICULAR","WOLFF"]),
                         ("hypertrophy",["VENTRICULAR_HYPERTROPHY","ATRIAL_ENLARGEMENT"]),
                         ("infarct",["INFARCT"])]:
            sub_elig = [c for c in elig if any(t in tu[c] for t in sct)]
            if sub_elig:
                sc_auc = fast_macro_auroc(lpt, rpt, sub_elig)
                W(f"ecgfounder_{scn}_auroc",sc_auc,pw_=True,blk="ecgfounder")
        # Brier score
        bs=[brier_score_loss(lpt[:,c], rpt[:,c]) for c in elig if lpt[:,c].sum()>=2]
        if bs: W("ecgfounder_brier_score",float(np.mean(bs)),pw_=True,blk="ecgfounder")

    # --- Block 3: EchoNext ---
    if (raw_dir/"echo_ref_probs.npy").exists():
        log.info("[%s] Block 3: EchoNext...",mid)
        eref=np.load(raw_dir/"echo_ref_probs.npy")
        erec=np.load(raw_dir/"echo_rec_probs.npy")
        elab=np.load(raw_dir/"echo_labels.npy")
        epid=np.load(raw_dir/"echo_patient_ids.npy")
        elpt = (pt_groupby_mean((elab > 0.5).astype(np.float32), epid) > 0.5).astype(int)
        erpt = pt_groupby_mean(erec, epid)
        egpt = pt_groupby_mean(eref, epid)
        echo_tasks = list(range(elpt.shape[1]))
        emac_r = fast_macro_auroc(elpt, erpt, echo_tasks)
        emac_g = fast_macro_auroc(elpt, egpt, echo_tasks)
        (cl_e_auc, ch_e_auc), (cl_e_del, ch_e_del) = boot_ci_auroc_and_delta(elpt, erpt, egpt, echo_tasks, n=nb_comp)
        W("echonext_shd_macro_12_auroc",emac_r,cl_e_auc,ch_e_auc,nb_comp,pw_=True,blk="echonext",ds="echonext")
        W("echonext_delta_auroc_vs_gt",emac_r-emac_g,cl_e_del,ch_e_del,nb_comp,pw_=True,blk="echonext",ds="echonext")
        W("echonext_pval_vs_gt",np.nan,lg_=True,blk="echonext",ds="echonext")
        ebs=[brier_score_loss(elpt[:,ti], erpt[:,ti]) for ti in range(elab.shape[1]) if elpt[:,ti].sum()>=2]
        if ebs: W("echonext_brier_score",float(np.mean(ebs)),pw_=True,blk="echonext",ds="echonext")

    log.info("[%s] CPU aggregation complete.",mid)

# ============================================================
# Main
# ============================================================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-id",required=True)
    ap.add_argument("--checkpoint-path",required=True)
    ap.add_argument("--mode",choices=["gpu-infer","cpu-aggregate","full"],default="full")
    ap.add_argument("--raw-dir",default=None)
    ap.add_argument("--db-path",default="results/sap_benchmark_v2/sap_metrics.db")
    ap.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    dev=torch.device(a.device)
    db=_ROOT/a.db_path
    ckpt=Path(a.checkpoint_path); ckpt=ckpt if ckpt.is_absolute() else _ROOT/ckpt
    raw=Path(a.raw_dir) if a.raw_dir else Path(f"/tmp/sap_raw_{a.model_id}")
    if a.mode in ("gpu-infer","full"): gpu_infer(a.model_id,ckpt,raw,dev,a.batch_size,a.smoke)
    if a.mode in ("cpu-aggregate","full"): cpu_aggregate(a.model_id,raw,db,a.smoke)
    if a.mode=="full" and not a.smoke:
        import shutil; shutil.rmtree(raw,ignore_errors=True)
        log.info("[%s] Raw arrays cleaned.",a.model_id)

if __name__=="__main__": main()
