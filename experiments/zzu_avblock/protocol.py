"""Locked endpoint, lead sets and evaluation primitives for ZZU-ZUA AV block."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
CONFIGS={'full12':LEADS,'I':['I'],'II':['II'],'V1':['V1'],'I_II':['I','II'],'II_V1':['II','V1'],'limb6':LEADS[:6],'precordial6':LEADS[6:]}
ENDPOINT='High-Grade/Third-Degree Atrioventricular Block'
RHYTHM={'Ventricular Flutter','Ventricular Fibrillation','Torsade de Pointes','Bidirectional Ventricular Tachycardia','Polymorphic Ventricular Tachycardia','Atrial Fibrillation with Pre-excitation','Long RR Interval (>=3 seconds)','R-on-T Ventricular Premature Contractions'}

def unit_normalize_windows(x):
    x=np.asarray(x,dtype=np.float32)
    x=x-x.mean(axis=-1,keepdims=True)
    sd=x.std(axis=-1,keepdims=True)
    x=np.divide(x,sd,out=np.zeros_like(x),where=sd>0)
    flat=x.reshape(len(x),-1)
    norm=np.linalg.norm(flat,axis=1,keepdims=True)
    return np.divide(flat,norm,out=np.zeros_like(flat),where=norm>0)

def candidates(features):
    """Eight deterministic 16-bit SimHash bands; approximate retrieval only."""
    rng=np.random.default_rng(42)
    bits=features@rng.standard_normal((features.shape[1],128),dtype=np.float32)>0
    bands=np.packbits(bits,axis=1).copy().view('<u2')
    pairs=set()
    for band in range(8):
        order=np.argsort(bands[:,band],kind='stable');v=bands[order,band]
        cuts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1,len(v)]
        for start,end in zip(cuts[:-1],cuts[1:]):
            ids=order[start:end]
            for i,a in enumerate(ids):
                pairs.update((int(min(a,b)),int(max(a,b))) for b in ids[i+1:])
        if len(pairs)>5000000:raise RuntimeError('Near-duplicate candidate bound exceeded; no split may be made')
    return np.array(sorted(pairs),dtype=np.int32).reshape(-1,2)

def locked_threshold(y,s,target=.95):
    """Lowest score threshold satisfying development specificity, ties preserved."""
    y=np.asarray(y);s=np.asarray(s)
    if not np.isfinite(s).all() or not (y==0).any():raise ValueError('Invalid development scores')
    negatives=np.sort(s[y==0])[::-1]
    allowed=int(np.floor((1-target)*len(negatives)+1e-9))
    return float(np.nextafter(negatives[allowed],np.inf))

def metrics(y,score,status,threshold):
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float);status=np.asarray(status)
    valid=(status=='completed')&np.isfinite(score);pred=valid&(score>=threshold)
    pos=y==1;neg=~pos
    out={'n':len(y),'positive_n':int(pos.sum()),'negative_n':int(neg.sum()),'completed_n':int(valid.sum()),
         'failed_or_indeterminate_n':int((~valid).sum()),'sensitivity_all_eligible':float(pred[pos].mean()) if pos.any() else None,
         'inappropriate_positive_fraction_all_controls':float(pred[neg].mean()) if neg.any() else None,
         'specificity_completed':float((~pred[neg&valid]).mean()) if (neg&valid).any() else None,
         'positive_coverage':float(valid[pos].mean()) if pos.any() else None,'negative_coverage':float(valid[neg].mean()) if neg.any() else None}
    if len(np.unique(y[valid]))==2:
        out.update(auroc=float(roc_auc_score(y[valid],score[valid])),auprc=float(average_precision_score(y[valid],score[valid])),brier=float(brier_score_loss(y[valid],score[valid])))
    return out
