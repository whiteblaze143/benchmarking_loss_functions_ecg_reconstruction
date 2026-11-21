"""Fixed local RR features; no learned transforms or rhythm labels."""
import numpy as np
NAMES=['valid_rr_fraction','valid_pair_fraction','mean_rr_ms','median_rr_ms','sdnn_ms','rmssd_ms','pnn50','pnn20','cv','iqr_ms','q10_ms','q90_ms','mad_ms','range_ms','turning_point_fraction','hist_entropy']

def layout(length,excluded):
    if length-excluded<60:raise ValueError('Daily file has fewer than 60 RR intervals after exclusion')
    starts=list(range(excluded,length-59,60));assigned=starts.copy();ends=[s+60 for s in starts]
    if ends[-1]<length:
        starts.append(length-60);assigned.append(ends[-1]);ends.append(length)
    return np.array(starts),np.array(assigned),np.array(ends)

def features(windows):
    w=np.asarray(windows,dtype=float)
    if w.ndim!=2 or w.shape[1]!=60:raise ValueError('Expected actual 60-RR windows')
    valid=np.isfinite(w)&(w>=200)&(w<=4000)
    pair=valid[:,:-1]&valid[:,1:]
    usable=(valid.sum(1)>=4)&(pair.sum(1)>=2)
    out=np.full((len(w),len(NAMES)),np.nan)
    if not usable.any():return out,usable
    a=w[usable];v=valid[usable];pa=pair[usable];masked=np.where(v,a,np.nan)
    diff=np.diff(a,axis=1);d=np.where(pa,diff,np.nan)
    mean=np.nanmean(masked,axis=1);median=np.nanmedian(masked,axis=1);sd=np.nanstd(masked,axis=1,ddof=1)
    q=np.nanquantile(masked,[.1,.25,.75,.9],axis=1)
    triples=v[:,:-2]&v[:,1:-1]&v[:,2:]
    turns=((a[:,1:-1]-a[:,:-2])*(a[:,2:]-a[:,1:-1])<0)&triples
    turn=np.divide(turns.sum(1),triples.sum(1),out=np.full(len(a),np.nan),where=triples.sum(1)>0)
    hist=np.zeros((len(a),16),dtype=float);edges=np.linspace(200,4000,17)
    ri,ci=np.where(v);bins=np.minimum(np.searchsorted(edges,a[ri,ci],side='right')-1,15)
    np.add.at(hist,(ri,bins),1);prob=hist/hist.sum(1)[:,None]
    log=np.zeros_like(prob);np.log(prob,out=log,where=prob>0)
    out[usable]=np.column_stack([v.mean(1),pa.mean(1),mean,median,sd,np.sqrt(np.nanmean(d*d,axis=1)),((np.abs(diff)>50)&pa).sum(1)/pa.sum(1),((np.abs(diff)>20)&pa).sum(1)/pa.sum(1),sd/mean,q[2]-q[1],q[0],q[3],np.nanmedian(np.abs(masked-median[:,None]),axis=1),np.nanmax(masked,axis=1)-np.nanmin(masked,axis=1),turn,-(prob*log).sum(1)])
    return out,usable
