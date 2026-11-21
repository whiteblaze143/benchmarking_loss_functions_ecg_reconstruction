"""Local observed-voltage processing; never reads other channels or labels."""
import numpy as np
from wfdb.processing import xqrs_detect
from rr_features import NAMES


def rr_statistics(rr_ms):
    rr=np.asarray(rr_ms,dtype=np.float64)
    if rr.ndim!=1:raise ValueError('RR intervals must be one-dimensional')
    v=np.isfinite(rr)&(rr>=200)&(rr<=4000)
    pair=v[:-1]&v[1:]
    result=np.full(len(NAMES),np.nan,dtype=np.float32)
    if v.sum()<4 or pair.sum()<2:return result,False
    good=rr[v];diff=np.diff(rr);d=diff[pair];q=np.quantile(good,[.1,.25,.75,.9]);mean=good.mean();median=np.median(good)
    triple=v[:-2]&v[1:-1]&v[2:]
    turning=((rr[1:-1]-rr[:-2])*(rr[2:]-rr[1:-1])<0)&triple
    turn=float(turning.sum()/triple.sum()) if triple.any() else np.nan
    hist=np.histogram(good,bins=np.linspace(200,4000,17))[0];prob=hist[hist>0]/len(good)
    result[:]=[v.mean(),pair.mean(),mean,median,good.std(ddof=1),np.sqrt(np.mean(d*d)),np.mean(abs(d)>50),np.mean(abs(d)>20),good.std(ddof=1)/mean,q[2]-q[1],q[0],q[3],np.median(abs(good-median)),good.max()-good.min(),turn,-np.sum(prob*np.log(prob))]
    return result,True


def extract_channel(signal):
    x=np.asarray(signal,dtype=np.float32)
    if x.shape!=(6000,):raise ValueError('Expected exactly one observed 30s channel at 200Hz')
    status=0;error='';peaks=np.empty(0,dtype=np.int32)
    if not np.isfinite(x).all():status=2;error='nonfinite_input'
    elif np.ptp(x)==0:status=1;error='constant_input'
    else:
        try:
            peaks=np.asarray(xqrs_detect(x,fs=200,learn=True,verbose=False),dtype=np.int32)
            if peaks.ndim!=1 or (len(peaks) and (peaks[0]<0 or peaks[-1]>=6000 or not (np.diff(peaks)>0).all())):
                raise RuntimeError('Detector returned invalid local sample coordinates')
        except (ValueError,IndexError,RuntimeError) as e:
            peaks=np.empty(0,dtype=np.int32);status=3;error=type(e).__name__+': '+str(e)
    samples=np.diff(peaks);rr=samples.astype(np.float64)*5
    accepted=(samples>=60)&(samples<=400)
    features,rr_valid=rr_statistics(rr)
    return {'peaks':peaks,'rr_features':features,'rr_valid':rr_valid,'phase_valid':int(accepted.sum())>=2,'accepted_intervals':int(accepted.sum()),'represented_fraction':float(samples[accepted].sum()/6000),'status':status,'error':error}
