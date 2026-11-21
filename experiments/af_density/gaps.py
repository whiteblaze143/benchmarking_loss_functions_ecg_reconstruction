"""Explicit published shutdown strategies; observed-only changes the clock."""
import numpy as np


def apply_gap_strategy(reference, predicted, missing, strategy):
    r=np.asarray(reference);p=np.asarray(predicted);m=np.asarray(missing)
    if r.ndim!=1 or p.shape!=r.shape or m.shape!=r.shape or m.dtype!=bool or not len(r):
        raise ValueError('Expected equal nonempty vectors and boolean missing mask')
    if not np.isin(r,[0,1]).all() or not np.isin(p,[0,1]).all():
        raise ValueError('Expected binary reference and prediction')
    if strategy=='observed_only':
        if m.all():
            return {'status':'indeterminate_no_observed_time','clock':'compressed_observed_bins','reference':None,'predicted':None}
        return {'status':'ok','clock':'compressed_observed_bins','reference':r[~m].copy(),'predicted':p[~m].copy()}
    if strategy=='nonaf':
        x=p.copy();x[m]=0
        return {'status':'ok','clock':'original_bins','reference':r.copy(),'predicted':x}
    if strategy!='context':raise ValueError('Unknown gap strategy')
    if m[0] or m[-1]:
        return {'status':'indeterminate_missing_flanking_context','clock':'original_bins','reference':r.copy(),'predicted':None}
    x=p.copy()
    starts=np.flatnonzero(m & ~np.r_[False,m[:-1]])
    ends=np.flatnonzero(m & ~np.r_[m[1:],False])+1
    for lo,hi in zip(starts,ends):
        left=p[lo-1];right=p[hi]
        if left==right:x[lo:hi]=left
        else:
            # For odd gaps the midpoint bin belongs to the right half.
            mid=lo+(hi-lo)//2
            x[lo:mid]=left;x[mid:hi]=right
    return {'status':'ok','clock':'original_bins','reference':r.copy(),'predicted':x}
