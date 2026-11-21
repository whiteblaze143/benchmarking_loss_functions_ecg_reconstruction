"""Explicit continuous-time Charitos density and small-array discrete references.

Continuous density follows the minimum contiguous observation definition.
Discrete references implement Butkuviene et al., TBME 2023, equations 3--5.
They use beats, not elapsed time. Only the literal short-array reference is quadratic.
No-AF and continuous-AF density are undefined (None), never silently zero.
"""
import math
from bisect import bisect_left


def _absolute_linear_integral(slope, intercept, lo, hi):
    def primitive(x):
        return slope * x * x / 2 + intercept * x
    root = -intercept / slope if slope else math.inf
    if lo < root < hi:
        return abs(primitive(root) - primitive(lo)) + abs(primitive(hi) - primitive(root))
    return abs(primitive(hi) - primitive(lo))


def charitos_density(intervals, observation_seconds):
    """Integrate the exact lower envelope for sorted, disjoint AF intervals.

    Input intervals are half-open seconds on one uninterrupted observation clock.
    Missing/excluded time must be handled by the caller, never concatenated here.
    Every possible consecutive episode block contributes its accumulated AF
    duration and intervening non-AF gap. For AF amount x, the shortest window
    is x plus the smallest gap of a block with total AF duration >= x.
    This step envelope is integrated exactly, including its discontinuities.
    """
    t = float(observation_seconds)
    if not math.isfinite(t) or t <= 0:
        raise ValueError('Observation duration must be finite and positive')
    episodes = [(float(a), float(b)) for a, b in intervals]
    previous = 0.0
    for i, (a, b) in enumerate(episodes):
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= t):
            raise ValueError('Invalid AF interval')
        if i and a <= previous:
            raise ValueError('AF intervals must be sorted, separated and nonoverlapping')
        previous = b
    total = math.fsum(b - a for a, b in episodes)
    if total == 0 or total == t:
        return None
    durations = [b-a for a,b in episodes]
    prefix = [0.0] + [math.fsum(durations[:i]) for i in range(1,len(durations)+1)]
    blocks = []
    for i, (start, _) in enumerate(episodes):
        for j in range(i, len(episodes)):
            accumulated = durations[i] if i==j else prefix[j+1]-prefix[i]
            gap = 0.0 if i==j else episodes[j][1]-start-accumulated
            blocks.append((accumulated,gap))
    blocks.sort()
    amounts = [amount for amount,_ in blocks]
    suffix_gap = [0.0]*len(blocks)
    smallest = math.inf
    for i in range(len(blocks)-1,-1,-1):
        smallest = min(smallest,blocks[i][1])
        suffix_gap[i] = smallest
    cuts = sorted({0.0,total,*amounts})
    area = 0.0
    for lo,hi in zip(cuts,cuts[1:]):
        mid = (lo+hi)/2
        gap = suffix_gap[bisect_left(amounts,mid)]
        area += _absolute_linear_integral(1/t-1/total,gap/t,lo,hi)/total
    return 2 * area / (1 - total/t)


def discrete_pattern_reference(binary):
    """Literal equations 3--5; intended only for short verification fixtures."""
    x = list(binary)
    if any(v not in (0, 1) for v in x) or not x:
        raise ValueError('Expected nonempty binary sequence')
    n, k = len(x), sum(x)
    if not 0 < k < n:
        return {'density': None, 'aggregation': None}
    prefix = [0]
    for value in x:
        prefix.append(prefix[-1] + value)
    maximum = [max(prefix[min(i+length,n)]-prefix[i] for i in range(n))
               for length in range(1,n+1)]
    shortest = [next(i+1 for i,value in enumerate(maximum) if value >= p)
                for p in range(1,k+1)]
    aggregation = 2/(n*k)*sum(abs(value-length*k/n)
                              for length,value in enumerate(maximum,1))
    density = 2*n/((k+1)*(n-k))*sum(abs(length/n-p/k)
                                   for p,length in enumerate(shortest,1))
    return {'density': density, 'aggregation': aggregation}


def discrete_pattern_from_runs(intervals, observation_beats):
    """Exact equations 3--5 from half-open AF beat runs, without O(N**2)."""
    import numpy as np
    n = observation_beats
    if not isinstance(n, int) or n <= 0:
        raise ValueError('Expected positive integer beat count')
    runs = list(intervals)
    previous = -1
    for a,b in runs:
        if not isinstance(a,int) or not isinstance(b,int) or not 0 <= a < b <= n or a <= previous:
            raise ValueError('Expected sorted separated integer AF runs')
        previous = b
    k = sum(b-a for a,b in runs)
    if not 0 < k < n:
        return {'density': None, 'aggregation': None}
    blocks = []
    for i,(start,_) in enumerate(runs):
        amount = 0
        for j in range(i,len(runs)):
            amount += runs[j][1]-runs[j][0]
            blocks.append((amount,runs[j][1]-start-amount))
    blocks.sort()
    suffix = [0]*len(blocks)
    smallest = n
    for i in range(len(blocks)-1,-1,-1):
        smallest = min(smallest,blocks[i][1])
        suffix[i] = smallest
    gaps = np.empty(k,dtype=np.int64)
    previous = 0
    for i,(amount,_) in enumerate(blocks):
        if amount>previous:
            gaps[previous:amount] = suffix[i]
            previous = amount
    shortest = np.arange(1,k+1,dtype=np.int64)+gaps
    maximum = np.cumsum(np.bincount(shortest,minlength=n+1))[1:]
    aggregation = 2/(n*k)*np.abs(maximum-np.arange(1,n+1)*k/n).sum()
    density = 2*n/((k+1)*(n-k))*np.abs(shortest/n-np.arange(1,k+1)/k).sum()
    return {'density':float(density),'aggregation':float(aggregation)}


def charitos_author_binned_port(data, timeunits=1440, minim=0.001, limit=True):
    """Port of Charitos's public 2019 R v3 function; retains its conventions.

    Inputs are AF amounts per equal observation bin, with timeunits the bin's
    capacity in the same units. This is not continuous interval density.
    """
    import numpy as np
    x = np.asarray(data,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not math.isfinite(timeunits) or timeunits<=0 or np.any(x<0) or np.any(x>timeunits):
        raise ValueError('Invalid binned AF data')
    if not math.isfinite(minim) or not 0<minim<1:
        raise ValueError('Invalid integration step')
    steps=round(1/minim)
    if not math.isclose(steps*minim,1) or steps<2:
        raise ValueError('Integration step must partition the unit interval')
    total=float(x.sum())
    if total==0:return 0.0
    proportions=x/total
    area=0.0
    for m in np.arange(1,steps)*minim:
        amount=0.0;start=0;shortest=len(x)+1
        for end,value in enumerate(proportions):
            amount+=value
            while amount>=m:
                shortest=min(shortest,end-start+1)
                amount-=proportions[start];start+=1
        area+=abs(m-shortest/len(x))*minim
    burden=total/timeunits/len(x)
    raw=2*area/(1-burden) if burden!=1 else math.inf
    return min(raw,1.0) if limit else raw
