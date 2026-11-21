"""Binary explicit-duration HSMM Viterbi core; no parameter fitting or tuning.

Durations are ordered window steps. A partial final assignment remains one
step; callers must retain its actual duration for elapsed-time evaluation.
Unknown windows must split decoding into observed blocks, not receive fabricated
emissions. Distributions and initial occupancy must come from development data.
"""
import numpy as np


def hsmm_viterbi(log_emissions, duration_pmf, initial_occupancy, *,
                 left_censored, right_censored):
    """Decode alternating non-AF/AF states with explicit endpoint censoring.

    pmf[state,d-1] is P(run length=d), normalized on its declared finite support.
    Censored starts use the equilibrium residual-life distribution S(d)/E[D];
    censored ends use survival S(d). A run spanning both boundaries uses residual
    survival. This assumes random observation within a stationary state run.
    Initial occupancy is the state probability at observation start, not the
    probability of a new run's class. Full positive probability at d=1 is not
    required; impossible paths fail instead of falling back to D0.
    """
    emission = np.asarray(log_emissions, dtype=np.float64)
    pmf = np.asarray(duration_pmf, dtype=np.float64)
    prior = np.asarray(initial_occupancy, dtype=np.float64)
    if emission.ndim != 2 or emission.shape[1] != 2 or not len(emission) or not np.isfinite(emission).all():
        raise ValueError('Expected finite (windows,2) log emissions')
    if (pmf.ndim != 2 or pmf.shape[0] != 2 or not pmf.shape[1] or
            not np.isfinite(pmf).all() or (pmf < 0).any() or
            not np.allclose(pmf.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError('Expected normalized nonnegative duration PMFs')
    if prior.shape != (2,) or not np.isfinite(prior).all() or (prior <= 0).any() or not np.isclose(prior.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError('Expected positive normalized initial state occupancy')
    if not isinstance(left_censored, bool) or not isinstance(right_censored, bool):
        raise ValueError('Endpoint censoring must be explicit booleans')
    survival = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
    residual = survival / survival.sum(axis=1, keepdims=True)
    residual_survival = np.cumsum(residual[:, ::-1], axis=1)[:, ::-1]
    with np.errstate(divide='ignore'):
        log_pmf, log_survival = np.log(pmf), np.log(survival)
        log_residual, log_residual_survival = np.log(residual), np.log(residual_survival)
    n = len(emission)
    cumulative = np.vstack((np.zeros(2), np.cumsum(emission, axis=0)))
    score = np.full((n+1, 2), -np.inf)
    previous = np.full((n+1, 2), -1, dtype=np.int64)
    for stop in range(1, n+1):
        lengths = np.arange(1, min(stop, pmf.shape[1])+1)
        starts = stop-lengths
        for state in [0, 1]:
            log_duration = (log_survival if stop == n and right_censored else log_pmf)[state, lengths-1].copy()
            initial = starts == 0
            if left_censored and initial.any():
                law = log_residual_survival if stop == n and right_censored else log_residual
                log_duration[initial] = law[state, lengths[initial]-1]
            preceding = score[starts, 1-state].copy()
            preceding[initial] = np.log(prior[state])
            candidates = preceding + log_duration + cumulative[stop, state]-cumulative[starts, state]
            best = int(np.argmax(candidates))
            score[stop, state] = candidates[best]
            previous[stop, state] = starts[best]
    state = int(np.argmax(score[n]))
    if not np.isfinite(score[n, state]):
        raise ValueError('No feasible HSMM path on the declared duration support')
    best_score = float(score[n, state])
    states = np.empty(n, dtype=np.int8)
    stop = n
    while stop:
        start = previous[stop, state]
        assert 0 <= start < stop
        states[start:stop] = state
        stop, state = start, 1-state
    return {'states': states, 'log_score': best_score}
