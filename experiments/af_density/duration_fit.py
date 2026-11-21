"""Censored discrete-lognormal duration likelihood for the binary HSMM."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp


def duration_log_laws(mu, sigma, support):
    """Round positive continuous durations to steps; last bin contains the tail.

    Bin 1 is (0,1.5), bin d is [d-.5,d+.5), final bin [support-.5,inf).
    The last bin is an explicitly declared duration cap, not an infinite tail.
    Stable log differences avoid arbitrary probability floors during fitting.
    """
    if not np.isfinite([mu, sigma]).all() or sigma <= 0 or not isinstance(support, int) or support < 2:
        raise ValueError('Invalid lognormal duration parameters or support')
    boundaries = np.r_[0., np.arange(1.5, support, 1), np.inf]
    with np.errstate(divide='ignore'):
        z = (np.log(boundaries)-mu)/sigma
    log_cdf, log_sf = log_ndtr(z), log_ndtr(-z)
    use_sf = z[:-1] >= 0
    high = np.where(use_sf, log_sf[:-1], log_cdf[1:])
    low = np.where(use_sf, log_sf[1:], log_cdf[:-1])
    log_pmf = high + np.log(-np.expm1(low-high))
    log_survival = np.logaddexp.accumulate(log_pmf[::-1])[::-1]
    log_mean = logsumexp(log_pmf+np.log(np.arange(1, support+1)))
    log_residual = log_survival-log_mean
    log_residual_survival = np.logaddexp.accumulate(log_residual[::-1])[::-1]
    return log_pmf, log_survival, log_residual, log_residual_survival


def fit_duration_law(lengths, left_censored, right_censored, weights, *, support):
    lengths = np.asarray(lengths)
    left = np.asarray(left_censored)
    right = np.asarray(right_censored)
    weights = np.asarray(weights, dtype=np.float64)
    if (lengths.ndim != 1 or not len(lengths) or lengths.dtype.kind not in 'iu' or
            (lengths < 1).any() or (lengths > support).any() or
            left.shape != lengths.shape or right.shape != lengths.shape or
            left.dtype != bool or right.dtype != bool or weights.shape != lengths.shape or
            not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError('Invalid observed durations, censoring or likelihood weights')
    if not (~left & ~right).any():
        raise ValueError('At least one fully observed run required for duration fitting')
    indices = lengths-1
    def objective(parameters):
        laws = duration_log_laws(parameters[0], np.exp(parameters[1]), support)
        category = left.astype(np.int8)*2+right.astype(np.int8)
        log_probability = np.stack(laws)[category, indices]
        return -float(np.dot(weights, log_probability)/weights.sum())
    mu = float(np.average(np.log(lengths), weights=weights))
    bounds = [(np.log(.1), np.log(2*support)), (np.log(.1), np.log(5.))]
    result = minimize(objective, [mu, 0.], method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Duration fit failed: '+str(result.message))
    laws = duration_log_laws(result.x[0], np.exp(result.x[1]), support)
    pmf = np.exp(laws[0])
    assert np.isclose(pmf.sum(), 1, rtol=0, atol=1e-12)
    return {
        'mu_log_steps': float(result.x[0]), 'sigma_log_steps': float(np.exp(result.x[1])),
        'weighted_mean_negative_log_likelihood': float(result.fun),
        'optimizer_iterations': int(result.nit), 'optimizer_message': str(result.message),
        'parameter_near_bound': [bool(abs(value-lo) < 1e-5 or abs(value-hi) < 1e-5)
                                 for value, (lo, hi) in zip(result.x, bounds)],
        'runs': len(lengths), 'fully_observed_runs': int((~left & ~right).sum()),
        'left_only_runs': int((left & ~right).sum()), 'right_only_runs': int((~left & right).sum()),
        'both_censored_runs': int((left & right).sum()), 'pmf': pmf,
    }
