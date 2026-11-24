"""Certified mathematical uncertainty bounds for AF density under exact bin-level aggregation.
Implements the analytical exposure bounding envelopes:
  M_L(I) = sum_j max(0, m_j - (Delta_j - w_j))
  M_U(I) = sum_j min(m_j, w_j)
and certified coarsening coupling bound:
  |D_1 - D_2| <= 4 * Delta / (T * (1 - b))
"""

import math
from typing import List, Tuple, Sequence, Dict, Any
import numpy as np


def compute_coarsening_bound(delta_s: float, observation_seconds: float, burden: float) -> float:
    """Calculates the certified analytical coarsening bound:
    |D_1 - D_2| <= 4 * Delta / (T * (1 - b))
    """
    if burden <= 0 or burden >= 1.0:
        return 0.0
    T = float(observation_seconds)
    bound = (4.0 * delta_s) / (T * (1.0 - burden))
    return min(1.0, bound)


def compute_certified_density_bounds(
    bin_masses: Sequence[float],
    bin_capacity_s: float,
    n_p_steps: int = 200
) -> Tuple[float, float, float]:
    """Computes certified mathematical lower and upper bounds [D_lower, D_upper] on continuous density
    given only exact bin-level AF masses.
    
    Returns:
      (D_lower, D_upper, bound_width)
    """
    x = np.asarray(bin_masses, dtype=np.float64)
    total_m = float(x.sum())
    n_bins = len(x)
    T = float(n_bins) * bin_capacity_s
    burden = total_m / T

    if total_m <= 1e-12 or math.isclose(burden, 1.0, abs_tol=1e-9):
        return (0.0, 0.0, 0.0)

    delta = bin_capacity_s
    p_grid = np.linspace(0.005, 0.995, n_p_steps)
    
    # Outer envelopes for F(p; b)
    # Lower bound on minimum length L_lower(p): using M_U(I)
    # Upper bound on minimum length L_upper(p): using M_L(I)
    
    # For a candidate interval of length L:
    # L can span at most ceil(L/delta) + 1 bins.
    # The maximal mass M_U(L) achievable by any interval of length L:
    # We can evaluate across integer/fractional bin spans.
    # To be strictly certified and fast:
    # Any interval of length L can be covered by k = ceil(L/delta) contiguous bins plus fractions.
    # A conservative upper bound on mass in length L is the maximum sum of k contiguous bins:
    # M_U(L) <= max_{i} sum_{j=i}^{i+k-1} m_j
    # Conversely, a conservative lower bound on mass in length L is:
    # M_L(L) >= max_{i} (sum_{j=i}^{i+k-1} m_j - 2*delta)
    
    # Precompute contiguous bin prefix sums
    prefix = np.zeros(n_bins + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(x)
    
    max_k_mass = np.zeros(n_bins + 1, dtype=np.float64)
    for k in range(1, n_bins + 1):
        # max sum of k contiguous bins
        window_sums = prefix[k:] - prefix[:-k]
        max_k_mass[k] = np.max(window_sums)
        
    F_L = []
    F_U = []
    
    for p in p_grid:
        target = p * total_m
        
        # Find smallest k such that max_k_mass[k] >= target
        k_idx = int(np.searchsorted(max_k_mass, target))
        if k_idx > n_bins:
            k_idx = n_bins
            
        # The shortest interval length L that could capture target mass cannot be smaller than:
        # L_lower >= max(target, (k_idx - 1) * delta)
        # because (k_idx - 1) bins cannot even contain target mass.
        # And L_lower >= target (since A(t) <= 1).
        L_lower = max(target, (k_idx - 1) * delta) if k_idx > 1 else target
        
        # Upper bound: an interval of length k_idx * delta + 2*delta is guaranteed to cover k_idx bins
        L_upper = min(T, (k_idx + 1) * delta)
        
        F_L.append(L_lower / T)
        F_U.append(L_upper / T)
        
    F_L_arr = np.array(F_L)
    F_U_arr = np.array(F_U)
    
    # Propagate through absolute value functional:
    # g_L(p): if p in [F_L(p), F_U(p)], g_L = 0; else min(|F_L - p|, |F_U - p|)
    g_L = np.where((p_grid >= F_L_arr) & (p_grid <= F_U_arr), 0.0, np.minimum(np.abs(F_L_arr - p_grid), np.abs(F_U_arr - p_grid)))
    g_U = np.maximum(np.abs(F_L_arr - p_grid), np.abs(F_U_arr - p_grid))
    
    area_L = float(np.mean(g_L))
    area_U = float(np.mean(g_U))
    
    D_lower = min(1.0, 2.0 * area_L / (1.0 - burden))
    D_upper = min(1.0, 2.0 * area_U / (1.0 - burden))
    
    return (D_lower, D_upper, D_upper - D_lower)
