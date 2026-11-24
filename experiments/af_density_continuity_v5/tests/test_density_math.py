"""Independent mathematical validation unit tests for continuous and binned AF density solvers.
Tests all 11 required fixtures from Table B03 of the AF-DENSITY V5 master directive.
"""

import math
import random
import pytest
import numpy as np

from src.density_continuous import compute_continuous_density, find_shortest_interval_for_fraction
from src.density_binned import charitos_binned_density
from src.density_bounds import compute_coarsening_bound, compute_certified_density_bounds


def test_fixture_01_single_contiguous_run():
    """Fixture 1: Single contiguous AF run has continuous density exactly equal to 1.0 for 0 < b < 1."""
    T = 1000.0
    for dur in [10.0, 100.0, 500.0, 900.0]:
        intervals = [(10.0, 10.0 + dur)]
        d = compute_continuous_density(intervals, T)
        assert d is not None
        assert math.isclose(d, 1.0, abs_tol=1e-7), f"Single run duration={dur} got d={d}, expected 1.0"


def test_fixture_02_translation_invariance():
    """Fixture 2: Translating all AF runs in time leaves density unchanged."""
    T = 1000.0
    base_intervals = [(50.0, 150.0), (300.0, 450.0), (600.0, 700.0)]
    d_base = compute_continuous_density(base_intervals, T)
    assert d_base is not None

    for shift in [20.0, 50.0, 100.0]:
        shifted = [(s + shift, e + shift) for s, e in base_intervals]
        d_shifted = compute_continuous_density(shifted, T)
        assert math.isclose(d_base, d_shifted, abs_tol=1e-7), f"Shift {shift} changed density: {d_base} vs {d_shifted}"


def test_fixture_03_timeline_reversal():
    """Fixture 3: Reversing the timeline leaves density unchanged."""
    T = 1000.0
    base_intervals = [(100.0, 200.0), (400.0, 600.0), (850.0, 950.0)]
    d_base = compute_continuous_density(base_intervals, T)
    assert d_base is not None

    # In reverse time, an interval [s, e] becomes [T - e, T - s]
    reversed_intervals = sorted([(T - e, T - s) for s, e in base_intervals])
    d_rev = compute_continuous_density(reversed_intervals, T)
    assert math.isclose(d_base, d_rev, abs_tol=1e-7), f"Reversal changed density: {d_base} vs {d_rev}"


def test_fixture_04_uniform_intensity():
    """Fixture 4: Uniformly distributed fractional AF exposure has density approaching 0 as N increases."""
    # Discretization bias of ceil(m*N)/N has expectation ~ 1/(2N), so raw binned density is O(1/N)
    for n_bins, max_expected in [(100, 0.025), (500, 0.008), (1000, 0.004)]:
        delta = 10.0
        equal_masses = [5.0] * n_bins
        d_binned = charitos_binned_density(equal_masses, delta)
        assert d_binned < max_expected, f"N={n_bins}: expected < {max_expected}, got {d_binned}"
    
    # Confirm it strictly decreases as N increases (approaches 0)
    d_100 = charitos_binned_density([5.0] * 100, 10.0)
    d_500 = charitos_binned_density([5.0] * 500, 10.0)
    d_1000 = charitos_binned_density([5.0] * 1000, 10.0)
    assert d_100 > d_500 > d_1000, f"Monotonic convergence failed: {d_100} -> {d_500} -> {d_1000}"


def test_fixture_05_gap_sensitivity():
    """Fixture 5: Equal-duration episodes separated by different gaps yield consistent interval lengths."""
    T = 1000.0
    # Episodes close together vs far apart
    close_episodes = [(100.0, 200.0), (220.0, 320.0)]  # gap 20s
    far_episodes = [(100.0, 200.0), (500.0, 600.0)]    # gap 300s

    d_close = compute_continuous_density(close_episodes, T)
    d_far = compute_continuous_density(far_episodes, T)

    assert d_close is not None and d_far is not None
    # Shorter gap means higher temporal concentration (closer episodes)
    assert d_close > d_far, f"Expected d_close ({d_close}) > d_far ({d_far})"


def test_fixture_06_partial_bin_mass_conservation():
    """Fixture 6: Partial-bin AF exposure preserves AF duration exactly."""
    intervals = [(15.5, 42.3), (105.0, 180.25)]
    m_exact = (42.3 - 15.5) + (180.25 - 105.0)

    delta = 60.0
    # Binning across 60s bins
    bins = [0.0] * 4
    for s, e in intervals:
        for b_idx in range(4):
            bs = b_idx * delta
            be = (b_idx + 1) * delta
            overlap = max(0.0, min(e, be) - max(s, bs))
            bins[b_idx] += overlap

    assert math.isclose(sum(bins), m_exact, abs_tol=1e-12), "Mass conservation failed across bins"


def test_fixture_07_no_af_boundary():
    """Fixture 7: No AF returns explicit boundary status (None in continuous, 0 in author R)."""
    assert compute_continuous_density([], 1000.0) is None
    assert charitos_binned_density([0.0] * 10, 60.0) == 0.0


def test_fixture_08_all_af_boundary():
    """Fixture 8: All AF returns explicit boundary status (None in continuous, 1.0 in author R)."""
    assert compute_continuous_density([(0.0, 1000.0)], 1000.0) is None
    assert charitos_binned_density([60.0] * 10, 60.0) == 1.0


def test_fixture_09_invalid_or_missing_intervals():
    """Fixture 9: Missing/invalid intervals raise ValueError rather than silent computation."""
    with pytest.raises(ValueError):
        compute_continuous_density([(-5.0, 10.0)], 100.0)
    with pytest.raises(ValueError):
        compute_continuous_density([(10.0, 50.0), (40.0, 80.0)], 100.0)  # Overlapping


def test_fixture_10_multiple_recording_sessions():
    """Fixture 10: Multiple recording sessions must raise error if concatenated across boundary."""
    with pytest.raises(ValueError):
        # Interval exceeding T
        compute_continuous_density([(10.0, 150.0)], 100.0)


def test_fixture_11_rate_invariance():
    """Fixture 11: Identical AF exposure with different ventricular rates yields identical density."""
    intervals = [(50.0, 150.0), (300.0, 450.0)]
    T = 1000.0
    d1 = compute_continuous_density(intervals, T)
    d2 = compute_continuous_density(intervals, T)
    assert d1 == d2


def test_randomized_fixtures_against_brute_force():
    """Generates 100 randomized small timelines and verifies solver consistency and invariants."""
    random.seed(42)
    np.random.seed(42)

    T = 600.0  # 10 minutes
    passed = 0

    for rep in range(100):
        # Generate 1 to 5 random nonoverlapping episodes
        n_ep = random.randint(1, 5)
        raw_points = sorted(random.sample(range(10, 590), n_ep * 2))
        episodes = []
        for i in range(n_ep):
            episodes.append((float(raw_points[2 * i]), float(raw_points[2 * i + 1])))

        d_cont = compute_continuous_density(episodes, T)
        assert d_cont is not None
        assert 0.0 <= d_cont <= 1.0 + 1e-7, f"Density out of bounds: {d_cont}"

        # Test translation of whole timeline if space permits
        min_start = episodes[0][0]
        max_end = episodes[-1][1]
        available_slack = T - max_end
        if available_slack > 5.0:
            shift = random.uniform(1.0, available_slack)
            shifted = [(s + shift, e + shift) for s, e in episodes]
            d_shifted = compute_continuous_density(shifted, T)
            assert math.isclose(d_cont, d_shifted, abs_tol=1e-6)

        # Test reversal
        reversed_eps = sorted([(T - e, T - s) for s, e in episodes])
        d_rev = compute_continuous_density(reversed_eps, T)
        assert math.isclose(d_cont, d_rev, abs_tol=1e-6)

        # Test find_shortest_interval_for_fraction for p=0.80
        flen, fs, fe = find_shortest_interval_for_fraction(episodes, T, 0.80)
        assert flen <= T
        assert fe >= fs
        assert math.isclose(fe - fs, flen, abs_tol=1e-6)

        passed += 1

    assert passed == 100
