"""Unit tests for PMV core extraction, alignment, and mathematical properties."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from scipy import signal

from experiments.pmv_temporal_specificity.scripts.pmv_core import PMVExtractor, PMVSubsegmentResult


def create_synthetic_p_wave(duration_s=0.140, fs=200.0, amp=0.1, center_s=0.070, width_s=0.025):
    """Gaussian synthetic P-wave for testing mathematical invariants."""
    t = np.linspace(0, duration_s, int(round(duration_s * fs)), endpoint=False)
    p = amp * np.exp(-0.5 * ((t - center_s) / width_s)**2)
    return p


def test_identical_p_waves_zero_mad():
    """Identical P-waves must produce exactly 0 MAD, 0 MAD energy, and 0 PMV."""
    extractor = PMVExtractor(fs=200.0)
    base_p = create_synthetic_p_wave(fs=200.0, amp=0.15)
    ensemble = np.tile(base_p, (30, 1))

    Ei = extractor.compute_reference_energy(ensemble)
    D, _, V, Em, mad_mean, mad_max = extractor.calculate_pmv_from_ensemble(ensemble, fixed_reference_energy=Ei)

    assert np.isclose(D, 0.0, atol=1e-12)
    assert np.isclose(V, 0.0, atol=1e-12)
    assert np.isclose(mad_mean, 0.0, atol=1e-12)
    assert np.isclose(mad_max, 0.0, atol=1e-12)
    assert Ei > 0.0


def test_no_silent_fallback_missing_fixed_reference():
    """When fixed_reference_energy is omitted, function returns NaNs and marks invalid; NO silent fallback."""
    extractor = PMVExtractor(fs=200.0)
    base_p = create_synthetic_p_wave(fs=200.0, amp=0.15)
    ensemble = np.tile(base_p, (30, 1))

    D, Ei, V, _, _, _ = extractor.calculate_pmv_from_ensemble(ensemble, fixed_reference_energy=None)
    assert np.isnan(Ei)
    assert np.isnan(V)
    assert np.isclose(D, 0.0, atol=1e-12)


def test_fixed_reference_energy_scale_invariance():
    """PMV V = D / Ei must be invariant to uniform amplitude scaling."""
    extractor = PMVExtractor(fs=200.0)
    base_p = create_synthetic_p_wave(fs=200.0, amp=0.10)
    rng = np.random.RandomState(42)

    variations = rng.normal(0, 0.02, size=(40, len(base_p)))
    ensemble1 = base_p + variations
    Ei1 = extractor.compute_reference_energy(ensemble1)
    D1, _, V1, _, _, _ = extractor.calculate_pmv_from_ensemble(ensemble1, fixed_reference_energy=Ei1)

    scale_factor = 3.5
    ensemble2 = ensemble1 * scale_factor
    Ei2 = extractor.compute_reference_energy(ensemble2)
    D2, _, V2, _, _, _ = extractor.calculate_pmv_from_ensemble(ensemble2, fixed_reference_energy=Ei2)

    assert np.isclose(Ei2, (scale_factor**2) * Ei1, rtol=1e-5)
    assert np.isclose(D2, (scale_factor**2) * D1, rtol=1e-5)
    assert np.isclose(V1, V2, rtol=1e-5)


def test_fixed_denominator_prevents_local_template_explosion():
    """Fixed reference Ei prevents spurious PMV inflation when local template energy attenuates."""
    extractor = PMVExtractor(fs=200.0)
    base_p = create_synthetic_p_wave(fs=200.0, amp=0.15)
    rng = np.random.RandomState(42)

    # Reference ensemble: robust 0.15 mV P-waves
    ref_ensemble = base_p + rng.normal(0, 0.01, size=(40, len(base_p)))
    Ei = extractor.compute_reference_energy(ref_ensemble)

    # Subsegment ensemble: P-wave amplitude reduced by 10x due to phase/lead cancellation
    # but with SAME absolute MAD dispersion
    attenuated_ensemble = 0.1 * base_p + rng.normal(0, 0.01, size=(40, len(base_p)))

    # Compute with published fixed Ei vs local template Em
    D, _, V_fixed, Em_local, _, _ = extractor.calculate_pmv_from_ensemble(attenuated_ensemble, fixed_reference_energy=Ei)
    V_local = D / Em_local

    # Fixed PMV remains stable; local template PMV explodes by ~100x!
    assert V_fixed < 1.0
    assert V_local > 10.0 * V_fixed


def test_insufficient_beats_gating():
    """When valid beats < min_beats (default 20), subsegment returns invalid and NaNs."""
    extractor = PMVExtractor(fs=200.0, min_beats_per_subsegment=20, min_beat_snr_db=10.0)
    fs = 200.0
    duration_s = 120.0
    ecg = np.random.randn(int(duration_s * fs)) * 0.01

    qrs_indices = np.linspace(200, int(duration_s * fs) - 200, 12, dtype=int)
    res = extractor.compute_subsegment_pmv(ecg, qrs_indices, time_center_s=60.0)

    assert not res.is_valid
    assert "INSUFFICIENT_BEATS" in res.status
    assert np.isnan(res.mad_energy_D)
    assert np.isnan(res.pmv_V)


def test_pr_alignment_recovers_jitter():
    """PR-interval jitter artificially inflates unaligned PMV; alignment recovers near-zero PMV."""
    extractor = PMVExtractor(fs=200.0, enable_pr_alignment=True, max_pr_shift_ms=25.0)
    base_p = create_synthetic_p_wave(fs=200.0, amp=0.20, center_s=0.070)
    L = len(base_p)

    # Bimodal PR jitter (e.g., dual AV nodal pathways: fast vs slow) separated by >= 50 ms (10 samples)
    shifts = [-5]*12 + [0]*11 + [5]*12
    ensemble_jittered = np.zeros((35, L))
    for i, s in enumerate(shifts):
        if s == 0:
            ensemble_jittered[i] = base_p
        elif s < 0:
            ensemble_jittered[i] = np.pad(base_p[-s:], (0, -s), mode="edge")
        else:
            ensemble_jittered[i] = np.pad(base_p[:-s], (s, 0), mode="edge")

    Ei = extractor.compute_reference_energy(ensemble_jittered)
    D_unaligned, _, V_unaligned, _, _, _ = extractor.calculate_pmv_from_ensemble(ensemble_jittered, fixed_reference_energy=Ei)

    m_k = np.median(ensemble_jittered, axis=0)
    aligned_p, mean_shift, did_align = extractor.align_pr_jitter(ensemble_jittered, m_k)
    D_aligned, _, V_aligned, _, _, _ = extractor.calculate_pmv_from_ensemble(aligned_p, fixed_reference_energy=Ei)

    assert did_align
    assert mean_shift > 0
    assert D_aligned < D_unaligned * 0.35
    assert V_aligned < V_unaligned * 0.35


def test_spatial_pca_fixed_eigenvector():
    """Fixed spatial PCA correctly projects 2 leads using reference eigenvector."""
    extractor = PMVExtractor(fs=200.0, min_beats_per_subsegment=20, min_beat_snr_db=10.0)
    fs = 200.0
    n_samples = int(120.0 * fs)

    ecg2 = np.zeros((2, n_samples))
    qrs_indices = np.arange(400, n_samples - 400, 200)

    base_p = create_synthetic_p_wave(fs=200.0, amp=0.20)
    L = len(base_p)
    start_offset = extractor.pre_qrs_start_samples

    for q in qrs_indices:
        s = q + start_offset
        ecg2[0, s:s+L] += base_p
        ecg2[1, s:s+L] += 0.5 * base_p
        # Add synthetic QRS spike at q
        ecg2[0, q-2:q+3] += np.array([0.2, 0.8, 1.2, 0.8, 0.2])
        ecg2[1, q-2:q+3] += 0.5 * np.array([0.2, 0.8, 1.2, 0.8, 0.2])

    # Reference eigenvector [0.8944, 0.4472]
    u_ref = np.array([2.0, 1.0]) / np.sqrt(5.0)
    fixed_Ei = 0.04

    res = extractor.compute_subsegment_pmv(
        ecg2, qrs_indices, time_center_s=60.0, fixed_eigenvector=u_ref, fixed_reference_energy=fixed_Ei
    )

    assert res.is_valid
    assert res.num_beats >= 20
    assert not np.isnan(res.pmv_V)
