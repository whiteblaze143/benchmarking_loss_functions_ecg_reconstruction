"""Tests for VCG Dynamics and Intrinsic Descriptors (PRD Section 66)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rep_stat_ecg.src.vcg.dynamics import (
    compute_vcg_derivatives,
    compute_intrinsic_descriptor,
    extract_vcg_microstate_features,
)


def test_derivative_straight_line():
    """Constant velocity straight line should have zero acceleration and zero curvature."""
    fs = 500.0
    t = np.linspace(0, 2.0, int(2.0 * fs), endpoint=False)
    # v(t) = c_0 + velocity * t
    velocity = np.array([[2.0], [-1.0], [0.5]])  # [3, 1]
    v = velocity * t[None, :]  # [3, T]

    v_dot, v_ddot, xi = extract_vcg_microstate_features(v, fs=fs, window_length_ms=25.0)

    # Interior samples (excluding boundary filter artifacts)
    interior = slice(50, -50)
    expected_v_dot = np.repeat(velocity, len(t), axis=1)

    assert np.allclose(v_dot[:, interior], expected_v_dot[:, interior], atol=1e-3)
    assert np.allclose(v_ddot[:, interior], 0.0, atol=1e-3)

    # Curvature should be approximately log(1 + 0) = 0
    kappa = xi[2, interior]
    assert np.allclose(kappa, 0.0, atol=1e-3)


def test_circle_curvature():
    """For a unit circle traversed at constant speed omega, curvature is 1/r."""
    fs = 500.0
    duration = 4.0
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    r = 2.0
    omega = 2.0 * np.pi * 1.0  # 1 Hz

    # v(t) = [r cos(omega t), r sin(omega t), 0]
    v = np.stack([
        r * np.cos(omega * t),
        r * np.sin(omega * t),
        np.zeros_like(t),
    ], axis=0)  # [3, T]

    v_dot, v_ddot, xi = extract_vcg_microstate_features(v, fs=fs, window_length_ms=25.0)

    # Analytic curvature for circle of radius r is 1/r
    expected_curv_raw = 1.0 / r
    expected_kappa = np.log1p(expected_curv_raw)

    interior = slice(100, -100)
    measured_kappa = xi[2, interior]

    err = np.max(np.abs(measured_kappa - expected_kappa))
    assert err < 0.05, f"Measured kappa {np.mean(measured_kappa):.4f} deviates from {expected_kappa:.4f}"


def test_rigid_rotation_preserves_intrinsic_descriptor():
    """Rigid 3D rotation SO(3) must preserve the intrinsic descriptor xi."""
    fs = 500.0
    t = np.linspace(0, 2.0, int(2.0 * fs), endpoint=False)
    # 3D helix trajectory
    v = np.stack([
        np.cos(2.0 * np.pi * t),
        np.sin(2.0 * np.pi * t),
        0.5 * t,
    ], axis=0)  # [3, T]

    _, _, xi_original = extract_vcg_microstate_features(v, fs=fs)

    # Apply random 3D rotation
    rot = Rotation.from_euler("zyx", [35, 45, 60], degrees=True).as_matrix()
    v_rotated = rot @ v

    _, _, xi_rotated = extract_vcg_microstate_features(v_rotated, fs=fs)

    interior = slice(50, -50)
    diff = np.max(np.abs(xi_original[:, interior] - xi_rotated[:, interior]))
    assert diff < 1e-4, f"Intrinsic descriptor changed under rotation: max diff {diff}"


def test_derivatives_computed_before_phase_resampling():
    """Verify that physical sampling rate preserves true physical units."""
    # Slower signal (1 Hz) vs faster signal (2 Hz)
    fs = 500.0
    t = np.linspace(0, 1.0, 500, endpoint=False)
    v1 = np.stack([np.sin(2 * np.pi * 1.0 * t), np.zeros_like(t), np.zeros_like(t)])
    v2 = np.stack([np.sin(2 * np.pi * 2.0 * t), np.zeros_like(t), np.zeros_like(t)])

    v1_dot, _, xi1 = extract_vcg_microstate_features(v1, fs=fs)
    v2_dot, _, xi2 = extract_vcg_microstate_features(v2, fs=fs)

    # Max speed of v2 should be exactly twice that of v1
    max_s1 = np.max(np.exp(xi1[0, 50:-50]))
    max_s2 = np.max(np.exp(xi2[0, 50:-50]))
    ratio = max_s2 / max_s1
    assert np.isclose(ratio, 2.0, atol=0.05), f"Expected 2.0x velocity ratio, got {ratio}"
