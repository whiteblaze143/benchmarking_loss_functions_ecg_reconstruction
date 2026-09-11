"""Physical-Time VCG Dynamics and Intrinsic Descriptor Module.

Computes:
- First and second time derivatives dot{v}(t), ddot{v}(t) via Savitzky-Golay filtering
  strictly in original physical time (e.g. 500 Hz, delta_t = 0.002 s).
- Intrinsic descriptor xi(t) = [s(t), rho(t), kappa(t)]:
    s(t) = log(||dot{v}(t)|| + eps)
    rho(t) = (v(t)^T dot{v}(t)) / (||v(t)|| + eps)
    kappa(t) = log(1 + ||dot{v}(t) x ddot{v}(t)|| / (||dot{v}(t)|| + eps)^3)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
import torch


def compute_vcg_derivatives(
    v: np.ndarray,
    fs: float = 500.0,
    window_length_ms: float = 21.0,
    polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes first and second derivatives of 3D VCG in physical time.

    Args:
        v: [3, T] or [B, 3, T] VCG trajectory in physical mV.
        fs: Sampling frequency in Hz (default 500.0).
        window_length_ms: Window length in milliseconds (default 21 ms).
        polyorder: Polynomial order for Savitzky-Golay filter (default 2).

    Returns:
        v_dot: [..., 3, T] velocity in mV/s.
        v_ddot: [..., 3, T] acceleration in mV/s^2.
    """
    dt = 1.0 / fs
    # Convert window length from ms to odd number of samples >= polyorder + 2
    n_samples = int(round((window_length_ms / 1000.0) * fs))
    if n_samples % 2 == 0:
        n_samples += 1
    min_samples = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3
    window_samples = max(min_samples, n_samples)

    if v.shape[-1] < window_samples:
        raise ValueError(
            f"Trajectory length {v.shape[-1]} is shorter than filter window {window_samples}"
        )

    # Apply Savitzky-Golay filter along the time dimension (-1)
    v_dot = savgol_filter(v, window_length=window_samples, polyorder=polyorder, deriv=1, delta=dt, axis=-1)
    v_ddot = savgol_filter(v, window_length=window_samples, polyorder=polyorder, deriv=2, delta=dt, axis=-1)

    return v_dot, v_ddot


def compute_intrinsic_descriptor(
    v: np.ndarray,
    v_dot: np.ndarray,
    v_ddot: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Computes intrinsic geometric descriptor xi = [s, rho, kappa].

    Args:
        v: [..., 3, T] VCG position.
        v_dot: [..., 3, T] velocity dot{v}.
        v_ddot: [..., 3, T] acceleration ddot{v}.
        eps: Small positive constant for numerical stability.

    Returns:
        xi: [..., 3, T] descriptor array containing [s, rho, kappa].
    """
    # Speed: s(t) = log(||dot{v}(t)|| + eps)
    speed_raw = np.linalg.norm(v_dot, axis=-2)  # [..., T]
    s = np.log(speed_raw + eps)

    # Radial velocity: rho(t) = (v(t)^T dot{v}(t)) / (||v(t)|| + eps)
    pos_norm = np.linalg.norm(v, axis=-2)  # [..., T]
    radial_inner = np.sum(v * v_dot, axis=-2)  # [..., T]
    rho = radial_inner / (pos_norm + eps)

    # Cross product: dot{v} x ddot{v}
    # np.cross operates along axis=-2 for 3D vectors
    cross = np.cross(v_dot, v_ddot, axis=-2)  # [..., 3, T]
    cross_norm = np.linalg.norm(cross, axis=-2)  # [..., T]

    # Curvature: kappa(t) = log(1 + ||dot{v} x ddot{v}|| / (||dot{v}|| + eps)^3)
    denom = (speed_raw + eps) ** 3
    raw_curv = cross_norm / denom
    kappa = np.log1p(raw_curv)

    # Stack along coordinate axis (-2) -> [..., 3, T]
    xi = np.stack([s, rho, kappa], axis=-2)
    return xi


def extract_vcg_microstate_features(
    v: np.ndarray,
    fs: float = 500.0,
    window_length_ms: float = 21.0,
    polyorder: int = 2,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience pipeline function extracting derivatives and xi descriptor."""
    v_dot, v_ddot = compute_vcg_derivatives(
        v, fs=fs, window_length_ms=window_length_ms, polyorder=polyorder
    )
    xi = compute_intrinsic_descriptor(v, v_dot, v_ddot, eps=eps)
    return v_dot, v_ddot, xi
