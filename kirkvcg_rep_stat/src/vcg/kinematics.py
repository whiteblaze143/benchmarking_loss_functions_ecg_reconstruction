r"""Differential geometry and kinematics of 3D VCG cardiac dipole trajectories.

Computes instantaneous kinematic and Frenet-Serret geometric descriptors
along the cardiac dipole trajectory s(t) in R^3:
- Dipole velocity vector \dot{s}(t) and velocity magnitude (speed) ||\dot{s}(t)||
- Dipole acceleration vector \ddot{s}(t) and jerk \dddot{s}(t)
- Instantaneous trajectory curvature \kappa(t)
- Instantaneous trajectory torsion \tau(t)
- Tangential and normal acceleration components
"""
from __future__ import annotations

import numpy as np


def compute_vcg_kinematics(
    vcg: np.ndarray,
    fs: float = 500.0,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    r"""Computes differential geometric properties of the 3D VCG loop.

    Parameters
    ----------
    vcg : np.ndarray
        Array of shape [n_samples, 3] containing (Vx, Vy, Vz) dipole trajectory.
    fs : float
        Sampling frequency in Hz (default: 500.0).
    eps : float
        Small constant for numerical stability against zero-division.

    Returns
    -------
    kinematics : dict
        {
            'velocity': np.ndarray [n_samples, 3],
            'speed': np.ndarray [n_samples], # ||\dot{s}||
            'acceleration': np.ndarray [n_samples, 3],
            'accel_magnitude': np.ndarray [n_samples],
            'jerk': np.ndarray [n_samples, 3],
            'curvature': np.ndarray [n_samples], # \kappa
            'torsion': np.ndarray [n_samples], # \tau
            'tangential_accel': np.ndarray [n_samples],
            'normal_accel': np.ndarray [n_samples],
        }
    """
    s = np.asarray(vcg, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError(f"Expected VCG trajectory of shape [n_samples, 3], got {s.shape}")

    n_samples = len(s)
    if n_samples < 4:
        raise ValueError("VCG trajectory must have at least 4 temporal samples for kinematics.")

    dt = 1.0 / float(fs)

    # 1. Velocity vector: \dot{s}(t) [n_samples, 3]
    velocity = np.gradient(s, dt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)  # [n_samples]

    # 2. Acceleration vector: \ddot{s}(t) [n_samples, 3]
    acceleration = np.gradient(velocity, dt, axis=0)
    accel_mag = np.linalg.norm(acceleration, axis=1)

    # 3. Jerk vector: \dddot{s}(t) [n_samples, 3]
    jerk = np.gradient(acceleration, dt, axis=0)

    # 4. Cross product: \dot{s}(t) x \ddot{s}(t) [n_samples, 3]
    cross_v_a = np.cross(velocity, acceleration)
    cross_mag = np.linalg.norm(cross_v_a, axis=1)  # [n_samples]

    # 5. 3D Frenet-Serret Curvature: \kappa(t) = ||\dot{s} x \ddot{s}|| / (||\dot{s}||^3 + eps)
    curvature = cross_mag / (speed**3 + eps)

    # 6. 3D Frenet-Serret Torsion: \tau(t) = ((\dot{s} x \ddot{s}) . \dddot{s}) / (||\dot{s} x \ddot{s}||^2 + eps)
    dot_cross_jerk = np.sum(cross_v_a * jerk, axis=1)
    torsion = dot_cross_jerk / (cross_mag**2 + eps)

    # 7. Tangential & Normal acceleration components
    # a_t = (\dot{s} . \ddot{s}) / (||\dot{s}|| + eps)
    tangential_accel = np.sum(velocity * acceleration, axis=1) / (speed + eps)
    # a_n = ||\dot{s} x \ddot{s}|| / (||\dot{s}|| + eps)
    normal_accel = cross_mag / (speed + eps)

    # Sanitize any infs or nans from extreme baseline flatlines
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    torsion = np.nan_to_num(torsion, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "velocity": velocity,
        "speed": speed,
        "acceleration": acceleration,
        "accel_magnitude": accel_mag,
        "jerk": jerk,
        "curvature": curvature,
        "torsion": torsion,
        "tangential_accel": tangential_accel,
        "normal_accel": normal_accel,
    }


def compute_vcg_loop_planarity(vcg: np.ndarray) -> float:
    """Computes the planarity index of the 3D VCG trajectory.

    In normal ventricular activation, the QRS loop lies primarily in a 2D plane
    (Ghosal et al., 2024; Cholaquidis et al., 2023). Planarity measures the fraction
    of spatial variance explained by the first two principal axes:
    Planarity = (lambda_1 + lambda_2) / (lambda_1 + lambda_2 + lambda_3 + eps).

    Parameters
    ----------
    vcg : np.ndarray
        3D VCG coordinates of shape [n_samples, 3].

    Returns
    -------
    planarity : float
        Planarity ratio bounded in [0.0, 1.0], where 1.0 indicates perfect planar geometry.
    """
    s = np.asarray(vcg, dtype=np.float64)
    if len(s) < 3:
        return 1.0

    centered = s - np.mean(s, axis=0)
    cov = (centered.T @ centered) / float(max(1, len(s) - 1))
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total_var = float(np.sum(eigenvalues)) + 1e-12
    planarity = float((eigenvalues[0] + eigenvalues[1]) / total_var)
    return float(np.clip(planarity, 0.0, 1.0))


def compute_spatial_qrst_angle(
    qrs_vector: np.ndarray,
    t_vector: np.ndarray,
    degrees: bool = True,
    eps: float = 1e-8,
) -> float:
    """Computes the 3D spatial angle between dominant QRS and T-wave vectors.

    A key clinical discriminator for myocardial electrical heterogeneity and risk
    stratification (Hughes et al., 2024; Stabenau et al., 2023).

    Parameters
    ----------
    qrs_vector : np.ndarray
        Dominant QRS vector in R^3.
    t_vector : np.ndarray
        Dominant T-wave vector in R^3.
    degrees : bool
        Whether to return angle in degrees (default: True, range [0, 180]).
    eps : float
        Numerical epsilon.

    Returns
    -------
    angle : float
        Spatial QRS-T angle.
    """
    v_qrs = np.asarray(qrs_vector, dtype=np.float64)
    v_t = np.asarray(t_vector, dtype=np.float64)

    norm_qrs = np.linalg.norm(v_qrs) + eps
    norm_t = np.linalg.norm(v_t) + eps
    cos_theta = float(np.dot(v_qrs, v_t) / (norm_qrs * norm_t))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    angle_rad = float(np.arccos(cos_theta))

    return float(np.degrees(angle_rad) if degrees else angle_rad)
