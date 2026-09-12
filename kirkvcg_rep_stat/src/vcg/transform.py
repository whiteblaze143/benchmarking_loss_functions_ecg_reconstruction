"""Vectorcardiogram (VCG) transformation from 12-lead ECG.

Maps 12-lead ECG signals to 3D orthogonal cardiac dipole trajectories
s(t) = [V_x(t), V_y(t), V_z(t)]^T in R^3.

Supports:
1. Kors quasi-orthogonal regression matrix (Kors et al., 1990)
2. Classic Dower affine transformation matrix (Dower et al., 1980)
"""
from __future__ import annotations

import numpy as np

# Standard 12-lead order: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# Indices in standard 12-lead:
# V1=6, V2=7, V3=8, V4=9, V5=10, V6=11, I=0, II=1
DEFAULT_8LEAD_INDICES = [6, 7, 8, 9, 10, 11, 0, 1]

# Kors matrix for (V1, V2, V3, V4, V5, V6, I, II) -> (X, Y, Z)
KORS_MATRIX = np.array([
    [-0.130,  0.050, -0.010,  0.140,  0.260,  0.110,  0.380, -0.070],  # X
    [ 0.060, -0.020, -0.050,  0.060,  0.170,  0.130, -0.070,  0.930],  # Y
    [-0.430, -0.060, -0.040, -0.050, -0.080, -0.090,  0.110, -0.230],  # Z
], dtype=np.float64)

# Dower matrix for (V1, V2, V3, V4, V5, V6, I, II) -> (X, Y, Z)
DOWER_MATRIX = np.array([
    [-0.172, -0.074,  0.122,  0.231,  0.239,  0.194,  0.156, -0.009],  # X
    [ 0.057, -0.019, -0.106, -0.022,  0.041,  0.048, -0.227,  0.887],  # Y
    [-0.229, -0.310, -0.246, -0.063,  0.055,  0.108,  0.022,  0.102],  # Z
], dtype=np.float64)


def ecg_to_vcg_kors(
    ecg: np.ndarray,
    lead_indices: list[int] | None = None,
) -> np.ndarray:
    """Transforms 12-lead ECG to 3D VCG using Kors quasi-orthogonal matrix.

    Parameters
    ----------
    ecg : np.ndarray
        ECG array of shape [n_samples, 12] or [12, n_samples] or [n_samples, 8].
    lead_indices : list of int, optional
        Lead indices for [V1, V2, V3, V4, V5, V6, I, II]. Defaults to standard 12-lead indices.

    Returns
    -------
    vcg : np.ndarray
        3D VCG trajectory array of shape [n_samples, 3] representing (Vx, Vy, Vz).
    """
    return ecg_to_vcg(ecg, transform_matrix=KORS_MATRIX, lead_indices=lead_indices)


def ecg_to_vcg_dower(
    ecg: np.ndarray,
    lead_indices: list[int] | None = None,
) -> np.ndarray:
    """Transforms 12-lead ECG to 3D VCG using classic Dower transform.

    Parameters
    ----------
    ecg : np.ndarray
        ECG array of shape [n_samples, 12] or [12, n_samples] or [n_samples, 8].
    lead_indices : list of int, optional
        Lead indices for [V1, V2, V3, V4, V5, V6, I, II].

    Returns
    -------
    vcg : np.ndarray
        3D VCG trajectory array of shape [n_samples, 3] representing (Vx, Vy, Vz).
    """
    return ecg_to_vcg(ecg, transform_matrix=DOWER_MATRIX, lead_indices=lead_indices)


def ecg_to_vcg(
    ecg: np.ndarray,
    transform_matrix: np.ndarray = KORS_MATRIX,
    lead_indices: list[int] | None = None,
) -> np.ndarray:
    """Generic transformation from multi-lead ECG to 3D orthogonal VCG dipole space.

    Parameters
    ----------
    ecg : np.ndarray
        ECG array of shape [n_samples, n_leads] or [n_leads, n_samples].
    transform_matrix : np.ndarray
        Projection matrix of shape [3, 8].
    lead_indices : list of int, optional
        Indices for 8 independent leads.

    Returns
    -------
    vcg : np.ndarray
        Trajectory points s(t) in R^3 of shape [n_samples, 3].
    """
    arr = np.asarray(ecg, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ECG array, got shape {arr.shape}")

    # Ensure orientation is [n_samples, n_leads]
    if arr.shape[0] in (8, 12) and arr.shape[1] > 12:
        arr = arr.T

    n_samples, n_leads = arr.shape
    if n_leads == 8:
        # Already 8 leads: assume order [V1, V2, V3, V4, V5, V6, I, II]
        leads_8 = arr
    elif n_leads >= 12:
        indices = lead_indices if lead_indices is not None else DEFAULT_8LEAD_INDICES
        leads_8 = arr[:, indices]
    elif n_leads == 3:
        # Already 3-channel VCG
        return arr
    else:
        raise ValueError(f"Unsupported number of leads: {n_leads}. Expected 3, 8, or 12.")

    # Matrix multiplication: [n_samples, 8] @ [8, 3] -> [n_samples, 3]
    vcg = leads_8 @ transform_matrix.T
    return vcg
