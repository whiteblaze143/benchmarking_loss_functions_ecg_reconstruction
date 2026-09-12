"""Sample-level electrophysiological feature extraction for VCG repSpat.

Constructs the p-dimensional attribute vector X(t) at each time sample t_i
along the VCG trajectory, combining:
1. Lead voltages (12 leads or 8 independent leads)
2. Local 1st and 2nd temporal differences (depolarization/repolarization velocities)
3. 3D VCG differential geometry: speed, curvature, torsion, normal acceleration
4. Localized RMS energy across sliding windows
5. Continuous standardized z-scores and binary thresholded markers (for Jaccard distance).
"""
from __future__ import annotations

import numpy as np
from ..vcg.transform import ecg_to_vcg_kors, DEFAULT_8LEAD_INDICES
from ..vcg.kinematics import compute_vcg_kinematics

LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAMES_8 = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "II"]


def extract_sample_features(
    ecg: np.ndarray,
    vcg: np.ndarray | None = None,
    fs: float = 500.0,
    include_derivatives: bool = True,
    include_kinematics: bool = True,
    include_energy: bool = True,
    window_samples: int = 15,
) -> dict[str, np.ndarray | list[str]]:
    """Extracts continuous and binary multivariate electrophysiological features per sample.

    Parameters
    ----------
    ecg : np.ndarray
        Multi-lead ECG array of shape [n_samples, n_leads] or [n_leads, n_samples].
    vcg : np.ndarray, optional
        Precomputed 3D VCG array of shape [n_samples, 3]. If None, computed via Kors transform.
    fs : float
        Sampling frequency in Hz (default: 500.0).
    include_derivatives : bool
        Whether to compute 1st and 2nd temporal differences of voltages.
    include_kinematics : bool
        Whether to include 3D VCG curvature, torsion, and velocity.
    include_energy : bool
        Whether to include sliding window localized RMS energy.
    window_samples : int
        Window size for localized RMS energy (default: 15 samples = 30 ms at 500 Hz).

    Returns
    -------
    feature_dict : dict
        {
            'continuous': np.ndarray [n_samples, p_cont],
            'binary': np.ndarray [n_samples, p_bin],
            'vcg': np.ndarray [n_samples, 3],
            'feature_names': list of str,
            'feature_names_bin': list of str,
        }
    """
    arr = np.asarray(ecg, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ECG array, got shape {arr.shape}")

    if arr.shape[0] in (8, 12) and arr.shape[1] > 12:
        arr = arr.T

    n_samples, n_leads = arr.shape
    dt = 1.0 / float(fs)

    if vcg is None:
        vcg = ecg_to_vcg_kors(arr)
    else:
        vcg = np.asarray(vcg, dtype=np.float64)

    feature_cols: list[np.ndarray] = []
    feature_names: list[str] = []

    # 1. Lead voltages
    if n_leads == 12:
        lead_labels = LEAD_NAMES_12
    elif n_leads == 8:
        lead_labels = LEAD_NAMES_8
    else:
        lead_labels = [f"lead_{i}" for i in range(n_leads)]

    for i in range(n_leads):
        feature_cols.append(arr[:, i])
        feature_names.append(f"volt_{lead_labels[i]}")

    # 2. Local 1st and 2nd temporal derivatives (dV/dt, d^2V/dt^2)
    if include_derivatives:
        d1 = np.gradient(arr, dt, axis=0)  # [n_samples, n_leads]
        d2 = np.gradient(d1, dt, axis=0)
        for i in range(n_leads):
            feature_cols.append(d1[:, i])
            feature_names.append(f"d1_{lead_labels[i]}")
            feature_cols.append(d2[:, i])
            feature_names.append(f"d2_{lead_labels[i]}")

    # 3. 3D VCG kinematics and differential geometry
    if include_kinematics:
        kin = compute_vcg_kinematics(vcg, fs=fs)
        feature_cols.append(kin["speed"])
        feature_names.append("vcg_speed")
        feature_cols.append(kin["accel_magnitude"])
        feature_names.append("vcg_accel_mag")
        feature_cols.append(kin["curvature"])
        feature_names.append("vcg_curvature")
        feature_cols.append(kin["torsion"])
        feature_names.append("vcg_torsion")
        feature_cols.append(kin["normal_accel"])
        feature_names.append("vcg_normal_accel")

    # 4. Localized RMS energy across sliding window
    if include_energy and window_samples > 1:
        pad_width = window_samples // 2
        # Compute RMS energy of VCG vector: sqrt(mean(Vx^2 + Vy^2 + Vz^2))
        vcg_norm_sq = np.sum(vcg**2, axis=1)
        padded = np.pad(vcg_norm_sq, pad_width, mode="edge")
        window = np.ones(window_samples) / window_samples
        local_rms = np.sqrt(np.convolve(padded, window, mode="valid")[:n_samples])
        feature_cols.append(local_rms)
        feature_names.append("vcg_local_rms")

    # Stack continuous feature matrix [n_samples, p]
    X_continuous = np.column_stack(feature_cols)

    # 5. Nonparametric median binarization per repSpat protocol
    medians = np.median(X_continuous, axis=0)
    X_binary = (X_continuous > medians).astype(np.float64)
    feature_names_bin = [f"{name}_above_median" for name in feature_names]

    return {
        "continuous": X_continuous,
        "binary": X_binary,
        "vcg": vcg,
        "feature_names": feature_names,
        "feature_names_bin": feature_names_bin,
    }


def standardize_features(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardizes continuous features (z-score normalization).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix [n, p].
    eps : float
        Small constant to prevent zero standard deviation division.

    Returns
    -------
    X_z : np.ndarray
        Standardized feature matrix [n, p] with mean 0 and variance 1.
    """
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    return (X - mean) / (std + eps)


def binarize_features(X: np.ndarray, threshold: str = "median") -> np.ndarray:
    """Binarizes feature matrix into {0, 1} for Jaccard distance calculation.

    Parameters
    ----------
    X : np.ndarray
        Continuous feature matrix [n, p].
    threshold : str
        'median' (default) or 'zero'.

    Returns
    -------
    X_bin : np.ndarray
        Binary feature matrix of shape [n, p].
    """
    if threshold == "median":
        thresh_vals = np.median(X, axis=0, keepdims=True)
        return (X > thresh_vals).astype(np.float64)
    elif threshold == "zero":
        return (X > 0.0).astype(np.float64)
    else:
        raise ValueError(f"Unknown thresholding mode '{threshold}'")
