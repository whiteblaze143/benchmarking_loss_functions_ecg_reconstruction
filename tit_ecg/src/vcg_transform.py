"""Fixed ECG to 3D VCG (Vectorcardiogram) Dipole Transformation.

Transforms 8 independent surface ECG leads (I, II, V1..V6) into the 3D cardiac electrical dipole
trajectory v(t) = [V_x(t), V_y(t), V_z(t)]^T via Kors quasi-orthogonal regression.

Reference: Kors JA, et al. Reconstruction of the Frank vectorcardiogram from standard
electrocardiographic leads: Diagnostic utility. J Electrocardiol 1990; 23:297-303.
"""
from __future__ import annotations

import numpy as np

# Kors regression matrix mapping [I, II, V1, V2, V3, V4, V5, V6] -> [Vx, Vy, Vz]
# Shape [3, 8]
KORS_REGRESSION_MATRIX = np.array([
    #  I        II        V1       V2       V3       V4       V5       V6
    [ 0.38,   -0.07,    -0.13,    0.05,   -0.01,    0.14,    0.06,    0.54 ],  # Vx (right-to-left)
    [-0.07,    0.93,     0.06,   -0.02,   -0.05,    0.06,   -0.17,    0.13 ],  # Vy (superior-to-inferior)
    [-0.11,   -0.23,    -0.43,   -0.06,   -0.14,   -0.20,   -0.11,    0.31 ],  # Vz (posterior-to-anterior)
], dtype=float)


def ecg_to_vcg_kors(ecg: np.ndarray) -> np.ndarray:
    """Projects 12-lead ECG to 3D Frank VCG coordinates using the Kors regression matrix.

    Parameters
    ----------
    ecg : np.ndarray of shape [n_samples, 12] or [12, n_samples]
        Standard 12-lead ECG recording with leads ordered:
        [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6].

    Returns
    -------
    vcg : np.ndarray of shape [n_samples, 3]
        Continuous 3D dipole trajectory [Vx, Vy, Vz].
    """
    ecg = np.asarray(ecg, dtype=float)
    transposed = False
    if ecg.shape[0] == 12 and ecg.shape[1] != 12:
        ecg = ecg.T
        transposed = True

    if ecg.shape[1] < 8:
        raise ValueError(f"Expected at least 8 independent leads, got shape {ecg.shape}")

    # Standard lead index mapping:
    # 0: I, 1: II, 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
    lead_indices = [0, 1, 6, 7, 8, 9, 10, 11]
    ecg_8 = ecg[:, lead_indices]  # [n_samples, 8]

    # Matrix multiplication: [n_samples, 8] @ [8, 3] -> [n_samples, 3]
    vcg = ecg_8 @ KORS_REGRESSION_MATRIX.T
    return vcg
