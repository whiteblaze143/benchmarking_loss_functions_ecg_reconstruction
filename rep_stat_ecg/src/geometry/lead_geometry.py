"""Lead Geometry Registry for GRAIL-ECG.

Contains canonical angles (theta, phi) in radians for standard 12 leads and
the primary 8 independent leads (I, II, V1-V6). Values imported from NEF-NET PTB-XL
configuration and lead-field volume conduction prior.
"""

from __future__ import annotations

import numpy as np
import torch

# Standard 12-lead names in canonical order
CANONICAL_12_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
)

# 8 independent physical leads
INDEPENDENT_8_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")

# Independent lead indices relative to canonical 12-lead tensor
INDEPENDENT_8_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)

# Angle definitions (theta: colatitude/polar angle, phi: azimuthal angle in radians)
# Theta in [0, pi], Phi in [-pi, pi]
STANDARD_12_ANGLES = {
    "I":   (np.pi / 2,        np.pi / 2),
    "II":  (np.pi * 5 / 6,    np.pi / 2),
    "III": (np.pi * 5 / 6,   -np.pi / 2),
    "aVR": (np.pi / 3,       -np.pi / 2),
    "aVL": (np.pi / 3,        np.pi / 2),
    "aVF": (np.pi,            np.pi / 2),
    "V1":  (np.pi / 2,       -np.pi / 18),
    "V2":  (np.pi / 2,        np.pi / 18),
    "V3":  (np.pi * 19 / 36,  np.pi / 12),
    "V4":  (np.pi * 11 / 20,  np.pi / 6),
    "V5":  (np.pi * 16 / 30,  np.pi / 3),
    "V6":  (np.pi * 16 / 30,  np.pi / 2),
}

INDEPENDENT_8_ANGLES = {
    lead: STANDARD_12_ANGLES[lead] for lead in INDEPENDENT_8_LEADS
}


def get_angles_tensor(leads: tuple[str, ...] = INDEPENDENT_8_LEADS, device=None, dtype=torch.float32) -> torch.Tensor:
    """Returns a [len(leads), 2] tensor of (theta, phi) angles."""
    arr = np.array([STANDARD_12_ANGLES[l] for l in leads], dtype=np.float32)
    return torch.tensor(arr, device=device, dtype=dtype)


def independent_from_standard(ecg: torch.Tensor) -> torch.Tensor:
    """Extracts 8 independent leads [I, II, V1-V6] from [B, 12, T] or [12, T]."""
    if ecg.shape[-2] != 12:
        raise ValueError(f"Expected 12 leads at dim -2, got shape {tuple(ecg.shape)}")
    return ecg[..., INDEPENDENT_8_INDICES, :]


def standard_from_independent(ecg_indep: torch.Tensor) -> torch.Tensor:
    """Algebraically derives the 4 dependent limb leads from I and II."""
    if ecg_indep.shape[-2] != 8:
        raise ValueError(f"Expected 8 independent leads at dim -2, got shape {tuple(ecg_indep.shape)}")
    lead_i = ecg_indep[..., 0, :]
    lead_ii = ecg_indep[..., 1, :]
    lead_iii = lead_ii - lead_i
    lead_avr = -(lead_i + lead_ii) / 2.0
    lead_avl = lead_i - lead_ii / 2.0
    lead_avf = lead_ii - lead_i / 2.0
    precordial = [ecg_indep[..., idx, :] for idx in range(2, 8)]
    return torch.stack(
        (lead_i, lead_ii, lead_iii, lead_avr, lead_avl, lead_avf, *precordial),
        dim=-2,
    )
