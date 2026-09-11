"""Mandatory Unit Tests for GRAIL-ECG Contracts and Geometry (PRD §35)."""

import pytest
import torch
import numpy as np

from grail_ecg.src.geometry.lead_geometry import (
    CANONICAL_12_LEADS,
    INDEPENDENT_8_LEADS,
    INDEPENDENT_8_INDICES,
    STANDARD_12_ANGLES,
    get_angles_tensor,
    independent_from_standard,
    standard_from_independent,
)
from grail_ecg.src.geometry.theta_encoder import ThetaEncoder


def test_input_requires_12_canonical_leads():
    """Asserts that standard functions require exact 12-lead dimension."""
    valid_ecg = torch.randn(2, 12, 5000)
    out = independent_from_standard(valid_ecg)
    assert out.shape == (2, 8, 5000)

    invalid_ecg = torch.randn(2, 11, 5000)
    with pytest.raises(ValueError, match="Expected 12 leads"):
        independent_from_standard(invalid_ecg)


def test_independent_8_extraction():
    """Verifies indices map precisely to [I, II, V1, V2, V3, V4, V5, V6]."""
    assert INDEPENDENT_8_INDICES == (0, 1, 6, 7, 8, 9, 10, 11)
    for idx, expected_lead in zip(INDEPENDENT_8_INDICES, INDEPENDENT_8_LEADS):
        assert CANONICAL_12_LEADS[idx] == expected_lead


def test_einthoven_relations():
    """Verifies that derived limb leads satisfy Kirchhoff & Goldberger laws machine-precisely."""
    # Synthetic 8 independent leads
    I = torch.randn(4, 5000)
    II = torch.randn(4, 5000)
    precordial = [torch.randn(4, 5000) for _ in range(6)]
    indep = torch.stack([I, II, *precordial], dim=1)  # [4, 8, 5000]

    derived_12 = standard_from_independent(indep)  # [4, 12, 5000]
    lead_I = derived_12[:, 0, :]
    lead_II = derived_12[:, 1, :]
    lead_III = derived_12[:, 2, :]
    lead_aVR = derived_12[:, 3, :]
    lead_aVL = derived_12[:, 4, :]
    lead_aVF = derived_12[:, 5, :]

    # Einthoven: I - II + III == 0  <=> III == II - I
    assert torch.allclose(lead_III, lead_II - lead_I, atol=1e-7)

    # Goldberger definitions
    assert torch.allclose(lead_aVR, -(lead_I + lead_II) / 2.0, atol=1e-7)
    assert torch.allclose(lead_aVL, lead_I - lead_II / 2.0, atol=1e-7)
    assert torch.allclose(lead_aVF, lead_II - lead_I / 2.0, atol=1e-7)


def test_theta_encoder_shape():
    """Verifies ThetaEncoder maps [B, L, 2] to [B, L, 12]."""
    encoder = ThetaEncoder(omega=1.0)
    angles = torch.tensor([[[np.pi / 2, np.pi / 2], [np.pi * 5 / 6, np.pi / 2]]], dtype=torch.float32)
    harmonics = encoder(angles)
    assert harmonics.shape == (1, 2, 12)

    # Assert non-trivial trigonometric outputs
    assert not torch.isnan(harmonics).any()
    assert not torch.isinf(harmonics).any()


def test_theta_registry_order():
    """Verifies all standard angles are present and within valid spherical bounds."""
    for lead in CANONICAL_12_LEADS:
        assert lead in STANDARD_12_ANGLES
        theta, phi = STANDARD_12_ANGLES[lead]
        assert 0.0 <= theta <= np.pi + 1e-5
        assert -np.pi - 1e-5 <= phi <= np.pi + 1e-5
