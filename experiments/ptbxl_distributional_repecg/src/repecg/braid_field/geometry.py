from __future__ import annotations

import math

import torch


Q8_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")

# Lead-direction chart used by the Nef/Theta experiments.  This is deliberately
# named a chart, not a torso-electrode map: it is a diagnostic view manifold.
_Q8_VIEW_DEGREES = {
    "I": (90.0, 90.0),
    "II": (150.0, 90.0),
    "V1": (90.0, -10.0),
    "V2": (90.0, 10.0),
    "V3": (95.0, 15.0),
    "V4": (99.0, 30.0),
    "V5": (96.0, 60.0),
    "V6": (96.0, 90.0),
}


def canonical_q8_geometry(
    *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical Q8 operators and their fixed 2-D lead-direction chart.

    operators: [8, 8] identity basis in the independent-lead span.
    coords: [8, 2] angular chart in radians, ordered as Q8_LEADS.
    """
    operators = torch.eye(8, dtype=dtype, device=device)
    coords = torch.tensor(
        [[math.radians(v) for v in _Q8_VIEW_DEGREES[name]] for name in Q8_LEADS],
        dtype=dtype,
        device=device,
    )
    return operators, coords


def make_interpolated_query_bank(
    n_theta: int = 7,
    n_phi: int = 7,
    *,
    sigma: float = 0.45,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dense *diagnostic* query bank over the Q8 angular chart.

    Each chart point receives an 8-D measurement operator formed by radial-basis
    interpolation of the Q8 basis operators, followed by L2 normalization.

    This provides a reproducible smoke-test manifold for braid experiments.  It
    is not a claim that arbitrary wearable electrodes are physically equivalent
    to this interpolation.  Publication-grade experiments should replace this
    bank with a validated physiological query manifold (e.g. dense Nef queries
    or measured torso-view coordinates).
    """
    if n_theta < 2 or n_phi < 2:
        raise ValueError("n_theta and n_phi must both be >= 2")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    basis, anchor = canonical_q8_geometry(dtype=dtype, device=device)
    theta = torch.linspace(anchor[:, 0].min(), anchor[:, 0].max(), n_theta, dtype=dtype, device=device)
    phi = torch.linspace(anchor[:, 1].min(), anchor[:, 1].max(), n_phi, dtype=dtype, device=device)
    grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing="ij")
    coords = torch.stack((grid_theta.reshape(-1), grid_phi.reshape(-1)), dim=-1)

    d2 = torch.cdist(coords, anchor).square()
    weights = torch.softmax(-d2 / (2.0 * sigma * sigma), dim=-1)
    operators = weights @ basis
    operators = operators / operators.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return operators, coords
