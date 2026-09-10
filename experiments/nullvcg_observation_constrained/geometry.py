"""Fixed LVCG geometry for the validation-only NullVCG oracle."""

from __future__ import annotations

import torch


INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

LEAD_DIRECTIONS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.5, 0.86603, 0.0],
        [-0.33682, 0.17365, 0.92542],
        [0.33682, 0.17365, 0.92542],
        [0.75441, 0.17365, 0.63302],
        [0.96985, 0.17365, 0.17101],
        [0.92542, 0.17365, -0.33682],
        [0.63302, 0.17365, -0.75441],
    ],
    dtype=torch.float64,
)


def _matrix_like(signal: torch.Tensor) -> torch.Tensor:
    return LEAD_DIRECTIONS.to(device=signal.device, dtype=signal.dtype)


def independent_from_standard(ecg: torch.Tensor) -> torch.Tensor:
    """Select I, II, V1--V6 from a standard-order 12-lead tensor."""
    if ecg.shape[-2] != 12:
        raise ValueError(f"expected 12 leads, got shape {tuple(ecg.shape)}")
    return ecg[..., (0, 1, 6, 7, 8, 9, 10, 11), :]


def standard_from_independent(ecg: torch.Tensor) -> torch.Tensor:
    """Derive III/aVR/aVL/aVF exactly from I and II."""
    if ecg.shape[-2] != 8:
        raise ValueError(f"expected 8 independent leads, got shape {tuple(ecg.shape)}")
    lead_i, lead_ii = ecg[..., 0, :], ecg[..., 1, :]
    return torch.stack(
        (
            lead_i,
            lead_ii,
            lead_ii - lead_i,
            -(lead_i + lead_ii) / 2,
            lead_i - lead_ii / 2,
            lead_ii - lead_i / 2,
            *(ecg[..., index, :] for index in range(2, 8)),
        ),
        dim=-2,
    )


def project_vcg(vcg: torch.Tensor) -> torch.Tensor:
    """Project [..., 3, time] VCG coordinates to eight independent leads."""
    if vcg.shape[-2] != 3:
        raise ValueError(f"expected 3 VCG coordinates, got shape {tuple(vcg.shape)}")
    return torch.einsum("lv,...vt->...lt", _matrix_like(vcg), vcg)


def full_oracle(ecg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Best fixed-geometry least-squares reconstruction using all eight leads."""
    observed = independent_from_standard(ecg)
    inverse = torch.linalg.pinv(_matrix_like(observed))
    vcg = torch.einsum("vl,...lt->...vt", inverse, observed)
    return standard_from_independent(project_vcg(vcg)), vcg


def constrained_oracle(ecg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Best y/z reconstruction with VCG x fixed exactly to measured Lead I."""
    observed = independent_from_standard(ecg)
    matrix = _matrix_like(observed)
    x = observed[..., 0, :]
    residual = observed[..., 1:, :] - matrix[1:, 0, None] * x[..., None, :]
    yz = torch.einsum("vl,...lt->...vt", torch.linalg.pinv(matrix[1:, 1:]), residual)
    vcg = torch.cat((x[..., None, :], yz), dim=-2)
    return standard_from_independent(project_vcg(vcg)), vcg


def residual_subspace_oracle(
    ecg: torch.Tensor,
    lead_i_coefficients: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle reconstruction in ``Y = c I + B z`` for seven missing leads.

    ``basis`` is expected to have orthonormal columns. The returned latent is the
    least-squares projection of the true residual and is therefore an oracle,
    not a deployable predictor.
    """
    observed = independent_from_standard(ecg)
    lead_i = observed[..., 0, :]
    missing = observed[..., 1:, :]
    coefficients = lead_i_coefficients.to(device=ecg.device, dtype=ecg.dtype)
    projection = basis.to(device=ecg.device, dtype=ecg.dtype)
    if coefficients.shape != (7,):
        raise ValueError(f"expected seven Lead-I coefficients, got {tuple(coefficients.shape)}")
    if projection.ndim != 2 or projection.shape[0] != 7:
        raise ValueError(f"expected a [7, rank] basis, got {tuple(projection.shape)}")
    residual = missing - coefficients[:, None] * lead_i[..., None, :]
    latent = torch.einsum("lr,...lt->...rt", projection, residual)
    residual_hat = torch.einsum("lr,...rt->...lt", projection, latent)
    missing_hat = coefficients[:, None] * lead_i[..., None, :] + residual_hat
    independent = torch.cat((lead_i[..., None, :], missing_hat), dim=-2)
    return standard_from_independent(independent), latent


def geometry_diagnostics() -> dict[str, float | int | list[int]]:
    missing_yz = LEAD_DIRECTIONS[1:, 1:]
    return {
        "shape": list(LEAD_DIRECTIONS.shape),
        "rank": int(torch.linalg.matrix_rank(LEAD_DIRECTIONS)),
        "missing_yz_rank": int(torch.linalg.matrix_rank(missing_yz)),
        "missing_yz_condition_number": float(torch.linalg.cond(missing_yz)),
        "trainable_geometry_parameters": 0,
    }
