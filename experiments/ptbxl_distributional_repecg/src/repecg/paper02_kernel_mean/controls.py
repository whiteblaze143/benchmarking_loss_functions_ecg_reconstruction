from __future__ import annotations

import torch


def sample_covariance(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2 or values.shape[-2] < 2:
        raise ValueError("sample covariance requires at least two observations")
    centered = values - values.mean(dim=-2, keepdim=True)
    return centered.transpose(-1, -2) @ centered / (values.shape[-2] - 1)


def mean_covariance_features(values: torch.Tensor) -> torch.Tensor:
    """Return mean and lower-triangular sample covariance without duplication."""
    dimension = values.shape[-1]
    triangle = torch.tril_indices(dimension, dimension, device=values.device)
    covariance = sample_covariance(values)
    return torch.cat(
        (values.mean(dim=-2), covariance[..., triangle[0], triangle[1]]),
        dim=-1,
    )


def _symmetric_power(matrix: torch.Tensor, exponent: float, floor: float = 0.0) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    if exponent < 0:
        largest = eigenvalues[..., -1:].clamp_min(1e-30)
        threshold = torch.maximum(largest * 1e-12, torch.full_like(largest, 1e-15))
        powered = torch.where(eigenvalues >= threshold, eigenvalues.pow(exponent), 0.0)
    else:
        powered = eigenvalues.clamp_min(floor).pow(exponent)
    return (eigenvectors * powered.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)


def exact_moment_matched_gaussian(
    values: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    tolerance: float = 1e-8,
) -> torch.Tensor:
    """Create a Gaussian draw with the realized mean and covariance of ``values``.

    Computation and validation are deliberately float64. The leading dimensions
    are arbitrary batches and the final two dimensions are observations and
    features.
    """
    if values.ndim < 2 or values.shape[-2] <= values.shape[-1]:
        raise ValueError("exact matching requires more observations than dimensions")
    target = values.to(dtype=torch.float64)
    target_mean = target.mean(dim=-2, keepdim=True)
    target_covariance = sample_covariance(target)
    noise = torch.randn(
        target.shape,
        dtype=torch.float64,
        device=target.device,
        generator=generator,
    )
    noise = noise - noise.mean(dim=-2, keepdim=True)
    noise_inverse_root = _symmetric_power(sample_covariance(noise), -0.5)
    target_root = _symmetric_power(target_covariance, 0.5, floor=0.0)
    matched = noise @ noise_inverse_root @ target_root + target_mean
    matched = matched - matched.mean(dim=-2, keepdim=True) + target_mean

    mean_error = (matched.mean(dim=-2, keepdim=True) - target_mean).abs().amax()
    covariance_error = (sample_covariance(matched) - target_covariance).abs().amax()
    if float(mean_error) >= tolerance or float(covariance_error) >= tolerance:
        raise RuntimeError(
            "moment matching failed: "
            f"mean_error={float(mean_error):.3e}, covariance_error={float(covariance_error):.3e}"
        )
    return matched
