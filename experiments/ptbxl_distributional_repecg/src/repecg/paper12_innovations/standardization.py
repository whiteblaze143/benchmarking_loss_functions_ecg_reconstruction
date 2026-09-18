"""Train-only phase-coordinate standardization for Paper 12 V2."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhaseCoordinateStandardizer:
    mean: torch.Tensor
    scale: torch.Tensor
    floor: float = 1e-6

    @classmethod
    def fit(cls, values: torch.Tensor, *, floor: float = 1e-6) -> "PhaseCoordinateStandardizer":
        if values.ndim != 3 or values.shape[1] != 16:
            raise ValueError("expected (record,16,coordinate) values")
        if floor != 1e-6:
            raise ValueError("Paper 12 V2 scale floor is frozen at 1e-6")
        precise = values.double()
        mean = precise.mean(dim=0, keepdim=True)
        scale = precise.std(dim=0, keepdim=True, unbiased=False).clamp_min(floor)
        return cls(mean=mean, scale=scale, floor=floor)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1:] != self.mean.shape[1:]:
            raise ValueError("phase-coordinate shape differs from fitted standardizer")
        return ((values.double() - self.mean) / self.scale).to(values.dtype)

    def floored_fraction(self, original_values: torch.Tensor) -> torch.Tensor:
        raw_scale = original_values.double().std(dim=0, unbiased=False)
        return (raw_scale <= self.floor).float().mean()
