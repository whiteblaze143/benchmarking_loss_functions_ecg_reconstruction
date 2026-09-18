"""Conditional location-scale innovations over frozen phase features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LocationScaleInnovationModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        phases: int = 16,
        width: int = 64,
        classes: int = 5,
        *,
        unit_scale_init: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.phases = phases
        self.predictor = nn.GRU(input_dim, width, batch_first=True)
        self.mean = nn.Linear(width, input_dim)
        self.log_scale = nn.Linear(width, input_dim)
        if unit_scale_init:
            nn.init.zeros_(self.log_scale.weight)
            nn.init.zeros_(self.log_scale.bias)
        self.head = nn.Sequential(nn.Linear((phases - 1) * input_dim, 128), nn.GELU(), nn.Linear(128, classes))

    def components(
        self, phase_features: torch.Tensor, history_permutation: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if phase_features.ndim != 3 or phase_features.shape[1] != self.phases or phase_features.shape[2] != self.input_dim:
            raise ValueError(f"expected (batch,{self.phases},{self.input_dim}) frozen phase features")
        history = phase_features
        if history_permutation is not None:
            if history_permutation.shape != (len(phase_features),):
                raise ValueError("history_permutation must contain one index per record")
            history = history[history_permutation]
        shifted = torch.cat((torch.zeros_like(history[:, :1]), history[:, :-1]), dim=1)
        hidden, _ = self.predictor(shifted)
        mean = self.mean(hidden)
        log_scale = self.log_scale(hidden).clamp(-4.0, 2.0)
        innovation = (phase_features - mean) * torch.exp(-log_scale)
        return mean, log_scale, innovation

    def raw_log_scale(self, phase_features: torch.Tensor) -> torch.Tensor:
        shifted = torch.cat((torch.zeros_like(phase_features[:, :1]), phase_features[:, :-1]), dim=1)
        hidden, _ = self.predictor(shifted)
        return self.log_scale(hidden)

    def forward(self, phase_features: torch.Tensor, representation: str = "innovation") -> torch.Tensor:
        mean, _, innovation = self.components(phase_features)
        if representation == "innovation":
            values = innovation
        elif representation == "mean_residual":
            values = phase_features - mean
        elif representation == "state":
            values = phase_features
        else:
            raise ValueError(f"unknown representation: {representation}")
        # Phase 0 has no observed history and is excluded from every diagnosis
        # representation. It remains available only as context for phases 1:.
        return self.head(values[:, 1:].flatten(1))

    def density_loss(self, phase_features: torch.Tensor) -> torch.Tensor:
        mean, log_scale, innovation = self.components(phase_features)
        return (0.5 * innovation[:, 1:].square() + log_scale[:, 1:]).mean()

    def freeze_density(self) -> None:
        for module in (self.predictor, self.mean, self.log_scale):
            module.requires_grad_(False)

    def diagnosis_loss(self, phase_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self(phase_features), labels)
