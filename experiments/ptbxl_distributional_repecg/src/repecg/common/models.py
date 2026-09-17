from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CircularResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=0)
        self.conv2 = nn.Conv1d(width, width, 3, padding=0)
        self.activation = nn.GELU()

    def _conv(self, x: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
        return layer(F.pad(x, (1, 1), mode="circular"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self._conv(x, self.conv1))
        x = self._conv(x, self.conv2)
        return self.activation(x + residual)


class PhaseCNN(nn.Module):
    def __init__(self, input_dim: int, width: int = 128, blocks: int = 3, classes: int = 5):
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 1)
        self.blocks = nn.Sequential(*(CircularResidualBlock(width) for _ in range(blocks)))
        self.head = nn.Linear(width, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if phase_features.ndim != 3:
            raise ValueError("expected (batch,phase,feature)")
        x = phase_features.transpose(1, 2)
        x = self.blocks(self.input(x)).mean(dim=-1)
        return self.head(x)


class RecurrenceCNN(nn.Module):
    def __init__(self, classes: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(32, 128)
        self.head = nn.Linear(128, classes)

    def forward(self, operator: torch.Tensor) -> torch.Tensor:
        if operator.ndim == 3:
            operator = operator[:, None]
        x = self.encoder(operator).flatten(1)
        return self.head(torch.nn.functional.gelu(self.projection(x)))
