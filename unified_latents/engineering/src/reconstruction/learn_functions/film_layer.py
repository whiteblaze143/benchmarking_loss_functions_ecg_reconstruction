"""Feature-wise linear modulation (FiLM) layer."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class FiLMModulation(nn.Module):
    """Apply FiLM scaling and shifting to convolutional feature maps.

    Args:
        channel_dim: Number of channels in the feature map to modulate.
        cond_dim: Dimensionality of the conditioning vector.
    """

    def __init__(self, channel_dim: int, cond_dim: int = 64) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.cond_dim = cond_dim

        self.proj = nn.Linear(cond_dim, channel_dim * 2)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
            with torch.no_grad():
                self.proj.bias[: self.channel_dim].fill_(1.0)  # gamma bias
                self.proj.bias[self.channel_dim :].zero_()  # beta bias

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation."""
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0)

        gamma_beta = self.proj(conditioning)
        gamma, beta = torch.split(gamma_beta, self.channel_dim, dim=-1)

        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        return gamma * features + beta

