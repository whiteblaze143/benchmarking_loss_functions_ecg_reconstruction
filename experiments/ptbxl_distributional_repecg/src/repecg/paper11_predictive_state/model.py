"""Minimal causal-prefix predictive-state model for Paper 11 gates."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PredictiveStateModel(nn.Module):
    def __init__(self, input_dim: int, states: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.state_classifier = nn.Linear(hidden_dim, states)
        self.state_embedding = nn.Parameter(torch.randn(states, hidden_dim) * 0.05)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def state_posteriors(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encoder(sequence)
        return F.softmax(self.state_classifier(hidden), dim=-1)

    def predict_next(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or sequence.shape[1] < 2:
            raise ValueError("sequence must have shape (batch, time>=2, feature)")
        states = self.state_posteriors(sequence[:, :-1])
        return self.decoder(states @ self.state_embedding), states

    def predict_from_prefix(self, prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if prefix.ndim != 3 or prefix.shape[1] < 1:
            raise ValueError("prefix must have shape (batch, time>=1, feature)")
        state = self.state_posteriors(prefix)[:, -1]
        return self.decoder(state @ self.state_embedding), state

    def future_loss(self, sequence: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        prediction, _ = self.predict_next(sequence)
        target = sequence[:, 1:] if targets is None else targets
        if target.shape != prediction.shape:
            raise ValueError("future targets must match next-step prediction shape")
        return F.mse_loss(prediction, target)


class ContinuousPredictor(nn.Module):
    """Capacity-matched causal encoder without a categorical bottleneck."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def encode_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encoder(prefix)
        return hidden[:, -1]

    def predict_from_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode_prefix(prefix))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.predict_from_prefix(prefix)


class SetPredictor(nn.Module):
    """Order-free prefix baseline using exactly the same beat contents."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def predict_from_prefix(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(prefix).mean(dim=1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.predict_from_prefix(prefix)
