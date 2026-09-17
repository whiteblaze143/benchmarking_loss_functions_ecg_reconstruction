from __future__ import annotations

import torch
from torch import nn

from repecg.common.models import PhaseCNN


class OperatorSetModel(nn.Module):
    def __init__(self, response_dim: int = 128, classes: int = 5):
        super().__init__()
        self.operator = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 64))
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
        layer = nn.TransformerEncoderLayer(d_model=192, nhead=4, batch_first=True, dim_feedforward=384, activation="gelu")
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, 192))
        self.pool = nn.MultiheadAttention(192, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(192, 256), nn.GELU())
        self.head = nn.Linear(256, classes)

    def forward(self, operators: torch.Tensor, responses: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if operators.ndim != 3 or responses.ndim != 4:
            raise ValueError("expected operators (batch,set,8), responses (batch,set,phase,feature)")
        batch, count = operators.shape[:2]
        q = self.operator(operators)
        response = self.response(responses.reshape(batch * count, responses.shape[2], responses.shape[3])).reshape(batch, count, 128)
        encoded = self.set_encoder(torch.cat((q, response), dim=-1), src_key_padding_mask=mask)
        query = self.pool_query.expand(batch, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        return self.head(self.latent(pooled[:, 0]))
