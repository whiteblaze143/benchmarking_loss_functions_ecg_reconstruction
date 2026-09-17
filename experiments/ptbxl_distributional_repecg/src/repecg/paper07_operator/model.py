from __future__ import annotations

import torch
from torch import nn

from repecg.common.models import PhaseCNN


class OperatorSetModel(nn.Module):
    """Diagnose and optionally reconstruct from operator-response pairs."""

    def __init__(
        self,
        response_dim: int = 128,
        classes: int = 5,
        *,
        operator_mode: str = "continuous",
        vocabulary_size: int = 0,
    ):
        super().__init__()
        if operator_mode not in {"continuous", "categorical"}:
            raise ValueError("operator_mode must be continuous or categorical")
        if operator_mode == "categorical" and vocabulary_size < 1:
            raise ValueError("categorical models require a non-empty training vocabulary")
        self.operator_mode = operator_mode

        # Matched modules are initialized before the mode-specific encoder, so
        # they have identical initial states under the same random seed.
        self.response = PhaseCNN(response_dim, width=128, blocks=2, classes=128)
        layer = nn.TransformerEncoderLayer(
            d_model=192, nhead=4, batch_first=True, dim_feedforward=384, activation="gelu"
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, 192))
        self.pool = nn.MultiheadAttention(192, 4, batch_first=True)
        self.latent = nn.Sequential(nn.Linear(192, 256), nn.GELU())
        self.head = nn.Linear(256, classes)
        self.decoder = nn.Sequential(
            nn.Linear(320, 256), nn.GELU(), nn.Linear(256, 16 * response_dim)
        )

        if operator_mode == "continuous":
            self.operator = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 64))
            self.known_operator = None
            self.register_buffer("unknown_operator", torch.empty(0), persistent=False)
        else:
            self.operator = None
            self.known_operator = nn.Embedding(vocabulary_size, 64)
            self.register_buffer("unknown_operator", torch.zeros(64), persistent=True)

    def encode_operator(
        self, operators: torch.Tensor | None, operator_ids: torch.Tensor | None
    ) -> torch.Tensor:
        if self.operator_mode == "continuous":
            if operators is None or operators.shape[-1] != 8:
                raise ValueError("continuous models require (...,8) operators")
            assert self.operator is not None
            return self.operator(operators)
        if operator_ids is None:
            raise ValueError("categorical models require operator_ids")
        assert self.known_operator is not None
        encoded = self.known_operator(operator_ids.clamp_min(0))
        return torch.where(
            (operator_ids >= 0).unsqueeze(-1), encoded, self.unknown_operator.to(encoded.dtype)
        )

    def forward(
        self,
        operators: torch.Tensor | None,
        responses: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        operator_ids: torch.Tensor | None = None,
        target_operator: torch.Tensor | None = None,
        target_operator_ids: torch.Tensor | None = None,
        return_reconstruction: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if responses.ndim != 4:
            raise ValueError("responses must have shape (batch,set,phase,feature)")
        batch, count = responses.shape[:2]
        latent = self.encode_context(operators, responses, mask, operator_ids=operator_ids)
        logits = self.head(latent)
        if not return_reconstruction:
            return logits
        target_q = self.encode_operator(target_operator, target_operator_ids)
        if target_q.ndim != 2 or target_q.shape[0] != batch:
            raise ValueError("target operator must contain one operator per record")
        reconstruction = self.decoder(torch.cat((latent, target_q), dim=-1)).reshape(
            batch, 16, responses.shape[-1]
        )
        return logits, reconstruction

    def encode_context(
        self,
        operators: torch.Tensor | None,
        responses: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        operator_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if responses.ndim != 4:
            raise ValueError("responses must have shape (batch,set,phase,feature)")
        batch, count = responses.shape[:2]
        q = self.encode_operator(operators, operator_ids)
        if q.shape[:2] != (batch, count):
            raise ValueError("operator and response set dimensions differ")
        response = self.response(
            responses.reshape(batch * count, responses.shape[2], responses.shape[3])
        ).reshape(batch, count, 128)
        encoded = self.set_encoder(torch.cat((q, response), dim=-1), src_key_padding_mask=mask)
        query = self.pool_query.expand(batch, -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=mask)
        return self.latent(pooled[:, 0])


def symmetric_bernoulli_kl(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    """Mean symmetric KL for independent Bernoulli diagnostic outputs."""
    left = torch.sigmoid(left_logits).clamp(1e-6, 1.0 - 1e-6)
    right = torch.sigmoid(right_logits).clamp(1e-6, 1.0 - 1e-6)
    left_right = left * (left.log() - right.log()) + (1 - left) * (
        (1 - left).log() - (1 - right).log()
    )
    right_left = right * (right.log() - left.log()) + (1 - right) * (
        (1 - right).log() - (1 - left).log()
    )
    return 0.5 * (left_right + right_left).mean()
