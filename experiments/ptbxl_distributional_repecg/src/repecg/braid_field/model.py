from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from repecg.paper07_operator import OperatorSetModel


@dataclass(frozen=True)
class BraidFieldConfig:
    response_dim: int = 128
    classes: int = 5
    slots_per_sign: int = 3
    temperature: float = 0.15
    braid_dim: int = 128
    event_dim: int = 64
    mc_samples: int = 4
    variant: str = "field_braid_event"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SoftBraidReadout(nn.Module):
    """Differentiable braid surrogate from a scalar field on a supplied 2-D chart.

    The module tracks multiple soft maxima and minima and summarizes pairwise
    winding.  It is intentionally a *surrogate* used for optimization; exact
    critical-point topology should be audited separately.
    """

    def __init__(
        self,
        *,
        slots_per_sign: int = 3,
        temperature: float = 0.15,
        braid_dim: int = 128,
        event_dim: int = 64,
    ):
        super().__init__()
        if slots_per_sign < 1:
            raise ValueError("slots_per_sign must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.slots_per_sign = slots_per_sign
        self.temperature = temperature
        self.track_count = 2 * slots_per_sign

        self.max_bias = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, slots_per_sign))
        self.min_bias = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, slots_per_sign))

        pair_count = self.track_count * (self.track_count - 1) // 2
        self.braid_mlp = nn.Sequential(
            nn.Linear(pair_count * 6, max(128, braid_dim)),
            nn.GELU(),
            nn.Linear(max(128, braid_dim), braid_dim),
        )
        self.event_mlp = nn.Sequential(
            nn.Linear(self.track_count * 4, max(64, event_dim)),
            nn.GELU(),
            nn.Linear(max(64, event_dim), event_dim),
        )

    @staticmethod
    def _coord_features(coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError("coords must be [query,2]")
        x, y = coords.unbind(dim=-1)
        return torch.stack((x, y, torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)), dim=-1)

    def _tracks(
        self, field: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if field.ndim != 3:
            raise ValueError("field must be [batch,time,query]")
        if field.shape[-1] != len(coords):
            raise ValueError("field/query geometry mismatch")

        features = self._coord_features(coords)
        max_bias = self.max_bias(features)
        min_bias = self.min_bias(features)

        max_score = field.unsqueeze(-1) / self.temperature + max_bias[None, None]
        min_score = -field.unsqueeze(-1) / self.temperature + min_bias[None, None]
        max_weight = torch.softmax(max_score, dim=2)
        min_weight = torch.softmax(min_score, dim=2)

        max_pos = torch.einsum("btqk,qd->btkd", max_weight, coords)
        min_pos = torch.einsum("btqk,qd->btkd", min_weight, coords)
        tracks = torch.cat((max_pos, min_pos), dim=2)

        # Concentration is high when a slot is localized rather than diffuse.
        max_conf = max_weight.square().sum(dim=2)
        min_conf = min_weight.square().sum(dim=2)
        confidence = torch.cat((max_conf, min_conf), dim=2)
        return tracks, confidence

    def _pair_features(self, tracks: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = []
        eps = 1e-6
        for left, right in itertools.combinations(range(self.track_count), 2):
            relative = tracks[:, :, left] - tracks[:, :, right]
            r0 = relative[:, :-1]
            r1 = relative[:, 1:]
            cross = r0[..., 0] * r1[..., 1] - r0[..., 1] * r1[..., 0]
            dot = (r0 * r1).sum(dim=-1)
            dtheta = torch.atan2(cross, dot + eps)

            distance = relative.norm(dim=-1)
            pair_conf = 0.5 * (confidence[:, :, left] + confidence[:, :, right])
            features.extend(
                (
                    dtheta.sum(dim=1, keepdim=True) / (2.0 * math.pi),
                    dtheta.abs().sum(dim=1, keepdim=True) / (2.0 * math.pi),
                    dtheta.abs().mean(dim=1, keepdim=True),
                    distance.amin(dim=1, keepdim=True),
                    distance.mean(dim=1, keepdim=True),
                    pair_conf.mean(dim=1, keepdim=True),
                )
            )
        return torch.cat(features, dim=-1)

    def _event_features(self, confidence: torch.Tensor) -> torch.Tensor:
        occupancy = torch.sigmoid((confidence - 0.15) / 0.05)
        delta = occupancy[:, 1:] - occupancy[:, :-1]
        births = torch.relu(delta).sum(dim=1)
        deaths = torch.relu(-delta).sum(dim=1)
        mean_active = occupancy.mean(dim=1)
        mean_conf = confidence.mean(dim=1)
        return torch.cat((births, deaths, mean_active, mean_conf), dim=-1)

    def separation_loss(self, tracks: torch.Tensor) -> torch.Tensor:
        penalties = []
        for left, right in itertools.combinations(range(self.track_count), 2):
            d2 = (tracks[:, :, left] - tracks[:, :, right]).square().sum(dim=-1)
            penalties.append(torch.exp(-d2 / 0.02).mean())
        return torch.stack(penalties).mean()

    def forward(self, field: torch.Tensor, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        tracks, confidence = self._tracks(field, coords)
        raw_braid = self._pair_features(tracks, confidence)
        raw_event = self._event_features(confidence)
        return {
            "braid_embedding": self.braid_mlp(raw_braid),
            "event_embedding": self.event_mlp(raw_event),
            "tracks": tracks,
            "track_confidence": confidence,
            "raw_braid": raw_braid,
            "raw_event": raw_event,
            "separation_loss": self.separation_loss(tracks),
        }


class BraidFieldClassifier(nn.Module):
    """Operator-conditioned canonical-field classifier with braid readouts.

    Variants:
      field                 : latent field/context only
      field_braid           : field + winding readout
      field_braid_event     : field + winding + soft birth/death readout
      field_braid_inv       : same architecture as event; paired loss is external
      field_braid_prob      : variational latent + MC field/braid marginalization
    """

    VARIANTS = {
        "field",
        "field_braid",
        "field_braid_event",
        "field_braid_inv",
        "field_braid_prob",
    }

    def __init__(self, config: BraidFieldConfig):
        super().__init__()
        if config.variant not in self.VARIANTS:
            raise ValueError(f"unknown braid-field variant: {config.variant}")
        self.config = config
        self.backbone = OperatorSetModel(
            response_dim=config.response_dim,
            classes=config.classes,
            operator_mode="continuous",
        )
        self.field_scalar = nn.Sequential(
            nn.LayerNorm(config.response_dim),
            nn.Linear(config.response_dim, 1),
        )
        self.braid = SoftBraidReadout(
            slots_per_sign=config.slots_per_sign,
            temperature=config.temperature,
            braid_dim=config.braid_dim,
            event_dim=config.event_dim,
        )

        self.posterior_mu = nn.Linear(256, 256)
        self.posterior_logvar = nn.Linear(256, 256)

        fusion_dim = 256
        if config.variant != "field":
            fusion_dim += config.braid_dim
        if config.variant in {"field_braid_event", "field_braid_inv", "field_braid_prob"}:
            fusion_dim += config.event_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(256, config.classes)

    def encode_context(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backbone.encode_context(operators, responses, mask)

    def decode_responses(self, latent: torch.Tensor, query_operators: torch.Tensor) -> torch.Tensor:
        """Decode response features at arbitrary 8-D operators.

        latent: [B,256]
        query_operators: [Q,8] or [B,Q,8]
        returns [B,Q,16,response_dim]
        """
        if query_operators.ndim == 2:
            query_operators = query_operators.unsqueeze(0).expand(len(latent), -1, -1)
        if query_operators.ndim != 3 or query_operators.shape[0] != len(latent):
            raise ValueError("query_operators must be [Q,8] or [B,Q,8]")
        encoded_q = self.backbone.encode_operator(query_operators, None)
        expanded = latent.unsqueeze(1).expand(-1, query_operators.shape[1], -1)
        decoded = self.backbone.decoder(torch.cat((expanded, encoded_q), dim=-1))
        return decoded.reshape(
            len(latent), query_operators.shape[1], 16, self.config.response_dim
        )

    def decode_scalar_field(self, latent: torch.Tensor, query_operators: torch.Tensor) -> torch.Tensor:
        responses = self.decode_responses(latent, query_operators)
        # [B,Q,T,F] -> [B,T,Q]
        return self.field_scalar(responses).squeeze(-1).transpose(1, 2)

    def _fuse(
        self,
        latent: torch.Tensor,
        query_operators: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        field = self.decode_scalar_field(latent, query_operators)
        if self.config.variant == "field":
            fused = self.fusion(latent)
            return {
                "logits": self.head(fused),
                "latent": latent,
                "field": field,
                "braid_embedding": torch.zeros(len(latent), 0, device=latent.device),
                "event_embedding": torch.zeros(len(latent), 0, device=latent.device),
                "separation_loss": field.new_zeros(()),
            }

        topology = self.braid(field, query_coords)
        pieces = [latent, topology["braid_embedding"]]
        if self.config.variant in {"field_braid_event", "field_braid_inv", "field_braid_prob"}:
            pieces.append(topology["event_embedding"])
        fused = self.fusion(torch.cat(pieces, dim=-1))
        return {"logits": self.head(fused), "latent": latent, "field": field, **topology}

    def forward(
        self,
        operators: torch.Tensor,
        responses: torch.Tensor,
        query_operators: torch.Tensor,
        query_coords: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode_context(operators, responses, mask)
        if self.config.variant != "field_braid_prob":
            output = self._fuse(latent, query_operators, query_coords)
            output["kl"] = latent.new_zeros(())
            return output

        mu = self.posterior_mu(latent)
        logvar = self.posterior_logvar(latent).clamp(-8.0, 4.0)
        draws = []
        for _ in range(self.config.mc_samples):
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            draws.append(self._fuse(z, query_operators, query_coords))
        logits = torch.stack([item["logits"] for item in draws], dim=0)
        result = draws[0]
        result["logits"] = logits.mean(dim=0)
        result["predictive_logit_var"] = logits.var(dim=0, unbiased=False)
        result["posterior_mu"] = mu
        result["posterior_logvar"] = logvar
        return result
