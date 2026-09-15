"""Frozen NEF representation graft for the A0 pre-axial-decoder grid."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


NEF_SOURCE_ROOT = Path("/data/mithunmanivannan/nef-net-aim/src")
if str(NEF_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(NEF_SOURCE_ROOT))

from nef_net_aim.components.angle_embedding import PTBXL_12LEAD_ANGLES_RAD  # noqa: E402
from nef_net_aim.core import NEFCore  # noqa: E402


class FrozenNEFPreDecoderGraft(nn.Module):
    """Inject target-conditioned NEF features into missing-lead A0 grid slots."""

    def __init__(self, mode: str, checkpoint: str | Path, width: int = 768, patches: int = 200):
        super().__init__()
        if mode not in {"AE_VE", "FULL"}:
            raise ValueError(f"unsupported hybrid NEF mode: {mode}")
        self.mode = mode
        self.patches = int(patches)
        # Construct the trainable adapter before the mode-dependent core so H1/H2
        # receive identical projection initialization under the same seed.
        self.projection = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width))
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.core = NEFCore(mode=mode, dim_geom=128)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("mode") != mode:
            raise ValueError(f"NEF checkpoint mode mismatch: {payload.get('mode')} != {mode}")
        self.core.load_state_dict(payload["model_state_dict"], strict=True)
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        self.core.eval()
        self.register_buffer(
            "standard_angles", PTBXL_12LEAD_ANGLES_RAD.clone(), persistent=True
        )
        self.last_identity_grid_error = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.core.eval()
        return self

    def forward(
        self,
        grid: torch.Tensor,
        normalized: torch.Tensor,
        inherited: torch.Tensor,
        artificial: torch.Tensor,
    ) -> torch.Tensor:
        if grid.shape[1:] != (12, self.patches, self.projection[0].out_features):
            raise ValueError(f"unexpected A0 pre-decoder grid shape: {tuple(grid.shape)}")
        if normalized.shape[1] != 12 or normalized.shape[-1] != 5000:
            raise ValueError(f"unexpected normalized ECG shape: {tuple(normalized.shape)}")
        # This experiment is strictly Lead-I-only. Fail instead of generalizing.
        expected = torch.ones_like(inherited, dtype=torch.bool)
        expected[:, 0] = False
        if not torch.equal(inherited, expected):
            raise ValueError("A0 NEF Stage-1 requires Lead I as the only observed lead")

        batch = normalized.shape[0]
        source = normalized[:, 0:1]
        query_angles = self.standard_angles[1:12].to(normalized).unsqueeze(0).expand(batch, -1, -1)
        source_angles = self.standard_angles[0:1].to(normalized).unsqueeze(0).expand(batch, -1, -1)
        q_embedding = self.core.angle_embed(query_angles)
        source_embedding = self.core.angle_embed(source_angles)
        features = self.core.extract_features(source, q_embedding, source_embedding)
        b, q, channels, samples = features.shape
        adapted = F.adaptive_avg_pool1d(
            features.reshape(b * q, channels, samples), self.patches
        ).reshape(b, q, channels, self.patches).permute(0, 1, 3, 2)
        residual = self.projection(adapted)
        output = grid.clone()
        output[:, 1:12] = grid[:, 1:12] + self.alpha * residual
        self.last_identity_grid_error = float((output.detach() - grid.detach()).abs().max())
        return output
