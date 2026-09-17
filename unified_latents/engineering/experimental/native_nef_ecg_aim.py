"""Native, jointly trainable NEF components for the ECG-AIM latent grid."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


NEF_SOURCE_ROOT = Path("/data/mithunmanivannan/nef-net-aim/src")
if str(NEF_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(NEF_SOURCE_ROOT))

from nef_net_aim.components.angle_embedding import (  # noqa: E402
    NEFAngleEmbedding,
    PTBXL_12LEAD_ANGLES_RAD,
)
from nef_net_aim.components.geovt import (  # noqa: E402
    GeometricAwareAttention,
    HierarchicalViewTransformer,
)
from nef_net_aim.components.view_encoder import QueryConditionedViewEncoder  # noqa: E402


def _component_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    selected = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"FULL checkpoint has no parameters with prefix {prefix}")
    return selected


class NativeNEFECGAIM(nn.Module):
    """Transplant pretrained NEF mechanisms into ECG-AIM before its decoder."""

    def __init__(
        self,
        mode: str,
        checkpoint: str | Path,
        width: int = 768,
        patches: int = 200,
        fusion_init: float = 0.1,
    ):
        super().__init__()
        if mode not in {"AE_VE", "FULL"}:
            raise ValueError(f"unsupported native NEF mode: {mode}")
        self.mode = mode
        self.patches = int(patches)
        self.contribution_enabled = True

        # Shared C1/C2 modules are constructed in the same order. The runner
        # also reseeds before each arm and verifies exact shared-state equality.
        self.angle_embed = NEFAngleEmbedding(dim_geom=128, num_leads=48)
        self.view_encoder = QueryConditionedViewEncoder(
            in_channels=1, feat_dim=128, query_dim=128
        )
        self.angle_projection = nn.Linear(128, width, bias=False)
        self.view_projection = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width))
        self.beta_angle = nn.Parameter(torch.tensor(float(fusion_init)))
        self.beta_view = nn.Parameter(torch.tensor(float(fusion_init)))

        self.geovt_attn = None
        self.view_transformer = None
        self.geovt_projection = None
        self.beta_geovt = None
        if mode == "FULL":
            self.geovt_attn = GeometricAwareAttention(dim_geom=128, dim_key=128)
            self.view_transformer = HierarchicalViewTransformer(num_layers=4, feat_dim=128)
            self.geovt_projection = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width))
            self.beta_geovt = nn.Parameter(torch.tensor(float(fusion_init)))

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("mode") != "FULL":
            raise ValueError("native C1/C2 initialization requires the FULL checkpoint")
        state = payload["model_state_dict"]
        self.angle_embed.load_state_dict(_component_state(state, "angle_embed."), strict=True)
        self.view_encoder.load_state_dict(_component_state(state, "view_encoder."), strict=True)
        if mode == "FULL":
            self.geovt_attn.load_state_dict(_component_state(state, "geovt_attn."), strict=True)
            self.view_transformer.load_state_dict(
                _component_state(state, "view_transformer."), strict=True
            )

        self.register_buffer(
            "standard_angles", PTBXL_12LEAD_ANGLES_RAD.clone(), persistent=True
        )
        self.last_disabled_grid_error = None

    def _require_lead_i_only(self, inherited: torch.Tensor) -> None:
        expected = torch.ones_like(inherited, dtype=torch.bool)
        expected[:, 0] = False
        if not torch.equal(inherited, expected):
            raise ValueError("native NEF Stage-1 requires Lead I as the only observed lead")

    def augment_conditioning(self, conditioning: torch.Tensor, inherited: torch.Tensor) -> torch.Tensor:
        self._require_lead_i_only(inherited)
        if not self.contribution_enabled:
            return conditioning
        batch = conditioning.shape[0]
        query_angles = self.standard_angles[1:12].to(conditioning).unsqueeze(0).expand(batch, -1, -1)
        query = self.angle_embed(query_angles)
        output = conditioning.clone()
        output[:, 1:12] = output[:, 1:12] + self.beta_angle * self.angle_projection(query)
        return output

    def augment_grid(
        self,
        grid: torch.Tensor,
        normalized: torch.Tensor,
        inherited: torch.Tensor,
        artificial: torch.Tensor,
    ) -> torch.Tensor:
        del artificial
        self._require_lead_i_only(inherited)
        if grid.shape[1:] != (12, self.patches, self.view_projection[0].out_features):
            raise ValueError(f"unexpected A0 pre-decoder grid shape: {tuple(grid.shape)}")
        if normalized.shape[1:] != (12, 5000):
            raise ValueError(f"unexpected normalized ECG shape: {tuple(normalized.shape)}")
        if not self.contribution_enabled:
            self.last_disabled_grid_error = 0.0
            return grid

        batch = normalized.shape[0]
        source = normalized[:, 0:1]
        query_angles = self.standard_angles[1:12].to(normalized).unsqueeze(0).expand(batch, -1, -1)
        source_angles = self.standard_angles[0:1].to(normalized).unsqueeze(0).expand(batch, -1, -1)
        query = self.angle_embed(query_angles)
        source_query = self.angle_embed(source_angles)

        view_features = []
        geovt_features = []
        for index in range(11):
            target_query = query[:, index:index + 1]
            view = self.view_encoder(source, target_query)
            view_features.append(view)
            if self.mode == "FULL":
                geovt_features.append(
                    self.view_transformer(
                        source_query, target_query, [view], self.geovt_attn
                    )
                )

        view = torch.stack(view_features, dim=1)
        b, q, channels, samples = view.shape
        view = F.adaptive_avg_pool1d(
            view.reshape(b * q, channels, samples), self.patches
        ).reshape(b, q, channels, self.patches).permute(0, 1, 3, 2)

        output = grid.clone()
        output[:, 1:12] = output[:, 1:12] + self.beta_view * self.view_projection(view)
        if self.mode == "FULL":
            geovt = torch.stack(geovt_features, dim=1)
            geovt = F.adaptive_avg_pool1d(
                geovt.reshape(b * q, channels, geovt.shape[-1]), self.patches
            ).reshape(b, q, channels, self.patches).permute(0, 1, 3, 2)
            output[:, 1:12] = (
                output[:, 1:12] + self.beta_geovt * self.geovt_projection(geovt)
            )
        return output

