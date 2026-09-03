#!/usr/bin/env python3
"""3DRECON-QT and Electrocardio-Panorama Inspired 3D Theta Spatial Reconstruction Model.

This module isolates the physical spatial query hypothesis:
Does target-lead spatial coordinate conditioning (theta, phi) improve strict 1->12 ECG
reconstruction compared to arbitrary categorical lead embeddings and permuted coordinates?

Mechanistic Principles:
- H_source = SourceEncoder(x_source) # [B, P=200, D=768]
- For target lead l: u_l = ThetaEncoder(theta_l, phi_l) -> 12-D
- g_l = MLP_theta(u_l) -> D-D
- Z_l = H_source * g_l (multiplicative conditioning, initialized to identity)
- Xhat_l = SharedECGDecoder(Z_l) (identical temporal decoder parameters across all leads)

NO learned categorical lead IDs in theta cells.
NO source theta conditioning.
NO cross-attention or inter-lead self-attention.
NO metadata, patient tokens, or foundation model features.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard 12-lead order in ECG-AIM:
# 0: I, 1: II, 2: III, 3: aVR, 4: aVL, 5: aVF, 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

ECGAIM_THETA = torch.tensor(
    [
        [math.pi / 2, math.pi / 2],          # I
        [5 * math.pi / 6, math.pi / 2],      # II
        [5 * math.pi / 6, -math.pi / 2],     # III
        [math.pi / 3, -math.pi / 2],         # aVR
        [math.pi / 3, math.pi / 2],          # aVL
        [math.pi, math.pi / 2],              # aVF
        [math.pi / 2, -math.pi / 18],        # V1
        [math.pi / 2, math.pi / 18],         # V2
        [19 * math.pi / 36, math.pi / 12],   # V3
        [11 * math.pi / 20, math.pi / 6],    # V4
        [16 * math.pi / 30, math.pi / 3],    # V5
        [16 * math.pi / 30, math.pi / 2],    # V6
    ],
    dtype=torch.float32,
)


class ThetaEncoder(nn.Module):
    """Preserves the released Panorama angular encoding exactly.

    Input: theta: [B, L, 2] (spherical angles theta, phi)
    Output: encoded: [B, L, 12]
    """

    def __init__(self, encoder_len: int = 1):
        super().__init__()
        self.encoder_len = encoder_len
        self.omega = 1.0

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        b, lead_num = theta.shape[:2]
        sum_theta = theta[..., 0:1] + theta[..., 1:2]
        sub_theta = theta[..., 0:1] - theta[..., 1:2]
        before_encode = torch.cat(
            [theta, sum_theta, sub_theta],
            dim=-1,
        )  # [B, L, 4]
        out_all = [
            before_encode,
            torch.sin(before_encode * self.omega),
            torch.cos(before_encode * self.omega),
        ]
        # IMPORTANT: Preserve original stack->view ordering rather than replacing
        # with torch.cat, which alters 12-D feature alignment.
        return torch.stack(out_all, dim=-1).view(b, lead_num, -1)


class ThreeDSpatialConditioner(nn.Module):
    """3DRECON / Panorama-style TARGET lead spatial conditioner.

    Isolates the spatial query hypothesis:
    - no source theta conditioning
    - no learned lead ID in pure theta cells
    - no cross-attention
    - no inter-lead self-attention
    """

    def __init__(
        self,
        width: int = 768,
        code_mode: Literal["theta", "permuted_theta", "learned", "random_fixed", "learned_additive"] = "theta",
        fusion: Literal["mul", "add"] = "mul",
        random_seed: int = 20260902,
    ):
        super().__init__()
        self.width = width
        self.code_mode = code_mode
        self.fusion = fusion

        self.theta_encoder = ThetaEncoder(encoder_len=1)
        self.register_buffer("lead_angles", ECGAIM_THETA.clone(), persistent=True)

        # Fixed derangement permutation: target labels remain unchanged in evaluation,
        # but the physical coordinates are assigned incorrectly to test spatial sensitivity.
        perm = (torch.arange(12) + 5) % 12
        self.register_buffer("theta_permutation", perm, persistent=True)

        if code_mode == "learned":
            # Match ThetaEncoder dimensionality (12) before projection
            self.learned_codes = nn.Parameter(torch.randn(12, 12) * 0.02)
        else:
            self.learned_codes = None

        if code_mode == "random_fixed":
            g = torch.Generator()
            g.manual_seed(random_seed)
            random_codes = torch.randn(12, 12, generator=g)
            # Standardize feature dimensions so scale alone cannot distinguish this control
            random_codes = (random_codes - random_codes.mean(dim=0, keepdim=True)) / (
                random_codes.std(dim=0, keepdim=True) + 1e-6
            )
            self.register_buffer("random_codes", random_codes, persistent=True)
        else:
            self.random_codes = None

        if code_mode == "learned_additive":
            # Baseline A0-style additive categorical embedding
            self.lead_embed = nn.Parameter(torch.randn(12, width) * 0.02)
            self.code_projection = None
        else:
            self.lead_embed = None
            self.code_projection = nn.Sequential(
                nn.Linear(12, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            final = self.code_projection[-1]
            if fusion == "mul":
                # Start as identity: H * 1 = H
                nn.init.zeros_(final.weight)
                nn.init.ones_(final.bias)
            else:
                # Start as identity: H + 0 = H
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def _codes(self, batch: int) -> torch.Tensor:
        if self.code_mode in {"theta", "permuted_theta"}:
            theta = self.lead_angles
            if self.code_mode == "permuted_theta":
                theta = theta[self.theta_permutation]
            theta = theta.unsqueeze(0).expand(batch, -1, -1)
            return self.theta_encoder(theta)
        if self.code_mode == "learned":
            return self.learned_codes.unsqueeze(0).expand(batch, -1, -1)
        if self.code_mode == "random_fixed":
            return self.random_codes.unsqueeze(0).expand(batch, -1, -1)
        raise ValueError(f"Unknown code_mode: {self.code_mode}")

    def forward(self, h_source: torch.Tensor) -> torch.Tensor:
        """h_source: [B, P, D]

        returns: h_grid [B, 12, P, D]
        """
        batch = h_source.shape[0]

        if self.code_mode == "learned_additive":
            # Traditional A0 additive broadcast
            return h_source[:, None, :, :] + self.lead_embed[None, :, None, :]

        codes = self._codes(batch)  # [B, 12, 12]
        condition = self.code_projection(codes)  # [B, 12, D]

        source = h_source[:, None, :, :]  # [B, 1, P, D]
        condition = condition[:, :, None, :]  # [B, 12, 1, D]

        if self.fusion == "mul":
            return source * condition
        if self.fusion == "add":
            return source + condition
        raise ValueError(f"Unknown fusion mode: {self.fusion}")


class SharedECGDecoder(nn.Module):
    """One shared temporal transformer decoder for all target leads.

    All lead-specific distinction must arrive strictly through the spatial code.
    """

    def __init__(
        self,
        width: int = 768,
        patch_size: int = 25,
        depth: int = 4,
        heads: int = 12,
    ):
        super().__init__()
        if width % heads != 0:
            for h in [12, 8, 4, 2, 1]:
                if width % h == 0:
                    heads = h
                    break
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(width),
        )
        self.patch_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, patch_size),
        )

    def forward(self, h_grid: torch.Tensor) -> torch.Tensor:
        """h_grid: [B, 12, P, D]

        return: [B, 12, P * patch_size = 5000]
        """
        B, L, P, D = h_grid.shape
        x = h_grid.reshape(B * L, P, D)
        # Shared temporal transformer across all leads
        x = self.decoder(x)
        patches = self.patch_head(x)  # [B * L, P, patch_size]
        return patches.reshape(B, L, -1)


class ThreeDThetaECGAIM(nn.Module):
    """Full 1->12 ECG Reconstruction Model with 3D Theta Spatial Conditioning.

    Controlled Cell Modes (Stage A):
    - D0_current_id_currentloss: code_mode='learned_additive', fusion='add'
    - D1_theta_mul_currentloss: code_mode='theta', fusion='mul'
    - D2_current_id_l1: code_mode='learned_additive', fusion='add'
    - D3_theta_mul_l1: code_mode='theta', fusion='mul' (PRIMARY 3DRECON)
    - D4_learned12_mul_l1: code_mode='learned', fusion='mul' (Capacity control)
    - D5_permuted_theta_mul_l1: code_mode='permuted_theta', fusion='mul' (Geometry control)
    """

    def __init__(
        self,
        code_mode: str = "theta",
        fusion: str = "mul",
        patch_size: int = 25,
        width: int = 768,
        encoder_depth: int = 8,
        decoder_depth: int = 4,
        heads: int = 12,
        use_delineation_head: bool = False,
    ):
        super().__init__()
        self.code_mode = code_mode
        self.fusion = fusion
        self.patch_size = patch_size
        self.width = width
        self.num_patches = 5000 // patch_size

        # 1. Source Lead Morphology Encoder (exact parameter-matched backbone)
        self.patch_proj = nn.Linear(patch_size, width)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, width) * 0.02)
        enc_heads = heads
        if width % enc_heads != 0:
            for h in [12, 8, 4, 2, 1]:
                if width % h == 0:
                    enc_heads = h
                    break
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=enc_heads,
            dim_feedforward=4 * width,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        # 2. 3D Spatial Conditioner
        self.conditioner = ThreeDSpatialConditioner(
            width=width,
            code_mode=code_mode,
            fusion=fusion,
        )

        # 3. Shared ECG Decoder
        self.decoder = SharedECGDecoder(
            width=width,
            patch_size=patch_size,
            depth=decoder_depth,
            heads=heads,
        )

        # 4. Multi-Task Delineation Head (Stage B option)
        if use_delineation_head:
            self.delineation_head = nn.Sequential(
                nn.Conv1d(width, 96, kernel_size=15, padding=7),
                nn.GELU(),
                nn.Conv1d(96, 4 * patch_size, kernel_size=1),
            )
        else:
            self.delineation_head = None

    def forward(
        self,
        x_source: torch.Tensor,
        obs_lead_idx: int = 0,
        compute_delineation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """x_source: [B, 1, 5000] (raw physical millivolts of observed lead)

        returns:
          y_pred: [B, 12, 5000] (reconstructed 12 leads)
          h_grid: [B, 12, 200, 768] (conditioned spatial representations)
          seg_logits: [B, 12, 5000, 4] (if compute_delineation and enabled)
        """
        B = x_source.shape[0]

        # 1. Patchify source lead: [B, 200, 25] -> [B, 200, width]
        patches = x_source.reshape(B, self.num_patches, self.patch_size)
        h_source = self.patch_proj(patches) + self.pos_embed
        h_source = self.source_encoder(h_source)  # [B, 200, width]

        # 2. Condition on 3D spatial target codes: [B, 12, 200, width]
        h_grid = self.conditioner(h_source)

        # 3. Decode all target leads through identical shared decoder
        waveform_pred = self.decoder(h_grid)  # [B, 12, 5000]

        # Strictly enforce observed lead identity on output
        waveform_pred[:, obs_lead_idx, :] = x_source[:, 0, :]

        seg_logits = None
        if compute_delineation and self.delineation_head is not None:
            h_flat = h_grid.reshape(B * 12, self.num_patches, self.width).permute(0, 2, 1)
            del_out = self.delineation_head(h_flat)  # [B * 12, 4 * 25, 200]
            seg_logits = (
                del_out.reshape(B, 12, 4, self.patch_size, self.num_patches)
                .permute(0, 1, 4, 3, 2)
                .reshape(B, 12, 5000, 4)
            )

        return {
            "y_pred": waveform_pred,
            "h_grid": h_grid,
            "seg_logits": seg_logits,
        }


def missing_lead_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    observed_lead: int = 0,
) -> torch.Tensor:
    """Genuine missing-lead L1 loss.

    Strictly computes mean absolute error over the 11 unobserved leads.
    """
    keep = torch.ones(pred.shape[1], dtype=torch.bool, device=pred.device)
    keep[observed_lead] = False
    return F.l1_loss(pred[:, keep], target[:, keep])
