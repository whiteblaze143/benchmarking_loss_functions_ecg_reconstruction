import math
from typing import Optional
import torch
import torch.nn as nn

LEAD_NAMES = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

# Panorama source ordering:
# I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF
PANORAMA_ANGLES = torch.tensor([
    [math.pi / 2,        math.pi / 2],   # I
    [5 * math.pi / 6,    math.pi / 2],   # II
    [math.pi / 2,       -math.pi / 18],  # V1
    [math.pi / 2,        math.pi / 18],  # V2
    [19 * math.pi / 36,  math.pi / 12],  # V3
    [11 * math.pi / 20,  math.pi / 6],   # V4
    [16 * math.pi / 30,  math.pi / 3],   # V5
    [16 * math.pi / 30,  math.pi / 2],   # V6
    [5 * math.pi / 6,   -math.pi / 2],   # III
    [math.pi / 3,       -math.pi / 2],   # aVR
    [math.pi / 3,        math.pi / 2],   # aVL
    [math.pi,            math.pi / 2],   # aVF
], dtype=torch.float32)

# Permutation mapping Panorama ordering -> Standard 12-lead ordering:
# [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
PANORAMA_TO_STANDARD = torch.tensor(
    [0, 1, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7],
    dtype=torch.long,
)

STANDARD_ANGLES = PANORAMA_ANGLES[PANORAMA_TO_STANDARD]


class ThetaEncoder(nn.Module):
    """
    Keep behavior identical to the released Electrocardio-Panorama ThetaEncoder
    (WhatAShot/Electrocardio-Panorama, codes/network/utils/theta_encoder.py).
    
    Input:
        theta: [B, lead_num, 2]
    Output:
        encoded: [B, lead_num, 12]
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
        )

        out_all = [
            before_encode,
            torch.sin(before_encode * self.omega),
            torch.cos(before_encode * self.omega),
        ]

        # Exact public implementation ordering: torch.stack(...).view(...)
        after_encode = torch.stack(
            out_all,
            dim=-1,
        ).view(b, lead_num, -1)

        return after_encode


class ThetaQueryEncoder(nn.Module):
    """
    Query MLP with Mish activation as visually confirmed in 3DRECON-QT Fig. 2:
    angular encoding -> flatten -> Linear -> Mish -> Linear
    """

    def __init__(
        self,
        latent_channels: int = 256,
        hidden: int = 128,
    ):
        super().__init__()
        self.theta = ThetaEncoder(1)
        self.mlp = nn.Sequential(
            nn.Linear(12, hidden),
            nn.Mish(),
            nn.Linear(hidden, latent_channels),
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        # [B, L, 2] -> [B, L, 12]
        z = self.theta(angles)
        # [B, L, 12] -> [B, L, C]
        return self.mlp(z)


class SpatialCodebook(nn.Module):
    """
    Unified spatial codebook supporting 5 distinct spatial conditioning modes:
    - 'theta': True physical spherical angles
    - 'permuted_theta': Fixed derangement (arange(12) + 5) % 12
    - 'learned': Trainable 12-D continuous code
    - 'random_fixed': Fixed normalized random 12-D code
    - 'constant': Zero condition (no target viewpoint semantics)
    """

    def __init__(
        self,
        mode: str,
        seed: int = 20260903,
    ):
        super().__init__()
        self.mode = mode

        self.register_buffer(
            "angles",
            STANDARD_ANGLES.clone(),
        )

        # Fixed derangement permutation: no element maps to its original index.
        # (0+5)%12=5, (1+5)%12=6, ..., (11+5)%12=4
        self.register_buffer(
            "permutation",
            (torch.arange(12) + 5) % 12,
        )

        if mode == "learned":
            self.learned = nn.Parameter(
                torch.randn(12, 12) * 0.02
            )
        else:
            self.learned = None

        if mode == "random_fixed":
            gen = torch.Generator()
            gen.manual_seed(seed)
            x = torch.randn(12, 12, generator=gen)
            x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
            self.register_buffer(
                "random_fixed",
                x,
            )
        else:
            self.random_fixed = None

    def forward(
        self,
        batch_size: int,
        theta_encoder: ThetaEncoder,
        target_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns spatial code representation:
          - If target_indices is None: [B, 12, 12] for all 12 leads
          - If target_indices is [B]:  [B, 12] for the specific target lead per sample
        """
        if target_indices is not None:
            assert target_indices.shape[0] == batch_size, f"Shape mismatch: {target_indices.shape[0]} vs {batch_size}"
            if self.mode == "theta":
                selected_angles = self.angles[target_indices]  # [B, 2]
                return theta_encoder(selected_angles[:, None, :]).squeeze(1)  # [B, 12]

            if self.mode == "permuted_theta":
                perm_angles = self.angles[self.permutation]  # [12, 2]
                selected_angles = perm_angles[target_indices]  # [B, 2]
                return theta_encoder(selected_angles[:, None, :]).squeeze(1)  # [B, 12]

            if self.mode == "learned":
                return self.learned[target_indices]  # [B, 12]

            if self.mode == "random_fixed":
                return self.random_fixed[target_indices]  # [B, 12]

            if self.mode == "constant":
                return torch.zeros(batch_size, 12, device=self.angles.device)

            raise ValueError(f"Unknown code mode: {self.mode}")

        # All 12 leads simultaneously
        if self.mode == "theta":
            angles = self.angles[None].expand(batch_size, -1, -1)
            return theta_encoder(angles)

        if self.mode == "permuted_theta":
            angles = self.angles[self.permutation]
            angles = angles[None].expand(batch_size, -1, -1)
            return theta_encoder(angles)

        if self.mode == "learned":
            return self.learned[None].expand(batch_size, -1, -1)

        if self.mode == "random_fixed":
            return self.random_fixed[None].expand(batch_size, -1, -1)

        if self.mode == "constant":
            return torch.zeros(
                batch_size,
                12,
                12,
                device=self.angles.device,
            )

        raise ValueError(f"Unknown code mode: {self.mode}")
