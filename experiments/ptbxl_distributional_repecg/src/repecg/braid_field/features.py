from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class FrozenResponseFeatureMap:
    """Torch implementation of the frozen Paper-07 response map."""

    def __init__(self, artifact: str | Path, *, device: torch.device | str = "cpu"):
        with np.load(artifact) as item:
            self.mean = torch.as_tensor(item["whitening_mean"], dtype=torch.float32, device=device)
            self.components = torch.as_tensor(item["whitening_components"], dtype=torch.float32, device=device)
            self.scales = torch.as_tensor(item["whitening_scales"], dtype=torch.float32, device=device)
            self.landmarks = torch.as_tensor(item["landmarks"], dtype=torch.float32, device=device)
            self.inverse_root = torch.as_tensor(item["inverse_root"], dtype=torch.float32, device=device)
            self.c2 = float(np.asarray(item["c2"]))
            self.voltage_scale = float(np.asarray(item["voltage_scale"]))

    @property
    def response_dim(self) -> int:
        return int(self.landmarks.shape[0])

    def phase_means_from_waveform(self, waveform_mv: torch.Tensor) -> torch.Tensor:
        """Map [B,beats,256] projected waveforms to [B,16,response_dim]."""
        if waveform_mv.ndim != 3 or waveform_mv.shape[-1] != 256:
            raise ValueError("waveform_mv must be [batch,beat,256]")
        derivative = torch.gradient(waveform_mv / self.voltage_scale, dim=-1)[0]
        atoms = torch.stack((waveform_mv / self.voltage_scale, derivative), dim=-1)
        batch, beats = waveform_mv.shape[:2]
        cells = atoms.reshape(batch, beats, 16, 16, 2).permute(0, 2, 1, 3, 4)
        cells = cells.reshape(batch, 16, -1, 2)

        white = (cells - self.mean) @ self.components * self.scales
        flat = white.reshape(-1, white.shape[-2], white.shape[-1])
        kernel = torch.rsqrt(torch.cdist(flat, self.landmarks.unsqueeze(0)).square() + self.c2)
        mapped = kernel @ self.inverse_root
        return mapped.mean(dim=1).reshape(batch, 16, self.response_dim)

    def project_basis(
        self, beats_mv: torch.Tensor, operators: torch.Tensor
    ) -> torch.Tensor:
        """Project physical Q8 beats onto operators and return response features.

        beats_mv: [B,beat,256,8]
        operators: [Q,8] or [B,Q,8]
        returns: [B,Q,16,response_dim]
        """
        if beats_mv.ndim != 4 or beats_mv.shape[-2:] != (256, 8):
            raise ValueError("beats_mv must be [batch,beat,256,8]")
        if operators.ndim == 2:
            operators = operators.unsqueeze(0).expand(len(beats_mv), -1, -1)
        if operators.ndim != 3 or operators.shape[0] != len(beats_mv) or operators.shape[-1] != 8:
            raise ValueError("operators must be [Q,8] or [B,Q,8]")
        operators = operators / operators.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        waveform = torch.einsum("btli,bqi->bqtl", beats_mv, operators)
        b, q, beat, length = waveform.shape
        mapped = self.phase_means_from_waveform(waveform.reshape(b * q, beat, length))
        return mapped.reshape(b, q, 16, self.response_dim)


def phase_scalar_field(beats_mv: torch.Tensor, operators: torch.Tensor, voltage_scale: float) -> torch.Tensor:
    """Ground-truth phase-mean scalar field, normalized by frozen voltage scale."""
    if operators.ndim == 2:
        operators = operators.unsqueeze(0).expand(len(beats_mv), -1, -1)
    operators = operators / operators.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    waveform = torch.einsum("btli,bqi->bqtl", beats_mv, operators) / float(voltage_scale)
    field = waveform.reshape(len(beats_mv), operators.shape[1], beats_mv.shape[1], 16, 16).mean(dim=(2, 4))
    return field.transpose(1, 2)
