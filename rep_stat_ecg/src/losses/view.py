"""Any-Pairs Auxiliary View Loss for GRAIL-ECG.

Computes L1 reconstruction loss between reconstructed held-out view and true lead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewReconstructionLoss(nn.Module):
    """L1 Loss on held-out target view."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_waveform: torch.Tensor, target_waveform: torch.Tensor) -> torch.Tensor:
        """Args:

            pred_waveform: [B, 1, 5000]
            target_waveform: [B, 1, 5000] or [B, 5000]

        Returns:
            Scalar L1 loss.
        """
        if target_waveform.ndim == 2:
            target_waveform = target_waveform.unsqueeze(1)
        return F.l1_loss(pred_waveform, target_waveform)
