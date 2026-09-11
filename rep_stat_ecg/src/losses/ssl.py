"""Self-Supervised Learning Losses for GRAIL-ECG.

Contains mathematically verified, corrected VICReg loss implementation
adhering to PRD §4, §14, and §35 with proper batch-size denominators (B - 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """Corrected VICReg (Variance-Invariance-Covariance Regularization) Loss.

    Properly computes variance and covariance using the batch size B (with B - 1 sample
    correction) rather than dataset size N.
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:

            z_a: [B, D] representations for view A.
            z_b: [B, D] representations for view B.

        Returns:
            dict containing total loss and individual components:
            'loss', 'sim_loss', 'var_loss', 'cov_loss'
        """
        B, D = z_a.shape
        if B < 2:
            raise ValueError(f"VICReg requires batch size >= 2, got {B}")

        # 1. Invariance / Similarity loss (Mean Squared Euclidean Distance)
        sim_loss = F.mse_loss(z_a, z_b)

        # 2. Variance loss: encourages std(z_j) >= gamma along each feature dimension
        std_a = torch.sqrt(z_a.var(dim=0, unbiased=True) + self.epsilon)
        std_b = torch.sqrt(z_b.var(dim=0, unbiased=True) + self.epsilon)
        var_loss = torch.mean(F.relu(self.gamma - std_a)) + torch.mean(F.relu(self.gamma - std_b))

        # 3. Covariance loss: decorrelates feature dimensions (off-diagonals of cov matrix)
        z_a_centered = z_a - z_a.mean(dim=0)
        z_b_centered = z_b - z_b.mean(dim=0)

        cov_a = (z_a_centered.T @ z_a_centered) / (B - 1)  # [D, D]
        cov_b = (z_b_centered.T @ z_b_centered) / (B - 1)  # [D, D]

        # Zero out diagonal
        cov_a_offdiag = cov_a - torch.diag_embed(torch.diagonal(cov_a))
        cov_b_offdiag = cov_b - torch.diag_embed(torch.diagonal(cov_b))

        cov_loss = (cov_a_offdiag.pow(2).sum() / D) + (cov_b_offdiag.pow(2).sum() / D)

        total_loss = (
            self.sim_coeff * sim_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        return {
            "loss": total_loss,
            "sim_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Functional interface for VICReg loss."""
    loss_module = VICRegLoss(
        sim_coeff=sim_coeff,
        var_coeff=std_coeff,
        cov_coeff=cov_coeff,
        gamma=gamma,
        epsilon=epsilon,
    )
    return loss_module(z_a, z_b)

