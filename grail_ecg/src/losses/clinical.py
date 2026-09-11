"""Clinical Multi-Task Anchor Loss for GRAIL-ECG.

Computes multi-label Binary Cross-Entropy with logits across the 4 structured slots:
rhythm, conduction, morphology, and stt.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalAnchorLoss(nn.Module):
    """Multi-domain BCEWithLogitsLoss for anchor concepts."""

    def __init__(self, domain_slices: dict[str, slice]):
        """Args:

            domain_slices: dict mapping domain names to slices in the anchor target vector [25].
        """
        super().__init__()
        self.domain_slices = domain_slices

    def forward(
        self,
        domain_logits: dict[str, torch.Tensor],
        target_labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Args:

            domain_logits: dict mapping domain name to logits [B, n_classes_in_domain].
            target_labels: [B, 25] binary multi-hot anchor labels.
            sample_weights: Optional [B] or [B, 25] weights.

        Returns:
            dict containing total clinical loss and per-domain loss.
        """
        total_loss = torch.tensor(0.0, device=target_labels.device)
        losses_by_domain = {}

        for domain, logits in domain_logits.items():
            if domain not in self.domain_slices:
                continue
            sl = self.domain_slices[domain]
            sub_targets = target_labels[:, sl]

            if sample_weights is not None:
                if sample_weights.ndim == 1:
                    w = sample_weights.unsqueeze(1)
                else:
                    w = sample_weights[:, sl]
                loss_d = F.binary_cross_entropy_with_logits(logits, sub_targets, weight=w)
            else:
                loss_d = F.binary_cross_entropy_with_logits(logits, sub_targets)

            losses_by_domain[domain] = loss_d
            total_loss = total_loss + loss_d

        losses_by_domain["loss"] = total_loss
        return losses_by_domain
