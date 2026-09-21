from __future__ import annotations

import torch
from torch.nn import functional as F


def braid_invariance_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Cosine distance between same-record braid embeddings under two views."""
    if left.shape != right.shape:
        raise ValueError("paired embeddings must have the same shape")
    return (1.0 - F.cosine_similarity(left, right, dim=-1)).mean()


def embedding_variance_floor_loss(
    embedding: torch.Tensor, *, target_std: float = 0.5, eps: float = 1e-4
) -> torch.Tensor:
    """VICReg-style anti-collapse penalty applied across records in a batch."""
    if embedding.ndim != 2:
        raise ValueError("embedding must be [batch, feature]")
    std = torch.sqrt(embedding.var(dim=0, unbiased=False) + eps)
    return torch.relu(target_std - std).mean()


def symmetric_bernoulli_kl(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL for independent Bernoulli outputs."""
    left = torch.sigmoid(left_logits).clamp(1e-6, 1.0 - 1e-6)
    right = torch.sigmoid(right_logits).clamp(1e-6, 1.0 - 1e-6)
    lr = left * (left.log() - right.log()) + (1.0 - left) * (
        (1.0 - left).log() - (1.0 - right).log()
    )
    rl = right * (right.log() - left.log()) + (1.0 - right) * (
        (1.0 - right).log() - (1.0 - left).log()
    )
    return 0.5 * (lr + rl).mean()


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL[q(z|x)||N(0,I)] for diagonal Gaussian q."""
    return -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()
