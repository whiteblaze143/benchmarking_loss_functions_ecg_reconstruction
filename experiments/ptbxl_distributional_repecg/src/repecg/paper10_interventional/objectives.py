"""Losses for the preclinical Paper 10 same-record matching audit."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def paired_matching_loss(
    clean_state: torch.Tensor,
    transformed_state: torch.Tensor,
    *,
    norm_weight: float = 1.0,
) -> torch.Tensor:
    """Match direction and magnitude so cosine agreement cannot hide norm leakage."""
    if clean_state.shape != transformed_state.shape or clean_state.ndim != 2:
        raise ValueError("paired states must have identical (batch, feature) shape")
    direction = (F.normalize(clean_state, dim=-1) - F.normalize(transformed_state, dim=-1)).square().sum(dim=-1)
    log_norm = (
        clean_state.norm(dim=-1).clamp_min(1e-8).log()
        - transformed_state.norm(dim=-1).clamp_min(1e-8).log()
    ).square()
    return (direction + norm_weight * log_norm).mean()


def coral_loss(states: torch.Tensor, environments: torch.Tensor) -> torch.Tensor:
    """Mean-plus-covariance population alignment; intentionally unpaired."""
    if states.ndim != 2 or environments.ndim != 1 or len(states) != len(environments):
        raise ValueError("states must be (batch, feature) with one environment per row")
    unique = torch.unique(environments)
    if len(unique) < 2:
        raise ValueError("CORAL needs at least two environments")
    summaries = []
    for environment in unique:
        subset = states[environments == environment]
        if len(subset) < 2:
            raise ValueError("each environment needs at least two rows")
        centered = subset - subset.mean(dim=0, keepdim=True)
        summaries.append((subset.mean(dim=0), centered.T @ centered / (len(subset) - 1)))
    terms = []
    for i, (mean_i, covariance_i) in enumerate(summaries):
        for mean_j, covariance_j in summaries[i + 1 :]:
            terms.append((mean_i - mean_j).square().mean() + (covariance_i - covariance_j).square().mean())
    return torch.stack(terms).mean()


def irmv1_penalty(logits: torch.Tensor, labels: torch.Tensor, environments: torch.Tensor) -> torch.Tensor:
    """IRMv1 scalar-classifier penalty calculated separately for each environment."""
    if logits.shape != labels.shape or logits.ndim != 2:
        raise ValueError("logits and labels must have equal (batch, class) shape")
    if environments.ndim != 1 or len(environments) != len(logits):
        raise ValueError("environments must have one row per logit")
    scale = torch.ones((), device=logits.device, requires_grad=True)
    penalties = []
    for environment in torch.unique(environments):
        mask = environments == environment
        if not torch.any(mask):
            continue
        risk = F.binary_cross_entropy_with_logits(scale * logits[mask], labels[mask])
        gradient = torch.autograd.grad(risk, scale, create_graph=True)[0]
        penalties.append(gradient.square())
    if len(penalties) < 2:
        raise ValueError("IRMv1 needs at least two nonempty environments")
    return torch.stack(penalties).mean()
