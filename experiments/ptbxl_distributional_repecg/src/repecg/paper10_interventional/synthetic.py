"""Frozen synthetic worlds for Paper 10's identifiability gate."""

from __future__ import annotations

import torch


def make_world(world: str, split: str, count: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Generate one predeclared world; no fitted parameters or data-dependent choices."""
    if world not in {"A", "B", "C", "D"}:
        raise ValueError("world must be one of A, B, C, D")
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    generator = torch.Generator().manual_seed(seed + (0 if split == "train" else 10_000))
    signal = torch.randn(count, 4, generator=generator)
    labels = (signal[:, 0] > 0).to(torch.long)
    if world == "A":
        environment = torch.randint(0, 2, (count,), generator=generator)
    elif world == "B":
        environment = labels.clone() if split == "train" else torch.randint(0, 2, (count,), generator=generator)
    elif world == "C":
        environment = torch.randint(0, 2, (count,), generator=generator)
        labels = environment.clone()
    else:
        environment = torch.randint(0, 2, (count,), generator=generator)
    environment_feature = (2.0 * environment.float() - 1.0).unsqueeze(1)
    clean = torch.cat([signal, torch.zeros_like(environment_feature)], dim=1)
    transformed = torch.cat([signal, environment_feature], dim=1)
    phase_noise = 0.02 * torch.randn(count, 16, 5, generator=generator)
    transformed_phase = transformed.unsqueeze(1).expand(-1, 16, -1) + phase_noise
    if world == "D":
        transformed_phase[:, :, :4] = 0.0
    return {
        "clean": clean.unsqueeze(1).expand(-1, 16, -1) + phase_noise,
        "transformed": transformed_phase,
        "environment": environment,
        "labels": labels,
        "record_id": torch.arange(count),
    }
