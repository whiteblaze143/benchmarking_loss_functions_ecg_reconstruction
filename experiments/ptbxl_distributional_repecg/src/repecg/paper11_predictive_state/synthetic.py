"""Frozen synthetic worlds for predictive-state recovery and falsification."""

from __future__ import annotations

import torch


def hmm_parameters(states: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    if states not in {3, 4}:
        raise ValueError("synthetic HMM supports 3 or 4 known states")
    transition = 0.90 * torch.eye(states, device=device) + 0.10 * torch.roll(torch.eye(states, device=device), shifts=1, dims=1)
    centers = torch.tensor(
        [[2.0, -1.0, 0.5, 1.5], [-1.5, 2.0, 1.0, -0.5], [0.5, 0.0, -2.0, 2.0], [-2.0, -1.5, 1.5, 0.5]],
        device=device,
    )[:states]
    return transition, centers, 0.20


def make_hmm_sequences(
    count: int, length: int, seed: int, device: torch.device, states: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    transition, centers, noise_std = hmm_parameters(states, device)
    latent = torch.empty(count, length, dtype=torch.long, device=device)
    latent[:, 0] = torch.randint(0, states, (count,), generator=generator, device=device)
    uniforms = torch.rand(count, length - 1, generator=generator, device=device)
    for step in range(1, length):
        cumulative = transition[latent[:, step - 1]].cumsum(dim=-1)
        latent[:, step] = (uniforms[:, step - 1, None] > cumulative).sum(dim=-1)
    sequence = centers[latent] + noise_std * torch.randn(count, length, 4, generator=generator, device=device)
    return sequence, latent


def hmm_filter_beliefs(sequence: torch.Tensor, states: int) -> torch.Tensor:
    """Exact causal filtering posteriors for the frozen Gaussian-emission HMM."""
    transition, centers, noise_std = hmm_parameters(states, sequence.device)
    log_transition = transition.log()
    log_prior = torch.full(
        (sequence.shape[0], states),
        -torch.log(torch.tensor(float(states), device=sequence.device)),
        device=sequence.device,
    )
    beliefs = []
    for observation in sequence.unbind(dim=1):
        log_emission = -0.5 * ((observation[:, None] - centers[None]) / noise_std).square().sum(dim=-1)
        log_prior = torch.logsumexp(log_prior[:, :, None] + log_transition[None], dim=1) + log_emission
        log_prior = log_prior - torch.logsumexp(log_prior, dim=-1, keepdim=True)
        beliefs.append(log_prior.exp())
    return torch.stack(beliefs, dim=1)


def hmm_next_mean_from_state(latent: torch.Tensor, states: int) -> torch.Tensor:
    transition, centers, _ = hmm_parameters(states, latent.device)
    return transition[latent[:, :-1]] @ centers


def hmm_next_mean_from_belief(beliefs: torch.Tensor, states: int) -> torch.Tensor:
    transition, centers, _ = hmm_parameters(states, beliefs.device)
    return beliefs[:, :-1] @ transition @ centers


def make_iid_sequences(count: int, length: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(count, length, 4, generator=generator, device=device)


def make_linear_state_space_sequences(
    count: int, length: int, seed: int, device: torch.device, latent_dim: int = 4, nuisance_dim: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Observed process with a continuous filtering state and independent nuisance dimensions."""
    if latent_dim != 4:
        raise ValueError("frozen continuous-state control uses latent dimension 4")
    generator = torch.Generator(device=device).manual_seed(seed)
    dynamics = torch.tensor(
        [[0.76, -0.53, 0.00, 0.00], [0.53, 0.76, 0.00, 0.00], [0.00, 0.00, 0.62, 0.60], [0.00, 0.00, -0.60, 0.62]],
        device=device,
    )
    state = torch.empty(count, length, latent_dim, device=device)
    state[:, 0] = torch.randn(count, latent_dim, generator=generator, device=device)
    for step in range(1, length):
        state[:, step] = state[:, step - 1] @ dynamics.T + 0.15 * torch.randn(count, latent_dim, generator=generator, device=device)
    predictive_observation = state + 0.60 * torch.randn(count, length, latent_dim, generator=generator, device=device)
    nuisance = 0.20 * torch.randn(count, length, nuisance_dim, generator=generator, device=device)
    return torch.cat((predictive_observation, nuisance), dim=-1), state


def make_history_required_sequences(count: int, length: int, seed: int, device: torch.device) -> torch.Tensor:
    """An order-sensitive second-order process: X[t+1] is the older of the last two bits."""
    generator = torch.Generator(device=device).manual_seed(seed)
    bits = torch.empty(count, length, dtype=torch.long, device=device)
    bits[:, :2] = torch.randint(0, 2, (count, 2), generator=generator, device=device)
    for step in range(2, length):
        bits[:, step] = bits[:, step - 2]
    centers = torch.tensor([[-1.5, 1.0, -0.5, 0.5], [1.5, -1.0, 0.5, -0.5]], device=device)
    return centers[bits] + 0.05 * torch.randn(count, length, 4, generator=generator, device=device)
