import torch
import numpy as np

def get_semantic_permutations(num_leads: int = 12, seed: int = 2026, count: int = 5) -> list[torch.Tensor]:
    """
    E11: Return `count` fixed random permutations of length `num_leads`.
    """
    rng = torch.Generator().manual_seed(seed)
    perms = []
    for _ in range(count):
        # generate random permutation
        perm = torch.randperm(num_leads, generator=rng)
        perms.append(perm)
    return perms

def apply_semantic_randomization(
    x: torch.Tensor,
    permutation: torch.Tensor,
    model_type: str = "fixed_channel"
) -> torch.Tensor:
    """
    E11: Apply semantic wiring ablation.
    
    Args:
        x: Input tensor [B, ..., num_leads]
        permutation: Tensor of indices to permute
        model_type: The way the permutation is applied.
            - fixed_channel: Permute lead identities in the input tensor.
            - operator: (Assumes Q is provided separately, this is a placeholder if Q is embedded in x)
    """
    if model_type == "fixed_channel" or model_type == "conditional_residual" or model_type == "tokens":
        # Permute the last dimension (leads/features)
        # Note: If x is [B, phases, leads], we index the leads dimension
        return x[..., permutation]
    elif model_type == "operator":
        # For operator models, the operator Q is permuted, but the waveform remains fixed.
        # It's usually better to permute Q explicitly outside of this function,
        # but if Q is the input, we permute it here.
        return x[..., permutation]
    else:
        raise ValueError(f"Unknown model_type {model_type}")
