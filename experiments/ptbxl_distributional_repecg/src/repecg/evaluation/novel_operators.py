import torch
import numpy as np

def get_ood_operator_bank(seed: int = 2026, dim: int = 8, count: int = 100) -> torch.Tensor:
    """
    E10: Generate 100 fixed, unseen measurement operators q_i \in R^8 with ||q_i|| = 1.
    """
    rng = torch.Generator().manual_seed(seed)
    q_ood = torch.randn(count, dim, generator=rng)
    # Normalize
    q_ood = q_ood / torch.norm(q_ood, p=2, dim=1, keepdim=True)
    return q_ood

def apply_icm_heuristic(
    x: torch.Tensor, 
    heuristic: str, 
    v3_idx: int = 8, 
    v2_idx: int = 7
) -> torch.Tensor:
    """
    E9: Q-unaware models heuristic for ICM.
    ICM = V3 - V2.
    """
    icm_waveform = x[..., v3_idx] - x[..., v2_idx]
    
    x_novel = torch.zeros_like(x)
    if heuristic == "ICM-H0":
        # Place ICM in channel 0 (usually Lead I), zero all others
        x_novel[..., 0] = icm_waveform
    elif heuristic == "ICM-V3":
        # Place ICM in the V3 slot
        x_novel[..., v3_idx] = icm_waveform
    elif heuristic == "ICM-ALL":
        # Copy the same ICM waveform into every input channel
        for i in range(x.shape[-1]):
            x_novel[..., i] = icm_waveform
    else:
        raise ValueError(f"Unknown heuristic {heuristic}")
        
    return x_novel
