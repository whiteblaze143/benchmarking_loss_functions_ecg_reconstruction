import torch
from .configurations import ECGConfiguration

def apply_missingness(
    x: torch.Tensor, 
    config: ECGConfiguration, 
    lead_names: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies missingness according to the E3/E4 rules.
    x is [batch, features] or [batch, time, features] or similar depending on the model,
    but typically in repECG it's [batch, phase, lead] where lead dimension matches lead_names.
    Returns: (modified_x, mask)
    """
    if config.name == "12lead" and config.missing_mode != "Z0":
        # Full input, mask is all ones
        mask = torch.ones(x.shape[0], x.shape[1], len(lead_names), device=x.device, dtype=x.dtype)
        return x, mask

    # Calculate mask based on observed leads
    # Assumes the last dimension of x corresponds to leads, if x is 3D
    # For now, create a mask tensor
    mask = torch.zeros(x.shape[0], x.shape[1], len(lead_names), device=x.device, dtype=x.dtype)
    
    for i, lead in enumerate(lead_names):
        if lead in config.observed_leads:
            mask[..., i] = 1.0

    if config.missing_mode == "Z0":
        # Zero-fill, no mask concatenated
        # Note: some models don't take mask at all in Z0
        return x * mask, mask
    elif config.missing_mode == "ZM":
        # Zero-fill + mask
        # Models need to handle mask concatenation explicitly
        return x * mask, mask
    elif config.missing_mode == "native":
        # Native omission - handled inside the model architecture
        return x * mask, mask
    else:
        raise ValueError(f"Unknown missing mode {config.missing_mode}")
