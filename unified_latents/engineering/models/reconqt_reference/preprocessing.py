import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from typing import Tuple, Optional
from .theta import LEAD_NAMES


def bandpass_3drecon(
    signal: np.ndarray,
    fs: float = 500.0,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 5,
) -> np.ndarray:
    """
    Fifth-order zero-phase Butterworth bandpass filter (0.5 - 40 Hz)
    as confirmed in Ansari et al. (2026).
    
    Args:
        signal: numpy array of shape [..., T]
        fs: sampling rate (default 500.0 Hz)
        low: lower cutoff frequency in Hz
        high: upper cutoff frequency in Hz
        order: filter order (default 5)
    Returns:
        filtered signal with identical shape
    """
    sos = butter(
        order,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(
        sos,
        signal,
        axis=-1,
    )


def extract_source_vector(
    target_12lead: torch.Tensor,
    source_mode: str = "lead_I",
) -> torch.Tensor:
    """
    Extracts or computes the single-lead source vector.
    
    Modes:
      - 'lead_I': Observed Lead I (index 0)
      - 'v3_minus_v2': Differential precordial vector (V3 - V2),
                       the subcutaneous ICM-like development signal from Ansari et al.
    """
    if source_mode == "lead_I":
        idx_I = LEAD_NAMES.index("I")
        return target_12lead[:, idx_I : idx_I + 1, :].clone()

    elif source_mode == "v3_minus_v2":
        idx_v2 = LEAD_NAMES.index("V2")
        idx_v3 = LEAD_NAMES.index("V3")
        v2 = target_12lead[:, idx_v2 : idx_v2 + 1, :]
        v3 = target_12lead[:, idx_v3 : idx_v3 + 1, :]
        diff = v3 - v2
        return diff

    else:
        raise ValueError(f"Unknown source mode: {source_mode}")


def normalize_record(
    source: torch.Tensor,
    target: torch.Tensor,
    mode: str = "strict_deployable",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Explicit Normalization Protocol.
    
    Modes:
      1. 'strict_deployable':
         Source statistics (mean_s, std_s) computed strictly from the single observed
         source channel. Zero target information is leaked. Target is standardized by
         the source statistics so relative amplitudes are preserved.
      2. 'per_channel_record':
         Every individual lead is standardized to zero mean and unit variance independently.
      3. 'shared_12lead_record':
         One global mean and std computed across all 12 simultaneous leads.
         TAGGED: NONDEPLOYABLE_FOR_SINGLE_LEAD.
    """
    metadata = {"normalization_mode": mode}

    if mode == "strict_deployable":
        # Source is [B, 1, T]
        mean_s = source.mean(dim=-1, keepdim=True)
        std_s = source.std(dim=-1, keepdim=True).clamp_min(eps)

        norm_source = (source - mean_s) / std_s
        norm_target = (target - mean_s) / std_s

        metadata["mean"] = mean_s
        metadata["std"] = std_s
        metadata["deployable"] = True
        return norm_source, norm_target, metadata

    elif mode == "per_channel_record":
        mean_s = source.mean(dim=-1, keepdim=True)
        std_s = source.std(dim=-1, keepdim=True).clamp_min(eps)
        norm_source = (source - mean_s) / std_s

        mean_t = target.mean(dim=-1, keepdim=True)
        std_t = target.std(dim=-1, keepdim=True).clamp_min(eps)
        norm_target = (target - mean_t) / std_t

        metadata["mean_s"] = mean_s
        metadata["std_s"] = std_s
        metadata["mean_t"] = mean_t
        metadata["std_t"] = std_t
        metadata["deployable"] = False
        return norm_source, norm_target, metadata

    elif mode == "shared_12lead_record":
        # Global stats across leads and time
        mean_global = target.mean(dim=(-2, -1), keepdim=True)
        std_global = target.std(dim=(-2, -1), keepdim=True).clamp_min(eps)

        norm_source = (source - mean_global) / std_global
        norm_target = (target - mean_global) / std_global

        metadata["mean_global"] = mean_global
        metadata["std_global"] = std_global
        metadata["deployable"] = False
        metadata["warning"] = "NONDEPLOYABLE_FOR_SINGLE_LEAD"
        return norm_source, norm_target, metadata

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
