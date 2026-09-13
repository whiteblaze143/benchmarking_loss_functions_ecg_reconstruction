"""Construct the frozen eight-view Theta-repSpat representation."""

from __future__ import annotations

import numpy as np
import torch


CANONICAL_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
INPUT_INDICES = (0, 1, 4)
PREDICTED_INDICES = (2, 3, 5, 6, 7)

# Exact PTB-XL angles from the authors' dataset/PTBXL.py, in the same lead order.
CANONICAL_ANGLES_RAD = np.asarray(
    [
        [np.pi / 2, np.pi / 2],
        [np.pi * 5 / 6, np.pi / 2],
        [np.pi / 2, -np.pi / 18],
        [np.pi / 2, np.pi / 18],
        [np.pi * 19 / 36, np.pi / 12],
        [np.pi * 11 / 20, np.pi / 6],
        [np.pi * 16 / 30, np.pi / 3],
        [np.pi * 16 / 30, np.pi / 2],
    ],
    dtype=np.float32,
)


def global_minmax_normalize(ecg: np.ndarray) -> np.ndarray:
    """Apply the authors' per-record global min-max normalization exactly."""
    values = np.asarray(ecg, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(CANONICAL_LEADS):
        raise ValueError("ecg must have shape [8, time] in I,II,V1,...,V6 order")
    lo = float(values.min())
    hi = float(values.max())
    if not hi > lo:
        raise ValueError("record-wide min-max normalization is undefined for a constant ECG")
    return (values - lo) / (hi - lo)


@torch.inference_mode()
def build_canonical_panorama(model: torch.nn.Module, input_ecg: torch.Tensor) -> torch.Tensor:
    """Complete [I, II, V3] into [I, II, V1, ..., V6].

    Args:
        model: Frozen author Nef-Net v2 model.
        input_ecg: Already normalized tensor with shape [batch, 3, time], ordered
            [I, II, V3].

    Returns:
        Tensor of shape [batch, 8, time]. Observed leads are copied exactly;
        only V1, V2, V4, V5, and V6 are predicted.
    """
    if input_ecg.ndim != 3 or input_ecg.shape[1] != 3:
        raise ValueError("input_ecg must have shape [batch, 3, time]")
    if input_ecg.shape[-1] % 4:
        raise ValueError("Nef-Net v2 requires a time length divisible by four")
    if model.training:
        raise ValueError("model must be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("all Nef-Net parameters must be frozen")

    batch, _, time = input_ecg.shape
    device = input_ecg.device
    dtype = input_ecg.dtype
    angles = torch.as_tensor(CANONICAL_ANGLES_RAD, device=device, dtype=dtype)
    input_angles = angles[list(INPUT_INDICES)].unsqueeze(0).expand(batch, -1, -1)

    panorama = input_ecg.new_empty((batch, len(CANONICAL_LEADS), time))
    panorama[:, 0] = input_ecg[:, 0]
    panorama[:, 1] = input_ecg[:, 1]
    panorama[:, 4] = input_ecg[:, 2]

    for output_index in PREDICTED_INDICES:
        query = angles[output_index].unsqueeze(0).expand(batch, -1)
        prediction = model(input_ecg, input_angles, query)
        if prediction.shape != (batch, 1, time):
            raise ValueError(
                f"Nef-Net output must have shape {(batch, 1, time)}, got {tuple(prediction.shape)}"
            )
        panorama[:, output_index] = prediction[:, 0]
    return panorama
