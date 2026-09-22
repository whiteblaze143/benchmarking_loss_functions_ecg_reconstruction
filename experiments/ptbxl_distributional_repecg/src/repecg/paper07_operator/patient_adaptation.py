"""Bounded deployment-time adaptation for an observed lead and patient context."""

from __future__ import annotations

import torch
from torch import nn

from .model import Full12LeadWaveformDecoder, OperatorSetModel


def configure_freeze_thaw(
    model: OperatorSetModel, decoder: Full12LeadWaveformDecoder, *, mode: str
) -> list[nn.Parameter]:
    """Freeze diagnosis weights; optionally thaw only context and reconstruction.

    ``rapid_reconstruction_adaptation`` is self-supervised: callers may use an
    observed-lead waveform loss only. It never exposes task labels or withheld
    leads, and the diagnostic head is always frozen.
    """
    if mode not in {"frozen_inference", "rapid_reconstruction_adaptation"}:
        raise ValueError("mode must be frozen_inference or rapid_reconstruction_adaptation")
    for parameter in list(model.parameters()) + list(decoder.parameters()):
        parameter.requires_grad_(False)
    if mode == "frozen_inference":
        return []
    if model.patient_context is None:
        raise ValueError("rapid adaptation requires a patient-context model")
    for parameter in model.patient_context.parameters():
        parameter.requires_grad_(True)
    for parameter in decoder.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in list(model.patient_context.parameters()) + list(decoder.parameters()) if parameter.requires_grad]
    if any(parameter.requires_grad for parameter in model.head.parameters()):
        raise AssertionError("diagnostic head must remain frozen during rapid adaptation")
    return trainable


def observed_lead_mse(
    reconstruction: torch.Tensor, observed_waveforms: torch.Tensor, lead_indices: torch.Tensor
) -> torch.Tensor:
    """Self-supervised loss restricted to the actually observed lead(s)."""
    if reconstruction.ndim != 3 or observed_waveforms.ndim != 3 or lead_indices.ndim != 1:
        raise ValueError("expected reconstruction [batch,12,time], observations [batch,lead,time], and lead indices")
    if reconstruction.shape[0] != observed_waveforms.shape[0] or reconstruction.shape[2] != observed_waveforms.shape[2]:
        raise ValueError("reconstruction and observed waveform batch/time dimensions must align")
    if observed_waveforms.shape[1] != len(lead_indices) or len(lead_indices) < 1:
        raise ValueError("each observed waveform needs one valid lead index")
    if torch.any(lead_indices < 0) or torch.any(lead_indices >= reconstruction.shape[1]):
        raise ValueError("observed lead index is out of range")
    return nn.functional.mse_loss(reconstruction[:, lead_indices], observed_waveforms)
