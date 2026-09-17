from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from repecg.common.preprocess import LeadScaler


KINGSTON_SOURCE_LEADS = ("I", "II", "III", "V")
CANONICAL_INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
KINGSTON_OBSERVED_MASK = np.asarray([True, True, False, False, False, False, False, False])


@dataclass(frozen=True)
class ZeroPaddedBasis:
    physical_mv: np.ndarray
    standardized: np.ndarray
    observed_lead_mask: np.ndarray


def zero_padded_basis(
    filtered_signal_mv: np.ndarray,
    source_leads: tuple[str, ...],
    scaler: LeadScaler,
) -> ZeroPaddedBasis:
    """Map Kingston I/II into the frozen eight-lead input and zero-fill the rest.

    Kingston's III is derived and its generic V channel has no V1-V6 location,
    so neither is admitted into the independent spatial basis.
    """
    values = np.asarray(filtered_signal_mv, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(source_leads):
        raise ValueError("filtered signal shape does not match source lead names")
    if len(set(source_leads)) != len(source_leads):
        raise ValueError("source lead names must be unique")
    if not np.isfinite(values).all():
        raise ValueError("filtered signal contains non-finite samples")
    if "I" not in source_leads or "II" not in source_leads:
        raise ValueError("Kingston negative control requires explicitly named I and II")
    if scaler.mean.shape != (8,) or scaler.std.shape != (8,):
        raise ValueError("frozen scaler must have eight independent-lead entries")

    physical = np.zeros((len(values), 8), dtype=np.float64)
    standardized = np.zeros_like(physical)
    for target_index, lead in enumerate(("I", "II")):
        source_index = source_leads.index(lead)
        physical[:, target_index] = values[:, source_index]
        standardized[:, target_index] = (
            values[:, source_index] - scaler.mean[target_index]
        ) / scaler.std[target_index]
    return ZeroPaddedBasis(
        physical_mv=physical,
        standardized=standardized,
        observed_lead_mask=KINGSTON_OBSERVED_MASK.copy(),
    )
