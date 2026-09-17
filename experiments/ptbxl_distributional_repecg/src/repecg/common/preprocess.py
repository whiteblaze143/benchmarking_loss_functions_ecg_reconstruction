from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .ptbxl import CANONICAL_LEADS, INDEPENDENT_LEADS


INDEPENDENT_INDEX = np.asarray([CANONICAL_LEADS.index(x) for x in INDEPENDENT_LEADS])


def bandpass_ecg(
    signal: np.ndarray,
    *,
    fs: int = 500,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(f"expected (time,12), received {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains non-finite samples")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=0)


@dataclass(frozen=True)
class LeadScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, basis_mv: np.ndarray) -> np.ndarray:
        return (basis_mv - self.mean) / self.std


@dataclass
class LeadScalerAccumulator:
    """Numerically stable streaming per-lead moments for full PTB-XL."""

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def merge(self, count: int, mean: np.ndarray, m2: np.ndarray) -> None:
        if count < 0:
            raise ValueError("count must be nonnegative")
        if count == 0:
            return
        batch_mean = np.asarray(mean, dtype=np.float64)
        batch_m2 = np.asarray(m2, dtype=np.float64)
        if batch_mean.shape != (8,) or batch_m2.shape != (8,):
            raise ValueError("moments must have shape (8,)")
        if self.count == 0:
            self.count = count
            self.mean = batch_mean.copy()
            self.m2 = batch_m2.copy()
            return
        assert self.mean is not None and self.m2 is not None
        total = self.count + count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * count / total
        self.m2 = self.m2 + batch_m2 + np.square(delta) * self.count * count / total
        self.count = total

    def update(self, basis_mv: np.ndarray) -> None:
        values = np.asarray(basis_mv, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 8:
            raise ValueError("signal must have shape (time,8)")
        if not np.isfinite(values).all():
            raise ValueError("signal contains non-finite samples")
        batch_count = len(values)
        if batch_count == 0:
            return
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.square(centered).sum(axis=0)
        self.merge(batch_count, batch_mean, batch_m2)

    def finalize(self) -> LeadScaler:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("cannot finalize an empty scaler")
        std = np.sqrt(self.m2 / self.count)
        std[std < 1e-8] = 1.0
        return LeadScaler(mean=self.mean.copy(), std=std)


def fit_lead_scaler(signals: list[np.ndarray]) -> LeadScaler:
    if not signals:
        raise ValueError("at least one training signal is required")
    stacked = np.concatenate(signals, axis=0)
    if stacked.ndim != 2 or stacked.shape[1] != 8:
        raise ValueError("training signals must have shape (time,8)")
    mean = stacked.mean(axis=0, dtype=np.float64)
    std = stacked.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    return LeadScaler(mean=mean, std=std)


def independent_basis(filtered_canonical_mv: np.ndarray) -> np.ndarray:
    return filtered_canonical_mv[:, INDEPENDENT_INDEX].copy()
