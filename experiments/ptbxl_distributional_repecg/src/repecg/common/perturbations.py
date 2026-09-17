from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly


def resample_raw_waveform(signal_mv: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    """Polyphase-antialiased resampling of an unnormalized physical waveform."""
    values = np.asarray(signal_mv, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("signal must have shape (time, leads)")
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("sampling rates must be positive")
    divisor = math.gcd(source_hz, target_hz)
    return resample_poly(values, target_hz // divisor, source_hz // divisor, axis=0)


def add_noise_at_snr(
    signal_mv: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent Gaussian noise with exact realized per-lead SNR."""
    values = np.asarray(signal_mv, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("signal must have shape (time, leads)")
    signal_rms = np.sqrt(np.mean(np.square(values), axis=0))
    noise = rng.normal(size=values.shape)
    noise = noise - noise.mean(axis=0, keepdims=True)
    noise_rms = np.sqrt(np.mean(np.square(noise), axis=0))
    target_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    scaled = noise * np.divide(target_rms, noise_rms, out=np.zeros_like(target_rms), where=noise_rms > 0)
    return values + scaled


def scale_global_amplitude(signal_mv: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0:
        raise ValueError("amplitude factor must be positive")
    return np.asarray(signal_mv, dtype=np.float64) * factor


def jitter_rpeaks(
    peaks: np.ndarray,
    *,
    max_ms: float,
    sampling_hz: int,
    signal_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply independent bounded integer jitter while preserving peak order."""
    values = np.asarray(peaks, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("peaks must be a nonempty vector")
    radius = int(round(max_ms * sampling_hz / 1000.0))
    if radius < 0 or signal_length <= 0:
        raise ValueError("invalid jitter parameters")
    shifted = values + rng.integers(-radius, radius + 1, size=len(values))
    shifted = np.clip(shifted, 0, signal_length - 1)
    if np.any(np.diff(shifted) <= 0):
        raise ValueError("jitter destroyed chronological peak order")
    return shifted


def retain_random_beats(
    beats: np.ndarray,
    *,
    removal_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove an exact rounded fraction of beats and retain chronological order."""
    values = np.asarray(beats)
    if values.ndim < 1 or len(values) == 0:
        raise ValueError("beats must be nonempty")
    if not 0.0 <= removal_fraction < 1.0:
        raise ValueError("removal_fraction must be in [0,1)")
    retain = len(values) - int(round(removal_fraction * len(values)))
    indices = np.sort(rng.choice(len(values), size=retain, replace=False))
    return values[indices], indices
