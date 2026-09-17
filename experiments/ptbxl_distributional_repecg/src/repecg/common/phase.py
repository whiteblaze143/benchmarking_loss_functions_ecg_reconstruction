from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def phase_normalize(
    basis: np.ndarray,
    intervals: np.ndarray,
    *,
    samples: int = 256,
) -> np.ndarray:
    if basis.ndim != 2:
        raise ValueError("basis must have shape (time,channels)")
    beats = []
    target = np.linspace(0.0, 1.0, samples, endpoint=False)
    for start, stop in np.asarray(intervals, dtype=np.int64):
        cell = basis[start:stop]
        if len(cell) < 2:
            continue
        source = np.linspace(0.0, 1.0, len(cell), endpoint=False)
        beats.append(PchipInterpolator(source, cell, axis=0, extrapolate=True)(target))
    if not beats:
        return np.empty((0, samples, basis.shape[1]), dtype=np.float64)
    return np.stack(beats)


def phase_cells(beats: np.ndarray, groups: int) -> np.ndarray:
    if beats.ndim != 3:
        raise ValueError("beats must have shape (beat,phase,lead)")
    if beats.shape[1] % groups:
        raise ValueError("phase samples must be divisible by groups")
    width = beats.shape[1] // groups
    return beats.reshape(beats.shape[0], groups, width, beats.shape[2])
