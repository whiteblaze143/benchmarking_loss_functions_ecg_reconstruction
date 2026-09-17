from __future__ import annotations

import numpy as np


def logsignature_descriptor(cell: np.ndarray, *, depth: int = 3) -> np.ndarray:
    try:
        import iisignature
    except ImportError as exc:
        raise RuntimeError("Paper 3 requires iisignature; install the frozen environment") from exc
    values = np.asarray(cell, dtype=np.float64)
    if values.shape != (16, 8):
        raise ValueError(f"expected a (16,8) beat-cell, received {values.shape}")
    preparation = iisignature.prepare(8, depth)
    logsignature = np.asarray(iisignature.logsig(values, preparation), dtype=np.float64)
    return np.concatenate((values[0], values[-1], values.mean(axis=0), logsignature))


def order_destroy(cell: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(cell).copy()
    values[1:-1] = values[1:-1][rng.permutation(14)]
    return values
