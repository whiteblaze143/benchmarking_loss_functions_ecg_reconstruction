from __future__ import annotations

import numpy as np


def normalize_operator(q: np.ndarray) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        raise ValueError("zero measurement operator")
    return value / norm


def operator_waveform(basis_mv: np.ndarray, q: np.ndarray) -> np.ndarray:
    value = normalize_operator(q)
    if basis_mv.ndim != 2 or basis_mv.shape[1] != len(value):
        raise ValueError("basis/operator dimensions disagree")
    return basis_mv @ value


def sample_sparse_operators(count: int, rng: np.random.Generator) -> np.ndarray:
    result = []
    for _ in range(count):
        q = rng.normal(size=8)
        keep = np.argpartition(np.abs(q), -3)[-3:]
        sparse = np.zeros(8)
        sparse[keep] = q[keep]
        result.append(normalize_operator(sparse))
    return np.stack(result)
