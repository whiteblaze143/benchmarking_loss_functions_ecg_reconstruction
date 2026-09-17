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


def response_atoms(beats_mv: np.ndarray, q: np.ndarray, voltage_scale: float) -> np.ndarray:
    if voltage_scale <= 0 or not np.isfinite(voltage_scale):
        raise ValueError("voltage_scale must be positive and finite")
    values = np.asarray(beats_mv, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (256, 8):
        raise ValueError(f"expected physical beats with shape (beat,256,8), got {values.shape}")
    waveform = values @ normalize_operator(q) / voltage_scale
    derivative = np.gradient(waveform, axis=1)
    return np.stack((waveform, derivative), axis=-1)


def sample_sparse_operators(count: int, rng: np.random.Generator) -> np.ndarray:
    result = []
    for _ in range(count):
        q = rng.normal(size=8)
        keep = np.argpartition(np.abs(q), -3)[-3:]
        sparse = np.zeros(8)
        sparse[keep] = q[keep]
        result.append(normalize_operator(sparse))
    return np.stack(result)


def canonical_operators() -> np.ndarray:
    return np.eye(8, dtype=np.float64)


def derived_limb_operators() -> np.ndarray:
    coefficients = np.asarray(
        [
            [-1.0, 1.0, 0, 0, 0, 0, 0, 0],
            [-0.5, -0.5, 0, 0, 0, 0, 0, 0],
            [1.0, -0.5, 0, 0, 0, 0, 0, 0],
            [-0.5, 1.0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    return np.stack([normalize_operator(value) for value in coefficients])


def interpolation_operators() -> np.ndarray:
    """Nine frozen interior points from lead I to V2 (alpha=0.1,...,0.9)."""
    lead_i = canonical_operators()[0]
    lead_v2 = canonical_operators()[3]
    return np.stack(
        [normalize_operator((1.0 - alpha) * lead_i + alpha * lead_v2) for alpha in np.arange(0.1, 1.0, 0.1)]
    )


def _overlaps_up_to_sign(candidate: np.ndarray, reference: np.ndarray, tolerance: float) -> bool:
    return bool(
        np.any(np.linalg.norm(reference - candidate, axis=1) <= tolerance)
        or np.any(np.linalg.norm(reference + candidate, axis=1) <= tolerance)
    )


def frozen_operator_banks(tolerance: float = 1e-8) -> dict[str, np.ndarray]:
    seen = np.concatenate(
        (canonical_operators(), sample_sparse_operators(100, np.random.default_rng(1701)))
    )
    dense = np.random.default_rng(1702).normal(size=(100, 8))
    dense = np.stack([normalize_operator(value) for value in dense])
    proposed = {
        "derived_limb": derived_limb_operators(),
        "dense": dense,
        "i_to_v2": interpolation_operators(),
    }
    unseen = {
        name: np.stack(
            [value for value in values if not _overlaps_up_to_sign(value, seen, tolerance)]
        )
        for name, values in proposed.items()
    }
    return {"seen": seen, **unseen}


def record_training_operators(record_id: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(record_id)]))
    canonical = canonical_operators()[rng.choice(8, size=4, replace=False)]
    return np.concatenate((canonical, sample_sparse_operators(4, rng)))
