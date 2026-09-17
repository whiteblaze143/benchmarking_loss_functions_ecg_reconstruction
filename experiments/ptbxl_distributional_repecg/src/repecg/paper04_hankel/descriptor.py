from __future__ import annotations

import hashlib

import numpy as np
from scipy.interpolate import PchipInterpolator


DESCRIPTOR_DIMENSION = 30


def hankel_descriptor(cell: np.ndarray, *, delay: int = 6, max_rank: int = 8) -> np.ndarray:
    values = np.asarray(cell, dtype=np.float64)
    if values.shape != (32, 8):
        raise ValueError(f"expected a (32,8) beat-cell, received {values.shape}")
    states = np.stack([values[i : i + delay].reshape(-1) for i in range(32 - delay + 1)])
    x = states[:-1].T
    y = states[1:].T
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    supported = int(np.sum(singular / max(singular[0], 1e-12) >= 1e-6))
    rank = min(max_rank, supported)
    if rank == 0:
        raise ValueError("Hankel matrix has zero supported rank")
    ur = u[:, :rank]
    sr = np.maximum(singular[:rank], 1e-8)
    vr = vt[:rank].T
    operator = ur.T @ y @ vr @ np.diag(1.0 / sr)
    eigenvalues = np.linalg.eigvals(operator)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    padded = np.zeros(max_rank, dtype=np.complex128)
    padded[:rank] = eigenvalues[:rank]
    normalized = np.zeros(max_rank)
    normalized[: min(max_rank, len(singular))] = singular[:max_rank] / max(singular.sum(), 1e-12)
    probability = singular / max(singular.sum(), 1e-12)
    effective_rank = np.exp(-(probability * np.log(np.maximum(probability, 1e-12))).sum())
    cumulative = np.cumsum(singular**2) / max(np.square(singular).sum(), 1e-12)
    rank95 = float(np.searchsorted(cumulative, 0.95) + 1)
    reconstructed = ur @ operator @ np.diag(sr) @ vr.T
    residual = np.linalg.norm(y - reconstructed) / max(np.linalg.norm(y), 1e-12)
    descriptor = np.concatenate(
        (
            normalized,
            [effective_rank, rank95, cumulative[0], cumulative[min(2, len(cumulative) - 1)]],
            np.abs(padded),
            np.angle(padded),
            [residual, np.max(np.abs(padded))],
        )
    )
    if not np.isfinite(descriptor).all():
        raise ValueError("non-finite Hankel descriptor")
    if descriptor.shape != (DESCRIPTOR_DIMENSION,):
        raise ValueError(f"unexpected Hankel descriptor shape: {descriptor.shape}")
    return descriptor


def transform_cell(
    cell: np.ndarray,
    *,
    kind: str,
    record_id: int,
    beat: int,
    phase: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(cell, dtype=np.float64)
    if values.shape != (32, 8) or not np.isfinite(values).all():
        raise ValueError("expected a finite (32,8) beat-cell")
    if kind == "identity":
        return values.copy()
    token = f"paper04:{seed}:{record_id}:{beat}:{phase}:{kind}".encode()
    record_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    rng = np.random.default_rng(record_seed)
    if kind == "time_shuffle":
        order = np.arange(32)
        order[1:-1] = rng.permutation(order[1:-1])
        return values[order]
    if kind == "monotone_warp_sham":
        amplitude = (-0.1, 0.1)[int(rng.integers(0, 2))]
        parameter = np.linspace(0.0, 1.0, 32)
        warped = parameter + amplitude * np.sin(2.0 * np.pi * parameter) / (2.0 * np.pi)
        return PchipInterpolator(parameter, values, axis=0)(warped)
    raise ValueError(f"unsupported Paper 4 intervention: {kind}")
