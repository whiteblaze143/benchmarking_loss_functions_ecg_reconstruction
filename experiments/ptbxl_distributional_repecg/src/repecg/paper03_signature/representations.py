from __future__ import annotations

import hashlib

import iisignature
import numpy as np
from scipy.interpolate import PchipInterpolator


PATH_SAMPLES = 16
PATH_DIMENSION = 8
SIGNATURE_DEPTH = 3
LOGSIGNATURE_DIMENSION = iisignature.logsiglength(PATH_DIMENSION, SIGNATURE_DEPTH)
DESCRIPTOR_DIMENSION = 3 * PATH_DIMENSION + LOGSIGNATURE_DIMENSION
_PREPARED = iisignature.prepare(PATH_DIMENSION, SIGNATURE_DEPTH)


def _rng(record_id: int, beat: int, phase: int, seed: int, kind: str) -> np.random.Generator:
    token = f"paper03:{seed}:{record_id}:{beat}:{phase}:{kind}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(value)


def transform_path(
    path: np.ndarray,
    *,
    kind: str,
    record_id: int,
    beat: int,
    phase: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    if values.shape != (PATH_SAMPLES, PATH_DIMENSION) or not np.isfinite(values).all():
        raise ValueError(f"expected finite {(PATH_SAMPLES, PATH_DIMENSION)} path")
    if kind == "identity":
        return values.copy()
    if kind == "time_reverse":
        return values[::-1].copy()
    if kind == "order_destroy":
        order = np.arange(PATH_SAMPLES)
        order[1:-1] = _rng(record_id, beat, phase, seed, kind).permutation(order[1:-1])
        return values[order]
    if kind == "monotone_warp_sham":
        rng = _rng(record_id, beat, phase, seed, kind)
        amplitude = (-0.1, 0.1)[int(rng.integers(0, 2))]
        parameter = np.linspace(0.0, 1.0, PATH_SAMPLES)
        warped = parameter + amplitude * np.sin(2.0 * np.pi * parameter) / (2.0 * np.pi)
        return PchipInterpolator(parameter, values, axis=0)(warped)
    raise ValueError(f"unsupported path intervention: {kind}")


def path_descriptor(path: np.ndarray) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    if values.shape != (PATH_SAMPLES, PATH_DIMENSION) or not np.isfinite(values).all():
        raise ValueError(f"expected finite {(PATH_SAMPLES, PATH_DIMENSION)} path")
    descriptor = np.concatenate(
        (values[0], values[-1], values.mean(axis=0), iisignature.logsig(values, _PREPARED))
    )
    if descriptor.shape != (DESCRIPTOR_DIMENSION,) or not np.isfinite(descriptor).all():
        raise ValueError("invalid depth-3 log-signature descriptor")
    return descriptor
