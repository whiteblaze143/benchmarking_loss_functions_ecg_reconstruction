"""Deterministic raw-waveform views and stagewise audits for Paper 10."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from repecg.common.beats import detect_rpeaks
from repecg.common.kernels import NystromMap, WhiteningTransform
from repecg.common.perturbations import add_noise_at_snr, resample_raw_waveform, scale_global_amplitude
from repecg.common.phase import phase_normalize
from repecg.common.preprocess import LeadScaler, bandpass_ecg, independent_basis


@dataclass(frozen=True)
class Environment:
    name: str
    kind: str
    value: float | None = None
    realization: int = 0
    tier: str = "primary"


def environment_bank() -> tuple[Environment, ...]:
    return (
        Environment("clean", "clean"),
        Environment("gain_0p8", "gain", 0.8),
        Environment("gain_1p2", "gain", 1.2),
        Environment("resample_500_250_500", "resample"),
        *(Environment(f"noise_20db_r{r}", "noise", 20.0, r) for r in range(3)),
        *(Environment(f"noise_10db_r{r}", "noise", 10.0, r, "boundary") for r in range(3)),
    )


def view_seed(*, patient_id: int, ecg_id: int, environment: Environment, master_seed: int) -> int:
    token = f"paper10:{master_seed}:{patient_id}:{ecg_id}:{environment.name}:{environment.realization}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)


def fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def apply_environment(
    raw_signal: np.ndarray,
    environment: Environment,
    *,
    patient_id: int,
    ecg_id: int,
    master_seed: int,
) -> tuple[np.ndarray, int | None]:
    """Apply a declared intervention before filtering, scaling, and peak detection."""
    values = np.asarray(raw_signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("raw signal must have shape (time, 12)")
    if environment.kind == "clean":
        return values.copy(), None
    if environment.kind == "gain":
        assert environment.value is not None
        return scale_global_amplitude(values, environment.value), None
    if environment.kind == "resample":
        return resample_raw_waveform(resample_raw_waveform(values, 500, 250), 250, 500), None
    if environment.kind == "noise":
        assert environment.value is not None
        seed = view_seed(patient_id=patient_id, ecg_id=ecg_id, environment=environment, master_seed=master_seed)
        return add_noise_at_snr(values, environment.value, np.random.default_rng(seed)), seed
    raise ValueError(f"unsupported environment kind: {environment.kind}")


def relative_l2(reference: np.ndarray, observed: np.ndarray) -> float:
    if reference.shape != observed.shape:
        raise ValueError("relative L2 requires matching shapes")
    return float(np.linalg.norm(observed - reference) / max(np.linalg.norm(reference), 1e-12))


def snr_db(reference: np.ndarray, observed: np.ndarray) -> float:
    if reference.shape != observed.shape:
        raise ValueError("SNR requires matching shapes")
    return float(20.0 * np.log10(max(np.linalg.norm(reference), 1e-12) / max(np.linalg.norm(observed - reference), 1e-12)))


def phase_kme(beats: np.ndarray, whitening: WhiteningTransform, mapping: NystromMap) -> np.ndarray:
    """Apply the frozen train-only Phase-KME map without refitting it."""
    if beats.ndim != 3 or beats.shape[1:] != (256, 8):
        raise ValueError("beats must have shape (cycle, 256, 8)")
    if len(beats) == 0:
        return np.empty((16, len(mapping.landmarks)), dtype=np.float64)
    atoms = beats.reshape(len(beats), 16, 16, 8).transpose(1, 0, 2, 3).reshape(16, -1, 8)
    return np.stack([mapping.mean(whitening.transform(cell)) for cell in atoms])


def preprocess_view(
    signal: np.ndarray,
    *,
    scaler: LeadScaler,
    whitening: WhiteningTransform,
    mapping: NystromMap,
) -> dict[str, object]:
    filtered = bandpass_ecg(signal)
    basis = independent_basis(filtered)
    detection = detect_rpeaks(basis)
    beats = phase_normalize(scaler.transform(basis), detection.valid_intervals)
    return {
        "filtered": filtered,
        "rpeaks": detection.rpeaks,
        "valid_cycles": int(len(beats)),
        "detector_lead": detection.lead,
        "kme": phase_kme(beats, whitening, mapping),
    }
