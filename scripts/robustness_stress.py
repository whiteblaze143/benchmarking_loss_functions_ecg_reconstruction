"""Deterministic ECG input-degradation conditions for reconstruction stress tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly


@dataclass(frozen=True)
class StressCondition:
    id: str
    source: str
    noise_type: str
    snr_db: float | None = None


def build_conditions(
    snrs: tuple[int, ...] = (24, 12, 6, 0),
    *,
    include_fitbit: bool = False,
) -> list[StressCondition]:
    conditions = [
        StressCondition(f"gaussian_{snr}db", "synthetic", "gaussian", float(snr))
        for snr in snrs
    ]
    conditions.append(StressCondition("baseline_drift", "synthetic", "baseline_drift"))
    for noise_type in ("bw", "em", "ma"):
        conditions.extend(
            StressCondition(f"nstdb_{noise_type}_{snr}db", "NSTDB", noise_type, float(snr))
            for snr in snrs
        )
    if include_fitbit:
        conditions.extend([
            StressCondition("fitbit_20db", "Fitbit", "fitbit_noise", 20.0),
            StressCondition("fitbit_10db", "Fitbit", "fitbit_noise", 10.0),
            StressCondition(
                "fitbit_baseline_wander",
                "Fitbit",
                "fitbit_baseline_wander",
            ),
        ])
    return conditions


def load_nstdb_noise(noise_dir: Path, fs_target: int = 500) -> dict[str, np.ndarray]:
    noises: dict[str, np.ndarray] = {}
    for name in ("bw", "em", "ma"):
        record = wfdb.rdrecord(str(noise_dir / name))
        channels = np.asarray(record.p_signal, dtype=np.float64).T
        channels = resample_poly(channels, fs_target, int(record.fs), axis=-1)
        channels -= channels.mean(axis=-1, keepdims=True)
        channels /= np.maximum(channels.std(axis=-1, keepdims=True), 1e-8)
        noises[name] = channels.astype(np.float32)
    return noises


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fitbit_noise(noise_dir: Path, fs_target: int = 500) -> tuple[dict[str, np.ndarray], dict]:
    """Load wearable artifacts only after provenance and hashes are verified."""
    provenance_path = noise_dir / "PROVENANCE.json"
    if not provenance_path.exists():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text())
    required = {
        "source_dataset", "source_version", "source_records",
        "extraction_method", "sample_rate_hz", "units", "artifacts",
    }
    missing = sorted(required - provenance.keys())
    if missing:
        raise ValueError(f"Fitbit noise provenance is missing fields: {missing}")
    if not provenance["source_records"] or not provenance["extraction_method"]:
        raise ValueError("Fitbit provenance requires source records and extraction method")
    artifacts = provenance["artifacts"]
    expected_keys = {"fitbit_noise", "fitbit_baseline_wander"}
    if set(artifacts) != expected_keys:
        raise ValueError(f"Fitbit artifacts must be exactly {sorted(expected_keys)}")
    source_rate = int(provenance["sample_rate_hz"])
    if source_rate <= 0:
        raise ValueError("Fitbit sample_rate_hz must be positive")
    output: dict[str, np.ndarray] = {}
    for key, metadata in artifacts.items():
        if set(metadata) < {"file", "sha256"}:
            raise ValueError(f"Fitbit artifact {key} requires file and sha256")
        path = noise_dir / metadata["file"]
        if not path.exists() or _sha256(path) != metadata["sha256"]:
            raise ValueError(f"Fitbit artifact hash mismatch: {path}")
        values = np.load(path, allow_pickle=False).astype(np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[-1] < source_rate or not np.isfinite(values).all():
            raise ValueError(f"Invalid Fitbit artifact array {path}: {values.shape}")
        if source_rate != fs_target:
            values = resample_poly(values, fs_target, source_rate, axis=-1)
        values -= values.mean(axis=-1, keepdims=True)
        values /= np.maximum(values.std(axis=-1, keepdims=True), 1e-8)
        output[key] = values.astype(np.float32)
    return output, provenance


def _seed(condition: StressCondition, ecg_id: int, lead: int, base_seed: int) -> int:
    text = f"{condition.id}:{ecg_id}:{lead}:{base_seed}".encode()
    value = 2166136261
    for byte in text:
        value = (value ^ byte) * 16777619 & 0xFFFFFFFF
    return value


def _segment(noise: np.ndarray, length: int, seed: int) -> np.ndarray:
    channel = seed % noise.shape[0]
    source = noise[channel]
    if source.size < length:
        source = np.resize(source, length)
    max_start = max(source.size - length, 0)
    start = 0 if max_start == 0 else np.random.default_rng(seed).integers(0, max_start + 1)
    segment = source[start : start + length].astype(np.float32, copy=True)
    segment -= segment.mean()
    segment /= max(float(segment.std()), 1e-8)
    return segment


def apply_condition(
    clean: torch.Tensor,
    observed: list[int],
    ecg_ids: torch.Tensor,
    condition: StressCondition,
    nstdb_noises: dict[str, np.ndarray],
    base_seed: int = 1337,
    fs: int = 500,
) -> torch.Tensor:
    noisy = clean.clone()
    length = clean.shape[-1]
    for batch_index, ecg_id in enumerate(ecg_ids.tolist()):
        for lead in observed:
            signal = clean[batch_index, lead]
            signal_ac = signal - signal.mean()
            signal_std = signal_ac.std(unbiased=False).clamp_min(1e-8)
            seed = _seed(condition, int(ecg_id), lead, base_seed)
            if condition.noise_type == "gaussian":
                generator = torch.Generator(device=clean.device)
                generator.manual_seed(seed)
                artifact = torch.randn(length, generator=generator, device=clean.device)
            elif condition.noise_type == "baseline_drift":
                phase = (seed % 10000) / 10000.0 * 2.0 * np.pi
                frequency = 0.15 + ((seed // 10000) % 36) / 100.0
                time = torch.arange(length, device=clean.device, dtype=clean.dtype) / fs
                artifact = torch.sin(2.0 * np.pi * frequency * time + phase)
            else:
                artifact = torch.from_numpy(
                    _segment(nstdb_noises[condition.noise_type], length, seed)
                ).to(device=clean.device, dtype=clean.dtype)
            artifact = artifact - artifact.mean()
            artifact_std = artifact.std(unbiased=False).clamp_min(1e-8)
            if condition.snr_db is None:
                scale = 0.25 * signal_std / artifact_std
            else:
                scale = signal_std / (
                    (10.0 ** (condition.snr_db / 20.0)) * artifact_std
                )
            noisy[batch_index, lead] = signal + scale * artifact
    return noisy
