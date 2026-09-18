"""Frozen G4 probe-contract utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


PRIMARY_REPRESENTATIONS = ("state", "phase_standardized", "mean_residual", "innovation")


def probe_features(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or values.shape[1:] != (16, 1024):
        raise ValueError("G4 requires (record,16,1024) Phase-KME tensors")
    return values[:, 1:].flatten(1)


@dataclass(frozen=True)
class ProbeScaler:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, train: torch.Tensor) -> "ProbeScaler":
        precise = train.double()
        return cls(
            precise.mean(0, keepdim=True),
            precise.std(0, keepdim=True, unbiased=False).clamp_min(1e-6),
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return ((values.double() - self.mean) / self.scale).to(values.dtype)


def minibatch_order(count: int, seed: int, epoch: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed * 1000 + epoch)
    return torch.randperm(count, generator=generator, device=device)


def patient_bootstrap_multiplicity(patient_ids: np.ndarray, seed: int, draws: int) -> np.ndarray:
    patients = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(patients), size=(draws, len(patients)))
    multiplicity = np.zeros((draws, len(patients)), dtype=np.int16)
    for draw in range(draws):
        multiplicity[draw] = np.bincount(sampled[draw], minlength=len(patients))
    return multiplicity


def record_weights(patient_ids: np.ndarray, patient_multiplicity: np.ndarray) -> np.ndarray:
    patients, inverse, counts = np.unique(patient_ids, return_inverse=True, return_counts=True)
    if len(patient_multiplicity) != len(patients):
        raise ValueError("one multiplicity required per unique patient")
    return patient_multiplicity[inverse] / counts[inverse]


def valid_multilabel_weights(labels: np.ndarray, weights: np.ndarray) -> bool:
    for column in range(labels.shape[1]):
        if weights[labels[:, column] > 0.5].sum() <= 0 or weights[labels[:, column] <= 0.5].sum() <= 0:
            return False
    return True


def require_confirmatory_mode(mode: str) -> None:
    if mode != "confirmatory":
        raise ValueError("fold-8 runner forbids hyperparameter search")
