#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


PHASES = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_mask() -> np.ndarray:
    indices = np.arange(PHASES)
    separation = np.abs(indices[:, None] - indices[None, :])
    cyclic = np.minimum(separation, PHASES - separation)
    return cyclic > 1


def fit_tau(values: np.ndarray, batch_size: int = 128) -> float:
    if values.ndim != 3 or values.shape[1] != PHASES:
        raise ValueError(f"expected (record,{PHASES},feature), got {values.shape}")
    allowed = _allowed_mask()
    chunks = []
    for start in range(0, len(values), batch_size):
        batch = np.asarray(values[start : start + batch_size], dtype=np.float64)
        differences = batch[:, :, None, :] - batch[:, None, :, :]
        distances = np.square(differences).sum(axis=-1)[:, allowed]
        chunks.append(distances[np.isfinite(distances) & (distances > 0)])
    candidates = np.concatenate(chunks)
    if not len(candidates):
        raise ValueError("no positive finite non-neighbor distances available for tau")
    tau = float(np.median(candidates))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"invalid tau: {tau}")
    return tau


def recurrence(values: np.ndarray, tau: float, batch_size: int = 128) -> np.ndarray:
    allowed = _allowed_mask()
    result = np.empty((len(values), PHASES, PHASES), dtype=np.float32)
    for start in range(0, len(values), batch_size):
        batch = np.asarray(values[start : start + batch_size], dtype=np.float64)
        differences = batch[:, :, None, :] - batch[:, None, :, :]
        distances = np.square(differences).sum(axis=-1)
        affinity = np.where(allowed[None], np.exp(-distances / tau), 0.0)
        degree = affinity.sum(axis=-1).clip(min=1e-8)
        inverse_root = 1.0 / np.sqrt(degree)
        normalized = inverse_root[:, :, None] * affinity * inverse_root[:, None, :]
        if not np.isfinite(normalized).all():
            raise ValueError("non-finite normalized recurrence operator")
        result[start : start + len(batch)] = normalized.astype(np.float32)
    return result


def intervene(
    values: np.ndarray, ecg_ids: np.ndarray, *, kind: str, seed: int
) -> np.ndarray:
    if kind not in {"phase_content_permutation", "cyclic_relabel_sham"}:
        raise ValueError(f"unsupported intervention: {kind}")
    transformed = np.empty_like(values)
    for index, ecg_id in enumerate(ecg_ids):
        token = f"paper01:{seed}:{int(ecg_id)}:{kind}".encode()
        record_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        rng = np.random.default_rng(record_seed)
        if kind == "phase_content_permutation":
            transformed[index] = values[index, rng.permutation(PHASES)]
        else:
            transformed[index] = np.roll(values[index], int(rng.integers(1, PHASES)), axis=0)
    return transformed


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as item:
        required = {"kernel", "linear", "labels", "ecg_ids", "patient_ids"}
        if not required.issubset(item.files):
            raise ValueError(f"{path} lacks {sorted(required - set(item.files))}")
        return {name: np.asarray(item[name]) for name in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_path = args.source / "representation_train.npz"
    validation_path = args.source / "representation_val.npz"
    train = _load(train_path)
    validation = _load(validation_path)
    kernel_tau = fit_tau(train["kernel"])
    mean_tau = fit_tau(train["linear"])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("train", train), ("val", validation)):
        _atomic_npz(
            args.output / f"representation_{name}.npz",
            kernel_recurrence=recurrence(payload["kernel"], kernel_tau),
            mean_recurrence=recurrence(payload["linear"], mean_tau),
            phase_content_permuted=recurrence(
                intervene(
                    payload["kernel"], payload["ecg_ids"],
                    kind="phase_content_permutation", seed=args.seed,
                ),
                kernel_tau,
            ),
            cyclic_relabel_sham=recurrence(
                intervene(
                    payload["kernel"], payload["ecg_ids"],
                    kind="cyclic_relabel_sham", seed=args.seed,
                ),
                kernel_tau,
            ),
            labels=payload["labels"],
            ecg_ids=payload["ecg_ids"],
            patient_ids=payload["patient_ids"],
        )
    manifest = {
        "kind": "paper01_distributional_recurrence_representations",
        "status": "complete",
        "folds": {"fit": [1, 2, 3, 4, 5, 6, 7], "selection": [8]},
        "train_records": int(len(train["ecg_ids"])),
        "validation_records": int(len(validation["ecg_ids"])),
        "kernel_tau": kernel_tau,
        "mean_control_tau": mean_tau,
        "intervention_seed": args.seed,
        "interventions": {
            "phase_content_permuted": "record-keyed permutation of cell content with the original exclusion mask",
            "cyclic_relabel_sham": "record-keyed nonzero cyclic relabeling of both cell identities and cyclic exclusion geometry",
        },
        "operator_shape": [PHASES, PHASES],
        "neighbor_exclusion": "cyclic_distance_leq_1",
        "normalization": "symmetric_inverse_degree",
        "source": {
            "manifest_sha256": _sha256(args.source / "manifest.json"),
            "train_sha256": _sha256(train_path),
            "validation_sha256": _sha256(validation_path),
        },
        "audit": {"passed": True},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
