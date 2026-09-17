#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repecg.common.kernels import NystromMap, WhiteningTransform, audit_nystrom, biased_mmd2
from repecg.paper07_operator import (
    frozen_operator_banks,
    record_training_operators,
    response_atoms,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible(cache: Path) -> pd.DataFrame:
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("kind") != "production_phase_cache":
        raise ValueError(f"not a production phase cache: {cache}")
    return pd.read_csv(cache / "qc.csv").query("eligible").reset_index(drop=True)


def _load(cache: Path, row: pd.Series, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache / str(row.artifact)) as item:
        standardized = np.asarray(item["beats"], dtype=np.float64)
        labels = np.asarray(item["labels"], dtype=np.float32)
    return standardized * std + mean, labels


def _record_order(frame: pd.DataFrame, seed: int) -> np.ndarray:
    keyed = []
    for index, ecg_id in enumerate(frame.ecg_id):
        token = f"paper07-reservoir:{seed}:{int(ecg_id)}".encode()
        keyed.append((hashlib.sha256(token).digest(), index))
    return np.asarray([index for _, index in sorted(keyed)], dtype=np.int64)


def _physical_reservoir(
    cache: Path, frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray, count: int, seed: int
) -> np.ndarray:
    chunks = []
    observed = 0
    for index in _record_order(frame, seed):
        beats, _ = _load(cache, frame.iloc[index], mean, std)
        values = beats.reshape(-1, 8)
        take = min(len(values), count - observed)
        chunks.append(values[:take])
        observed += take
        if observed == count:
            break
    if observed != count:
        raise ValueError(f"only {observed} physical samples for reservoir of {count}")
    return np.concatenate(chunks)


def _response_reservoir(
    cache: Path,
    frame: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    voltage_scale: float,
    count: int,
    seed: int,
) -> np.ndarray:
    chunks = []
    observed = 0
    for index in _record_order(frame, seed):
        row = frame.iloc[index]
        beats, _ = _load(cache, row, mean, std)
        for q in record_training_operators(int(row.ecg_id), seed):
            values = response_atoms(beats, q, voltage_scale).reshape(-1, 2)
            take = min(len(values), count - observed)
            chunks.append(values[:take])
            observed += take
            if observed == count:
                return np.concatenate(chunks)
    raise ValueError(f"only {observed} response atoms for reservoir of {count}")


def _phase_cells(atoms: np.ndarray) -> list[np.ndarray]:
    cells = atoms.reshape(len(atoms), 16, 16, 2).transpose(1, 0, 2, 3)
    return [cell.reshape(-1, 2) for cell in cells]


def _audit_cells(
    train_cache: Path,
    train: pd.DataFrame,
    validation_cache: Path,
    validation: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    voltage_scale: float,
    seed: int,
) -> list[np.ndarray]:
    cells = []
    for cache, frame in ((train_cache, train), (validation_cache, validation)):
        for index in _record_order(frame, seed)[:32]:
            row = frame.iloc[index]
            beats, _ = _load(cache, row, mean, std)
            q = record_training_operators(int(row.ecg_id), seed)[0]
            cells.extend(_phase_cells(response_atoms(beats, q, voltage_scale)))
    return cells


def _audit_map(
    cells: list[np.ndarray], whitening: WhiteningTransform, mapping: NystromMap, pairs: int, seed: int
) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    whitened = [whitening.transform(cell) for cell in cells]
    exact = np.empty(pairs)
    approximate = np.empty(pairs)
    for pair in range(pairs):
        left, right = rng.choice(len(cells), size=2, replace=False)
        exact[pair] = biased_mmd2(whitened[left], whitened[right])
        approximate[pair] = np.square(
            mapping.mean(whitened[left]) - mapping.mean(whitened[right])
        ).sum()
    return audit_nystrom(exact, approximate)


class TorchResponseMap:
    def __init__(self, whitening: WhiteningTransform, mapping: NystromMap, device: torch.device):
        self.mean = torch.as_tensor(whitening.mean, device=device, dtype=torch.float32)
        self.components = torch.as_tensor(whitening.components, device=device, dtype=torch.float32)
        self.scales = torch.as_tensor(whitening.scales, device=device, dtype=torch.float32)
        self.landmarks = torch.as_tensor(mapping.landmarks, device=device, dtype=torch.float32)
        self.landmark_count = len(mapping.landmarks)
        self.inverse_root = torch.as_tensor(mapping.inverse_root, device=device, dtype=torch.float32)
        self.c2 = mapping.c2
        self.device = device

    def phase_means(self, atoms: np.ndarray) -> np.ndarray:
        cells = atoms.reshape(len(atoms), 16, 16, 2).transpose(1, 0, 2, 3).reshape(16, -1, 2)
        values = torch.as_tensor(cells, device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            white = (values - self.mean) @ self.components * self.scales
            features = torch.rsqrt(
                torch.cdist(white, self.landmarks.expand(16, -1, -1)).square() + self.c2
            ) @ self.inverse_root
            result = features.mean(dim=1)
        return result.cpu().numpy().astype(np.float32)


def _validation_operators(record_id: int, seen: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(record_id), 7]))
    selected = np.concatenate((np.arange(8), rng.choice(np.arange(8, len(seen)), size=8, replace=False)))
    return seen[selected]


def _represent(
    cache: Path,
    frame: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    voltage_scale: float,
    mapping: TorchResponseMap,
    seed: int,
    seen: np.ndarray,
    validation: bool,
) -> dict[str, np.ndarray]:
    operators_per_record = 16 if validation else 8
    responses = np.empty((len(frame), operators_per_record, 16, mapping.landmark_count), dtype=np.float32)
    negative_responses = np.empty_like(responses)
    operators = np.empty((len(frame), operators_per_record, 8), dtype=np.float32)
    labels = np.empty((len(frame), 5), dtype=np.float32)
    for index, row in frame.iterrows():
        beats, labels[index] = _load(cache, row, mean, std)
        current = (
            _validation_operators(int(row.ecg_id), seen, seed)
            if validation else record_training_operators(int(row.ecg_id), seed)
        )
        operators[index] = current
        for operator_index, q in enumerate(current):
            responses[index, operator_index] = mapping.phase_means(
                response_atoms(beats, q, voltage_scale)
            )
            negative_responses[index, operator_index] = mapping.phase_means(
                response_atoms(beats, -q, voltage_scale)
            )
        if (index + 1) % 500 == 0:
            print(json.dumps({"split": cache.name, "represented": index + 1}), flush=True)
    return {
        "operators": operators,
        "responses": responses,
        "negative_responses": negative_responses,
        "labels": labels,
        "ecg_ids": frame.ecg_id.to_numpy(dtype=np.int64),
        "patient_ids": frame.patient_id.to_numpy(dtype=np.int64),
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-reservoir", type=int, default=500_000)
    parser.add_argument("--response-reservoir", type=int, default=500_000)
    parser.add_argument("--audit-pairs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train = _eligible(args.train_cache)
    validation = _eligible(args.validation_cache)
    scaler = json.loads(args.scaler.read_text())
    physical_mean = np.asarray(scaler["mean"], dtype=np.float64)
    physical_std = np.asarray(scaler["std"], dtype=np.float64)
    physical = _physical_reservoir(
        args.train_cache, train, physical_mean, physical_std, args.physical_reservoir, args.seed
    )
    voltage_scale = float(np.std(physical))
    reservoir = _response_reservoir(
        args.train_cache, train, physical_mean, physical_std, voltage_scale,
        args.response_reservoir, args.seed,
    )
    whitening = WhiteningTransform.fit(reservoir)
    whitened = whitening.transform(reservoir)
    cells = _audit_cells(
        args.train_cache, train, args.validation_cache, validation,
        physical_mean, physical_std, voltage_scale, args.seed,
    )
    audits = {}
    mapping = None
    for landmarks in (128, 256):
        mapping = NystromMap.fit(whitened, landmarks=landmarks, seed=args.seed)
        audits[str(landmarks)] = _audit_map(cells, whitening, mapping, args.audit_pairs, args.seed)
        if audits[str(landmarks)]["passed"]:
            break
    assert mapping is not None
    args.output.mkdir(parents=True, exist_ok=True)
    if not audits[str(len(mapping.landmarks))]["passed"]:
        manifest = {
            "kind": "paper07_measurement_operator_response_sets",
            "status": "ineligible_nystrom_fidelity",
            "audits": audits,
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"Paper 7 Nyström fidelity failed: {audits}")
    banks = frozen_operator_banks()
    device_map = TorchResponseMap(whitening, mapping, torch.device(args.device))
    train_output = _represent(
        args.train_cache, train, physical_mean, physical_std, voltage_scale,
        device_map, args.seed, banks["seen"], False,
    )
    validation_output = _represent(
        args.validation_cache, validation, physical_mean, physical_std, voltage_scale,
        device_map, args.seed, banks["seen"], True,
    )
    finite = all(
        np.isfinite(payload[key]).all()
        for payload in (train_output, validation_output)
        for key in ("responses", "negative_responses")
    )
    norms = np.concatenate((train_output["operators"].reshape(-1, 8), validation_output["operators"].reshape(-1, 8)))
    norm_error = float(np.max(np.abs(np.linalg.norm(norms, axis=1) - 1.0)))
    audit = {"passed": bool(finite and norm_error < 1e-6), "finite": bool(finite), "operator_norm_max_abs_error": norm_error}
    if not audit["passed"]:
        raise RuntimeError(f"Paper 7 representation audit failed: {audit}")
    _atomic_npz(
        args.output / "response_fit.npz",
        physical_mean=physical_mean,
        physical_std=physical_std,
        voltage_scale=np.asarray(voltage_scale),
        whitening_mean=whitening.mean,
        whitening_components=whitening.components,
        whitening_scales=whitening.scales,
        landmarks=mapping.landmarks,
        inverse_root=mapping.inverse_root,
        c2=np.asarray(mapping.c2),
    )
    _atomic_npz(args.output / "operator_banks.npz", **banks)
    _atomic_npz(args.output / "representation_train.npz", **train_output)
    _atomic_npz(args.output / "representation_val.npz", **validation_output)
    manifest = {
        "kind": "paper07_measurement_operator_response_sets",
        "status": "complete",
        "folds": {"fit": [1, 2, 3, 4, 5, 6, 7], "selection": [8]},
        "train_records": len(train),
        "validation_records": len(validation),
        "training_pairs_per_record": 8,
        "validation_pairs_per_record": 16,
        "context_size": [1, 6],
        "response_atom": "shared-scale_[x_q,dx_q/ds]",
        "voltage_scale": voltage_scale,
        "landmarks": len(mapping.landmarks),
        "audits": audits,
        "banks": {name: len(values) for name, values in banks.items()},
        "interpolation_alphas": [round(value, 1) for value in np.arange(0.1, 1.0, 0.1)],
        "orientation_pairs": "explicit_F(q)_and_F(-q)_responses",
        "audit": audit,
        "seed": args.seed,
        "source": {
            "train_manifest_sha256": _sha256(args.train_cache / "manifest.json"),
            "validation_manifest_sha256": _sha256(args.validation_cache / "manifest.json"),
            "scaler_sha256": _sha256(args.scaler),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
