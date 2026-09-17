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
from sklearn.cluster import MiniBatchKMeans

from repecg.common.kernels import NystromMap, WhiteningTransform, audit_nystrom, biased_mmd2
from repecg.paper06_conditional import conditional_distance, decompose_macro_residual, soft_membership


PHASES = 16
MACRO_STATES = 16


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
    if standardized.ndim != 3 or standardized.shape[1:] != (256, 8):
        raise ValueError(f"invalid Paper 6 beats for ECG {row.ecg_id}: {standardized.shape}")
    return standardized * std + mean, labels


def _record_order(frame: pd.DataFrame, seed: int) -> np.ndarray:
    keyed = []
    for index, ecg_id in enumerate(frame.ecg_id):
        token = f"paper06-reservoir:{seed}:{int(ecg_id)}".encode()
        keyed.append((hashlib.sha256(token).digest(), index))
    return np.asarray([index for _, index in sorted(keyed)], dtype=np.int64)


def _reservoir(
    cache: Path,
    frame: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
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
        raise ValueError(f"only {observed} physical samples available for reservoir of {count}")
    return np.concatenate(chunks)


def _fit_decomposition(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / (len(values) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:3]]
    return mean, basis


def _median_nearest_squared(values: np.ndarray, anchors: np.ndarray) -> float:
    nearest = ((values[:, None] - anchors[None]) ** 2).sum(axis=-1).min(axis=1)
    result = float(np.median(nearest))
    if not np.isfinite(result) or result <= 0:
        raise ValueError("invalid train-only macrostate bandwidth")
    return result


def _phase_cells(values: np.ndarray) -> list[np.ndarray]:
    cells = values.reshape(len(values), PHASES, 16, values.shape[-1]).transpose(1, 0, 2, 3)
    return [cell.reshape(-1, values.shape[-1]) for cell in cells]


def _audit_cells(
    train_cache: Path,
    train: pd.DataFrame,
    validation_cache: Path,
    validation: pd.DataFrame,
    physical_mean: np.ndarray,
    physical_std: np.ndarray,
    decomposition_mean: np.ndarray,
    basis: np.ndarray,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    macro_cells: list[np.ndarray] = []
    residual_cells: list[np.ndarray] = []
    for cache, frame in ((train_cache, train), (validation_cache, validation)):
        for index in _record_order(frame, seed)[:32]:
            beats, _ = _load(cache, frame.iloc[index], physical_mean, physical_std)
            macro, residual = decompose_macro_residual(
                beats.reshape(-1, 8), decomposition_mean, basis
            )
            macro_cells.extend(_phase_cells(macro.reshape(beats.shape[0], 256, 3)))
            residual_cells.extend(_phase_cells(residual.reshape(beats.shape)))
    return macro_cells, residual_cells


def _audit_map(
    cells: list[np.ndarray], whitening: WhiteningTransform, mapping: NystromMap, pairs: int, seed: int
) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    exact = np.empty(pairs)
    approximate = np.empty(pairs)
    whitened = [whitening.transform(cell) for cell in cells]
    for pair in range(pairs):
        left, right = rng.choice(len(cells), size=2, replace=False)
        exact[pair] = biased_mmd2(whitened[left], whitened[right])
        approximate[pair] = np.square(
            mapping.mean(whitened[left]) - mapping.mean(whitened[right])
        ).sum()
    return audit_nystrom(exact, approximate)


def _fit_map(
    values: np.ndarray,
    cells: list[np.ndarray],
    pairs: int,
    seed: int,
) -> tuple[WhiteningTransform, NystromMap, dict[str, dict[str, float | bool]]]:
    whitening = WhiteningTransform.fit(values)
    whitened = whitening.transform(values)
    audits: dict[str, dict[str, float | bool]] = {}
    mapping = None
    for landmarks in (128, 256):
        mapping = NystromMap.fit(whitened, landmarks=landmarks, seed=seed)
        audits[str(landmarks)] = _audit_map(cells, whitening, mapping, pairs, seed)
        if audits[str(landmarks)]["passed"]:
            break
    assert mapping is not None
    return whitening, mapping, audits


class TorchMap:
    def __init__(self, whitening: WhiteningTransform, mapping: NystromMap, device: torch.device):
        self.mean = torch.as_tensor(whitening.mean, device=device, dtype=torch.float32)
        self.components = torch.as_tensor(whitening.components, device=device, dtype=torch.float32)
        self.scales = torch.as_tensor(whitening.scales, device=device, dtype=torch.float32)
        self.landmarks = torch.as_tensor(mapping.landmarks, device=device, dtype=torch.float32)
        self.inverse_root = torch.as_tensor(mapping.inverse_root, device=device, dtype=torch.float32)
        self.c2 = mapping.c2
        self.device = device

    def transform(self, values: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(values, device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            white = (tensor - self.mean) @ self.components * self.scales
            features = torch.rsqrt(torch.cdist(white, self.landmarks).square() + self.c2)
            result = features @ self.inverse_root
        return result.cpu().numpy().astype(np.float64)


def _permutation(length: int, record_id: int, phase: int, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"paper06:{seed}:{record_id}:{phase}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(length)


def _squared_distances(features: list[np.ndarray]) -> np.ndarray:
    means = np.stack([cell.mean(axis=0) for cell in features])
    return np.square(means[:, None] - means[None]).sum(axis=-1)


def _represent_raw(
    cache: Path,
    frame: pd.DataFrame,
    physical_mean: np.ndarray,
    physical_std: np.ndarray,
    decomposition_mean: np.ndarray,
    basis: np.ndarray,
    macro_anchors: np.ndarray,
    macro_sigma2: float,
    macro_map: TorchMap,
    residual_map: TorchMap,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    retained: list[int] = []
    conditional: list[np.ndarray] = []
    destroyed: list[np.ndarray] = []
    sham: list[np.ndarray] = []
    macro_marginal: list[np.ndarray] = []
    residual_marginal: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ineligible: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        beats, target = _load(cache, row, physical_mean, physical_std)
        flat = beats.reshape(-1, 8)
        macro, residual = decompose_macro_residual(flat, decomposition_mean, basis)
        macro_cells = _phase_cells(macro.reshape(len(beats), 256, 3))
        residual_cells = _phase_cells(residual.reshape(beats.shape))
        memberships = [soft_membership(cell, macro_anchors, macro_sigma2) for cell in macro_cells]
        macro_features = [macro_map.transform(cell) for cell in macro_cells]
        residual_features = [residual_map.transform(cell) for cell in residual_cells]
        destroyed_features = []
        sham_features = []
        sham_memberships = []
        for phase, (features, membership) in enumerate(zip(residual_features, memberships, strict=True)):
            order = _permutation(len(features), int(row.ecg_id), phase, seed)
            destroyed_features.append(features[order])
            sham_features.append(features[order])
            sham_memberships.append(membership[order])
        try:
            conditional.append(conditional_distance(residual_features, memberships))
            destroyed.append(conditional_distance(destroyed_features, memberships))
            sham.append(conditional_distance(sham_features, sham_memberships))
        except ValueError as error:
            ineligible.append({"ecg_id": int(row.ecg_id), "reason": str(error)})
            continue
        macro_marginal.append(_squared_distances(macro_features))
        residual_marginal.append(_squared_distances(residual_features))
        labels.append(target)
        retained.append(index)
        if len(retained) % 500 == 0:
            print(json.dumps({"split": cache.name, "represented": len(retained)}), flush=True)
    selected = frame.iloc[retained].reset_index(drop=True)
    return {
        "conditional": np.stack(conditional),
        "destroyed": np.stack(destroyed),
        "sham": np.stack(sham),
        "macro": np.stack(macro_marginal),
        "residual": np.stack(residual_marginal),
        "labels": np.stack(labels),
        "ecg_ids": selected.ecg_id.to_numpy(dtype=np.int64),
        "patient_ids": selected.patient_id.to_numpy(dtype=np.int64),
    }, pd.DataFrame(ineligible, columns=["ecg_id", "reason"])


def _allowed_mask() -> np.ndarray:
    indices = np.arange(PHASES)
    separation = np.abs(indices[:, None] - indices[None])
    return np.minimum(separation, PHASES - separation) > 1


def _fit_tau(distances: np.ndarray) -> float:
    values = distances[:, _allowed_mask()]
    candidates = values[np.isfinite(values) & (values > 0)]
    if not len(candidates):
        raise ValueError("no positive train-only Paper 6 distances")
    return float(np.median(candidates))


def _recurrence(distances: np.ndarray, tau: float) -> np.ndarray:
    allowed = _allowed_mask()
    affinity = np.where(allowed[None], np.exp(-distances / tau), 0.0)
    degree = affinity.sum(axis=-1).clip(min=1e-8)
    inverse_root = 1.0 / np.sqrt(degree)
    result = inverse_root[:, :, None] * affinity * inverse_root[:, None, :]
    if not np.isfinite(result).all():
        raise ValueError("non-finite Paper 6 recurrence")
    return result.astype(np.float32)


def _load_full_signal(path: Path, ecg_ids: np.ndarray) -> np.ndarray:
    with np.load(path) as item:
        lookup = {int(ecg_id): index for index, ecg_id in enumerate(item["ecg_ids"])}
        missing = [int(ecg_id) for ecg_id in ecg_ids if int(ecg_id) not in lookup]
        if missing:
            raise ValueError(f"Paper 1 full-signal control lacks ECG IDs: {missing[:10]}")
        return np.asarray(item["kernel_recurrence"])[[lookup[int(value)] for value in ecg_ids]]


def _assemble(raw: dict[str, np.ndarray], full_signal: np.ndarray, taus: dict[str, float]) -> dict[str, np.ndarray]:
    conditional = _recurrence(raw["conditional"], taus["conditional"])
    destroyed = _recurrence(raw["destroyed"], taus["conditional"])
    sham = _recurrence(raw["sham"], taus["conditional"])
    macro = _recurrence(raw["macro"], taus["macro"])
    residual = _recurrence(raw["residual"], taus["residual"])
    duplicate = lambda value: np.stack((value, value), axis=1)
    return {
        "full": duplicate(conditional),
        "full_signal": duplicate(full_signal.astype(np.float32)),
        "macro_component": duplicate(macro),
        "residual_marginal": duplicate(residual),
        "macro_residual_marginals": np.stack((macro, residual), axis=1),
        "shuffled_residual": duplicate(destroyed),
        "joint_pair_sham": duplicate(sham),
        "labels": raw["labels"],
        "ecg_ids": raw["ecg_ids"],
        "patient_ids": raw["patient_ids"],
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
    parser.add_argument("--paper01-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reservoir", type=int, default=500_000)
    parser.add_argument("--audit-pairs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train = _eligible(args.train_cache)
    validation = _eligible(args.validation_cache)
    scaler = json.loads(args.scaler.read_text())
    physical_mean = np.asarray(scaler["mean"], dtype=np.float64)
    physical_std = np.asarray(scaler["std"], dtype=np.float64)
    reservoir = _reservoir(
        args.train_cache, train, physical_mean, physical_std, args.reservoir, args.seed
    )
    decomposition_mean, basis = _fit_decomposition(reservoir)
    macro_reservoir, residual_reservoir = decompose_macro_residual(
        reservoir, decomposition_mean, basis
    )
    macro_cluster = MiniBatchKMeans(
        n_clusters=MACRO_STATES, batch_size=4096, n_init=10, random_state=args.seed
    ).fit(macro_reservoir)
    macro_anchors = np.asarray(macro_cluster.cluster_centers_, dtype=np.float64)
    macro_sigma2 = _median_nearest_squared(macro_reservoir, macro_anchors)
    macro_cells, residual_cells = _audit_cells(
        args.train_cache, train, args.validation_cache, validation,
        physical_mean, physical_std, decomposition_mean, basis, args.seed,
    )
    macro_whitening, macro_mapping, macro_audits = _fit_map(
        macro_reservoir, macro_cells, args.audit_pairs, args.seed
    )
    residual_whitening, residual_mapping, residual_audits = _fit_map(
        residual_reservoir, residual_cells, args.audit_pairs, args.seed
    )
    selected_macro_audit = macro_audits[str(len(macro_mapping.landmarks))]
    selected_residual_audit = residual_audits[str(len(residual_mapping.landmarks))]
    args.output.mkdir(parents=True, exist_ok=True)
    if not selected_macro_audit["passed"] or not selected_residual_audit["passed"]:
        manifest = {
            "kind": "paper06_conditional_residual_representations",
            "status": "ineligible_nystrom_fidelity",
            "macro_audits": macro_audits,
            "residual_audits": residual_audits,
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"Paper 6 Nyström fidelity failed: {manifest}")
    device = torch.device(args.device)
    train_raw, train_ineligible = _represent_raw(
        args.train_cache, train, physical_mean, physical_std, decomposition_mean, basis,
        macro_anchors, macro_sigma2, TorchMap(macro_whitening, macro_mapping, device),
        TorchMap(residual_whitening, residual_mapping, device), args.seed,
    )
    validation_raw, validation_ineligible = _represent_raw(
        args.validation_cache, validation, physical_mean, physical_std, decomposition_mean, basis,
        macro_anchors, macro_sigma2, TorchMap(macro_whitening, macro_mapping, device),
        TorchMap(residual_whitening, residual_mapping, device), args.seed,
    )
    taus = {name: _fit_tau(train_raw[name]) for name in ("conditional", "macro", "residual")}
    train_output = _assemble(
        train_raw,
        _load_full_signal(args.paper01_source / "representation_train.npz", train_raw["ecg_ids"]),
        taus,
    )
    validation_output = _assemble(
        validation_raw,
        _load_full_signal(args.paper01_source / "representation_val.npz", validation_raw["ecg_ids"]),
        taus,
    )
    sham_distance_error = max(
        float(np.max(np.abs(raw["conditional"] - raw["sham"])))
        for raw in (train_raw, validation_raw)
    )
    finite = all(
        np.isfinite(payload[key]).all()
        for payload in (train_output, validation_output)
        for key in (
            "full", "full_signal", "macro_component", "residual_marginal",
            "macro_residual_marginals", "shuffled_residual", "joint_pair_sham",
        )
    )
    audit = {
        "passed": bool(finite and sham_distance_error < 1e-10),
        "finite": bool(finite),
        "joint_pair_sham_distance_max_abs_error": sham_distance_error,
        "train_effective_mass_ineligible": len(train_ineligible),
        "validation_effective_mass_ineligible": len(validation_ineligible),
    }
    if not audit["passed"]:
        raise RuntimeError(f"Paper 6 representation audit failed: {audit}")
    _atomic_npz(
        args.output / "conditional_fit.npz",
        physical_mean=physical_mean,
        physical_std=physical_std,
        decomposition_mean=decomposition_mean,
        macro_basis=basis,
        macro_anchors=macro_anchors,
        macro_sigma2=np.asarray(macro_sigma2),
        macro_whitening_mean=macro_whitening.mean,
        macro_whitening_components=macro_whitening.components,
        macro_whitening_scales=macro_whitening.scales,
        macro_landmarks=macro_mapping.landmarks,
        macro_inverse_root=macro_mapping.inverse_root,
        residual_whitening_mean=residual_whitening.mean,
        residual_whitening_components=residual_whitening.components,
        residual_whitening_scales=residual_whitening.scales,
        residual_landmarks=residual_mapping.landmarks,
        residual_inverse_root=residual_mapping.inverse_root,
        conditional_tau=np.asarray(taus["conditional"]),
        macro_tau=np.asarray(taus["macro"]),
        residual_tau=np.asarray(taus["residual"]),
    )
    _atomic_npz(args.output / "representation_train.npz", **train_output)
    _atomic_npz(args.output / "representation_val.npz", **validation_output)
    train_ineligible.to_csv(args.output / "ineligible_train.csv", index=False)
    validation_ineligible.to_csv(args.output / "ineligible_val.csv", index=False)
    manifest = {
        "kind": "paper06_conditional_residual_representations",
        "status": "complete",
        "folds": {"fit": [1, 2, 3, 4, 5, 6, 7], "selection": [8]},
        "train_records": len(train_output["ecg_ids"]),
        "validation_records": len(validation_output["ecg_ids"]),
        "rank": 3,
        "macro_states": MACRO_STATES,
        "minimum_effective_mass": 2.0,
        "denominator_floor": 1e-8,
        "macro_landmarks": len(macro_mapping.landmarks),
        "residual_landmarks": len(residual_mapping.landmarks),
        "macro_audits": macro_audits,
        "residual_audits": residual_audits,
        "taus": taus,
        "controls": [
            "full_signal", "macro_component", "residual_marginal",
            "macro_residual_marginals", "shuffled_residual", "joint_pair_sham",
        ],
        "audit": audit,
        "seed": args.seed,
        "source": {
            "train_manifest_sha256": _sha256(args.train_cache / "manifest.json"),
            "validation_manifest_sha256": _sha256(args.validation_cache / "manifest.json"),
            "scaler_sha256": _sha256(args.scaler),
            "paper01_manifest_sha256": _sha256(args.paper01_source / "manifest.json"),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
