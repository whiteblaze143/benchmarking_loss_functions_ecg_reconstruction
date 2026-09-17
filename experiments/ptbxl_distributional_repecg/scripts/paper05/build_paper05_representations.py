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
from sklearn.decomposition import PCA

from repecg.paper05_koopman import (
    chronological_permutation,
    koopman_operator,
    median_anchor_bandwidth,
    operator_descriptor,
    soft_observables,
)


ANCHORS = 32
RIDGE = 1e-2
PCA_COMPONENTS = 64
SPECTRAL_COMPONENTS = 2 * ANCHORS + 4
FULL_COMPONENTS = ANCHORS + PCA_COMPONENTS + SPECTRAL_COMPONENTS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible(cache: Path) -> tuple[pd.DataFrame, int]:
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("kind") != "production_phase_cache":
        raise ValueError(f"not a production phase cache: {cache}")
    frame = pd.read_csv(cache / "qc.csv")
    source_eligible = frame[frame.eligible].copy()
    admitted = source_eligible[source_eligible.valid_cycles >= 3].reset_index(drop=True)
    return admitted, int(len(source_eligible) - len(admitted))


def _load(cache: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache / str(row.artifact)) as item:
        beats = np.asarray(item["beats"], dtype=np.float32)
        labels = np.asarray(item["labels"], dtype=np.float32)
    if beats.ndim != 3 or beats.shape[1:] != (256, 8) or len(beats) < 3:
        raise ValueError(f"invalid Paper 5 beats for ECG {row.ecg_id}: {beats.shape}")
    return beats, labels


def _record_order(frame: pd.DataFrame, seed: int) -> np.ndarray:
    keyed = []
    for index, ecg_id in enumerate(frame.ecg_id):
        token = f"paper05-reservoir:{seed}:{int(ecg_id)}".encode()
        keyed.append((hashlib.sha256(token).digest(), index))
    return np.asarray([index for _, index in sorted(keyed)], dtype=np.int64)


class KernelEmbedder:
    def __init__(self, fit_path: Path, device: torch.device):
        with np.load(fit_path) as item:
            self.mean = torch.as_tensor(item["whitening_mean"], device=device, dtype=torch.float32)
            self.components = torch.as_tensor(
                item["whitening_components"], device=device, dtype=torch.float32
            )
            self.scales = torch.as_tensor(item["whitening_scales"], device=device, dtype=torch.float32)
            self.landmarks = torch.as_tensor(item["landmarks"], device=device, dtype=torch.float32)
            self.inverse_root = torch.as_tensor(item["inverse_root"], device=device, dtype=torch.float32)
            self.c2 = float(item["c2"])
        self.device = device

    def beat_cell_means(self, beats: np.ndarray) -> np.ndarray:
        count = len(beats) * 16
        cells = torch.as_tensor(
            beats.reshape(count, 16, 8), device=self.device, dtype=torch.float32
        )
        with torch.inference_mode():
            white = (cells - self.mean) @ self.components * self.scales
            distances = torch.cdist(white, self.landmarks.expand(count, -1, -1))
            features = torch.rsqrt(distances.square() + self.c2) @ self.inverse_root
            means = features.mean(dim=1)
        return means.cpu().numpy().astype(np.float32)


def _reservoir(
    cache: Path, frame: pd.DataFrame, embedder: KernelEmbedder, count: int, seed: int
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    observed = 0
    for index in _record_order(frame, seed):
        beats, _ = _load(cache, frame.iloc[index])
        values = embedder.beat_cell_means(beats)
        take = min(len(values), count - observed)
        chunks.append(values[:take])
        observed += take
        if observed == count:
            break
    if observed != count:
        raise ValueError(f"only {observed} beat-cell means available for reservoir of {count}")
    return np.concatenate(chunks)


def _raw_descriptors(
    cache: Path,
    frame: pd.DataFrame,
    embedder: KernelEmbedder,
    anchors: np.ndarray,
    tau: float,
    seed: int,
) -> dict[str, np.ndarray]:
    records = len(frame)
    occupancy = np.empty((records, ANCHORS), dtype=np.float32)
    operators = np.empty((records, ANCHORS * ANCHORS), dtype=np.float32)
    shuffled_operators = np.empty_like(operators)
    spectral = np.empty((records, SPECTRAL_COMPONENTS), dtype=np.float32)
    shuffled_spectral = np.empty_like(spectral)
    labels = np.empty((records, 5), dtype=np.float32)
    occupancy_error = 0.0
    for index, row in frame.iterrows():
        beats, labels[index] = _load(cache, row)
        means = embedder.beat_cell_means(beats)
        observable = soft_observables(means, anchors, tau)
        permutation = chronological_permutation(len(observable), int(row.ecg_id), seed)
        shuffled = observable[permutation]
        occupancy[index] = observable.mean(axis=0)
        occupancy_error = max(
            occupancy_error, float(np.max(np.abs(observable.mean(axis=0) - shuffled.mean(axis=0))))
        )
        operator = koopman_operator(observable, ridge=RIDGE)
        shuffled_operator = koopman_operator(shuffled, ridge=RIDGE)
        operators[index] = operator.reshape(-1)
        shuffled_operators[index] = shuffled_operator.reshape(-1)
        spectral[index] = operator_descriptor(operator, observable)
        shuffled_spectral[index] = operator_descriptor(shuffled_operator, shuffled)
        if (index + 1) % 500 == 0:
            print(json.dumps({"split": cache.name, "represented": index + 1}), flush=True)
    return {
        "occupancy": occupancy,
        "operators": operators,
        "shuffled_operators": shuffled_operators,
        "spectral": spectral,
        "shuffled_spectral": shuffled_spectral,
        "labels": labels,
        "ecg_ids": frame.ecg_id.to_numpy(dtype=np.int64),
        "patient_ids": frame.patient_id.to_numpy(dtype=np.int64),
        "occupancy_error": np.asarray(occupancy_error),
    }


def _assemble(raw: dict[str, np.ndarray], pca: PCA) -> dict[str, np.ndarray]:
    projected = pca.transform(raw["operators"]).astype(np.float32)
    shuffled_projected = pca.transform(raw["shuffled_operators"]).astype(np.float32)
    full = np.concatenate((raw["occupancy"], projected, raw["spectral"]), axis=1)
    chronology_shuffled = np.concatenate(
        (raw["occupancy"], shuffled_projected, raw["shuffled_spectral"]), axis=1
    )
    occupancy_only = np.zeros_like(full)
    occupancy_only[:, :ANCHORS] = raw["occupancy"]
    return {
        "full": full.astype(np.float32),
        "occupancy_only": occupancy_only.astype(np.float32),
        "chronology_shuffled": chronology_shuffled.astype(np.float32),
        "identity_order_sham": full.astype(np.float32).copy(),
        "labels": raw["labels"],
        "ecg_ids": raw["ecg_ids"],
        "patient_ids": raw["patient_ids"],
    }


def _scale(
    train: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    mean = train["full"].mean(axis=0, dtype=np.float64)
    scale = train["full"].std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    for payload in (train, validation):
        for key in ("full", "chronology_shuffled", "identity_order_sham"):
            payload[key] = ((payload[key] - mean) / scale).astype(np.float32)
        occupancy = ((payload["occupancy_only"][:, :ANCHORS] - mean[:ANCHORS]) / scale[:ANCHORS])
        payload["occupancy_only"].fill(0.0)
        payload["occupancy_only"][:, :ANCHORS] = occupancy.astype(np.float32)
    return mean, scale


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--paper02-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reservoir", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_frame, train_ineligible = _eligible(args.train_cache)
    validation_frame, validation_ineligible = _eligible(args.validation_cache)
    device = torch.device(args.device)
    embedder = KernelEmbedder(args.paper02_source / "kernel_fit.npz", device)
    reservoir = _reservoir(args.train_cache, train_frame, embedder, args.reservoir, args.seed)
    clusterer = MiniBatchKMeans(
        n_clusters=ANCHORS,
        batch_size=4096,
        n_init=10,
        random_state=args.seed,
    ).fit(reservoir)
    anchors = np.asarray(clusterer.cluster_centers_, dtype=np.float64)
    tau = median_anchor_bandwidth(reservoir, anchors)
    train_raw = _raw_descriptors(
        args.train_cache, train_frame, embedder, anchors, tau, args.seed
    )
    validation_raw = _raw_descriptors(
        args.validation_cache, validation_frame, embedder, anchors, tau, args.seed
    )
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=args.seed)
    pca.fit(train_raw["operators"])
    train = _assemble(train_raw, pca)
    validation = _assemble(validation_raw, pca)
    feature_mean, feature_scale = _scale(train, validation)
    finite = all(
        np.isfinite(payload[key]).all()
        for payload in (train, validation)
        for key in ("full", "occupancy_only", "chronology_shuffled", "identity_order_sham")
    )
    sham_error = max(
        float(np.max(np.abs(payload["full"] - payload["identity_order_sham"])))
        for payload in (train, validation)
    )
    occupancy_error = max(
        float(train_raw["occupancy_error"]), float(validation_raw["occupancy_error"])
    )
    audit = {
        "passed": bool(finite and sham_error == 0.0 and occupancy_error <= 1e-12),
        "finite": bool(finite),
        "identity_sham_max_abs_error": sham_error,
        "shuffle_occupancy_max_abs_error": occupancy_error,
        "train_transition_ineligible": train_ineligible,
        "validation_transition_ineligible": validation_ineligible,
    }
    if not audit["passed"]:
        raise RuntimeError(f"Paper 5 representation audit failed: {audit}")
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        args.output / "koopman_fit.npz",
        anchors=anchors,
        tau=np.asarray(tau),
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    _atomic_npz(args.output / "representation_train.npz", **train)
    _atomic_npz(args.output / "representation_val.npz", **validation)
    manifest = {
        "kind": "paper05_record_specific_koopman_representations",
        "status": "complete",
        "folds": {"fit": [1, 2, 3, 4, 5, 6, 7], "selection": [8]},
        "train_records": len(train_frame),
        "validation_records": len(validation_frame),
        "transition_gate": {"minimum_cycles": 3, "minimum_transitions": 32},
        "anchors": ANCHORS,
        "anchor_fit": "training_only_minibatch_kmeans",
        "bandwidth": tau,
        "bandwidth_fit": "training_median_squared_distance_to_nearest_anchor",
        "ridge": RIDGE,
        "operator_pca_components": PCA_COMPONENTS,
        "spectral_components": SPECTRAL_COMPONENTS,
        "representation_components": FULL_COMPONENTS,
        "controls": {
            "occupancy_only": "same_width_zero_padded_after_32_occupancy_coordinates",
            "chronology_shuffled": "deterministic_record_specific_permutation_preserving_occupancy",
            "identity_order_sham": "identity_order_with_identical_descriptor_pipeline",
        },
        "audit": audit,
        "seed": args.seed,
        "source": {
            "train_manifest_sha256": _sha256(args.train_cache / "manifest.json"),
            "validation_manifest_sha256": _sha256(args.validation_cache / "manifest.json"),
            "paper02_manifest_sha256": _sha256(args.paper02_source / "manifest.json"),
            "paper02_kernel_fit_sha256": _sha256(args.paper02_source / "kernel_fit.npz"),
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
