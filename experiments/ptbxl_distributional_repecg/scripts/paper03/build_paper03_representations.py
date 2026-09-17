#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from repecg.common.kernels import NystromMap, TruncatedPCAWhitening, audit_nystrom, biased_mmd2
from repecg.paper03_signature.representations import path_descriptor, transform_path


KINDS = ("identity", "order_destroy", "time_reverse", "monotone_warp_sham")


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
    frame = pd.read_csv(cache / "qc.csv")
    return frame[frame.eligible].reset_index(drop=True)


def _load(cache: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache / str(row.artifact)) as item:
        beats = np.asarray(item["beats"], dtype=np.float64)
        labels = np.asarray(item["labels"], dtype=np.float32)
    if beats.ndim != 3 or beats.shape[1:] != (256, 8) or len(beats) < 2:
        raise ValueError(f"invalid Paper 3 beats for ECG {row.ecg_id}: {beats.shape}")
    return beats, labels


def _descriptors(beats: np.ndarray, record_id: int, kind: str, seed: int) -> np.ndarray:
    cells = beats.reshape(len(beats), 16, 16, 8)
    output = np.empty((len(beats), 16, 228), dtype=np.float64)
    for beat in range(len(beats)):
        for phase in range(16):
            path = transform_path(
                cells[beat, phase], kind=kind, record_id=record_id,
                beat=beat, phase=phase, seed=seed,
            )
            output[beat, phase] = path_descriptor(path)
    return output


def _record_order(frame: pd.DataFrame, seed: int) -> np.ndarray:
    keys = []
    for index, ecg_id in enumerate(frame.ecg_id):
        token = f"paper03-reservoir:{seed}:{int(ecg_id)}".encode()
        keys.append((hashlib.sha256(token).digest(), index))
    return np.asarray([index for _, index in sorted(keys)], dtype=np.int64)


def _reservoir(cache: Path, frame: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    chunks = []
    observed = 0
    for index in _record_order(frame, seed):
        row = frame.iloc[index]
        beats, _ = _load(cache, row)
        values = _descriptors(beats, int(row.ecg_id), "identity", seed).reshape(-1, 228)
        take = min(len(values), count - observed)
        chunks.append(values[:take])
        observed += take
        if observed == count:
            break
    if observed != count:
        raise ValueError(f"only {observed} descriptors available for reservoir of {count}")
    return np.concatenate(chunks)


def _audit_cells(
    train_cache: Path,
    train: pd.DataFrame,
    validation_cache: Path,
    validation: pd.DataFrame,
    whitening: TruncatedPCAWhitening,
    seed: int,
) -> list[np.ndarray]:
    cells = []
    for cache, frame in ((train_cache, train), (validation_cache, validation)):
        for index in _record_order(frame, seed)[:32]:
            row = frame.iloc[index]
            beats, _ = _load(cache, row)
            descriptors = _descriptors(beats, int(row.ecg_id), "identity", seed)
            cells.extend(whitening.transform(descriptors[:, phase]) for phase in range(16))
    return cells


def _audit_mapping(
    cells: list[np.ndarray], mapping: NystromMap, pairs: int, seed: int
) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    exact = np.empty(pairs)
    approximate = np.empty(pairs)
    for pair in range(pairs):
        left, right = rng.choice(len(cells), size=2, replace=False)
        exact[pair] = biased_mmd2(cells[left], cells[right])
        approximate[pair] = np.square(mapping.mean(cells[left]) - mapping.mean(cells[right])).sum()
    return audit_nystrom(exact, approximate)


def _represent(
    cache: Path,
    frame: pd.DataFrame,
    whitening: TruncatedPCAWhitening,
    mapping: NystromMap,
    unordered_whitening: TruncatedPCAWhitening,
    paper02_path: Path,
    seed: int,
) -> dict[str, np.ndarray]:
    outputs = {
        kind: np.empty((len(frame), 16, len(mapping.landmarks)), dtype=np.float32)
        for kind in KINDS
    }
    labels = np.empty((len(frame), 5), dtype=np.float32)
    for index, row in frame.iterrows():
        beats, labels[index] = _load(cache, row)
        for kind in KINDS:
            descriptors = _descriptors(beats, int(row.ecg_id), kind, seed)
            white = whitening.transform(descriptors.reshape(-1, 228))
            features = mapping.transform(white).reshape(len(beats), 16, -1)
            outputs[kind][index] = features.mean(axis=0)
    with np.load(paper02_path) as item:
        if not np.array_equal(frame.ecg_id.to_numpy(dtype=np.int64), item["ecg_ids"]):
            raise ValueError("Paper 2 control ECG IDs disagree with Paper 3 cache")
        unordered_source = np.asarray(item["kernel"], dtype=np.float64)
        unordered = unordered_whitening.transform(
            unordered_source.reshape(-1, unordered_source.shape[-1])
        ).reshape(unordered_source.shape[0], 16, -1).astype(np.float32)
    return {
        "signature": outputs["identity"],
        "order_scrambled": outputs["order_destroy"],
        "time_reversed": outputs["time_reverse"],
        "monotone_warp_sham": outputs["monotone_warp_sham"],
        "unordered_kme": unordered,
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
    parser.add_argument("--paper02-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reservoir", type=int, default=100_000)
    parser.add_argument("--audit-pairs", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train = _eligible(args.train_cache)
    validation = _eligible(args.validation_cache)
    reservoir = _reservoir(args.train_cache, train, args.reservoir, args.seed)
    whitening = TruncatedPCAWhitening.fit(reservoir, n_components=64)
    whitened = whitening.transform(reservoir)
    with np.load(args.paper02_source / "representation_train.npz") as item:
        unordered_source = np.asarray(item["kernel"], dtype=np.float64)
    unordered_whitening = TruncatedPCAWhitening.fit(
        unordered_source.reshape(-1, unordered_source.shape[-1]), n_components=128
    )
    audit_cells = _audit_cells(
        args.train_cache, train, args.validation_cache, validation, whitening, args.seed
    )
    audits = {}
    mapping = None
    for landmarks in (128, 256):
        mapping = NystromMap.fit(whitened, landmarks=landmarks, seed=args.seed)
        audits[str(landmarks)] = _audit_mapping(audit_cells, mapping, args.audit_pairs, args.seed)
        if audits[str(landmarks)]["passed"]:
            break
    assert mapping is not None
    if not audits[str(len(mapping.landmarks))]["passed"]:
        raise RuntimeError(f"Paper 3 Nyström audit failed: {audits}")
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        args.output / "signature_fit.npz",
        whitening_mean=whitening.mean,
        whitening_components=whitening.components,
        whitening_scales=whitening.scales,
        landmarks=mapping.landmarks,
        inverse_root=mapping.inverse_root,
        c2=np.asarray(mapping.c2),
    )
    _atomic_npz(
        args.output / "unordered_control_fit.npz",
        mean=unordered_whitening.mean,
        components=unordered_whitening.components,
        scales=unordered_whitening.scales,
    )
    _atomic_npz(
        args.output / "representation_train.npz",
        **_represent(
            args.train_cache, train, whitening, mapping, unordered_whitening,
            args.paper02_source / "representation_train.npz", args.seed,
        ),
    )
    _atomic_npz(
        args.output / "representation_val.npz",
        **_represent(
            args.validation_cache, validation, whitening, mapping, unordered_whitening,
            args.paper02_source / "representation_val.npz", args.seed,
        ),
    )
    manifest = {
        "kind": "paper03_path_signature_representations",
        "status": "complete",
        "folds": {"fit": [1, 2, 3, 4, 5, 6, 7], "selection": [8]},
        "train_records": len(train),
        "validation_records": len(validation),
        "descriptor": {"dimension": 228, "path_dimension": 8, "depth": 3},
        "pca_components": len(whitening.scales),
        "reservoir_descriptors": args.reservoir,
        "landmarks": len(mapping.landmarks),
        "unordered_control_components": len(unordered_whitening.scales),
        "unordered_control_fit": "folds_1_7_truncated_pca_whitening",
        "audit": audits[str(len(mapping.landmarks))],
        "audits": audits,
        "seed": args.seed,
        "source": {
            "train_manifest_sha256": _sha256(args.train_cache / "manifest.json"),
            "validation_manifest_sha256": _sha256(args.validation_cache / "manifest.json"),
            "paper02_manifest_sha256": _sha256(args.paper02_source / "manifest.json"),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
