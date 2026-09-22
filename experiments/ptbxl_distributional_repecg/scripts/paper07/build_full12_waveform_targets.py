#!/usr/bin/env python3
"""Materialize exact PTB-XL full-12-lead targets for the P07 auxiliary variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from repecg.common import load_config
from repecg.common.preprocess import bandpass_ecg
from repecg.common.ptbxl import PTBXLStore


SCHEMA_VERSION = "paper07_full12_waveform_targets_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _representation_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        required = {"ecg_ids", "patient_ids", "labels"}
        missing = required - set(item.files)
        if missing:
            raise ValueError(f"representation split missing {sorted(missing)}")
        result = {name: np.asarray(item[name]) for name in required}
    if result["ecg_ids"].ndim != 1 or result["patient_ids"].shape != result["ecg_ids"].shape:
        raise ValueError("representation ECG and patient IDs must be aligned vectors")
    if result["labels"].shape != (len(result["ecg_ids"]), 5):
        raise ValueError("representation labels must have shape [record,5]")
    if len(np.unique(result["ecg_ids"])) != len(result["ecg_ids"]):
        raise ValueError("representation ECG IDs must be unique within a split")
    return result


def _fit_scaler(store: PTBXLStore, train: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    mean = np.zeros(12, dtype=np.float64)
    m2 = np.zeros(12, dtype=np.float64)
    for ecg_id, patient_id, labels in zip(train["ecg_ids"], train["patient_ids"], train["labels"], strict=True):
        record = store.read_record(int(ecg_id))
        if record.patient_id != int(patient_id) or not np.array_equal(record.labels, labels):
            raise ValueError(f"PTB-XL source mismatch for ECG {ecg_id}")
        if record.fold not in set(range(1, 8)):
            raise ValueError(f"training target includes non-training fold for ECG {ecg_id}")
        signal = bandpass_ecg(record.signal_mv)
        if signal.shape != (5000, 12):
            raise ValueError(f"unexpected waveform shape for ECG {ecg_id}: {signal.shape}")
        batch_count = len(signal)
        batch_mean = signal.mean(axis=0, dtype=np.float64)
        batch_m2 = np.square(signal - batch_mean).sum(axis=0, dtype=np.float64)
        total = count + batch_count
        delta = batch_mean - mean
        mean += delta * batch_count / total
        m2 += batch_m2 + np.square(delta) * count * batch_count / total
        count = total
    if count == 0:
        raise ValueError("cannot fit a scaler with no training samples")
    std = np.sqrt(m2 / count)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 1e-8):
        raise ValueError("invalid full-12-lead training scaler")
    return mean, std


def _write_split(
    destination: Path,
    split_name: str,
    store: PTBXLStore,
    payload: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    values = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(len(payload["ecg_ids"]), 12, 5000),
    )
    expected_fold = set(range(1, 8)) if split_name == "train" else {8}
    for index, (ecg_id, patient_id, labels) in enumerate(
        zip(payload["ecg_ids"], payload["patient_ids"], payload["labels"], strict=True)
    ):
        record = store.read_record(int(ecg_id))
        if record.patient_id != int(patient_id) or not np.array_equal(record.labels, labels):
            raise ValueError(f"PTB-XL source mismatch for ECG {ecg_id}")
        if record.fold not in expected_fold:
            raise ValueError(f"{split_name} target has unexpected fold for ECG {ecg_id}")
        signal = bandpass_ecg(record.signal_mv)
        if signal.shape != (5000, 12):
            raise ValueError(f"unexpected waveform shape for ECG {ecg_id}: {signal.shape}")
        standardized = (signal - mean) / std
        if not np.isfinite(standardized).all():
            raise ValueError(f"non-finite standardized target for ECG {ecg_id}")
        values[index] = standardized.T.astype(np.float16)
        if (index + 1) % 500 == 0:
            print(json.dumps({"split": split_name, "materialized": index + 1}), flush=True)
    values.flush()
    del values
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/common.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated target artifact: {args.output}")
    manifest_path = args.representations / "manifest.json"
    representation_manifest = json.loads(manifest_path.read_text())
    if representation_manifest.get("kind") != "paper07_measurement_operator_response_sets" or representation_manifest.get("status") != "complete":
        raise ValueError("requires a complete Paper 07 representation artifact")
    train_path = args.representations / "representation_train.npz"
    validation_path = args.representations / "representation_val.npz"
    train = _representation_split(train_path)
    validation = _representation_split(validation_path)
    if set(train["ecg_ids"]) & set(validation["ecg_ids"]):
        raise ValueError("training and selection ECG IDs overlap")
    config = load_config(args.config)
    store = PTBXLStore(config)
    mean, std = _fit_scaler(store, train)
    args.output.mkdir(parents=True, exist_ok=False)
    _write_split(args.output / "train_waveforms.npy", "train", store, train, mean, std)
    _write_split(args.output / "validation_waveforms.npy", "validation", store, validation, mean, std)
    np.save(args.output / "train_ecg_ids.npy", train["ecg_ids"])
    np.save(args.output / "validation_ecg_ids.npy", validation["ecg_ids"])
    np.save(args.output / "train_patient_ids.npy", train["patient_ids"])
    np.save(args.output / "validation_patient_ids.npy", validation["patient_ids"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "target": "bandpass_filtered_canonical_12_lead_500Hz_10s_train_standardized",
        "folds": {"train": list(range(1, 8)), "selection": [8]},
        "shape": {"train": [len(train["ecg_ids"]), 12, 5000], "validation": [len(validation["ecg_ids"]), 12, 5000]},
        "scaler": {"mean": mean.tolist(), "std": std.tolist(), "fit_records": len(train["ecg_ids"])},
        "source": {
            "representation_manifest_sha256": _sha256(manifest_path),
            "train_representation_sha256": _sha256(train_path),
            "validation_representation_sha256": _sha256(validation_path),
        },
        "files": {
            name: _sha256(args.output / name)
            for name in (
                "train_waveforms.npy", "validation_waveforms.npy", "train_ecg_ids.npy", "validation_ecg_ids.npy",
                "train_patient_ids.npy", "validation_patient_ids.npy",
            )
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output), "shape": manifest["shape"]}))


if __name__ == "__main__":
    main()
