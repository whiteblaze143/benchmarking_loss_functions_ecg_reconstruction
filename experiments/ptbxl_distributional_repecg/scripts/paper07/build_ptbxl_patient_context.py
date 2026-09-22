#!/usr/bin/env python3
"""Materialize exact, train-normalized PTB-XL age/sex context for P07."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = "paper07_ptbxl_patient_context_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(path: Path, split: str) -> dict[str, np.ndarray]:
    name = "representation_train.npz" if split == "train" else "representation_val.npz"
    with np.load(path / name, allow_pickle=False) as item:
        required = {"ecg_ids", "patient_ids"}
        if missing := required - set(item.files):
            raise ValueError(f"representation split missing {sorted(missing)}")
        values = {key: np.asarray(item[key]) for key in required}
    if values["ecg_ids"].ndim != 1 or values["patient_ids"].shape != values["ecg_ids"].shape:
        raise ValueError("representation ID vectors must align")
    return values


def _context(frame: pd.DataFrame, ids: np.ndarray, patients: np.ndarray, expected_folds: set[int]) -> np.ndarray:
    rows = frame.reindex(ids)
    if rows.isna().all(axis=1).any():
        raise ValueError("representation contains ECG absent from PTB-XL metadata")
    if not np.array_equal(rows.patient_id.to_numpy(dtype=np.int64), patients):
        raise ValueError("PTB-XL patient IDs do not match frozen representations")
    if set(rows.strat_fold.astype(int)) - expected_folds:
        raise ValueError("representation contains an out-of-contract fold")
    if not np.isfinite(rows.age.to_numpy(dtype=np.float64)).all() or not set(rows.sex.astype(int)) <= {0, 1}:
        raise ValueError("PTB-XL age/sex contract is not complete and binary")
    # PTB-XL v1.0.2 explicitly stores every age above 89 as 300 for privacy.
    age = rows.age.to_numpy(dtype=np.float32)
    return np.stack((np.minimum(age, 89.0), (age == 300.0).astype(np.float32), rows.sex.to_numpy(dtype=np.float32)), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated context artifact: {args.output}")
    representation_manifest = args.representations / "manifest.json"
    manifest = json.loads(representation_manifest.read_text())
    if manifest.get("kind") != "paper07_measurement_operator_response_sets" or manifest.get("status") != "complete":
        raise ValueError("requires complete frozen P07 representations")
    frame = pd.read_csv(args.metadata, index_col="ecg_id")
    required = {"patient_id", "strat_fold", "age", "sex"}
    if missing := required - set(frame.columns):
        raise ValueError(f"PTB-XL metadata missing {sorted(missing)}")
    if frame.index.has_duplicates:
        raise ValueError("PTB-XL metadata has duplicate ECG IDs")
    train, validation = _load_split(args.representations, "train"), _load_split(args.representations, "validation")
    train_context = _context(frame, train["ecg_ids"], train["patient_ids"], set(range(1, 8)))
    validation_context = _context(frame, validation["ecg_ids"], validation["patient_ids"], {8})
    age_mean, age_std = float(train_context[:, 0].mean()), float(train_context[:, 0].std())
    if not np.isfinite(age_mean) or not np.isfinite(age_std) or age_std <= 0:
        raise ValueError("invalid train-only age scale")
    train_context[:, 0] = (train_context[:, 0] - age_mean) / age_std
    validation_context[:, 0] = (validation_context[:, 0] - age_mean) / age_std
    args.output.mkdir(parents=True, exist_ok=False)
    for split, values in (("train", train_context), ("validation", validation_context)):
        temporary = args.output / f"{split}_context.tmp.npy"
        np.save(temporary, values.astype(np.float32))
        os.replace(temporary, args.output / f"{split}_context.npy")
    for split, values in (("train", train), ("validation", validation)):
        np.save(args.output / f"{split}_ecg_ids.npy", values["ecg_ids"])
        np.save(args.output / f"{split}_patient_ids.npy", values["patient_ids"])
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "features": ["age_years_capped_at_documented_89_year_privacy_boundary", "age_90_plus_censoring_indicator", "recorded_sex"],
        "folds": {"train": list(range(1, 8)), "selection": [8]},
        "age_scaling": {"fit": "train_only", "mean": age_mean, "std": age_std},
        "source": {"representation_manifest_sha256": _sha256(representation_manifest), "ptbxl_metadata_sha256": _sha256(args.metadata)},
        "files": {name: _sha256(args.output / name) for name in (
            "train_context.npy", "validation_context.npy", "train_ecg_ids.npy", "validation_ecg_ids.npy", "train_patient_ids.npy", "validation_patient_ids.npy",
        )},
    }
    (args.output / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "shape": {"train": list(train_context.shape), "validation": list(validation_context.shape)}}))


if __name__ == "__main__":
    main()
