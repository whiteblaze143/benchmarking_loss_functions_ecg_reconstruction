#!/usr/bin/env python3
"""Materialize standard-12-lead P07 responses with the existing frozen Nyström map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repecg.common.kernels import NystromMap, WhiteningTransform
from repecg.paper07_operator import (
    canonical_operators,
    derived_limb_operators,
    response_atoms,
    time_indexed_response_atoms,
)


SCHEMA_VERSION = "paper07_frozen_full12_context_responses_v1"
LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _eligible(cache: Path) -> pd.DataFrame:
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("kind") != "production_phase_cache":
        raise ValueError(f"not a production phase cache: {cache}")
    frame = pd.read_csv(cache / "qc.csv").query("eligible").reset_index(drop=True)
    required = {"ecg_id", "patient_id", "artifact"}
    if missing := required - set(frame.columns):
        raise ValueError(f"cache QC missing {sorted(missing)}")
    return frame


def _base_split(path: Path, split: str) -> dict[str, np.ndarray]:
    file_name = "representation_train.npz" if split == "train" else "representation_val.npz"
    with np.load(path / file_name, allow_pickle=False) as item:
        required = {"ecg_ids", "patient_ids", "labels"}
        if missing := required - set(item.files):
            raise ValueError(f"base P07 split missing {sorted(missing)}")
        values = {name: np.asarray(item[name]) for name in required}
    if values["labels"].shape != (len(values["ecg_ids"]), 5) or values["patient_ids"].shape != values["ecg_ids"].shape:
        raise ValueError("base P07 split metadata is malformed")
    return values


class TorchResponseMap:
    def __init__(self, response_fit: Path, device: torch.device):
        with np.load(response_fit, allow_pickle=False) as item:
            required = {"physical_mean", "physical_std", "voltage_scale", "whitening_mean", "whitening_components", "whitening_scales", "landmarks", "inverse_root", "c2"}
            if missing := required - set(item.files):
                raise ValueError(f"response fit missing {sorted(missing)}")
            self.physical_mean = np.asarray(item["physical_mean"], dtype=np.float64)
            self.physical_std = np.asarray(item["physical_std"], dtype=np.float64)
            self.voltage_scale = float(item["voltage_scale"])
            whitening = WhiteningTransform(
                mean=np.asarray(item["whitening_mean"], dtype=np.float64),
                components=np.asarray(item["whitening_components"], dtype=np.float64),
                scales=np.asarray(item["whitening_scales"], dtype=np.float64),
            )
            mapping = NystromMap(
                landmarks=np.asarray(item["landmarks"], dtype=np.float64),
                inverse_root=np.asarray(item["inverse_root"], dtype=np.float64),
                c2=float(item["c2"]),
            )
        if self.physical_mean.shape != (8,) or self.physical_std.shape != (8,) or self.voltage_scale <= 0:
            raise ValueError("invalid frozen response fit physical scaling")
        self.mean = torch.as_tensor(whitening.mean, device=device, dtype=torch.float32)
        self.components = torch.as_tensor(whitening.components, device=device, dtype=torch.float32)
        self.scales = torch.as_tensor(whitening.scales, device=device, dtype=torch.float32)
        self.landmarks = torch.as_tensor(mapping.landmarks, device=device, dtype=torch.float32)
        self.inverse_root = torch.as_tensor(mapping.inverse_root, device=device, dtype=torch.float32)
        self.c2 = mapping.c2
        self.atom_dim = len(whitening.mean)
        self.landmark_count = len(mapping.landmarks)
        self.device = device

    def phase_means(self, atoms: np.ndarray) -> np.ndarray:
        if atoms.ndim != 3 or atoms.shape[1] != 256:
            raise ValueError("response atoms must have shape [beat,256,features]")
        cells = atoms.reshape(len(atoms), 16, 16, atoms.shape[-1]).transpose(1, 0, 2, 3).reshape(16, -1, atoms.shape[-1])
        values = torch.as_tensor(cells, device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            white = (values - self.mean) @ self.components * self.scales
            features = torch.rsqrt(torch.cdist(white, self.landmarks.expand(16, -1, -1)).square() + self.c2) @ self.inverse_root
        return features.mean(dim=1).cpu().numpy().astype(np.float32)


def _standard_12_operators() -> np.ndarray:
    independent = canonical_operators()
    derived = derived_limb_operators()
    result = np.stack((independent[0], independent[1], derived[0], derived[1], derived[2], derived[3], *independent[2:]))
    if result.shape != (12, 8):
        raise AssertionError("standard operator construction failed")
    return result.astype(np.float32)


def _write_split(
    output: Path, split: str, cache: Path, base: dict[str, np.ndarray], response_map: TorchResponseMap, operators: np.ndarray,
    atom_fn,
) -> None:
    frame = _eligible(cache)
    if not np.array_equal(frame.ecg_id.to_numpy(dtype=np.int64), base["ecg_ids"]) or not np.array_equal(frame.patient_id.to_numpy(dtype=np.int64), base["patient_ids"]):
        raise ValueError(f"{split} cache ordering does not exactly match frozen P07 representations")
    destination = output / f"{split}_responses.npy"
    temporary = destination.with_suffix(".tmp.npy")
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(frame), 12, 16, response_map.landmark_count))
    for index, row in frame.iterrows():
        with np.load(cache / str(row.artifact), allow_pickle=False) as item:
            beats = np.asarray(item["beats"], dtype=np.float64)
            labels = np.asarray(item["labels"], dtype=np.float32)
        if beats.ndim != 3 or beats.shape[1:] != (256, 8) or not np.array_equal(labels, base["labels"][index]):
            raise ValueError(f"{split} phase cache payload disagrees at ECG {row.ecg_id}")
        physical_beats = beats * response_map.physical_std + response_map.physical_mean
        for lead_index, q in enumerate(operators):
            values[index, lead_index] = response_map.phase_means(atom_fn(physical_beats, q, response_map.voltage_scale))
        if (index + 1) % 500 == 0:
            print(json.dumps({"split": split, "materialized": index + 1}), flush=True)
    values.flush()
    del values
    os.replace(temporary, destination)
    for name in ("labels", "ecg_ids", "patient_ids"):
        np.save(output / f"{split}_{name}.npy", base[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True, help="Complete frozen P07 representation artifact")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-indexed", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated context artifact: {args.output}")
    base_manifest_path = args.representations / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    if base_manifest.get("kind") != "paper07_measurement_operator_response_sets" or base_manifest.get("status") != "complete":
        raise ValueError("requires a complete frozen P07 representation artifact")
    response_fit = args.representations / "response_fit.npz"
    train, validation = _base_split(args.representations, "train"), _base_split(args.representations, "validation")
    if set(train["ecg_ids"]) & set(validation["ecg_ids"]):
        raise ValueError("training and selection ECG IDs overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    response_map = TorchResponseMap(response_fit, torch.device(args.device))
    atom_fn = time_indexed_response_atoms if args.time_indexed else response_atoms
    expected_features = 3 if args.time_indexed else 2
    if response_map.atom_dim != expected_features:
        raise ValueError("response-fit dimensionality does not match requested response atoms")
    operators = _standard_12_operators()
    np.save(args.output / "operators.npy", operators)
    _write_split(args.output, "train", args.train_cache, train, response_map, operators, atom_fn)
    _write_split(args.output, "validation", args.validation_cache, validation, response_map, operators, atom_fn)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "lead_order": list(LEAD_ORDER),
        "operators": "standard_12_lead_operator_bank_in_canonical_target_order",
        "response_map": "frozen_time_indexed_p07_response_fit_no_refit" if args.time_indexed else "frozen_shared_p07_response_fit_no_refit",
        "time_indexed": bool(args.time_indexed),
        "folds": {"train": list(range(1, 8)), "selection": [8]},
        "shape": {"train": [len(train["labels"]), 12, 16, response_map.landmark_count], "validation": [len(validation["labels"]), 12, 16, response_map.landmark_count]},
        "source": {
            "base_manifest_sha256": _sha256(base_manifest_path),
            "response_fit_sha256": _sha256(response_fit),
            "train_cache_manifest_sha256": _sha256(args.train_cache / "manifest.json"),
            "validation_cache_manifest_sha256": _sha256(args.validation_cache / "manifest.json"),
        },
        "files": {name: _sha256(args.output / name) for name in (
            "operators.npy", "train_responses.npy", "validation_responses.npy", "train_labels.npy", "validation_labels.npy",
            "train_ecg_ids.npy", "validation_ecg_ids.npy", "train_patient_ids.npy", "validation_patient_ids.npy",
        )},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "shape": manifest["shape"]}))


if __name__ == "__main__":
    main()
