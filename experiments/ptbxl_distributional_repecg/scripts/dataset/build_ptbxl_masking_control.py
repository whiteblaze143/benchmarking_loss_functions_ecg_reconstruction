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

from repecg.common.kernels import NystromMap, WhiteningTransform
from repecg.paper02_kernel_mean.controls import exact_moment_matched_gaussian, mean_covariance_features


MASK_I_II = np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_standardized_beats(beats: np.ndarray) -> np.ndarray:
    values = np.asarray(beats, dtype=np.float32).copy()
    if values.ndim != 3 or values.shape[1:] != (256, 8):
        raise ValueError(f"expected standardized beats shaped [B,256,8], got {values.shape}")
    values[..., MASK_I_II == 0] = 0.0
    return values


def _embed(
    cells: torch.Tensor,
    whitening: WhiteningTransform,
    mapping: NystromMap,
    generator: torch.Generator,
) -> dict[str, np.ndarray]:
    device = cells.device
    white_mean = torch.as_tensor(whitening.mean, device=device, dtype=torch.float32)
    components = torch.as_tensor(whitening.components, device=device, dtype=torch.float32)
    scales = torch.as_tensor(whitening.scales, device=device, dtype=torch.float32)
    anchors = torch.as_tensor(mapping.landmarks, device=device, dtype=torch.float32)
    inverse_root = torch.as_tensor(mapping.inverse_root, device=device, dtype=torch.float32)

    def kernel_mean(values: torch.Tensor) -> torch.Tensor:
        white = (values.float() - white_mean) @ components * scales
        features = torch.rsqrt(torch.cdist(white, anchors.expand(len(values), -1, -1)).square() + mapping.c2)
        return (features @ inverse_root).mean(dim=1)

    with torch.inference_mode():
        matched = exact_moment_matched_gaussian(cells, generator=generator, tolerance=1e-5)
        return {
            "kernel": kernel_mean(cells).cpu().numpy(),
            "gaussian": kernel_mean(matched).cpu().numpy(),
            "linear": (((cells - white_mean) @ components) * scales).mean(dim=1).cpu().numpy(),
            "moments": mean_covariance_features(cells.double()).float().cpu().numpy(),
        }


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--kernel-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    cache_manifest_path = args.cache / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if cache_manifest.get("kind") != "production_phase_cache":
        raise ValueError("matched masking control requires a production phase cache")
    if cache_manifest.get("split") != "development_select":
        raise ValueError("matched masking control is frozen to PTB-XL development_select fold 8")
    frame = pd.read_csv(args.cache / "qc.csv")
    frame = frame.loc[frame["eligible"]].reset_index(drop=True)
    if set(frame["fold"]) != {8}:
        raise ValueError(f"matched masking control expected only fold 8, got {sorted(set(frame['fold']))}")
    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("--max-records must be positive")
        frame = frame.iloc[: args.max_records].copy()

    with np.load(args.kernel_fit) as item:
        whitening = WhiteningTransform(item["whitening_mean"], item["whitening_components"], item["whitening_scales"])
        mapping = NystromMap(item["landmarks"], item["inverse_root"], float(item["c2"]))

    keys = ("kernel", "gaussian", "linear", "moments")
    full = {key: [] for key in keys}
    masked = {key: [] for key in keys}
    labels: list[np.ndarray] = []
    device = torch.device("cuda")
    for row_index, row in enumerate(frame.itertuples(index=False)):
        with np.load(args.cache / str(row.artifact)) as item:
            beats = np.asarray(item["beats"], dtype=np.float32)
            labels.append(np.asarray(item["labels"], dtype=np.float32))
        masked_beats = mask_standardized_beats(beats)
        if not np.all(masked_beats[..., 2:] == 0.0):
            raise AssertionError("masked standardized channels are not exact zero")

        def cells(values: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                values.reshape(len(values), 16, 16, 8)
                .transpose(1, 0, 2, 3)
                .reshape(16, len(values) * 16, 8),
                device=device,
            )

        pair_seed = args.seed + row_index
        full_generator = torch.Generator(device=device).manual_seed(pair_seed)
        masked_generator = torch.Generator(device=device).manual_seed(pair_seed)
        full_embedded = _embed(cells(beats), whitening, mapping, full_generator)
        masked_embedded = _embed(cells(masked_beats), whitening, mapping, masked_generator)
        for key in keys:
            full[key].append(full_embedded[key])
            masked[key].append(masked_embedded[key])

    common = {
        "labels": np.stack(labels),
        "ecg_ids": frame["ecg_id"].to_numpy(dtype=np.int64),
        "patient_ids": frame["patient_id"].to_numpy(dtype=np.int64),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    full_path = args.output / "representation_ptbxl_full_fold8.npz"
    masked_path = args.output / "representation_ptbxl_masked_i_ii_fold8.npz"
    _atomic_npz(
        full_path,
        {**{key: np.stack(values) for key, values in full.items()}, **common,
         "observed_lead_mask": np.tile(np.ones(8, dtype=np.uint8), (len(frame), 1))},
    )
    _atomic_npz(
        masked_path,
        {**{key: np.stack(values) for key, values in masked.items()}, **common,
         "observed_lead_mask": np.tile(MASK_I_II, (len(frame), 1))},
    )
    manifest = {
        "kind": "paired_ptbxl_missing_lead_control",
        "evaluation_role": "negative_control_matched_reference",
        "split": "development_select",
        "fold": 8,
        "records": len(frame),
        "patients": int(frame["patient_id"].nunique()),
        "conditions": {
            "full": {"mask": [1] * 8, "archive": full_path.name, "sha256": _sha256(full_path)},
            "masked_i_ii": {"mask": MASK_I_II.tolist(), "archive": masked_path.name, "sha256": _sha256(masked_path)},
        },
        "padding": "exact_zero_in_frozen_ptbxl_standardized_coordinates",
        "pairing": "same_record_same_beats_same_gaussian_rng_seed",
        "cache_manifest_sha256": _sha256(cache_manifest_path),
        "kernel_fit_sha256": _sha256(args.kernel_fit),
        "seed": args.seed,
        "max_records": args.max_records,
    }
    (args.output / "ptbxl_masking_control_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
