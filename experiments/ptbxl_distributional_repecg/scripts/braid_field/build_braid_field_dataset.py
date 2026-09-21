#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repecg.braid_field.features import FrozenResponseFeatureMap, phase_scalar_field
from repecg.braid_field.geometry import canonical_q8_geometry, make_interpolated_query_bank


def _eligible(cache: Path) -> pd.DataFrame:
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("kind") != "production_phase_cache":
        raise ValueError(f"not a production phase cache: {cache}")
    return pd.read_csv(cache / "qc.csv").query("eligible").reset_index(drop=True)


def _load_physical(cache: Path, row: pd.Series, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache / str(row.artifact)) as item:
        beats = np.asarray(item["beats"], dtype=np.float32)
        labels = np.asarray(item["labels"], dtype=np.float32)
    return beats * std.astype(np.float32) + mean.astype(np.float32), labels


def _build_split(
    cache: Path,
    frame: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    fmap: FrozenResponseFeatureMap,
    query_ops: torch.Tensor,
    query_coords: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    context_ops, _ = canonical_q8_geometry(device=device)
    n = len(frame)
    responses = np.empty((n, 8, 16, fmap.response_dim), np.float32)
    field = np.empty((n, 16, len(query_ops)), np.float32)
    labels = np.empty((n, 5), np.float32)

    # Beat counts vary between records, so process records independently.
    # Do not np.stack record beat tensors: that silently assumes equal beat counts.
    for index in range(n):
        beats, y = _load_physical(cache, frame.iloc[index], mean, std)
        beats_t = torch.as_tensor(beats[None], device=device)
        with torch.inference_mode():
            responses[index] = fmap.project_basis(beats_t, context_ops)[0].cpu().numpy()
            field[index] = phase_scalar_field(
                beats_t, query_ops, fmap.voltage_scale
            )[0].cpu().numpy()
        labels[index] = y
        if (index + 1) % max(1, batch) == 0 or index + 1 == n:
            print(json.dumps({"split": cache.name, "built": index + 1, "total": n}), flush=True)

    return {
        "context_operators": context_ops.cpu().numpy().astype(np.float32),
        "context_responses": responses,
        "query_operators": query_ops.cpu().numpy().astype(np.float32),
        "query_coords": query_coords.cpu().numpy().astype(np.float32),
        "target_field": field,
        "labels": labels,
        "ecg_ids": frame.ecg_id.to_numpy(dtype=np.int64),
        "patient_ids": frame.patient_id.to_numpy(dtype=np.int64),
    }


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--response-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-theta", type=int, default=7)
    parser.add_argument("--query-phi", type=int, default=7)
    parser.add_argument("--query-sigma", type=float, default=0.45)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    scaler = json.loads(args.scaler.read_text())
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    fmap = FrozenResponseFeatureMap(args.response_fit, device=device)
    query_ops, query_coords = make_interpolated_query_bank(
        args.query_theta, args.query_phi, sigma=args.query_sigma, device=device
    )

    args.output.mkdir(parents=True, exist_ok=True)
    train_frame = _eligible(args.train_cache)
    val_frame = _eligible(args.validation_cache)
    train = _build_split(
        args.train_cache, train_frame, mean, std, fmap, query_ops, query_coords,
        batch=args.batch, device=device,
    )
    val = _build_split(
        args.validation_cache, val_frame, mean, std, fmap, query_ops, query_coords,
        batch=args.batch, device=device,
    )
    _atomic_npz(args.output / "train.npz", train)
    _atomic_npz(args.output / "val.npz", val)
    manifest = {
        "kind": "braid_field_q8_dataset",
        "status": "complete",
        "context": "canonical_Q8",
        "query_bank": {
            "construction": "diagnostic_RBF_interpolation_smoke_manifold",
            "theta": args.query_theta,
            "phi": args.query_phi,
            "sigma": args.query_sigma,
            "count": len(query_ops),
            "publication_warning": "replace_or_validate_with_physiological_dense_query_manifold",
        },
        "train_records": len(train_frame),
        "validation_records": len(val_frame),
        "response_dim": fmap.response_dim,
        "voltage_scale": fmap.voltage_scale,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
