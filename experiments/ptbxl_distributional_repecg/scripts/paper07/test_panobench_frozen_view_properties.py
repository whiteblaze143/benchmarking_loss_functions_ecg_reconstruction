#!/usr/bin/env python3
"""Test SetOperator set properties on the sealed native 48-view PanoBench bank.

This is a structural architecture test. PanoBench view directions are 3-D and
P07 clinical measurement operators are 8-D, so this script deliberately makes
no clinical unseen-operator reconstruction or transfer-performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io
import torch

from repecg.paper07_operator import OperatorSetModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bank(coordinates: Path) -> np.ndarray:
    payload = json.loads(coordinates.read_text())["panobench"]
    values = sorted(payload["leads"].values(), key=lambda item: item["index"])
    indices = [item["index"] for item in values]
    if indices != list(range(48)):
        raise ValueError(f"PanoBench coordinate registry must contain exact indices 0..47, got {indices}")
    bank = np.asarray([[item["unit_sphere_cartesian"][axis] for axis in ("x", "y", "z")] for item in values], dtype=np.float32)
    if bank.shape != (48, 3) or not np.isfinite(bank).all() or not np.allclose(np.linalg.norm(bank, axis=1), 1.0, atol=1e-5):
        raise ValueError("invalid frozen PanoBench spatial operator bank")
    return bank


def _responses(values: np.ndarray) -> np.ndarray:
    if values.shape != (48, 2500) or not np.isfinite(values).all():
        raise ValueError(f"expected finite native PanoBench [48,2500], got {values.shape}")
    cells = np.array_split(values, 16, axis=1)
    means = np.stack([cell.mean(axis=1) for cell in cells], axis=1)[..., None]
    derivative = np.diff(values, axis=1, prepend=values[:, :1])
    derivative_cells = np.array_split(derivative, 16, axis=1)
    derivatives = np.stack([cell.mean(axis=1) for cell in derivative_cells], axis=1)[..., None]
    return np.concatenate((means, derivatives), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=Path("/data/mithunmanivannan/panobench/test"))
    parser.add_argument("--coordinates", type=Path, default=Path("/data/mithunmanivannan/nef-net-aim/datasets/coordinate_registry.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite property test output: {args.output}")
    files = sorted(args.test_root.glob("*.mat"), key=lambda path: int(path.stem))
    if len(files) != 1030 or args.records < 1 or args.records > len(files):
        raise ValueError(f"requires 1..1030 records from the sealed PanoBench test split, found {len(files)}")
    bank = _bank(args.coordinates)
    selection = np.random.default_rng(args.seed).choice(len(files), size=args.records, replace=False)
    selected = [files[index] for index in np.sort(selection)]
    values = np.stack([scipy.io.loadmat(path)["Panobench"].astype(np.float32) for path in selected])
    responses = np.stack([_responses(item) for item in values])
    torch.manual_seed(args.seed)
    model = OperatorSetModel(response_dim=2, classes=1, operator_mode="continuous", operator_dim=3).eval()
    operators = torch.as_tensor(bank).unsqueeze(0).expand(args.records, -1, -1)
    features = torch.as_tensor(responses)
    generator = torch.Generator().manual_seed(args.seed + 1)
    order_full = torch.randperm(48, generator=generator)
    subset = torch.arange(0, 48, 4)
    order_subset = subset[torch.randperm(len(subset), generator=generator)]
    with torch.inference_mode():
        full = model.encode_context(operators, features)
        full_permuted = model.encode_context(operators[:, order_full], features[:, order_full])
        sparse = model.encode_context(operators[:, subset], features[:, subset])
        sparse_permuted = model.encode_context(operators[:, order_subset], features[:, order_subset])
    full_error = float((full - full_permuted).abs().max())
    sparse_error = float((sparse - sparse_permuted).abs().max())
    tolerance = 1e-6
    if full_error > tolerance or sparse_error > tolerance:
        raise AssertionError(f"set permutation property failed: full={full_error}, sparse={sparse_error}")
    output = {
        "schema_version": "p07_panobench_frozen_native_view_property_test_v1",
        "status": "pass",
        "scope": "architecture_property_only_not_p07_8d_unseen_operator_performance",
        "test_split": "PanoBench sealed test",
        "records": len(selected),
        "views": 48,
        "operator_bank": {"dimension": 3, "coordinate_sha256": _sha256(args.coordinates), "all_48_native_views": True},
        "response_source": "real_PanoBench_waveform_16_equal_time_cells_mean_and_derivative",
        "properties": {
            "all_48_view_permutation_max_abs_error": full_error,
            "12_view_subset_permutation_max_abs_error": sparse_error,
            "tolerance": tolerance,
        },
        "records_sha256": {path.name: _sha256(path) for path in selected},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "records": output["records"], "views": output["views"], "max_error": max(full_error, sparse_error)}))


if __name__ == "__main__":
    main()
