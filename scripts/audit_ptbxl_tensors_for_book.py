#!/usr/bin/env python3
"""Audit every canonical PTB-XL tensor used by the reconstruction loaders.

This script is intentionally separate from the Quarto chapter so a full 5 GB
inventory is not repeated on every book render. The chapter loads the derived
record-level Parquet and JSON summary, checks their provenance, and performs a
paired raw-WFDB comparison on a deterministic stratified sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SPLIT_FROM_FOLD = {
    "train": set(range(1, 9)),
    "val": {9},
    "test": {10},
}


def audit_tensor(path: Path, split: str, root: Path) -> dict[str, object]:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} contains {type(tensor)!r}, not torch.Tensor")
    values = (
        tensor.detach().to(device="cpu", dtype=torch.float32)
        .contiguous().numpy()
    )
    finite = np.isfinite(values)
    finite_values = values[finite]
    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    lead_ranges = np.ptp(safe_values, axis=-1)
    return {
        "split": split,
        "ecg_id": int(path.stem),
        "relative_path": str(path.relative_to(root)),
        "file_bytes": path.stat().st_size,
        "ndim": tensor.ndim,
        "channels": tensor.shape[0] if tensor.ndim >= 1 else np.nan,
        "samples": tensor.shape[-1] if tensor.ndim >= 1 else np.nan,
        "dtype": str(tensor.dtype),
        "nonfinite_values": int((~finite).sum()),
        "flat_leads": int((lead_ranges <= 1e-8).sum()),
        "minimum": float(finite_values.min()) if finite_values.size else np.nan,
        "maximum": float(finite_values.max()) if finite_values.size else np.nan,
        "mean": float(finite_values.mean()) if finite_values.size else np.nan,
        "rms": (
            float(np.sqrt(np.mean(np.square(finite_values, dtype=np.float64))))
            if finite_values.size
            else np.nan
        ),
        "signal_sha256": hashlib.sha256(
            values.tobytes(order="C")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/book_eda"),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    tensor_root = root / "data/ptb_xl/tensors"
    metadata_path = root / "data/ptb_xl/ptbxl_database.csv"
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        files = sorted(
            (tensor_root / split).glob("*.pt"),
            key=lambda value: int(value.stem),
        )
        print(f"Auditing {split}: {len(files):,} tensors", flush=True)
        for index, path in enumerate(files, start=1):
            rows.append(audit_tensor(path, split, root))
            if index % 500 == 0:
                print(f"  {split}: {index:,}/{len(files):,}", flush=True)

    inventory = pd.DataFrame(rows).sort_values(["split", "ecg_id"])
    expected_rows = []
    for split, folds in SPLIT_FROM_FOLD.items():
        expected_rows.extend(
            {"split": split, "ecg_id": int(ecg_id)}
            for ecg_id in metadata.loc[
                metadata.strat_fold.isin(folds), "ecg_id"
            ]
        )
    expected = pd.DataFrame(expected_rows)
    alignment = expected.merge(
        inventory[["split", "ecg_id"]],
        on=["split", "ecg_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    duplicate_sizes = inventory.groupby("signal_sha256").size()
    summary = {
        "schema_version": 1,
        "source_metadata": str(metadata_path.relative_to(root)),
        "tensor_root": str(tensor_root.relative_to(root)),
        "record_count": int(len(inventory)),
        "split_counts": {
            key: int(value)
            for key, value in inventory.split.value_counts().sort_index().items()
        },
        "expected_split_counts": {
            split: int(metadata.strat_fold.isin(folds).sum())
            for split, folds in SPLIT_FROM_FOLD.items()
        },
        "metadata_tensor_alignment": {
            key: int(value)
            for key, value in alignment["_merge"].value_counts().items()
        },
        "shape_12_by_5000": int(
            ((inventory.channels == 12) & (inventory.samples == 5000)).sum()
        ),
        "float32": int(inventory.dtype.eq("torch.float32").sum()),
        "records_with_nonfinite_values": int(
            inventory.nonfinite_values.gt(0).sum()
        ),
        "records_with_flat_leads": int(inventory.flat_leads.gt(0).sum()),
        "unique_signal_hashes": int(inventory.signal_sha256.nunique()),
        "duplicate_hash_groups": int(duplicate_sizes.gt(1).sum()),
        "records_in_duplicate_hash_groups": int(
            duplicate_sizes[duplicate_sizes.gt(1)].sum()
        ),
        "minimum": float(inventory.minimum.min()),
        "maximum": float(inventory.maximum.max()),
        "record_rms_quantiles": {
            str(quantile): float(inventory.rms.quantile(quantile))
            for quantile in (0.0, 0.01, 0.5, 0.99, 1.0)
        },
        "audit_script": str(Path(__file__).resolve().relative_to(root)),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    inventory_path = output_dir / "ptbxl_tensor_inventory.parquet"
    summary_path = output_dir / "ptbxl_tensor_audit_summary.json"
    inventory.to_parquet(inventory_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {inventory_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
