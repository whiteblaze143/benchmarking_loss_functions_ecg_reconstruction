#!/usr/bin/env python3
"""Materialize the PTB-XL sex-only context aligned to a frozen P07 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SOURCE_SCHEMA = "paper07_ptbxl_patient_context_v1"
SCHEMA = "paper07_ptbxl_sex_context_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated context artifact: {args.output}")
    source_manifest = args.source / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    if manifest.get("schema_version") != SOURCE_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("requires complete PTB-XL age/sex context source")
    if manifest.get("features", [])[-1:] != ["recorded_sex"]:
        raise ValueError("source context has no final recorded-sex field")
    values: dict[str, np.ndarray] = {}
    for split in ("train", "validation"):
        source = np.load(args.source / f"{split}_context.npy", allow_pickle=False)
        if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all() or not set(np.unique(source[:, 2])) <= {0.0, 1.0}:
            raise ValueError(f"invalid source sex field for {split}")
        values[split] = source[:, 2:3].astype(np.float32, copy=False)
    args.output.mkdir(parents=True, exist_ok=False)
    for split, value in values.items():
        temporary = args.output / f"{split}_context.tmp.npy"
        np.save(temporary, value)
        os.replace(temporary, args.output / f"{split}_context.npy")
        for suffix in ("ecg_ids", "patient_ids"):
            np.save(args.output / f"{split}_{suffix}.npy", np.load(args.source / f"{split}_{suffix}.npy", allow_pickle=False))
    files = tuple(
        f"{split}_{suffix}.npy" for split in ("train", "validation") for suffix in ("context", "ecg_ids", "patient_ids")
    )
    output = {
        "schema_version": SCHEMA, "status": "complete", "features": ["recorded_sex"],
        "source_patient_context_manifest_sha256": _sha256(source_manifest),
        "files": {name: _sha256(args.output / name) for name in files},
    }
    (args.output / "manifest.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "shape": {key: list(value.shape) for key, value in values.items()}}))


if __name__ == "__main__":
    main()
