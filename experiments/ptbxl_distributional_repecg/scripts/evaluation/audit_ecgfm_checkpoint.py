#!/usr/bin/env python3
"""Read-only provenance audit for the archived ECG-FM checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("ECG-FM payload is not a checkpoint dictionary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "sha256": digest(args.checkpoint),
        "bytes": args.checkpoint.stat().st_size, "top_level_keys": sorted(payload),
        "status": "inspected_only_not_admitted",
        "reason": "pretraining provenance and a non-proxy sparse-lead interface are unverified",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
