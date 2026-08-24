#!/usr/bin/env python3
"""Create an exact, provenance-bound subset of a model registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    selected = [
        model for model in registry["models"] if model["id"].startswith(args.prefix)
    ]
    if len(selected) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} models with prefix {args.prefix!r}; "
            f"found {len(selected)}"
        )
    registry["models"] = selected
    registry["subset_provenance"] = {
        "source_registry": str(args.registry),
        "source_registry_sha256": sha256(args.registry),
        "selection_prefix": args.prefix,
        "selected_ids": [model["id"] for model in selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(selected)} models to {args.output}")


if __name__ == "__main__":
    main()
