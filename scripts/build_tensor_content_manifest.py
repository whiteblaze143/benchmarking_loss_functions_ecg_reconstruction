#!/usr/bin/env python3
"""Build byte-level PTB-XL tensor manifests and split Merkle-style roots."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tensor-root",
        type=Path,
        default=Path("data/ptb_xl/tensors"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("refine-logs/ptbxl_tensor_content_manifest.json"),
    )
    args = parser.parse_args()

    splits = {}
    for split in ("train", "val", "test"):
        split_path = args.tensor_root / split
        entries = []
        root = hashlib.sha256()
        for path in sorted(split_path.glob("*.pt"), key=lambda item: int(item.stem)):
            digest = sha256_file(path)
            entry = {
                "record_id": int(path.stem),
                "relative_path": str(path.relative_to(args.tensor_root)),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
            entries.append(entry)
            root.update(
                f"{entry['record_id']}:{entry['size_bytes']}:{digest}\n".encode()
            )
        splits[split] = {
            "records": len(entries),
            "content_root_sha256": root.hexdigest(),
            "entries": entries,
        }

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tensor_root": str(args.tensor_root),
        "root_definition": "sha256(concat(record_id:size_bytes:file_sha256\\n))",
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                split: {
                    "records": value["records"],
                    "content_root_sha256": value["content_root_sha256"],
                }
                for split, value in splits.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
