#!/usr/bin/env python3
"""Pin and acquire the official HuBERT-ECG release; no evaluation occurs here."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Edoardo-Coppola/hubert-ecg-large")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    info = HfApi().model_info(args.repo_id)
    if not info.sha:
        raise RuntimeError("official HuBERT model has no immutable revision SHA")
    snapshot_download(repo_id=args.repo_id, revision=info.sha, local_dir=args.output)
    files = [path for path in sorted(args.output.rglob("*")) if path.is_file()]
    if not files:
        raise RuntimeError("official HuBERT snapshot download produced no files")
    (args.output / "admission.json").write_text(json.dumps({
        "repo_id": args.repo_id, "revision": info.sha,
        "files": [{"path": str(path.relative_to(args.output)), "bytes": path.stat().st_size, "sha256": digest(path)} for path in files],
        "status": "acquired_not_evaluation_admitted",
        "next_gate": "verify author preprocessing and native 12-lead embedding extraction",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
