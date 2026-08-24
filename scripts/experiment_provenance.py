"""Exact source provenance helpers for controlled experiment checkpoints."""

from __future__ import annotations

import hashlib
import base64
import subprocess
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_provenance(root: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    root = root.resolve()
    paths = sorted(set(relative_paths))
    hashes = {}
    contents = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Provenance source does not exist: {path}")
        source_bytes = path.read_bytes()
        hashes[relative] = hashlib.sha256(source_bytes).hexdigest()
        contents[relative] = base64.b64encode(source_bytes).decode("ascii")
    bundle = hashlib.sha256()
    for relative, digest in hashes.items():
        bundle.update(f"{relative}:{digest}\n".encode())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", *paths],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    return {
        "git_commit": revision,
        "git_dirty_for_sources": bool(diff),
        "source_file_sha256": hashes,
        "source_file_contents_base64": contents,
        "source_bundle_sha256": bundle.hexdigest(),
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_diff_base64": base64.b64encode(diff).decode("ascii"),
    }
