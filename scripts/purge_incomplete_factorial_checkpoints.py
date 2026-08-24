#!/usr/bin/env python3
"""Remove only provably incomplete checkpoints for pending factorial jobs.

A completed training writes a schema-v2 metadata sidecar after epoch 10. A
pending job with a checkpoint but no sidecar is an intermediate "best so far"
artifact and cannot support final analysis. This tool records hashes and log
sentinels before deletion so the operational failure remains auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    state = json.loads(args.state.read_text())
    candidates = []
    for job in state["jobs"]:
        if job["status"] != "pending":
            continue
        match = re.search(r"--checkpoint_path\s+(\S+)", job["cmd"])
        if not match:
            continue
        checkpoint = Path(match.group(1))
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        sidecar = checkpoint.with_suffix(".metadata.json")
        if not checkpoint.is_file() or sidecar.exists():
            continue

        log = root / "refine-logs/queue/logs" / f"{job['id']}.log"
        exitcode_file = log.with_suffix(log.suffix + ".exitcode")
        exitcode = (
            exitcode_file.read_text().strip()
            if exitcode_file.is_file() else None
        )
        candidates.append({
            "job_id": job["id"],
            "state_status": job["status"],
            "checkpoint": str(checkpoint.relative_to(root)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "metadata_sidecar_present": False,
            "log": (
                str(log.relative_to(root)) if log.is_file() else None
            ),
            "exitcode": exitcode,
        })

    audit = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "rule": (
            "state=pending AND checkpoint exists AND final metadata sidecar "
            "is absent"
        ),
        "execute": args.execute,
        "files": candidates,
        "total_bytes": sum(item["bytes"] for item in candidates),
    }
    output = (
        args.audit_output
        if args.audit_output.is_absolute()
        else root / args.audit_output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n")

    if args.execute:
        for item in candidates:
            (root / item["checkpoint"]).unlink()

    print(json.dumps({
        "execute": args.execute,
        "files": len(candidates),
        "bytes": audit["total_bytes"],
        "audit_output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
