#!/usr/bin/env python3
"""Atomically add a retained mask block to the live spatial one-lead queue."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "refine-logs/queue_spatial_1lead/queue_state.json"
TRANSIENT_FIELDS = {
    "started", "completed", "checkpoint_size_bytes", "failed_at",
    "returncode", "error", "cancel_reason",
}


def atomic_write(path: Path, payload: dict) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask")
    parser.add_argument("--source-mask", default="1000000")
    parser.add_argument("--recover-running", action="store_true")
    args = parser.parse_args()
    if len(args.mask) != 7 or not args.mask.isdigit():
        raise ValueError("mask must contain exactly seven digits")
    data = json.loads(QUEUE.read_text())
    jobs = data["jobs"]
    existing_ids = {job["id"] for job in jobs}
    target_existing = [job for job in jobs if args.mask in job["id"]]
    if target_existing:
        raise RuntimeError(f"target mask already exists in {len(target_existing)} queue jobs")
    source = [
        job for job in jobs
        if args.source_mask in job["id"] and job.get("status") != "cancelled"
    ]
    if len(source) != 20:
        raise RuntimeError(f"expected 20 retained source jobs, found {len(source)}")
    additions = []
    for original in source:
        clone = {
            key: value for key, value in original.items()
            if key not in TRANSIENT_FIELDS
        }
        clone["id"] = original["id"].replace(args.source_mask, args.mask)
        clone["cmd"] = original["cmd"].replace(args.source_mask, args.mask)
        clone["status"] = "pending"
        clone["attempts"] = 0
        if clone["id"] in existing_ids:
            raise RuntimeError(f"duplicate job id: {clone['id']}")
        if f"--factorial_mask {args.mask}" not in clone["cmd"]:
            raise RuntimeError(f"mask substitution failed for {clone['id']}")
        if f"--run_name {clone['id']}" not in clone["cmd"]:
            raise RuntimeError(f"run-name identity mismatch for {clone['id']}")
        additions.append(clone)
        existing_ids.add(clone["id"])
    running = [job for job in jobs if job.get("status") == "running"]
    if running and not args.recover_running:
        raise RuntimeError("queue still contains a running job; stop the worker first")
    for job in running:
        job["status"] = "pending"
        job["restart_reason"] = "controlled queue expansion for mask 1110000"
        for field in ("started", "failed_at", "returncode", "error"):
            job.pop(field, None)
    jobs.extend(additions)
    data["purpose"] = (
        "Priority A0-E1, PA1 fusion, canonical exact-Theta equivalence, "
        "capacity-matched, and permuted-geometry one-lead controls across "
        "masks 1010010, 1000000, and 1110000 before resuming the 3-lead queue"
    )
    retained_counts = {
        mask: sum(mask in job["id"] and job.get("status") != "cancelled" for job in jobs)
        for mask in ("1010010", "1000000", "1110000")
    }
    if retained_counts != {"1010010": 20, "1000000": 20, "1110000": 20}:
        raise RuntimeError(f"unexpected retained grid: {retained_counts}")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise RuntimeError("queue contains duplicate ids")
    backup = QUEUE.with_name("queue_state.pre_1110000.json")
    if not backup.exists():
        shutil.copy2(QUEUE, backup)
    atomic_write(QUEUE, data)
    print(json.dumps({
        "added": len(additions), "requeued_running": [job["id"] for job in running],
        "retained_counts": retained_counts, "total_jobs": len(jobs),
        "backup": str(backup),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
