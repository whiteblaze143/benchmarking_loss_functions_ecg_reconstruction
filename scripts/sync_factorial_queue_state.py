#!/usr/bin/env python3
"""Synchronize a stopped experiment-queue state with its generated manifest.

Completed/running/failed status is preserved by job id while commands, expected
outputs, phase membership, and phase dependencies are refreshed from the
manifest.  The scheduler must be stopped before this script is run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--drop-orphans",
        action="store_true",
        help="Drop stopped jobs removed from the new manifest (recorded in state metadata).",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    state = json.loads(args.state.read_text())
    if any(job.get("status") == "running" for job in state.get("jobs", [])):
        raise SystemExit("Refusing to synchronize while a queue job is running")

    manifest_jobs: dict[str, tuple[str, dict]] = {}
    for phase in manifest.get("phases", []):
        for job in phase.get("jobs", []):
            job_id = job["id"]
            if job_id in manifest_jobs:
                raise ValueError(f"Duplicate manifest job id: {job_id}")
            manifest_jobs[job_id] = (phase["name"], job)

    existing = {job["id"]: job for job in state.get("jobs", [])}
    orphaned = sorted(set(existing) - set(manifest_jobs))
    if orphaned and not args.drop_orphans:
        raise ValueError(f"State contains jobs absent from manifest: {orphaned}")

    synchronized_jobs = []
    for job_id, (phase_name, manifest_job) in manifest_jobs.items():
        old = existing.get(job_id, {})
        synchronized_jobs.append({
            "id": job_id,
            "phase": phase_name,
            "cmd": manifest_job["cmd"],
            "expected_output": manifest_job.get("expected_output"),
            "status": old.get("status", "pending"),
            "gpu": old.get("gpu"),
            "screen_name": old.get("screen_name"),
            "pid": old.get("pid"),
            "attempts": old.get("attempts", 0),
            "started": old.get("started"),
            "completed": old.get("completed"),
            "error": old.get("error"),
        })

    synchronized_phases = []
    for phase in manifest.get("phases", []):
        jobs = [job for job in synchronized_jobs if job["phase"] == phase["name"]]
        statuses = {job["status"] for job in jobs}
        status = "completed" if jobs and statuses == {"completed"} else "pending"
        synchronized_phases.append({
            "name": phase["name"],
            "depends_on": phase.get("depends_on", []),
            "status": status,
        })

    state["phases"] = synchronized_phases
    state["jobs"] = synchronized_jobs
    state.setdefault("meta", {})["manifest_synchronized_at"] = datetime.now(timezone.utc).isoformat()
    if orphaned:
        state["meta"]["dropped_stopped_jobs"] = orphaned
    backup = args.state.with_suffix(args.state.suffix + ".pre_sync.bak")
    shutil.copy2(args.state, backup)
    temporary = args.state.with_suffix(args.state.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, args.state)
    print(json.dumps({
        "jobs": len(synchronized_jobs),
        "phases": len(synchronized_phases),
        "added_jobs": sorted(set(manifest_jobs) - set(existing)),
        "dropped_jobs": orphaned,
        "backup": str(backup),
    }, indent=2))


if __name__ == "__main__":
    main()
