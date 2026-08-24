#!/usr/bin/env python3
"""Recover scheduler-caused factorial attempts without losing audit evidence.

Run only after the queue manager and the named job screens have stopped. The
script backs up queue state and logs, quarantines partial checkpoints, and
returns only the named jobs to pending.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def checkpoint_from_command(command: str, root: Path) -> Path:
    match = re.search(r"--checkpoint_path\s+(\S+)", command)
    if not match:
        raise ValueError(f"No --checkpoint_path in command: {command}")
    path = Path(match.group(1))
    return path if path.is_absolute() else root / path


def copy_if_present(source: Path, destination_dir: Path) -> None:
    if source.is_file():
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / source.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    state_path = args.state.resolve()
    recovery = args.recovery_dir.resolve()
    recovery.mkdir(parents=True, exist_ok=False)

    state = json.loads(state_path.read_text())
    by_id = {job["id"]: job for job in state["jobs"]}
    missing = sorted(set(args.job) - set(by_id))
    if missing:
        raise KeyError(f"Jobs absent from state: {missing}")

    shutil.copy2(state_path, recovery / "queue_state.before.json")
    canonical_log_dir = root / "refine-logs/queue/logs"
    root_log_dir = recovery / "root_attempt_logs"
    canonical_backup_dir = recovery / "canonical_logs_before_retry"
    artifact_dir = recovery / "partial_artifacts"

    repaired = []
    for job_id in args.job:
        job = by_id[job_id]
        checkpoint = checkpoint_from_command(job["cmd"], root)
        sidecar = checkpoint.with_suffix(".metadata.json")

        # Partial artifacts are recoverable but must not masquerade as completed
        # outputs when the attempt is returned to pending.
        for source in (checkpoint, sidecar):
            if source.is_file():
                artifact_dir.mkdir(parents=True, exist_ok=True)
                destination = artifact_dir / source.name
                if destination.exists():
                    raise FileExistsError(destination)
                shutil.move(str(source), destination)

        for suffix in (".log", ".log.exitcode"):
            copy_if_present(
                root / f"{job_id}{suffix}",
                root_log_dir,
            )
            copy_if_present(
                canonical_log_dir / f"{job_id}{suffix}",
                canonical_backup_dir,
            )

        repaired.append({
            "job_id": job_id,
            "previous_status": job.get("status"),
            "previous_attempts": job.get("attempts"),
            "previous_started": job.get("started"),
            "previous_completed": job.get("completed"),
            "previous_error": job.get("error"),
            "previous_screen_name": job.get("screen_name"),
            "previous_pid": job.get("pid"),
        })
        job.update({
            "status": "pending",
            "gpu": None,
            "screen_name": None,
            "pid": None,
            "attempts": 0,
            "started": None,
            "completed": None,
            "error": None,
        })

    event = {
        "timestamp": utc_now(),
        "reason": args.reason,
        "jobs": repaired,
        "recovery_dir": str(recovery.relative_to(root)),
    }
    state.setdefault("meta", {}).setdefault("manual_repairs", []).append(event)
    temporary = state_path.with_suffix(state_path.suffix + ".repair.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(state_path)
    (recovery / "repair_event.json").write_text(
        json.dumps(event, indent=2) + "\n"
    )
    print(json.dumps(event, indent=2))


if __name__ == "__main__":
    main()
