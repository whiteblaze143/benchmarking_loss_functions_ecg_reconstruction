#!/usr/bin/env python3
"""Wait for an upstream experiment queue, then exec a dependent queue manager.

This process is intentionally only a cross-queue handoff.  Each queue remains
owned by its normal queue manager; this script never polls or launches
individual experiment jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def upstream_ready(
    state_path: Path,
    *,
    expected_jobs: int,
    required_output: Path,
) -> tuple[bool, str]:
    state = load_json(state_path)
    if state is None:
        return False, f"upstream state unavailable: {state_path}"

    jobs = state.get("jobs")
    if not isinstance(jobs, list):
        return False, "upstream state has no job list"
    if len(jobs) != expected_jobs:
        return False, f"upstream job count {len(jobs)}/{expected_jobs}"

    counts: dict[str, int] = {}
    for job in jobs:
        status = str(job.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    if counts.get("completed", 0) != expected_jobs:
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        return False, f"upstream incomplete ({summary})"

    if not required_output.is_file() or required_output.stat().st_size == 0:
        return False, f"required upstream output missing: {required_output}"
    return True, f"upstream complete ({expected_jobs}/{expected_jobs})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-state", type=Path, required=True)
    parser.add_argument("--expected-jobs", type=int, required=True)
    parser.add_argument("--required-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--queue-manager", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_jobs <= 0:
        raise SystemExit("--expected-jobs must be positive")
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")

    last_reason: str | None = None
    while True:
        ready, reason = upstream_ready(
            args.upstream_state,
            expected_jobs=args.expected_jobs,
            required_output=args.required_output,
        )
        if reason != last_reason:
            print(f"[dependent-queue] {reason}", flush=True)
            last_reason = reason
        if ready:
            break
        time.sleep(args.poll_seconds)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(args.queue_manager),
        "--manifest",
        str(args.manifest),
        "--state",
        str(args.state),
        "--log-dir",
        str(args.log_dir),
    ]
    print(f"[dependent-queue] launching: {' '.join(command)}", flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
