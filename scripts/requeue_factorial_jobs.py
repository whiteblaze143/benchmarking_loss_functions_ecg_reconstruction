#!/usr/bin/env python3
"""Safely reset a stopped factorial queue's jobs by ID prefix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    live_screens = subprocess.run(
        ["screen", "-ls"], capture_output=True, text=True
    ).stdout
    if f"EQ_{args.prefix}" in live_screens:
        raise RuntimeError(f"Refusing to requeue live jobs:\n{live_screens}")

    state = json.loads(args.state.read_text())
    selected = [job for job in state["jobs"] if job["id"].startswith(args.prefix)]
    if not selected:
        raise ValueError(f"No jobs match prefix {args.prefix!r}")
    affected_phases = {job["phase"] for job in selected}
    for job in selected:
        expected = job.get("expected_output")
        if expected:
            Path(args.project_root, expected).unlink(missing_ok=True)
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
    for phase in state["phases"]:
        if phase["name"] in affected_phases:
            phase["status"] = "pending"

    smoke_root = args.project_root / "checkpoints/factorial_v2/smoke"
    if args.prefix == "smoke_cnvae__":
        for directory in smoke_root.glob("cnvae__*"):
            shutil.rmtree(directory)
    temporary = args.state.with_suffix(args.state.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, args.state)
    print(f"Requeued {len(selected)} jobs across {sorted(affected_phases)}")


if __name__ == "__main__":
    main()
