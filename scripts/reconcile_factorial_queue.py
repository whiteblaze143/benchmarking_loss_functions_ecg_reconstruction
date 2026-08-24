#!/usr/bin/env python3
"""Mark independently verified factorial artifacts complete in a stopped queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


VERIFIED_JOB_IDS = (
    "analyze_factorial",
    "echonext_preflight",
    "evaluate_smartwatch_factorial",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate(project_root: Path) -> None:
    results = project_root / "results/factorial_v2"
    statistics = json.loads((results / "statistics.json").read_text())
    if len(statistics.get("effects", [])) != 147:
        raise ValueError("Expected 147 prespecified three-factor effect rows")
    if len(statistics.get("supplemental_effects", [])) != 315:
        raise ValueError("Expected 315 supplemental four-factor effect rows")
    if len(statistics.get("ecgfounder_noninferiority", [])) != 48:
        raise ValueError("Expected 48 ECGFounder non-inferiority rows")
    if len(statistics.get("confirmation_runs", [])) != 18:
        raise ValueError("Expected 18 confirmation rows")

    preflight = json.loads((results / "echonext_preflight.json").read_text())
    if preflight.get("status") != "pass":
        raise ValueError("EchoNext preflight did not pass")

    registry = json.loads(
        (project_root / "experiment_queue/factorial_v2/model_registry.json").read_text()
    )
    expected = {model["id"] for model in registry["models"]}
    smartwatch = json.loads((results / "smartwatch_results.json").read_text())
    if set(smartwatch.get("models", {})) != expected:
        raise ValueError("Smartwatch model IDs do not match the registry")
    if smartwatch.get("protocol", {}).get("conditions") != ["clean"]:
        raise ValueError("Smartwatch artifact is not the repaired clean-only protocol")
    if not all(
        set(conditions) == {"clean"}
        for devices in smartwatch["models"].values()
        for conditions in devices.values()
    ):
        raise ValueError("Smartwatch artifact contains unvalidated stress labels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    python_processes = subprocess.run(
        ["pgrep", "-x", "python"], capture_output=True, text=True
    )
    manager_pids = []
    for raw_pid in python_processes.stdout.split():
        command_path = Path("/proc") / raw_pid / "cmdline"
        command = command_path.read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        ) if command_path.exists() else ""
        if command.strip().endswith(
            ".agents/skills/experiment-queue/scripts/queue_manager.py "
            "--manifest experiment_queue/factorial_v2/manifest.json "
            "--state experiment_queue/factorial_v2/queue_state.json "
            "--log-dir experiment_queue/factorial_v2/logs"
        ):
            manager_pids.append(raw_pid)
    if manager_pids:
        raise RuntimeError(
            f"Refusing to edit queue state while queue manager PIDs run: {manager_pids}"
        )

    validate(args.project_root)
    state = json.loads(args.state.read_text())
    manifest = json.loads(args.manifest.read_text())

    phase_specs = {
        phase["name"]: phase.get("depends_on", []) for phase in manifest["phases"]
    }
    existing_phases = {phase["name"]: phase for phase in state["phases"]}
    for name, dependencies in phase_specs.items():
        phase = existing_phases.get(name)
        if phase is None:
            phase = {"name": name, "depends_on": dependencies, "status": "pending"}
            state["phases"].append(phase)
            existing_phases[name] = phase
        else:
            phase["depends_on"] = dependencies

    verified_phases: set[str] = set()
    for job in state["jobs"]:
        if job["id"] not in VERIFIED_JOB_IDS:
            continue
        job.update({
            "status": "completed",
            "gpu": None,
            "screen_name": None,
            "pid": None,
            "completed": now(),
            "error": None,
        })
        verified_phases.add(job["phase"])

    for phase_name in verified_phases:
        jobs = [job for job in state["jobs"] if job["phase"] == phase_name]
        if jobs and all(job["status"] == "completed" for job in jobs):
            existing_phases[phase_name]["status"] = "completed"

    temporary = args.state.with_suffix(args.state.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, args.state)
    print(
        json.dumps(
            {
                "verified_jobs": list(VERIFIED_JOB_IDS),
                "verified_phases": sorted(verified_phases),
                "phase_count": len(state["phases"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
