#!/usr/bin/env python3
"""Combine the live repair and downstream evaluation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = json.loads(args.training.read_text())
    postprocess = json.loads(args.postprocess.read_text())
    training["project"] = "factorial_v4_complete_repaired_2x4"

    existing_phase_names = {phase["name"] for phase in training["phases"]}
    for phase in postprocess["phases"]:
        if phase["name"] in existing_phase_names:
            raise ValueError(f"Duplicate phase name: {phase['name']}")
        phase = dict(phase)
        if phase["name"] == "repair_attestation":
            phase["depends_on"] = ["replace_msvae_mse_off"]
        training["phases"].append(phase)

    job_ids = [
        job["id"]
        for phase in training["phases"]
        for job in phase["jobs"]
    ]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Duplicate job IDs in combined manifest")
    args.output.write_text(json.dumps(training, indent=2, allow_nan=False) + "\n")
    print(
        f"Wrote {len(training['phases'])} phases and {len(job_ids)} jobs "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
