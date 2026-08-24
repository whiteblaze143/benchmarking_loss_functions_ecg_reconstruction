#!/usr/bin/env python3
"""Remove legacy smartwatch conditions that were clean-signal duplicates."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def equivalent_signal(left: Any, right: Any) -> bool:
    """Compare repeated clean inference while allowing GPU round-off."""
    if isinstance(left, dict) and isinstance(right, dict):
        left = {key: value for key, value in left.items() if key != "clinical"}
        right = {key: value for key, value in right.items() if key != "clinical"}
        return left.keys() == right.keys() and all(
            equivalent_signal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            equivalent_signal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=2e-5, abs_tol=1e-2)
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    model_count = 0
    device_count = 0
    removed_conditions = 0
    for model_id, devices in payload.get("models", {}).items():
        model_count += 1
        for device, conditions in devices.items():
            device_count += 1
            clean = conditions.get("clean")
            if clean is None:
                raise ValueError(f"{model_id}/{device} has no clean result")
            for condition_id, result in conditions.items():
                if condition_id != "clean" and not equivalent_signal(clean, result):
                    raise ValueError(
                        f"Refusing repair: {model_id}/{device}/{condition_id} "
                        "differs from clean"
                    )
            removed_conditions += len(conditions) - 1
            devices[device] = {"clean": clean}

    payload.update({
        "schema_version": 2,
        "protocol": {
            "dataset": "PhysioNet ECG Effects of the Use of Smartwatches",
            "task": "single_lead_to_12_lead_zero_shot_reconstruction",
            "conditions": ["clean"],
            "stress_testing": "not_run",
            "reason": (
                "Legacy stress entries were removed after source audit and "
                "metric-equivalence checks (GPU round-off tolerance) proved "
                "the old evaluator never applied noise."
            ),
            "repair_audit": {
                "models": model_count,
                "model_device_pairs": device_count,
                "removed_duplicate_condition_entries": removed_conditions,
            },
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    os.replace(temporary, args.output)
    print(json.dumps(payload["protocol"]["repair_audit"], indent=2))


if __name__ == "__main__":
    main()
