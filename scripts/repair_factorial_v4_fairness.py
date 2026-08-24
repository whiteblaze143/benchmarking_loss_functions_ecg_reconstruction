#!/usr/bin/env python3
"""Recompute subgroup gaps on identical task sets from saved per-record outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_comprehensive_registry import subgroup_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    registry = json.loads(args.registry.read_text())
    project_root = Path(__file__).resolve().parents[1]
    tasks_path = project_root / registry["ecgfounder_tasks"]
    class_names = [
        line.strip() for line in tasks_path.read_text().splitlines() if line.strip()
    ]
    repaired = 0
    for model in payload["models"].values():
        frame = pd.read_parquet(
            project_root / model["ecgfounder_per_record_parquet"],
            columns=["probabilities", "labels", "age", "sex"],
        )
        model["fairness"] = subgroup_metrics(
            np.stack(frame["probabilities"].to_numpy()),
            np.stack(frame["labels"].to_numpy()),
            frame["age"].to_numpy(dtype=float),
            frame["sex"].to_numpy(),
            class_names,
        )
        repaired += 1
    payload.setdefault("repairs", {})["fairness_common_task_sets"] = {
        "status": "applied",
        "models": repaired,
        "method": (
            "Each sex/age gap compares subgroup macro AUROCs over the exact "
            "intersection of tasks containing both classes in every compared group."
        ),
    }
    temporary = args.results.with_suffix(args.results.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(args.results)
    print(json.dumps(payload["repairs"]["fairness_common_task_sets"], indent=2))


if __name__ == "__main__":
    main()
