#!/usr/bin/env python3
"""Validation-only checkpoint/state selection followed by locked LUDB testing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42"
EVALUATOR = ROOT / "scripts/evaluate_semiseg_ludb.py"
OUTPUT_DB = ROOT / "results/semiseg_ludb_training/semiseg_ludb_mt_evaluation.sqlite"
SUMMARY = RUN_DIR / "finalization_summary.json"


def run_evaluation(checkpoint: Path, split: str, state: str, padding: str, tolerance: float) -> dict:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--checkpoint", str(checkpoint),
        "--output-db", str(OUTPUT_DB),
        "--split", split,
        "--state", state,
        "--padding", padding,
        "--evaluation-window", "annotated",
        "--device", "cpu",
        "--batch-size", "64",
        "--tolerance-ms", str(tolerance),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=True, text=True, capture_output=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"evaluator emitted no JSON: {completed.stdout}\n{completed.stderr}")
    return json.loads(lines[-1])


def training_rows() -> list[dict]:
    path = RUN_DIR / "log.txt"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-epochs", type=int, default=100)
    args = parser.parse_args()
    rows = training_rows()
    observed_epochs = sorted({int(row["epoch"]) for row in rows})
    if observed_epochs != list(range(args.expected_epochs)):
        raise RuntimeError(f"training incomplete: observed epochs {observed_epochs[:3]}...{observed_epochs[-3:]}")

    candidates = []
    seen_hashes = set()
    import hashlib
    for criterion in ("best-MeanIoU.pth", "best-loss.pth"):
        checkpoint = RUN_DIR / criterion
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint_hash in seen_hashes:
            continue
        seen_hashes.add(checkpoint_hash)
        for state in ("model", "model_ema"):
            result = run_evaluation(checkpoint, "valid", state, "none", 150.0)
            candidates.append({
                "criterion": criterion,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "state": state,
                "validation": result,
            })
    # The paper selects by validation mIoU. Exact ties default to the student.
    selected = max(
        candidates,
        key=lambda item: (item["validation"]["miou_including_background"], item["state"] == "model"),
    )
    checkpoint = Path(selected["checkpoint"])
    state = selected["state"]
    test_clinical = run_evaluation(checkpoint, "test", state, "none", 150.0)
    test_20ms = run_evaluation(checkpoint, "test", state, "none", 20.0)
    test_notebook = run_evaluation(checkpoint, "test", state, "notebook", 150.0)
    summary = {
        "protocol": "semiseg_ludb_mt_finalization_v1",
        "selection_rule": "maximum official-validation mIoU including background; exact tie defaults to student",
        "training_epochs": len(rows),
        "best_logged_student_validation": max(rows, key=lambda row: row["MeanIoU"]),
        "candidates": candidates,
        "selected": {
            "criterion": selected["criterion"],
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "state": state,
            "validation_miou": selected["validation"]["miou_including_background"],
        },
        "test_untouched_input_annotated_window_150ms": test_clinical,
        "test_untouched_input_annotated_window_20ms": test_20ms,
        "test_author_notebook_padding_parity": test_notebook,
    }
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(SUMMARY)
    print(json.dumps({"event": "SEMISEG_LUDB_MT_FINALIZED", "summary": str(SUMMARY), **summary["selected"]}), flush=True)


if __name__ == "__main__":
    main()
