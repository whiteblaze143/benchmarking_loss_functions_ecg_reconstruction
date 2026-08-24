#!/usr/bin/env python3
"""Verify repaired MultiScale-VAE MSE-off checkpoints before 2^4 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
MASK_SUFFIXES = [f"{value:03b}" for value in range(8)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no nonempty model state")
    return state


def tensor_difference(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = state_dict(torch.load(left_path, map_location="cpu", weights_only=False))
    right = state_dict(torch.load(right_path, map_location="cpu", weights_only=False))
    if set(left) != set(right):
        raise ValueError(f"State keys differ: {left_path} vs {right_path}")
    maximum = 0.0
    changed = 0
    compared = 0
    for key in left:
        if not torch.is_tensor(left[key]) or not torch.is_tensor(right[key]):
            continue
        compared += 1
        difference = float((left[key] - right[key]).abs().max().item())
        if not math.isfinite(difference):
            raise ValueError(f"Non-finite tensor difference for {key}")
        maximum = max(maximum, difference)
        changed += int(difference > 0)
    return {
        "state_tensors_compared": compared,
        "changed_state_tensors": changed,
        "maximum_absolute_tensor_difference": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints/factorial_v4/primary",
    )
    parser.add_argument("--queue-state", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queue = json.loads(args.queue_state.read_text())
    manifest = json.loads(args.manifest.read_text())
    repair_jobs = [
        job for job in queue["jobs"] if job["id"].startswith("repair_msvae__")
    ]
    queue_complete = len(repair_jobs) == 8 and all(
        job["status"] == "completed" for job in repair_jobs
    )
    command_by_id = {
        job["id"]: job["cmd"]
        for phase in manifest["phases"]
        for job in phase["jobs"]
        if job["id"].startswith("repair_msvae__")
    }

    pairs = []
    for suffix in MASK_SUFFIXES:
        off_id = f"msvae__e0c{suffix[0]}m{suffix[1]}d{suffix[2]}__s42"
        on_id = f"msvae__e1c{suffix[0]}m{suffix[1]}d{suffix[2]}__s42"
        off_path = args.checkpoint_root / off_id / "ul_ecp_best.pt"
        on_path = args.checkpoint_root / on_id / "ul_ecp_best.pt"
        if not off_path.exists() or not on_path.exists():
            pairs.append({
                "mse_off_id": off_id,
                "mse_on_id": on_id,
                "status": "missing_checkpoint",
            })
            continue
        comparison = tensor_difference(off_path, on_path)
        job_id = f"repair_{off_id}"
        command = command_by_id.get(job_id, "")
        valid = (
            comparison["changed_state_tensors"] > 0
            and "--lambda_mse 0.0" in command
        )
        pairs.append({
            "mse_off_id": off_id,
            "mse_on_id": on_id,
            "status": "pass" if valid else "fail",
            "mse_off_checkpoint_sha256": sha256(off_path),
            "mse_on_checkpoint_sha256": sha256(on_path),
            "training_command": command,
            **comparison,
        })

    source_paths = [
        ROOT / "unified_latents/engineering/trainers/train_multi_scale_vae.py",
        ROOT / "unified_latents/engineering/experimental/Multi_Scale_VAE.py",
        ROOT / "scripts/common_loss.py",
        ROOT / "tests/test_factorial_losses.py",
    ]
    status = (
        "PASS"
        if queue_complete
        and len(pairs) == 8
        and all(pair.get("status") == "pass" for pair in pairs)
        else "FAIL"
    )
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "repaired MultiScale-VAE MSE-off cells for complete 2^4 analysis",
        "queue_complete": queue_complete,
        "gradient_test": (
            "tests/test_factorial_losses.py::"
            "test_msvae_mse_toggle_changes_reconstruction_gradient"
        ),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"{status}: {sum(pair.get('status') == 'pass' for pair in pairs)}/8 pairs")
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
