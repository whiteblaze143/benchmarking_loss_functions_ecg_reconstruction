#!/usr/bin/env python3
"""Run every release-compatible factorial model on one real ECG, fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_store import (
    DEFAULT_DB,
    connect,
    load_checkpoint_with_identity,
    prune_cache,
    store_lock,
)
from scripts.infer_factorial_checkpoint import (
    DEFAULT_COMPATIBILITY_AUDIT,
    DEFAULT_TRAINING_CONTRACT,
    compatible_identities,
    prepare_input,
)
from scripts.train_mcma_3lead import MCMAModel


LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MISSING_LEAD_INDICES = (2, 3, 4, 5, 6, 8, 9, 10, 11)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/ptb_xl/tensors/test/100.pt",
    )
    result.add_argument("--db", type=Path, default=DEFAULT_DB)
    result.add_argument(
        "--compatibility-audit",
        type=Path,
        default=DEFAULT_COMPATIBILITY_AUDIT,
    )
    result.add_argument(
        "--training-contract", type=Path, default=DEFAULT_TRAINING_CONTRACT
    )
    result.add_argument(
        "--cache-dir", type=Path, default=ROOT / "checkpoints/inference_audit_cache"
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/factorial_mixed_level/inference_readiness",
    )
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument(
        "--if-stale",
        action="store_true",
        help="Exit without model loading when every bound artifact is current",
    )
    return result


def artifact_is_current(args: argparse.Namespace) -> bool:
    summary_path = args.output_dir / "summary.json"
    model_csv = args.output_dir / "per_model_inference_readiness.csv"
    lead_csv = args.output_dir / "per_model_per_lead_case_metrics.csv"
    try:
        summary = json.loads(summary_path.read_text())
        return (
            summary.get("compatibility_audit_sha256")
            == sha256_file(args.compatibility_audit)
            and summary.get("benchmark_code_sha256") == sha256_file(Path(__file__))
            and summary.get("input_sha256") == sha256_file(args.input)
            and summary.get("csv_sha256") == sha256_file(model_csv)
            and summary.get("per_lead_case_metrics_csv_sha256")
            == sha256_file(lead_csv)
            and summary.get("cache_retained_bytes") == 0
            and summary.get("all_finite") is True
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError):
        return False


def main() -> None:
    args = parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.if_stale and artifact_is_current(args):
        print(json.dumps({"status": "current", "action": "skipped"}))
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ready = compatible_identities(args.compatibility_audit, args.training_contract)
    if not ready:
        raise RuntimeError("No release-compatible models are available")
    signal, original_length = prepare_input(args.input)
    target = torch.load(args.input, map_location="cpu", weights_only=True).float()
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3 or target.shape[1] != 12:
        raise ValueError("Readiness benchmark requires a full 12-lead target tensor")
    target = target[..., :original_length]
    rows = []
    lead_rows = []
    started = datetime.now(timezone.utc)
    try:
        for model_id in sorted(ready):
            load_start = time.perf_counter()
            payload, identity = load_checkpoint_with_identity(
                model_id,
                db_path=args.db,
                cache_dir=args.cache_dir,
                map_location="cpu",
                weights_only=False,
            )
            load_seconds = time.perf_counter() - load_start
            if identity["sha256"] != ready[model_id]["checkpoint_sha256"]:
                raise RuntimeError(f"{model_id} catalog/audit digest mismatch")
            model = MCMAModel(in_channels=3, out_channels=12)
            model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
            model.eval()
            latencies = []
            reconstruction = None
            with torch.inference_mode():
                for _ in range(args.repeats):
                    forward_start = time.perf_counter()
                    reconstruction = model(signal)[..., :original_length]
                    latencies.append(time.perf_counter() - forward_start)
            assert reconstruction is not None
            if reconstruction.shape != (signal.shape[0], 12, original_length):
                raise RuntimeError(
                    f"{model_id} emitted unexpected shape {tuple(reconstruction.shape)}"
                )
            if not torch.isfinite(reconstruction).all():
                raise RuntimeError(f"{model_id} emitted non-finite reconstruction")
            rows.append(
                {
                    "model_id": model_id,
                    "factorial_mask": ready[model_id]["factorial_mask"],
                    "seed": int(ready[model_id]["seed"]),
                    "checkpoint_sha256": identity["sha256"],
                    "checkpoint_size_bytes": int(identity["size_bytes"]),
                    "load_and_materialize_seconds": load_seconds,
                    "forward_median_seconds": float(pd.Series(latencies).median()),
                    "forward_minimum_seconds": min(latencies),
                    "forward_maximum_seconds": max(latencies),
                    "repeats": args.repeats,
                    "output_mean": float(reconstruction.mean()),
                    "output_std": float(reconstruction.std()),
                    "finite": True,
                    "output_shape": "x".join(map(str, reconstruction.shape)),
                }
            )
            for lead_index in MISSING_LEAD_INDICES:
                real = target[:, lead_index].reshape(-1)
                predicted = reconstruction[:, lead_index].reshape(-1)
                real_centered = real - real.mean()
                predicted_centered = predicted - predicted.mean()
                denominator = torch.sqrt(
                    (real_centered.square().sum())
                    * (predicted_centered.square().sum())
                )
                pearson = (
                    float((real_centered * predicted_centered).sum() / denominator)
                    if denominator > 0
                    else float("nan")
                )
                lead_rows.append(
                    {
                        "model_id": model_id,
                        "factorial_mask": ready[model_id]["factorial_mask"],
                        "seed": int(ready[model_id]["seed"]),
                        "checkpoint_sha256": identity["sha256"],
                        "lead_index": lead_index,
                        "lead": LEAD_NAMES[lead_index],
                        "samples": int(real.numel()),
                        "mse": float(torch.mean((predicted - real).square())),
                        "mae": float(torch.mean(torch.abs(predicted - real))),
                        "pearson": pearson,
                        "target_mean": float(real.mean()),
                        "reconstruction_mean": float(predicted.mean()),
                        "target_std": float(real.std()),
                        "reconstruction_std": float(predicted.std()),
                        "variance_ratio": float(
                            predicted.var(unbiased=True) / real.var(unbiased=True)
                        ),
                    }
                )
            del reconstruction, model, payload
            with store_lock(args.db):
                connection = connect(args.db)
                try:
                    prune_cache(connection, args.cache_dir, 0)
                finally:
                    connection.close()
    finally:
        with store_lock(args.db):
            connection = connect(args.db)
            try:
                prune_cache(connection, args.cache_dir, 0)
            finally:
                connection.close()

    frame = pd.DataFrame(rows)
    lead_frame = pd.DataFrame(lead_rows)
    if len(frame) != len(ready) or not frame["finite"].all():
        raise RuntimeError("Inference audit did not complete every compatible model")
    if len(lead_frame) != len(ready) * len(MISSING_LEAD_INDICES):
        raise RuntimeError("Inference audit has incomplete per-lead case metrics")
    metric_columns = ["mse", "mae", "pearson", "variance_ratio"]
    if not torch.isfinite(torch.tensor(lead_frame[metric_columns].to_numpy())).all():
        raise RuntimeError("Inference audit produced non-finite case metrics")
    csv_content = frame.to_csv(index=False)
    csv_path = args.output_dir / "per_model_inference_readiness.csv"
    atomic_text(csv_path, csv_content)
    lead_csv_content = lead_frame.to_csv(index=False)
    lead_csv_path = args.output_dir / "per_model_per_lead_case_metrics.csv"
    atomic_text(lead_csv_path, lead_csv_content)
    completed = datetime.now(timezone.utc)
    contract = json.loads(args.training_contract.read_text())
    summary = {
        "schema_version": 1,
        "status": "complete_for_current_compatible_cohort",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_seconds": (completed - started).total_seconds(),
        "models_expected": len(ready),
        "models_completed": len(frame),
        "all_finite": bool(frame["finite"].all()),
        "device": "cpu",
        "torch_threads": 1,
        "repeats_per_model": args.repeats,
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "prepared_input_shape": list(signal.shape),
        "output_shape": [int(signal.shape[0]), 12, original_length],
        "training_contract_id": contract["contract_id"],
        "approved_source_bundle_sha256": contract[
            "approved_source_bundle_sha256"
        ],
        "compatibility_audit_sha256": sha256_file(args.compatibility_audit),
        "benchmark_code_sha256": sha256_file(Path(__file__)),
        "csv_sha256": hashlib.sha256(csv_content.encode()).hexdigest(),
        "per_lead_case_metrics_csv_sha256": hashlib.sha256(
            lead_csv_content.encode()
        ).hexdigest(),
        "per_lead_case_metric_rows": len(lead_frame),
        "missing_leads": [LEAD_NAMES[index] for index in MISSING_LEAD_INDICES],
        "checkpoint_logical_bytes": int(frame["checkpoint_size_bytes"].sum()),
        "cache_retained_bytes": sum(
            path.stat().st_size for path in args.cache_dir.glob("factorial_*.pt")
        ),
        "forward_median_seconds_across_models": float(
            frame["forward_median_seconds"].median()
        ),
        "load_and_materialize_median_seconds": float(
            frame["load_and_materialize_seconds"].median()
        ),
        "limitations": [
            "single real PTB-XL record",
            "CPU-only operational readiness benchmark",
            "forward timings are not a hardware-normalized performance comparison",
            "cohort covers only checkpoints compatible at the audit timestamp",
        ],
    }
    atomic_text(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
