#!/usr/bin/env python3
"""Build generation-bound convergence diagnostics from compatible training logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOSS_RE = re.compile(
    r"^(Train|Val) Loss:\s*([0-9.eE+-]+)\s*"
    r"\(MSE:\s*([0-9.eE+-]+),\s*MMD:\s*([0-9.eE+-]+),\s*ED:\s*([0-9.eE+-]+)\)\s*$",
    re.MULTILINE,
)


def decode_factorial_mask(mask: str) -> dict[str, int]:
    """Decode and validate the mixed-level seven-position loss identity."""
    if re.fullmatch(r"1[01]{5}[0-4]", mask) is None:
        raise ValueError(
            f"Invalid factorial mask {mask!r}; expected 1 + five binary digits + kernel 0-4"
        )
    return {
        "mse_active": 1,
        "correlation_active": int(mask[1]),
        "derivative_active": int(mask[2]),
        "vcg_active": int(mask[3]),
        "energy_distance_active": int(mask[4]),
        "lead_consistency_active": int(mask[5]),
        "mmd_kernel": int(mask[6]),
    }


def build_controlled_kernel_contrasts(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Contrast kernel levels against level 0 within seed and binary prefix."""
    frame = model_frame.copy()
    frame["binary_prefix"] = frame.factorial_mask.astype(str).str.zfill(7).str[:6]
    rows = []
    for (seed, binary_prefix), group in frame.groupby(
        ["seed", "binary_prefix"], sort=True
    ):
        if group.mmd_kernel.duplicated().any():
            raise RuntimeError(
                f"Duplicate kernel level for seed={seed}, binary_prefix={binary_prefix}"
            )
        baseline = group.loc[group.mmd_kernel.eq(0)]
        if len(baseline) != 1 or len(group) < 2:
            continue
        baseline = baseline.iloc[0]
        for candidate in group.loc[~group.mmd_kernel.eq(0)].itertuples(index=False):
            rows.append({
                "seed": int(seed),
                "binary_prefix": binary_prefix,
                "baseline_model_id": baseline.model_id,
                "candidate_model_id": candidate.model_id,
                "candidate_kernel": int(candidate.mmd_kernel),
                "baseline_first_val_mse": float(baseline.first_val_mse),
                "candidate_first_val_mse": float(candidate.first_val_mse),
                "delta_first_val_mse": float(
                    candidate.first_val_mse - baseline.first_val_mse
                ),
                "baseline_last_val_mse": float(baseline.last_val_mse),
                "candidate_last_val_mse": float(candidate.last_val_mse),
                "delta_last_val_mse": float(
                    candidate.last_val_mse - baseline.last_val_mse
                ),
                "baseline_duration_minutes": float(baseline.duration_seconds / 60),
                "candidate_duration_minutes": float(candidate.duration_seconds / 60),
                "delta_duration_minutes": float(
                    (candidate.duration_seconds - baseline.duration_seconds) / 60
                ),
                "baseline_checkpoint_sha256": baseline.checkpoint_sha256,
                "candidate_checkpoint_sha256": candidate.checkpoint_sha256,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["seed", "binary_prefix", "candidate_kernel"])


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "results/checkpoint_store/compatibility_audit.json",
    )
    parser.add_argument(
        "--queue-state",
        type=Path,
        default=ROOT / "refine-logs/queue/queue_state.json",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "refine-logs/queue/logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/factorial_mixed_level/training_diagnostics",
    )
    return parser.parse_args()


def parse_time(value: str | None):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> None:
    args = arguments()
    audit = json.loads(args.audit.read_text())
    state = json.loads(args.queue_state.read_text())
    jobs = {job["id"]: job for job in state["jobs"]}
    compatible = [model for model in audit["models"] if model.get("compatible")]
    rows = []
    model_rows = []

    for model in compatible:
        model_id = model["model_id"]
        mask = str(model["factorial_mask"])
        factors = decode_factorial_mask(mask)
        log_path = args.log_dir / f"{model_id}.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"Compatible model has no training log: {log_path}")
        content = log_path.read_text(errors="replace")
        matches = LOSS_RE.findall(content)
        train = [tuple(map(float, values)) for split, *values in matches if split == "Train"]
        validation = [tuple(map(float, values)) for split, *values in matches if split == "Val"]
        if len(train) != len(validation) or not train:
            raise RuntimeError(
                f"{model_id}: expected paired train/validation epochs, got "
                f"{len(train)}/{len(validation)}"
            )
        job = jobs[model_id]
        started = parse_time(job.get("started"))
        completed = parse_time(job.get("completed"))
        duration_seconds = (
            (completed - started).total_seconds()
            if started is not None and completed is not None
            else None
        )
        common = {
            "model_id": model_id,
            "factorial_mask": mask,
            "seed": int(model["seed"]),
            **factors,
            "checkpoint_sha256": model["checkpoint_sha256"],
            "checkpoint_size_bytes": int(model["checkpoint_size_bytes"]),
            "source_bundle_sha256": model["source_bundle_sha256"],
            "training_contract_id": audit["contract"]["contract_id"],
            "batch_size": 1024,
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "duration_seconds": duration_seconds,
        }
        for epoch, (train_values, val_values) in enumerate(
            zip(train, validation, strict=True), start=1
        ):
            rows.append({
                **common,
                "epoch": epoch,
                "train_total": train_values[0],
                "train_mse": train_values[1],
                "train_mmd": train_values[2],
                "train_ed": train_values[3],
                "val_total": val_values[0],
                "val_mse": val_values[1],
                "val_mmd": val_values[2],
                "val_ed": val_values[3],
            })
        model_rows.append({
            **common,
            "epochs": len(train),
            "first_val_mse": validation[0][1],
            "last_val_mse": validation[-1][1],
            "val_mse_relative_change": validation[-1][1] / validation[0][1] - 1,
            "first_val_total": validation[0][0],
            "last_val_total": validation[-1][0],
            "cuda_oom_mentions": content.lower().count("out of memory"),
            "traceback_mentions": content.count("Traceback"),
            "nonfinite_mentions": len(
                re.findall(r"\b(?:nan|inf|infinite|non-finite)\b", content.lower())
            ),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    epoch_frame = pd.DataFrame(rows).sort_values(["model_id", "epoch"])
    model_frame = pd.DataFrame(model_rows).sort_values("model_id")
    controlled_kernel_frame = build_controlled_kernel_contrasts(model_frame)
    scale_frame = (
        model_frame.groupby("energy_distance_active", as_index=False)
        .agg(
            compatible_models=("model_id", "size"),
            median_first_val_total=("first_val_total", "median"),
            minimum_first_val_total=("first_val_total", "min"),
            maximum_first_val_total=("first_val_total", "max"),
            median_first_val_mse=("first_val_mse", "median"),
            median_last_val_mse=("last_val_mse", "median"),
        )
        .sort_values("energy_distance_active")
    )
    generated_at = datetime.now(timezone.utc)
    eta_rows = []
    global_duration = model_frame.duration_seconds.dropna()
    if global_duration.empty:
        raise RuntimeError("No compatible completed duration is available for ETA")
    global_median_minutes = float(global_duration.median() / 60)
    for kernel in range(5):
        sample = model_frame.loc[
            model_frame.mmd_kernel.eq(kernel), "duration_seconds"
        ].dropna() / 60
        kernel_min = float(sample.min()) if not sample.empty else global_median_minutes
        kernel_median = (
            float(sample.median()) if not sample.empty else global_median_minutes
        )
        kernel_max = float(sample.max()) if not sample.empty else global_median_minutes
        remaining_jobs = [
            job for job in state["jobs"]
            if int(re.search(r"--factorial_mask\s+(\d{7})", job["cmd"]).group(1)[-1])
            == kernel
            and job.get("status") in {"pending", "running"}
        ]
        pending_jobs = sum(job.get("status") == "pending" for job in remaining_jobs)
        running_jobs = [job for job in remaining_jobs if job.get("status") == "running"]
        running_elapsed_minutes = sum(
            max((generated_at - parse_time(job.get("started"))).total_seconds() / 60, 0)
            for job in running_jobs
            if parse_time(job.get("started")) is not None
        )
        eta_rows.append({
            "mmd_kernel": kernel,
            "compatible_duration_samples": len(sample),
            "observed_min_minutes": kernel_min,
            "observed_median_minutes": kernel_median,
            "observed_max_minutes": kernel_max,
            "pending_jobs": pending_jobs,
            "running_jobs": len(running_jobs),
            "running_elapsed_minutes": running_elapsed_minutes,
            "estimated_remaining_minutes": (
                pending_jobs * kernel_median
                + sum(max(kernel_median - (
                    (generated_at - parse_time(job.get("started"))).total_seconds() / 60
                    if parse_time(job.get("started")) is not None else 0
                ), 0) for job in running_jobs)
            ),
            "observed_min_scenario_minutes": (
                pending_jobs * kernel_min
                + sum(max(kernel_min - (
                    (generated_at - parse_time(job.get("started"))).total_seconds() / 60
                    if parse_time(job.get("started")) is not None else 0
                ), 0) for job in running_jobs)
            ),
            "observed_max_scenario_minutes": (
                pending_jobs * kernel_max
                + sum(max(kernel_max - (
                    (generated_at - parse_time(job.get("started"))).total_seconds() / 60
                    if parse_time(job.get("started")) is not None else 0
                ), 0) for job in running_jobs)
            ),
            "fallback_to_global_median": sample.empty,
        })
    eta_frame = pd.DataFrame(eta_rows)

    for name, frame in (
        ("compatible_epoch_curves.csv", epoch_frame),
        ("compatible_model_summary.csv", model_frame),
        ("optimization_scale_by_energy_distance.csv", scale_frame),
        ("controlled_kernel_training_contrasts.csv", controlled_kernel_frame),
        ("operational_eta_by_kernel.csv", eta_frame),
    ):
        destination = args.output_dir / name
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)

    durations = model_frame.duration_seconds.dropna()
    eta_minutes = float(eta_frame.estimated_remaining_minutes.sum())
    eta_min_scenario = float(eta_frame.observed_min_scenario_minutes.sum())
    eta_max_scenario = float(eta_frame.observed_max_scenario_minutes.sum())
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_contract_id": audit["contract"]["contract_id"],
        "source_bundle_sha256": audit["contract"]["approved_source_bundle_sha256"],
        "builder_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compatible_models": len(model_frame),
        "epochs": len(epoch_frame),
        "epoch_counts": dict(Counter(model_frame.epochs.astype(int))),
        "duration_minutes": {
            "minimum": float(durations.min() / 60),
            "median": float(durations.median() / 60),
            "maximum": float(durations.max() / 60),
        },
        "operational_eta": {
            "method": "serial_single_GPU_kernel_specific_observed_duration_medians",
            "remaining_pending_or_running_jobs": int(
                eta_frame[["pending_jobs", "running_jobs"]].to_numpy().sum()
            ),
            "estimated_remaining_hours": eta_minutes / 60,
            "estimated_remaining_days": eta_minutes / 1440,
            "estimated_completion_utc": (
                generated_at + timedelta(minutes=eta_minutes)
            ).isoformat(),
            "observed_min_scenario_hours": eta_min_scenario / 60,
            "observed_max_scenario_hours": eta_max_scenario / 60,
            "limitations": [
                "only compatible completed runs contribute duration samples",
                "kernel-specific samples remain sparse and other loss bits may change runtime",
                "the estimate assumes uninterrupted serial execution on the current hardware",
                "the min/max scenarios are observed-run bounds, not statistical confidence intervals",
            ],
        },
        "models_with_cuda_oom": int((model_frame.cuda_oom_mentions > 0).sum()),
        "models_with_traceback": int((model_frame.traceback_mentions > 0).sum()),
        "models_with_nonfinite_mentions": int((model_frame.nonfinite_mentions > 0).sum()),
        "files": {
            "compatible_epoch_curves.csv": hashlib.sha256(
                (args.output_dir / "compatible_epoch_curves.csv").read_bytes()
            ).hexdigest(),
            "compatible_model_summary.csv": hashlib.sha256(
                (args.output_dir / "compatible_model_summary.csv").read_bytes()
            ).hexdigest(),
            "optimization_scale_by_energy_distance.csv": hashlib.sha256(
                (
                    args.output_dir / "optimization_scale_by_energy_distance.csv"
                ).read_bytes()
            ).hexdigest(),
            "controlled_kernel_training_contrasts.csv": hashlib.sha256(
                (
                    args.output_dir / "controlled_kernel_training_contrasts.csv"
                ).read_bytes()
            ).hexdigest(),
            "operational_eta_by_kernel.csv": hashlib.sha256(
                (args.output_dir / "operational_eta_by_kernel.csv").read_bytes()
            ).hexdigest(),
        },
    }
    atomic_text(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
