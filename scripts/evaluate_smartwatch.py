#!/usr/bin/env python3
"""Protocol-safe smartwatch simulator and zero-shot reconstruction benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wfdb
from tqdm import tqdm

from scripts.clinical_metrics import probability_fidelity_metrics
from scripts.ecgfounder_classifier import (
    load_ecgfounder,
    load_task_names,
    preprocess_ecgfounder,
)
from scripts.evaluate_comprehensive_registry import (
    LEAD_NAMES,
    StreamingSignalMetrics,
    json_safe,
    load_adapter,
)
from scripts.smartwatch_protocol import (
    align_watch_to_reference,
    canonical_record_key,
    measure_protocol_signal,
    parse_protocol_target,
    resample_signals,
    validate_single_watch_lead,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICES = [
    "applewatch_serie8",
    "fitbitsense2",
    "samsunggalaxy6",
    "withingsscanwatch",
]
GT_DEVICE = "philips_tc30"
EXPERIMENT_DIRS = ["amp_test", "freq_test", "sqr-2hz", "st-segment"]
TARGET_FS = 500
TARGET_LEN = 5000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))
    os.replace(staging, path)


def discover_records(base_dir: Path, device: str) -> list[Path]:
    records: list[Path] = []
    for experiment in EXPERIMENT_DIRS:
        experiment_dir = base_dir / device / experiment
        if experiment_dir.exists():
            records.extend(
                header.with_suffix("")
                for header in experiment_dir.glob("**/*.hea")
            )
    return sorted(records)


def load_physical_record(path: Path) -> tuple[np.ndarray, int, list[str], list[str]]:
    record = wfdb.rdrecord(str(path))
    if record.p_signal is None:
        raise ValueError(f"WFDB record has no physical signal: {path}")
    units = [str(unit) for unit in record.units]
    if set(units) != {"mV"}:
        raise ValueError(f"Expected physical mV WFDB signals at {path}, found {units}")
    raw = np.asarray(record.p_signal, dtype=np.float64)
    finite_rows = np.isfinite(raw).all(axis=1)
    if not finite_rows.all():
        finite_indices = np.flatnonzero(finite_rows)
        if not finite_indices.size:
            raise ValueError(f"WFDB record has no finite samples: {path}")
        first, last = int(finite_indices[0]), int(finite_indices[-1])
        if not finite_rows[first:last + 1].all():
            raise ValueError(f"WFDB record has interior non-finite samples: {path}")
        raw = raw[first:last + 1]
    values = resample_signals(raw, float(record.fs), TARGET_FS)
    return values, TARGET_FS, list(record.sig_name), units


def prepare_device_datasets(data_dir: Path) -> dict[str, dict[str, Any]]:
    reference_records: dict[str, Path] = {}
    for reference_path in discover_records(data_dir, GT_DEVICE):
        relative = reference_path.relative_to(data_dir / GT_DEVICE)
        key = canonical_record_key(relative)
        if key in reference_records:
            raise ValueError(
                "Ambiguous case-insensitive Philips record key "
                f"{key}: {reference_records[key]} and {reference_path}"
            )
        reference_records[key] = reference_path

    datasets: dict[str, dict[str, Any]] = {}
    for device in DEVICES:
        sparse_inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        unmatched_record_ids: list[str] = []
        for watch_path in discover_records(data_dir, device):
            relative = watch_path.relative_to(data_dir / device)
            reference_path = reference_records.get(canonical_record_key(relative))
            if reference_path is None:
                unmatched_record_ids.append(str(relative))
                continue
            watch, _, watch_names, _ = load_physical_record(watch_path)
            reference, _, reference_names, _ = load_physical_record(reference_path)
            watch_lead_name, watch_lead_index = validate_single_watch_lead(watch_names)
            if reference_names != LEAD_NAMES:
                raise ValueError(
                    f"Philips lead order mismatch at {reference_path}: {reference_names}"
                )
            if watch_lead_name not in reference_names:
                raise ValueError(
                    f"Watch lead {watch_lead_name} missing from Philips reference"
                )
            aligned_watch, aligned_reference, alignment = align_watch_to_reference(
                watch[:, 0],
                reference,
                reference_lead_index=watch_lead_index,
                fs=TARGET_FS,
                target_len=TARGET_LEN,
            )
            sparse = np.zeros((12, TARGET_LEN), dtype=np.float32)
            sparse[watch_lead_index] = aligned_watch
            protocol_target = parse_protocol_target(relative)
            watch_measurement = measure_protocol_signal(
                aligned_watch,
                protocol_target,
                TARGET_FS,
            )
            reference_measurement = measure_protocol_signal(
                aligned_reference[:, watch_lead_index],
                protocol_target,
                TARGET_FS,
            )
            sparse_inputs.append(sparse)
            targets.append(aligned_reference.T.astype(np.float32))
            rows.append(
                {
                    "record_id": str(relative),
                    "device": device,
                    "input_lead": watch_lead_name,
                    "input_lead_index": watch_lead_index,
                    **protocol_target,
                    "watch_measurement": watch_measurement,
                    "reference_measurement": reference_measurement,
                    **alignment,
                }
            )
        if not rows:
            raise ValueError(f"No paired records found for {device}")
        input_leads = {row["input_lead"] for row in rows}
        if input_leads != {"II"}:
            raise ValueError(f"Unexpected smartwatch input leads for {device}: {input_leads}")
        datasets[device] = {
            "inputs": np.stack(sparse_inputs),
            "targets": np.stack(targets),
            "metadata": pd.DataFrame.from_records(rows),
            "pairing_audit": {
                "discovered_watch_records": len(rows) + len(unmatched_record_ids),
                "paired_records": len(rows),
                "unmatched_watch_record_ids": unmatched_record_ids,
                "pairing_key": "casefolded_relative_posix_path",
            },
        }
    return datasets


def aggregate_protocol_ground_truth(metadata: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for experiment_type, group in metadata.groupby("experiment_type", sort=True):
        valid = group["watch_measurement"].notna()
        reference_valid = group["reference_measurement"].notna()
        entry: dict[str, Any] = {
            "n_records": int(len(group)),
            "n_watch_measurements": int(valid.sum()),
            "n_reference_measurements": int(reference_valid.sum()),
            "evaluation_type": (
                "paired_device_reference_under_simulated_2hz_stimulus"
                if experiment_type == "square_wave"
                else "simulation_only_calibrated_protocol"
            ),
        }
        if valid.any():
            error = (
                group.loc[valid, "watch_measurement"].to_numpy(dtype=float)
                - group.loc[valid, "target_value"].to_numpy(dtype=float)
            )
            entry["watch_vs_simulator"] = {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "bias": float(np.mean(error)),
                "unit": str(group["target_unit"].iloc[0]),
            }
        if reference_valid.any():
            error = (
                group.loc[reference_valid, "reference_measurement"].to_numpy(dtype=float)
                - group.loc[reference_valid, "target_value"].to_numpy(dtype=float)
            )
            entry["philips_vs_simulator"] = {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "bias": float(np.mean(error)),
                "unit": str(group["target_unit"].iloc[0]),
            }
        if experiment_type == "square_wave":
            entry["watch_vs_philips_max_aligned_cross_correlation"] = float(
                group["alignment_pearson"].mean()
            )
        output[str(experiment_type)] = entry
    return output


def per_record_reconstruction_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    observed: list[int],
    prefix: str,
) -> list[dict[str, float]]:
    missing = [index for index in range(12) if index not in observed]
    target_missing = target[:, missing].float().cpu().numpy()
    recon_missing = reconstruction[:, missing].float().cpu().numpy()
    rows: list[dict[str, float]] = []
    for truth, prediction in zip(target_missing, recon_missing):
        error = prediction - truth
        correlations: list[float] = []
        r2_values: list[float] = []
        for lead_truth, lead_prediction in zip(truth, prediction):
            if np.std(lead_truth) > 1e-12 and np.std(lead_prediction) > 1e-12:
                correlations.append(float(np.corrcoef(lead_truth, lead_prediction)[0, 1]))
            denominator = float(np.sum((lead_truth - lead_truth.mean()) ** 2))
            if denominator > 1e-12:
                r2_values.append(1.0 - float(np.sum((lead_prediction - lead_truth) ** 2)) / denominator)
        rows.append(
            {
                f"{prefix}_mse": float(np.mean(error ** 2)),
                f"{prefix}_rmse": float(np.sqrt(np.mean(error ** 2))),
                f"{prefix}_mae": float(np.mean(np.abs(error))),
                f"{prefix}_pearson": (
                    float(np.mean(correlations))
                    if correlations
                    else float("nan")
                ),
                f"{prefix}_r2": (
                    float(np.mean(r2_values))
                    if r2_values
                    else float("nan")
                ),
            }
        )
    return rows


@torch.inference_mode()
def classifier_probabilities(
    classifier: torch.nn.Module,
    signals: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tensor = torch.as_tensor(signals, dtype=torch.float32)
    probabilities: list[np.ndarray] = []
    for start in range(0, tensor.shape[0], batch_size):
        batch = tensor[start:start + batch_size].to(device)
        probabilities.append(
            torch.sigmoid(classifier(preprocess_ecgfounder(batch))).cpu().numpy()
        )
    return np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/physionet.org/files/ecg-capable-smartwatches/1.0.0"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse complete model cells from <output>.partial.",
    )
    parser.add_argument(
        "--robustness_samples",
        type=int,
        default=0,
        help="Must remain zero; no validated extra smartwatch stress protocol exists.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.robustness_samples:
        raise ValueError(
            "Smartwatch robustness is not implemented; simulator conditions are "
            "analyzed as protocol strata, not mislabeled as injected noise."
        )

    registry_path = Path(args.registry)
    registry_path = registry_path if registry_path.is_absolute() else PROJECT_ROOT / registry_path
    data_dir = args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    per_record_dir = output.parent / "smartwatch_per_record"
    per_record_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(registry_path.read_text())
    if not registry.get("models"):
        raise ValueError("Registry contains no models")
    missing = [
        spec["checkpoint"]
        for spec in registry["models"]
        if not (PROJECT_ROOT / spec["checkpoint"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing registered checkpoints: {missing}")

    device = torch.device(args.device)
    tasks = load_task_names(PROJECT_ROOT / registry["ecgfounder_tasks"])
    classifier = load_ecgfounder(
        PROJECT_ROOT / registry["ecgfounder_repo"],
        PROJECT_ROOT / registry["ecgfounder_checkpoint"],
        device,
        len(tasks),
    )
    datasets = prepare_device_datasets(data_dir)
    for dataset in datasets.values():
        dataset["reference_probabilities"] = classifier_probabilities(
            classifier,
            dataset["targets"],
            device,
            args.batch_size,
        )

    payload: dict[str, Any] = {
        "schema_version": 3,
        "models": {},
        "device_protocol_ground_truth": {
            name: aggregate_protocol_ground_truth(dataset["metadata"])
            for name, dataset in datasets.items()
        },
    }
    temporary = output.with_suffix(output.suffix + ".partial")
    if args.resume and temporary.is_file():
        resumed = json.loads(temporary.read_text())
        registered_ids = {spec["id"] for spec in registry["models"]}
        unexpected = set(resumed.get("models", {})) - registered_ids
        if unexpected:
            raise ValueError(
                f"Partial smartwatch output has unregistered models: {sorted(unexpected)}"
            )
        payload["models"] = resumed.get("models", {})
        print(f"Resuming smartwatch evaluation from {len(payload['models'])} models")

    for spec in tqdm(registry["models"], desc="Smartwatch models"):
        existing = payload["models"].get(spec["id"])
        if existing is not None:
            complete = set(existing) == set(DEVICES)
            for device_name in DEVICES:
                result = existing.get(device_name, {})
                parquet_path = PROJECT_ROOT / result.get(
                    "per_record_parquet", "__missing__"
                )
                complete &= (
                    result.get("n_paired_records")
                    == len(datasets[device_name]["metadata"])
                    and result.get("ecgfounder_probability_fidelity", {}).get(
                        "n_tasks"
                    )
                    == len(tasks)
                    and parquet_path.is_file()
                    and len(pd.read_parquet(parquet_path, columns=["record_id"]))
                    == len(datasets[device_name]["metadata"])
                )
            if complete:
                print(f"Resume: verified and skipped {spec['id']}")
                continue
            print(f"Resume: recomputing incomplete cell {spec['id']}")
            payload["models"].pop(spec["id"], None)

        adapter = load_adapter(spec, device)
        model_result: dict[str, Any] = {}
        for device_name, dataset in datasets.items():
            actual_observed = [LEAD_NAMES.index("II")]
            streaming_nine = StreamingSignalMetrics(adapter.observed)
            streaming_eleven = StreamingSignalMetrics(actual_observed)
            reconstruction_probabilities: list[np.ndarray] = []
            metric_rows: list[dict[str, Any]] = []
            inputs = dataset["inputs"]
            targets = dataset["targets"]
            metadata = dataset["metadata"]
            for start in range(0, len(inputs), args.batch_size):
                stop = min(start + args.batch_size, len(inputs))
                input_batch = torch.from_numpy(inputs[start:stop]).to(device)
                target_batch = torch.from_numpy(targets[start:stop]).to(device)
                reconstruction = adapter.reconstruct(
                    input_batch,
                    preserve_observed=actual_observed,
                )
                if reconstruction.shape != target_batch.shape or not torch.isfinite(reconstruction).all():
                    raise RuntimeError(f"Invalid smartwatch reconstruction from {spec['id']}")
                if not torch.equal(
                    reconstruction[:, actual_observed],
                    input_batch[:, actual_observed],
                ):
                    raise RuntimeError(
                        f"Acquired lead II was not preserved by {spec['id']}"
                    )
                streaming_nine.update(target_batch, reconstruction)
                streaming_eleven.update(target_batch, reconstruction)
                nine_metrics = per_record_reconstruction_metrics(
                    target_batch,
                    reconstruction,
                    adapter.observed,
                    "trained_target_nine",
                )
                eleven_metrics = per_record_reconstruction_metrics(
                    target_batch,
                    reconstruction,
                    actual_observed,
                    "wearable_missing_eleven",
                )
                for offset, (nine_values, eleven_values) in enumerate(
                    zip(nine_metrics, eleven_metrics)
                ):
                    row = metadata.iloc[start + offset].to_dict()
                    row.update(nine_values)
                    row.update(eleven_values)
                    metric_rows.append(row)
                reconstruction_probabilities.append(
                    classifier_probabilities(
                        classifier,
                        reconstruction,
                        device,
                        args.batch_size,
                    )
                )
            reconstructed_probs = np.concatenate(reconstruction_probabilities)
            reference_probs = dataset["reference_probabilities"]
            frame = pd.DataFrame.from_records(metric_rows)
            frame["ecgfounder_reconstruction_probabilities"] = list(
                reconstructed_probs.astype(np.float32)
            )
            frame["ecgfounder_reference_probabilities"] = list(
                reference_probs.astype(np.float32)
            )
            parquet_path = (
                per_record_dir / f"{spec['id']}__{device_name}.parquet"
            )
            frame.to_parquet(parquet_path, index=False)
            model_result[device_name] = {
                "signal": streaming_nine.finalize(),
                "signal_trained_target_nine": streaming_nine.finalize(),
                "signal_wearable_missing_eleven": streaming_eleven.finalize(),
                "n_paired_records": len(frame),
                "input_leads_available": ["II"],
                "model_observed_lead_contract": [
                    LEAD_NAMES[index] for index in adapter.observed
                ],
                "zero_filled_model_input_channels": [
                    LEAD_NAMES[index]
                    for index in adapter.observed
                    if index != LEAD_NAMES.index("II")
                ],
                "reconstruction_passthrough_leads": ["II"],
                "ecgfounder_probability_fidelity": probability_fidelity_metrics(
                    reconstructed_probs,
                    reference_probs,
                    tasks,
                ),
                "per_record_parquet": str(
                    parquet_path.relative_to(PROJECT_ROOT)
                    if parquet_path.is_relative_to(PROJECT_ROOT)
                    else parquet_path
                ),
            }
        payload["models"][spec["id"]] = model_result
        write_json_atomic(temporary, payload)
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["protocol"] = {
        "dataset": "Electrocardiogram-Capable Smartwatches v1.0.0",
        "dataset_doi": "10.13026/7018-y383",
        "evaluation_taxonomy": {
            "simulator_targets": "simulation_only_calibrated_protocol",
            "philips_pairing": "paired_device_signal_reference",
            "square_wave": (
                "paired_device_reference_under_simulated_2hz_stimulus; "
                "reported endpoint is the cross-correlation maximized during "
                "lag alignment, not an independent accuracy metric"
            ),
            "ecgfounder": "frozen_classifier_probability_fidelity_proxy",
            "human_diagnostic_ground_truth": "unavailable",
        },
        "task": "lead_II_to_eleven_missing_leads_zero_shot_OOD",
        "reported_signal_endpoints": {
            "primary_wearable": "eleven leads missing given acquired lead II",
            "ptbxl_comparable": (
                "nine model-designated target leads given zero-filled I/V2 "
                "input channels"
            ),
        },
        "input_lead_source": "WFDB sig_name; validated as II for every watch record",
        "source_sample_rates": "read from each WFDB header",
        "evaluation_sample_rate_hz": TARGET_FS,
        "evaluation_length_samples": TARGET_LEN,
        "resampling": "scipy.signal.resample_poly",
        "alignment": "bandpass_0.5_40Hz_normalized_cross_correlation",
        "fiducial_detector": "NeuroKit2 engzeemod2012 with explicit scipy fallback",
        "conditions": EXPERIMENT_DIRS,
        "extra_injected_stress": "not_run",
        "ecgfounder_task_count": len(tasks),
        "ecgfounder_checkpoint_sha256": sha256_file(
            PROJECT_ROOT / registry["ecgfounder_checkpoint"]
        ),
        "record_counts": {
            name: int(len(dataset["metadata"]))
            for name, dataset in datasets.items()
        },
        "pairing_audit": {
            name: dataset["pairing_audit"]
            for name, dataset in datasets.items()
        },
    }
    write_json_atomic(temporary, payload)
    os.replace(temporary, output)
    print(f"Saved protocol-safe smartwatch benchmark: {output}")


if __name__ == "__main__":
    main()
