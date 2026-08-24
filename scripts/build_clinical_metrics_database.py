#!/usr/bin/env python3
"""Build normalized task-level databases for poster figures and audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def classification_rows(
    metrics: dict[str, Any],
    *,
    dataset: str,
    evaluation_type: str,
    classifier: str,
    signal_variant: str,
    model_id: str,
    family: str,
    factorial_mask: str,
    condition: str,
    source_parquet: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_name, task in metrics["per_task"].items():
        rows.append(
            {
                "dataset": dataset,
                "evaluation_type": evaluation_type,
                "classifier": classifier,
                "signal_variant": signal_variant,
                "model_id": model_id,
                "family": family,
                "factorial_mask": factorial_mask,
                "condition": condition,
                "task_name": task_name,
                "source_parquet": source_parquet,
                **task,
            }
        )
    return rows


def fidelity_rows(
    metrics: dict[str, Any],
    *,
    dataset: str,
    model_id: str,
    family: str,
    factorial_mask: str,
    device: str,
    source_parquet: str,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "evaluation_type": metrics["evaluation_type"],
            "ground_truth_status": metrics["ground_truth_status"],
            "model_id": model_id,
            "family": family,
            "factorial_mask": factorial_mask,
            "device": device,
            "task_name": task_name,
            "source_parquet": source_parquet,
            **task,
        }
        for task_name, task in metrics["per_task"].items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl", type=Path, required=True)
    parser.add_argument("--echonext", type=Path, required=True)
    parser.add_argument("--smartwatch", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ptbxl_path = args.ptbxl if args.ptbxl.is_absolute() else PROJECT_ROOT / args.ptbxl
    echo_path = args.echonext if args.echonext.is_absolute() else PROJECT_ROOT / args.echonext
    watch_path = args.smartwatch if args.smartwatch.is_absolute() else PROJECT_ROOT / args.smartwatch
    registry_path = args.registry if args.registry.is_absolute() else PROJECT_ROOT / args.registry
    for path in (ptbxl_path, echo_path, watch_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    classification = pd.read_parquet(ptbxl_path)
    classification["condition"] = "clean"
    echo = json.loads(echo_path.read_text())
    registry = json.loads(registry_path.read_text())
    registry_by_id = {spec["id"]: spec for spec in registry["models"]}
    echo_rows: list[dict[str, Any]] = []
    reference = echo["shd_reference"]
    echo_rows.extend(
        classification_rows(
            reference["clinical"],
            dataset="EchoNext_test",
            evaluation_type="real_gt",
            classifier="EchoNext_Mini_12_SHD",
            signal_variant="reference",
            model_id="__official_reference__",
            family="reference",
            factorial_mask="reference",
            condition="clean",
            source_parquet=reference["per_record_parquet"],
        )
    )
    for model_id, model in echo["models"].items():
        spec = registry_by_id[model_id]
        echo_rows.extend(
            classification_rows(
                model["shd_clinical"],
                dataset="EchoNext_test",
                evaluation_type="real_gt",
                classifier="EchoNext_Mini_12_SHD",
                signal_variant="reconstructed",
                model_id=model_id,
                family=spec["family"],
                factorial_mask=spec["factorial_mask"],
                condition="clean",
                source_parquet=model["shd_per_record_parquet"],
            )
        )
        for condition, condition_result in model["robustness"]["conditions"].items():
            echo_rows.extend(
                classification_rows(
                    condition_result["shd_clinical"],
                    dataset="EchoNext_test",
                    evaluation_type="simulation_only_perturbation_real_labels",
                    classifier="EchoNext_Mini_12_SHD",
                    signal_variant="reconstructed",
                    model_id=model_id,
                    family=spec["family"],
                    factorial_mask=spec["factorial_mask"],
                    condition=condition,
                    source_parquet=model["shd_per_record_parquet"],
                )
            )
    classification = pd.concat(
        [classification, pd.DataFrame.from_records(echo_rows)],
        ignore_index=True,
        sort=False,
    )
    classification_path = output_dir / "classification_task_metrics.parquet"
    classification.to_parquet(classification_path, index=False)
    classification.to_csv(classification_path.with_suffix(".csv"), index=False)

    watch = json.loads(watch_path.read_text())
    watch_rows: list[dict[str, Any]] = []
    for model_id, devices in watch["models"].items():
        spec = registry_by_id[model_id]
        for device, result in devices.items():
            watch_rows.extend(
                fidelity_rows(
                    result["ecgfounder_probability_fidelity"],
                    dataset="PhysioNet_smartwatch_simulator",
                    model_id=model_id,
                    family=spec["family"],
                    factorial_mask=spec["factorial_mask"],
                    device=device,
                    source_parquet=result["per_record_parquet"],
                )
            )
    watch_fidelity = pd.DataFrame.from_records(watch_rows)
    watch_fidelity_path = output_dir / "smartwatch_task_fidelity.parquet"
    watch_fidelity.to_parquet(watch_fidelity_path, index=False)
    watch_fidelity.to_csv(watch_fidelity_path.with_suffix(".csv"), index=False)

    protocol_rows: list[dict[str, Any]] = []
    for device, experiments in watch["device_protocol_ground_truth"].items():
        for experiment, metrics in experiments.items():
            row: dict[str, Any] = {
                "dataset": "PhysioNet_smartwatch_simulator",
                "evaluation_type": metrics["evaluation_type"],
                "device": device,
                "experiment_type": experiment,
                "n_records": metrics["n_records"],
                "n_watch_measurements": metrics["n_watch_measurements"],
                "n_reference_measurements": metrics["n_reference_measurements"],
            }
            for prefix in ("watch_vs_simulator", "philips_vs_simulator"):
                for metric_name, value in metrics.get(prefix, {}).items():
                    row[f"{prefix}_{metric_name}"] = value
            if "watch_vs_philips_max_aligned_cross_correlation" in metrics:
                row["watch_vs_philips_max_aligned_cross_correlation"] = metrics[
                    "watch_vs_philips_max_aligned_cross_correlation"
                ]
            protocol_rows.append(row)
    protocol = pd.DataFrame.from_records(protocol_rows)
    protocol_path = output_dir / "smartwatch_protocol_metrics.parquet"
    protocol.to_parquet(protocol_path, index=False)
    protocol.to_csv(protocol_path.with_suffix(".csv"), index=False)

    manifest = {
        "schema_version": 1,
        "classification_rows": len(classification),
        "smartwatch_fidelity_rows": len(watch_fidelity),
        "smartwatch_protocol_rows": len(protocol),
        "classification_path": str(classification_path),
        "smartwatch_fidelity_path": str(watch_fidelity_path),
        "smartwatch_protocol_path": str(protocol_path),
        "evaluation_type_separation": {
            "real_gt": "classification_task_metrics",
            "simulation_only": (
                "condition/evaluation_type fields and calibrated scalar "
                "smartwatch protocol rows"
            ),
            "paired_device_reference": (
                "2-Hz square-wave watch-versus-Philips maximum aligned "
                "cross-correlation"
            ),
            "frozen_classifier_probability_fidelity_proxy": "smartwatch_task_fidelity",
        },
    }
    (output_dir / "clinical_database_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
