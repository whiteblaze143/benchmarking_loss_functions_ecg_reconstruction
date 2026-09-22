#!/usr/bin/env python3
"""Compute preregistered paired inference from frozen task-native predictions.

This script only consumes aligned prediction bundles emitted by
``run_task_native_queue.py``.  It never reorders observations, refits a head,
or constructs substitute predictions.  Every available model pair is evaluated
for every requested configuration and binary class.  Multi-label tasks are
therefore reported class-by-class; macro-AUROC has no valid single DeLong test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from repecg.evaluation.paired_inference import paired_delong_auc, patient_equal_paired_bootstrap
from repecg.evaluation.prediction_artifacts import PredictionBundle, load_prediction_bundle, require_paired_bundles


SCHEMA_VERSION = "task_native_paired_statistics_v1"
ALPHA = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_bootstrap(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    """GraphECG-style paired test-record bootstrap, with no row substitutions."""
    observed = float(roc_auc_score(labels, scores_a) - roc_auc_score(labels, scores_b))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[indices]).size != 2:
            continue
        deltas.append(float(
            roc_auc_score(labels[indices], scores_a[indices])
            - roc_auc_score(labels[indices], scores_b[indices])
        ))
    if not deltas:
        raise ValueError("all record bootstrap resamples were metric-ineligible")
    distribution = np.asarray(deltas, dtype=np.float64)
    return {
        "observed_delta": observed,
        "ci_lower": float(np.quantile(distribution, 0.025)),
        "ci_upper": float(np.quantile(distribution, 0.975)),
        "replicates": replicates,
        "valid_replicates": len(distribution),
    }


def _bundles_by_dataset(prediction_root: Path) -> dict[str, dict[str, PredictionBundle]]:
    if not prediction_root.is_dir():
        raise FileNotFoundError(f"prediction directory does not exist: {prediction_root}")
    paths = sorted(prediction_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no prediction bundles in {prediction_root}")
    result: dict[str, dict[str, PredictionBundle]] = {}
    for path in paths:
        bundle = load_prediction_bundle(path)
        per_dataset = result.setdefault(bundle.dataset, {})
        if bundle.model in per_dataset:
            raise ValueError(f"duplicate prediction bundle for {bundle.dataset}/{bundle.model}")
        per_dataset[bundle.model] = bundle
    return result


def _comparison_rows(
    left: PredictionBundle,
    right: PredictionBundle,
    *,
    configurations: set[str] | None,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    require_paired_bundles(left, right)
    rows: list[dict[str, Any]] = []
    for cfg_index, configuration in enumerate(left.configurations):
        if configurations is not None and configuration not in configurations:
            continue
        for class_index in range(left.y_true.shape[1]):
            labels = left.y_true[:, class_index].astype(np.int8, copy=False)
            scores_a = left.probabilities[cfg_index, :, class_index]
            scores_b = right.probabilities[cfg_index, :, class_index]
            row: dict[str, Any] = {
                "dataset": left.dataset,
                "task_name": left.task_name,
                "configuration": configuration,
                "class_index": class_index,
                "model_a": left.model,
                "model_b": right.model,
                "observations": int(len(labels)),
                "positives": int(labels.sum()),
                "negatives": int((1 - labels).sum()),
                "alpha_unadjusted": ALPHA,
                "status": "complete",
            }
            try:
                delong = paired_delong_auc(labels, scores_a, scores_b)
            except ValueError as error:
                row.update({"status": "ineligible", "reason": str(error)})
                rows.append(row)
                continue
            row.update({f"delong_{key}": value for key, value in asdict(delong).items()})
            if left.patient_ids is not None and len(np.unique(left.patient_ids)) != len(left.patient_ids):
                row.update({
                    "bootstrap_status": "ineligible",
                    "bootstrap_reason": "repeated patient IDs require an explicit endpoint aggregation rule",
                })
            elif left.patient_ids is not None:
                bootstrap = patient_equal_paired_bootstrap(
                    labels, scores_a, scores_b, left.patient_ids,
                    replicates=replicates, seed=seed,
                )
                row.update({"bootstrap_status": "complete", "bootstrap_unit": "patient"})
                row.update({f"bootstrap_{key}": value for key, value in asdict(bootstrap).items()})
            else:
                bootstrap = _record_bootstrap(
                    labels, scores_a, scores_b, replicates=replicates, seed=seed,
                )
                row.update({"bootstrap_status": "complete", "bootstrap_unit": "record"})
                row.update({f"bootstrap_{key}": value for key, value in bootstrap.items()})
            rows.append(row)
    return rows


def run(
    prediction_root: Path,
    *,
    models: set[str] | None,
    configurations: set[str] | None,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run all valid model pairs and return a provenance manifest plus rows."""
    if replicates != 2000:
        raise ValueError("the frozen GraphECG-compatible contract requires exactly 2,000 bootstrap replicates")
    groups = _bundles_by_dataset(prediction_root)
    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for dataset, by_model in sorted(groups.items()):
        selected = sorted(model for model in by_model if models is None or model in models)
        if len(selected) < 2:
            continue
        for model_a, model_b in itertools.combinations(selected, 2):
            left, right = by_model[model_a], by_model[model_b]
            input_hashes[str(left.path)] = _sha256(left.path)
            input_hashes[str(right.path)] = _sha256(right.path)
            rows.extend(_comparison_rows(
                left, right, configurations=configurations, replicates=replicates, seed=seed,
            ))
    if not rows:
        raise ValueError("fewer than two selected models with aligned prediction bundles")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "test": "two-sided paired DeLong AUROC",
        "bootstrap": {"replicates": replicates, "seed": seed, "confidence_interval": "percentile_95"},
        "alpha_unadjusted": ALPHA,
        "input_sha256": input_hashes,
        "rows": len(rows),
    }
    return manifest, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="all", help="Comma-separated model keys, or all")
    parser.add_argument("--configurations", default="all", help="Comma-separated configuration keys, or all")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated statistics directory: {args.output_dir}")
    models = None if args.models == "all" else {item.strip() for item in args.models.split(",") if item.strip()}
    configurations = None if args.configurations == "all" else {
        item.strip() for item in args.configurations.split(",") if item.strip()
    }
    manifest, rows = run(
        args.prediction_root,
        models=models,
        configurations=configurations,
        replicates=2000,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    columns = sorted({key for row in rows for key in row})
    with (args.output_dir / "paired_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "rows": len(rows), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
