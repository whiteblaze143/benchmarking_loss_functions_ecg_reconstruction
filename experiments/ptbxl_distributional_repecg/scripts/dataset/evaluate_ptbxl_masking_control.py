#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from repecg.common.models import PhaseCNN


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_equal_weights(patient_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(patient_ids)
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _metrics(target: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    classwise_auroc = [
        float(roc_auc_score(target[:, column], probability[:, column], sample_weight=weights))
        for column in range(target.shape[1])
    ]
    classwise_auprc = [
        float(average_precision_score(target[:, column], probability[:, column], sample_weight=weights))
        for column in range(target.shape[1])
    ]
    patient_total = float(weights.sum())
    return {
        "macro_auroc": float(np.mean(classwise_auroc)),
        "macro_auprc": float(np.mean(classwise_auprc)),
        "brier": float(np.average(np.square(target - probability).mean(axis=1), weights=weights)),
        "classwise_auroc": dict(zip(CLASSES, classwise_auroc, strict=True)),
        "classwise_auprc": dict(zip(CLASSES, classwise_auprc, strict=True)),
        "positive_patient_equivalents": {
            name: float(np.dot(weights, target[:, column]))
            for column, name in enumerate(CLASSES)
        },
        "negative_patient_equivalents": {
            name: patient_total - float(np.dot(weights, target[:, column]))
            for column, name in enumerate(CLASSES)
        },
    }


def _predict(checkpoint: Path, values: np.ndarray, batch_size: int) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PhaseCNN(values.shape[-1])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.cuda().eval()
    batches = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy(values[start : start + batch_size]).cuda()
            batches.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(batches)


def _best_checkpoint(cells: Path, variant: str) -> tuple[Path, dict[str, object]]:
    candidates = []
    for summary_path in cells.glob(f"{variant}_*/summary.json"):
        summary = json.loads(summary_path.read_text())
        candidates.append((summary_path.parent, summary))
    if not candidates:
        raise FileNotFoundError(f"no completed {variant} checkpoints in {cells}")
    path, summary = max(candidates, key=lambda item: float(item[1]["metrics"]["macro_auroc"]))
    checkpoint = path / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return checkpoint, summary


def _bootstrap_delta(
    target: np.ndarray,
    full: np.ndarray,
    masked: np.ndarray,
    patient_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    patients = np.unique(patient_ids)
    rows = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        index = np.concatenate([rows[patient] for patient in sampled])
        occurrence = np.concatenate(
            [np.full(len(rows[patient]), occurrence_id, dtype=np.int64) for occurrence_id, patient in enumerate(sampled)]
        )
        weights = patient_equal_weights(occurrence)
        full_score = _metrics(target[index], full[index], weights)["macro_auroc"]
        masked_score = _metrics(target[index], masked[index], weights)["macro_auroc"]
        draws[replicate] = float(masked_score) - float(full_score)
    return draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    torch.backends.cudnn.enabled = False
    torch.set_float32_matmul_precision("high")

    manifest_path = args.representations / "ptbxl_masking_control_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("max_records") is not None or manifest.get("records") != 2173:
        raise ValueError("negative-control metrics require the complete 2,173-record fold-8 build")
    full_path = args.representations / manifest["conditions"]["full"]["archive"]
    masked_path = args.representations / manifest["conditions"]["masked_i_ii"]["archive"]
    with np.load(full_path) as item:
        full_archive = {name: np.asarray(item[name]) for name in item.files}
    with np.load(masked_path) as item:
        masked_archive = {name: np.asarray(item[name]) for name in item.files}
    for key in ("labels", "ecg_ids", "patient_ids"):
        if not np.array_equal(full_archive[key], masked_archive[key]):
            raise ValueError(f"paired archive mismatch for {key}")

    target = full_archive["labels"]
    patient_ids = full_archive["patient_ids"]
    weights = patient_equal_weights(patient_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "kind": "paired_ptbxl_missing_lead_metrics",
        "evaluation_role": "negative_control_matched_reference",
        "records": len(target),
        "patients": int(np.unique(patient_ids).size),
        "aggregation": "inverse_records_per_patient_weighting",
        "bootstrap_unit": "patient",
        "replicates": args.replicates,
        "delta_direction": "masked_minus_full",
        "selection_fold_warning": "descriptive robustness control on fold 8 used for checkpoint selection; not independent test evidence",
        "representation_manifest_sha256": _sha256(manifest_path),
        "variants": {},
    }
    prediction_arrays: dict[str, np.ndarray] = {
        "labels": target,
        "ecg_ids": full_archive["ecg_ids"],
        "patient_ids": patient_ids,
    }
    bootstrap_arrays = {}
    for variant in ("kernel", "moments"):
        checkpoint, checkpoint_summary = _best_checkpoint(args.cells, variant)
        full_probability = _predict(checkpoint, full_archive[variant], args.batch_size)
        masked_probability = _predict(checkpoint, masked_archive[variant], args.batch_size)
        draws = _bootstrap_delta(
            target, full_probability, masked_probability, patient_ids, args.replicates, args.seed
        )
        full_metrics = _metrics(target, full_probability, weights)
        masked_metrics = _metrics(target, masked_probability, weights)
        result["variants"][variant] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "selected_full_record_macro_auroc": checkpoint_summary["metrics"]["macro_auroc"],
            "full_patient_equal": full_metrics,
            "masked_patient_equal": masked_metrics,
            "macro_auroc_delta": float(masked_metrics["macro_auroc"]) - float(full_metrics["macro_auroc"]),
            "macro_auroc_delta_ci95": np.quantile(draws, (0.025, 0.975)).tolist(),
        }
        prediction_arrays[f"probability_full_{variant}"] = full_probability
        prediction_arrays[f"probability_masked_{variant}"] = masked_probability
        bootstrap_arrays[f"macro_auroc_delta_{variant}"] = draws

    temporary_predictions = args.output / "predictions.tmp.npz"
    np.savez_compressed(temporary_predictions, **prediction_arrays)
    os.replace(temporary_predictions, args.output / "predictions.npz")
    temporary_bootstrap = args.output / "bootstrap.tmp.npz"
    np.savez_compressed(temporary_bootstrap, **bootstrap_arrays)
    os.replace(temporary_bootstrap, args.output / "bootstrap.npz")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
