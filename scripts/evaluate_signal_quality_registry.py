#!/usr/bin/env python3
"""Evaluate PTB-XL signal-quality preservation for all factorial checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.clinical_metrics import probability_fidelity_metrics
from scripts.evaluate_comprehensive_registry import classification_metrics, load_adapter
from scripts.signal_quality_classifier import load_classifier, preprocess
from scripts.train_signal_quality_classifier import ARTIFACT_COLUMNS, quality_labels


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QualityDataset(Dataset):
    def __init__(self, directory: Path, labels: dict[int, np.ndarray]):
        self.files = sorted(directory.glob("*.pt"), key=lambda path: int(path.stem))
        self.labels = labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        signal = torch.load(path, weights_only=True).float()
        if signal.shape != (12, 5000):
            raise ValueError(f"Unexpected tensor shape at {path}: {tuple(signal.shape)}")
        ecg_id = int(path.stem)
        return signal, ecg_id, torch.from_numpy(self.labels[ecg_id])


@torch.inference_mode()
def source_predictions(model, loader, device):
    ids, labels, probabilities = [], [], []
    for signals, batch_ids, batch_labels in tqdm(loader, desc="PTB-XL quality reference"):
        probabilities.append(
            torch.sigmoid(model(preprocess(signals.to(device, non_blocking=True))))
            .cpu()
            .numpy()
        )
        ids.append(batch_ids.numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(ids), np.concatenate(labels), np.concatenate(probabilities)


@torch.inference_mode()
def evaluate_model(spec, loader, classifier, device, reference_by_id, class_names):
    adapter = load_adapter(spec, device)
    ids_all, labels_all, probs_all, source_all = [], [], [], []
    observed_error = 0.0
    for signals, batch_ids, labels in tqdm(loader, desc=spec["id"]):
        signals = signals.to(device, non_blocking=True)
        reconstruction = adapter.reconstruct(signals, preserve_observed=adapter.observed)
        if reconstruction.shape != signals.shape or not torch.isfinite(reconstruction).all():
            raise RuntimeError(f"{spec['id']} produced invalid reconstruction")
        observed_error = max(
            observed_error,
            float(
                (reconstruction[:, adapter.observed] - signals[:, adapter.observed])
                .abs()
                .max()
                .cpu()
            ),
        )
        probabilities = torch.sigmoid(classifier(preprocess(reconstruction))).cpu().numpy()
        ids = batch_ids.numpy()
        ids_all.append(ids)
        labels_all.append(labels.numpy())
        probs_all.append(probabilities)
        source_all.append(np.stack([reference_by_id[int(ecg_id)] for ecg_id in ids]))

    ids = np.concatenate(ids_all)
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probs_all)
    source_probabilities = np.concatenate(source_all)
    result = {
        "id": spec["id"],
        "family": spec["family"],
        "mask": spec["factorial_mask"],
        "factors": spec["factors"],
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": sha256(ROOT / spec["checkpoint"]),
        "n_records": int(len(ids)),
        "observed_lead_max_abs_error": observed_error,
        "classification": classification_metrics(probabilities, labels, class_names),
        "source_classification": classification_metrics(
            source_probabilities, labels, class_names
        ),
        "probability_fidelity": probability_fidelity_metrics(
            source_probabilities,
            probabilities,
            class_names,
        ),
    }
    return result, pd.DataFrame(
        {
            "ecg_id": ids.astype(np.int64),
            "labels": list(labels.astype(np.float32)),
            "probabilities": list(probabilities.astype(np.float32)),
            "source_probabilities": list(source_probabilities.astype(np.float32)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "experiment_queue/factorial_v4/model_registry.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/factorial_v4/parity/ecgfm_signal_quality.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ptbxl_signal_quality_factorial_v4_2x4",
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.registry = args.registry.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_record_dir = args.output_dir / "per_record"
    per_record_dir.mkdir(parents=True, exist_ok=True)

    registry = json.loads(args.registry.read_text())
    specs = registry["models"]
    if args.models:
        requested = set(args.models)
        specs = [spec for spec in specs if spec["id"] in requested]
        unknown = requested - {spec["id"] for spec in specs}
        if unknown:
            raise ValueError(f"Unknown model ids: {sorted(unknown)}")
    database_path = ROOT / registry["ptbxl_csv"]
    database = pd.read_csv(database_path)
    labels = quality_labels(database)
    dataset = QualityDataset(ROOT / registry["data_dir"], labels)
    if len(dataset) != 2198:
        raise RuntimeError(f"Expected 2,198 PTB-XL fold-10 tensors, found {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    device = torch.device(args.device)
    classifier, class_names = load_classifier(
        ROOT / registry["five_superclass_backbone"], args.checkpoint, device
    )
    ids, labels_np, source_probs = source_predictions(classifier, loader, device)
    reference_by_id = {
        int(ecg_id): source_probs[index] for index, ecg_id in enumerate(ids)
    }
    reference = classification_metrics(source_probs, labels_np, class_names)

    partial_path = args.output_dir / "signal_quality_results.json.partial"
    output_path = args.output_dir / "signal_quality_results.json"
    models: dict[str, Any] = {}
    if args.resume and partial_path.exists():
        models = json.loads(partial_path.read_text()).get("models", {})
    payload = {
        "schema_version": 1,
        "protocol": {
            "dataset": "PTB-XL",
            "split": "fold 10 (test)",
            "n_records": len(dataset),
            "classification_task": "PTB-XL signal-quality annotations",
            "classes": class_names,
            "annotation_semantics": "non-null value in the corresponding PTB-XL metadata field",
            "annotation_columns": ARTIFACT_COLUMNS,
            "training_folds": "1-8",
            "selection_fold": 9,
            "test_used_for_training_or_selection": False,
            "classifier": "frozen ECG-FM backbone plus train/validation-only linear head",
            "preprocessing": "per-lead record z-score",
            "electrodes_problems_caveat": "exploratory endpoint with only 3 fold-10 positives",
            "registry": str(args.registry.relative_to(ROOT)),
            "registry_sha256": sha256(args.registry),
            "quality_checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "quality_checkpoint_sha256": sha256(args.checkpoint),
            "database_sha256": sha256(database_path),
        },
        "label_counts": {
            name: {
                "positive": int(labels_np[:, index].sum()),
                "negative": int(len(labels_np) - labels_np[:, index].sum()),
            }
            for index, name in enumerate(class_names)
        },
        "source_reference": reference,
        "models": models,
    }
    for spec in specs:
        if spec["id"] in models:
            print(f"resume: skipping {spec['id']}", flush=True)
            continue
        result, frame = evaluate_model(
            spec, loader, classifier, device, reference_by_id, class_names
        )
        path = per_record_dir / f"{spec['id']}.parquet"
        frame.to_parquet(path, index=False)
        result["per_record_parquet"] = str(path.relative_to(ROOT))
        models[spec["id"]] = result
        partial_path.write_text(
            json.dumps(json_safe(payload), indent=2, allow_nan=False)
        )
        gc.collect()
        torch.cuda.empty_cache()

    payload["completeness"] = {
        "expected_models": len(specs),
        "completed_models": len(models),
        "expected_records_per_model": len(dataset),
        "status": "complete" if len(models) == len(specs) else "incomplete",
    }
    if payload["completeness"]["status"] != "complete":
        raise RuntimeError(f"Incomplete evaluation: {payload['completeness']}")
    output_path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))
    partial_path.unlink(missing_ok=True)
    print(output_path)


if __name__ == "__main__":
    main()
