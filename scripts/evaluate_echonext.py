#!/usr/bin/env python3
"""Evaluate a registry on EchoNext with a strict acquisition/provenance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

from scripts.evaluate_comprehensive_registry import (
    LEAD_NAMES,
    StreamingSignalMetrics,
    json_safe,
    load_adapter,
    morphology_metrics,
    record_signal_metrics,
    save_example_plot,
)
from scripts.ecgfounder_classifier import preprocess_ecgfounder
from scripts.clinical_metrics import (
    multilabel_classification_metrics,
    probability_fidelity_metrics,
)
from scripts.echonext_classifier import (
    EchoNextMiniModel,
    SHD_LABEL_COLUMNS,
    SHD_TASKS,
    load_echonext_test_metadata,
)
from scripts.robustness_stress import apply_condition

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_waveform(data_dir: Path) -> Path:
    matches = list(data_dir.glob("**/EchoNext_test_waveforms.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one EchoNext_test_waveforms.npy under {data_dir}; found {matches}"
        )
    return matches[0]


class EchoNextWaveforms:
    """Memory-mapped EchoNext waveforms with batch-local provenance inversion."""

    def __init__(self, data: np.ndarray, provenance: dict[str, Any]):
        self.data = data
        self.provenance = provenance
        if data.ndim == 4 and data.shape[1] == 1 and data.shape[-1] == 12:
            self.layout = "n_1_time_lead"
            source_length = data.shape[2]
        elif data.ndim == 3 and data.shape[1] == 12:
            self.layout = "n_lead_time"
            source_length = data.shape[2]
        elif data.ndim == 3 and data.shape[-1] == 12:
            self.layout = "n_time_lead"
            source_length = data.shape[1]
        else:
            raise ValueError(f"Unsupported EchoNext waveform shape: {data.shape}")
        if source_length != 2500:
            raise ValueError(f"Expected 10 seconds at 250 Hz (2500 samples), got {source_length}")

        normalization = provenance["normalization"]
        if normalization in (None, "none", {"kind": "none"}):
            self.mean = None
            self.std = None
        elif isinstance(normalization, dict) and normalization.get("kind") == "dataset_zscore":
            if "mean" not in normalization or "std" not in normalization:
                raise ValueError("dataset_zscore provenance requires mean and std for inversion")
            self.mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(1, 12, 1)
            self.std = np.asarray(normalization["std"], dtype=np.float32).reshape(1, 12, 1)
            if self.mean.shape != (1, 12, 1) or self.std.shape != (1, 12, 1):
                raise ValueError("EchoNext normalization mean/std must contain 12 lead values")
            if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
                raise ValueError("EchoNext normalization statistics must be finite with positive std")
        else:
            raise ValueError(f"Unsupported or non-invertible normalization: {normalization}")

        unit_scale = {"mV": 1.0, "uV": 1e-3, "V": 1e3}.get(provenance["units"])
        if unit_scale is None:
            raise ValueError(f"Unsupported ECG units: {provenance['units']}")
        self.unit_scale = unit_scale

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def batch(self, start: int, stop: int) -> np.ndarray:
        values = self.data[start:stop]
        if self.layout == "n_1_time_lead":
            values = values[:, 0].transpose(0, 2, 1)
        elif self.layout == "n_time_lead":
            values = values.transpose(0, 2, 1)
        values = np.asarray(values, dtype=np.float32)
        if self.mean is not None:
            values = values * self.std + self.mean
        values = values * self.unit_scale
        values = resample_poly(values, up=2, down=1, axis=-1).astype(np.float32)
        if values.shape[1:] != (12, 5000) or not np.isfinite(values).all():
            raise ValueError(f"Expected finite [B,12,5000] after resampling; got {values.shape}")
        return values

    def official_normalized_batch(self, start: int, stop: int) -> np.ndarray:
        """Return the released normalized layout expected by EchoNext Mini."""
        values = self.data[start:stop]
        if self.layout == "n_1_time_lead":
            output = np.asarray(values, dtype=np.float32)
        elif self.layout == "n_lead_time":
            output = np.asarray(values, dtype=np.float32).transpose(0, 2, 1)[:, None]
        else:
            output = np.asarray(values, dtype=np.float32)[:, None]
        if output.shape[1:] != (1, 2500, 12) or not np.isfinite(output).all():
            raise ValueError(
                f"Expected finite official [B,1,2500,12] batch; got {output.shape}"
            )
        return output


def ordered_value_hash(values: np.ndarray) -> str:
    serialized = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))
    os.replace(staging, path)


def repeat_condition_tabular(
    tabular_batch: np.ndarray,
    condition_count: int,
) -> np.ndarray:
    """Match condition-major concatenation used for grouped stress inference."""
    values = np.asarray(tabular_batch, dtype=np.float32)
    if values.ndim != 2 or condition_count < 1:
        raise ValueError("Expected 2-D tabular batch and a positive condition count")
    return np.tile(values, (condition_count, 1))


def load_and_validate(data_dir: Path) -> tuple[EchoNextWaveforms, dict[str, Any]]:
    provenance_path = data_dir / "PROVENANCE.json"
    if not provenance_path.exists():
        raise FileNotFoundError(f"Missing required provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    required = {"source", "version", "sampling_rate_hz", "lead_order", "units", "normalization"}
    missing = sorted(required - provenance.keys())
    if missing:
        raise ValueError(f"EchoNext provenance is missing fields: {missing}")
    if float(provenance["sampling_rate_hz"]) != 250.0:
        raise ValueError("EchoNext source sampling_rate_hz must be 250")
    if provenance["lead_order"] != LEAD_NAMES:
        raise ValueError(f"Lead order mismatch: {provenance['lead_order']}")

    waveform_path = resolve_waveform(data_dir)
    data = np.load(waveform_path, mmap_mode="r")
    waveforms = EchoNextWaveforms(data, provenance)
    waveforms.batch(0, min(1, len(waveforms)))
    return waveforms, {**provenance, "waveform_path": str(waveform_path)}


@torch.no_grad()
def evaluate(
    data: EchoNextWaveforms,
    registry: dict[str, Any],
    device: torch.device,
    batch_size: int,
    output_path: Path,
    morphology_samples: int,
    classifier: torch.nn.Module | None,
    stress_conditions: list[Any],
    nstdb_noises: dict[str, np.ndarray],
    robustness_samples: int,
    robustness_condition_batch: int,
    data_dir: Path,
    resume: bool,
    shd_classifier: EchoNextMiniModel,
    shd_metadata: pd.DataFrame,
    shd_tabular: np.ndarray,
    shd_labels: np.ndarray,
    ecgfounder_tasks: list[str] | None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_dir = output_path.parent / "echonext_per_record"
    record_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    payload: dict[str, Any] = {"models": {}}
    reference_shd_path = output_path.parent / "echonext_reference_shd.parquet"
    reference_shd_probabilities: list[np.ndarray] = []
    reference_batch_size = max(batch_size, 256)
    for start in range(0, len(data), reference_batch_size):
        stop = min(start + reference_batch_size, len(data))
        reference_shd_probabilities.append(
            shd_classifier.predict_official_waveforms(
                data.official_normalized_batch(start, stop),
                shd_tabular[start:stop],
            )
        )
    reference_shd = np.concatenate(reference_shd_probabilities)
    if reference_shd.shape != shd_labels.shape:
        raise RuntimeError(
            f"EchoNext reference prediction/label mismatch: "
            f"{reference_shd.shape} vs {shd_labels.shape}"
        )
    pd.DataFrame(
        {
            "row_index": np.arange(len(data), dtype=np.int64),
            "ecg_key": shd_metadata["ecg_key"].to_numpy(),
            "labels": list(shd_labels.astype(np.int8)),
            "probabilities": list(reference_shd.astype(np.float32)),
        }
    ).to_parquet(reference_shd_path, index=False)
    reference_shd_metrics = multilabel_classification_metrics(
        reference_shd,
        shd_labels,
        SHD_TASKS,
        thresholds=0.5,
        threshold_source="official_fixed_0.5",
    )
    reference_macro_auroc = reference_shd_metrics["macro"]["auroc"]
    if reference_macro_auroc is None or not 0.79 <= reference_macro_auroc <= 0.82:
        raise RuntimeError(
            "Official EchoNext reference sanity gate failed: expected macro "
            f"AUROC in [0.79,0.82], found {reference_macro_auroc}. This usually "
            "indicates waveform/metadata misalignment or preprocessing drift."
        )
    if resume and partial_path.exists():
        payload = json.loads(partial_path.read_text())
        registered_ids = {spec["id"] for spec in registry["models"]}
        unexpected = set(payload.get("models", {})) - registered_ids
        if unexpected:
            raise ValueError(f"Partial EchoNext output has unregistered models: {unexpected}")
        print(f"Resuming EchoNext evaluation from {len(payload.get('models', {}))} models")
    ids = torch.arange(len(data), dtype=torch.long)
    for spec in registry["models"]:
        existing = payload.get("models", {}).get(spec["id"])
        expected_condition_ids = {condition.id for condition in stress_conditions}
        if existing is not None:
            existing_paths = [
                PROJECT_ROOT / existing.get("per_record_parquet", "__missing__"),
                PROJECT_ROOT / existing.get("shd_per_record_parquet", "__missing__"),
            ]
            if classifier is not None:
                existing_paths.append(
                    PROJECT_ROOT / existing.get(
                        "ecgfounder_per_record_parquet", "__missing__"
                    )
                )
            complete = (
                existing.get("signal", {}).get("n_samples") == len(data)
                and existing.get("robustness", {}).get("n_samples")
                == min(robustness_samples, len(data))
                and existing.get("morphology_evaluated_records")
                == min(morphology_samples, len(data))
                and set(existing.get("robustness", {}).get("conditions", {}))
                == expected_condition_ids
                and existing.get("shd_clinical", {}).get("n_tasks") == 12
                and all(
                    condition.get("shd_clinical", {}).get("n_tasks") == 12
                    for condition in existing.get("robustness", {})
                    .get("conditions", {})
                    .values()
                )
                and all(path.exists() for path in existing_paths)
            )
            if complete:
                print(f"Resume: verified and skipped {spec['id']}")
                continue
            print(f"Resume: recomputing incomplete cell {spec['id']}")
            payload["models"].pop(spec["id"], None)
        adapter = load_adapter(spec, device)
        family_batch_size = (
            max(batch_size, 128)
            if spec["family"] == "unet"
            else max(batch_size, 32)
        )
        metrics = StreamingSignalMetrics(adapter.observed)
        
        robust_clean_metrics = StreamingSignalMetrics(adapter.observed)
        robust_metrics = {
            condition.id: StreamingSignalMetrics(adapter.observed)
            for condition in stress_conditions
        }
        
        rows: list[dict[str, Any]] = []
        morphology_target: list[np.ndarray] = []
        morphology_recon: list[np.ndarray] = []
        morphology_seen = 0
        example_plot: str | None = None
        
        robust_seen = 0
        probabilities: list[np.ndarray] = []
        reference_probabilities: list[np.ndarray] = []
        ecg_ids_all: list[np.ndarray] = []
        clean_shd_probabilities: list[np.ndarray] = []
        stress_shd_probabilities: dict[str, list[np.ndarray]] = {
            condition.id: [] for condition in stress_conditions
        }
        family_condition_batch = min(
            robustness_condition_batch,
            10
            if spec["family"] in {"multiscale_vae", "ecg_aim"}
            else robustness_condition_batch,
        )

        for start in range(0, len(data), family_batch_size):
            stop = min(start + family_batch_size, len(data))
            target = torch.from_numpy(data.batch(start, stop)).to(device)
            reconstruction = adapter.reconstruct(target)
            if reconstruction.shape != target.shape or not torch.isfinite(reconstruction).all():
                raise RuntimeError(f"Invalid reconstruction from {spec['id']}")
            metrics.update(target, reconstruction)
            clean_shd_probabilities.append(
                shd_classifier.predict_reconstruction_500hz(
                    reconstruction,
                    shd_tabular[start:stop],
                )
            )
            rows.extend(record_signal_metrics(
                target.cpu(), reconstruction.cpu(), ids[start:stop], adapter.observed, "clean"
            ))
            if morphology_seen < morphology_samples:
                take = min(stop - start, morphology_samples - morphology_seen)
                morphology_target.extend(target[:take].cpu().numpy())
                morphology_recon.extend(reconstruction[:take].cpu().numpy())
                morphology_seen += take
            if example_plot is None:
                example_plot = save_example_plot(
                    spec["id"],
                    target[0].cpu().numpy(),
                    reconstruction[0].cpu().numpy(),
                    output_path.parent / "echonext_example_plots",
                )
                
            # Robustness stress testing
            if robust_seen < robustness_samples:
                take = min(target.shape[0], robustness_samples - robust_seen)
                robust_clean_metrics.update(target[:take], reconstruction[:take])
                for condition_start in range(0, len(stress_conditions), family_condition_batch):
                    condition_group = stress_conditions[
                        condition_start:condition_start + family_condition_batch
                    ]
                    noisy_group = [
                        apply_condition(
                            target[:take],
                            adapter.observed,
                            ids[start:start+take],
                            condition,
                            nstdb_noises,
                        )
                        for condition in condition_group
                    ]
                    reconstructed_group = adapter.reconstruct(torch.cat(noisy_group, dim=0))
                    if reconstructed_group.shape[0] != take * len(condition_group):
                        raise RuntimeError("Robustness condition batching changed output batch size")
                    repeated_tabular = repeat_condition_tabular(
                        shd_tabular[start:start + take],
                        len(condition_group),
                    )
                    grouped_shd_probabilities = (
                        shd_classifier.predict_reconstruction_500hz(
                            reconstructed_group,
                            repeated_tabular,
                        )
                    )
                    for group_index, condition in enumerate(condition_group):
                        noisy_recon = reconstructed_group[
                            group_index * take:(group_index + 1) * take
                        ]
                        robust_metrics[condition.id].update(
                            target[:take], noisy_recon
                        )
                        stress_shd_probabilities[condition.id].append(
                            grouped_shd_probabilities[
                                group_index * take:(group_index + 1) * take
                            ]
                        )
                robust_seen += take

            if classifier is not None:
                logits = classifier(preprocess_ecgfounder(reconstruction))
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                reference_probabilities.append(
                    torch.sigmoid(classifier(preprocess_ecgfounder(target))).cpu().numpy()
                )
                ecg_ids_all.append(ids[start:stop].numpy())
                
        record_path = record_dir / f"{spec['id']}.parquet"
        pd.DataFrame.from_records(rows).to_parquet(record_path, index=False)
        
        clean = metrics.finalize()
        clean_shd = np.concatenate(clean_shd_probabilities)
        clean_shd_metrics = multilabel_classification_metrics(
            clean_shd,
            shd_labels,
            SHD_TASKS,
            thresholds=0.5,
            threshold_source="official_fixed_0.5",
        )
        robust_clean = robust_clean_metrics.finalize() if robust_seen else {}
        robust_conditions: dict[str, Any] = {}
        for condition in stress_conditions:
            stressed = robust_metrics[condition.id].finalize() if robust_seen else {}
            condition_result: dict[str, Any] = {
                "source": condition.source,
                "noise_type": condition.noise_type,
                "snr_db": condition.snr_db,
                "n_samples": robust_seen,
                "signal": stressed,
                "missing_lead_mse_delta": (
                    stressed["missing_leads"]["mse"]
                    - robust_clean["missing_leads"]["mse"]
                    if stressed
                    else float("nan")
                ),
            }
            condition_shd = (
                np.concatenate(stress_shd_probabilities[condition.id])
                if stress_shd_probabilities[condition.id]
                else np.empty((0, len(SHD_TASKS)), dtype=np.float32)
            )
            condition_result["shd_clinical"] = (
                multilabel_classification_metrics(
                    condition_shd,
                    shd_labels[:robust_seen],
                    SHD_TASKS,
                    thresholds=0.5,
                    threshold_source="official_fixed_0.5",
                )
                if robust_seen
                else {}
            )
            robust_conditions[condition.id] = condition_result

        shd_frames = [
            pd.DataFrame(
                {
                    "row_index": np.arange(len(data), dtype=np.int64),
                    "ecg_key": shd_metadata["ecg_key"].to_numpy(),
                    "condition": "clean",
                    "labels": list(shd_labels.astype(np.int8)),
                    "probabilities": list(clean_shd.astype(np.float32)),
                }
            )
        ]
        for condition in stress_conditions:
            if not stress_shd_probabilities[condition.id]:
                continue
            condition_probabilities = np.concatenate(
                stress_shd_probabilities[condition.id]
            )
            shd_frames.append(
                pd.DataFrame(
                    {
                        "row_index": np.arange(robust_seen, dtype=np.int64),
                        "ecg_key": shd_metadata["ecg_key"]
                        .to_numpy()[:robust_seen],
                        "condition": condition.id,
                        "labels": list(shd_labels[:robust_seen].astype(np.int8)),
                        "probabilities": list(
                            condition_probabilities.astype(np.float32)
                        ),
                    }
                )
            )
        shd_path = record_dir / f"{spec['id']}__echonext_shd.parquet"
        pd.concat(shd_frames, ignore_index=True).to_parquet(shd_path, index=False)
            
        model_payload = {
            "id": spec["id"],
            "family": spec["family"],
            "factorial_mask": spec["factorial_mask"],
            "signal": clean,
            "shd_clinical": clean_shd_metrics,
            "shd_probability_fidelity": probability_fidelity_metrics(
                clean_shd,
                reference_shd,
                SHD_TASKS,
            ),
            "shd_per_record_parquet": str(
                shd_path.relative_to(PROJECT_ROOT)
                if shd_path.is_relative_to(PROJECT_ROOT)
                else shd_path
            ),
            "robustness": {
                "n_samples": robust_seen,
                "clean_subset": robust_clean,
                "conditions": robust_conditions,
            },
            "morphology": morphology_metrics(morphology_target, morphology_recon, adapter.observed),
            "morphology_evaluated_records": morphology_seen,
            "example_plot": example_plot,
            "per_record_parquet": str(record_path.relative_to(PROJECT_ROOT)) if record_path.is_relative_to(PROJECT_ROOT) else str(record_path),
            "per_record_conditions": ["clean"],
            "batch_size": family_batch_size,
            "robustness_condition_batch": family_condition_batch if stress_conditions else 0,
        }
        
        if probabilities:
            probs = np.concatenate(probabilities)
            reference_probs = np.concatenate(reference_probabilities)
            clinical_path = record_dir / f"{spec['id']}__ecgfounder_echonext.parquet"
            clinical_dict = {
                "ecg_id": np.concatenate(ecg_ids_all).astype(np.int64),
                "probabilities": list(probs.astype(np.float32)),
                "reference_probabilities": list(reference_probs.astype(np.float32)),
            }
            if ecgfounder_tasks is not None:
                model_payload["ecgfounder_probability_fidelity"] = (
                    probability_fidelity_metrics(
                        probs,
                        reference_probs,
                        ecgfounder_tasks,
                    )
                )
            
            clinical_dict["ecg_key"] = shd_metadata["ecg_key"].to_numpy()
            for col in SHD_LABEL_COLUMNS:
                clinical_dict[col] = shd_metadata[col].to_numpy()
            
            clinical_frame = pd.DataFrame(clinical_dict)
            record_dir.mkdir(parents=True, exist_ok=True)
            clinical_frame.to_parquet(clinical_path, index=False)
            model_payload["ecgfounder_per_record_parquet"] = str(clinical_path.relative_to(PROJECT_ROOT)) if clinical_path.is_relative_to(PROJECT_ROOT) else str(clinical_path)
            
        payload["models"][spec["id"]] = model_payload
        
        write_json_atomic(partial_path, payload)
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload["shd_reference"] = {
        "clinical": reference_shd_metrics,
        "per_record_parquet": str(
            reference_shd_path.relative_to(PROJECT_ROOT)
            if reference_shd_path.is_relative_to(PROJECT_ROOT)
            else reference_shd_path
        ),
        "classifier": shd_classifier.provenance,
    }
    payload["metadata_waveform_alignment"] = {
        "contract": (
            "PhysioNet EchoNext v1.1.1 states that split-specific NumPy row "
            "order matches metadata row order; test rows are retained in CSV order."
        ),
        "n_rows": len(shd_metadata),
        "ecg_key_order_sha256": ordered_value_hash(
            shd_metadata["ecg_key"].to_numpy()
        ),
        "metadata_original_row_order_sha256": ordered_value_hash(
            shd_metadata["Unnamed: 0"].to_numpy()
        ),
        "required_label_columns": SHD_LABEL_COLUMNS,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--morphology_samples", type=int, default=512)
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--robustness_samples", type=int, default=512)
    parser.add_argument("--robustness_condition_batch", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse complete cells from <output>.partial.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--echonext_model_root",
        type=Path,
        default=Path(
            "ecg_fm_integration/echonext_minimodel_repo/7-EchoNext Minimodel"
        ),
    )
    parser.add_argument(
        "--skip_ecgfounder",
        action="store_true",
        help="Skip optional 150-task probability-fidelity evaluation on EchoNext.",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry_path = registry_path if registry_path.is_absolute() else PROJECT_ROOT / registry_path
    data_dir = args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    registry = json.loads(registry_path.read_text())
    
    missing_checkpoints = [spec["checkpoint"] for spec in registry["models"] if not (PROJECT_ROOT / spec["checkpoint"]).exists()]
    if missing_checkpoints:
        raise FileNotFoundError(f"Missing registered checkpoints: {missing_checkpoints}")
    data, provenance = load_and_validate(data_dir)
    
    classifier = None
    class_names = None
    if "ecgfounder_repo" in registry and not args.skip_ecgfounder:
        from scripts.ecgfounder_classifier import load_ecgfounder, load_task_names
        class_names = load_task_names(PROJECT_ROOT / registry["ecgfounder_tasks"])
        classifier = load_ecgfounder(
            PROJECT_ROOT / registry["ecgfounder_repo"],
            PROJECT_ROOT / registry["ecgfounder_checkpoint"],
            torch.device(args.device),
            n_classes=len(class_names),
        )

    echonext_model_root = (
        args.echonext_model_root
        if args.echonext_model_root.is_absolute()
        else PROJECT_ROOT / args.echonext_model_root
    )
    shd_classifier = EchoNextMiniModel(
        echonext_model_root,
        torch.device(args.device),
    )
    shd_metadata, shd_tabular, shd_labels = load_echonext_test_metadata(
        data_dir / "echonext_metadata_100k.csv",
        shd_classifier.transformer_path,
    )
    if len(shd_metadata) != len(data):
        raise ValueError(
            f"EchoNext metadata/waveform count mismatch: {len(shd_metadata)} vs {len(data)}"
        )

    stress_conditions = []
    nstdb_noises = {}
    if not args.skip_robustness:
        from scripts.robustness_stress import build_conditions, load_nstdb_noise
        stress_conditions = build_conditions(include_fitbit=False)
        if "nstdb_dir" in registry:
            nstdb_noises = load_nstdb_noise(PROJECT_ROOT / registry["nstdb_dir"])

    payload = evaluate(
        data,
        registry,
        torch.device(args.device),
        args.batch_size,
        output_path,
        args.morphology_samples,
        classifier=classifier,
        stress_conditions=stress_conditions,
        nstdb_noises=nstdb_noises,
        robustness_samples=args.robustness_samples,
        robustness_condition_batch=args.robustness_condition_batch,
        data_dir=data_dir,
        resume=args.resume,
        shd_classifier=shd_classifier,
        shd_metadata=shd_metadata,
        shd_tabular=shd_tabular,
        shd_labels=shd_labels,
        ecgfounder_tasks=class_names,
    )
    payload.update({
        "schema_version": 3,
        "protocol": {
            "dataset": "EchoNext",
            "n_samples": len(data),
            "shape": [len(data), 12, 5000],
            "source_sample_rate_hz": 250,
            "evaluation_sample_rate_hz": 500,
            "resampling": "scipy.signal.resample_poly(up=2,down=1)",
            "per_record_minmax": False,
            "morphology_samples_per_model": min(args.morphology_samples, len(data)),
            "streamed_from_memory_map": True,
            "signal_value_space": "physical_mV_after_official_zscore_inversion",
            "shd_classifier": shd_classifier.provenance,
            "provenance": provenance,
        },
    })
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    write_json_atomic(temporary, payload)
    os.replace(temporary, output_path)
    print(f"Saved EchoNext factorial results: {output_path}")


if __name__ == "__main__":
    main()
