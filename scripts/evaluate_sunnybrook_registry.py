#!/usr/bin/env python3
"""Zero-shot Sunnybrook Cath Lab evaluation for the locked factorial registry.

The Philips Sierra XML waveform is decoded in its declared physical scale
(parsedwaveforms@resolution, microvolts/bit), center-cropped to 10 seconds, and
passed through the same registry adapters and frozen classifiers used by the
PTB-XL factorial benchmark. No per-record amplitude normalization is applied to
the reconstruction metrics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sierraecg
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.clinical_metrics import probability_fidelity_metrics
from scripts.ecgfounder_classifier import (
    load_ecgfounder,
    load_task_names,
    preprocess_ecgfounder,
)
from scripts.evaluate_comprehensive_registry import (
    LEAD_NAMES,
    StreamingSignalMetrics,
    classification_metrics,
    json_safe,
    load_adapter,
    morphology_metrics,
    morphology_record_metrics,
    record_signal_metrics,
)
from scripts.five_superclass_parity import (
    load_classifier as load_five_superclass,
    preprocess as preprocess_five_superclass,
)


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
MI_CODES = {"IMIC", "QMML"}
STTC_CODES = {"MSTEA", "T0DI", "T0IN", "T1IN", "T6AN", "EREPOL", "REPILA", "SD0IN"}
CD_CODES = {"RBBB", "IRBBB", "NIVCD"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_name(element: ET.Element) -> str:
    return element.tag.split("}")[-1].lower()


def text_values(root: ET.Element, name: str) -> list[str]:
    values: list[str] = []
    for element in root.iter():
        if local_name(element) == name.lower():
            value = (element.text or "").strip()
            if value:
                values.append(value)
    return list(dict.fromkeys(values))


def parsed_waveform_resolution_uv(root: ET.Element) -> float:
    for element in root.iter():
        if local_name(element) == "parsedwaveforms":
            value = element.attrib.get("resolution")
            if value is None:
                break
            resolution = float(value)
            if resolution <= 0:
                raise ValueError(f"Invalid parsed-waveform resolution: {resolution}")
            return resolution
    raise ValueError("parsedwaveforms@resolution is missing")


def mapped_superclass_labels(codes: set[str], severities: list[str]) -> np.ndarray:
    """Map high-confidence Philips statements to the frozen five-class head.

    This is a device-statement proxy, not adjudicated clinical ground truth.
    HYP is deliberately left unmapped because the present Philips atrial
    enlargement statements do not establish the PTB-XL hypertrophy superclass.
    """
    labels = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    labels[0] = float(any(value.strip().upper() == "- NORMAL ECG -" for value in severities))
    labels[1] = float(bool(codes & MI_CODES))
    labels[2] = float(bool(codes & STTC_CODES))
    labels[3] = float(bool(codes & CD_CODES))
    labels[4] = 0.0
    return labels


def load_record(xml_path: Path, target_samples: int = 5000) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    resolution_uv = parsed_waveform_resolution_uv(root)
    record = sierraecg.read_file(str(xml_path))
    signal_map = {lead.label: np.asarray(lead.samples, dtype=np.float32) for lead in record.leads}
    missing = [lead for lead in LEAD_NAMES if lead not in signal_map]
    if missing:
        raise ValueError(f"{xml_path.name} missing leads: {missing}")
    rates = {int(lead.sampling_freq) for lead in record.leads}
    if rates != {500}:
        raise ValueError(f"{xml_path.name} unexpected sample rates: {sorted(rates)}")
    lengths = {int(signal_map[lead].shape[0]) for lead in LEAD_NAMES}
    if len(lengths) != 1:
        raise ValueError(f"{xml_path.name} inconsistent lead lengths: {sorted(lengths)}")
    original_samples = lengths.pop()
    if original_samples < target_samples:
        raise ValueError(f"{xml_path.name} has only {original_samples} samples")
    start = (original_samples - target_samples) // 2
    end = start + target_samples
    counts = np.stack([signal_map[lead][start:end] for lead in LEAD_NAMES])
    signal_mv = counts * (resolution_uv / 1000.0)
    codes = set(text_values(root, "statementcode"))
    severities = text_values(root, "severity")
    qa_actions = text_values(root, "qaactioncode")
    return {
        "record_id": xml_path.stem,
        "signal": signal_mv.astype(np.float32),
        "source_sha256": sha256(xml_path),
        "sample_rate_hz": 500,
        "original_samples": original_samples,
        "crop_start": start,
        "crop_end": end,
        "resolution_uv_per_bit": resolution_uv,
        "statement_codes": sorted(codes),
        "severities": severities,
        "qa_actions": qa_actions,
        "superclass_labels": mapped_superclass_labels(codes, severities),
    }


@torch.inference_mode()
def sigmoid_probabilities(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    preprocess,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, inputs.shape[0], batch_size):
        logits = model(preprocess(inputs[start:start + batch_size]))
        chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def dataset_preflight(records: list[dict[str, Any]]) -> dict[str, Any]:
    signals = np.stack([record["signal"] for record in records])
    labels = np.stack([record["superclass_labels"] for record in records])
    resolution_values = sorted({record["resolution_uv_per_bit"] for record in records})
    rates = sorted({record["sample_rate_hz"] for record in records})
    lengths = sorted({record["original_samples"] for record in records})
    counts = {
        CLASS_NAMES[index]: {
            "positive": int(labels[:, index].sum()),
            "negative": int(labels.shape[0] - labels[:, index].sum()),
            "auroc_evaluable": bool(np.unique(labels[:, index]).size == 2),
        }
        for index in range(len(CLASS_NAMES))
    }
    return {
        "status": "pass",
        "n_records": len(records),
        "tensor_shape": list(signals.shape),
        "finite": bool(np.isfinite(signals).all()),
        "sample_rates_hz": rates,
        "source_lengths": lengths,
        "resolution_uv_per_bit": resolution_values,
        "physical_range_mv": [float(signals.min()), float(signals.max())],
        "physical_std_mv": float(signals.std()),
        "superclass_label_counts": counts,
        "superclass_label_provenance": (
            "high-confidence mapping from Philips PageWriter automated statement codes; "
            "not clinician-adjudicated ground truth"
        ),
        "signal_quality_label_status": (
            "not_evaluable: QA action is invariant and not an adjudicated binary label"
        ),
    }


def save_example(
    output_dir: Path,
    model_id: str,
    target: np.ndarray,
    reconstruction: np.ndarray,
) -> str:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [2, 3, 4, 5, 6, 8, 9, 10, 11]
    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
    time = np.arange(target.shape[-1]) / 500.0
    for axis, lead in zip(axes.flat, missing):
        axis.plot(time, target[lead], color="#111827", lw=0.8, label="Philips")
        axis.plot(time, reconstruction[lead], color="#0072B2", lw=0.7, alpha=0.85, label="Recon")
        axis.set_title(LEAD_NAMES[lead])
        axis.grid(alpha=0.15)
    axes.flat[0].legend(frameon=False, ncol=2)
    fig.supxlabel("Time (s)")
    fig.supylabel("mV")
    fig.suptitle(f"Sunnybrook zero-shot reconstruction — {model_id}")
    fig.tight_layout()
    path = output_dir / f"{model_id}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=ROOT / "experiment_queue/factorial_v4/model_registry.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/sunnybrook_12_lead_ecg_samples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sunnybrook_factorial_v4_2x4")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.registry = args.registry.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    registry = json.loads(args.registry.read_text())
    specs = registry["models"]
    if args.models:
        requested = set(args.models)
        specs = [spec for spec in specs if spec["id"] in requested]
        missing = sorted(requested - {spec["id"] for spec in specs})
        if missing:
            raise ValueError(f"Unknown requested model ids: {missing}")
    xml_paths = sorted(args.data_dir.glob("*.xml"))
    records = [load_record(path) for path in tqdm(xml_paths, desc="Sunnybrook XML preflight")]
    if len(records) != 20:
        raise RuntimeError(f"Expected 20 Sunnybrook XML files, found {len(records)}")
    preflight = dataset_preflight(records)
    if not preflight["finite"] or preflight["sample_rates_hz"] != [500]:
        raise RuntimeError(f"Sunnybrook preflight failed: {preflight}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_dir = args.output_dir / "per_record"
    plot_dir = args.output_dir / "plots"
    record_dir.mkdir(parents=True, exist_ok=True)
    signals = torch.from_numpy(np.stack([record["signal"] for record in records]))
    ids = torch.arange(1, len(records) + 1, dtype=torch.long)
    labels = np.stack([record["superclass_labels"] for record in records])
    device = torch.device(args.device)

    tasks = load_task_names(ROOT / registry["ecgfounder_tasks"])
    ecgfounder = load_ecgfounder(
        ROOT / registry["ecgfounder_repo"],
        ROOT / registry["ecgfounder_checkpoint"],
        device,
        len(tasks),
    )
    parity_classifier, parity_classes = load_five_superclass(
        ROOT / registry["five_superclass_backbone"],
        ROOT / registry["five_superclass_checkpoint"],
        device,
    )
    if parity_classes != CLASS_NAMES:
        raise RuntimeError(f"Unexpected five-superclass order: {parity_classes}")
    signals_device = signals.to(device)
    reference_ecgfounder = sigmoid_probabilities(
        ecgfounder, signals_device, preprocess_ecgfounder, args.batch_size
    )
    reference_superclass = sigmoid_probabilities(
        parity_classifier, signals_device, preprocess_five_superclass, args.batch_size
    )
    reference_classification = classification_metrics(reference_superclass, labels, CLASS_NAMES)

    output: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "dataset": "Sunnybrook Cath Lab Clinical Validation Set",
            "evaluation_type": "real_measured_12lead_zero_shot_external_validation",
            "n_records": len(records),
            "sample_rate_hz": 500,
            "source_duration_seconds": 11.0,
            "evaluation_duration_seconds": 10.0,
            "crop": "deterministic_center_crop_250_to_5250",
            "lead_order": LEAD_NAMES,
            "observed_leads": ["I", "II", "V2"],
            "missing_leads": ["III", "aVR", "aVL", "aVF", "V1", "V3", "V4", "V5", "V6"],
            "units": "mV",
            "amplitude_conversion": "decoded_ADC_counts * parsedwaveforms_resolution_uv_per_bit / 1000",
            "normalization_for_signal_metrics": "none",
            "ecgfounder_preprocessing": "official per-record global z-score",
            "five_superclass_preprocessing": "frozen parity-head per-lead z-score",
            "classification_ground_truth": (
                "Philips PageWriter automated statements mapped to high-confidence PTB-XL "
                "superclasses; non-adjudicated proxy labels"
            ),
            "ecgfounder_ground_truth": "unavailable; probability fidelity only",
            "signal_quality_ground_truth": "unavailable; invariant device QA action",
            "registry": str(args.registry.relative_to(ROOT)),
            "registry_sha256": sha256(args.registry),
            "ecgfounder_checkpoint_sha256": sha256(ROOT / registry["ecgfounder_checkpoint"]),
            "five_superclass_checkpoint_sha256": sha256(ROOT / registry["five_superclass_checkpoint"]),
        },
        "preflight": preflight,
        "reference": {
            "five_superclass": reference_classification,
            "ecgfounder_probability_shape": list(reference_ecgfounder.shape),
        },
        "records": [
            {
                key: value
                for key, value in record.items()
                if key not in {"signal", "superclass_labels"}
            }
            | {"superclass_labels": record["superclass_labels"].tolist()}
            for record in records
        ],
        "models": {},
    }

    for spec in tqdm(specs, desc="Sunnybrook models"):
        adapter = load_adapter(spec, device)
        if adapter.observed != [0, 1, 7]:
            raise RuntimeError(f"{spec['id']} observed leads differ: {adapter.observed}")
        reconstructed_parts: list[torch.Tensor] = []
        for start in range(0, signals.shape[0], args.batch_size):
            reconstructed_parts.append(
                adapter.reconstruct(signals_device[start:start + args.batch_size]).cpu()
            )
        reconstruction = torch.cat(reconstructed_parts, dim=0)
        if reconstruction.shape != signals.shape or not torch.isfinite(reconstruction).all():
            raise RuntimeError(
                f"{spec['id']} invalid output {tuple(reconstruction.shape)}, "
                f"finite={bool(torch.isfinite(reconstruction).all())}"
            )
        observed_error = float(
            (reconstruction[:, adapter.observed] - signals[:, adapter.observed]).abs().max()
        )
        if observed_error != 0.0:
            raise RuntimeError(f"{spec['id']} did not preserve observed leads: {observed_error}")

        streaming = StreamingSignalMetrics(adapter.observed)
        streaming.update(signals, reconstruction)
        record_rows = record_signal_metrics(
            signals, reconstruction, ids, adapter.observed, "clean"
        )
        morphology = morphology_metrics(
            list(signals.numpy()),
            list(reconstruction.numpy()),
            adapter.observed,
            list(labels),
        )
        morphology_rows = {
            int(row["ecg_id"]): row
            for row in morphology_record_metrics(
                list(signals.numpy()),
                list(reconstruction.numpy()),
                ids.tolist(),
                adapter.observed,
            )
        }
        reconstructed_device = reconstruction.to(device)
        reconstructed_ecgfounder = sigmoid_probabilities(
            ecgfounder, reconstructed_device, preprocess_ecgfounder, args.batch_size
        )
        reconstructed_superclass = sigmoid_probabilities(
            parity_classifier,
            reconstructed_device,
            preprocess_five_superclass,
            args.batch_size,
        )
        ecgfounder_fidelity = probability_fidelity_metrics(
            reconstructed_ecgfounder, reference_ecgfounder, tasks
        )
        superclass = classification_metrics(
            reconstructed_superclass, labels, CLASS_NAMES
        )
        for index, row in enumerate(record_rows):
            record = records[index]
            row.update({
                key: value
                for key, value in morphology_rows.get(index + 1, {}).items()
                if key != "ecg_id"
            })
            row.update({
                "record_id": record["record_id"],
                "statement_codes": record["statement_codes"],
                "superclass_labels": labels[index].astype(np.float32),
                "superclass_probabilities": reconstructed_superclass[index].astype(np.float32),
                "reference_superclass_probabilities": reference_superclass[index].astype(np.float32),
                "ecgfounder_probabilities": reconstructed_ecgfounder[index].astype(np.float32),
                "reference_ecgfounder_probabilities": reference_ecgfounder[index].astype(np.float32),
            })
        record_path = record_dir / f"{spec['id']}.parquet"
        pd.DataFrame.from_records(record_rows).to_parquet(record_path, index=False)
        example_plot = save_example(
            plot_dir,
            spec["id"],
            signals[0].numpy(),
            reconstruction[0].numpy(),
        )
        output["models"][spec["id"]] = {
            "id": spec["id"],
            "family": spec["family"],
            "kind": spec["kind"],
            "mask": spec["factorial_mask"],
            "factors": spec["factors"],
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": sha256(ROOT / spec["checkpoint"]),
            "observed_leads": spec["observed_leads"],
            "observed_lead_max_abs_error": observed_error,
            "signal": streaming.finalize(),
            "morphology": morphology,
            "five_superclass": superclass,
            "five_superclass_reference": reference_classification,
            "ecgfounder_probability_fidelity": ecgfounder_fidelity,
            "per_record_parquet": str(record_path.relative_to(ROOT)),
            "example_plot": example_plot,
        }
        del adapter, reconstruction, reconstructed_device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output["completeness"] = {
        "expected_models": len(specs),
        "completed_models": len(output["models"]),
        "expected_records_per_model": len(records),
        "completed_records_per_model": {
            model_id: int(pd.read_parquet(ROOT / model["per_record_parquet"]).shape[0])
            for model_id, model in output["models"].items()
        },
        "status": "complete" if len(output["models"]) == len(specs) else "incomplete",
    }
    output_path = args.output_dir / "sunnybrook_results.json"
    output_path.write_text(json.dumps(json_safe(output), indent=2, allow_nan=False))
    (args.output_dir / "preflight.json").write_text(
        json.dumps(json_safe(preflight), indent=2, allow_nan=False)
    )
    print(output_path)


if __name__ == "__main__":
    main()
