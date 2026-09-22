#!/usr/bin/env python3
"""Frozen-encoder PTB-XL single-lead conditional waveform probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from sklearn.linear_model import Ridge

from repecg.paper07_operator import OperatorSetModel
from scripts.paper07.run_vitaldb_increment_probe import IMQNystromMap, _paired_bootstrap


SCHEMA_VERSION = "paper07_frozen_ptbxl_conditional_probe_v1"
INDEPENDENT_STANDARD_INDICES = np.asarray([0, 1, 6, 7, 8, 9, 10, 11])
INDEPENDENT_NAMES = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
STANDARD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CONTEXTS = {"lead_I": 0, "lead_II": 1}


def _load_contexts(root: Path, split: str) -> dict[str, np.ndarray]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != "paper07_frozen_full12_context_responses_v1" or manifest.get("status") != "complete":
        raise ValueError("requires complete frozen full-12 response contexts")
    return {
        "responses": np.load(root / f"{split}_responses.npy", mmap_mode="r"),
        "operators": np.load(root / "operators.npy", allow_pickle=False),
        "ecg_ids": np.load(root / f"{split}_ecg_ids.npy", allow_pickle=False),
        "patient_ids": np.load(root / f"{split}_patient_ids.npy", allow_pickle=False),
    }


def _load_targets(root: Path, split: str, expected: dict[str, np.ndarray], samples: int) -> np.ndarray:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != "paper07_full12_waveform_targets_v1" or manifest.get("status") != "complete":
        raise ValueError("requires complete full-12 waveform targets")
    ecg_ids = np.load(root / f"{split}_ecg_ids.npy", allow_pickle=False)
    patient_ids = np.load(root / f"{split}_patient_ids.npy", allow_pickle=False)
    if not np.array_equal(ecg_ids, expected["ecg_ids"]) or not np.array_equal(patient_ids, expected["patient_ids"]):
        raise ValueError(f"{split} target/context IDs disagree")
    source = np.load(root / f"{split}_waveforms.npy", mmap_mode="r")
    result = np.empty((len(source), 8, samples), dtype=np.float32)
    for start in range(0, len(source), 256):
        stop = min(start + 256, len(source))
        result[start:stop] = signal.resample(
            np.asarray(source[start:stop, INDEPENDENT_STANDARD_INDICES], dtype=np.float32), samples, axis=-1,
        ).astype(np.float32)
    return result


def _checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    metadata = {
        "path": str(path),
        "variant": payload.get("variant"),
        "schema_version": payload.get("schema_version"),
    }
    if "summary" in payload:
        metadata["training_summary"] = payload["summary"]
    if "val_metrics" in payload:
        metadata["validation_metrics"] = payload["val_metrics"]
    return state, metadata


def _latents(
    checkpoint: Path, data: dict[str, np.ndarray], lead_index: int, device: torch.device, batch: int,
) -> tuple[np.ndarray, dict[str, object]]:
    state, metadata = _checkpoint_state(checkpoint)
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {key for key in missing if key.startswith("decoder.")}
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"checkpoint mismatch for {checkpoint}: missing={missing}, unexpected={unexpected}")
    model.eval()
    result = np.empty((len(data["responses"]), 256), dtype=np.float32)
    operator = torch.as_tensor(data["operators"][lead_index:lead_index + 1], device=device)
    with torch.inference_mode():
        for start in range(0, len(result), batch):
            stop = min(start + batch, len(result))
            responses = torch.as_tensor(
                np.asarray(data["responses"][start:stop, lead_index:lead_index + 1]), device=device,
            )
            operators = operator.expand(stop - start, -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                latent = model.encode_context(operators, responses)
            result[start:stop] = latent.float().cpu().numpy()
    return result, metadata


def _fit_select(
    train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, dict[str, object]]:
    candidates = []
    best = None
    best_prediction = None
    for alpha in alphas:
        prediction = Ridge(alpha=alpha).fit(train_x, train_y).predict(validation_x).astype(np.float32)
        rmse = float(np.sqrt(np.mean(np.square(prediction - validation_y))))
        candidates.append({"alpha": alpha, "validation_rmse": rmse})
        key = (rmse, alpha)
        if best is None or key < best:
            best, best_prediction = key, prediction
    assert best is not None and best_prediction is not None
    return best_prediction, {"selected_alpha": best[1], "candidates": candidates}


def _derive_standard(independent: np.ndarray, observed_index: int, observed: np.ndarray) -> np.ndarray:
    if independent.ndim != 3 or independent.shape[1] != 8:
        raise ValueError("independent prediction must have shape [record,8,time]")
    values = independent.copy()
    values[:, observed_index] = observed
    lead_i, lead_ii = values[:, 0], values[:, 1]
    return np.stack(
        (lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2,
         lead_i - lead_ii / 2, lead_ii - lead_i / 2,
         *(values[:, index] for index in range(2, 8))), axis=1,
    )


def _patient_equal_by_lead(
    target: np.ndarray, prediction: np.ndarray, patient_ids: np.ndarray, heldout: np.ndarray,
) -> dict[str, object]:
    per_lead = {}
    rmse_matrix = np.sqrt(np.mean(np.square(prediction - target), axis=2))
    correlation = np.empty_like(rmse_matrix)
    variance_ratio = np.var(prediction, axis=2) / np.maximum(np.var(target, axis=2), 1e-12)
    for lead in range(target.shape[1]):
        left = target[:, lead] - target[:, lead].mean(axis=1, keepdims=True)
        right = prediction[:, lead] - prediction[:, lead].mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        correlation[:, lead] = np.divide(
            np.sum(left * right, axis=1), denominator, out=np.zeros(len(left)), where=denominator > 1e-12,
        )
    for lead, name in enumerate(STANDARD_NAMES):
        mask = heldout[lead]
        rows = []
        for patient in np.unique(patient_ids):
            index = patient_ids == patient
            rows.append((rmse_matrix[index, lead].mean(), correlation[index, lead].mean(), variance_ratio[index, lead].mean()))
        values = np.asarray(rows)
        per_lead[name] = {
            "heldout": bool(mask), "patients": int(len(values)), "records": int(len(patient_ids)),
            "rmse": float(values[:, 0].mean()), "correlation": float(values[:, 1].mean()),
            "variance_ratio": float(values[:, 2].mean()),
        }
    patient_rmse = []
    for patient in np.unique(patient_ids):
        index = patient_ids == patient
        patient_rmse.append(float(rmse_matrix[index][:, heldout].mean()))
    return {
        "per_lead": per_lead,
        "heldout_macro_patient_equal_rmse": float(np.mean(patient_rmse)),
        "patients": int(len(patient_rmse)), "records": int(len(patient_ids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--landmarks", type=int, default=512)
    parser.add_argument("--c2", type=float, default=1.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output: {args.output}")
    checkpoints = {}
    for specification in args.checkpoint:
        if "=" not in specification:
            raise ValueError("checkpoint must be NAME=PATH")
        name, raw_path = specification.split("=", 1)
        if not name or name in checkpoints:
            raise ValueError(f"invalid or duplicate checkpoint name: {name}")
        checkpoints[name] = Path(raw_path)

    train, validation = _load_contexts(args.contexts, "train"), _load_contexts(args.contexts, "validation")
    train_target = _load_targets(args.targets, "train", train, args.samples)
    validation_target = _load_targets(args.targets, "validation", validation, args.samples)
    scaler = json.loads((args.targets / "manifest.json").read_text())["scaler"]
    target_mean = np.asarray(scaler["mean"])[INDEPENDENT_STANDARD_INDICES]
    target_std = np.asarray(scaler["std"])[INDEPENDENT_STANDARD_INDICES]
    validation_units = validation_target * target_std[None, :, None] + target_mean[None, :, None]
    # This host's cuDNN PhaseCNN path raises ``ptrDesc->finalize()`` on
    # otherwise valid frozen inference. Match the existing P07 trainer.
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = {}
    results = {}
    checkpoint_metadata = {}
    for encoder_name, checkpoint in checkpoints.items():
        results[encoder_name] = {}
        for context_name, lead_index in CONTEXTS.items():
            train_latent, metadata = _latents(checkpoint, train, lead_index, device, args.batch)
            validation_latent, _ = _latents(checkpoint, validation, lead_index, device, args.batch)
            checkpoint_metadata[encoder_name] = metadata
            flat_train_y = train_target.reshape(len(train_target), -1)
            flat_validation_y = validation_target.reshape(len(validation_target), -1)
            ridge_prediction, ridge_selection = _fit_select(
                train_latent, flat_train_y, validation_latent, flat_validation_y, args.alphas,
            )
            mapping = IMQNystromMap.fit(
                train_latent, args.landmarks, args.c2, args.seed + lead_index,
            )
            imq_prediction, imq_selection = _fit_select(
                mapping.transform(train_latent), flat_train_y,
                mapping.transform(validation_latent), flat_validation_y, args.alphas,
            )
            results[encoder_name][context_name] = {}
            observed_independent = lead_index
            heldout_standard = np.ones(12, dtype=bool)
            heldout_standard[lead_index] = False
            for model_name, flat_prediction, selection in (
                ("ridge", ridge_prediction, ridge_selection),
                ("imq_nystrom", imq_prediction, {**imq_selection, "landmarks": args.landmarks, "c2": args.c2}),
            ):
                independent = flat_prediction.reshape(len(validation_target), 8, args.samples)
                independent_units = independent * target_std[None, :, None] + target_mean[None, :, None]
                observed_units = validation_units[:, observed_independent]
                standard_prediction = _derive_standard(independent_units, observed_independent, observed_units)
                standard_target = _derive_standard(
                    validation_units, observed_independent, observed_units,
                )
                key = f"{encoder_name}__{context_name}__{model_name}"
                predictions[key] = standard_prediction.astype(np.float32)
                metrics = _patient_equal_by_lead(
                    standard_target, standard_prediction, validation["patient_ids"], heldout_standard,
                )
                results[encoder_name][context_name][model_name] = {
                    "selection": selection,
                    "selection_fold_metrics": metrics,
                    "interpretation": "fold-8 readout selection only; not test evidence",
                }

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "fold8_predictions.npz",
        ecg_ids=validation["ecg_ids"], patient_ids=validation["patient_ids"],
        **predictions,
    )
    output = {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "folds": {"readout_fit": list(range(1, 8)), "readout_selection": [8], "test": "not_opened"},
        "target": "eight independent leads at 256 samples; dependent limb leads derived by exact algebra",
        "checkpoint_metadata": checkpoint_metadata, "results": results,
        "samples": args.samples, "seed": args.seed,
    }
    (args.output / "results.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "encoders": sorted(checkpoints), "contexts": sorted(CONTEXTS)}))


if __name__ == "__main__":
    main()
