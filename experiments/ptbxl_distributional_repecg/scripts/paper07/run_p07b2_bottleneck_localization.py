#!/usr/bin/env python3
"""Frozen P07-B2 ridge probes that localize loss of waveform information.

The extractor exposes the exact single-lead path at six points.  It neither
trains nor modifies either encoder; fold 8 is used solely to select ridge
regularization, matching P07-B.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

from repecg.paper07_operator import OperatorSetModel
from scripts.paper07.run_frozen_ptbxl_conditional_probe import (
    CONTEXTS, INDEPENDENT_NAMES, INDEPENDENT_STANDARD_INDICES, _checkpoint_state, _load_contexts, _load_targets,
)


LEVELS = ("Z0_raw_waveform", "Z1_phase_kme", "Z3_phasecnn", "Z4_phase_averaged", "Z5_lead_token", "Z6_pooled_latent")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 1e-12 else 0.0


def _extract(
    checkpoint: Path, data: dict[str, np.ndarray], lead_index: int, device: torch.device, batch: int,
) -> dict[str, np.ndarray]:
    state, _ = _checkpoint_state(checkpoint)
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {key for key in missing if key.startswith("decoder.")} or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    count = len(data["responses"])
    dimensions = {"Z1_phase_kme": 2048, "Z3_phasecnn": 2048, "Z4_phase_averaged": 128, "Z5_lead_token": 192, "Z6_pooled_latent": 256}
    result = {name: np.empty((count, width), dtype=np.float32) for name, width in dimensions.items()}
    operator = torch.as_tensor(data["operators"][lead_index:lead_index + 1], device=device)
    with torch.inference_mode():
        for start in range(0, count, batch):
            stop = min(start + batch, count)
            response = torch.as_tensor(np.asarray(data["responses"][start:stop, lead_index:lead_index + 1]), device=device)
            responses = response[:, 0]
            operators = operator.expand(stop - start, -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                # PhaseCNN is written as a single forward call, so expose the
                # pre-pooling tensor without altering its computation.
                convolution = model.response.blocks(model.response.input(responses.transpose(1, 2)))
                phasecnn = convolution.transpose(1, 2)
                phase_averaged = model.response.head(convolution.mean(dim=-1))
                q = model.encode_operator(operators, None)[:, 0]
                token = torch.cat((q, phase_averaged), dim=-1)
                encoded = model.set_encoder(token[:, None])
                pooled, _ = model.pool(model.pool_query.expand(stop - start, -1, -1), encoded, encoded)
                latent = model.latent(pooled[:, 0])
            result["Z1_phase_kme"][start:stop] = responses.flatten(1).float().cpu().numpy()
            result["Z3_phasecnn"][start:stop] = phasecnn.flatten(1).float().cpu().numpy()
            result["Z4_phase_averaged"][start:stop] = phase_averaged.float().cpu().numpy()
            result["Z5_lead_token"][start:stop] = token.float().cpu().numpy()
            result["Z6_pooled_latent"][start:stop] = latent.float().cpu().numpy()
    return result


def _select_predict(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, alphas: list[float]) -> tuple[np.ndarray, dict[str, object]]:
    candidates: list[dict[str, float]] = []
    best: tuple[float, float] | None = None
    prediction: np.ndarray | None = None
    for alpha in alphas:
        candidate = Ridge(alpha=alpha).fit(train_x, train_y).predict(val_x).astype(np.float32)
        rmse = float(np.sqrt(np.mean((candidate - val_y) ** 2)))
        candidates.append({"alpha": alpha, "selection_rmse_standardized": rmse})
        if best is None or (rmse, alpha) < best:
            best, prediction = (rmse, alpha), candidate
    assert best is not None and prediction is not None
    return prediction, {"selected_alpha": best[1], "candidates": candidates}


def _rows(
    prediction: np.ndarray, target: np.ndarray, patient_ids: np.ndarray, target_leads: list[int],
    variant: str, context: str, level: str, scope: str, mean: np.ndarray, scale: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for patient in np.unique(patient_ids):
        index = np.flatnonzero(patient_ids == patient)
        for column, lead in enumerate(target_leads):
            true = target[index, lead] * scale[lead] + mean[lead]
            guess = prediction[index, column] * scale[lead] + mean[lead]
            true_variance, predicted_variance = float(np.var(true)), float(np.var(guess))
            rows.append({
                "patient_id": int(patient), "context_lead": context, "target_lead": INDEPENDENT_NAMES[lead],
                "encoder_variant": variant, "level": level, "scope": scope,
                "mae_mv": float(np.mean(np.abs(guess - true))), "rmse_mv": float(np.sqrt(np.mean((guess - true) ** 2))),
                "pearson_r": float(np.mean([_correlation(a, b) for a, b in zip(true, guess)])),
                "true_variance": true_variance, "pred_variance": predicted_variance,
                "variance_ratio": predicted_variance / max(true_variance, 1e-12),
            })
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, float]:
    true_variance = np.asarray([row["true_variance"] for row in rows], dtype=float)
    predicted_variance = np.asarray([row["pred_variance"] for row in rows], dtype=float)
    return {
        "patients": len({int(row["patient_id"]) for row in rows}),
        "mae_mv": float(np.mean([float(row["mae_mv"]) for row in rows])),
        "rmse_mv": float(np.mean([float(row["rmse_mv"]) for row in rows])),
        "pearson_r": float(np.mean([float(row["pearson_r"]) for row in rows])),
        "variance_ratio": float(np.mean([float(row["variance_ratio"]) for row in rows])),
        "patient_variance_correlation": _correlation(true_variance, predicted_variance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-contexts", type=Path, required=True)
    parser.add_argument("--time-contexts", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--time-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = {
        "original_kme": (args.original_contexts, args.original_checkpoint),
        "time_indexed_kme": (args.time_contexts, args.time_checkpoint),
    }
    all_rows: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    scaler = json.loads((args.targets / "manifest.json").read_text())["scaler"]
    mean = np.asarray(scaler["mean"], dtype=np.float32)[INDEPENDENT_STANDARD_INDICES]
    scale = np.asarray(scaler["std"], dtype=np.float32)[INDEPENDENT_STANDARD_INDICES]
    for variant, (context_root, checkpoint) in variants.items():
        train, validation = _load_contexts(context_root, "train"), _load_contexts(context_root, "validation")
        train_y = _load_targets(args.targets, "train", train, args.samples)
        validation_y = _load_targets(args.targets, "validation", validation, args.samples)
        for context_name, lead_index in CONTEXTS.items():
            train_features = _extract(checkpoint, train, lead_index, device, args.batch)
            validation_features = _extract(checkpoint, validation, lead_index, device, args.batch)
            train_features["Z0_raw_waveform"] = train_y[:, lead_index]
            validation_features["Z0_raw_waveform"] = validation_y[:, lead_index]
            heldout = [lead for lead in range(8) if lead != lead_index]
            for level in LEVELS:
                for scope, leads in (("self", [lead_index]), ("heldout", heldout)):
                    prediction, selection = _select_predict(
                        train_features[level], train_y[:, leads].reshape(len(train_y), -1),
                        validation_features[level], validation_y[:, leads].reshape(len(validation_y), -1), args.alphas,
                    )
                    prediction = prediction.reshape(len(validation_y), len(leads), args.samples)
                    key = f"{variant}__{context_name}__{level}__{scope}"
                    selections[key] = selection
                    all_rows.extend(_rows(prediction, validation_y, validation["patient_ids"], leads, variant, context_name, level, scope, mean, scale))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "patient_level_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in all_rows:
        key = "__".join(str(row[name]) for name in ("encoder_variant", "context_lead", "level", "scope"))
        grouped.setdefault(key, []).append(row)
    result = {
        "schema_version": "p07_b2_frozen_bottleneck_localization_v1", "status": "complete",
        "folds": {"readout_fit": list(range(1, 8)), "selection": [8], "test": "not_opened"},
        "records": int(len(next(iter(variants.values())) and _load_contexts(args.original_contexts, "validation")["ecg_ids"])),
        "levels": list(LEVELS), "ridge_selections": selections,
        "metrics": {key: _summary(rows) for key, rows in grouped.items()},
        "patient_table": "patient_level_metrics.csv",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "rows": len(all_rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
