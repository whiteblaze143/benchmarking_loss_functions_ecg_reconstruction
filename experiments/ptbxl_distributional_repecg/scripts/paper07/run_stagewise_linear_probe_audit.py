#!/usr/bin/env python3
"""Frozen P07 full-Q8 to sparse linear-probe transfer audit.

Fits each linear probe only on canonical-Q8 representations from folds 1--7.
An internal patient-disjoint split of that pool selects C; fold 8 is never used
for selection.  The same fitted Q8 probe is then evaluated unchanged on each
standard sparse configuration at every fixed-width representation stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from repecg.paper07_operator import OperatorSetModel


SCHEMA_VERSION = "p07_stagewise_linear_probe_audit_v1"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CONFIGURATIONS = {
    "Q8": (0, 1, 6, 7, 8, 9, 10, 11),
    "S6_precordial": (6, 7, 8, 9, 10, 11),
    "S6_limb": (0, 1, 2, 3, 4, 5),
    "S3_icu_v1": (0, 1, 6),
    "S3_icu_v5": (0, 1, 10),
    "S2_bipolar": (0, 1),
    "S1_lead_I": (0,),
    "S1_lead_II": (1,),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _macro_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    values = []
    for column in range(labels.shape[1]):
        if len(np.unique(labels[:, column])) == 2:
            values.append(roc_auc_score(labels[:, column], probability[:, column]))
    if len(values) != labels.shape[1]:
        raise ValueError("each superclass must have both outcomes")
    return float(np.mean(values))


def _patient_split(patient_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    patients = np.unique(patient_ids)
    order = sorted(
        range(len(patients)),
        key=lambda index: hashlib.sha256(f"p07-probe:{seed}:{int(patients[index])}".encode()).digest(),
    )
    selected = set(patients[np.asarray(order[:max(1, round(len(order) * 0.2))])].tolist())
    validation = np.fromiter((value in selected for value in patient_ids), dtype=bool, count=len(patient_ids))
    if not validation.any() or validation.all():
        raise ValueError("invalid patient split")
    return ~validation, validation


def _load_checkpoint(path: Path) -> tuple[OperatorSetModel, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    variant = str(payload.get("variant", ""))
    response_mode = payload.get("response_mode", "linear" if "linear_response" in variant or "linear_bottleneck" in variant or "linear" == variant else "phase_cnn")
    model = OperatorSetModel(
        response_dim=128, classes=5,
        operator_mode="raw_q" if variant in {"linear_q_encoder", "raw_q_from_set_operator_robust"} else "continuous",
        response_mode=response_mode,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {"path": str(path), "sha256": _sha256(path), "variant": payload.get("variant"), "response_mode": response_mode}


def _stage_features(model: OperatorSetModel, operators: torch.Tensor, responses: torch.Tensor) -> dict[str, torch.Tensor]:
    """Fixed-width, mean-token summaries at stages before learned pooling."""
    batch, count = operators.shape[:2]
    q = model.encode_operator(operators, None)
    phase = model.response(responses.reshape(batch * count, *responses.shape[2:])).reshape(batch, count, 128)
    token = torch.cat((q, phase), dim=-1)
    contextual = model.set_encoder(token)
    query = model.pool_query.expand(batch, -1, -1)
    pooled, _ = model.pool(query, contextual, contextual)
    latent = model.latent(pooled[:, 0])
    logits = model.head(latent)
    return {
        # E0 is phase KME before PhaseCNN, averaged over observed operators.
        "E0_mean_phase_kme": responses.mean(dim=1).flatten(1),
        "E1_response_embedding": phase.mean(dim=1),
        "E2_operator_embedding": q.mean(dim=1),
        "E3_combined_token": token.mean(dim=1),
        "E4_contextual_token": contextual.mean(dim=1),
        "E5_attention_pooled": pooled[:, 0],
        "E6_latent": latent,
        "E7_frozen_logits": logits,
    }


def _extract(
    model: OperatorSetModel, responses: np.ndarray, operators: np.ndarray, indices: tuple[int, ...], device: torch.device, batch: int,
) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {}
    with torch.inference_mode():
        for start in range(0, len(responses), batch):
            stop = min(start + batch, len(responses))
            response = torch.as_tensor(np.asarray(responses[start:stop, indices]), device=device)
            operator = torch.as_tensor(np.asarray(operators[list(indices)]), device=device).expand(stop - start, -1, -1)
            values = _stage_features(model, operator, response)
            for name, value in values.items():
                result.setdefault(name, []).append(value.float().cpu().numpy())
    return {name: np.concatenate(chunks).astype(np.float32) for name, chunks in result.items()}


def _fit_probe(
    train_x: np.ndarray, train_y: np.ndarray, select_x: np.ndarray, select_y: np.ndarray, c_values: tuple[float, ...], seed: int,
) -> tuple[StandardScaler, Ridge, float, list[dict[str, float]]]:
    candidates = []
    for c_value in c_values:
        scaler = StandardScaler().fit(train_x)
        model = Ridge(alpha=1.0 / c_value).fit(scaler.transform(train_x), train_y)
        probability = model.predict(scaler.transform(select_x))
        candidates.append({"C": c_value, "selection_macro_auroc": _macro_auc(select_y, probability)})
    selected = max(candidates, key=lambda row: (row["selection_macro_auroc"], -row["C"]))["C"]
    scaler = StandardScaler().fit(np.concatenate((train_x, select_x)))
    fit_x = scaler.transform(np.concatenate((train_x, select_x)))
    fit_y = np.concatenate((train_y, select_y))
    model = Ridge(alpha=1.0 / selected).fit(fit_x, fit_y)
    return scaler, model, float(selected), candidates


def _predict(scaler: StandardScaler, model: Ridge, values: np.ndarray) -> np.ndarray:
    standardized = scaler.transform(values)
    return model.predict(standardized).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-values", type=float, nargs="+", default=(1e-3, 1e-2, 1e-1, 1.0, 10.0))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output: {args.output}")
    manifest = json.loads((args.contexts / "manifest.json").read_text())
    if manifest.get("schema_version") != "paper07_frozen_full12_context_responses_v1" or manifest.get("status") != "complete":
        raise ValueError("requires complete frozen full-12 response contexts")
    if tuple(manifest.get("lead_order", ())) != LEADS:
        raise ValueError("unexpected standard-lead order")
    checkpoints = {}
    for item in args.checkpoint:
        if "=" not in item:
            raise ValueError("checkpoint must be NAME=PATH")
        name, raw = item.split("=", 1)
        if not name or name in checkpoints:
            raise ValueError("checkpoint names must be unique and nonempty")
        checkpoints[name] = Path(raw)
    if set(checkpoints) != {"robust", "aux", "fulllead"}:
        raise ValueError("requires exactly robust, aux, and fulllead checkpoints")
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = {}
    for split in ("train", "validation"):
        payload[split] = {
            "responses": np.load(args.contexts / f"{split}_responses.npy", mmap_mode="r"),
            "labels": np.load(args.contexts / f"{split}_labels.npy", allow_pickle=False),
            "patient_ids": np.load(args.contexts / f"{split}_patient_ids.npy", allow_pickle=False),
            "ecg_ids": np.load(args.contexts / f"{split}_ecg_ids.npy", allow_pickle=False),
        }
    operators = np.load(args.contexts / "operators.npy", allow_pickle=False)
    fit_mask, select_mask = _patient_split(payload["train"]["patient_ids"], args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "seed": args.seed,
        "protocol": {
            "encoder": "frozen", "probe": "standardized multi-output L2 ridge linear probe",
            "probe_fit_folds": [1, 2, 3, 4, 5, 6, 7], "probe_selection": "20% deterministic patient-disjoint split of folds 1-7",
            "sealed_test_fold": 8, "source_train_configuration": "Q8 only",
            "sparse_evaluation": "same Q8-trained probe; no re-fit or calibration",
            "early_stages": "mean over observed tokens to retain fixed width under changing cardinality",
        },
        "contexts_manifest_sha256": _sha256(args.contexts / "manifest.json"),
        "configurations": {name: {"lead_indices": list(indices), "leads": [LEADS[index] for index in indices]} for name, indices in CONFIGURATIONS.items()},
        "models": {},
    }
    all_predictions: dict[str, np.ndarray] = {"ecg_ids": payload["validation"]["ecg_ids"], "patient_ids": payload["validation"]["patient_ids"], "labels": payload["validation"]["labels"]}
    for name, path in checkpoints.items():
        model, metadata = _load_checkpoint(path)
        model = model.to(device)
        features_train = _extract(model, payload["train"]["responses"], operators, CONFIGURATIONS["Q8"], device, args.batch)
        features_validation = {configuration: _extract(model, payload["validation"]["responses"], operators, indices, device, args.batch) for configuration, indices in CONFIGURATIONS.items()}
        model_summary: dict[str, object] = {"checkpoint": metadata, "stages": {}}
        for stage, values in features_train.items():
            scaler, probe, selected_c, candidates = _fit_probe(values[fit_mask], payload["train"]["labels"][fit_mask], values[select_mask], payload["train"]["labels"][select_mask], tuple(args.c_values), args.seed)
            stage_result = {"dimension": int(values.shape[1]), "selected_C": selected_c, "selection_candidates": candidates, "fold8": {}}
            for configuration, extracted in features_validation.items():
                probability = _predict(scaler, probe, extracted[stage])
                all_predictions[f"{name}__{stage}__{configuration}"] = probability
                stage_result["fold8"][configuration] = {"macro_auroc": _macro_auc(payload["validation"]["labels"], probability)}
            model_summary["stages"][stage] = stage_result
        summary["models"][name] = model_summary
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(args.output / "fold8_predictions.npz", **all_predictions)
    (args.output / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output), "models": sorted(checkpoints)}))


if __name__ == "__main__":
    main()
