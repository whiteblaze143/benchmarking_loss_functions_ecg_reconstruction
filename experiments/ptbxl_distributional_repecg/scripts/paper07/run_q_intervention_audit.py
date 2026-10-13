#!/usr/bin/env python3
"""Stage 2A: frozen operator-response correspondence interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from run_stagewise_linear_probe_audit import (
    CONFIGURATIONS, LEADS, _extract, _load_checkpoint, _macro_auc, _patient_split,
)


SCHEMA_VERSION = "p07_q_intervention_audit_v1"
STAGES = ("E4_contextual_token", "E5_attention_pooled", "E6_latent", "E7_frozen_logits")


def _random_operators(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape).astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def _fit_probe(train_x, train_y, select_x, select_y):
    scaler = StandardScaler().fit(train_x)
    choices = []
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        probe = Ridge(alpha=alpha).fit(scaler.transform(train_x), train_y)
        choices.append((alpha, _macro_auc(select_y, probe.predict(scaler.transform(select_x)))))
    alpha = max(choices, key=lambda item: (item[1], -item[0]))[0]
    scaler = StandardScaler().fit(np.concatenate((train_x, select_x)))
    probe = Ridge(alpha=alpha).fit(scaler.transform(np.concatenate((train_x, select_x))), np.concatenate((train_y, select_y)))
    return scaler, probe, float(alpha), [{"alpha": float(a), "selection_macro_auroc": float(s)} for a, s in choices]


def _predict(scaler, probe, values):
    return probe.predict(scaler.transform(values)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output: {args.output}")
    manifest = json.loads((args.contexts / "manifest.json").read_text())
    if manifest.get("status") != "complete" or tuple(manifest.get("lead_order", ())) != LEADS:
        raise ValueError("requires complete standard full-12 contexts")
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = {key: np.load(args.contexts / f"train_{key}.npy", mmap_mode="r") for key in ("responses", "labels", "patient_ids")}
    validation = {key: np.load(args.contexts / f"validation_{key}.npy", mmap_mode="r") for key in ("responses", "labels", "patient_ids")}
    operators = np.load(args.contexts / "operators.npy", allow_pickle=False)
    fit_mask, select_mask = _patient_split(train["patient_ids"], args.seed)
    model, metadata = _load_checkpoint(args.checkpoint)
    model = model.to(device).eval()
    # The intervention changes q only; the response array is always unchanged.
    interventions = ("correct", "zero", "wrong", "random")
    result = {"schema_version": SCHEMA_VERSION, "status": "complete", "seed": args.seed, "checkpoint": metadata, "protocol": {"encoder": "frozen robust primary", "probe_fit": "correct Q8 only, folds 1-7", "probe_selection": "20% patient-disjoint split", "test_fold": 8, "interventions": "operator only; response fixed", "stages": list(STAGES)}, "results": {}}
    predictions = {"ecg_ids": np.load(args.contexts / "validation_ecg_ids.npy", allow_pickle=False), "patient_ids": validation["patient_ids"], "labels": validation["labels"]}
    train_correct = _extract(model, train["responses"], operators, CONFIGURATIONS["Q8"], device, args.batch)
    select_features = {stage: values[select_mask] for stage, values in train_correct.items()}
    probes = {}
    for stage in STAGES:
        scaler, probe, alpha, candidates = _fit_probe(train_correct[stage][fit_mask], train["labels"][fit_mask], select_features[stage], train["labels"][select_mask])
        probes[stage] = (scaler, probe)
        result["results"][stage] = {"dimension": int(train_correct[stage].shape[1]), "selected_alpha": alpha, "selection_candidates": candidates, "fold8": {}}
    for configuration, indices in CONFIGURATIONS.items():
        base_ops = operators[list(indices)]
        random_ops = _random_operators(base_ops.shape, args.seed + sum(indices) + len(indices))
        intervention_ops = {"correct": base_ops, "zero": np.zeros_like(base_ops), "wrong": np.roll(base_ops, 1, axis=0), "random": random_ops}
        for intervention, ops in intervention_ops.items():
            # Select the same physical response columns as the configuration;
            # only the operator token is intervened on afterward.
            response_subset = validation["responses"][:, list(indices)]
            extracted = _extract(model, response_subset, ops, tuple(range(len(indices))), device, args.batch)
            for stage in STAGES:
                scaler, probe = probes[stage]
                probability = _predict(scaler, probe, extracted[stage])
                key = f"{configuration}__{intervention}__{stage}"
                predictions[key] = probability
                result["results"][stage]["fold8"][f"{configuration}__{intervention}"] = {"macro_auroc": _macro_auc(validation["labels"], probability)}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "fold8_predictions.npz", **predictions)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
