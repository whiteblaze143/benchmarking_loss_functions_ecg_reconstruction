#!/usr/bin/env python3
"""Stage 2: frozen configuration/cardinality nuisance probes for P07."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from run_stagewise_linear_probe_audit import (
    CONFIGURATIONS,
    LEADS,
    _extract,
    _load_checkpoint,
    _patient_split,
    _stage_features,
)


SCHEMA_VERSION = "p07_configuration_nuisance_probe_v1"


def _accuracy(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    return float(np.mean(np.argmax(scores, axis=1) == y_true))


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_select: np.ndarray, y_select: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, float]:
    scaler = StandardScaler().fit(x_train)
    candidates = []
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(scaler.transform(x_train), y_train)
        candidates.append((alpha, _accuracy(y_select, model.predict(scaler.transform(x_select)))))
    selected = max(candidates, key=lambda item: (item[1], -item[0]))[0]
    scaler = StandardScaler().fit(np.concatenate((x_train, x_select)))
    model = Ridge(alpha=selected).fit(scaler.transform(np.concatenate((x_train, x_select))), np.concatenate((y_train, y_select)))
    return model.predict(scaler.transform(x_test)), float(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Match the established P07 A100 inference contract.
    torch.backends.cudnn.enabled = False
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output: {args.output}")
    manifest = json.loads((args.contexts / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("requires complete contexts")
    checkpoints = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        checkpoints[name] = Path(path)
    train = {key: np.load(args.contexts / f"train_{key}.npy", mmap_mode="r") for key in ("responses", "labels", "patient_ids")}
    validation = {key: np.load(args.contexts / f"validation_{key}.npy", mmap_mode="r") for key in ("responses", "labels", "patient_ids")}
    operators = np.load(args.contexts / "operators.npy", allow_pickle=False)
    fit_mask, select_mask = _patient_split(train["patient_ids"], args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": SCHEMA_VERSION, "status": "complete", "seed": args.seed, "protocol": {"encoder": "frozen", "probe": "multi-output ridge", "fit_folds": list(range(1, 8)), "selection": "20% patient-disjoint split", "test_fold": 8}, "models": {}}
    for name, path in checkpoints.items():
        model, metadata = _load_checkpoint(path)
        model = model.cuda().eval()
        # Keep only one configuration in RAM at a time.  The full design is
        # written as temporary .npy arrays and each stage is fitted separately;
        # retaining all stages/configurations caused an avoidable OOM.
        with tempfile.TemporaryDirectory(prefix="p07_nuisance_") as temp_name:
            temp = Path(temp_name)
            for cfg, indices in CONFIGURATIONS.items():
                train_features = _extract(model, train["responses"], operators, indices, torch.device("cuda"), args.batch)
                validation_features = _extract(model, validation["responses"], operators, indices, torch.device("cuda"), args.batch)
                for stage, values in train_features.items():
                    np.save(temp / f"train_{cfg}_{stage}.npy", values)
                    np.save(temp / f"validation_{cfg}_{stage}.npy", validation_features[stage])
                del train_features, validation_features
            model_result = {"checkpoint": metadata, "stages": {}}
            stage_names = tuple(_stage_features(model, torch.zeros(1, 1, 8, device="cuda"), torch.zeros(1, 1, 16, 128, device="cuda")))
            cardinality = np.asarray([len(CONFIGURATIONS[cfg]) for cfg in CONFIGURATIONS], dtype=np.int64)
            config_y = np.eye(len(CONFIGURATIONS), dtype=np.float32)
            card_values = sorted(set(cardinality.tolist()))
            card_y = np.eye(len(card_values), dtype=np.float32)[np.searchsorted(card_values, cardinality)]
            y_config_train = np.concatenate([np.repeat(config_y[i:i + 1], fit_mask.sum(), axis=0) for i in range(len(CONFIGURATIONS))])
            y_config_select = np.concatenate([np.repeat(config_y[i:i + 1], select_mask.sum(), axis=0) for i in range(len(CONFIGURATIONS))])
            y_config_test = np.concatenate([np.repeat(config_y[i:i + 1], len(validation["labels"]), axis=0) for i in range(len(CONFIGURATIONS))])
            y_card_train = np.concatenate([np.repeat(card_y[i:i + 1], fit_mask.sum(), axis=0) for i in range(len(CONFIGURATIONS))])
            y_card_select = np.concatenate([np.repeat(card_y[i:i + 1], select_mask.sum(), axis=0) for i in range(len(CONFIGURATIONS))])
            y_card_test = np.concatenate([np.repeat(card_y[i:i + 1], len(validation["labels"]), axis=0) for i in range(len(CONFIGURATIONS))])
            for stage in stage_names:
                x_train = np.concatenate([np.load(temp / f"train_{cfg}_{stage}.npy", mmap_mode="r")[fit_mask] for cfg in CONFIGURATIONS])
                x_select = np.concatenate([np.load(temp / f"train_{cfg}_{stage}.npy", mmap_mode="r")[select_mask] for cfg in CONFIGURATIONS])
                x_test = np.concatenate([np.load(temp / f"validation_{cfg}_{stage}.npy", mmap_mode="r") for cfg in CONFIGURATIONS])
                config_scores, config_alpha = _fit_predict(x_train, y_config_train, x_select, y_config_select, x_test)
                card_scores, card_alpha = _fit_predict(x_train, y_card_train, x_select, y_card_select, x_test)
                model_result["stages"][stage] = {"dimension": int(x_train.shape[1]), "configuration_accuracy": _accuracy(np.argmax(y_config_test, axis=1), config_scores), "cardinality_accuracy": _accuracy(np.argmax(y_card_test, axis=1), card_scores), "configuration_alpha": config_alpha, "cardinality_alpha": card_alpha, "chance_configuration": 1.0 / len(CONFIGURATIONS), "chance_cardinality": 1.0 / len(card_values)}
            result["models"][name] = model_result
        result["models"][name] = model_result
        del model
        torch.cuda.empty_cache()
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
