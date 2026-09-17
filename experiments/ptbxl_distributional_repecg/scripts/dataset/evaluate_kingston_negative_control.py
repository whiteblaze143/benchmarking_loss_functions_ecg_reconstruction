#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPRESENTATIONS = ("kernel", "gaussian", "linear", "moments")
C_GRID = (0.01, 0.1, 1.0, 10.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_targets(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels).astype(str)
    unexpected = set(values) - {"SINUS", "AFIB_AFLT"}
    if unexpected:
        raise ValueError(f"unexpected Kingston labels: {sorted(unexpected)}")
    return (values == "AFIB_AFLT").astype(np.uint8)


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "positive_records": int(target.sum()),
        "negative_records": int(len(target) - target.sum()),
    }


def _bootstrap(target: np.ndarray, probability: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = []
    while len(draws) < replicates:
        index = rng.integers(0, len(target), len(target))
        if np.unique(target[index]).size == 2:
            draws.append(roc_auc_score(target[index], probability[index]))
    return np.asarray(draws, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")

    train_path = args.representations / "representation_negative_control_kingston_icu_train.npz"
    test_path = args.representations / "representation_negative_control_kingston_icu_test.npz"
    train_manifest_path = args.representations / "kingston_manifest_train.json"
    test_manifest_path = args.representations / "kingston_manifest_test.json"
    train_manifest = json.loads(train_manifest_path.read_text())
    test_manifest = json.loads(test_manifest_path.read_text())
    for manifest, expected in ((train_manifest, "Train"), (test_manifest, "Test")):
        if manifest.get("split") != expected or manifest.get("max_records") is not None:
            raise ValueError(f"Kingston evaluation requires the complete official {expected} build")
        if manifest.get("task_id") != "kingston_rhythm":
            raise ValueError("Kingston manifest has the wrong native task")
    with np.load(train_path) as item:
        train = {name: np.asarray(item[name]) for name in item.files}
    with np.load(test_path) as item:
        test = {name: np.asarray(item[name]) for name in item.files}
    train_y = binary_targets(train["rhythm_labels"])
    test_y = binary_targets(test["rhythm_labels"])
    train_index, validation_index = train_test_split(
        np.arange(len(train_y)), test_size=0.2, random_state=args.seed, stratify=train_y
    )

    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "kind": "kingston_native_rhythm_negative_control",
        "evaluation_role": "zero_padding_negative_control",
        "task_id": "kingston_rhythm",
        "positive_class": "AFIB_AFLT",
        "selection": "stratified 80/20 carve from official Train only",
        "final_fit": "complete eligible official Train split",
        "evaluation": "complete eligible official Test split",
        "record_level_limitation": "Kingston release supplies no patient identifier; confidence intervals resample records",
        "pooling_status": "never pooled with PTB-XL or other native tasks",
        "train_manifest_sha256": _sha256(train_manifest_path),
        "test_manifest_sha256": _sha256(test_manifest_path),
        "representations": {},
    }
    predictions = {
        "ecg_ids": test["ecg_ids"],
        "labels": test_y,
    }
    bootstraps = {}
    for representation in REPRESENTATIONS:
        train_x = train[representation].reshape(len(train_y), -1)
        test_x = test[representation].reshape(len(test_y), -1)
        candidates = []
        for c_value in C_GRID:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, class_weight="balanced", max_iter=2_000, random_state=args.seed, solver="liblinear"),
            )
            model.fit(train_x[train_index], train_y[train_index])
            probability = model.predict_proba(train_x[validation_index])[:, 1]
            candidates.append((roc_auc_score(train_y[validation_index], probability), c_value))
        validation_auroc, selected_c = max(candidates)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=selected_c, class_weight="balanced", max_iter=2_000, random_state=args.seed, solver="liblinear"),
        )
        model.fit(train_x, train_y)
        probability = model.predict_proba(test_x)[:, 1]
        draws = _bootstrap(test_y, probability, args.replicates, args.seed)
        model_path = args.output / f"{representation}_head.joblib"
        joblib.dump(model, model_path)
        results["representations"][representation] = {
            "selected_c": selected_c,
            "selection_auroc": float(validation_auroc),
            "test": _metrics(test_y, probability),
            "test_auroc_ci95": np.quantile(draws, (0.025, 0.975)).tolist(),
            "model_sha256": _sha256(model_path),
        }
        predictions[f"probability_{representation}"] = probability
        bootstraps[f"auroc_{representation}"] = draws
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    np.savez_compressed(args.output / "bootstrap.npz", **bootstraps)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
