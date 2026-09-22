#!/usr/bin/env python3
"""Patient-clustered paired bootstrap for original versus time-indexed P07-A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as item:
        required = {"probability", "labels", "ecg_ids", "patient_ids"}
        if missing := required - set(item.files):
            raise ValueError(f"{path} missing {sorted(missing)}")
        return {name: np.asarray(item[name]) for name in required}


def _auroc(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return np.asarray([roc_auc_score(labels[:, index], probability[:, index]) for index in range(labels.shape[1])])


def _paired_bootstrap(labels: np.ndarray, original: np.ndarray, time_indexed: np.ndarray, patients: np.ndarray, seed: int, replicates: int) -> dict[str, object]:
    unique = np.unique(patients)
    groups = [np.flatnonzero(patients == patient) for patient in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, labels.shape[1]), dtype=np.float64)
    for replicate in range(replicates):
        chosen = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[index] for index in chosen])
        draws[replicate] = _auroc(labels[indices], time_indexed[indices]) - _auroc(labels[indices], original[indices])
    point = _auroc(labels, time_indexed) - _auroc(labels, original)
    return {
        "macro": {
            "delta_macro_auroc": float(point.mean()),
            "bootstrap_ci95_low": float(np.quantile(draws.mean(axis=1), 0.025)),
            "bootstrap_ci95_high": float(np.quantile(draws.mean(axis=1), 0.975)),
            "p_time_indexed_gt_original": float(np.mean(draws.mean(axis=1) > 0.0)),
        },
        "per_superclass": {
            name: {
                "delta_auroc": float(point[index]),
                "bootstrap_ci95_low": float(np.quantile(draws[:, index], 0.025)),
                "bootstrap_ci95_high": float(np.quantile(draws[:, index], 0.975)),
                "p_time_indexed_gt_original": float(np.mean(draws[:, index] > 0.0)),
            }
            for index, name in enumerate(SUPERCLASSES)
        },
        "patients": int(len(unique)), "records": int(len(labels)), "bootstrap_replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--time-indexed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output: {args.output}")
    original, time_indexed = _load(args.original), _load(args.time_indexed)
    for name in ("ecg_ids", "patient_ids", "labels"):
        if not np.array_equal(original[name], time_indexed[name]):
            raise ValueError(f"prediction alignment failed for {name}")
    labels = original["labels"]
    if labels.shape != original["probability"].shape or labels.shape != time_indexed["probability"].shape:
        raise ValueError("label/probability shape mismatch")
    original_auc, time_auc = _auroc(labels, original["probability"]), _auroc(labels, time_indexed["probability"])
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "fold8_paired_predictions.npz",
        ecg_ids=original["ecg_ids"], patient_ids=original["patient_ids"], labels=labels,
        original_probability=original["probability"], time_indexed_probability=time_indexed["probability"],
    )
    result = {
        "schema_version": "paper07_a_paired_patient_bootstrap_v1", "status": "complete",
        "fold": 8, "seed": args.seed,
        "original": {"macro_auroc": float(original_auc.mean()), "superclass_auroc": dict(zip(SUPERCLASSES, original_auc.tolist(), strict=True))},
        "time_indexed": {"macro_auroc": float(time_auc.mean()), "superclass_auroc": dict(zip(SUPERCLASSES, time_auc.tolist(), strict=True))},
        "paired_patient_bootstrap": _paired_bootstrap(labels, original["probability"], time_indexed["probability"], original["patient_ids"], args.seed, args.bootstrap),
        "raw_predictions": "fold8_paired_predictions.npz",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["paired_patient_bootstrap"]["macro"], sort_keys=True))


if __name__ == "__main__":
    main()
