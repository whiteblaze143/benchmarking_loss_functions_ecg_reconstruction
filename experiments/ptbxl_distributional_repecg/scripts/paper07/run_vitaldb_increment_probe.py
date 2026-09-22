#!/usr/bin/env python3
"""Subject-disjoint VitalDB test of incremental PPG information for II -> V5."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.linear_model import Ridge


SCHEMA_VERSION = "paper07_vitaldb_ppg_increment_v1"
ARMS = ("ii", "ii_ppg", "ppg")
MODEL_KINDS = ("ridge", "imq_nystrom")


def _subject_split(subject_id: str, seed: int) -> str:
    token = hashlib.sha256(f"p07-vitaldb:{seed}:{subject_id}".encode()).digest()
    value = int.from_bytes(token[:8], "big") / 2**64
    return "train" if value < 0.70 else "validation" if value < 0.85 else "test"


def _load_records(root: Path, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted((root / "records").glob("case_*.npz"))
    if not paths:
        raise ValueError(f"no VitalDB records under {root}")
    waveforms = np.empty((len(paths), 3, samples), dtype=np.float32)
    subjects = np.empty(len(paths), dtype="U32")
    case_ids = np.empty(len(paths), dtype=np.int64)
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as item:
            raw = np.asarray(item["waveforms"], dtype=np.float64)
            if raw.shape != (3, 5000) or int(item["sample_rate_hz"]) != 500:
                raise ValueError(f"invalid waveform contract: {path}")
            if tuple(item["tracks"].tolist()) != (
                "SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/PLETH",
            ):
                raise ValueError(f"invalid track order: {path}")
            if not np.isfinite(raw).all():
                raise ValueError(f"non-finite waveform: {path}")
            # ADC baselines are large device offsets. Linear detrending is
            # record-local and therefore cannot leak held-out subjects.
            detrended = signal.detrend(raw, axis=-1, type="linear")
            waveforms[index] = signal.resample(detrended, samples, axis=-1).astype(np.float32)
            subjects[index] = str(item["subject_id"])
            case_ids[index] = int(item["case_id"])
    return waveforms, subjects, case_ids


def _split_indices(subjects: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    assignment = {subject: _subject_split(str(subject), seed) for subject in np.unique(subjects)}
    result = {
        split: np.flatnonzero(np.asarray([assignment[str(subject)] == split for subject in subjects]))
        for split in ("train", "validation", "test")
    }
    if any(len(indices) == 0 for indices in result.values()):
        raise ValueError(f"empty subject split: { {key: len(value) for key, value in result.items()} }")
    subject_sets = {key: set(subjects[value]) for key, value in result.items()}
    if any(subject_sets[left] & subject_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("subject split overlap")
    return result


def _standardize(
    waveforms: np.ndarray, train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = waveforms[train_indices].astype(np.float64)
    mean = train.mean(axis=(0, 2))
    std = train.std(axis=(0, 2))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 1e-8):
        raise ValueError("invalid train-only channel scaler")
    normalized = (waveforms - mean[None, :, None]) / std[None, :, None]
    return normalized.astype(np.float32), mean, std


def _features(waveforms: np.ndarray, arm: str) -> np.ndarray:
    if arm == "ii":
        selected = waveforms[:, [0]]
    elif arm == "ii_ppg":
        selected = waveforms[:, [0, 2]]
    elif arm == "ppg":
        selected = waveforms[:, [2]]
    else:
        raise ValueError(f"unknown arm {arm}")
    return selected.reshape(len(selected), -1)


class IMQNystromMap:
    def __init__(self, landmarks: np.ndarray, inverse_root: np.ndarray, c2: float):
        self.landmarks = landmarks
        self.inverse_root = inverse_root
        self.c2 = c2

    @staticmethod
    def _kernel(left: np.ndarray, right: np.ndarray, c2: float) -> np.ndarray:
        left_norm = np.square(left).sum(axis=1, keepdims=True)
        right_norm = np.square(right).sum(axis=1, keepdims=True).T
        distance = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0) / left.shape[1]
        return np.power(distance + c2, -0.5)

    @classmethod
    def fit(cls, values: np.ndarray, landmarks: int, c2: float, seed: int) -> "IMQNystromMap":
        if landmarks > len(values):
            raise ValueError("more landmarks than training observations")
        rng = np.random.default_rng(seed)
        selected = values[rng.choice(len(values), size=landmarks, replace=False)].astype(np.float64)
        gram = cls._kernel(selected, selected, c2)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        floor = max(float(eigenvalues[-1]) * 1e-8, 1e-10)
        inverse_root = (eigenvectors * np.maximum(eigenvalues, floor) ** -0.5) @ eigenvectors.T
        return cls(selected, inverse_root, c2)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (self._kernel(values.astype(np.float64), self.landmarks, self.c2) @ self.inverse_root).astype(np.float32)


def _fit_selected_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    alphas: list[float],
) -> tuple[Ridge, float, list[dict[str, float]]]:
    candidates = []
    best: tuple[float, float] | None = None
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(train_x, train_y)
        rmse = float(np.sqrt(np.mean(np.square(model.predict(validation_x) - validation_y))))
        candidates.append({"alpha": alpha, "validation_rmse": rmse})
        key = (rmse, alpha)
        if best is None or key < best:
            best = key
    assert best is not None
    selected_alpha = best[1]
    return Ridge(alpha=selected_alpha).fit(train_x, train_y), selected_alpha, candidates


def _wrong_subject_permutation(subjects: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    for shift in range(1, len(subjects)):
        candidate = np.roll(order, shift)
        inverse = np.empty_like(order)
        inverse[order] = candidate
        if np.all(subjects[inverse] != subjects):
            return inverse
    raise RuntimeError("could not construct a wrong-subject PPG derangement")


def _record_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    residual = prediction - target
    rmse = np.sqrt(np.mean(np.square(residual), axis=1))
    mae = np.mean(np.abs(residual), axis=1)
    centered_target = target - target.mean(axis=1, keepdims=True)
    centered_prediction = prediction - prediction.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(centered_target, axis=1) * np.linalg.norm(centered_prediction, axis=1)
    correlation = np.divide(
        np.sum(centered_target * centered_prediction, axis=1), denominator,
        out=np.zeros(len(target), dtype=np.float64), where=denominator > 1e-12,
    )
    variance_ratio = np.var(prediction, axis=1) / np.maximum(np.var(target, axis=1), 1e-12)
    return {"rmse": rmse, "mae": mae, "correlation": correlation, "variance_ratio": variance_ratio}


def _subject_means(values: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, subject in zip(values, subjects, strict=True):
        grouped[str(subject)].append(float(value))
    keys = np.asarray(sorted(grouped))
    return keys, np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64)


def _summarize(target: np.ndarray, prediction: np.ndarray, subjects: np.ndarray) -> dict[str, object]:
    metrics = _record_metrics(target, prediction)
    result: dict[str, object] = {}
    for name, values in metrics.items():
        _, subject_values = _subject_means(values, subjects)
        result[name] = {
            "patient_equal_mean": float(subject_values.mean()),
            "patient_equal_std": float(subject_values.std(ddof=1)),
            "patients": int(len(subject_values)),
            "records": int(len(values)),
        }
    true_range = np.ptp(target, axis=1)
    predicted_range = np.ptp(prediction, axis=1)
    _, true_subject = _subject_means(true_range, subjects)
    _, predicted_subject = _subject_means(predicted_range, subjects)
    slope, intercept = np.polyfit(true_subject, predicted_subject, 1)
    fitted = slope * true_subject + intercept
    r2 = 1.0 - np.square(predicted_subject - fitted).sum() / max(
        np.square(predicted_subject - predicted_subject.mean()).sum(), 1e-12,
    )
    result["peak_to_peak_calibration"] = {
        "slope": float(slope), "intercept": float(intercept), "r2": float(r2),
    }
    return result


def _paired_bootstrap(
    left: np.ndarray, right: np.ndarray, subjects: np.ndarray, seed: int, replicates: int,
) -> dict[str, float]:
    left_keys, left_subject = _subject_means(left, subjects)
    right_keys, right_subject = _subject_means(right, subjects)
    if not np.array_equal(left_keys, right_keys):
        raise AssertionError("paired subjects disagree")
    difference = left_subject - right_subject
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(replicates, len(difference)))
    bootstrap = difference[draws].mean(axis=1)
    return {
        "left_minus_right": float(difference.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "patients": int(len(difference)),
        "bootstrap_replicates": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--landmarks", type=int, default=256)
    parser.add_argument("--c2", type=float, default=1.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output: {args.output}")
    if args.samples < 32 or args.landmarks < 2 or args.c2 <= 0 or args.bootstrap < 100:
        raise ValueError("invalid probe dimensions or bootstrap count")

    source_manifest = json.loads((args.data / "manifest.json").read_text())
    if source_manifest.get("status") != "complete" or source_manifest.get("failed_cases") != 0:
        raise ValueError("requires a complete failure-free VitalDB acquisition")
    waveforms, subjects, case_ids = _load_records(args.data, args.samples)
    split = _split_indices(subjects, args.seed)
    normalized, channel_mean, channel_std = _standardize(waveforms, split["train"])
    target = normalized[:, 1]
    wrong_test = _wrong_subject_permutation(subjects[split["test"]], args.seed + 991)

    predictions: dict[str, np.ndarray] = {}
    selection: dict[str, object] = {}
    test_y = target[split["test"]]
    for arm in ARMS:
        all_x = _features(normalized, arm)
        train_x, validation_x, test_x = (all_x[split[name]] for name in ("train", "validation", "test"))
        ridge, alpha, candidates = _fit_selected_ridge(
            train_x, target[split["train"]], validation_x, target[split["validation"]], args.alphas,
        )
        predictions[f"ridge_{arm}"] = ridge.predict(test_x).astype(np.float32)
        selection[f"ridge_{arm}"] = {"selected_alpha": alpha, "candidates": candidates}
        mapping = IMQNystromMap.fit(train_x, args.landmarks, args.c2, args.seed + ARMS.index(arm))
        train_phi, validation_phi, test_phi = (
            mapping.transform(values) for values in (train_x, validation_x, test_x)
        )
        kernel_ridge, alpha, candidates = _fit_selected_ridge(
            train_phi, target[split["train"]], validation_phi, target[split["validation"]], args.alphas,
        )
        predictions[f"imq_nystrom_{arm}"] = kernel_ridge.predict(test_phi).astype(np.float32)
        selection[f"imq_nystrom_{arm}"] = {
            "selected_alpha": alpha, "candidates": candidates, "landmarks": args.landmarks, "c2": args.c2,
        }
        if arm == "ii_ppg":
            wrong_x = np.concatenate(
                (normalized[split["test"], 0], normalized[split["test"], 2][wrong_test]), axis=1,
            )
            predictions["ridge_ii_wrong_subject_ppg"] = ridge.predict(wrong_x).astype(np.float32)
            predictions["imq_nystrom_ii_wrong_subject_ppg"] = kernel_ridge.predict(
                mapping.transform(wrong_x)
            ).astype(np.float32)

    # Convert normalized V5 predictions back to detrended ADC units before reporting.
    test_y_units = test_y * channel_std[1] + channel_mean[1]
    predictions_units = {
        name: value * channel_std[1] + channel_mean[1] for name, value in predictions.items()
    }
    test_subjects = subjects[split["test"]]
    summaries = {
        name: _summarize(test_y_units, value, test_subjects)
        for name, value in predictions_units.items()
    }
    inference: dict[str, object] = {}
    for kind in MODEL_KINDS:
        prefix = "imq_nystrom" if kind == "imq_nystrom" else kind
        errors = {
            arm: _record_metrics(test_y_units, predictions_units[f"{prefix}_{arm}"])["rmse"]
            for arm in ARMS
        }
        wrong = _record_metrics(
            test_y_units, predictions_units[f"{prefix}_ii_wrong_subject_ppg"],
        )["rmse"]
        inference[kind] = {
            "real_ppg_increment_rmse": _paired_bootstrap(
                errors["ii"], errors["ii_ppg"], test_subjects, args.seed + 100, args.bootstrap,
            ),
            "wrong_subject_specificity_rmse": _paired_bootstrap(
                wrong, errors["ii_ppg"], test_subjects, args.seed + 101, args.bootstrap,
            ),
        }

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "test_predictions.npz",
        target=test_y_units.astype(np.float32),
        case_ids=case_ids[split["test"]], subjects=test_subjects,
        wrong_subject_source_index=wrong_test,
        **{name: value.astype(np.float32) for name, value in predictions_units.items()},
    )
    split_counts = {
        name: {"records": int(len(indices)), "subjects": int(len(np.unique(subjects[indices])))}
        for name, indices in split.items()
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source_manifest": str(args.data / "manifest.json"),
        "fold_contract": "subject_hash_70_train_15_validation_15_test",
        "split_counts": split_counts,
        "preprocessing": {
            "record_local": "linear_detrend_then_Fourier_resample",
            "samples": args.samples,
            "train_channel_mean": channel_mean.tolist(),
            "train_channel_std": channel_std.tolist(),
        },
        "selection": selection,
        "test_metrics": summaries,
        "paired_patient_bootstrap": inference,
        "wrong_subject_control": "same fitted II+PPG model; PPG permuted with exact subject mismatch",
        "seed": args.seed,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "split_counts": split_counts, "paired_patient_bootstrap": inference}, sort_keys=True))


if __name__ == "__main__":
    main()
