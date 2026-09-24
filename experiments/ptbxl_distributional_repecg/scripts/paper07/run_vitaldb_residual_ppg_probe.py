#!/usr/bin/env python3
"""Leakage-safe VitalDB test of PPG information remaining after Lead-II ridge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from scripts.paper07.run_vitaldb_increment_probe import (
    _features, _fit_selected_ridge, _load_records, _split_indices, _standardize,
    _subject_means, _wrong_subject_permutation,
)


def _fold(subject: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"p07-vitaldb-residual:{seed}:{subject}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _cross_fitted_base(features: np.ndarray, target: np.ndarray, subjects: np.ndarray, alpha: float, seed: int, folds: int) -> np.ndarray:
    output = np.empty_like(target, dtype=np.float32)
    assignments = np.asarray([_fold(str(subject), seed, folds) for subject in subjects])
    for heldout in range(folds):
        test = assignments == heldout
        train = ~test
        if not test.any() or not train.any():
            raise ValueError("invalid cross-fitting partition")
        output[test] = Ridge(alpha=alpha).fit(features[train], target[train]).predict(features[test])
    return output


def _r2_and_mae(target: np.ndarray, prediction: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    r2 = 1.0 - float(np.square(residual).sum()) / max(float(np.square(target - target.mean()).sum()), 1e-12)
    _, mae_subject = _subject_means(np.mean(np.abs(residual), axis=1), subjects)
    return {"r2_pooled": r2, "residual_mae_patient_equal": float(mae_subject.mean()), "patients": int(len(mae_subject)), "records": int(len(target))}


def _bootstrap_r2(target: np.ndarray, prediction: np.ndarray, subjects: np.ndarray, seed: int, draws: int) -> dict[str, float]:
    keys = np.unique(subjects)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        selected = rng.choice(keys, size=len(keys), replace=True)
        index = np.concatenate([np.flatnonzero(subjects == subject) for subject in selected])
        values.append(_r2_and_mae(target[index], prediction[index], subjects[index])["r2_pooled"])
    return {"ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975)), "bootstrap_replicates": draws}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1., 10.])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("failed_cases") != 0:
        raise ValueError("requires complete failure-free VitalDB source")
    waveforms, subjects, case_ids = _load_records(args.data, args.samples)
    split = _split_indices(subjects, args.seed)
    normalized, mean, scale = _standardize(waveforms, split["train"])
    ii = _features(normalized, "ii")
    ppg = _features(normalized, "ppg")
    target = normalized[:, 1]
    train, validation, test = (split[name] for name in ("train", "validation", "test"))
    base, base_alpha, base_candidates = _fit_selected_ridge(ii[train], target[train], ii[validation], target[validation], args.alphas)
    train_base_oof = _cross_fitted_base(ii[train], target[train], subjects[train], base_alpha, args.seed + 17, args.folds)
    validation_base = base.predict(ii[validation]).astype(np.float32)
    test_base = base.predict(ii[test]).astype(np.float32)
    train_residual = target[train] - train_base_oof
    validation_residual = target[validation] - validation_base
    test_residual = target[test] - test_base
    g, g_alpha, g_candidates = _fit_selected_ridge(ppg[train], train_residual, ppg[validation], validation_residual, args.alphas)
    real = g.predict(ppg[test]).astype(np.float32)
    wrong_index = _wrong_subject_permutation(subjects[test], args.seed + 991)
    wrong = g.predict(ppg[test][wrong_index]).astype(np.float32)
    shifted = g.predict(np.roll(ppg[test].reshape(len(test), 1, args.samples), args.samples // 2, axis=-1).reshape(len(test), -1)).astype(np.float32)
    # Residuals have zero-offset units: scale, but do not add the V5 channel mean.
    unit = float(scale[1])
    true_units = test_residual * unit
    predictions = {"real_ppg": real * unit, "wrong_subject_ppg": wrong * unit, "time_shifted_ppg": shifted * unit}
    test_subjects = subjects[test]
    summaries = {name: {**_r2_and_mae(true_units, prediction, test_subjects), "r2_bootstrap": _bootstrap_r2(true_units, prediction, test_subjects, args.seed + index, args.bootstrap)} for index, (name, prediction) in enumerate(predictions.items())}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "residual_predictions.npz", case_ids=case_ids[test], subjects=test_subjects, true_residual=true_units.astype(np.float32), base_v5=(test_base * unit).astype(np.float32), wrong_subject_source_index=wrong_index, **{key: value.astype(np.float32) for key, value in predictions.items()})
    result = {
        "schema_version": "paper07_vitaldb_residual_ppg_v1", "status": "complete", "seed": args.seed,
        "split_counts": {name: {"records": int(len(index)), "subjects": int(len(np.unique(subjects[index])))} for name, index in split.items()},
        "base_ii_model": {"selected_alpha": base_alpha, "validation_candidates": base_candidates, "train_residual": "5-fold subject-disjoint out-of-fold II ridge"},
        "ppg_residual_model": {"selected_alpha": g_alpha, "validation_candidates": g_candidates},
        "controls": {"wrong_subject": "same fitted real-PPG residual model with a test-set subject derangement", "time_shifted": f"within-window circular shift by {args.samples // 2} samples"},
        "residual_target": "V5 - fitted_II(V5), evaluated in detrended ADC units", "results": summaries,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "results": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
