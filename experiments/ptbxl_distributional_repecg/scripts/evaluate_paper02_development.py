from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from repecg.common.bootstrap import clustered_paired_bootstrap, interval
from repecg.common.metrics import multilabel_metrics


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def macro_auroc(target: np.ndarray, probability: np.ndarray, index: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(target[index, column], probability[index, column]) for column in range(5)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    predictions = {}
    reference: pd.DataFrame | None = None
    for variant in ("kernel", "moments", "gaussian"):
        frame = pd.read_csv(args.training / f"predictions_{variant}.csv")
        if reference is None:
            reference = frame
        elif not np.array_equal(frame.ecg_id.to_numpy(), reference.ecg_id.to_numpy()):
            raise ValueError("prediction row order mismatch")
        predictions[variant] = frame[[f"p_{name}" for name in CLASSES]].to_numpy()
    assert reference is not None
    target = reference[[f"y_{name}" for name in CLASSES]].to_numpy()
    patient_ids = reference.patient_id.to_numpy(dtype=np.int64)
    metrics = []
    for variant, probability in predictions.items():
        row = {"variant": variant, **multilabel_metrics(target, probability)}
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(args.output / "metrics.csv", index=False)

    def task_statistic(index: np.ndarray) -> np.ndarray:
        kernel = macro_auroc(target, predictions["kernel"], index)
        moments = macro_auroc(target, predictions["moments"], index)
        gaussian = macro_auroc(target, predictions["gaussian"], index)
        return np.asarray((kernel - moments, kernel - gaussian))

    task_draws = clustered_paired_bootstrap(
        patient_ids,
        task_statistic,
        replicates=args.replicates,
        seed=args.seed,
    )
    with np.load(args.representations / "representation_val.npz") as item:
        kernel_rep = np.asarray(item["kernel"], dtype=np.float64).reshape(len(target), -1)
        gaussian_rep = np.asarray(item["gaussian"], dtype=np.float64).reshape(len(target), -1)
        rep_patient_ids = np.asarray(item["patient_ids"], dtype=np.int64)
    if not np.array_equal(rep_patient_ids, patient_ids):
        raise ValueError("representation/prediction patient order mismatch")
    denominator = np.linalg.norm(kernel_rep, axis=1) * np.linalg.norm(gaussian_rep, axis=1)
    cosine = np.divide(
        np.einsum("ij,ij->i", kernel_rep, gaussian_rep),
        denominator,
        out=np.zeros(len(target)),
        where=denominator > 0,
    )
    mechanism = 1.0 - cosine
    unique_patients = np.unique(patient_ids)
    patient_mechanism = np.asarray([mechanism[patient_ids == patient].mean() for patient in unique_patients])
    rng = np.random.default_rng(args.seed + 1)
    mechanism_draws = np.asarray(
        [patient_mechanism[rng.integers(0, len(unique_patients), len(unique_patients))].mean() for _ in range(args.replicates)]
    )
    task_moments_ci = interval(task_draws[:, 0])
    task_gaussian_ci = interval(task_draws[:, 1])
    mechanism_ci = interval(mechanism_draws)
    mechanism_pass = mechanism_ci[0] > 0.0
    task_pass = task_moments_ci[0] > 0.005
    task_fail = task_moments_ci[1] < -0.005
    if mechanism_pass and task_pass:
        decision = "PASS"
    elif not mechanism_pass or task_fail:
        decision = "FAIL"
    else:
        decision = "INCONCLUSIVE"
    bootstrap = pd.DataFrame(
        {
            "replicate": np.arange(args.replicates),
            "delta_kernel_minus_moments_macro_auroc": task_draws[:, 0],
            "delta_kernel_minus_gaussian_macro_auroc": task_draws[:, 1],
            "mechanism_one_minus_cosine": mechanism_draws,
        }
    )
    bootstrap.to_csv(args.output / "bootstrap.csv", index=False)
    gate = {
        "decision": decision,
        "task_delta_kernel_minus_moments": {
            "point": float(task_statistic(np.arange(len(target)))[0]),
            "ci95": task_moments_ci,
            "rope": [-0.005, 0.005],
        },
        "destroyer_task_delta_kernel_minus_gaussian": {
            "point": float(task_statistic(np.arange(len(target)))[1]),
            "ci95": task_gaussian_ci,
        },
        "mechanism_one_minus_cosine": {
            "point_patient_equal": float(patient_mechanism.mean()),
            "ci95": mechanism_ci,
        },
        "stability": "not_yet_measured",
        "replicates": args.replicates,
        "bootstrap_unit": "patient",
    }
    (args.output / "claim_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
