#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figures.clinical_common import (
    COLORS,
    FAMILY_LABELS,
    configure,
    resolve_project_path,
    save,
    selected_ids,
)


def reliability(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    centers, observed, counts = [], [], []
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities < edges[index + 1] if index < bins - 1 else probabilities <= 1
        )
        if mask.any():
            centers.append(float(probabilities[mask].mean()))
            observed.append(float(labels[mask].mean()))
            counts.append(int(mask.sum()))
    return np.asarray(centers), np.asarray(observed), np.asarray(counts)


def stack_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(frame[column].map(np.asarray).to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("results/factorial_v2/selected_masks.json"))
    parser.add_argument("--task", default="shd_moderate_or_greater")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    payload = json.loads(args.results.read_text())
    selection_path = resolve_project_path(args.selection)
    ids = selected_ids(selection_path)
    tasks = payload["shd_reference"]["classifier"]["tasks"]
    if args.task not in tasks:
        raise ValueError(f"Unknown EchoNext calibration task {args.task!r}")
    task_index = tasks.index(args.task)
    reference_path = Path(payload["shd_reference"]["per_record_parquet"])
    reference = pd.read_parquet(reference_path)
    labels = stack_column(reference, "labels")[:, task_index]
    ref_probs = stack_column(reference, "probabilities")[:, task_index]
    fig, ax = plt.subplots(figsize=(3.8, 3.2), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1)
    x, y, _ = reliability(ref_probs, labels)
    ax.plot(x, y, marker="o", color=COLORS["reference"], label="Original ECG")
    inputs = [args.results, reference_path, selection_path]
    for family, model_id in ids.items():
        model_path = Path(payload["models"][model_id]["shd_per_record_parquet"])
        frame = pd.read_parquet(model_path, filters=[("condition", "=", "clean")])
        probabilities = stack_column(frame, "probabilities")[:, task_index]
        x, y, _ = reliability(probabilities, labels)
        ax.plot(x, y, marker="o", color=COLORS[family], label=FAMILY_LABELS[family])
        inputs.append(model_path)
    ax.set_xlabel("Mean predicted composite-SHD probability")
    ax.set_ylabel("Observed composite-SHD frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    save(
        fig,
        args.output,
        inputs,
        metadata={
            "task": args.task,
            "task_index": task_index,
            "bins": 10,
            "selection": "validation-selected masks",
        },
    )


if __name__ == "__main__":
    main()
