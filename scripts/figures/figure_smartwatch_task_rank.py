#!/usr/bin/env python3
import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("results/factorial_v2/selected_masks.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    data = pd.read_parquet(args.database)
    selection_path = resolve_project_path(args.selection)
    ids = selected_ids(selection_path)
    data = data[data.model_id.isin(ids.values())].copy()
    data["family_label"] = data.family.map(FAMILY_LABELS)
    summary = data.groupby(["family", "task_name"], as_index=False).agg(
        probability_mae=("probability_mae", "mean"),
        threshold_agreement=("threshold_agreement", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for family in ids:
        subset = summary[summary.family == family].sort_values("probability_mae")
        rank = np.arange(1, len(subset) + 1)
        axes[0].plot(
            rank,
            subset.probability_mae,
            color=COLORS[family],
            linewidth=1.5,
            label=FAMILY_LABELS[family],
        )
        axes[1].plot(
            rank,
            subset.sort_values("threshold_agreement", ascending=False).threshold_agreement,
            color=COLORS[family],
            linewidth=1.5,
            label=FAMILY_LABELS[family],
        )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("ECGFounder task rank")
    axes[1].set_xlabel("ECGFounder task rank")
    axes[0].set_ylabel("Probability MAE vs Philips")
    axes[1].set_ylabel("0.5-threshold agreement vs Philips")
    axes[1].set_ylim(0, 1.01)
    axes[0].legend(frameon=False)
    for label, axis in zip(("A", "B"), axes):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
    save(
        fig,
        args.output,
        [args.database, selection_path],
        metadata={
            "tasks": 150,
            "aggregation": "mean across four watch devices",
            "evaluation_type": "frozen-classifier probability-fidelity proxy",
            "diagnostic_ground_truth": "unavailable",
        },
    )


if __name__ == "__main__":
    main()
