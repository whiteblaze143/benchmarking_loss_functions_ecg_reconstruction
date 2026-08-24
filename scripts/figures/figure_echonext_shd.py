#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    echo = data[
        (data.classifier == "EchoNext_Mini_12_SHD")
        & (data.condition == "clean")
    ]
    reference = echo[echo.signal_variant == "reference"].set_index("task_name")
    order = reference.sort_values("auroc").index
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), constrained_layout=True)
    axes[0].scatter(
        reference.loc[order, "auroc"],
        y,
        color=COLORS["reference"],
        marker="|",
        s=120,
        linewidths=2,
        label="Original ECG",
    )
    selection_path = resolve_project_path(args.selection)
    ids = selected_ids(selection_path)
    offsets = [-0.20, 0.0, 0.20]
    for offset, (family, model_id) in zip(offsets, ids.items()):
        selected = echo[echo.model_id == model_id].set_index("task_name")
        axes[0].scatter(
            selected.loc[order, "auroc"],
            y + offset,
            s=24,
            color=COLORS[family],
            label=FAMILY_LABELS[family],
        )
        axes[1].scatter(
            selected.loc[order, "auroc"] - reference.loc[order, "auroc"],
            y + offset,
            s=24,
            color=COLORS[family],
            label=FAMILY_LABELS[family],
        )
    axes[0].set_yticks(y, [name.replace("_", " ") for name in order])
    axes[0].set_xlabel("AUROC")
    axes[0].set_xlim(0.45, 1.0)
    axes[1].axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("Reconstruction − original AUROC")
    for label, axis in zip(("A", "B"), axes):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    save(
        fig,
        args.output,
        [args.database, selection_path],
        metadata={
            "tasks": 12,
            "ground_truth": "echocardiography-derived EchoNext labels",
            "selection": "validation-selected reconstruction masks",
            "classifier_inputs": (
                "reconstructed waveform plus unchanged original tabular "
                "metadata; retention is not attributable to waveform alone"
            ),
        },
    )


if __name__ == "__main__":
    main()
