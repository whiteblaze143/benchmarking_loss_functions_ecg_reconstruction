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
    if "condition" not in data:
        data["condition"] = "clean"
    founder = data[
        (data.classifier == "ECGFounder_150")
        & (data.condition == "clean")
    ].copy()
    reference = founder[
        (founder.signal_variant == "reference")
        & (founder.model_id == founder.model_id.iloc[0])
        & founder.auroc.notna()
    ].set_index("task_name")
    order = reference.sort_values("average_precision", ascending=False).index
    selection_path = resolve_project_path(args.selection)
    ids = selected_ids(selection_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    rank = np.arange(1, len(order) + 1)
    axes[0].plot(
        rank,
        reference.loc[order, "average_precision"],
        color=COLORS["reference"],
        linewidth=2,
        label="Original ECG",
    )
    for family, model_id in ids.items():
        selected = founder[
            (founder.model_id == model_id)
            & (founder.signal_variant == "reconstructed")
        ].set_index("task_name")
        axes[0].plot(
            rank,
            selected.loc[order, "average_precision"],
            color=COLORS[family],
            linewidth=1.5,
            label=FAMILY_LABELS[family],
        )
        delta = selected.loc[order, "auroc"] - reference.loc[order, "auroc"]
        axes[1].scatter(
            reference.loc[order, "prevalence"],
            delta,
            s=14 + 36 * reference.loc[order, "prevalence"],
            alpha=0.75,
            color=COLORS[family],
            label=FAMILY_LABELS[family],
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Task rank by original-ECG AP")
    axes[0].set_ylabel("Average precision")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Positive prevalence")
    axes[1].set_ylabel("Reconstruction − original AUROC")
    axes[1].legend(frameon=False)
    for label, axis in zip(("A", "B"), axes):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
    save(
        fig,
        args.output,
        [args.database, selection_path],
        metadata={
            "task_filter": "31/150 ECGFounder tasks with both classes",
            "selection": "validation-selected masks only",
            "undefined_tasks": "excluded from AUROC/AP plot and retained in database",
        },
    )


if __name__ == "__main__":
    main()
