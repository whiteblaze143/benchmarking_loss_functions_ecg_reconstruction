#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.figures.clinical_common import (
    FAMILY_LABELS,
    comparison_ids,
    configure,
    resolve_project_path,
    save,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("results/factorial_v2/selected_masks.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    data = pd.read_parquet(args.database)
    data = data[
        (data.classifier == "PTBXL_five_superclass")
        & (data.signal_variant == "reconstructed")
    ].copy()
    labels = []
    frames = []
    selection_path = resolve_project_path(args.selection)
    for family, configurations in comparison_ids(selection_path).items():
        for config, model_id in configurations.items():
            frame = data[data.model_id == model_id].copy()
            label = f"{FAMILY_LABELS[family]} · {config}"
            frame["configuration"] = label
            labels.append(label)
            frames.append(frame)
    selected = pd.concat(frames)
    matrix = selected.pivot(
        index="configuration", columns="task_name", values="auroc"
    ).loc[labels, ["NORM", "MI", "STTC", "CD", "HYP"]]
    fig, ax = plt.subplots(figsize=(5.7, 3.2), constrained_layout=True)
    sns.heatmap(
        matrix,
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "AUROC"},
        ax=ax,
    )
    ax.set_xlabel("PTB-XL diagnostic superclass")
    ax.set_ylabel("")
    save(
        fig,
        args.output,
        [args.database, selection_path],
        metadata={"selection": "base, validation-best, and full per family"},
    )


if __name__ == "__main__":
    main()
