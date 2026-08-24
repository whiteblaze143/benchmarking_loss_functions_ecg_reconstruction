#!/usr/bin/env python3
"""Poster Figure 1: complete repaired 2^4 loss landscape."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FAMILY_LABELS, FAMILY_ORDER, configure, panel_label, save_triplet


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/factorial_v4_2x4/main_48_cell_table.parquet"
OUTPUT = Path(__file__).resolve().parents[1] / "figures/fig1_loss_landscape"


def main() -> None:
    configure()
    data = pd.read_parquet(SOURCE)
    masks = [f"{value:04b}" for value in range(16)]
    metrics = [
        ("r2", "Missing-lead R²", "viridis"),
        ("qrs_correlation", "QRS correlation", "mako"),
        ("st_correlation", "ST correlation", "crest"),
        ("ecgfounder_auroc", "ECGFounder AUROC", "rocket"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(12.0, 6.2), constrained_layout=True
    )
    for label, axis, (metric, colorbar_label, cmap) in zip(
        "ABCD", axes.flat, metrics
    ):
        matrix = (
            data.pivot(index="family", columns="mask", values=metric)
            .reindex(index=FAMILY_ORDER, columns=masks)
            .rename(index=FAMILY_LABELS)
        )
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.6,
            linecolor="white",
            cbar_kws={"label": colorbar_label, "shrink": 0.78},
            annot_kws={"fontsize": 6.5},
        )
        axis.set_xlabel("Loss mask (MSE / correlation / MMD / derivative)")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)
        panel_label(axis, label)
    caption = (
        "Complete repaired 2^4 factorial landscape on 2,198 PTB-XL fold-10 "
        "ECGs. Loss effects vary by architecture; no single mask dominates "
        "all reconstruction, morphology, and frozen-classifier endpoints."
    )
    save_triplet(fig, OUTPUT, sources=[SOURCE], caption=caption)


if __name__ == "__main__":
    main()
