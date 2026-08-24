#!/usr/bin/env python3
"""Poster Figure 2: four-factor patient-cluster BCa main effects."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    COMPONENT_COLORS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    configure,
    panel_label,
    save_triplet,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/factorial_v4_2x4/factorial_effects_bca.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "figures/fig2_component_effects"


def main() -> None:
    configure()
    data = pd.read_csv(SOURCE)
    metrics = [
        ("r2", "Effect on missing-lead R²"),
        ("qrs_correlation", "Effect on QRS correlation"),
        ("st_correlation", "Effect on ST correlation"),
        ("ecgfounder_auroc", "Effect on ECGFounder AUROC"),
    ]
    components = ["mse", "correlation", "mmd", "derivative"]
    fig, axes = plt.subplots(
        1, 4, figsize=(13.6, 3.45), constrained_layout=True
    )
    y_positions = {
        (family, component): family_index * 4 + component_index
        for family_index, family in enumerate(FAMILY_ORDER)
        for component_index, component in enumerate(components)
    }
    ticks = [
        family_index * 4 + 1 for family_index in range(len(FAMILY_ORDER))
    ]
    for label, axis, (metric, xlabel) in zip("ABCD", axes, metrics):
        subset = data[
            (data.metric == metric) & (data.effect_type == "main")
        ]
        for row in subset.itertuples():
            y = y_positions[(row.family, row.effect)]
            axis.errorbar(
                row.estimate,
                y,
                xerr=np.array([
                    [row.estimate - row.ci_low],
                    [row.ci_high - row.estimate],
                ]),
                marker="o",
                color=COMPONENT_COLORS[row.effect],
                capsize=2.5,
                markersize=5,
                linewidth=1.3,
            )
        axis.axvline(0, color="#666666", linewidth=0.9, linestyle="--")
        axis.set_yticks(ticks)
        axis.set_yticklabels(
            [FAMILY_LABELS[family] for family in FAMILY_ORDER]
            if axis is axes[0] else []
        )
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.grid(axis="x", alpha=0.2)
        panel_label(axis, label)
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none",
            color=COMPONENT_COLORS[component],
            label={"mse": "MSE", "mmd": "MMD"}.get(
                component, component.title()
            ),
        )
        for component in components
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    caption = (
        "Average matched on/off MSE, correlation, MMD, and derivative effects "
        "within each architecture. "
        "Intervals are 95% patient-cluster BCa intervals over 1,904 patients "
        "(2,198 ECGs), conditional on the seed-42 trained models."
    )
    save_triplet(fig, OUTPUT, sources=[SOURCE], caption=caption)


if __name__ == "__main__":
    main()
