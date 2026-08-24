#!/usr/bin/env python3
"""Poster Figure 5: smartwatch zero-shot domain-gap heatmaps."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FAMILY_LABELS, FAMILY_ORDER, configure, panel_label, save_triplet


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/factorial_v4/poster_evidence/selected_smartwatch.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "figures/fig5_smartwatch_domain_gap"
DEVICE_LABELS = {
    "applewatch_serie8": "Apple",
    "fitbitsense2": "Fitbit",
    "samsunggalaxy6": "Samsung",
    "withingsscanwatch": "Withings",
}


def main() -> None:
    configure()
    data = pd.read_csv(SOURCE)
    data["family_label"] = data.family.map(FAMILY_LABELS)
    data["device_label"] = data.device.map(DEVICE_LABELS)
    fig, axes = plt.subplots(
        1, 3, figsize=(10.9, 3.25), constrained_layout=True
    )
    panels = [
        ("r2", "Missing-eleven-lead R²", "vlag", None),
        ("pearson", "Missing-eleven-lead Pearson", "viridis", (0, 1)),
        (
            "probability_pearson_proxy",
            "ECGFounder probability correlation",
            "viridis",
            (0, 1),
        ),
    ]
    for label, axis, (metric, cbar_label, cmap, limits) in zip(
        "ABC", axes, panels
    ):
        matrix = (
            data.pivot(
                index="family_label",
                columns="device_label",
                values=metric,
            )
            .reindex(
                index=[FAMILY_LABELS[family] for family in FAMILY_ORDER],
                columns=list(DEVICE_LABELS.values()),
            )
        )
        kwargs = {}
        if limits is not None:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.6,
            linecolor="white",
            cbar_kws={"label": cbar_label, "shrink": 0.78},
            **kwargs,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=25)
        axis.tick_params(axis="y", rotation=0)
        panel_label(axis, label)
    caption = (
        "Zero-shot smartwatch transfer using only the physically acquired lead "
        "II. All 12 selected family-device cells have negative missing-eleven-"
        "lead R². Panel C is a Philips-referenced frozen-classifier probability-"
        "fidelity proxy on simulator signals, not diagnostic accuracy."
    )
    save_triplet(fig, OUTPUT, sources=[SOURCE], caption=caption)


if __name__ == "__main__":
    main()
