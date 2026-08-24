#!/usr/bin/env python3
"""Poster Figure 3: full-minus-base morphology and diagnostic endpoints."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (
    FAMILY_COLORS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    configure,
    panel_label,
    save_triplet,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "results/factorial_v4_2x4/familywise_endpoint_tests.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "figures/fig3_morphology_utility"


def main() -> None:
    configure()
    data = pd.read_csv(SOURCE)
    endpoints = [
        ("QRS", "Δ QRS correlation"),
        ("ST", "Δ ST correlation"),
        ("diagnostic_utility", "Δ ECGFounder AUROC"),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(10.8, 3.25), constrained_layout=True
    )
    x = np.arange(len(FAMILY_ORDER))
    for label, axis, (endpoint, ylabel) in zip("ABC", axes, endpoints):
        subset = data[data.endpoint == endpoint].set_index("family")
        values = [subset.loc[family, "estimate"] for family in FAMILY_ORDER]
        low = [
            value - subset.loc[family, "ci_low"]
            for family, value in zip(FAMILY_ORDER, values)
        ]
        high = [
            subset.loc[family, "ci_high"] - value
            for family, value in zip(FAMILY_ORDER, values)
        ]
        bars = axis.bar(
            x,
            values,
            color=[FAMILY_COLORS[family] for family in FAMILY_ORDER],
            width=0.68,
        )
        axis.errorbar(
            x,
            values,
            yerr=np.array([low, high]),
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1,
        )
        axis.axhline(0, color="#666666", linewidth=0.9)
        axis.set_xticks(
            x, [FAMILY_LABELS[family] for family in FAMILY_ORDER],
            rotation=22, ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        for bar, family in zip(bars, FAMILY_ORDER):
            rejected = bool(subset.loc[family, "reject_null"])
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                "*" if rejected else "n.s.",
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=9,
                fontweight="bold" if rejected else "normal",
            )
        panel_label(axis, label)
    caption = (
        "Full objective minus MSE-only base for the three prespecified endpoint "
        "families. QRS and ST improve in all architectures, whereas diagnostic "
        "AUROC changes are not significant at within-architecture α=0.0167. "
        "Error bars are 95% patient-cluster BCa intervals."
    )
    save_triplet(fig, OUTPUT, sources=[SOURCE], caption=caption)


if __name__ == "__main__":
    main()
