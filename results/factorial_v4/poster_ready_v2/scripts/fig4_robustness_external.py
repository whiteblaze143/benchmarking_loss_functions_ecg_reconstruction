#!/usr/bin/env python3
"""Poster Figure 4: clean/noise robustness and EchoNext generalization."""

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
ROBUST = ROOT / "results/factorial_v4/poster_evidence/selected_robustness.csv"
ARCH = ROOT / "results/factorial_v4/poster_evidence/architecture_selected.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "figures/fig4_robustness_external"


def main() -> None:
    configure()
    robustness = pd.read_csv(ROBUST)
    architecture = pd.read_csv(ARCH).set_index("family")
    fig, axes = plt.subplots(
        1, 3, figsize=(11.6, 3.35), constrained_layout=True
    )
    clean = robustness[robustness.condition == "clean"].set_index("family")
    zero = robustness[
        robustness.condition.str.match(r"nstdb_(bw|em|ma)_0db")
    ]
    noise_order = ["bw", "em", "ma"]
    noise_labels = {"bw": "BW", "em": "EM", "ma": "MA"}
    for family in FAMILY_ORDER:
        values = [clean.loc[family, "missing_lead_mse"]]
        for noise in noise_order:
            value = zero[
                (zero.family == family) & (zero.noise_type == noise)
            ].missing_lead_mse.iloc[0]
            values.append(value)
        axes[0].plot(
            range(4),
            values,
            marker="o",
            linewidth=2,
            color=FAMILY_COLORS[family],
            label=FAMILY_LABELS[family],
        )
    axes[0].set_xticks(
        range(4), ["Clean", *[f"{noise_labels[n]}\n0 dB" for n in noise_order]]
    )
    axes[0].set_ylabel("Missing-lead MSE")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    panel_label(axes[0], "A")

    x = np.arange(len(FAMILY_ORDER))
    for axis, metric, label in [
        (axes[1], "echonext_r2", "EchoNext missing-lead R²"),
        (axes[2], "echonext_pearson", "EchoNext missing-lead Pearson"),
    ]:
        values = [architecture.loc[family, metric] for family in FAMILY_ORDER]
        axis.bar(
            x,
            values,
            color=[FAMILY_COLORS[family] for family in FAMILY_ORDER],
            width=0.68,
        )
        axis.axhline(0, color="#666666", linewidth=0.9)
        axis.set_xticks(
            x, [FAMILY_LABELS[family] for family in FAMILY_ORDER],
            rotation=22, ha="right",
        )
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    panel_label(axes[1], "B")
    panel_label(axes[2], "C")
    caption = (
        "Validation-selected models under severe NSTDB noise and external "
        "EchoNext shift. Latent models lead on clean PTB-XL, but U-Net degrades "
        "less at 0 dB. EchoNext Pearson can remain high despite negative R², "
        "revealing amplitude-calibration failure."
    )
    save_triplet(fig, OUTPUT, sources=[ROBUST, ARCH], caption=caption)


if __name__ == "__main__":
    main()
