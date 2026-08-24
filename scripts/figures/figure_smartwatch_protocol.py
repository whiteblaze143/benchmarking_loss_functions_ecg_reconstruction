#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.figures.clinical_common import configure, save


DEVICE_LABELS = {
    "applewatch_serie8": "Apple",
    "fitbitsense2": "Fitbit",
    "samsunggalaxy6": "Samsung",
    "withingsscanwatch": "Withings",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    data = pd.read_parquet(args.database)
    experiments = ["heart_rate", "r_wave_amplitude", "st_offset"]
    labels = ["Heart rate (bpm)", "R-wave amplitude (µV)", "ST offset (µV)"]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.5), constrained_layout=True)
    palette = sns.color_palette("colorblind", 4)
    for ax, experiment, label in zip(axes, experiments, labels):
        subset = data[data.experiment_type == experiment].copy()
        subset["device_label"] = subset.device.map(DEVICE_LABELS)
        sns.barplot(
            data=subset,
            x="device_label",
            y="watch_vs_simulator_mae",
            hue="device_label",
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(f"MAE · {label}")
        ax.tick_params(axis="x", rotation=35)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=6, padding=1)
    square = data[data.experiment_type == "square_wave"].copy()
    square["device_label"] = square.device.map(DEVICE_LABELS)
    sns.barplot(
        data=square,
        x="device_label",
        y="watch_vs_philips_max_aligned_cross_correlation",
        hue="device_label",
        palette=palette,
        legend=False,
        ax=axes[3],
    )
    axes[3].set_xlabel("")
    axes[3].set_ylabel("Max aligned xcorr · 2-Hz square wave")
    axes[3].set_ylim(0, 1.02)
    axes[3].tick_params(axis="x", rotation=35)
    for container in axes[3].containers:
        axes[3].bar_label(container, fmt="%.2f", fontsize=6, padding=1)
    for label, ax in zip(("A", "B", "C", "D"), axes):
        ax.text(
            -0.12,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
    save(
        fig,
        args.output,
        [args.database],
        metadata={
            "evaluation_type": (
                "calibrated simulator targets for panels A-C; paired-device "
                "reference under simulated 2-Hz stimulus for panel D"
            ),
            "not_attributed_to_reconstruction_model": True,
            "square_wave_endpoint": (
                "watch-versus-Philips cross-correlation maximized during lag "
                "alignment; not an independent accuracy estimate"
            ),
        },
    )


if __name__ == "__main__":
    main()
