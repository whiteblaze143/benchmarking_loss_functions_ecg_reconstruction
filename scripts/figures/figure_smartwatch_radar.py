#!/usr/bin/env python3
import argparse
import json
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
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("results/factorial_v2/selected_masks.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    payload = json.loads(args.results.read_text())
    selection_path = resolve_project_path(args.selection)
    ids = selected_ids(selection_path)
    rows = []
    for family, model_id in ids.items():
        for device, result in payload["models"][model_id].items():
            signal = result["signal_wearable_missing_eleven"]["missing_leads"]
            fidelity = result["ecgfounder_probability_fidelity"]["macro"]
            rows.append(
                {
                    "family": family,
                    "device": device,
                    "Pearson": signal["pearson"],
                    "R²": signal["r2"],
                    "SNR": signal["snr_db"],
                    "−RMSE": -signal["rmse"],
                    "−Probability MAE": -fidelity["probability_mae"],
                    "Probability correlation": fidelity["probability_pearson"],
                }
            )
    frame = pd.DataFrame(rows)
    metrics = ["Pearson", "R²", "SNR", "−RMSE", "−Probability MAE", "Probability correlation"]
    display_labels = [
        "Pearson",
        "R²",
        "SNR",
        "−RMSE",
        "−Prob. MAE",
        "Prob. corr.",
    ]
    for metric in metrics:
        low, high = frame[metric].min(), frame[metric].max()
        frame[metric] = 0.5 if high == low else (frame[metric] - low) / (high - low)
    means = frame.groupby("family")[metrics].mean()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(
        figsize=(4.8, 4.2),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    for family in ids:
        values = means.loc[family].to_numpy()
        closed = np.r_[values, values[0]]
        ax.plot(closed_angles, closed, color=COLORS[family], linewidth=2, label=FAMILY_LABELS[family])
        ax.fill(closed_angles, closed, color=COLORS[family], alpha=0.08)
    ax.set_xticks(angles, display_labels)
    ax.tick_params(axis="x", pad=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"])
    ax.set_ylim(0, 1)
    ax.set_title(
        "Relative min–max score across 12 selected family×device cells\n"
        "(not absolute performance)",
        fontsize=8,
        pad=18,
    )
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.27, 1.13))
    save(
        fig,
        args.output,
        [args.results, selection_path],
        metadata={
            "normalization": "direction-corrected min-max across 12 selected family-device rows",
            "interpretation": "higher is better; relative profile only, not absolute clinical accuracy",
            "ground_truth_status": "signal reference plus classifier probability-fidelity proxy",
        },
    )


if __name__ == "__main__":
    main()
