#!/usr/bin/env python3
"""Render reported frozen Fold-8 configuration-shift AUROCs with exact values."""
import json
from pathlib import Path

import numpy as np

from paper_plot_style import save, plt

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "outputs/configuration_shift_evaluations/ptbxl/configuration_shift_matrix.json"
MODELS = (
    ("GraphECG", "GraphECG"),
    ("P07_ROBUSTNESS_TRAINED", "SetOp robust"),
    ("P07_ROBUSTNESS_TRAINED_AUX", "SetOp robust + aux"),
    ("P07_FULLLEAD_ONLY", "SetOp full-lead only"),
    ("FixedTensor_P02", "FixedTensor P02"),
)
CONFIGS = ("Q8_indep", "S6_precordial", "S6_limb", "S3_icu_v1", "S3_icu_v5", "S2_bipolar", "S1_smartwatch_I", "S1_lead_II", "S_icm")
SHORT = ("Q8", "S6\nprecordial", "S6\nlimb", "S3\nV1", "S3\nV5", "S2", "S1\nI", "S1\nII", "ICM")


def main() -> None:
    data = json.loads(SOURCE.read_text())
    values = np.asarray([[data[key][config]["macro_auroc"] for config in CONFIGS] for key, _ in MODELS])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.40, vmax=0.95, aspect="auto")
    ax.set_xticks(range(len(CONFIGS)), SHORT)
    ax.set_yticks(range(len(MODELS)), [label for _, label in MODELS])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] < 0.64 else "black"
            ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Macro AUROC")
    save(fig, "fig3_ptbxl_configuration_shift_auroc")


if __name__ == "__main__":
    main()
