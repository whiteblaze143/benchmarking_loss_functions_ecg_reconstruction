#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.figures.clinical_common import (
    COLORS,
    FAMILY_LABELS,
    configure,
    resolve_project_path,
    save,
    selected_ids,
)


def parse_condition(condition: str) -> tuple[str, float | None]:
    match = re.search(r"_(24|12|6|0)db$", condition)
    snr = float(match.group(1)) if match else None
    if condition.startswith("gaussian"):
        return "Gaussian", snr
    if condition.startswith("nstdb_"):
        return condition.split("_")[1].upper(), snr
    return "Baseline drift", snr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("results/factorial_v2/selected_masks.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    data = pd.read_parquet(args.database)
    selection_path = resolve_project_path(args.selection)
    ids = selected_ids(selection_path)
    stress = data[
        (data.classifier == "EchoNext_Mini_12_SHD")
        & data.model_id.isin(ids.values())
        & (data.condition != "clean")
    ].copy()
    parsed = stress.condition.map(parse_condition)
    stress["noise"] = parsed.map(lambda value: value[0])
    stress["snr_db"] = parsed.map(lambda value: value[1])
    summary = (
        stress.dropna(subset=["snr_db", "auroc"])
        .groupby(["model_id", "noise", "snr_db"], as_index=False)
        .auroc.mean()
    )
    baseline = (
        stress[(stress.noise == "Baseline drift") & stress.auroc.notna()]
        .groupby("model_id", as_index=False)
        .auroc.mean()
    )
    noises = ["Gaussian", "BW", "EM", "MA"]
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.35), sharey=True, constrained_layout=True)
    for ax, noise in zip(axes[:4], noises):
        subset = summary[summary.noise == noise]
        for family, model_id in ids.items():
            curve = subset[subset.model_id == model_id].sort_values("snr_db")
            ax.plot(
                curve.snr_db,
                curve.auroc,
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.set_title(noise)
        ax.set_xlabel("Input SNR (dB)")
        ax.set_xticks([0, 6, 12, 24])
        ax.grid(axis="y", alpha=0.2)
    for family, model_id in ids.items():
        value = baseline.loc[baseline.model_id == model_id, "auroc"]
        if len(value):
            axes[4].scatter(
                [0],
                [value.iloc[0]],
                s=28,
                color=COLORS[family],
                label=FAMILY_LABELS[family],
            )
    axes[4].set_title("Baseline drift")
    axes[4].set_xticks([0], ["Synthetic"])
    axes[4].grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Macro SHD AUROC")
    axes[-1].legend(frameon=False, loc="lower right")
    for label, ax in zip(("A", "B", "C", "D", "E"), axes):
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
        [args.database, selection_path],
        metadata={
            "evaluation_type": "simulation_only perturbations with real fixed labels",
            "aggregation": "mean across 12 SHD tasks",
            "conditions": "Gaussian and NSTDB BW/EM/MA at four SNRs plus synthetic baseline drift",
            "record_scope": (
                "deterministic first 2,198 records of the 5,442-record clean "
                "test set"
            ),
            "classifier_inputs": (
                "perturbed/reconstructed waveform plus unchanged original "
                "tabular metadata"
            ),
        },
    )


if __name__ == "__main__":
    main()
