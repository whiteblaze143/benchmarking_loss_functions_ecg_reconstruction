from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
OUTPUT = EXPERIMENT / "outputs/paper02_kernel_mean"


def main() -> None:
    evaluation = OUTPUT / "development_evaluation"
    destination = evaluation / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(evaluation / "metrics.csv").set_index("variant")
    order = ["kernel", "moments", "gaussian", "linear"]

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    colors = ["#31688e", "#35b779", "#fde725", "#440154"]
    axis.bar(order, metrics.loc[order, "macro_auroc"], color=colors)
    axis.set_ylabel("Validation macro AUROC")
    axis.set_ylim(0.5, 1.0)
    axis.set_title("Paper 2 development representations")
    figure.tight_layout()
    figure.savefig(destination / "development_macro_auroc.png", dpi=200)
    figure.savefig(destination / "development_macro_auroc.pdf")
    plt.close(figure)

    stability = pd.read_csv(OUTPUT / "development_robustness/odd_even_stability.csv")
    patient = stability.groupby("patient_id", sort=False)[["kernel_similarity", "moments_similarity"]].mean()
    figure, axis = plt.subplots(figsize=(5.0, 5.0))
    axis.scatter(patient.moments_similarity, patient.kernel_similarity, s=7, alpha=0.25)
    axis.plot([0, 1], [0, 1], color="black", linewidth=1)
    axis.set_xlabel("Moments odd/even similarity")
    axis.set_ylabel("KME odd/even similarity")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(destination / "odd_even_stability.png", dpi=200)
    figure.savefig(destination / "odd_even_stability.pdf")
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
