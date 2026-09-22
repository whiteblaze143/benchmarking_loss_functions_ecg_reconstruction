#!/usr/bin/env python3
"""Plot PCA variance spectra; a scale-free comparison across latent dimensions."""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from paper_plot_style import save, plt

HERE = Path(__file__).resolve().parent
MODELS = (("graphecg", "GraphECG", "#4C4C4C"), ("setop_robust", "SetOp robust", "#0072B2"),
          ("setop_robust_aux", "SetOp robust + aux", "#009E73"), ("setop_fulllead", "SetOp full-lead only", "#D55E00"),
          ("fixed_tensor", "FixedTensor", "#7F7F7F"))


def main() -> None:
    item = np.load(HERE / "frozen_fold8_latents.npz")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for key, name, color in MODELS:
        z = item[key]
        z = (z - z.mean(0)) / (z.std(0) + 1e-8)
        cumulative = np.cumsum(PCA(svd_solver="full").fit(z).explained_variance_ratio_)
        x = np.arange(1, len(cumulative) + 1)
        ax.plot(x, cumulative, label=name, color=color, linewidth=1.5)
    ax.axhline(0.90, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Principal components retained")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_xlim(1, 256)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig2_fold8_latent_spectrum")


if __name__ == "__main__":
    main()
