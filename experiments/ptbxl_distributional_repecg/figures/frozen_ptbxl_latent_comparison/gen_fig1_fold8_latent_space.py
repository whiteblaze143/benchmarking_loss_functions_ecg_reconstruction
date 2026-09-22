#!/usr/bin/env python3
"""Plot per-encoder PCA views of frozen Fold-8 latent spaces."""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from paper_plot_style import COLORS, save, plt

HERE = Path(__file__).resolve().parent
MODELS = (("graphecg", "GraphECG"), ("setop_robust", "SetOp robust"),
          ("setop_robust_aux", "SetOp robust + aux"), ("setop_fulllead", "SetOp full-lead only"),
          ("fixed_tensor", "FixedTensor"))
CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def main() -> None:
    item = np.load(HERE / "frozen_fold8_latents.npz")
    labels = item["labels"].astype(bool)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.1), constrained_layout=True)
    for axis, (key, name) in zip(axes.flat, MODELS):
        z = item[key]
        xy = PCA(n_components=2, svd_solver="full").fit_transform((z - z.mean(0)) / (z.std(0) + 1e-8))
        axis.scatter(xy[:, 0], xy[:, 1], c="0.78", s=2.0, alpha=0.18, linewidths=0, rasterized=True)
        for i, label in enumerate(CLASSES):
            present = labels[:, i]
            axis.scatter(xy[present, 0], xy[present, 1], c=COLORS[label], s=3.0, alpha=0.52,
                         linewidths=0, label=label, rasterized=True)
        axis.set_xlabel("PC 1 (within-model standardized latent)")
        axis.set_ylabel("PC 2")
        axis.text(0.02, 0.98, name, transform=axis.transAxes, va="top", fontsize=9)
    axes.flat[-1].set_visible(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, "fig1_fold8_latent_spaces")


if __name__ == "__main__":
    main()
