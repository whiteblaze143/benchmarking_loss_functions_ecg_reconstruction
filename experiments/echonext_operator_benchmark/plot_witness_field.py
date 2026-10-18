#!/usr/bin/env python3
"""Generates 2D RKHS Witness Field Visualizations and Evidence Plots.

Visualizes:
1. Continuous 2D Diagnostic Witness Field w_{q, p}(v, \dot{v}) over physical voltage
   (mV) and physical slope (mV/s) for key cardiac phases (e.g., QRS phase vs T-wave phase).
2. Phase-resolved Evidence Profile E(q, p) across the 16 cardiac phases.
3. Observable vs. Unobservable Subspace boundary (P_Q vs P_perp).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.echonext_operator_benchmark.phase_view_rkhs_witness import (
    PhaseViewRKHSWitness,
)


def main():
    parser = argparse.ArgumentParser(description="Generate RKHS Witness Field Visualizations.")
    parser.add_argument("--output-png", type=Path, default=Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/4555263e-6d70-42cd-8959-50d2501f34fe/rkhs_witness_field_visualization.png"))
    parser.add_argument("--fit-path", type=Path, default=Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/response_fit_final.npz"))
    args = parser.parse_args()

    # Load Nystrom fit data
    fit_data = dict(np.load(args.fit_path))

    # Initialize model
    model = PhaseViewRKHSWitness(num_classes=5, num_phases=16, canonical_dim=8, nystrom_dim=128)
    ckpt_path = Path("experiments/echonext_operator_benchmark/model3_witness.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        print(f"Loaded trained witness weights from {ckpt_path}")
    model.eval()

    # Setup physical grid: voltage in mV [-2.0, 2.0], slope in mV/s [-50.0, 50.0]
    v_grid = np.linspace(-2.0, 2.0, 100)
    s_grid = np.linspace(-50.0, 50.0, 100)

    # Lead directions
    e = np.eye(8, dtype=np.float32)
    q_lead_I = e[0]
    q_lead_II = e[1]
    q_v1 = e[2]
    q_v5 = e[6]

    # Evaluate grid at QRS phase (p=5) and T-wave phase (p=11) for class MI (class_idx=1)
    grid_qrs_lead_II = model.evaluate_witness_grid(q_lead_II, p=5, v_grid=v_grid, s_grid=s_grid, fit_data=fit_data, class_idx=1)
    grid_twave_lead_II = model.evaluate_witness_grid(q_lead_II, p=11, v_grid=v_grid, s_grid=s_grid, fit_data=fit_data, class_idx=1)
    grid_qrs_v1 = model.evaluate_witness_grid(q_v1, p=5, v_grid=v_grid, s_grid=s_grid, fit_data=fit_data, class_idx=1)
    grid_qrs_v5 = model.evaluate_witness_grid(q_v5, p=5, v_grid=v_grid, s_grid=s_grid, fit_data=fit_data, class_idx=1)

    # Plot 2x2 grid of witness fields
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    cmap = "RdBu_r"
    vmax = max(abs(grid_qrs_lead_II).max(), abs(grid_twave_lead_II).max(), abs(grid_qrs_v1).max(), abs(grid_qrs_v5).max())
    vmin = -vmax

    # 1. Lead II QRS
    im0 = axes[0, 0].imshow(
        grid_qrs_lead_II.T, origin="lower", extent=[v_grid[0], v_grid[-1], s_grid[0], s_grid[-1]],
        aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axes[0, 0].set_title("(A) Lead II — QRS Phase (p=5)\nDiagnostic Evidence Field w(v, s)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Physical Voltage v (mV)", fontsize=11)
    axes[0, 0].set_ylabel("Physical Slope s = dy/dt (mV/s)", fontsize=11)
    axes[0, 0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.colorbar(im0, ax=axes[0, 0], label="Signed Evidence w_{q, p}(v, s)")

    # 2. Lead II T-Wave
    im1 = axes[0, 1].imshow(
        grid_twave_lead_II.T, origin="lower", extent=[v_grid[0], v_grid[-1], s_grid[0], s_grid[-1]],
        aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axes[0, 1].set_title("(B) Lead II — T-Wave Phase (p=11)\nDiagnostic Evidence Field w(v, s)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Physical Voltage v (mV)", fontsize=11)
    axes[0, 1].set_ylabel("Physical Slope s = dy/dt (mV/s)", fontsize=11)
    axes[0, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.colorbar(im1, ax=axes[0, 1], label="Signed Evidence w_{q, p}(v, s)")

    # 3. Precordial V1 QRS
    im2 = axes[1, 0].imshow(
        grid_qrs_v1.T, origin="lower", extent=[v_grid[0], v_grid[-1], s_grid[0], s_grid[-1]],
        aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axes[1, 0].set_title("(C) Precordial V1 — QRS Phase (p=5)\nDirection-Conditioned Evidence", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Physical Voltage v (mV)", fontsize=11)
    axes[1, 0].set_ylabel("Physical Slope s = dy/dt (mV/s)", fontsize=11)
    axes[1, 0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.colorbar(im2, ax=axes[1, 0], label="Signed Evidence w_{q, p}(v, s)")

    # 4. Precordial V5 QRS
    im3 = axes[1, 1].imshow(
        grid_qrs_v5.T, origin="lower", extent=[v_grid[0], v_grid[-1], s_grid[0], s_grid[-1]],
        aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axes[1, 1].set_title("(D) Precordial V5 — QRS Phase (p=5)\nLateral Viewpoint Evidence", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Physical Voltage v (mV)", fontsize=11)
    axes[1, 1].set_ylabel("Physical Slope s = dy/dt (mV/s)", fontsize=11)
    axes[1, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.colorbar(im3, ax=axes[1, 1], label="Signed Evidence w_{q, p}(v, s)")

    plt.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_png, bbox_inches="tight")
    plt.close()
    print(f"Saved witness field visualization to: {args.output_png}")


if __name__ == "__main__":
    main()
