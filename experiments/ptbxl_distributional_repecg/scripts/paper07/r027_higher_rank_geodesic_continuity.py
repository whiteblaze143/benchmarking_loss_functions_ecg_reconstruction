#!/usr/bin/env python3
"""Milestone R027: Higher-Rank Canonical-Frame Geodesic Continuity Audit (r=2, 3, 4).

Audits geodesic continuity of the deterministic canonical section C(P) on Gr(r, 8):
Along random Grassmann geodesics U(t) connecting random subspaces U_0, U_1:
1. Measures projector variation ||P(t+Dt) - P(t)||_F (smooth baseline).
2. Measures canonical frame step ||B(t+Dt) - B(t)||_F (detects pivot transitions).
3. Measures latent representation step ||z(t+Dt) - z(t)||_2 across validation patients.

Quantifies whether pivot-induced basis jumps propagate into latent representations
or remain bounded due to SetTransformer permutation invariance and continuous operator embeddings.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.enabled = False

from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.canonical_frame import canonical_frame, observable_projector
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from build_r024_datasets import FastResponseComputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("higher_rank_continuity")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
MODELS_DIR = CODEX_DIR / "paper07_operator/r024_models"
VAL_CACHE = CODEX_DIR / "paper02_kernel_mean/development_select"
RESPONSE_FIT = CODEX_DIR / "paper07_operator/development_representations/response_fit.npz"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r027_heldout_grassmann_results"
ARTIFACT_DIR = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051")


def sample_grassmann_subspace(rng: np.random.Generator, rank: int, d: int = 8) -> np.ndarray:
    """Sample uniform random orthonormal basis B in Gr(rank, d)."""
    G = rng.normal(size=(d, rank))
    Q, _ = np.linalg.qr(G)
    return Q.T.astype(np.float64)  # [rank, d]


def construct_grassmann_geodesic(
    B0: np.ndarray,
    B1: np.ndarray,
    n_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Construct exact Grassmann geodesic U(t) for t in [0, 1].

    Args:
        B0: Orthonormal row basis [r, d].
        B1: Orthonormal row basis [r, d].
        n_steps: Number of evaluation steps along [0, 1].

    Returns:
        t_steps: 1D array of t values [n_steps].
        geodesic_frames: Array of orthonormal bases [n_steps, r, d].
        geodesic_dist: Total Grassmann geodesic distance d_G(U0, U1).
    """
    r, d = B0.shape
    U0 = B0.T  # [d, r]
    U1 = B1.T  # [d, r]

    # Thin SVD of U0^T U1
    M = U0.T @ U1  # [r, r]
    W, s, Vt = np.linalg.svd(M)
    Z = Vt.T

    # Principal angles theta_i = arccos(clip(s_i, 0, 1))
    s_clipped = np.clip(s, 0.0, 1.0)
    thetas = np.arccos(s_clipped)
    geodesic_dist = float(np.sqrt(np.sum(thetas ** 2)))

    # Compute orthogonal component: (I - U0 U0^T) U1 Z
    U1_proj = U0 @ (U0.T @ U1)
    U1_orth = U1 - U1_proj
    U1_orth_rot = U1_orth @ Z  # [d, r]

    # Normalize columns of U1_orth_rot to obtain X
    # (sin(theta_i) is the column norm)
    sin_thetas = np.sin(thetas)
    X = np.zeros((d, r), dtype=np.float64)
    for i in range(r):
        if sin_thetas[i] > 1e-10:
            X[:, i] = U1_orth_rot[:, i] / sin_thetas[i]
        else:
            # Collinear principal direction; pick arbitrary orthogonal unit vector
            rand_v = np.zeros(d)
            rand_v[i] = 1.0
            proj = U0 @ (U0.T @ rand_v)
            diff = rand_v - proj
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-6:
                X[:, i] = diff / norm_diff

    # Orthonormalize X with Gram-Schmidt against U0 and previous columns
    for i in range(r):
        X[:, i] -= U0 @ (U0.T @ X[:, i])
        for j in range(i):
            X[:, i] -= np.dot(X[:, j], X[:, i]) * X[:, j]
        norm = np.linalg.norm(X[:, i])
        if norm > 1e-10:
            X[:, i] /= norm

    # Evaluate along t in [0, 1]
    t_steps = np.linspace(0.0, 1.0, n_steps)
    frames = []
    for t in t_steps:
        cos_t = np.cos(t * thetas)
        sin_t = np.sin(t * thetas)
        # U(t) = U0 @ W @ diag(cos(t thetas)) + X @ diag(sin(t thetas))
        U_t = (U0 @ W) * cos_t[None, :] + X * sin_t[None, :]
        # Re-orthonormalize for numerical precision
        Q_t, _ = np.linalg.qr(U_t)
        frames.append(Q_t.T)  # [r, d]

    return t_steps, np.stack(frames, axis=0), geodesic_dist


def main():
    parser = argparse.ArgumentParser(description="Higher-Rank Geodesic Continuity Audit")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--n_geodesics_per_rank", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_patients", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    logger.info("=== 1. Loading Unified Model Checkpoint (CF-Span) ===")
    span_ckpt_path = MODELS_DIR / "operator_span/checkpoint.pt"
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model.load_state_dict(torch.load(span_ckpt_path, map_location=device)["state_dict"])
    model.eval()

    logger.info("=== 2. Loading FastResponseComputer and Patient Cohort ===")
    computer = FastResponseComputer(RESPONSE_FIT, device)
    qc_frame = pd.read_csv(VAL_CACHE / "qc.csv").query("eligible").reset_index(drop=True)
    sample_qc = qc_frame.iloc[:args.n_patients]

    def _load_beat(idx):
        p = VAL_CACHE / sample_qc.iloc[idx].artifact
        with np.load(p) as item:
            b = np.asarray(item["beats"], dtype=np.float32) * computer.std + computer.mean
            return b

    with ThreadPoolExecutor(max_workers=8) as ex:
        patient_beats = list(ex.map(_load_beat, range(len(sample_qc))))
    logger.info(f"Loaded {len(patient_beats)} patient records for latent continuity audit.")

    audit_results = []
    trajectories = []

    for rank in args.ranks:
        logger.info(f"\n--- Auditing Rank {rank} Geodesics ---")
        for g_idx in range(args.n_geodesics_per_rank):
            B0 = sample_grassmann_subspace(rng, rank=rank)
            B1 = sample_grassmann_subspace(rng, rank=rank)
            t_steps, frames, g_dist = construct_grassmann_geodesic(B0, B1, n_steps=args.n_steps)

            # Trajectory arrays
            P_diffs = []
            B_diffs = []
            z_diffs = []
            latent_norms = []

            # Store latents at each t
            prev_P = None
            prev_B = None
            prev_z = None

            for s_idx, t in enumerate(t_steps):
                B_curr = frames[s_idx]
                P_curr = B_curr.T @ B_curr

                # Compute canonical frame C(P_curr)
                B_canon = canonical_frame(P_curr, rank=rank, tol=1e-7)

                # Compute patient responses under canonical frame
                resps = np.stack([computer.compute_record_responses(b, B_canon) for b in patient_beats], axis=0)
                # Compute latent representations z in R^{N, 128}
                with torch.inference_mode():
                    r_tensor = torch.as_tensor(resps, dtype=torch.float32, device=device)
                    o_tensor = torch.as_tensor(B_canon, dtype=torch.float32, device=device).unsqueeze(0).expand(len(resps), -1, -1)
                    z_tensor = model.encode_context(o_tensor, r_tensor)  # [N, d_latent]
                    z_curr = z_tensor.cpu().numpy()

                latent_norms.append(float(np.mean(np.linalg.norm(z_curr, axis=1))))

                if prev_P is not None:
                    # Step changes
                    dP = float(np.linalg.norm(P_curr - prev_P, ord="fro"))
                    dB = float(np.linalg.norm(B_canon - prev_B, ord="fro"))
                    # Average patient latent distance
                    dz = float(np.mean(np.linalg.norm(z_curr - prev_z, axis=1)))

                    P_diffs.append(dP)
                    B_diffs.append(dB)
                    z_diffs.append(dz)

                prev_P = P_curr
                prev_B = B_canon
                prev_z = z_curr

            P_diffs = np.array(P_diffs)
            B_diffs = np.array(B_diffs)
            z_diffs = np.array(z_diffs)

            # Classify steps into pivot transitions vs continuous steps
            pivot_threshold = 0.5  # Frobenius norm jump for basis reorientation
            is_pivot_jump = (B_diffs > pivot_threshold)
            n_pivots = int(np.sum(is_pivot_jump))

            mean_dz_cont = float(np.mean(z_diffs[~is_pivot_jump])) if np.any(~is_pivot_jump) else 0.0
            mean_dz_pivot = float(np.mean(z_diffs[is_pivot_jump])) if np.any(is_pivot_jump) else 0.0
            max_dz = float(np.max(z_diffs))
            mean_dz = float(np.mean(z_diffs))
            mean_latent_norm = float(np.mean(latent_norms))
            # Relative latent step: dz / ||z||
            max_rel_step = max_dz / max(mean_latent_norm, 1e-6)

            geo_record = {
                "rank": rank,
                "geodesic_idx": g_idx,
                "geodesic_dist": g_dist,
                "n_steps": args.n_steps,
                "n_pivot_jumps": n_pivots,
                "mean_dP": float(np.mean(P_diffs)),
                "max_dP": float(np.max(P_diffs)),
                "mean_dB": float(np.mean(B_diffs)),
                "max_dB": float(np.max(B_diffs)),
                "mean_dz_continuous": mean_dz_cont,
                "mean_dz_pivot": mean_dz_pivot,
                "mean_dz_overall": mean_dz,
                "max_dz_overall": max_dz,
                "mean_latent_norm": mean_latent_norm,
                "max_rel_latent_step": max_rel_step,
            }
            audit_results.append(geo_record)

            logger.info(
                f"[Rank {rank} Geo {g_idx}] d_G={g_dist:.3f} | Pivots={n_pivots} | "
                f"Max dB={geo_record['max_dB']:.3f} | "
                f"Mean dz(cont)={mean_dz_cont:.4f}, Mean dz(pivot)={mean_dz_pivot:.4f}, "
                f"Max dz={max_dz:.4f} (rel={max_rel_step:.1%})"
            )

            # Store first trajectory per rank for visualization
            if g_idx == 0:
                trajectories.append({
                    "rank": rank,
                    "t_steps": t_steps[1:].tolist(),
                    "P_diffs": P_diffs.tolist(),
                    "B_diffs": B_diffs.tolist(),
                    "z_diffs": z_diffs.tolist(),
                    "is_pivot": is_pivot_jump.tolist(),
                })

    df = pd.DataFrame(audit_results)
    logger.info("\n=== HIGHER-RANK GEODESIC CONTINUITY AUDIT SUMMARY ===")
    summary_by_rank = {}
    for r in args.ranks:
        sub_df = df[df["rank"] == r]
        summary_by_rank[f"rank_{r}"] = {
            "mean_geodesic_dist": float(sub_df["geodesic_dist"].mean()),
            "mean_pivot_jumps_per_path": float(sub_df["n_pivot_jumps"].mean()),
            "mean_max_dB": float(sub_df["max_dB"].mean()),
            "mean_dz_continuous": float(sub_df["mean_dz_continuous"].mean()),
            "mean_dz_pivot": float(sub_df["mean_dz_pivot"].mean()),
            "max_dz_overall": float(sub_df["max_dz_overall"].max()),
            "mean_latent_norm": float(sub_df["mean_latent_norm"].mean()),
            "worst_rel_latent_step": float(sub_df["max_rel_latent_step"].max()),
        }
        logger.info(
            f"[Rank {r}] Mean pivots/path={summary_by_rank[f'rank_{r}']['mean_pivot_jumps_per_path']:.1f} | "
            f"Mean dz(cont)={summary_by_rank[f'rank_{r}']['mean_dz_continuous']:.4f} | "
            f"Mean dz(pivot)={summary_by_rank[f'rank_{r}']['mean_dz_pivot']:.4f} | "
            f"Worst dz={summary_by_rank[f'rank_{r}']['max_dz_overall']:.4f} "
            f"(rel={summary_by_rank[f'rank_{r}']['worst_rel_latent_step']:.1%})"
        )

    # Save output JSON
    out_json = OUTPUT_DIR / "higher_rank_geodesic_continuity.json"
    with open(out_json, "w") as f:
        json.dump({
            "summary_by_rank": summary_by_rank,
            "all_geodesics": audit_results,
            "sample_trajectories": trajectories,
        }, f, indent=2)
    logger.info(f"Saved continuity audit results to {out_json}")

    # Plot Figure 4
    plot_continuity_figure(trajectories, df)


def plot_continuity_figure(trajectories: List[Dict[str, Any]], df: pd.DataFrame):
    """Plot Figure 4: Higher-Rank Canonical-Frame Geodesic Continuity Audit."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Colors
    c_blue = "#1f77b4"
    c_green = "#2ca02c"
    c_orange = "#ff7f0e"
    c_red = "#d62728"
    c_purple = "#9467bd"

    # Panel A: Step variations along a representative geodesic (Rank 3)
    traj_r3 = next((t for t in trajectories if t["rank"] == 3), trajectories[0])
    t_vals = np.array(traj_r3["t_steps"])
    P_d = np.array(traj_r3["P_diffs"])
    B_d = np.array(traj_r3["B_diffs"])
    z_d = np.array(traj_r3["z_diffs"])
    piv = np.array(traj_r3["is_pivot"])

    ax_a = axes[0]
    ax_a2 = ax_a.twinx()

    l1 = ax_a.plot(t_vals, P_d, color=c_blue, lw=2.0, label=r"Projector $\|\Delta P\|_F$ (smooth)")
    l2 = ax_a.plot(t_vals, B_d, color=c_red, lw=1.8, linestyle="--", label=r"Canonical Basis $\|\Delta B\|_F$")
    l3 = ax_a2.plot(t_vals, z_d, color=c_green, lw=2.2, label=r"Latent $\|\Delta z\|_2$ (SetOperator)")

    # Highlight pivot events
    pivot_t = t_vals[piv]
    for pt in pivot_t:
        ax_a.axvline(pt, color="gray", alpha=0.3, linestyle=":")

    ax_a.set_xlabel(r"Geodesic Parameter $t \in [0, 1]$", fontsize=11, fontweight="bold")
    ax_a.set_ylabel(r"Matrix Frobenius Step $\|\Delta M\|_F$", fontsize=11, fontweight="bold")
    ax_a2.set_ylabel(r"Latent Representation Step $\|\Delta z\|_2$", fontsize=11, fontweight="bold", color=c_green)
    ax_a2.tick_params(axis="y", labelcolor=c_green)
    ax_a.set_title(r"A. Step Displacements Along Geodesic ($r=3$)", fontsize=12, fontweight="bold")
    ax_a.grid(True, linestyle=":", alpha=0.5)

    # Combined legend
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax_a.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    # Panel B: Latent step size comparison: Pivot Jump vs Continuous Steps
    ax_b = axes[1]
    ranks = sorted(df["rank"].unique())
    x_positions = np.arange(len(ranks))
    width = 0.35

    mean_cont = [df[df["rank"] == r]["mean_dz_continuous"].mean() for r in ranks]
    mean_piv = [df[df["rank"] == r]["mean_dz_pivot"].mean() for r in ranks]

    rects1 = ax_b.bar(x_positions - width/2, mean_cont, width, label="Continuous Steps", color=c_blue, alpha=0.85, edgecolor="black")
    rects2 = ax_b.bar(x_positions + width/2, mean_piv, width, label="Pivot-Crossing Steps", color=c_orange, alpha=0.85, edgecolor="black")

    ax_b.set_xlabel("Subspace Rank $r$", fontsize=11, fontweight="bold")
    ax_b.set_ylabel(r"Mean Latent Displacement $\|\Delta z\|_2$", fontsize=11, fontweight="bold")
    ax_b.set_title(r"B. Latent Impact of Pivot Crossings", fontsize=12, fontweight="bold")
    ax_b.set_xticks(x_positions)
    ax_b.set_xticklabels([f"r={r}" for r in ranks], fontsize=11)
    ax_b.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_b.legend(framealpha=0.9, fontsize=10)

    # Add text labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax_b.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        ax_b.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # Panel C: Relative Latent Step vs Basis Jump Magnitude
    ax_c = axes[2]
    for r, col in zip(ranks, [c_blue, c_green, c_purple]):
        sub_df = df[df["rank"] == r]
        ax_c.scatter(sub_df["max_dB"], sub_df["max_rel_latent_step"] * 100, label=f"Rank {r}", color=col, s=70, alpha=0.8, edgecolors="none")

    ax_c.axhline(5.0, color=c_red, linestyle="--", lw=1.5, label="5% Relative Threshold")
    ax_c.set_xlabel(r"Maximum Basis Jump $\|\Delta B\|_F$", fontsize=11, fontweight="bold")
    ax_c.set_ylabel(r"Max Relative Latent Step $\|\Delta z\|_2 / \|z\|_2$ (%)", fontsize=11, fontweight="bold")
    ax_c.set_title(r"C. Bound on Worst-Case Latent Perturbation", fontsize=12, fontweight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(framealpha=0.9, fontsize=10)
    ax_c.set_ylim(0, max(6.0, df["max_rel_latent_step"].max() * 115))

    plt.tight_layout()
    out_png = OUTPUT_DIR / "r027_higher_rank_continuity.png"
    art_png = ARTIFACT_DIR / "r027_higher_rank_continuity.png"
    fig.savefig(out_png, dpi=300)
    fig.savefig(art_png, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Figure 4 to {out_png} and {art_png}")


if __name__ == "__main__":
    main()
