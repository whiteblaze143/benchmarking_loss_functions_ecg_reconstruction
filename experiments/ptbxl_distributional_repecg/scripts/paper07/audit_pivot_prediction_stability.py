#!/usr/bin/env python3
"""Milestone R027: Pivot-Boundary Prediction Stability Audit & Parseval Frame Comparison.

Audits whether the ~35-43% latent step jumps at hard canonical-frame pivot crossings
destabilize clinical predictions (logits, probabilities, and class decisions),
and compares against the smooth Projected Parseval Frame.

Evaluates along identical Grassmann geodesics (r=2, 3, 4, 15 paths, 100 steps, N=50 patients):
1. Condition 1: Original-Basis (model_basis on raw geodesic frame B(t))
2. Condition 2: Span-Only     (model_span on raw geodesic frame B(t))
3. Condition 3: CF-Span       (model_span on hard canonical frame B_Q(t) = C(P(t)))
4. Condition 4: Parseval-Span (model_span on smooth projected Parseval frame v_i = P(t) e_i)

Computes:
- Delta z: ||z(t+Dt) - z(t)||_2
- Delta l: ||l(t+Dt) - l(t)||_inf (max logit step)
- Delta p: ||sigma(l(t+Dt)) - sigma(l(t))||_inf (max probability step)
- Class Flips: indicator if any of the 5 superclass decisions flip
- Prediction Jump Ratio: J_pred = E[Delta p | pivot] / E[Delta p | nonpivot]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

# Disable cuDNN 1D conv bug on this environment
cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(EXPERIMENT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

sys.path.append(str(Path(__file__).resolve().parent))
from build_r024_datasets import FastResponseComputer
from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.canonical_frame import canonical_frame, observable_projector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pivot_stability_audit")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
MODELS_DIR = CODEX_DIR / "paper07_operator/r024_models"
VAL_CACHE = CODEX_DIR / "paper02_kernel_mean/development_select"
RESPONSE_FIT = CODEX_DIR / "paper07_operator/development_representations/response_fit.npz"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r027_heldout_grassmann_results"
ARTIFACT_DIR = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051")


def sample_grassmann_subspace(rng: np.random.Generator, rank: int, d: int = 8) -> np.ndarray:
    G = rng.normal(size=(d, rank))
    Q, _ = np.linalg.qr(G)
    return Q.T.astype(np.float64)  # [rank, d]


def construct_grassmann_geodesic(
    B0: np.ndarray,
    B1: np.ndarray,
    n_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r, d = B0.shape
    U0 = B0.T
    U1 = B1.T

    M = U0.T @ U1
    W, s, Vt = np.linalg.svd(M)
    Z = Vt.T

    s_clipped = np.clip(s, 0.0, 1.0)
    thetas = np.arccos(s_clipped)
    geodesic_dist = float(np.sqrt(np.sum(thetas ** 2)))

    U1_proj = U0 @ (U0.T @ U1)
    U1_orth = U1 - U1_proj
    U1_orth_rot = U1_orth @ Z

    sin_thetas = np.sin(thetas)
    X = np.zeros((d, r), dtype=np.float64)
    for i in range(r):
        if sin_thetas[i] > 1e-10:
            X[:, i] = U1_orth_rot[:, i] / sin_thetas[i]
        else:
            rand_v = np.zeros(d)
            rand_v[i] = 1.0
            proj = U0 @ (U0.T @ rand_v)
            diff = rand_v - proj
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-6:
                X[:, i] = diff / norm_diff

    for i in range(r):
        X[:, i] -= U0 @ (U0.T @ X[:, i])
        for j in range(i):
            X[:, i] -= np.dot(X[:, j], X[:, i]) * X[:, j]
        norm = np.linalg.norm(X[:, i])
        if norm > 1e-10:
            X[:, i] /= norm

    t_steps = np.linspace(0.0, 1.0, n_steps)
    frames = []
    for t in t_steps:
        cos_t = np.cos(t * thetas)
        sin_t = np.sin(t * thetas)
        U_t = (U0 @ W) * cos_t[None, :] + X * sin_t[None, :]
        Q_t, _ = np.linalg.qr(U_t)
        frames.append(Q_t.T)

    return t_steps, np.stack(frames, axis=0), geodesic_dist


def construct_projected_parseval_frame(P: np.ndarray, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the smooth Projected Parseval Frame: v_i = P e_i for i in {0..7}.
    
    Returns:
        u_active: Active normalized directions [m_active, 8] with ||u_i|| = 1.
        w_active: Weights w_i = ||v_i||^2 = P_ii [m_active].
    """
    d = P.shape[0]
    # v_i is the i-th column of P
    V = P.copy()  # [8, 8], each column is v_i
    w = np.diag(P).copy()  # w_i = P_ii = ||v_i||^2 (since P = P^2)
    w_clamped = np.maximum(w, 0.0)

    # Filter active coordinates
    active_idx = np.where(w_clamped > tol)[0]
    if len(active_idx) == 0:
        # Fallback to coordinate with largest diagonal
        active_idx = np.array([int(np.argmax(w_clamped))])

    u_list = []
    w_list = []
    for idx in active_idx:
        norm_i = np.sqrt(w_clamped[idx])
        u_i = V[:, idx] / norm_i
        u_list.append(u_i)
        w_list.append(w_clamped[idx])

    return np.stack(u_list, axis=0), np.array(w_list)


def evaluate_model_at_frame(
    model: OperatorSetModel,
    computer: FastResponseComputer,
    operators: np.ndarray,
    patient_beats: List[np.ndarray],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model context latent, logits, and probabilities.
    
    Returns:
        latents: [N, d_latent]
        logits: [N, 5]
        probs: [N, 5]
    """
    resps = np.stack([computer.compute_record_responses(b, operators) for b in patient_beats], axis=0)
    N = len(patient_beats)
    with torch.inference_mode():
        r_t = torch.as_tensor(resps, dtype=torch.float32, device=device)
        o_t = torch.as_tensor(operators, dtype=torch.float32, device=device).unsqueeze(0).expand(N, -1, -1)
        latent = model.encode_context(o_t, r_t)
        logits = model.head(latent)
        probs = torch.sigmoid(logits)
    return latent.cpu().numpy(), logits.cpu().numpy(), probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Pivot-Boundary Prediction Stability Audit")
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

    logger.info("=== 1. Loading Checkpoints ===")
    basis_ckpt_path = MODELS_DIR / "basis_only/checkpoint.pt"
    span_ckpt_path = MODELS_DIR / "operator_span/checkpoint.pt"

    model_basis = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model_basis.load_state_dict(torch.load(basis_ckpt_path, map_location=device)["state_dict"])
    model_basis.eval()

    model_span = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model_span.load_state_dict(torch.load(span_ckpt_path, map_location=device)["state_dict"])
    model_span.eval()

    logger.info("=== 2. Loading FastResponseComputer and Cohort Beats ===")
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
    logger.info(f"Loaded {len(patient_beats)} patient records.")

    all_path_records = []
    trajectories = []

    conditions = ["orig_basis", "span_only", "cf_span", "parseval_span"]

    for rank in args.ranks:
        logger.info(f"\n=======================================================")
        logger.info(f"--- Auditing Rank {rank} Geodesics across 4 Conditions ---")
        logger.info(f"=======================================================")

        for g_idx in range(args.n_geodesics_per_rank):
            B0 = sample_grassmann_subspace(rng, rank=rank)
            B1 = sample_grassmann_subspace(rng, rank=rank)
            t_steps, frames, g_dist = construct_grassmann_geodesic(B0, B1, n_steps=args.n_steps)

            # Trajectory records per condition
            traj_data = {cond: {"latents": [], "logits": [], "probs": []} for cond in conditions}
            B_cf_frames = []
            P_frames = []

            for s_idx, t in enumerate(t_steps):
                B_raw = frames[s_idx]
                P_curr = B_raw.T @ B_raw
                P_frames.append(P_curr)

                # 1. Hard Canonical Frame
                B_cf = canonical_frame(P_curr, rank=rank, tol=1e-7)
                B_cf_frames.append(B_cf)

                # 2. Smooth Projected Parseval Frame
                U_parseval, _ = construct_projected_parseval_frame(P_curr, tol=1e-5)

                # Condition 1: Original-Basis on raw B_raw
                z1, l1, p1 = evaluate_model_at_frame(model_basis, computer, B_raw, patient_beats, device)
                traj_data["orig_basis"]["latents"].append(z1)
                traj_data["orig_basis"]["logits"].append(l1)
                traj_data["orig_basis"]["probs"].append(p1)

                # Condition 2: Span-Only on raw B_raw
                z2, l2, p2 = evaluate_model_at_frame(model_span, computer, B_raw, patient_beats, device)
                traj_data["span_only"]["latents"].append(z2)
                traj_data["span_only"]["logits"].append(l2)
                traj_data["span_only"]["probs"].append(p2)

                # Condition 3: CF-Span on hard B_cf
                z3, l3, p3 = evaluate_model_at_frame(model_span, computer, B_cf, patient_beats, device)
                traj_data["cf_span"]["latents"].append(z3)
                traj_data["cf_span"]["logits"].append(l3)
                traj_data["cf_span"]["probs"].append(p3)

                # Condition 4: Parseval-Span on smooth U_parseval
                z4, l4, p4 = evaluate_model_at_frame(model_span, computer, U_parseval, patient_beats, device)
                traj_data["parseval_span"]["latents"].append(z4)
                traj_data["parseval_span"]["logits"].append(l4)
                traj_data["parseval_span"]["probs"].append(p4)

            # Compute step displacements along trajectory
            N_steps_eval = len(t_steps) - 1
            step_records = []

            # Determine pivot status from hard canonical frame changes
            is_pivot_jump = []
            dB_cf_list = []
            dP_list = []

            for s in range(N_steps_eval):
                dP = float(np.linalg.norm(P_frames[s+1] - P_frames[s], ord="fro"))
                dB_cf = float(np.linalg.norm(B_cf_frames[s+1] - B_cf_frames[s], ord="fro"))
                piv = (dB_cf > 0.5)
                dP_list.append(dP)
                dB_cf_list.append(dB_cf)
                is_pivot_jump.append(piv)

            is_pivot_jump = np.array(is_pivot_jump)
            n_pivots = int(np.sum(is_pivot_jump))

            path_summary = {
                "rank": rank,
                "geodesic_idx": g_idx,
                "geodesic_dist": g_dist,
                "n_pivot_jumps": n_pivots,
                "conditions": {},
            }

            for cond in conditions:
                dz_list = []
                dl_list = []
                dp_list = []
                flip_list = []

                latents = traj_data[cond]["latents"]
                logits = traj_data[cond]["logits"]
                probs = traj_data[cond]["probs"]

                for s in range(N_steps_eval):
                    # Mean patient latent distance
                    dz = float(np.mean(np.linalg.norm(latents[s+1] - latents[s], axis=1)))
                    # Max logit displacement across classes, averaged over patients
                    dl = float(np.mean(np.max(np.abs(logits[s+1] - logits[s]), axis=1)))
                    # Max probability displacement across classes, averaged over patients
                    dp = float(np.mean(np.max(np.abs(probs[s+1] - probs[s]), axis=1)))
                    # Fraction of patients with a decision flip in any of the 5 classes
                    dec1 = (probs[s] > 0.5)
                    dec2 = (probs[s+1] > 0.5)
                    has_flip = np.any(dec1 != dec2, axis=1)
                    flip_rate = float(np.mean(has_flip))

                    dz_list.append(dz)
                    dl_list.append(dl)
                    dp_list.append(dp)
                    flip_list.append(flip_rate)

                dz_arr = np.array(dz_list)
                dl_arr = np.array(dl_list)
                dp_arr = np.array(dp_list)
                flip_arr = np.array(flip_list)

                # Partition into pivot vs non-pivot steps
                dp_pivot = float(np.mean(dp_arr[is_pivot_jump])) if n_pivots > 0 else 0.0
                dp_nonpivot = float(np.mean(dp_arr[~is_pivot_jump])) if np.any(~is_pivot_jump) else 0.0
                dl_pivot = float(np.mean(dl_arr[is_pivot_jump])) if n_pivots > 0 else 0.0
                dl_nonpivot = float(np.mean(dl_arr[~is_pivot_jump])) if np.any(~is_pivot_jump) else 0.0
                dz_pivot = float(np.mean(dz_arr[is_pivot_jump])) if n_pivots > 0 else 0.0
                dz_nonpivot = float(np.mean(dz_arr[~is_pivot_jump])) if np.any(~is_pivot_jump) else 0.0
                flip_pivot = float(np.mean(flip_arr[is_pivot_jump])) if n_pivots > 0 else 0.0
                flip_nonpivot = float(np.mean(flip_arr[~is_pivot_jump])) if np.any(~is_pivot_jump) else 0.0

                j_pred = float(dp_pivot / max(dp_nonpivot, 1e-6)) if n_pivots > 0 else 1.0
                j_logit = float(dl_pivot / max(dl_nonpivot, 1e-6)) if n_pivots > 0 else 1.0

                path_summary["conditions"][cond] = {
                    "mean_dz_overall": float(np.mean(dz_arr)),
                    "mean_dz_pivot": dz_pivot,
                    "mean_dz_nonpivot": dz_nonpivot,
                    "mean_dl_overall": float(np.mean(dl_arr)),
                    "mean_dl_pivot": dl_pivot,
                    "mean_dl_nonpivot": dl_nonpivot,
                    "mean_dp_overall": float(np.mean(dp_arr)),
                    "mean_dp_pivot": dp_pivot,
                    "mean_dp_nonpivot": dp_nonpivot,
                    "max_dp_overall": float(np.max(dp_arr)),
                    "mean_flip_rate_overall": float(np.mean(flip_arr)),
                    "mean_flip_rate_pivot": flip_pivot,
                    "mean_flip_rate_nonpivot": flip_nonpivot,
                    "J_pred_ratio": j_pred,
                    "J_logit_ratio": j_logit,
                }

            all_path_records.append(path_summary)

            logger.info(
                f"[Rank {rank} Geo {g_idx}] d_G={g_dist:.3f} | Pivots={n_pivots} | "
                f"CF-Span: Dp(pivot)={path_summary['conditions']['cf_span']['mean_dp_pivot']:.4f}, "
                f"Dp(nonpiv)={path_summary['conditions']['cf_span']['mean_dp_nonpivot']:.4f} "
                f"(J_pred={path_summary['conditions']['cf_span']['J_pred_ratio']:.2f}x) | "
                f"Parseval: Dp={path_summary['conditions']['parseval_span']['mean_dp_overall']:.4f} | "
                f"Span-Only: Dp={path_summary['conditions']['span_only']['mean_dp_overall']:.4f}"
            )

            if g_idx == 0:
                trajectories.append({
                    "rank": rank,
                    "t_steps": t_steps[1:].tolist(),
                    "is_pivot": is_pivot_jump.tolist(),
                    "dP": dP_list,
                    "dB_cf": dB_cf_list,
                })

    # Summary by rank and condition
    logger.info("\n==========================================================================")
    logger.info("=== PIVOT-BOUNDARY PREDICTION STABILITY AUDIT SUMMARY ===")
    logger.info("==========================================================================")

    summary_by_rank = {}
    for rank in args.ranks:
        rank_records = [r for r in all_path_records if r["rank"] == rank]
        n_piv_total = sum(r["n_pivot_jumps"] for r in rank_records)
        summary_by_rank[f"rank_{rank}"] = {
            "n_paths": len(rank_records),
            "mean_pivots_per_path": float(np.mean([r["n_pivot_jumps"] for r in rank_records])),
            "conditions": {},
        }
        logger.info(f"\n--- Subspace Rank {rank} ({len(rank_records)} paths, {n_piv_total} total pivot crossings) ---")
        for cond in conditions:
            c_data = [r["conditions"][cond] for r in rank_records]
            c_summary = {
                "mean_dp_nonpivot": float(np.mean([c["mean_dp_nonpivot"] for c in c_data])),
                "mean_dp_pivot": float(np.mean([c["mean_dp_pivot"] for c in c_data])),
                "max_dp_overall": float(np.max([c["max_dp_overall"] for c in c_data])),
                "mean_dl_nonpivot": float(np.mean([c["mean_dl_nonpivot"] for c in c_data])),
                "mean_dl_pivot": float(np.mean([c["mean_dl_pivot"] for c in c_data])),
                "mean_dz_nonpivot": float(np.mean([c["mean_dz_nonpivot"] for c in c_data])),
                "mean_dz_pivot": float(np.mean([c["mean_dz_pivot"] for c in c_data])),
                "flip_rate_nonpivot": float(np.mean([c["mean_flip_rate_nonpivot"] for c in c_data])),
                "flip_rate_pivot": float(np.mean([c["mean_flip_rate_pivot"] for c in c_data])),
                "mean_J_pred_ratio": float(np.mean([c["J_pred_ratio"] for c in c_data])),
                "mean_J_logit_ratio": float(np.mean([c["J_logit_ratio"] for c in c_data])),
            }
            summary_by_rank[f"rank_{rank}"]["conditions"][cond] = c_summary

            logger.info(
                f"  [{cond:13s}] Delta_p(nonpivot)={c_summary['mean_dp_nonpivot']:.4f} | "
                f"Delta_p(pivot)={c_summary['mean_dp_pivot']:.4f} | "
                f"J_pred = {c_summary['mean_J_pred_ratio']:.2f}x | "
                f"Flips(nonpiv)={c_summary['flip_rate_nonpivot']:.1%} -> Flips(piv)={c_summary['flip_rate_pivot']:.1%} | "
                f"Max Delta_p={c_summary['max_dp_overall']:.4f}"
            )

    out_json = OUTPUT_DIR / "pivot_prediction_stability_audit.json"
    with open(out_json, "w") as f:
        json.dump({
            "summary_by_rank": summary_by_rank,
            "all_geodesics": all_path_records,
        }, f, indent=2)
    logger.info(f"\nSaved stability audit results to {out_json}")

    # Plot Figure 5
    plot_stability_figure(summary_by_rank, all_path_records)


def plot_stability_figure(summary_by_rank: Dict[str, Any], all_records: List[Dict[str, Any]]):
    """Plot Figure 5: Prediction Stability Across Pivot Boundaries."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    c_blue = "#1f77b4"
    c_orange = "#ff7f0e"
    c_green = "#2ca02c"
    c_red = "#d62728"
    c_purple = "#9467bd"

    ranks = [2, 3, 4]
    x = np.arange(len(ranks))
    width = 0.2

    # Panel A: Probability Step (Delta p) Pivot vs Non-Pivot for CF-Span
    ax_a = axes[0]
    cf_dp_nonpiv = [summary_by_rank[f"rank_{r}"]["conditions"]["cf_span"]["mean_dp_nonpivot"] for r in ranks]
    cf_dp_piv = [summary_by_rank[f"rank_{r}"]["conditions"]["cf_span"]["mean_dp_pivot"] for r in ranks]
    span_dp_all = [summary_by_rank[f"rank_{r}"]["conditions"]["span_only"]["mean_dp_nonpivot"] for r in ranks]
    parseval_dp_all = [summary_by_rank[f"rank_{r}"]["conditions"]["parseval_span"]["mean_dp_nonpivot"] for r in ranks]

    rects1 = ax_a.bar(x - 1.5*width, span_dp_all, width, label="Span-Only (Raw B)", color=c_blue, alpha=0.85, edgecolor="black")
    rects2 = ax_a.bar(x - 0.5*width, cf_dp_nonpiv, width, label="CF-Span (Non-Pivot)", color=c_green, alpha=0.85, edgecolor="black")
    rects3 = ax_a.bar(x + 0.5*width, cf_dp_piv, width, label="CF-Span (Pivot Crossing)", color=c_orange, alpha=0.85, edgecolor="black")
    rects4 = ax_a.bar(x + 1.5*width, parseval_dp_all, width, label="Parseval-Span (Smooth)", color=c_purple, alpha=0.85, edgecolor="black")

    ax_a.set_xlabel("Subspace Rank $r$", fontsize=11, fontweight="bold")
    ax_a.set_ylabel(r"Mean Output Probability Step $\|\Delta p\|_\infty$", fontsize=11, fontweight="bold")
    ax_a.set_title(r"A. Output Probability Displacements ($\|\Delta p\|_\infty$)", fontsize=12, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"r={r}" for r in ranks], fontsize=11)
    ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_a.legend(framealpha=0.9, fontsize=8.5, loc="upper right")

    for rect in rects3:
        h = rect.get_height()
        ax_a.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # Panel B: Decision Flip Rate Across Conditions
    ax_b = axes[1]
    cf_flip_nonpiv = [summary_by_rank[f"rank_{r}"]["conditions"]["cf_span"]["flip_rate_nonpivot"] * 100 for r in ranks]
    cf_flip_piv = [summary_by_rank[f"rank_{r}"]["conditions"]["cf_span"]["flip_rate_pivot"] * 100 for r in ranks]
    span_flip_all = [summary_by_rank[f"rank_{r}"]["conditions"]["span_only"]["flip_rate_nonpivot"] * 100 for r in ranks]
    parseval_flip_all = [summary_by_rank[f"rank_{r}"]["conditions"]["parseval_span"]["flip_rate_nonpivot"] * 100 for r in ranks]

    ax_b.bar(x - 1.5*width, span_flip_all, width, label="Span-Only", color=c_blue, alpha=0.85, edgecolor="black")
    ax_b.bar(x - 0.5*width, cf_flip_nonpiv, width, label="CF-Span (Non-Pivot)", color=c_green, alpha=0.85, edgecolor="black")
    rects_fp = ax_b.bar(x + 0.5*width, cf_flip_piv, width, label="CF-Span (Pivot Crossing)", color=c_orange, alpha=0.85, edgecolor="black")
    ax_b.bar(x + 1.5*width, parseval_flip_all, width, label="Parseval-Span", color=c_purple, alpha=0.85, edgecolor="black")

    ax_b.set_xlabel("Subspace Rank $r$", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Class Decision Flip Rate (%)", fontsize=11, fontweight="bold")
    ax_b.set_title(r"B. Superclass Decision Flip Rate", fontsize=12, fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"r={r}" for r in ranks], fontsize=11)
    ax_b.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_b.legend(framealpha=0.9, fontsize=8.5, loc="upper right")

    for rect in rects_fp:
        h = rect.get_height()
        ax_b.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # Panel C: Decisive Jump Ratio J_pred = E[Delta p | pivot] / E[Delta p | nonpivot]
    ax_c = axes[2]
    j_ratios = [summary_by_rank[f"rank_{r}"]["conditions"]["cf_span"]["mean_J_pred_ratio"] for r in ranks]
    j_logits = [summary_by_rank[f"rank_{r}"]["conditions"]["cf_span"]["mean_J_logit_ratio"] for r in ranks]

    ax_c.bar(x - width/2, j_ratios, width, label=r"Probability Jump Ratio $J_{\rm pred}$", color=c_red, alpha=0.85, edgecolor="black")
    ax_c.bar(x + width/2, j_logits, width, label=r"Logit Jump Ratio $J_{\rm logit}$", color=c_orange, alpha=0.85, edgecolor="black")
    ax_c.axhline(1.0, color="gray", linestyle="--", lw=1.5, label="1.0 (No Boundary Inflation)")

    ax_c.set_xlabel("Subspace Rank $r$", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("Boundary Jump Inflation Factor ($x$)", fontsize=11, fontweight="bold")
    ax_c.set_title(r"C. Decisive Prediction Jump Ratio ($J_{\rm pred}$)", fontsize=12, fontweight="bold")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"r={r}" for r in ranks], fontsize=11)
    ax_c.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax_c.legend(framealpha=0.9, fontsize=9, loc="upper right")

    for rect in ax_c.patches[:len(ranks)]:
        h = rect.get_height()
        ax_c.annotate(f"{h:.2f}x", xy=(rect.get_x() + rect.get_width()/2, h),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out_png = OUTPUT_DIR / "r027_pivot_prediction_stability.png"
    art_png = ARTIFACT_DIR / "r027_pivot_prediction_stability.png"
    fig.savefig(out_png, dpi=300)
    fig.savefig(art_png, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Figure 5 to {out_png} and {art_png}")


if __name__ == "__main__":
    main()
