#!/usr/bin/env python3
"""Final Methodological Experiment: Held-Out Grassmann Benchmark.

Evaluates the 4 cells of the 2x2 factorial matrix on frozen random subspaces
sampled uniformly from Gr(r, 8) for r in {1, 2, 3, 4, 6}.

For each rank r:
  1. Sample 20 random orthonormal subspaces B_U in Gr(r, 8). (Total 100 subspaces)
  2. Compute Grassmann geodesic distance d_G(U, V) = sqrt(sum theta_i^2)
     to the nearest coordinate training subspace in V_coord^(r).
  3. Evaluate Quotient Invariance V(U) = max_k ||f(Q_{U,k}, Y_{U,k}) - f(B_U, Y_U)||
     across 10 random invertible remixings R_k B_U and redundant sets [B_U; A B_U].
  4. Evaluate Macro AUROC on PTB-XL (N=2,173) for all 4 cells:
     - Cell (1,1): Orig-Basis
     - Cell (1,2): CF-Basis
     - Cell (2,1): Orig-Span
     - Cell (2,2): CF-Span
  5. Compute Delta_span(U) = AUROC_span(U) - AUROC_basis(U) vs d_train(U).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score, roc_auc_score

# Disable cuDNN to avoid PyTorch 2.6 conv1d issue on A100
cudnn.enabled = False

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
import sys
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(EXPERIMENT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR / "src"))
if str(EXPERIMENT_DIR / "scripts/paper07") not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR / "scripts/paper07"))

from repecg.paper07_operator import OperatorSetModel
from repecg.paper07_operator.canonical_frame import (
    canonical_frame,
    observable_projector,
)
from build_r024_datasets import FastResponseComputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("grassmann_benchmark")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
VAL_CACHE = CODEX_DIR / "paper02_kernel_mean/development_select"
RESPONSE_FIT = CODEX_DIR / "paper07_operator/development_representations/response_fit.npz"
MODELS_DIR = CODEX_DIR / "paper07_operator/r024_models"
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r027_heldout_grassmann_results"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    class_aurocs = []
    class_auprcs = []
    for c in range(y_true.shape[1]):
        labels_c = y_true[:, c]
        if len(np.unique(labels_c)) > 1:
            class_aurocs.append(float(roc_auc_score(labels_c, y_prob[:, c])))
            class_auprcs.append(float(average_precision_score(labels_c, y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
            class_auprcs.append(0.0)
    return {
        "macro_auroc": float(np.mean(class_aurocs)),
        "micro_auroc": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        "macro_auprc": float(np.mean(class_auprcs)),
    }


def predict_model_batched(
    model: OperatorSetModel,
    operators: np.ndarray,
    responses: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    N = len(responses)
    is_shared_ops = (operators.ndim == 2)
    if is_shared_ops:
        o_base = torch.as_tensor(operators, dtype=torch.float32, device=device).unsqueeze(0)
    all_logits = []
    with torch.inference_mode():
        for i in range(0, N, batch_size):
            r_b = torch.as_tensor(responses[i : i + batch_size], dtype=torch.float32, device=device)
            B_cur = len(r_b)
            if is_shared_ops:
                o_b = o_base.expand(B_cur, -1, -1)
            else:
                o_b = torch.as_tensor(operators[i : i + batch_size], dtype=torch.float32, device=device)
            latent = model.encode_context(o_b, r_b)
            logits = model.head(latent)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def sample_grassmann_subspace(rng: np.random.Generator, rank: int) -> np.ndarray:
    """Sample uniform random orthonormal basis B_U in Gr(rank, 8)."""
    G = rng.normal(size=(8, rank))
    Q, _ = np.linalg.qr(G)
    B_U = Q.T.astype(np.float64)  # [rank, 8]
    return B_U


def grassmann_geodesic_distance(B1: np.ndarray, B2: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute Grassmann geodesic distance d_G(U, V) = sqrt(sum theta_i^2).
    
    B1, B2 are [r, 8] orthonormal row matrices.
    """
    # Singular values of B1 @ B2.T
    M = B1 @ B2.T
    s = np.linalg.svd(M, compute_uv=False)
    # Principal angles: theta_i = arccos(clip(s_i, 0, 1))
    s_clamped = np.clip(s, 0.0, 1.0)
    thetas = np.arccos(s_clamped)
    dist = float(np.sqrt(np.sum(thetas ** 2)))
    return dist, thetas


def distance_to_training_support(B_U: np.ndarray, rank: int) -> float:
    """Compute min Grassmann geodesic distance from B_U to all coordinate subspaces in V_coord^(rank)."""
    # Coordinate subspaces are spanned by rank standard basis vectors e_i
    min_dist = float("inf")
    # All subsets of size `rank` from {0, ..., 7}
    for indices in itertools.combinations(range(8), rank):
        B_coord = np.zeros((rank, 8), dtype=np.float64)
        for row_idx, col_idx in enumerate(indices):
            B_coord[row_idx, col_idx] = 1.0
        d, _ = grassmann_geodesic_distance(B_U, B_coord)
        if d < min_dist:
            min_dist = d
    return min_dist


def main():
    parser = argparse.ArgumentParser(description="Held-Out Grassmann Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_subspaces_per_rank", type=int, default=20)
    parser.add_argument("--n_remixes", type=int, default=10)
    parser.add_argument("--n_invariance_patients", type=int, default=50)
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

    logger.info("=== 2. Loading Precomputed FastResponseComputer and Patient Beats ===")
    computer = FastResponseComputer(RESPONSE_FIT, device)
    qc_frame = pd.read_csv(VAL_CACHE / "qc.csv").query("eligible").reset_index(drop=True)
    N = len(qc_frame)
    logger.info(f"Loading {N} validation beat records from {VAL_CACHE}...")

    def _load_beat(idx):
        p = VAL_CACHE / qc_frame.iloc[idx].artifact
        with np.load(p) as item:
            b = np.asarray(item["beats"], dtype=np.float32) * computer.std + computer.mean
            lbl = np.asarray(item["labels"], dtype=np.float32)
            return b, lbl

    with ThreadPoolExecutor(max_workers=16) as ex:
        beat_data = list(ex.map(_load_beat, range(N)))

    all_beats = [bd[0] for bd in beat_data]
    labels_all = np.stack([bd[1] for bd in beat_data], axis=0)
    logger.info(f"Loaded all {N} beat records successfully.")

    # Ranks to evaluate: r in {1, 2, 3, 4, 6}
    ranks = [1, 2, 3, 4, 6]
    total_subspaces = len(ranks) * args.n_subspaces_per_rank
    logger.info(f"Sampling {args.n_subspaces_per_rank} subspaces per rank {ranks} (Total: {total_subspaces} subspaces)...")

    # Generate and freeze test subspaces
    subspace_manifest = []
    for r in ranks:
        for s_idx in range(args.n_subspaces_per_rank):
            B_U = sample_grassmann_subspace(rng, rank=r)
            d_train = distance_to_training_support(B_U, rank=r)
            subspace_manifest.append({
                "subspace_id": f"rank_{r}_sub_{s_idx:02d}",
                "rank": r,
                "subspace_index": s_idx,
                "d_train": float(d_train),
                "B_U": B_U,
            })

    logger.info(f"Generated {len(subspace_manifest)} frozen Grassmann subspaces.")

    # Storage for invariance audit
    invariance_records = []
    # Select subset of patients for invariance check
    sample_beats_inv = all_beats[:args.n_invariance_patients]

    # Storage for evaluation results
    subspace_eval_results = []

    t_start = time.time()
    for idx, sub in enumerate(subspace_manifest):
        sub_id = sub["subspace_id"]
        r = sub["rank"]
        B_U = sub["B_U"]
        d_train = sub["d_train"]

        t_sub0 = time.time()
        # Compute canonical frame B_Q for this subspace
        P_U, _ = observable_projector(B_U, tol=1e-7)
        B_Q = canonical_frame(P_U, rank=r, tol=1e-7)

        # 1. Evaluate Quotient Invariance on Sample Patients (K remixes + redundant)
        inv_ops_list = []
        inv_labels = []
        
        # Test K random invertible remixes R_k B_U
        for k in range(args.n_remixes):
            A = rng.normal(size=(r, r))
            U_svd, _, Vt_svd = np.linalg.svd(A)
            s_conditioned = np.linspace(1.5, 0.8, r)
            R = U_svd @ np.diag(s_conditioned) @ Vt_svd
            Q_remix = R @ B_U
            row_norms = np.linalg.norm(Q_remix, axis=1, keepdims=True)
            Q_remix = Q_remix / np.maximum(row_norms, 1e-12)
            inv_ops_list.append(Q_remix)
            inv_labels.append(f"remix_{k}")

        # Add 1 redundant description: [B_U; A B_U]
        if r <= 6:
            A_red = rng.normal(size=(2, r))
            Q_red = np.vstack([B_U, A_red @ B_U])
            Q_red = Q_red / np.maximum(np.linalg.norm(Q_red, axis=1, keepdims=True), 1e-12)
            inv_ops_list.append(Q_red)
            inv_labels.append("redundant")

        # Baseline predictions on sample patients
        resps_B_U_inv = np.stack([computer.compute_record_responses(b, B_U) for b in sample_beats_inv], axis=0)
        resps_B_Q_inv = np.stack([computer.compute_record_responses(b, B_Q) for b in sample_beats_inv], axis=0)
        l_orig_base = predict_model_batched(model_basis, B_U, resps_B_U_inv, device)
        l_cf_base = predict_model_batched(model_basis, B_Q, resps_B_Q_inv, device)

        cf_diffs = []
        orig_diffs = []

        for Q_alt in inv_ops_list:
            # Orig on alternative description
            resps_alt_orig = np.stack([computer.compute_record_responses(b, Q_alt) for b in sample_beats_inv], axis=0)
            l_orig_alt = predict_model_batched(model_basis, Q_alt, resps_alt_orig, device)
            orig_diffs.append(float(np.max(np.abs(l_orig_alt - l_orig_base))))

            # CF on alternative description: canonicalizes to B_Q_alt == B_Q
            P_alt, r_alt = observable_projector(Q_alt, tol=1e-7)
            B_Q_alt = canonical_frame(P_alt, rank=r_alt, tol=1e-7)
            resps_cf_alt = np.stack([computer.compute_record_responses(b, B_Q_alt) for b in sample_beats_inv], axis=0)
            l_cf_alt = predict_model_batched(model_basis, B_Q_alt, resps_cf_alt, device)
            cf_diffs.append(float(np.max(np.abs(l_cf_alt - l_cf_base))))

        v_cf = float(np.max(cf_diffs))
        v_orig = float(np.max(orig_diffs))

        invariance_records.append({
            "subspace_id": sub_id,
            "rank": r,
            "d_train": d_train,
            "V_cf": v_cf,
            "V_orig": v_orig,
        })

        # 2. Evaluate All 2,173 Patients on 4 Factorial Cells
        # Compute combined responses for B_U and B_Q in a single pass per record
        q_both = np.vstack([B_U, B_Q])
        resps_both = [computer.compute_record_responses(b, q_both) for b in all_beats]
        resps_both = np.stack(resps_both, axis=0)
        resps_B_U_all = resps_both[:, :r]
        resps_B_Q_all = resps_both[:, r:]

        # Cell (1,1): Orig-Basis
        l_1_1 = predict_model_batched(model_basis, B_U, resps_B_U_all, device)
        m_1_1 = compute_metrics(labels_all, torch.sigmoid(torch.tensor(l_1_1)).numpy())

        # Cell (1,2): CF-Basis
        l_1_2 = predict_model_batched(model_basis, B_Q, resps_B_Q_all, device)
        m_1_2 = compute_metrics(labels_all, torch.sigmoid(torch.tensor(l_1_2)).numpy())

        # Cell (2,1): Orig-Span
        l_2_1 = predict_model_batched(model_span, B_U, resps_B_U_all, device)
        m_2_1 = compute_metrics(labels_all, torch.sigmoid(torch.tensor(l_2_1)).numpy())

        # Cell (2,2): CF-Span
        l_2_2 = predict_model_batched(model_span, B_Q, resps_B_Q_all, device)
        m_2_2 = compute_metrics(labels_all, torch.sigmoid(torch.tensor(l_2_2)).numpy())

        delta_span = m_2_2["macro_auroc"] - m_1_2["macro_auroc"]
        delta_span_orig = m_2_1["macro_auroc"] - m_1_1["macro_auroc"]

        sub_res = {
            "subspace_id": sub_id,
            "rank": r,
            "d_train": d_train,
            "cell_1_1_orig_basis_auroc": m_1_1["macro_auroc"],
            "cell_1_2_cf_basis_auroc": m_1_2["macro_auroc"],
            "cell_2_1_orig_span_auroc": m_2_1["macro_auroc"],
            "cell_2_2_cf_span_auroc": m_2_2["macro_auroc"],
            "delta_span_cf": delta_span,
            "delta_span_orig": delta_span_orig,
            "V_cf": v_cf,
            "V_orig": v_orig,
        }
        subspace_eval_results.append(sub_res)

        elapsed = time.time() - t_sub0
        logger.info(
            f"[{idx+1:03d}/{total_subspaces}] {sub_id} (d_train={d_train:.3f}) | "
            f"V_orig={v_orig:.4f}, V_cf={v_cf:.2e} | "
            f"Orig-Basis: {m_1_1['macro_auroc']:.4f} -> Orig-Span: {m_2_1['macro_auroc']:.4f} | "
            f"CF-Basis: {m_1_2['macro_auroc']:.4f} -> CF-Span: {m_2_2['macro_auroc']:.4f} (Delta={delta_span:+.4f}) | "
            f"{elapsed:.1f}s"
        )

    # 3. Aggregate Analysis
    df = pd.DataFrame(subspace_eval_results)

    # Summary by rank
    rank_summary = {}
    for r in ranks:
        sub_df = df[df["rank"] == r]
        rank_summary[f"rank_{r}"] = {
            "n_subspaces": len(sub_df),
            "mean_d_train": float(sub_df["d_train"].mean()),
            "std_d_train": float(sub_df["d_train"].std()),
            "cell_1_1_orig_basis": float(sub_df["cell_1_1_orig_basis_auroc"].mean()),
            "cell_1_2_cf_basis": float(sub_df["cell_1_2_cf_basis_auroc"].mean()),
            "cell_2_1_orig_span": float(sub_df["cell_2_1_orig_span_auroc"].mean()),
            "cell_2_2_cf_span": float(sub_df["cell_2_2_cf_span_auroc"].mean()),
            "delta_span_cf": float(sub_df["delta_span_cf"].mean()),
            "mean_V_cf": float(sub_df["V_cf"].mean()),
            "max_V_cf": float(sub_df["V_cf"].max()),
            "mean_V_orig": float(sub_df["V_orig"].mean()),
            "max_V_orig": float(sub_df["V_orig"].max()),
        }

    # Correlation between distance to training coordinate support and span-support gain
    all_d = df["d_train"].to_numpy()
    all_delta = df["delta_span_cf"].to_numpy()
    corr_pearson = float(np.corrcoef(all_d, all_delta)[0, 1])
    # Spearman rank correlation
    rank_d = pd.Series(all_d).rank().to_numpy()
    rank_delta = pd.Series(all_delta).rank().to_numpy()
    corr_spearman = float(np.corrcoef(rank_d, rank_delta)[0, 1])

    overall_summary = {
        "total_subspaces": total_subspaces,
        "ranks": ranks,
        "n_subspaces_per_rank": args.n_subspaces_per_rank,
        "overall_mean_d_train": float(df["d_train"].mean()),
        "overall_cell_1_1_orig_basis": float(df["cell_1_1_orig_basis_auroc"].mean()),
        "overall_cell_1_2_cf_basis": float(df["cell_1_2_cf_basis_auroc"].mean()),
        "overall_cell_2_1_orig_span": float(df["cell_2_1_orig_span_auroc"].mean()),
        "overall_cell_2_2_cf_span": float(df["cell_2_2_cf_span_auroc"].mean()),
        "overall_delta_span_cf": float(df["delta_span_cf"].mean()),
        "correlation_d_train_vs_delta_span_pearson": corr_pearson,
        "correlation_d_train_vs_delta_span_spearman": corr_spearman,
        "invariance_cf_mean_max_diff": float(df["V_cf"].mean()),
        "invariance_cf_worst_case_diff": float(df["V_cf"].max()),
        "invariance_orig_mean_max_diff": float(df["V_orig"].mean()),
        "invariance_orig_worst_case_diff": float(df["V_orig"].max()),
        "by_rank": rank_summary,
    }

    logger.info("=== HELD-OUT GRASSMANN BENCHMARK COMPLETE ===")
    logger.info(f"Total time: {time.time() - t_start:.1f}s")
    logger.info(f"Correlation (d_train vs Delta_span): Pearson r = {corr_pearson:.4f}, Spearman rho = {corr_spearman:.4f}")
    logger.info(f"Invariance: V_cf max = {df['V_cf'].max():.2e} vs V_orig max = {df['V_orig'].max():.4f}")

    for r_key, r_data in rank_summary.items():
        logger.info(
            f"[{r_key}] d_train={r_data['mean_d_train']:.3f} | "
            f"Orig-Basis: {r_data['cell_1_1_orig_basis']:.4f} -> CF-Span: {r_data['cell_2_2_cf_span']:.4f} | "
            f"Delta_Span: {r_data['delta_span_cf']:+.4f} | "
            f"V_orig={r_data['mean_V_orig']:.4f}, V_cf={r_data['mean_V_cf']:.2e}"
        )

    # Save outputs
    # 1. Full JSON
    out_json = OUTPUT_DIR / "heldout_grassmann_benchmark.json"
    with open(out_json, "w") as f:
        json.dump({
            "overall_summary": overall_summary,
            "subspace_eval_results": subspace_eval_results,
        }, f, indent=2)
    logger.info(f"Saved benchmark results to {out_json}")

    # 2. Frozen Subspace Manifest
    manifest_clean = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in item.items()} for item in subspace_manifest]
    with open(OUTPUT_DIR / "frozen_grassmann_subspaces.json", "w") as f:
        json.dump(manifest_clean, f, indent=2)


if __name__ == "__main__":
    main()
