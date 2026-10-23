#!/usr/bin/env python3
"""Rigorous Rotation Sensitivity Audit for D1 ECG-Only SetOperator.

Tests geometric invariance properties on identical patient cohort under:
A. Identity U = I (numerical identity test)
B. Row permutation (set permutation invariance test)
C. Individual row sign flips (projective/sign invariance test)
D. Genuine random orthogonal mixing U in O(r) (continuous basis invariance test)
E. Numerical determinism / repetition test
F. Genuinely different subspace (cross-subspace comparison control)

Reports:
- Mean, median, and 95th percentile per-patient probability shifts
- AUROC and AUPRC before and after rotation
- Absolute AUROC shift
- Subspace projector distance ||Q.T @ Q - Q_rot.T @ Q_rot||_F
- Operator orthonormality error
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"
MODELS_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/models"

for p in [SRC_DIR, BENCHMARK_DIR, MODELS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from setoperator_echonext import build_setoperator_echonext
from repecg.ablation_setoperator.ablation_models import AblationSetOperatorEchoNext
from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("d1_rotation_audit")

DATA_DIR = Path("/home/mithunmanivannan/data/echonext")
CANONICAL_8_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]
DEV_BANK_PATH = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/echonext_dev_operator_bank.npz")
DEFAULT_D1_CKPT = Path("/tmp/d1_best_model.pt")


def evaluate_on_operator(
    model: nn.Module,
    operators: np.ndarray,
    extractor: EchoNextPhaseKMEExtractor,
    val_beats: list[torch.Tensor],
    val_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, float, float]:
    model.eval()
    k = len(operators)
    ops_t = torch.as_tensor(operators, dtype=torch.float32, device=device)
    ops_batch = ops_t.unsqueeze(0)

    all_probs = []
    n = len(val_beats)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_sz = min(batch_size, n - i)
            batch_resps = []
            for j in range(b_sz):
                beats = val_beats[i + j]
                resps = extractor.compute_phase_kme(beats, torch.from_numpy(operators))
                batch_resps.append(resps)
            t_resps = torch.stack(batch_resps).to(device)
            t_ops = ops_batch.expand(b_sz, -1, -1)
            # D1 is ECG-only (tabular_mode == none), tabular passed as dummy zeros
            dummy_tab = torch.zeros((b_sz, 7), dtype=torch.float32, device=device)
            out = model(operators=t_ops, responses=t_resps, tabular=dummy_tab, mask=None)
            probs = torch.sigmoid(out["logits"].squeeze(-1)).cpu().numpy()
            all_probs.extend(probs.tolist())

    y_prob = np.array(all_probs)
    y_true = val_labels
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    return y_prob, auroc, auprc


def main():
    parser = argparse.ArgumentParser(description="Audit D1 Rotation Invariance")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_D1_CKPT))
    parser.add_argument("--cohort-size", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--subspace-idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Loading D1 checkpoint from %s on device %s...", args.ckpt, device)
    chk = torch.load(args.ckpt, map_location="cpu")
    model = AblationSetOperatorEchoNext(
        operator_mode=chk.get("operator_mode", "continuous"),
        geometry_mode=chk.get("geometry_mode", "baseline"),
        phase_mode=chk.get("phase_mode", "standard_circular"),
        set_encoder_mode=chk.get("set_encoder_mode", "standard_transformer"),
        pooling_mode=chk.get("pooling_mode", "learned_query"),
        tabular_mode="none",
        classes=1,
    ).to(device)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    logger.info("Loading validation cohort (size %d)...", args.cohort_size)
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    val_mask = (df["split"] == "val").values
    y_val = df.loc[val_mask, "shd_moderate_or_greater_flag"].values[:args.cohort_size]

    cache_file = Path("/tmp/val_beats_cached.pt")
    if not cache_file.exists():
        cache_file = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/cache/val_beats_cached.pt")
    logger.info("Loading cached beats from %s...", cache_file)
    beats_all = torch.load(cache_file, map_location="cpu")
    val_beats = beats_all[:args.cohort_size]

    extractor = EchoNextPhaseKMEExtractor(fit_path=DEFAULT_FIT_PATH, device="cpu")

    # Load starting subspace from dev bank
    dev_bank = np.load(DEV_BANK_PATH)
    r = args.rank
    subspaces = dev_bank[f"rank_{r}"]
    Q_orig = subspaces[args.subspace_idx].astype(np.float32)  # shape (r, 8)
    P_orig = Q_orig.T @ Q_orig  # (8, 8) subspace projector

    ortho_err_orig = float(np.max(np.abs(Q_orig @ Q_orig.T - np.eye(r))))
    logger.info("Starting Subspace: Rank %d, shape %s, Orthonormality error: %.2e", r, Q_orig.shape, ortho_err_orig)

    # 0. Baseline evaluation on original Q
    t0 = time.time()
    p_orig, auc_orig, prc_orig = evaluate_on_operator(model, Q_orig, extractor, val_beats, y_val, device)
    logger.info("Baseline Q: AUROC = %.4f | AUPRC = %.4f (computed in %.1fs)", auc_orig, prc_orig, time.time() - t0)

    rng = np.random.default_rng(args.seed)
    results = {
        "baseline": {
            "rank": r,
            "subspace_idx": args.subspace_idx,
            "cohort_size": args.cohort_size,
            "auroc": auc_orig,
            "auprc": prc_orig,
            "orthonormality_error": ortho_err_orig,
        },
        "tests": {},
    }

    def summarize_shift(name: str, p_test: np.ndarray, auc_test: float, prc_test: float, Q_test: np.ndarray, U: np.ndarray):
        diffs = np.abs(p_orig - p_test)
        mean_diff = float(np.mean(diffs))
        median_diff = float(np.median(diffs))
        p95_diff = float(np.percentile(diffs, 95))
        max_diff = float(np.max(diffs))
        delta_auc = float(auc_test - auc_orig)
        delta_prc = float(prc_test - prc_orig)

        # Orthonormality error of Q_test
        ortho_err = float(np.max(np.abs(Q_test @ Q_test.T - np.eye(len(Q_test)))))
        # Subspace projector distance ||Q_orig.T @ Q_orig - Q_test.T @ Q_test||_F
        P_test = Q_test.T @ Q_test
        proj_dist = float(np.linalg.norm(P_orig - P_test, ord="fro"))

        logger.info(
            "[%s] MeanShift=%.4f (med=%.4f, p95=%.4f, max=%.4f) | AUROC=%.4f (Δ=%+.4f) | AUPRC=%.4f (Δ=%+.4f) | ProjDist=%.2e",
            name, mean_diff, median_diff, p95_diff, max_diff, auc_test, delta_auc, prc_test, delta_prc, proj_dist
        )
        return {
            "mean_abs_shift": mean_diff,
            "median_abs_shift": median_diff,
            "p95_abs_shift": p95_diff,
            "max_abs_shift": max_diff,
            "auroc": auc_test,
            "delta_auroc": delta_auc,
            "abs_delta_auroc": abs(delta_auc),
            "auprc": prc_test,
            "delta_auprc": delta_prc,
            "orthonormality_error": ortho_err,
            "subspace_projector_distance": proj_dist,
        }

    # Test A: Identity U = I
    logger.info("--- Test A: Identity U = I (Numerical Identity) ---")
    U_id = np.eye(r, dtype=np.float32)
    Q_id = (U_id @ Q_orig).astype(np.float32)
    p_id, auc_id, prc_id = evaluate_on_operator(model, Q_id, extractor, val_beats, y_val, device)
    results["tests"]["A_identity"] = summarize_shift("Test A (Identity)", p_id, auc_id, prc_id, Q_id, U_id)

    # Test B: Row Permutation (Set Permutation Invariance)
    logger.info("--- Test B: Row Permutation (Token Reordering) ---")
    perm = np.array([2, 0, 1] if r == 3 else list(reversed(range(r))))
    U_perm = np.eye(r, dtype=np.float32)[perm]
    Q_perm = (U_perm @ Q_orig).astype(np.float32)
    p_perm, auc_perm, prc_perm = evaluate_on_operator(model, Q_perm, extractor, val_beats, y_val, device)
    results["tests"]["B_row_permutation"] = summarize_shift("Test B (Row Permutation)", p_perm, auc_perm, prc_perm, Q_perm, U_perm)

    # Test C: Row Sign Flips (q_0 -> -q_0, q_2 -> -q_2)
    logger.info("--- Test C: Individual Row Sign Flips ---")
    signs = np.ones(r, dtype=np.float32)
    signs[0] = -1.0
    if r > 2:
        signs[2] = -1.0
    U_sign = np.diag(signs)
    Q_sign = (U_sign @ Q_orig).astype(np.float32)
    p_sign, auc_sign, prc_sign = evaluate_on_operator(model, Q_sign, extractor, val_beats, y_val, device)
    results["tests"]["C_sign_flips"] = summarize_shift("Test C (Sign Flips)", p_sign, auc_sign, prc_sign, Q_sign, U_sign)

    # Test D: Genuine Random Orthogonal Mixing (U in SO(r))
    logger.info("--- Test D: Genuine Random Orthogonal Mixing (Continuous Basis Rotation) ---")
    U_rot = ortho_group.rvs(r, random_state=rng).astype(np.float32)
    Q_rot = (U_rot @ Q_orig).astype(np.float32)
    p_rot, auc_rot, prc_rot = evaluate_on_operator(model, Q_rot, extractor, val_beats, y_val, device)
    results["tests"]["D_random_orthogonal_mixing"] = summarize_shift("Test D (Random Orthogonal Mixing)", p_rot, auc_rot, prc_rot, Q_rot, U_rot)

    # Test E: Repeatability / Determinism of Test D
    logger.info("--- Test E: Numerical Determinism (Repeat Test D) ---")
    p_rot2, auc_rot2, prc_rot2 = evaluate_on_operator(model, Q_rot, extractor, val_beats, y_val, device)
    repeat_diff = float(np.max(np.abs(p_rot - p_rot2)))
    logger.info("Test E Determinism: Max prediction discrepancy between identical runs = %.2e", repeat_diff)
    results["tests"]["E_determinism"] = {
        "max_discrepancy": repeat_diff,
        "is_deterministic": bool(repeat_diff < 1e-6),
    }

    # Test F: Genuinely Different Subspace (Cross-subspace control)
    logger.info("--- Test F: Genuinely Different Subspace (Cross-subspace control) ---")
    Q_other = subspaces[(args.subspace_idx + 1) % len(subspaces)].astype(np.float32)
    P_other = Q_other.T @ Q_other
    proj_dist_other = float(np.linalg.norm(P_orig - P_other, ord="fro"))
    p_other, auc_other, prc_other = evaluate_on_operator(model, Q_other, extractor, val_beats, y_val, device)
    results["tests"]["F_different_subspace_control"] = summarize_shift(
        f"Test F (Subspace {args.subspace_idx+1})", p_other, auc_other, prc_other, Q_other, np.zeros((r, r), dtype=np.float32)
    )

    out_file = DEV_ROOT / "scripts/d1_rotation_audit_report.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Audit complete! Report saved to %s", out_file)


if __name__ == "__main__":
    main()
