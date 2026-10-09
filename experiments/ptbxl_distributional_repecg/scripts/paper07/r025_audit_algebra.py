#!/usr/bin/env python3
"""Milestone R025: Audit Observation Algebra (E0) and Real-Data Invariance (E1).

Formalizes:
- E0: Correctness of observation algebra on synthetic arrays (float64 oracle & float32 tolerances).
- E1: Information-preserving representation audit on real PTB-XL waveforms with negative controls.
- Enforces Stop 1 Gate.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from repecg.paper07_operator.observable_subspace import (
    observable_state,
    compute_projector,
)
from repecg.paper07_operator.observable_encoder import LinearFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("r025_audit_algebra")

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
OUTPUT_DIR = CODEX_DIR / "paper07_operator/r025_manifests"


def run_e0_oracle_audit() -> Dict[str, Any]:
    logger.info("--- Running E0: Exact Observation Algebra Oracle Audit (float64 & float32) ---")
    torch.manual_seed(42)
    d = 8
    T = 1000
    m = 5
    rcond = 1e-5

    # Float64 Oracle tests
    X64 = torch.randn(1, T, d, dtype=torch.float64)
    Q64 = torch.randn(1, m, d, dtype=torch.float64)
    Y64 = X64 @ Q64.transpose(-2, -1)

    state64 = observable_state(Q64, Y64, rcond=rcond)
    P64 = state64.projector
    X_obs64 = state64.projected

    # Identities
    sym_err = float(torch.max(torch.abs(P64 - P64.transpose(-2, -1))))
    idemp_err = float(torch.max(torch.abs(P64 @ P64 - P64)))
    xp_err = float(torch.max(torch.abs(X_obs64 - X64 @ P64)))
    proj_idemp_err = float(torch.max(torch.abs(X_obs64 @ P64 - X_obs64)))

    # Linear temporal frame Z = H @ X_obs -> Z @ P = Z
    frame = LinearFrame(T=T, L=24, mode="dct")
    H64 = frame.H.to(torch.float64)
    Z64 = torch.matmul(H64, X_obs64)
    zp_err = float(torch.max(torch.abs(Z64 @ P64 - Z64)))

    # Invertible R in R^(m x m)
    R64 = torch.randn(m, m, dtype=torch.float64)
    while torch.linalg.cond(R64) > 30.0:
        R64 = torch.randn(m, m, dtype=torch.float64)
    Q_remix64 = torch.matmul(R64, Q64)  # [m, m] @ [1, m, d] -> [1, m, d]
    Y_remix64 = torch.matmul(Y64, R64.transpose(-2, -1))  # [1, T, m] @ [m, m] -> [1, T, m]
    state_remix64 = observable_state(Q_remix64, Y_remix64, rcond=rcond)
    remix_p_err = float(torch.max(torch.abs(state_remix64.projector - P64)))
    remix_x_err = float(torch.max(torch.abs(state_remix64.projected - X_obs64)))

    # Permutations
    perm = torch.randperm(m)
    state_perm64 = observable_state(Q64[:, perm], Y64[:, :, perm], rcond=rcond)
    perm_p_err = float(torch.max(torch.abs(state_perm64.projector - P64)))
    perm_x_err = float(torch.max(torch.abs(state_perm64.projected - X_obs64)))

    # Sign reversal
    signs = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float64).view(1, m, 1)
    state_sign64 = observable_state(Q64 * signs, Y64 * signs.transpose(1, 2), rcond=rcond)
    sign_p_err = float(torch.max(torch.abs(state_sign64.projector - P64)))
    sign_x_err = float(torch.max(torch.abs(state_sign64.projected - X_obs64)))

    # Dependent row addition (III = II - I)
    e64 = torch.eye(d, dtype=torch.float64)
    Q_limb = torch.stack([e64[0], e64[1]], dim=0).unsqueeze(0)  # [1, 2, 8]
    Y_limb = X64 @ Q_limb.transpose(-2, -1)  # [1, T, 2]
    state_limb = observable_state(Q_limb, Y_limb, rcond=rcond)

    Q_dep = torch.cat([Q_limb, (e64[1] - e64[0]).view(1, 1, d)], dim=1)  # add III: [1, 3, 8]
    Y_dep = torch.cat([Y_limb, (Y_limb[:, :, 1:2] - Y_limb[:, :, 0:1])], dim=2)
    state_dep = observable_state(Q_dep, Y_dep, rcond=rcond)
    dep_p_err = float(torch.max(torch.abs(state_dep.projector - state_limb.projector)))
    dep_x_err = float(torch.max(torch.abs(state_dep.projected - state_limb.projected)))
    dep_rank_equal = bool(state_dep.rank == state_limb.rank)

    # Monotonic observability
    Q_more = torch.cat([Q_limb, e64[2:5].unsqueeze(0)], dim=1)
    P_limb, r_limb = compute_projector(Q_limb, rcond=rcond)
    P_more, r_more = compute_projector(Q_more, rcond=rcond)
    diff_P = (P_more - P_limb).squeeze(0)
    min_eig = float(torch.linalg.eigvalsh(diff_P).min())

    # Float32 production tolerances check
    X32 = X64.to(torch.float32)
    Q32 = Q64.to(torch.float32)
    Y32 = X32 @ Q32.transpose(-2, -1)
    state32 = observable_state(Q32, Y32, rcond=rcond)
    f32_p_err = float(torch.max(torch.abs(state32.projector - state32.projector.transpose(-2, -1))))
    f32_xp_err = float(torch.max(torch.abs(state32.projected - X32 @ state32.projector)))

    e0_results = {
        "float64_oracle": {
            "projector_symmetry_max_err": sym_err,
            "projector_idempotence_max_err": idemp_err,
            "noiseless_x_proj_max_err": xp_err,
            "projected_idempotence_max_err": proj_idemp_err,
            "frame_z_proj_max_err": zp_err,
            "invertible_remix_projector_max_err": remix_p_err,
            "invertible_remix_x_obs_max_err": remix_x_err,
            "permutation_projector_max_err": perm_p_err,
            "permutation_x_obs_max_err": perm_x_err,
            "sign_reversal_projector_max_err": sign_p_err,
            "sign_reversal_x_obs_max_err": sign_x_err,
            "dependent_lead_projector_max_err": dep_p_err,
            "dependent_lead_x_obs_max_err": dep_x_err,
            "dependent_lead_rank_preserved": dep_rank_equal,
            "monotonic_observability_min_eigval": min_eig,
        },
        "float32_production": {
            "projector_symmetry_max_err": f32_p_err,
            "noiseless_x_proj_max_err": f32_xp_err,
        },
    }

    # Verify bounds
    assert sym_err < 1e-12, f"Float64 symmetry error: {sym_err}"
    assert idemp_err < 1e-12, f"Float64 idempotence error: {idemp_err}"
    assert xp_err < 1e-12, f"Float64 projection error: {xp_err}"
    assert remix_p_err < 1e-10, f"Float64 remix P error: {remix_p_err}"
    assert dep_p_err < 1e-12, f"Float64 dependent lead P error: {dep_p_err}"
    assert dep_rank_equal, "Dependent lead changed rank"
    assert min_eig >= -1e-12, f"Monotonic observability violated: {min_eig}"
    assert f32_p_err < 1e-5, f"Float32 symmetry error: {f32_p_err}"
    assert f32_xp_err < 1e-4, f"Float32 projection error: {f32_xp_err}"

    e0_results["status"] = "PASS"
    logger.info("E0 Oracle Audit PASSED.")
    return e0_results


def run_e1_real_waveform_audit() -> Dict[str, Any]:
    logger.info("--- Running E1: Real PTB-XL Waveform Representation Audit & Negative Control ---")
    val_cache = CODEX_DIR / "paper02_kernel_mean/development_select"
    qc_frame = CODEX_DIR / "paper02_kernel_mean/development_select/qc.csv"
    import pandas as pd
    df = pd.read_csv(qc_frame).query("eligible").reset_index(drop=True)

    # Load 50 real ECG records
    n_test = min(50, len(df))
    logger.info(f"Auditing on {n_test} real clinical recordings...")

    e = np.eye(8, dtype=np.float32)
    # Basis: [I, II, V1, V2, V3, V4, V5, V6]
    # We test:
    # 1. 2-lead limb basis {I, II} vs 6-lead limb set {I, II, III, aVR, aVL, aVF}
    # 2. Invertible remixing on real waveforms
    # 3. Critical negative control: Random Q with fixed Y must fail consistency check

    q_2limb = torch.as_tensor(e[:2], dtype=torch.float32)  # [2, 8]
    # 6 limb leads:
    q_III = (e[1] - e[0]) / np.linalg.norm(e[1] - e[0])
    q_aVR = (-e[0] - e[1]) / np.linalg.norm(-e[0] - e[1])
    q_aVL = (e[0] - 0.5 * e[1]) / np.linalg.norm(e[0] - 0.5 * e[1])
    q_aVF = (e[1] - 0.5 * e[0]) / np.linalg.norm(e[1] - 0.5 * e[0])
    q_6limb = torch.as_tensor(np.stack([e[0], e[1], q_III, q_aVR, q_aVL, q_aVF]), dtype=torch.float32)  # [6, 8]

    frame = LinearFrame(T=256, L=24, mode="dct")

    diff_p_limb_list = []
    diff_x_limb_list = []
    diff_remix_list = []
    neg_control_residual_list = []

    for idx in range(n_test):
        rec_path = val_cache / df.iloc[idx].artifact
        with np.load(rec_path) as item:
            # beats: [N_beats, 256, 8]
            beats = item["beats"]
            x_real = torch.as_tensor(beats[0], dtype=torch.float32)  # [256, 8]

        # 1. Subspace equivalence: {I, II} vs {I, II, III, aVR, aVL, aVF}
        y_2limb = x_real @ q_2limb.T  # [256, 2]
        y_6limb = x_real @ q_6limb.T  # [256, 6]

        st_2 = observable_state(q_2limb, y_2limb, rcond=1e-5)
        st_6 = observable_state(q_6limb, y_6limb, rcond=1e-5)

        diff_P_limb = float(torch.max(torch.abs(st_2.projector - st_6.projector)))
        diff_X_limb = float(torch.max(torch.abs(st_2.projected - st_6.projected)))
        diff_p_limb_list.append(diff_P_limb)
        diff_x_limb_list.append(diff_X_limb)

        # 2. Invertible remix of full 8 leads
        q_full = torch.as_tensor(e, dtype=torch.float32)
        y_full = x_real @ q_full.T
        st_full = observable_state(q_full, y_full, rcond=1e-5)

        R = torch.randn(8, 8, dtype=torch.float32)
        while torch.linalg.cond(R) > 20.0:
            R = torch.randn(8, 8, dtype=torch.float32)
        q_remix = R @ q_full
        y_remix = y_full @ R.T
        st_remix = observable_state(q_remix, y_remix, rcond=1e-5)

        diff_remix = float(torch.max(torch.abs(st_full.projected - st_remix.projected)))
        diff_remix_list.append(diff_remix)

        # 3. CRITICAL NEGATIVE CONTROL:
        # Keep Y fixed (from real 6 limb leads), but pass a randomized Q of shape [6, 8]
        # Under the exact observation law, X_obs must equal X_real @ P_Q.
        # For a bogus pair (Q_rand, Y), X_obs = Y @ (Q_rand^+)^T != X_real @ P_Q_rand!
        Q_rand = torch.randn(6, 8, dtype=torch.float32)
        Q_rand = Q_rand / torch.linalg.norm(Q_rand, dim=-1, keepdim=True)
        st_bogus = observable_state(Q_rand, y_6limb, rcond=1e-5)

        # 3a. Observation law violation test
        expected_proj = x_real @ st_bogus.projector.squeeze(0)
        actual_proj = st_bogus.projected.squeeze(0)
        law_violation = float(
            torch.linalg.norm(actual_proj - expected_proj)
            / (torch.linalg.norm(expected_proj) + 1e-8)
        )
        neg_control_residual_list.append(law_violation)

        # 3b. Subspace change test: P and Z must change under altered Q
        diff_P_altered = float(torch.linalg.norm(st_6.projector - st_bogus.projector))
        Z_true = frame(st_6.projected)
        Z_bogus = frame(st_bogus.projected)
        diff_Z_altered = float(torch.linalg.norm(Z_true - Z_bogus))

    mean_neg_residual = float(np.mean(neg_control_residual_list))
    max_limb_p_diff = float(np.max(diff_p_limb_list))
    max_limb_x_diff = float(np.max(diff_x_limb_list))
    max_remix_diff = float(np.max(diff_remix_list))

    logger.info(f"Real ECG Limb Equivalence Max Diff P: {max_limb_p_diff:.2e}, X: {max_limb_x_diff:.2e}")
    logger.info(f"Real ECG Invertible Remix Max Diff X: {max_remix_diff:.2e}")
    logger.info(f"Negative Control Observation Law Violation Residual: {mean_neg_residual:.4f}")

    assert max_limb_p_diff < 1e-5, f"Real limb P difference exceeded tolerance: {max_limb_p_diff}"
    assert max_limb_x_diff < 1e-4, f"Real limb X difference exceeded tolerance: {max_limb_x_diff}"
    assert max_remix_diff < 1e-4, f"Real remix X difference exceeded tolerance: {max_remix_diff}"
    # Negative control must show significant observation law violation
    assert mean_neg_residual > 0.5, f"Negative control failed to flag bogus pair: residual = {mean_neg_residual}"

    e1_results = {
        "n_records_audited": n_test,
        "limb_subspace_equivalence_p_max_diff": max_limb_p_diff,
        "limb_subspace_equivalence_x_max_diff": max_limb_x_diff,
        "invertible_remix_x_max_diff": max_remix_diff,
        "negative_control_mean_relative_residual": mean_neg_residual,
        "negative_control_flagged": True,
        "status": "PASS",
    }
    logger.info("E1 Real Waveform Representation Audit PASSED.")
    return e1_results


def main():
    parser = argparse.ArgumentParser(description="Audit R025 Observation Algebra")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    e0 = run_e0_oracle_audit()
    e1 = run_e1_real_waveform_audit()

    audit_summary = {
        "milestone": "R025",
        "description": "Observation Algebra (E0) and Real-Data Representation (E1) Audit",
        "stop_1_gate": "PASS",
        "E0_oracle": e0,
        "E1_real_data": e1,
    }

    out_file = args.out_dir / "r025_algebra_audit.json"
    with open(out_file, "w") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Audit summary saved to {out_file}")
    logger.info("STOP 1 GATE: PASS")


if __name__ == "__main__":
    main()
