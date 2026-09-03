#!/usr/bin/env python3
"""Stage A.2 Preflight: Limb-Lead Algebra Audit and D3 Post-Hoc Algebraic Derivation.

Evaluates:
1. Ground-Truth Dataset Algebra Audit across all N=2,183 PTB-XL validation records.
2. D3 Native vs. D3 Algebraic Derivation comparison.
3. Physics consistency violations before and after.
4. Error propagation sanity check: e_l^derived vs e_II.
5. 10,000 patient-level bootstrap confidence intervals for Delta_i = Derived - Native.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_latents.engineering.models.three_d_theta_reconstruction import (
    LEAD_NAMES,
    ThreeDThetaECGAIM,
)
from scripts.train_3dtheta_ablation import SimplePTBXLDataset

OUTPUT_JSON = ROOT / "results" / "stage_a2_limb_algebra_summary.json"
OUTPUT_MD = ROOT / "STAGE_A2_LIMB_ALGEBRA_REPORT.md"


def bootstrap_ci(delta: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(delta[indices], axis=1)

    point_estimate = float(np.mean(delta))
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    std_err = float(np.std(boot_means))
    p_pos = np.mean(boot_means <= 0.0)
    p_neg = np.mean(boot_means >= 0.0)
    p_val = float(2.0 * min(p_pos, p_neg))

    return {
        "mean": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "std_err": std_err,
        "p_value": min(p_val, 1.0),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Stage A.2 Limb-Lead Algebra Preflight on {device}...")

    val_ds = SimplePTBXLDataset(ROOT / "data" / "ptb_xl" / "tensors" / "val")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    n_patients = len(val_ds)
    print(f"Validation Cohort: N = {n_patients} patients")

    # ==========================================================================
    # 1. GROUND TRUTH DATASET ALGEBRA AUDIT
    # ==========================================================================
    print("\n--- 1. Ground Truth Dataset Algebra Audit ---")
    # Leads: 0=I, 1=II, 2=III, 3=aVR, 4=aVL, 5=aVF
    res_III_all = []
    res_aVR_all = []
    res_aVL_all = []
    res_aVF_all = []

    all_targets = []
    for batch in val_loader:
        # [B, 12, 5000]
        targets = batch.numpy()
        all_targets.append(targets)
        I = targets[:, 0, :]
        II = targets[:, 1, :]
        III = targets[:, 2, :]
        aVR = targets[:, 3, :]
        aVL = targets[:, 4, :]
        aVF = targets[:, 5, :]

        # Residuals in mV
        r_III = np.abs(II - I - III)
        r_aVR = np.abs(aVR + (I + II) / 2.0)
        r_aVL = np.abs(aVL - I + II / 2.0)
        r_aVF = np.abs(aVF - II + I / 2.0)

        res_III_all.append(r_III)
        res_aVR_all.append(r_aVR)
        res_aVL_all.append(r_aVL)
        res_aVF_all.append(r_aVF)

    targets_np = np.concatenate(all_targets, axis=0)  # [N, 12, 5000]
    res_III = np.concatenate(res_III_all, axis=0)
    res_aVR = np.concatenate(res_aVR_all, axis=0)
    res_aVL = np.concatenate(res_aVL_all, axis=0)
    res_aVF = np.concatenate(res_aVF_all, axis=0)

    def stats_res(r: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(r)),
            "median": float(np.median(r)),
            "p95": float(np.percentile(r, 95)),
            "max": float(np.max(r)),
        }

    dataset_stats = {
        "epsilon_III": stats_res(res_III),
        "epsilon_aVR": stats_res(res_aVR),
        "epsilon_aVL": stats_res(res_aVL),
        "epsilon_aVF": stats_res(res_aVF),
    }

    print("PTB-XL Target Residuals (in mV):")
    for k, v in dataset_stats.items():
        print(f"  {k}: mean={v['mean']:.6f} mV, median={v['median']:.6f} mV, p95={v['p95']:.6f} mV, max={v['max']:.6f} mV")

    # Safety check: if residuals are substantial, halt
    max_res = max(v["max"] for v in dataset_stats.values())
    p95_res = max(v["p95"] for v in dataset_stats.values())
    if p95_res > 0.01:  # > 10 microvolts at 95th percentile
        print(f"\n[CRITICAL ERROR] Target limb leads in PTB-XL violate Einthoven/Goldberger identities! p95={p95_res:.4f} mV")
        print("Halting post-hoc analysis until lead definitions/scaling are audited.")
        return
    else:
        print(f"  [PASS] Target dataset strictly adheres to Einthoven/Goldberger identities (p95 = {p95_res*1000:.2f} µV).")

    # ==========================================================================
    # 2. LOAD D3 CHECKPOINT AND GENERATE PREDICTIONS
    # ==========================================================================
    print("\n--- 2. Loading D3 Checkpoint & Forward Inference ---")
    ckpt_path = ROOT / "refine-logs/convergence_10e/runs/D3_theta_mul_l1_s42_l0/best.pt"
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} does not exist.")
        return

    model = ThreeDThetaECGAIM(code_mode="theta", fusion="mul", width=768, encoder_depth=8, decoder_depth=4, heads=12).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()

    native_preds_list = []
    with torch.inference_mode():
        for batch in val_loader:
            x_src = batch[:, 0:1, :].to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                p = model(x_src, obs_lead_idx=0)["y_pred"].float()
            native_preds_list.append(p.cpu().numpy())

    native_preds = np.concatenate(native_preds_list, axis=0)  # [N, 12, 5000]

    # ==========================================================================
    # 3. CONSTRUCT D3-DERIVED PREDICTIONS
    # ==========================================================================
    print("\n--- 3. Constructing D3-Derived Algebraic Predictions ---")
    derived_preds = np.copy(native_preds)

    # Inputs:
    # targets[:, 0, :] is exact observed Lead I
    I_obs = targets_np[:, 0, :]
    II_pred = native_preds[:, 1, :]

    # Derive limb leads algebraically:
    derived_preds[:, 0, :] = I_obs
    derived_preds[:, 1, :] = II_pred
    derived_preds[:, 2, :] = II_pred - I_obs  # III = II - I
    derived_preds[:, 3, :] = -(I_obs + II_pred) / 2.0  # aVR = -(I + II)/2
    derived_preds[:, 4, :] = I_obs - II_pred / 2.0  # aVL = I - II/2
    derived_preds[:, 5, :] = II_pred - I_obs / 2.0  # aVF = II - I/2
    # V1-V6 (indices 6..11) strictly unchanged

    # ==========================================================================
    # 4. SANITY CHECK: e_l^derived vs e_II
    # ==========================================================================
    print("\n--- 4. Mathematical Sanity Check on Error Propagation ---")
    e_II = derived_preds[:, 1, :] - targets_np[:, 1, :]
    e_III_der = derived_preds[:, 2, :] - targets_np[:, 2, :]
    e_aVR_der = derived_preds[:, 3, :] - targets_np[:, 3, :]
    e_aVL_der = derived_preds[:, 4, :] - targets_np[:, 4, :]
    e_aVF_der = derived_preds[:, 5, :] - targets_np[:, 5, :]

    # Note: e_III_der - e_II is mathematically equal to targets_II - targets_I - targets_III = epsilon_III!
    # Therefore, testing (e_III_der - e_II) - epsilon_III verifies exact numerical implementation.
    exact_diff_III = float(np.max(np.abs((e_III_der - e_II) - (targets_np[:, 1, :] - targets_np[:, 0, :] - targets_np[:, 2, :]))))
    exact_diff_aVR = float(np.max(np.abs((e_aVR_der - (-0.5 * e_II)) - (targets_np[:, 3, :] + (targets_np[:, 0, :] + targets_np[:, 1, :]) / 2.0))))
    exact_diff_aVL = float(np.max(np.abs((e_aVL_der - (-0.5 * e_II)) - (targets_np[:, 0, :] - targets_np[:, 1, :] / 2.0 - targets_np[:, 4, :]))))
    exact_diff_aVF = float(np.max(np.abs((e_aVF_der - e_II) - (targets_np[:, 1, :] - targets_np[:, 0, :] / 2.0 - targets_np[:, 5, :]))))

    diff_III_raw = float(np.max(np.abs(e_III_der - e_II)))
    diff_aVR_raw = float(np.max(np.abs(e_aVR_der - (-0.5 * e_II))))
    diff_aVL_raw = float(np.max(np.abs(e_aVL_der - (-0.5 * e_II))))
    diff_aVF_raw = float(np.max(np.abs(e_aVF_der - e_II)))

    p99_III = float(np.percentile(np.abs(e_III_der - e_II), 99))
    p99_aVR = float(np.percentile(np.abs(e_aVR_der - (-0.5 * e_II)), 99))
    p99_aVL = float(np.percentile(np.abs(e_aVL_der - (-0.5 * e_II)), 99))
    p99_aVF = float(np.percentile(np.abs(e_aVF_der - e_II), 99))

    print(f"  Exact Implementation Residual |(e_der - e_formula) - epsilon_target|:")
    print(f"    III: {exact_diff_III:.10f} mV | aVR: {exact_diff_aVR:.10f} mV | aVL: {exact_diff_aVL:.10f} mV | aVF: {exact_diff_aVF:.10f} mV")
    print(f"  Raw |e_l^der - e_II| (p99 / max in mV):")
    print(f"    III: p99={p99_III*1000:.2f} µV, max={diff_III_raw:.4f} mV")
    print(f"    aVR: p99={p99_aVR*1000:.2f} µV, max={diff_aVR_raw:.4f} mV")
    print(f"    aVL: p99={p99_aVL*1000:.2f} µV, max={diff_aVL_raw:.4f} mV")
    print(f"    aVF: p99={p99_aVF*1000:.2f} µV, max={diff_aVF_raw:.4f} mV")
    sanity_passed = max(exact_diff_III, exact_diff_aVR, exact_diff_aVL, exact_diff_aVF) < 1e-6
    print(f"  Sanity Check Verdict: {'PASS (Implementation is exact to machine precision)' if sanity_passed else 'FAIL'}")

    # ==========================================================================
    # 5. PHYSICS INCONSISTENCY EVALUATION (NATIVE VS DERIVED)
    # ==========================================================================
    print("\n--- 5. Physical Inconsistency Violations ---")
    def compute_violations(p: np.ndarray) -> dict[str, float]:
        v_III = np.abs(p[:, 1, :] - p[:, 0, :] - p[:, 2, :])
        v_aVR = np.abs(p[:, 3, :] + (p[:, 0, :] + p[:, 1, :]) / 2.0)
        v_aVL = np.abs(p[:, 4, :] - p[:, 0, :] + p[:, 1, :] / 2.0)
        v_aVF = np.abs(p[:, 5, :] - p[:, 1, :] + p[:, 0, :] / 2.0)
        return {
            "mean_III_violation_uV": float(np.mean(v_III) * 1000),
            "mean_aVR_violation_uV": float(np.mean(v_aVR) * 1000),
            "mean_aVL_violation_uV": float(np.mean(v_aVL) * 1000),
            "mean_aVF_violation_uV": float(np.mean(v_aVF) * 1000),
        }

    viol_native = compute_violations(native_preds)
    viol_derived = compute_violations(derived_preds)

    print("Native Mean Violations (µV):", viol_native)
    print("Derived Mean Violations (µV):", viol_derived)

    # ==========================================================================
    # 6. HEAD-TO-HEAD RECONSTRUCTION EVALUATION
    # ==========================================================================
    print("\n--- 6. Computing Per-Lead & Aggregate Reconstruction Metrics ---")
    def calc_metrics(preds: np.ndarray, targs: np.ndarray):
        # Center for Pearson
        p_c = preds - preds.mean(axis=-1, keepdims=True)
        t_c = targs - targs.mean(axis=-1, keepdims=True)
        cov = (p_c * t_c).sum(axis=-1)
        std_p = np.sqrt(np.clip((p_c**2).sum(axis=-1), 1e-8, None))
        std_t = np.sqrt(np.clip((t_c**2).sum(axis=-1), 1e-8, None))
        pearsons = np.clip(cov / (std_p * std_t), -1.0, 1.0)  # [N, 12]

        diff = preds - targs
        maes = np.abs(diff).mean(axis=-1)  # [N, 12]
        mses = (diff**2).mean(axis=-1)  # [N, 12]

        return pearsons, maes, mses

    r_nat, mae_nat, mse_nat = calc_metrics(native_preds, targets_np)
    r_der, mae_der, mse_der = calc_metrics(derived_preds, targets_np)

    missing_idx = list(range(1, 12))
    limb_missing_idx = [1, 2, 3, 4, 5]
    chest_idx = [6, 7, 8, 9, 10, 11]

    # Per-patient aggregates
    patient_all_r_nat = r_nat[:, missing_idx].mean(axis=-1)
    patient_all_r_der = r_der[:, missing_idx].mean(axis=-1)

    patient_limb_r_nat = r_nat[:, limb_missing_idx].mean(axis=-1)
    patient_limb_r_der = r_der[:, limb_missing_idx].mean(axis=-1)

    patient_chest_r_nat = r_nat[:, chest_idx].mean(axis=-1)
    patient_chest_r_der = r_der[:, chest_idx].mean(axis=-1)

    patient_l1_nat = mae_nat[:, missing_idx].mean(axis=-1)
    patient_l1_der = mae_der[:, missing_idx].mean(axis=-1)

    patient_mse_nat = mse_nat[:, missing_idx].mean(axis=-1)
    patient_mse_der = mse_der[:, missing_idx].mean(axis=-1)

    # 10,000 bootstrap CIs on patient-level paired differences
    delta_all_r = patient_all_r_der - patient_all_r_nat
    delta_limb_r = patient_limb_r_der - patient_limb_r_nat
    delta_chest_r = patient_chest_r_der - patient_chest_r_nat
    delta_l1 = patient_l1_der - patient_l1_nat

    boot_all_r = bootstrap_ci(delta_all_r, n_boot=10000)
    boot_limb_r = bootstrap_ci(delta_limb_r, n_boot=10000)
    boot_chest_r = bootstrap_ci(delta_chest_r, n_boot=10000)
    boot_l1 = bootstrap_ci(delta_l1, n_boot=10000)

    # Per-lead comparisons table
    lead_rows = []
    for idx, name in enumerate(LEAD_NAMES):
        r_n = float(np.mean(r_nat[:, idx]))
        r_d = float(np.mean(r_der[:, idx]))
        dr = r_d - r_n
        m_n = float(np.mean(mae_nat[:, idx]))
        m_d = float(np.mean(mae_der[:, idx]))
        dm = m_d - m_n
        s_n = float(np.mean(mse_nat[:, idx]))
        s_d = float(np.mean(mse_der[:, idx]))
        lead_rows.append({
            "lead": name,
            "nat_r": r_n,
            "der_r": r_d,
            "delta_r": dr,
            "nat_mae": m_n,
            "der_mae": m_d,
            "delta_mae": dm,
            "nat_mse": s_n,
            "der_mse": s_d,
        })

    # Summary table
    summary_metrics = {
        "mean_all_missing_r": {"nat": float(np.mean(patient_all_r_nat)), "der": float(np.mean(patient_all_r_der)), "boot": boot_all_r},
        "mean_limb_missing_r": {"nat": float(np.mean(patient_limb_r_nat)), "der": float(np.mean(patient_limb_r_der)), "boot": boot_limb_r},
        "mean_chest_r": {"nat": float(np.mean(patient_chest_r_nat)), "der": float(np.mean(patient_chest_r_der)), "boot": boot_chest_r},
        "all_missing_r_p05": {"nat": float(np.percentile(patient_all_r_nat, 5)), "der": float(np.percentile(patient_all_r_der, 5))},
        "all_missing_l1_mV": {"nat": float(np.mean(patient_l1_nat)), "der": float(np.mean(patient_l1_der)), "boot": boot_l1},
        "all_missing_mse": {"nat": float(np.mean(patient_mse_nat)), "der": float(np.mean(patient_mse_der))},
    }

    # Print summary
    print("\n--- Summary Results ---")
    print(f"  All-Missing Pearson:   Native={summary_metrics['mean_all_missing_r']['nat']:.4f} -> Derived={summary_metrics['mean_all_missing_r']['der']:.4f} (Delta={boot_all_r['mean']:+.4f}, CI={boot_all_r['ci_lower']:+.4f} to {boot_all_r['ci_upper']:+.4f}, p={boot_all_r['p_value']:.4f})")
    print(f"  Limb-Missing Pearson:  Native={summary_metrics['mean_limb_missing_r']['nat']:.4f} -> Derived={summary_metrics['mean_limb_missing_r']['der']:.4f} (Delta={boot_limb_r['mean']:+.4f}, CI={boot_limb_r['ci_lower']:+.4f} to {boot_limb_r['ci_upper']:+.4f}, p={boot_limb_r['p_value']:.4f})")
    print(f"  Chest Pearson:         Native={summary_metrics['mean_chest_r']['nat']:.4f} -> Derived={summary_metrics['mean_chest_r']['der']:.4f} (Delta={boot_chest_r['mean']:+.4f})")
    print(f"  All-Missing L1 (mV):   Native={summary_metrics['all_missing_l1_mV']['nat']:.4f} -> Derived={summary_metrics['all_missing_l1_mV']['der']:.4f} (Delta={boot_l1['mean']:+.4f}, CI={boot_l1['ci_lower']:+.4f} to {boot_l1['ci_upper']:+.4f})")

    # ==========================================================================
    # 7. EXPORT ARTIFACTS
    # ==========================================================================
    def to_builtin(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_builtin(v) for v in obj]
        return obj

    full_output = {
        "dataset_algebra_audit": dataset_stats,
        "sanity_check": {
            "exact_diff_III": exact_diff_III,
            "exact_diff_aVR": exact_diff_aVR,
            "exact_diff_aVL": exact_diff_aVL,
            "exact_diff_aVF": exact_diff_aVF,
            "diff_III_raw": diff_III_raw,
            "diff_aVR_raw": diff_aVR_raw,
            "diff_aVL_raw": diff_aVL_raw,
            "diff_aVF_raw": diff_aVF_raw,
            "passed": sanity_passed,
        },
        "physics_violations": {
            "native": viol_native,
            "derived": viol_derived,
        },
        "summary_metrics": summary_metrics,
        "lead_rows": lead_rows,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(to_builtin(full_output), indent=2))
    print(f"\nSaved JSON summary to: {OUTPUT_JSON}")

    # Build Markdown
    md_lines = [
        "# Stage A.2 Preflight: Limb-Lead Algebra & Post-Hoc Derivation Report",
        "",
        f"Evaluation Cohort: PTB-XL held-out validation cohort ($N = {n_patients}$ patients)  ",
        "Evaluated Checkpoint: `D3_theta_mul_l1_s42_l0/best.pt`  ",
        "",
        "## 1. Ground Truth Dataset Algebra Audit",
        "Testing whether PTB-XL targets strictly satisfy Einthoven's and Goldberger's identities prior to any modeling:",
        "",
        "| Residual Identity | Expression | Mean Absolute (mV) | Median (mV) | p95 Absolute (mV) | Max Absolute (mV) | Target Integrity Status |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
        f"| $\\epsilon_{{III}}$ | $II - I - III$ | {dataset_stats['epsilon_III']['mean']:.6f} | {dataset_stats['epsilon_III']['median']:.6f} | {dataset_stats['epsilon_III']['p95']:.6f} | {dataset_stats['epsilon_III']['max']:.6f} | **Verified Physics Identity** |",
        f"| $\\epsilon_{{aVR}}$ | $aVR + \\frac{{I + II}}{{2}}$ | {dataset_stats['epsilon_aVR']['mean']:.6f} | {dataset_stats['epsilon_aVR']['median']:.6f} | {dataset_stats['epsilon_aVR']['p95']:.6f} | {dataset_stats['epsilon_aVR']['max']:.6f} | **Verified Physics Identity** |",
        f"| $\\epsilon_{{aVL}}$ | $aVL - I + \\frac{{II}}{{2}}$ | {dataset_stats['epsilon_aVL']['mean']:.6f} | {dataset_stats['epsilon_aVL']['median']:.6f} | {dataset_stats['epsilon_aVL']['p95']:.6f} | {dataset_stats['epsilon_aVL']['max']:.6f} | **Verified Physics Identity** |",
        f"| $\\epsilon_{{aVF}}$ | $aVF - II + \\frac{{I}}{{2}}$ | {dataset_stats['epsilon_aVF']['mean']:.6f} | {dataset_stats['epsilon_aVF']['median']:.6f} | {dataset_stats['epsilon_aVF']['p95']:.6f} | {dataset_stats['epsilon_aVF']['max']:.6f} | **Verified Physics Identity** |",
        "",
        f"> **Integrity Finding**: Across all 2,183 validation patients, the 95th percentile residual is under ${p95_res*1000:.2f}\\ \\mu\\text{{V}}$. PTB-XL targets obey Einthoven's law and Goldberger's equations to recording/numerical precision. Lead ordering and scaling are verified.",
        "",
        "## 2. Mathematical Sanity Check (Error Propagation)",
        "Testing numerical fidelity of algebraically derived errors against theoretical bounds:",
        f"- $\\max |(e_{{III}}^{{\\text{{derived}}}} - e_{{II}}) - \\epsilon_{{III}}| = {exact_diff_III:.10f}\\text{{ mV}}$",
        f"- $\\max |(e_{{aVR}}^{{\\text{{derived}}}} - (-0.5 \\cdot e_{{II}})) - \\epsilon_{{aVR}}| = {exact_diff_aVR:.10f}\\text{{ mV}}$",
        f"- $\\max |(e_{{aVL}}^{{\\text{{derived}}}} - (-0.5 \\cdot e_{{II}})) - \\epsilon_{{aVL}}| = {exact_diff_aVL:.10f}\\text{{ mV}}$",
        f"- $\\max |(e_{{aVF}}^{{\\text{{derived}}}} - e_{{II}}) - \\epsilon_{{aVF}}| = {exact_diff_aVF:.10f}\\text{{ mV}}$",
        f"- Raw $|e_l^{{\\text{{der}}}} - e_{{II}}|$ 99th percentile: III={p99_III*1000:.2f} µV, aVR={p99_aVR*1000:.2f} µV, aVL={p99_aVL*1000:.2f} µV, aVF={p99_aVF*1000:.2f} µV",
        f"- **Sanity Check**: `{'PASSED (Identities hold exactly)' if sanity_passed else 'FAILED'}`",
        "",
        "## 3. Algebraic Inconsistency Violations (Before vs. After)",
        "",
        "| Residual Metric | D3 Native Prediction (µV) | D3 Derived Prediction (µV) | Inconsistency Reduction |",
        "|:---|:---:|:---:|:---:|",
        f"| Mean $|II - I - III|$ | {viol_native['mean_III_violation_uV']:.2f} µV | {viol_derived['mean_III_violation_uV']:.6f} µV | Reduced to numerical precision |",
        f"| Mean $|aVR + (I+II)/2|$ | {viol_native['mean_aVR_violation_uV']:.2f} µV | {viol_derived['mean_aVR_violation_uV']:.6f} µV | Reduced to numerical precision |",
        f"| Mean $|aVL - I + II/2|$ | {viol_native['mean_aVL_violation_uV']:.2f} µV | {viol_derived['mean_aVL_violation_uV']:.6f} µV | Reduced to numerical precision |",
        f"| Mean $|aVF - II + I/2|$ | {viol_native['mean_aVF_violation_uV']:.2f} µV | {viol_derived['mean_aVF_violation_uV']:.6f} µV | Reduced to numerical precision |",
        "",
        "## 4. Head-to-Head Reconstruction Performance (Native vs. Derived)",
        "",
        "| Lead | Anatomical Domain | D3 Native $r$ | D3 Derived $r$ | $\\Delta r$ | Native MAE (mV) | Derived MAE (mV) | $\\Delta$ MAE | Native MSE | Derived MSE |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for row in lead_rows:
        lead = row["lead"]
        dom = "Observed Passthrough" if lead == "I" else ("Limb Lead" if lead in ["II", "III", "aVR", "aVL", "aVF"] else "Precordial Chest")
        md_lines.append(
            f"| **{lead}** | {dom} | {row['nat_r']:.4f} | {row['der_r']:.4f} | **{row['delta_r']:+.4f}** | "
            f"{row['nat_mae']:.4f} | {row['der_mae']:.4f} | {row['delta_mae']:+.4f} | "
            f"{row['nat_mse']:.4f} | {row['der_mse']:.4f} |"
        )

    md_lines.extend([
        "",
        "### 5. Aggregate Performance & 10,000-Sample Paired Patient Bootstrap CIs",
        "",
        "| Aggregate Metric | D3 Native | D3 Derived | $\\mathbb{E}[\\Delta_i]$ (Derived - Native) | 95% Bootstrap CI | Emp. $p$-value |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
        f"| **All-Missing Pearson $\\bar{{r}}$** | **{summary_metrics['mean_all_missing_r']['nat']:.4f}** | **{summary_metrics['mean_all_missing_r']['der']:.4f}** | **{boot_all_r['mean']:+.4f}** | `{boot_all_r['ci_lower']:+.4f}, {boot_all_r['ci_upper']:+.4f}` | {boot_all_r['p_value']:.4f} |",
        f"| **Limb-Only Missing Pearson $\\bar{{r}}$** | **{summary_metrics['mean_limb_missing_r']['nat']:.4f}** | **{summary_metrics['mean_limb_missing_r']['der']:.4f}** | **{boot_limb_r['mean']:+.4f}** | `{boot_limb_r['ci_lower']:+.4f}, {boot_limb_r['ci_upper']:+.4f}` | {boot_limb_r['p_value']:.4f} |",
        f"| **Precordial $V_1$–$V_6$ Pearson $\\bar{{r}}$** | **{summary_metrics['mean_chest_r']['nat']:.4f}** | **{summary_metrics['mean_chest_r']['der']:.4f}** | **{boot_chest_r['mean']:+.4f}** | Strictly Identical | - |",
        f"| **Tail Robustness $r_{{p05}}$** | **{summary_metrics['all_missing_r_p05']['nat']:.4f}** | **{summary_metrics['all_missing_r_p05']['der']:.4f}** | {summary_metrics['all_missing_r_p05']['der'] - summary_metrics['all_missing_r_p05']['nat']:+.4f} | - | - |",
        f"| **Missing L1 Error (mV)** | **{summary_metrics['all_missing_l1_mV']['nat']:.4f}** | **{summary_metrics['all_missing_l1_mV']['der']:.4f}** | **{boot_l1['mean']:+.4f}** | `{boot_l1['ci_lower']:+.4f}, {boot_l1['ci_upper']:+.4f}` | {boot_l1['p_value']:.4f} |",
        f"| **Missing MSE Error** | **{summary_metrics['all_missing_mse']['nat']:.4f}** | **{summary_metrics['all_missing_mse']['der']:.4f}** | {summary_metrics['all_missing_mse']['der'] - summary_metrics['all_missing_mse']['nat']:+.4f} | - | - |",
        "",
        "## 6. Critical Takeaways & Architectural Decision Gate",
        "- Zero algebraic inconsistency does NOT equal zero reconstruction error.",
        "- In the derived formulation, any error in predicting Lead II propagates directly: $e_{III} = e_{II}$ and $e_{aVF} = e_{II}$.",
        "- If algebraic derivation improves aggregate accuracy without compromising tails or precordial fidelity, it justifies implementing **D9 (8-lead basis)** and **D10 (direct-11 + consistency loss)**.",
    ])

    OUTPUT_MD.write_text("\n".join(md_lines))
    print(f"Saved Markdown report to: {OUTPUT_MD}")
    print("\n=== Stage A.2 Limb Algebra Preflight Complete ===")


if __name__ == "__main__":
    main()
