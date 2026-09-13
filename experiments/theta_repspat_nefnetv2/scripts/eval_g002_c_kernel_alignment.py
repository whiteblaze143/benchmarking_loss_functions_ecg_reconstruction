"""G002-C: Scale-Aligned Metric Stress and Centered Kernel Alignment Audit.

Computes:
1. Scale-Aligned Metric Stress:
   a_X^* = <D^X, D^{GT}>_F / ||D^X||_F^2
   epsilon_D^{aligned}(X) = ||a_X^* D^X - D^{GT}||_F / ||D^{GT}||_F
   Removes arbitrary coordinate/amplitude scaling between R0 in R^3 and GT in R^8.

2. Centered Kernel Alignment (CKA) & Hilbert-Schmidt Distance:
   Directly measures the similarity geometry repSpat's MMD machinery sees.
   K_X(i, j) = [ D_X(i, j)^2 + (gamma * median(D_X))^2 ]^{-1/2}  (with gamma=1.0)
   H = I - (1/n) * 1 * 1^T
   K_tilde_X = (H @ K_X @ H) / ||H @ K_X @ H||_F
   A_K(X, GT) = <K_tilde_X, K_tilde_GT>_F  (Centered Kernel Alignment in [0, 1])
   d_HS(X, GT) = ||K_tilde_X - K_tilde_GT||_F = sqrt(2 * (1 - A_K(X, GT)))

Evaluated across:
- Clinical 8-lead reference: 100 HEEDB Emory WFDB records
- PanoBench 44-view reference: 100 held-out test records
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.stats
import torch

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from theta_repspat.vendor import load_author_model
from eval_g002 import (
    CANONICAL_ANGLES_RAD,
    INPUT_CANONICAL_INDICES,
    PREDICTED_CANONICAL_INDICES,
    KORS_MATRIX,
    EXPECTED_LEN,
    PANOBENCH_ANGLES_RAD,
    _upsample2x,
    load_clinical_record,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


def compute_aligned_stress_and_cka(
    X_arm: np.ndarray,
    X_ref: np.ndarray,
    gamma: float = 1.0,
    subsample: int = 500,
) -> dict[str, float]:
    """Computes scale-aligned metric stress and centered kernel alignment."""
    T = len(X_ref)
    if T > subsample:
        step = T // subsample
        X_arm = X_arm[::step][:subsample]
        X_ref = X_ref[::step][:subsample]
        n = len(X_ref)
    else:
        n = T

    D_arm = np.linalg.norm(X_arm[:, None, :] - X_arm[None, :, :], axis=-1)
    D_ref = np.linalg.norm(X_ref[:, None, :] - X_ref[None, :, :], axis=-1)

    # 1. Scale-aligned metric stress
    norm_arm_sq = float(np.sum(D_arm ** 2))
    norm_ref = float(np.linalg.norm(D_ref, "fro"))
    inner_prod = float(np.sum(D_arm * D_ref))

    if norm_arm_sq > 1e-12 and inner_prod > 0 and norm_ref > 1e-12:
        a_star = inner_prod / norm_arm_sq
        eps_aligned = float(np.linalg.norm(a_star * D_arm - D_ref, "fro") / norm_ref)
    else:
        a_star = 1.0
        eps_aligned = float("nan")

    # Raw unaligned Frobenius stress for comparison
    eps_raw = float(np.linalg.norm(D_arm - D_ref, "fro") / norm_ref) if norm_ref > 1e-12 else float("nan")

    # 2. Centered Kernel Alignment (CKA) under IMQ kernel with median heuristic
    # Distance-relative bandwidth: c = gamma * median_{i<j, D_ij > 0} D_ij
    nonzero_arm = D_arm[D_arm > 0]
    nonzero_ref = D_ref[D_ref > 0]
    c_arm = float(gamma * np.median(nonzero_arm)) if len(nonzero_arm) > 0 else 1.0
    c_ref = float(gamma * np.median(nonzero_ref)) if len(nonzero_ref) > 0 else 1.0

    K_arm = 1.0 / np.sqrt(D_arm ** 2 + c_arm ** 2)
    K_ref = 1.0 / np.sqrt(D_ref ** 2 + c_ref ** 2)

    # Centering matrix H = I - (1/n) * 1 * 1^T
    H = np.eye(n) - np.ones((n, n)) / float(n)
    HK_arm_H = H @ K_arm @ H
    HK_ref_H = H @ K_ref @ H

    norm_HK_arm = float(np.linalg.norm(HK_arm_H, "fro"))
    norm_HK_ref = float(np.linalg.norm(HK_ref_H, "fro"))

    if norm_HK_arm > 1e-12 and norm_HK_ref > 1e-12:
        K_tilde_arm = HK_arm_H / norm_HK_arm
        K_tilde_ref = HK_ref_H / norm_HK_ref
        cka = float(np.sum(K_tilde_arm * K_tilde_ref))
        # Clamp CKA to [-1.0, 1.0] for numerical safety
        cka = float(np.clip(cka, -1.0, 1.0))
        d_hs = float(np.linalg.norm(K_tilde_arm - K_tilde_ref, "fro"))
    else:
        cka = float("nan")
        d_hs = float("nan")

    return {
        "a_star": a_star,
        "eps_raw": eps_raw,
        "eps_aligned": eps_aligned,
        "cka": cka,
        "d_hs": d_hs,
    }


def main():
    parser = argparse.ArgumentParser(description="G002-C: Scale-Aligned Stress and CKA")
    parser.add_argument("--max-clinical", type=int, default=100)
    parser.add_argument("--max-pano", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/g002_eval/g002_c_kernel_alignment.json")
    args = parser.parse_args()

    ckpt_path = root_dir / "results/panobench_3view_variablethird_geovt_seed123/model_final.pt"
    if not ckpt_path.is_file():
        ckpt_path = root_dir / "results/geovt_lineage_seed123/model_final.pt"
    model = load_author_model(root_dir / "author_code" / "nefnet_v2", ckpt_path, DEVICE)
    model.eval()

    # 1. Clinical Records (HEEDB Emory)
    emory_dir = Path("/data/mithunmanivannan/heedb_emory/WFDB/2010")
    hea_files = []
    with os.scandir(emory_dir) as it:
        for entry in it:
            if entry.name.endswith(".hea") and not entry.name.startswith("._"):
                hea_files.append(Path(entry.path))
                if len(hea_files) >= args.max_clinical:
                    break

    clinical_records = []
    print(f"Auditing G002-C on {len(hea_files)} clinical records...")
    for idx, hea_path in enumerate(hea_files):
        x_gt_8 = load_clinical_record(hea_path)
        if x_gt_8 is None:
            continue
        E_ref = x_gt_8.T  # (4608, 8) observational reference
        E_R0 = E_ref @ KORS_MATRIX.T  # (4608, 3)

        obs_raw = x_gt_8[INPUT_CANONICAL_INDICES]
        lo_obs, hi_obs = float(obs_raw.min()), float(obs_raw.max())
        span_obs = hi_obs - lo_obs if hi_obs > lo_obs else 1.0
        inp_norm = (obs_raw - lo_obs) / span_obs
        inp_t = torch.from_numpy(inp_norm.astype(np.float32)).unsqueeze(0).to(DEVICE)
        inp_angles = torch.from_numpy(CANONICAL_ANGLES_RAD[INPUT_CANONICAL_INDICES]).unsqueeze(0).to(DEVICE)

        completed_8 = np.copy(x_gt_8)
        for q_idx in PREDICTED_CANONICAL_INDICES:
            tgt_angle = torch.from_numpy(CANONICAL_ANGLES_RAD[q_idx]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()
            pred_mv = pred_norm * span_obs + lo_obs
            completed_8[q_idx] = pred_mv

        E_R2 = completed_8.T  # (4608, 8)
        E_R3 = E_R2 @ KORS_MATRIX.T  # (4608, 3)

        m_R0 = compute_aligned_stress_and_cka(E_R0, E_ref)
        m_R2 = compute_aligned_stress_and_cka(E_R2, E_ref)
        m_R3 = compute_aligned_stress_and_cka(E_R3, E_ref)

        clinical_records.append({
            "record_id": hea_path.stem,
            "R0_Kors": m_R0,
            "R2_Nef": m_R2,
            "R3_KorsNef": m_R3,
            "deltas": {
                "Delta_eps_aligned_R2_minus_R0": m_R2["eps_aligned"] - m_R0["eps_aligned"],
                "Delta_eps_aligned_R2_minus_R3": m_R2["eps_aligned"] - m_R3["eps_aligned"],
                "Delta_cka_R2_minus_R0": m_R2["cka"] - m_R0["cka"],
                "Delta_cka_R2_minus_R3": m_R2["cka"] - m_R3["cka"],
                "Delta_d_hs_R2_minus_R0": m_R2["d_hs"] - m_R0["d_hs"],
            }
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(hea_files)} clinical records done")

    # 2. PanoBench Records (Held-out dense 44-view geometry)
    pano_dir = Path("/data/mithunmanivannan/panobench/test")
    pano_files = sorted(list(pano_dir.glob("*.mat")))[:args.max_pano]
    rng = np.random.default_rng(123)
    pano_records = []
    print(f"Auditing G002-C on {len(pano_files)} PanoBench records...")
    for idx, mat_path in enumerate(pano_files):
        raw = scipy.io.loadmat(mat_path)["Panobench"][:44].astype(np.float32)
        values_raw = _upsample2x(raw)
        crop_start = int(rng.integers(0, 5000 - EXPECTED_LEN + 1))
        values_raw = values_raw[:, crop_start: crop_start + EXPECTED_LEN]

        third_input = int(rng.integers(2, 44))
        input_indices = np.array([0, 1, third_input])
        obs_raw = values_raw[input_indices]
        lo_obs, hi_obs = float(obs_raw.min()), float(obs_raw.max())
        span_obs = hi_obs - lo_obs if hi_obs > lo_obs else 1.0

        inp_norm = (obs_raw - lo_obs) / span_obs
        inp_t = torch.from_numpy(inp_norm).unsqueeze(0).to(DEVICE)
        inp_angles = torch.from_numpy(PANOBENCH_ANGLES_RAD[input_indices]).unsqueeze(0).to(DEVICE)

        completed_44 = np.copy(values_raw)
        queries_to_pred = [q for q in range(44) if q not in input_indices]
        for q in queries_to_pred:
            tgt_angle = torch.from_numpy(PANOBENCH_ANGLES_RAD[q]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()
            pred_mv = pred_norm * span_obs + lo_obs
            completed_44[q] = pred_mv

        E_44_ref = values_raw.T
        E_44_theta = completed_44.T
        m_44 = compute_aligned_stress_and_cka(E_44_theta, E_44_ref)
        pano_records.append({
            "record_id": mat_path.stem,
            "dense_44": m_44,
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(pano_files)} pano records done")

    # Bootstrap CIs
    def bootstrap_ci(arr, n_boot=2000):
        arr = np.array([x for x in arr if np.isfinite(x)])
        b_rng = np.random.default_rng(42)
        means = [np.mean(b_rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
        return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    d_eps_al_20 = [r["deltas"]["Delta_eps_aligned_R2_minus_R0"] for r in clinical_records]
    d_eps_al_23 = [r["deltas"]["Delta_eps_aligned_R2_minus_R3"] for r in clinical_records]
    d_cka_20 = [r["deltas"]["Delta_cka_R2_minus_R0"] for r in clinical_records]
    d_cka_23 = [r["deltas"]["Delta_cka_R2_minus_R3"] for r in clinical_records]
    d_dhs_20 = [r["deltas"]["Delta_d_hs_R2_minus_R0"] for r in clinical_records]

    r0_eps_al = [r["R0_Kors"]["eps_aligned"] for r in clinical_records]
    r2_eps_al = [r["R2_Nef"]["eps_aligned"] for r in clinical_records]
    r3_eps_al = [r["R3_KorsNef"]["eps_aligned"] for r in clinical_records]

    r0_cka = [r["R0_Kors"]["cka"] for r in clinical_records]
    r2_cka = [r["R2_Nef"]["cka"] for r in clinical_records]
    r3_cka = [r["R3_KorsNef"]["cka"] for r in clinical_records]

    p44_eps_al = [r["dense_44"]["eps_aligned"] for r in pano_records]
    p44_cka = [r["dense_44"]["cka"] for r in pano_records]

    summary = {
        "n_clinical": len(clinical_records),
        "n_pano": len(pano_records),
        "clinical_scale_aligned_stress": {
            "R0_Kors": bootstrap_ci(r0_eps_al),
            "R2_Nef": bootstrap_ci(r2_eps_al),
            "R3_KorsNef": bootstrap_ci(r3_eps_al),
        },
        "clinical_centered_kernel_alignment": {
            "R0_Kors": bootstrap_ci(r0_cka),
            "R2_Nef": bootstrap_ci(r2_cka),
            "R3_KorsNef": bootstrap_ci(r3_cka),
        },
        "panobench_44": {
            "eps_aligned": bootstrap_ci(p44_eps_al),
            "cka": bootstrap_ci(p44_cka),
        },
        "paired_contrasts": {
            "Delta_eps_aligned_R2_minus_R0": bootstrap_ci(d_eps_al_20),
            "Delta_eps_aligned_R2_minus_R3": bootstrap_ci(d_eps_al_23),
            "Delta_cka_R2_minus_R0": bootstrap_ci(d_cka_20),
            "Delta_cka_R2_minus_R3": bootstrap_ci(d_cka_23),
            "Delta_d_hs_R2_minus_R0": bootstrap_ci(d_dhs_20),
        },
    }

    out_file = root_dir / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================== G002-C AUDIT SUMMARY ==================")
    print("1. Scale-Aligned Metric Stress epsilon_D^{aligned}:")
    print(f"  R0 (Kors VCG):      mean={summary['clinical_scale_aligned_stress']['R0_Kors'][0]:.4f} [95% CI: {summary['clinical_scale_aligned_stress']['R0_Kors'][1]:.4f}, {summary['clinical_scale_aligned_stress']['R0_Kors'][2]:.4f}]")
    print(f"  R2 (Nef Panorama):  mean={summary['clinical_scale_aligned_stress']['R2_Nef'][0]:.4f} [95% CI: {summary['clinical_scale_aligned_stress']['R2_Nef'][1]:.4f}, {summary['clinical_scale_aligned_stress']['R2_Nef'][2]:.4f}]")
    print(f"  R3 (Kors on Nef):   mean={summary['clinical_scale_aligned_stress']['R3_KorsNef'][0]:.4f} [95% CI: {summary['clinical_scale_aligned_stress']['R3_KorsNef'][1]:.4f}, {summary['clinical_scale_aligned_stress']['R3_KorsNef'][2]:.4f}]")
    print(f"  Paired Delta (R2 - R0): mean={summary['paired_contrasts']['Delta_eps_aligned_R2_minus_R0'][0]:+.4f} [95% CI: {summary['paired_contrasts']['Delta_eps_aligned_R2_minus_R0'][1]:+.4f}, {summary['paired_contrasts']['Delta_eps_aligned_R2_minus_R0'][2]:+.4f}]")
    print(f"  Paired Delta (R2 - R3): mean={summary['paired_contrasts']['Delta_eps_aligned_R2_minus_R3'][0]:+.4f} [95% CI: {summary['paired_contrasts']['Delta_eps_aligned_R2_minus_R3'][1]:+.4f}, {summary['paired_contrasts']['Delta_eps_aligned_R2_minus_R3'][2]:+.4f}]")

    print("\n2. Centered Kernel Alignment A_K(X, GT) (MMD Similarity Geometry):")
    print(f"  R0 (Kors VCG):      mean={summary['clinical_centered_kernel_alignment']['R0_Kors'][0]:.4f} [95% CI: {summary['clinical_centered_kernel_alignment']['R0_Kors'][1]:.4f}, {summary['clinical_centered_kernel_alignment']['R0_Kors'][2]:.4f}]")
    print(f"  R2 (Nef Panorama):  mean={summary['clinical_centered_kernel_alignment']['R2_Nef'][0]:.4f} [95% CI: {summary['clinical_centered_kernel_alignment']['R2_Nef'][1]:.4f}, {summary['clinical_centered_kernel_alignment']['R2_Nef'][2]:.4f}]")
    print(f"  R3 (Kors on Nef):   mean={summary['clinical_centered_kernel_alignment']['R3_KorsNef'][0]:.4f} [95% CI: {summary['clinical_centered_kernel_alignment']['R3_KorsNef'][1]:.4f}, {summary['clinical_centered_kernel_alignment']['R3_KorsNef'][2]:.4f}]")
    print(f"  Paired Delta (R2 - R0): mean={summary['paired_contrasts']['Delta_cka_R2_minus_R0'][0]:+.4f} [95% CI: {summary['paired_contrasts']['Delta_cka_R2_minus_R0'][1]:+.4f}, {summary['paired_contrasts']['Delta_cka_R2_minus_R0'][2]:+.4f}]")
    print(f"  Paired Delta (R2 - R3): mean={summary['paired_contrasts']['Delta_cka_R2_minus_R3'][0]:+.4f} [95% CI: {summary['paired_contrasts']['Delta_cka_R2_minus_R3'][1]:+.4f}, {summary['paired_contrasts']['Delta_cka_R2_minus_R3'][2]:+.4f}]")

    print("\n3. Dense 44-channel PanoBench (Held-out test split):")
    print(f"  Scale-aligned stress eps_aligned: mean={summary['panobench_44']['eps_aligned'][0]:.4f} [95% CI: {summary['panobench_44']['eps_aligned'][1]:.4f}, {summary['panobench_44']['eps_aligned'][2]:.4f}]")
    print(f"  Centered Kernel Alignment (CKA):  mean={summary['panobench_44']['cka'][0]:.4f} [95% CI: {summary['panobench_44']['cka'][1]:.4f}, {summary['panobench_44']['cka'][2]:.4f}]")


if __name__ == "__main__":
    main()
