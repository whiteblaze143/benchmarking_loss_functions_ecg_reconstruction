"""Audit script for normalized metric distortion and local k-NN metric distortion in G002.

Computes:
1. Global normalized Frobenius metric distortion:
   epsilon_D = ||D_arm - D_gt||_F / ||D_gt||_F
2. Local k-NN metric distortion (k=10):
   epsilon_{D,10} = sqrt(sum_t sum_{u in N_k^{GT}(t)} (D_arm(t,u) - D_gt(t,u))^2) /
                    sqrt(sum_t sum_{u in N_k^{GT}(t)} D_gt(t,u)^2)
3. Bootstrap 95% confidence intervals for paired contrasts.
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

import sys
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


def compute_metric_distortion(X_arm: np.ndarray, X_gt: np.ndarray, k: int = 10, subsample: int = 500) -> dict[str, float]:
    """Computes global Frobenius distortion and local k-NN metric distortion."""
    T = len(X_gt)
    if T > subsample:
        step = T // subsample
        X_arm = X_arm[::step][:subsample]
        X_gt = X_gt[::step][:subsample]
        n_pts = len(X_gt)
    else:
        n_pts = T

    D_arm = np.linalg.norm(X_arm[:, None, :] - X_arm[None, :, :], axis=-1)
    D_gt = np.linalg.norm(X_gt[:, None, :] - X_gt[None, :, :], axis=-1)

    # 1. Global Frobenius distortion
    norm_gt = float(np.linalg.norm(D_gt, "fro"))
    if norm_gt > 1e-12:
        eps_d = float(np.linalg.norm(D_arm - D_gt, "fro") / norm_gt)
    else:
        eps_d = float("nan")

    # 2. Local k-NN metric distortion (k=10)
    k_eff = min(k, n_pts - 2)
    local_sq_diff = 0.0
    local_sq_gt = 0.0
    for t in range(n_pts):
        d_gt_t = D_gt[t].copy()
        d_gt_t[t] = np.inf
        nn_gt_indices = np.argpartition(d_gt_t, k_eff)[:k_eff]
        diffs = D_arm[t, nn_gt_indices] - D_gt[t, nn_gt_indices]
        local_sq_diff += float(np.sum(diffs ** 2))
        local_sq_gt += float(np.sum(D_gt[t, nn_gt_indices] ** 2))

    if local_sq_gt > 1e-12:
        eps_d_k = float(np.sqrt(local_sq_diff / local_sq_gt))
    else:
        eps_d_k = float("nan")

    return {
        "epsilon_d": eps_d,
        "epsilon_d_k10": eps_d_k,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit G002 Metric Distortion")
    parser.add_argument("--max-clinical", type=int, default=100)
    parser.add_argument("--max-pano", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/g002_eval/g002_metric_distortion.json")
    args = parser.parse_args()

    # Load model
    ckpt_path = Path("results/panobench_3view_variablethird_geovt_seed123/model_final.pt")
    if not ckpt_path.is_file():
        ckpt_path = Path("results/geovt_lineage_seed123/model_final.pt")
    model = load_author_model(
        root_dir / "author_code" / "nefnet_v2",
        ckpt_path if ckpt_path.is_absolute() else (root_dir / ckpt_path),
        DEVICE,
    )

    # 1. Clinical Records
    emory_dir = Path("/data/mithunmanivannan/heedb_emory/WFDB/2010")
    hea_files = []
    with os.scandir(emory_dir) as it:
        for entry in it:
            if entry.name.endswith(".hea") and not entry.name.startswith("._"):
                hea_files.append(Path(entry.path))
                if len(hea_files) >= args.max_clinical:
                    break

    clinical_results = []
    print(f"Auditing metric distortion on {len(hea_files)} clinical records...")
    for idx, hea_path in enumerate(hea_files):
        x_gt_8 = load_clinical_record(hea_path)
        if x_gt_8 is None:
            continue
        E_gt = x_gt_8.T  # (4608, 8)
        E_R0 = E_gt @ KORS_MATRIX.T  # (4608, 3)

        obs_raw = x_gt_8[INPUT_CANONICAL_INDICES]
        lo_obs, hi_obs = float(obs_raw.min()), float(obs_raw.max())
        span_obs = hi_obs - lo_obs if hi_obs > lo_obs else 1.0
        inp_norm = (obs_raw - lo_obs) / span_obs
        inp_t = torch.from_numpy(inp_norm).unsqueeze(0).to(DEVICE)
        inp_angles = torch.from_numpy(CANONICAL_ANGLES_RAD[INPUT_CANONICAL_INDICES]).unsqueeze(0).to(DEVICE)

        completed_8 = np.copy(x_gt_8)
        for q_idx in PREDICTED_CANONICAL_INDICES:
            tgt_angle = torch.from_numpy(CANONICAL_ANGLES_RAD[q_idx]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()
            pred_mv = pred_norm * span_obs + lo_obs
            completed_8[q_idx] = pred_mv

        E_R2 = completed_8.T
        E_R3 = E_R2 @ KORS_MATRIX.T

        m_R0 = compute_metric_distortion(E_R0, E_gt)
        m_R2 = compute_metric_distortion(E_R2, E_gt)
        m_R3 = compute_metric_distortion(E_R3, E_gt)

        clinical_results.append({
            "record_id": hea_path.stem,
            "R0_Kors": m_R0,
            "R2_Nef": m_R2,
            "R3_KorsNef": m_R3,
            "deltas": {
                "R2_minus_R0_eps": m_R2["epsilon_d"] - m_R0["epsilon_d"],
                "R2_minus_R3_eps": m_R2["epsilon_d"] - m_R3["epsilon_d"],
                "R2_minus_R0_eps_k10": m_R2["epsilon_d_k10"] - m_R0["epsilon_d_k10"],
                "R2_minus_R3_eps_k10": m_R2["epsilon_d_k10"] - m_R3["epsilon_d_k10"],
            }
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(hea_files)} done")

    # 2. PanoBench Records
    pano_dir = Path("/data/mithunmanivannan/panobench/test")
    pano_files = sorted(list(pano_dir.glob("*.mat")))[:args.max_pano]
    rng = np.random.default_rng(123)
    pano_results = []
    print(f"Auditing metric distortion on {len(pano_files)} PanoBench records...")
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

        E_44_gt = values_raw.T
        E_44_theta = completed_44.T
        m_44 = compute_metric_distortion(E_44_theta, E_44_gt)
        pano_results.append({
            "record_id": mat_path.stem,
            "dense_44": m_44,
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(pano_files)} done")

    # Bootstrap CIs
    def bootstrap_ci(arr, n_boot=2000):
        arr = np.array([x for x in arr if np.isfinite(x)])
        b_rng = np.random.default_rng(42)
        means = [np.mean(b_rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
        return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    d20_eps = [r["deltas"]["R2_minus_R0_eps"] for r in clinical_results]
    d23_eps = [r["deltas"]["R2_minus_R3_eps"] for r in clinical_results]
    d20_eps_k10 = [r["deltas"]["R2_minus_R0_eps_k10"] for r in clinical_results]
    d23_eps_k10 = [r["deltas"]["R2_minus_R3_eps_k10"] for r in clinical_results]

    r0_eps = [r["R0_Kors"]["epsilon_d"] for r in clinical_results]
    r2_eps = [r["R2_Nef"]["epsilon_d"] for r in clinical_results]
    r3_eps = [r["R3_KorsNef"]["epsilon_d"] for r in clinical_results]

    r0_eps_k10 = [r["R0_Kors"]["epsilon_d_k10"] for r in clinical_results]
    r2_eps_k10 = [r["R2_Nef"]["epsilon_d_k10"] for r in clinical_results]
    r3_eps_k10 = [r["R3_KorsNef"]["epsilon_d_k10"] for r in clinical_results]

    p44_eps = [r["dense_44"]["epsilon_d"] for r in pano_results]
    p44_eps_k10 = [r["dense_44"]["epsilon_d_k10"] for r in pano_results]

    summary = {
        "n_clinical": len(clinical_results),
        "n_pano": len(pano_results),
        "clinical_frobenius_distortion": {
            "R0_Kors": bootstrap_ci(r0_eps),
            "R2_Nef": bootstrap_ci(r2_eps),
            "R3_KorsNef": bootstrap_ci(r3_eps),
        },
        "clinical_local_k10_distortion": {
            "R0_Kors": bootstrap_ci(r0_eps_k10),
            "R2_Nef": bootstrap_ci(r2_eps_k10),
            "R3_KorsNef": bootstrap_ci(r3_eps_k10),
        },
        "panobench_44_distortion": {
            "epsilon_d": bootstrap_ci(p44_eps),
            "epsilon_d_k10": bootstrap_ci(p44_eps_k10),
        },
        "paired_contrasts": {
            "Delta_eps_R2_minus_R0": bootstrap_ci(d20_eps),
            "Delta_eps_R2_minus_R3": bootstrap_ci(d23_eps),
            "Delta_eps_k10_R2_minus_R0": bootstrap_ci(d20_eps_k10),
            "Delta_eps_k10_R2_minus_R3": bootstrap_ci(d23_eps_k10),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================== METRIC DISTORTION SUMMARY ==================")
    print(f"Clinical Frobenius Distortion epsilon_D:")
    print(f"  R0 (Kors VCG):      mean={summary['clinical_frobenius_distortion']['R0_Kors'][0]:.4f} [95% CI: {summary['clinical_frobenius_distortion']['R0_Kors'][1]:.4f}, {summary['clinical_frobenius_distortion']['R0_Kors'][2]:.4f}]")
    print(f"  R2 (Nef Panorama):  mean={summary['clinical_frobenius_distortion']['R2_Nef'][0]:.4f} [95% CI: {summary['clinical_frobenius_distortion']['R2_Nef'][1]:.4f}, {summary['clinical_frobenius_distortion']['R2_Nef'][2]:.4f}]")
    print(f"  R3 (Kors on Nef):   mean={summary['clinical_frobenius_distortion']['R3_KorsNef'][0]:.4f} [95% CI: {summary['clinical_frobenius_distortion']['R3_KorsNef'][1]:.4f}, {summary['clinical_frobenius_distortion']['R3_KorsNef'][2]:.4f}]")
    print(f"Clinical Local k-NN Distortion epsilon_{{D,10}}:")
    print(f"  R0 (Kors VCG):      mean={summary['clinical_local_k10_distortion']['R0_Kors'][0]:.4f} [95% CI: {summary['clinical_local_k10_distortion']['R0_Kors'][1]:.4f}, {summary['clinical_local_k10_distortion']['R0_Kors'][2]:.4f}]")
    print(f"  R2 (Nef Panorama):  mean={summary['clinical_local_k10_distortion']['R2_Nef'][0]:.4f} [95% CI: {summary['clinical_local_k10_distortion']['R2_Nef'][1]:.4f}, {summary['clinical_local_k10_distortion']['R2_Nef'][2]:.4f}]")
    print(f"PanoBench Dense 44-channel Distortion:")
    print(f"  epsilon_D:          mean={summary['panobench_44_distortion']['epsilon_d'][0]:.4f} [95% CI: {summary['panobench_44_distortion']['epsilon_d'][1]:.4f}, {summary['panobench_44_distortion']['epsilon_d'][2]:.4f}]")
    print(f"  epsilon_{{D,10}}:     mean={summary['panobench_44_distortion']['epsilon_d_k10'][0]:.4f} [95% CI: {summary['panobench_44_distortion']['epsilon_d_k10'][1]:.4f}, {summary['panobench_44_distortion']['epsilon_d_k10'][2]:.4f}]")
    print(f"Paired Contrast Delta epsilon_D (R2 - R0):")
    print(f"  mean={summary['paired_contrasts']['Delta_eps_R2_minus_R0'][0]:+.4f} [95% CI: {summary['paired_contrasts']['Delta_eps_R2_minus_R0'][1]:+.4f}, {summary['paired_contrasts']['Delta_eps_R2_minus_R0'][2]:+.4f}]")


if __name__ == "__main__":
    main()
