#!/usr/bin/env python3
"""G002-D: Nonlocal / Lag-Controlled Geometry & PCA-3 Dimensionality Diagnostic.

Addresses user requests:
1. PCA-3 Non-candidate diagnostic on R2:
   - Project R2 (8D) onto top 3 principal components (fitted purely on R2(t), no labels).
   - Evaluate whether 3D compression per se is inadequate or Kors projection specifically.
2. Nonlocal / Lag-Controlled Geometry:
   - Exclude pairs inside temporal locality radius Delta_max = 50 samples (100 ms at 500 Hz).
   - Stratify pairs by temporal lag:
       Local:        1 <= |t - u| <= 50  (0 - 100 ms)
       Intra-beat:  50 <  |t - u| <= 250 (100 - 500 ms)
       Cross-beat:        |t - u| >  250 (> 500 ms)
   - Recompute rho_D and Centered Kernel Alignment (CKA) across lag strata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.spatial.distance as dist
import scipy.stats as stats
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theta_repspat.panorama import (
    CANONICAL_ANGLES_RAD,
    INPUT_INDICES,
    PREDICTED_INDICES,
)
from theta_repspat.vendor import load_author_model


# Standard Kors regression transformation matrix (8 leads -> 3-lead Frank VCG X, Y, Z)
# Lead order: I, II, V1, V2, V3, V4, V5, V6
KORS_MATRIX = np.array([
    [ 0.38,  -0.07,  -0.13,   0.05,  -0.01,   0.14,   0.06,   0.54],   # X
    [-0.07,   0.93,   0.06,  -0.02,  -0.05,   0.06,  -0.17,   0.13],   # Y
    [-0.11,  -0.23,  -0.43,  -0.06,  -0.14,  -0.20,  -0.11,   0.31],   # Z
], dtype=np.float64)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    valid = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(valid, size=len(valid), replace=True))) for _ in range(n_boot)]
    return float(np.mean(valid)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_distance_matrix(X: np.ndarray) -> np.ndarray:
    return dist.squareform(dist.pdist(X, metric="euclidean"))


def compute_imq_kernel(D: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    d_med = float(np.median(D[iu]))
    if d_med < 1e-8:
        d_med = 1.0
    c = gamma * d_med
    K = 1.0 / np.sqrt((D / c) ** 2 + 1.0)
    return K


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n, dtype=np.float64) - (1.0 / n) * np.ones((n, n), dtype=np.float64)
    return H @ K @ H


def fit_pca_3(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA-3 on X [T, C] without any external labels. Returns (X_pca3, exp_var_ratio)."""
    X_cent = X - np.mean(X, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_cent, full_matrices=False)
    X_pca3 = X_cent @ Vt[:3, :].T  # [T, 3]
    var_exp = (S ** 2) / np.sum(S ** 2)
    return X_pca3, var_exp


def evaluate_record_geometry(
    ecg_8lead: np.ndarray,
    model,
    delta_max: int = 50,
    W_global_pca: np.ndarray | None = None,
) -> dict:
    """Evaluate full, nonlocal, and lag-stratified geometry across R0, R1, R2, R3, R2_PCA3, and R2_Global_PCA3."""
    # ecg_8lead: [T, 8]
    T = ecg_8lead.shape[0]
    x_gt_8 = ecg_8lead.T  # [8, T]

    # Ground truth reference: R1 in R^8
    R1 = ecg_8lead

    # Baseline R0: Kors VCG in R^3
    R0 = ecg_8lead @ KORS_MATRIX.T

    # Nef 8D Panorama R2
    input_indices = list(INPUT_INDICES)
    obs_raw = x_gt_8[input_indices]
    lo_obs, hi_obs = float(obs_raw.min()), float(obs_raw.max())
    span_obs = hi_obs - lo_obs if hi_obs > lo_obs else 1.0
    inp_norm = (obs_raw - lo_obs) / span_obs
    inp_t = torch.from_numpy(inp_norm.astype(np.float32)).unsqueeze(0).to(DEVICE)
    inp_angles = torch.from_numpy(CANONICAL_ANGLES_RAD[input_indices]).unsqueeze(0).to(DEVICE)

    completed_8 = np.copy(x_gt_8)
    for q_idx in PREDICTED_INDICES:
        tgt_angle = torch.from_numpy(CANONICAL_ANGLES_RAD[q_idx]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()
        pred_mv = pred_norm * span_obs + lo_obs
        completed_8[q_idx] = pred_mv

    R2 = completed_8.T  # [T, 8]

    # R3: Kors on Nef completion
    R3 = R2 @ KORS_MATRIX.T

    # Non-candidate diagnostic: PCA-3 of R2
    R2_pca3, var_exp = fit_pca_3(R2)

    arms_raw = {
        "R1": R1,
        "R0": R0,
        "R2": R2,
        "R3": R3,
        "R2_PCA3": R2_pca3,
    }
    if W_global_pca is not None:
        R2_cent = R2 - np.mean(R2, axis=0, keepdims=True)
        arms_raw["R2_Global_PCA3"] = R2_cent @ W_global_pca

    # Subsample along time to T_sub = 500 for pairwise state-space evaluation
    step = T // 500
    sub_slice = slice(None, None, step)
    arms = {name: arr[sub_slice][:500] for name, arr in arms_raw.items()}
    T_sub = arms["R1"].shape[0]

    # Distance matrices
    D_dict = {name: compute_distance_matrix(arr) for name, arr in arms.items()}
    K_dict = {name: compute_imq_kernel(D, gamma=1.0) for name, D in D_dict.items()}
    K_tilde_dict = {name: center_kernel(K) for name, K in K_dict.items()}

    D_GT = D_dict["R1"]
    K_tilde_GT = K_tilde_dict["R1"]

    # Generate lag masks: exact physical lag in samples (at 500 Hz)
    t_idx, u_idx = np.triu_indices(T_sub, k=1)
    lags_samples = np.abs(t_idx - u_idx) * step

    masks = {
        "all": np.ones_like(lags_samples, dtype=bool),
        "local": lags_samples <= delta_max,
        "nonlocal": lags_samples > delta_max,
        "intra_beat": (lags_samples > delta_max) & (lags_samples <= 250),
        "cross_beat": lags_samples > 250,
    }

    record_results = {"var_exp_pca": var_exp.tolist()}

    for arm_name in ["R0", "R2", "R3", "R2_PCA3"]:
        arm_res = {}
        D_X = D_dict[arm_name]
        K_tilde_X = K_tilde_dict[arm_name]

        # 1. Scale-aligned metric stress (global)
        d_x_all = D_X[t_idx, u_idx]
        d_gt_all = D_GT[t_idx, u_idx]
        denom = np.sum(d_x_all ** 2)
        numer = np.sum(d_x_all * d_gt_all)
        a_star = numer / denom if denom > 1e-12 else 1.0
        stress_aligned = np.linalg.norm(a_star * d_x_all - d_gt_all) / np.linalg.norm(d_gt_all)
        arm_res["stress_aligned"] = float(stress_aligned)

        # 2. Lag-stratified CKA and rho_D
        for stratum, mask in masks.items():
            if not np.any(mask):
                continue
            # Distance rank correlation
            sub_x = D_X[t_idx[mask], u_idx[mask]]
            sub_gt = D_GT[t_idx[mask], u_idx[mask]]
            rho, _ = stats.spearmanr(sub_x, sub_gt)
            arm_res[f"rho_{stratum}"] = float(rho)

            # Centered Kernel Alignment (CKA) on subset of pairs
            k_x_sub = K_tilde_X[t_idx[mask], u_idx[mask]]
            k_gt_sub = K_tilde_GT[t_idx[mask], u_idx[mask]]
            norm_x = np.linalg.norm(k_x_sub)
            norm_gt = np.linalg.norm(k_gt_sub)
            if norm_x > 1e-12 and norm_gt > 1e-12:
                cka_sub = float(np.dot(k_x_sub, k_gt_sub) / (norm_x * norm_gt))
            else:
                cka_sub = 0.0
            arm_res[f"cka_{stratum}"] = cka_sub

        record_results[arm_name] = arm_res

    return record_results


def main():
    parser = argparse.ArgumentParser(description="G002-D Nonlocal Geometry and PCA-3 Diagnostic")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--delta-max", type=int, default=50, help="Maximum temporal locality radius (samples)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    model_path = ROOT / "results/panobench_3view_variablethird_geovt_seed123/model_final.pt"
    if not model_path.is_file():
        model_path = ROOT / "results/geovt_lineage_seed123/model_final.pt"
    model = load_author_model(ROOT / "author_code" / "nefnet_v2", model_path, DEVICE)
    model.eval()

    from audit_metric_distortion import load_clinical_record

    # Load records from HEEDB Emory
    emory_dir = Path("/data/mithunmanivannan/heedb_emory/WFDB/2010")
    hea_files = []
    with os.scandir(emory_dir) as it:
        for entry in it:
            if entry.name.endswith(".hea") and not entry.name.startswith("._"):
                hea_files.append(Path(entry.path))
                if len(hea_files) >= args.max_records:
                    break

    print(f"\n================== G002-D NONLOCAL & PCA-3 DIAGNOSTIC ==================")
    print(f"Evaluated Records: {len(hea_files)} clinical records (HEEDB Emory)")
    print(f"Locality Radius Delta_max: {args.delta_max} samples")
    print(f"Stratification: Local (<=50), Nonlocal (>50), Intra-beat (50-250), Cross-beat (>250)")
    print(f"=======================================================================\n")

    # Fit global PCA-3 basis W_PCA in R^{8 x 3} on disjoint development set (PTB-XL train)
    print("Fitting global PCA-3 basis W_PCA in R^{8 x 3} on disjoint PTB-XL train records...")
    ptbxl_files = sorted(Path("/home/mithunmanivannan/data/ptb_xl/tensors/train").glob("*.pt"))[:50]
    r2_train_list = []
    input_indices = list(INPUT_INDICES)
    for p_file in ptbxl_files:
        p_data = torch.load(p_file, map_location="cpu", weights_only=True)
        if isinstance(p_data, dict):
            p_data = p_data["ecg"]
        p_8 = p_data[[0, 1, 6, 7, 8, 9, 10, 11], :4608].float().numpy()
        obs_raw = p_8[input_indices]
        lo_obs, hi_obs = float(obs_raw.min()), float(obs_raw.max())
        span_obs = hi_obs - lo_obs if hi_obs > lo_obs else 1.0
        inp_norm = (obs_raw - lo_obs) / span_obs
        inp_t = torch.from_numpy(inp_norm.astype(np.float32)).unsqueeze(0).to(DEVICE)
        inp_angles = torch.from_numpy(CANONICAL_ANGLES_RAD[input_indices]).unsqueeze(0).to(DEVICE)
        completed_8 = np.copy(p_8)
        for q_idx in PREDICTED_INDICES:
            tgt_angle = torch.from_numpy(CANONICAL_ANGLES_RAD[q_idx]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()
            completed_8[q_idx] = pred_norm * span_obs + lo_obs
        r2_t = completed_8.T  # [4608, 8]
        r2_cent = r2_t - np.mean(r2_t, axis=0, keepdims=True)
        sub_idx = np.linspace(0, 4608 - 1, 250, dtype=int)
        r2_train_list.append(r2_cent[sub_idx])

    r2_concat = np.concatenate(r2_train_list, axis=0)  # [N*250, 8]
    _, S_glob, Vt_glob = np.linalg.svd(r2_concat, full_matrices=False)
    W_global_pca = Vt_glob[:3, :].T  # [8, 3]
    var_exp_glob = (S_glob ** 2) / np.sum(S_glob ** 2)
    print(f"Global PCA Top-3 Variance Explained: {var_exp_glob[:3]} (sum = {np.sum(var_exp_glob[:3]):.4%})\n")

    results = []
    for idx, hea_path in enumerate(hea_files):
        x_gt_8 = load_clinical_record(hea_path)
        if x_gt_8 is None:
            continue
        # x_gt_8: [8, 4608]
        # Keep full length 4608 for GeoVT model forward pass
        ecg_8lead = x_gt_8.T  # [4608, 8]

        res = evaluate_record_geometry(ecg_8lead, model, delta_max=args.delta_max, W_global_pca=W_global_pca)
        results.append(res)
        if (len(results)) % 25 == 0 or idx == len(hea_files) - 1:
            print(f"Processed {len(results)}/{len(hea_files)} records...")


    # Aggregate metrics across records
    summary = {}
    arms = ["R0", "R2", "R3", "R2_PCA3", "R2_Global_PCA3"]

    print("\n--- 1. Scale-Aligned Stress & CKA (All Pairs vs Nonlocal Pairs) ---")
    print(f"{'Arm':18s} | {'Stress (Aligned)':18s} | {'CKA (All)':16s} | {'CKA (Nonlocal)':16s} | {'CKA (Cross-beat)':16s}")
    print("-" * 94)

    for arm in arms:
        stress_vals = np.array([r[arm]["stress_aligned"] for r in results])
        cka_all = np.array([r[arm]["cka_all"] for r in results])
        cka_nonlocal = np.array([r[arm]["cka_nonlocal"] for r in results])
        cka_cross = np.array([r[arm]["cka_cross_beat"] for r in results])

        m_s, lo_s, hi_s = bootstrap_ci(stress_vals)
        m_a, lo_a, hi_a = bootstrap_ci(cka_all)
        m_nl, lo_nl, hi_nl = bootstrap_ci(cka_nonlocal)
        m_cb, lo_cb, hi_cb = bootstrap_ci(cka_cross)

        print(f"{arm:18s} | {m_s:.4f} [{lo_s:.4f}, {hi_s:.4f}] | {m_a:.4f} [{lo_a:.4f}, {hi_a:.4f}] | {m_nl:.4f} [{lo_nl:.4f}, {hi_nl:.4f}] | {m_cb:.4f} [{lo_cb:.4f}, {hi_cb:.4f}]")

        summary[arm] = {
            "stress_aligned": [m_s, lo_s, hi_s],
            "cka_all": [m_a, lo_a, hi_a],
            "cka_nonlocal": [m_nl, lo_nl, hi_nl],
            "cka_cross_beat": [m_cb, lo_cb, hi_cb],
        }

    # Paired contrasts
    print("\n--- 2. Paired Contrasts for Nonlocal CKA (Bootstrap 95% CI) ---")
    r0_nl = np.array([r["R0"]["cka_nonlocal"] for r in results])
    r2_nl = np.array([r["R2"]["cka_nonlocal"] for r in results])
    r3_nl = np.array([r["R3"]["cka_nonlocal"] for r in results])
    rp_nl = np.array([r["R2_PCA3"]["cka_nonlocal"] for r in results])
    rg_nl = np.array([r["R2_Global_PCA3"]["cka_nonlocal"] for r in results])

    # Delta R2 - R0 (nonlocal)
    d_20 = r2_nl - r0_nl
    m, lo, hi = bootstrap_ci(d_20)
    print(f"Delta CKA Nonlocal (R2 - R0):          mean={m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # Delta R2 - R3 (nonlocal)
    d_23 = r2_nl - r3_nl
    m, lo, hi = bootstrap_ci(d_23)
    print(f"Delta CKA Nonlocal (R2 - R3):          mean={m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # Delta R2 - R2_PCA3 (local 3D compression penalty)
    d_2p = r2_nl - rp_nl
    m, lo, hi = bootstrap_ci(d_2p)
    print(f"Delta CKA Nonlocal (R2 - Local PCA3):  mean={m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # Delta R2 - R2_Global_PCA3 (global 3D compression penalty)
    d_2g = r2_nl - rg_nl
    m, lo, hi = bootstrap_ci(d_2g)
    print(f"Delta CKA Nonlocal (R2 - Global PCA3): mean={m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # Delta Global PCA3 - Kors
    d_gk = rg_nl - r3_nl
    m, lo, hi = bootstrap_ci(d_gk)
    print(f"Delta CKA Nonlocal (Global PCA3 - Kors): mean={m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # Local PCA-3 variance explained
    var_3 = np.array([sum(r["var_exp_pca"][:3]) for r in results])
    print(f"\nR2 Local Top-3 Principal Components Variance Explained: {np.mean(var_3)*100:.2f}% ± {np.std(var_3)*100:.2f}%")

    out_file = ROOT / "results/g002_eval/g002_d_nonlocal_pca.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved G002-D results to: {out_file}")


if __name__ == "__main__":
    main()
