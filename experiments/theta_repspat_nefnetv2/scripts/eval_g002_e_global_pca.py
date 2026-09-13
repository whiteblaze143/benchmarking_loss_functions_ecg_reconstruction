#!/usr/bin/env python3
"""Global PCA-3 diagnostic for Nef panorama R2 on Emory (N=100).

Tests whether a SINGLE, GLOBALLY FROZEN linear basis W_PCA in R^{8 x 3}
fitted on a disjoint training set (PTB-XL train) preserves observational
geometry alignment (nonlocal CKA) as effectively as per-record local PCA-3:
    R2_global_PCA3 = R2 @ W_PCA
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.spatial.distance as dist
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theta_repspat.clinical_dataset import STANDARD_12LEAD_ANGLES_RAD
from theta_repspat.vendor import load_author_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

# Kors synthesis matrix (8 leads -> 3D VCG: X, Y, Z)
# 8 leads: I, II, V1, V2, V3, V4, V5, V6
KORS_8LEAD_TO_VCG = np.asarray([
    [-0.13,  0.05, -0.01,  0.14,  0.06,  0.54,  0.38,  0.07],  # X
    [ 0.06, -0.02, -0.05,  0.02, -0.19, -0.23, -0.02,  0.41],  # Y
    [-0.43, -0.06, -0.14, -0.03, -0.11,  0.31,  0.11, -0.23],  # Z
], dtype=np.float64)


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


def compute_nonlocal_cka(K_X: np.ndarray, K_Y: np.ndarray, nonlocal_mask: np.ndarray) -> float:
    k_x = K_X[nonlocal_mask]
    k_y = K_Y[nonlocal_mask]
    norm_x = np.linalg.norm(k_x)
    norm_y = np.linalg.norm(k_y)
    if norm_x < 1e-12 or norm_y < 1e-12:
        return float("nan")
    return float(np.dot(k_x, k_y) / (norm_x * norm_y))


def load_n002_model(ckpt_path: Path):
    source_root = ROOT / "author_code/nefnet_v2"
    model = load_author_model(source_root, ckpt_path, device=DEVICE)
    model.eval()
    return model


def fit_global_pca(model, ptbxl_train_dir: Path, n_records: int = 200, seed: int = 123) -> np.ndarray:
    """Fit a global PCA-3 basis W_PCA in R^{8 x 3} on R2 representations from disjoint PTB-XL train."""
    files = sorted(ptbxl_train_dir.glob("*.pt"))
    rng = np.random.default_rng(seed)
    chosen_files = rng.choice(files, size=min(n_records, len(files)), replace=False)
    
    # Precordial query angles
    precordial_angles = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[6:12]).float().to(DEVICE)
    limb_angles = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[[0, 1, 7]]).float().to(DEVICE).unsqueeze(0)
    
    r2_all = []
    print(f"Fitting global PCA-3 basis on {len(chosen_files)} PTB-XL train records...")
    with torch.no_grad():
        for f in chosen_files:
            data = torch.load(f, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                data = data["ecg"]
            # Shape [12, 5000]. Disk: 0:I, 1:II, 7:V2
            crop = data[:, :2500].float()  # 5 seconds
            # N002 observed normalization on I, II, V2
            inp_3 = crop[[0, 1, 7]]  # [3, 2500]
            m_obs = float(inp_3.min())
            M_obs = float(inp_3.max())
            diff = max(M_obs - m_obs, 1e-6)
            inp_norm = (inp_3 - m_obs) / diff
            x_in = inp_norm.unsqueeze(0).to(DEVICE)  # [1, 3, 2500]
            
            # Predict V1..V6
            recons = []
            for q_idx in range(6):
                q_angle = precordial_angles[q_idx : q_idx + 1].unsqueeze(1)  # [1, 1, 2]
                pred = model(x_in, limb_angles, q_angle)
                recons.append(pred.squeeze(0).squeeze(0).cpu().numpy() * diff + m_obs)
            
            # R2: [I, II, V1..V6]
            i_lead = crop[0].cpu().numpy()
            ii_lead = crop[1].cpu().numpy()
            r2 = np.stack([i_lead, ii_lead] + recons, axis=1)  # [2500, 8]
            # Center per record
            r2_cent = r2 - np.mean(r2, axis=0, keepdims=True)
            # Sample 250 timepoints
            sub_idx = np.linspace(0, 2500 - 1, 250, dtype=int)
            r2_all.append(r2_cent[sub_idx])
            
    r2_concat = np.concatenate(r2_all, axis=0)  # [N * 250, 8]
    # SVD
    U, S, Vt = np.linalg.svd(r2_concat, full_matrices=False)
    W_pca = Vt[:3, :].T  # [8, 3]
    var_exp = (S ** 2) / np.sum(S ** 2)
    print(f"Global PCA Top-3 Variance Explained: {var_exp[:3]} (sum = {np.sum(var_exp[:3]):.4%})")
    return W_pca


def evaluate_emory_global_pca(
    model,
    W_pca: np.ndarray,
    emory_dir: Path,
    n_records: int = 100,
    max_records: int = 100,
) -> dict:
    files = sorted(emory_dir.glob("*.pt"))[:max_records]
    print(f"\nEvaluating Global PCA-3 on {len(files)} Emory records...")
    
    precordial_angles = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[6:12]).float().to(DEVICE)
    limb_angles = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[[0, 1, 7]]).float().to(DEVICE).unsqueeze(0)
    
    cka_r0 = []
    cka_r2 = []
    cka_r2_local_pca = []
    cka_r2_global_pca = []
    cka_r3 = []
    
    # Pre-compute nonlocal mask for 250 samples (100 ms exclusion: Delta_max = 50 samples at 500 Hz)
    # Downsampled to 250 points across 2500 samples -> step = 10 samples (20 ms)
    # So 100 ms corresponds to |i - j| > 5 in the downsampled grid
    N_SUB = 250
    iu_row, iu_col = np.triu_indices(N_SUB, k=1)
    nonlocal_mask = np.abs(iu_row - iu_col) > 5  # > 100 ms
    
    with torch.no_grad():
        for idx, f in enumerate(files):
            data = torch.load(f, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                data = data["ecg"]
            # Disk order on Emory: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
            crop = data[:, :2500].float()
            # GT 8-lead reference: [I, II, V1, V2, V3, V4, V5, V6]
            gt_8lead = crop[[0, 1, 6, 7, 8, 9, 10, 11]].cpu().numpy().T  # [2500, 8]
            
            # Predict R2 via N002 from I, II, V2
            inp_3 = crop[[0, 1, 7]]  # I, II, V2
            m_obs = float(inp_3.min())
            M_obs = float(inp_3.max())
            diff = max(M_obs - m_obs, 1e-6)
            inp_norm = (inp_3 - m_obs) / diff
            x_in = inp_norm.unsqueeze(0).to(DEVICE)
            
            recons = []
            for q_idx in range(6):
                q_angle = precordial_angles[q_idx : q_idx + 1].unsqueeze(1)
                pred = model(x_in, limb_angles, q_angle)
                recons.append(pred.squeeze(0).squeeze(0).cpu().numpy() * diff + m_obs)
                
            i_lead = crop[0].cpu().numpy()
            ii_lead = crop[1].cpu().numpy()
            r2_8lead = np.stack([i_lead, ii_lead] + recons, axis=1)  # [2500, 8]
            
            # Subsample to 250 points
            sub_idx = np.linspace(0, 2500 - 1, N_SUB, dtype=int)
            gt_sub = gt_8lead[sub_idx]
            r2_sub = r2_8lead[sub_idx]
            
            # R0: Kors VCG from GT 8-lead
            r0_sub = gt_sub @ KORS_8LEAD_TO_VCG.T  # [250, 3]
            
            # R3: Kors on R2
            r3_sub = r2_sub @ KORS_8LEAD_TO_VCG.T  # [250, 3]
            
            # Local PCA-3 on R2
            r2_cent = r2_sub - np.mean(r2_sub, axis=0, keepdims=True)
            _, _, Vt = np.linalg.svd(r2_cent, full_matrices=False)
            r2_local_pca_sub = r2_cent @ Vt[:3, :].T  # [250, 3]
            
            # Global PCA-3 on R2
            r2_global_pca_sub = r2_cent @ W_pca  # [250, 3]
            
            # Distance matrices
            D_gt = compute_distance_matrix(gt_sub)
            D_r0 = compute_distance_matrix(r0_sub)
            D_r2 = compute_distance_matrix(r2_sub)
            D_r2_local = compute_distance_matrix(r2_local_pca_sub)
            D_r2_global = compute_distance_matrix(r2_global_pca_sub)
            D_r3 = compute_distance_matrix(r3_sub)
            
            # Kernels
            K_gt = center_kernel(compute_imq_kernel(D_gt))
            K_r0 = center_kernel(compute_imq_kernel(D_r0))
            K_r2 = center_kernel(compute_imq_kernel(D_r2))
            K_r2_local = center_kernel(compute_imq_kernel(D_r2_local))
            K_r2_global = center_kernel(compute_imq_kernel(D_r2_global))
            K_r3 = center_kernel(compute_imq_kernel(D_r3))
            
            # Nonlocal CKA
            cka_r0.append(compute_nonlocal_cka(K_r0, K_gt, nonlocal_mask))
            cka_r2.append(compute_nonlocal_cka(K_r2, K_gt, nonlocal_mask))
            cka_r2_local_pca.append(compute_nonlocal_cka(K_r2_local, K_gt, nonlocal_mask))
            cka_r2_global_pca.append(compute_nonlocal_cka(K_r2_global, K_gt, nonlocal_mask))
            cka_r3.append(compute_nonlocal_cka(K_r3, K_gt, nonlocal_mask))
            
    m_r0, l_r0, u_r0 = bootstrap_ci(np.asarray(cka_r0))
    m_r2, l_r2, u_r2 = bootstrap_ci(np.asarray(cka_r2))
    m_loc, l_loc, u_loc = bootstrap_ci(np.asarray(cka_r2_local_pca))
    m_glob, l_glob, u_glob = bootstrap_ci(np.asarray(cka_r2_global_pca))
    m_r3, l_r3, u_r3 = bootstrap_ci(np.asarray(cka_r3))
    
    diff_glob_r2 = np.asarray(cka_r2_global_pca) - np.asarray(cka_r2)
    m_dg, l_dg, u_dg = bootstrap_ci(diff_glob_r2)
    
    diff_glob_kors = np.asarray(cka_r2_global_pca) - np.asarray(cka_r3)
    m_dk, l_dk, u_dk = bootstrap_ci(diff_glob_kors)
    
    print("\n" + "="*75)
    print("GLOBAL PCA-3 vs LOCAL PCA-3 vs KORS ON EMORY (N=100, Nonlocal > 100ms):")
    print(f"  R0 (Kors VCG GT):        CKA = {m_r0:.4f} [{l_r0:.4f}, {u_r0:.4f}]")
    print(f"  R2 (Nef 8-lead):         CKA = {m_r2:.4f} [{l_r2:.4f}, {u_r2:.4f}]")
    print(f"  R2 Local PCA-3:          CKA = {m_loc:.4f} [{l_loc:.4f}, {u_loc:.4f}]")
    print(f"  R2 Global PCA-3:         CKA = {m_glob:.4f} [{l_glob:.4f}, {u_glob:.4f}]")
    print(f"  R3 (Kors on Nef):        CKA = {m_r3:.4f} [{l_r3:.4f}, {u_r3:.4f}]")
    print(f"  Delta(Global PCA-3 - R2):   {m_dg:+.4f} [{l_dg:+.4f}, {u_dg:+.4f}]")
    print(f"  Delta(Global PCA-3 - Kors): {m_dk:+.4f} [{l_dk:+.4f}, {u_dk:+.4f}]")
    print("="*75 + "\n")
    
    return {
        "cka_r0": {"mean": m_r0, "ci_lower": l_r0, "ci_upper": u_r0},
        "cka_r2": {"mean": m_r2, "ci_lower": l_r2, "ci_upper": u_r2},
        "cka_r2_local_pca": {"mean": m_loc, "ci_lower": l_loc, "ci_upper": u_loc},
        "cka_r2_global_pca": {"mean": m_glob, "ci_lower": l_glob, "ci_upper": u_glob},
        "cka_r3": {"mean": m_r3, "ci_lower": l_r3, "ci_upper": u_r3},
        "delta_global_pca_vs_r2": {"mean": m_dg, "ci_lower": l_dg, "ci_upper": u_dg},
        "delta_global_pca_vs_kors": {"mean": m_dk, "ci_lower": l_dk, "ci_upper": u_dk},
    }


def main():
    exp_root = ROOT
    ckpt_path = exp_root / "results/panobench_3view_variablethird_geovt_seed123/model_final.pt"
    ptbxl_train_dir = Path("/home/mithunmanivannan/data/ptb_xl/tensors/train")
    emory_dir = exp_root / "data/heedb_emory_eval"
    
    model = load_n002_model(ckpt_path)
    W_pca = fit_global_pca(model, ptbxl_train_dir, n_records=200)
    
    results = evaluate_emory_global_pca(model, W_pca, emory_dir)
    
    out_file = exp_root / "results/g002_eval/g002_e_global_pca.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()
