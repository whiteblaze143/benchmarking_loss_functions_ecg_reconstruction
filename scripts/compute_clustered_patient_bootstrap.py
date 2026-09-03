#!/usr/bin/env python3
"""
Clustered Patient-Level Bootstrap Analysis for Stage A/A.1 Models (D3, D5, D8).
Addresses repeated ECGs per patient in the PTB-XL validation cohort (Fold 9).

Performs:
1. Patient-ID audit: N_ECG vs N_patient and record-to-patient clustering.
2. Clustered patient bootstrap (10,000 resamples):
   - Sample patient IDs with replacement (M = 1,942 unique patients).
   - Include all ECG records belonging to each sampled patient.
   - Compute metric difference for each contrast.
3. Full Frontal-Lead Decomposition for D3 vs D5:
   - Delta_II (independent frontal plane lead when Lead I is observed)
   - Delta_dependent_limb = mean(Delta_III, Delta_aVR, Delta_aVL, Delta_aVF)
   - Delta_frontal = mean(Delta_II, Delta_III, Delta_aVR, Delta_aVL, Delta_aVF)
   - Delta_chest = mean(V1..V6)
   - Delta_precordial_transition = mean(V_{k+1} - V_k)
   - Delta_missing_11 = mean(all 11 missing leads)
4. D3 vs D8 Clustered Comparison:
   - Delta_missing_11
   - Delta_chest
   - Delta_precordial_transition
5. Comparison between Unclustered (Record-Level) vs Clustered (Patient-Level) CIs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unified_latents.engineering.models.three_d_theta_reconstruction import (
    LEAD_NAMES,
    ThreeDThetaECGAIM,
)
from scripts.train_3dtheta_ablation import CELL_CONFIGS, SimplePTBXLDataset


class IndexedValDataset(Dataset):
    def __init__(self, folder: Path):
        self.files = sorted(list(folder.glob("*.pt")))
        self.ecg_ids = [int(p.stem) for p in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "waveform" in obj:
            waveform = torch.as_tensor(obj["waveform"], dtype=torch.float32)
        elif isinstance(obj, torch.Tensor):
            waveform = obj.float()
        else:
            raise TypeError(f"Unexpected type in {path}: {type(obj)}")
        return waveform[..., :5000], idx, self.ecg_ids[idx]


def load_model(cell: str, seed: int, device: torch.device) -> ThreeDThetaECGAIM:
    cfg = CELL_CONFIGS[cell]
    model = ThreeDThetaECGAIM(
        code_mode=cfg["code_mode"],
        fusion=cfg["fusion"],
        width=768,
        encoder_depth=8,
        decoder_depth=4,
        heads=12,
    ).to(device)

    ckpt_path = ROOT / f"refine-logs/convergence_10e/runs/{cell}_s{seed}_l0/best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {
        k: v.float() if v.is_floating_point() else v
        for k, v in payload["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model.eval()
    return model


def compute_per_record_lead_pearsons(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Computes per-record Pearson r for all 12 leads and 5 precordial transitions."""
    model.eval()
    all_pearsons = []
    all_trans_pearsons = []
    all_l1 = []

    with torch.inference_mode():
        for batch in loader:
            waveforms, _, _ = batch
            waveforms = waveforms.to(device)  # [B, 12, 5000]
            x_src = waveforms[:, 0:1, :]      # Lead I observed

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(x_src, obs_lead_idx=0)["y_pred"].float()

            # Center along time dimension
            p_c = preds - preds.mean(dim=-1, keepdim=True)
            t_c = waveforms - waveforms.mean(dim=-1, keepdim=True)
            cov = (p_c * t_c).sum(dim=-1)
            std_p = torch.sqrt((p_c**2).sum(dim=-1).clamp_min(1e-8))
            std_t = torch.sqrt((t_c**2).sum(dim=-1).clamp_min(1e-8))
            r_12 = (cov / (std_p * std_t)).clamp(-1.0, 1.0)  # [B, 12]

            # Precordial transition: delta V_k = V_{k+1} - V_k for V1..V6 (indices 6..11)
            chest_idx = [6, 7, 8, 9, 10, 11]
            pv = preds[:, chest_idx, :]
            tv = waveforms[:, chest_idx, :]
            dp = pv[:, 1:, :] - pv[:, :-1, :]  # [B, 5, 5000]
            dt_ = tv[:, 1:, :] - tv[:, :-1, :]
            dpc = dp - dp.mean(dim=-1, keepdim=True)
            dtc = dt_ - dt_.mean(dim=-1, keepdim=True)
            cov_d = (dpc * dtc).sum(dim=-1)
            std_dp = torch.sqrt((dpc**2).sum(dim=-1).clamp_min(1e-8))
            std_dt = torch.sqrt((dtc**2).sum(dim=-1).clamp_min(1e-8))
            r_trans = (cov_d / (std_dp * std_dt)).clamp(-1.0, 1.0)  # [B, 5]

            l1 = torch.abs(preds[:, 1:, :] - waveforms[:, 1:, :]).mean(dim=(-2, -1))

            all_pearsons.append(r_12.cpu().numpy())
            all_trans_pearsons.append(r_trans.cpu().numpy())
            all_l1.append(l1.cpu().numpy())

    return {
        "lead_r": np.concatenate(all_pearsons, axis=0),        # [N_ECG, 12]
        "trans_r": np.concatenate(all_trans_pearsons, axis=0),  # [N_ECG, 5]
        "l1_missing": np.concatenate(all_l1, axis=0),          # [N_ECG]
    }


def clustered_bootstrap_ci(
    delta_record: np.ndarray,
    patient_ids: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Clustered bootstrap: samples unique patient IDs with replacement.
    Includes all ECG records belonging to each sampled patient.
    """
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    n_records = len(delta_record)

    # Pre-aggregate record delta sums and counts per unique patient
    patient_to_idx = {pid: i for i, pid in enumerate(unique_patients)}
    record_patient_idx = np.array([patient_to_idx[pid] for pid in patient_ids])

    # Pre-compute sums and counts per patient
    patient_sums = np.bincount(record_patient_idx, weights=delta_record, minlength=n_patients)
    patient_counts = np.bincount(record_patient_idx, minlength=n_patients)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        # Sample patient IDs with replacement
        sampled_pts = rng.integers(0, n_patients, size=n_patients)
        # Vectorized weighted aggregate across sampled clusters
        sampled_counts = np.bincount(sampled_pts, minlength=n_patients)
        sum_delta = np.dot(patient_sums, sampled_counts)
        tot_records = np.dot(patient_counts, sampled_counts)
        boot_means[b] = sum_delta / tot_records

    mean_val = float(np.mean(delta_record))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    std_err = float(np.std(boot_means, ddof=1))

    # Two-tailed empirical p-value for H0: delta == 0
    p_pos = np.mean(boot_means >= 0.0)
    p_neg = np.mean(boot_means <= 0.0)
    p_value = float(2.0 * min(p_pos, p_neg))
    p_value = min(1.0, max(0.0, p_value))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err,
        "p_value": p_value,
    }


def unclustered_bootstrap_ci(
    delta_record: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Standard unclustered (record-level) bootstrap for comparison."""
    rng = np.random.default_rng(seed)
    n = len(delta_record)
    boot_means = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(delta_record[idx])

    mean_val = float(np.mean(delta_record))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    std_err = float(np.std(boot_means, ddof=1))

    p_pos = np.mean(boot_means >= 0.0)
    p_neg = np.mean(boot_means <= 0.0)
    p_value = float(2.0 * min(p_pos, p_neg))
    p_value = min(1.0, max(0.0, p_value))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err,
        "p_value": p_value,
    }


def main():
    print("=" * 80)
    print("  CLUSTERED PATIENT-LEVEL BOOTSTRAP & FRONTAL DECOMPOSITION")
    print("=" * 80)

    # 1. Audit PTB-XL Folds and Patient IDs
    meta_path = ROOT / "data/ptb_xl/ptbxl_database.csv"
    df_meta = pd.read_csv(meta_path).set_index("ecg_id")

    val_ds = IndexedValDataset(ROOT / "data/ptb_xl/tensors/val")
    val_ecg_ids = val_ds.ecg_ids
    val_df = df_meta.loc[val_ecg_ids]
    val_patient_ids = val_df["patient_id"].values

    n_ecg = len(val_ecg_ids)
    n_patient = len(np.unique(val_patient_ids))

    print(f"\n[1] Patient-ID Audit on Validation Fold 9:")
    print(f"    Total ECG Records:    N_ECG = {n_ecg}")
    print(f"    Unique Patient IDs:   N_patient = {n_patient}")
    val_counts = pd.Series(val_patient_ids).value_counts()
    print(f"    Patients with 1 ECG:  {(val_counts == 1).sum()} ({(val_counts == 1).sum() / n_patient * 100:.1f}%)")
    print(f"    Patients with >1 ECG: {(val_counts > 1).sum()} ({(val_counts > 1).sum() / n_patient * 100:.1f}%)")
    print(f"    Max ECGs for 1 pt:    {val_counts.max()}")

    # 2. Extract per-record metrics for D3, D5, D8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2] Computing Per-Record Predictions on {device}...")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    print("    Loading D3 (true theta, mul)...")
    m_d3 = load_model("D3_theta_mul_l1", seed=42, device=device)
    res_d3 = compute_per_record_lead_pearsons(m_d3, val_loader, device)

    print("    Loading D5 (permuted theta, mul)...")
    m_d5 = load_model("D5_permuted_theta_mul_l1", seed=42, device=device)
    res_d5 = compute_per_record_lead_pearsons(m_d5, val_loader, device)

    print("    Loading D8 (random 12-D, mul)...")
    m_d8 = load_model("D8_random12_mul_l1", seed=42, device=device)
    res_d8 = compute_per_record_lead_pearsons(m_d8, val_loader, device)

    # Lead Indices:
    # 0: I, 1: II, 2: III, 3: aVR, 4: aVL, 5: aVF
    # 6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
    idx_ii = 1
    idx_dep_limb = [2, 3, 4, 5]   # III, aVR, aVL, aVF
    idx_frontal = [1, 2, 3, 4, 5]  # II, III, aVR, aVL, aVF
    idx_chest = [6, 7, 8, 9, 10, 11]
    idx_missing = list(range(1, 12))

    # 3. D3 vs D5 Metric Differences
    diff_lead_d3_d5 = res_d3["lead_r"] - res_d5["lead_r"]          # [N_ECG, 12]
    diff_trans_d3_d5 = res_d3["trans_r"] - res_d5["trans_r"]      # [N_ECG, 5]

    delta_ii = diff_lead_d3_d5[:, idx_ii]
    delta_dep_limb = diff_lead_d3_d5[:, idx_dep_limb].mean(axis=1)
    delta_frontal = diff_lead_d3_d5[:, idx_frontal].mean(axis=1)
    delta_chest = diff_lead_d3_d5[:, idx_chest].mean(axis=1)
    delta_trans = diff_trans_d3_d5.mean(axis=1)
    delta_missing = diff_lead_d3_d5[:, idx_missing].mean(axis=1)

    # 4. D3 vs D8 Metric Differences
    diff_lead_d3_d8 = res_d3["lead_r"] - res_d8["lead_r"]
    diff_trans_d3_d8 = res_d3["trans_r"] - res_d8["trans_r"]

    delta_missing_d8 = diff_lead_d3_d8[:, idx_missing].mean(axis=1)
    delta_chest_d8 = diff_lead_d3_d8[:, idx_chest].mean(axis=1)
    delta_trans_d8 = diff_trans_d3_d8.mean(axis=1)

    # 5. Run Clustered vs Unclustered Bootstrap (10,000 resamples)
    print("\n[3] Running 10,000 Clustered Patient-Level Bootstrap Resamples...")
    metrics_to_run = {
        "D3 - D5: Delta_II (Independent Frontal Lead)": delta_ii,
        "D3 - D5: Delta_dependent_limb (III, aVR, aVL, aVF)": delta_dep_limb,
        "D3 - D5: Delta_frontal (All 5 Frontal Leads)": delta_frontal,
        "D3 - D5: Delta_chest (Precordial V1-V6)": delta_chest,
        "D3 - D5: Precordial Transition Delta V_k": delta_trans,
        "D3 - D5: All Missing 11 Leads": delta_missing,
        "D3 - D8: All Missing 11 Leads": delta_missing_d8,
        "D3 - D8: Delta_chest (Precordial V1-V6)": delta_chest_d8,
        "D3 - D8: Precordial Transition Delta V_k": delta_trans_d8,
    }

    results = {}
    print("\n" + "=" * 105)
    print(f"{'Contrast / Metric':<50} | {'Mean Delta':<10} | {'Clustered 95% CI':<24} | {'p-val':<7} | {'Unclustered 95% CI':<24}")
    print("=" * 105)

    for name, delta_arr in metrics_to_run.items():
        clustered = clustered_bootstrap_ci(delta_arr, val_patient_ids, n_boot=10000, seed=42)
        unclustered = unclustered_bootstrap_ci(delta_arr, n_boot=10000, seed=42)

        results[name] = {
            "clustered": clustered,
            "unclustered": unclustered,
        }

        c_ci_str = f"[{clustered['ci_lower']:+.6f}, {clustered['ci_upper']:+.6f}]"
        u_ci_str = f"[{unclustered['ci_lower']:+.6f}, {unclustered['ci_upper']:+.6f}]"
        print(f"{name:<50} | {clustered['mean']:+.6f} | {c_ci_str:<24} | {clustered['p_value']:<7.4f} | {u_ci_str:<24}")

    print("=" * 105)

    # Save to JSON
    out_json = ROOT / "results/clustered_patient_bootstrap_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nFull results exported to: {out_json}")


if __name__ == "__main__":
    main()
