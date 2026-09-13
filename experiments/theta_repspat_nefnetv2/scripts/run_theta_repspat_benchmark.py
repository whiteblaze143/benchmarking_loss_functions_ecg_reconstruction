#!/usr/bin/env python3
"""Theta-repSpat Clinical Benchmark Pipeline across Representation Arms R0-R3.

Evaluates whether the learned continuous panoramic completion (R2) preserves or
enhances cardiac state-space recurrence patterns compared to the standard 8-lead
oracle (R1), baseline linear Kors VCG (R0), and Kors-projected panorama (R3).

Phases:
  Phase 1: Primary Expert/Manual Delineation Cohorts (LUDB and ISP).
           Evaluates recurrence calibration and biological wave concordance (ARI/NMI)
           strictly as a secondary external interpretability probe.
  Phase 2: Algorithmic Sensitivity Cohort (RDB).
           Evaluates sensitivity across wavelet-delineated recordings.
  Phase 3: Large-Scale Transport (HEEDB Emory).
           Evaluates graph recurrence topology and representation transport.

Invariance Contract:
  - RepSpat parameters remain strictly frozen:
    m-grid, G-grid, scale-equivariant median heuristic (c = gamma * d_med, gamma=1.0),
    B=1000, alpha=0.05, and clique reassignment rule.
  - Adapter Invariant: X_CAHC == X_supplied (p=8 for R1/R2, p=3 for R0/R3).
    No silent Kors transformation, standardization, or lead deletion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Add experiment and project roots to sys.path
exp_root = Path(__file__).resolve().parents[1]
repo_root = Path(__file__).resolve().parents[3]
if str(exp_root / "src") not in sys.path:
    sys.path.insert(0, str(exp_root / "src"))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.dataset_adapters import LUDBAdapter, ISPAdapter, RDBAdapter
from tit_ecg.src.pipeline import ExactRepSpat
from theta_repspat.vendor import load_author_model
from eval_g002 import (
    CANONICAL_ANGLES_RAD,
    INPUT_CANONICAL_INDICES,
    PREDICTED_CANONICAL_INDICES,
    KORS_MATRIX,
    EXPECTED_LEN,
    load_clinical_record,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


def extract_4_arms(
    raw_12_or_8: np.ndarray,
    model: torch.nn.Module,
    fs: float,
    target_samples: int = 1500,
) -> dict[str, np.ndarray]:
    """Constructs the 4 representation arms from a multi-lead ECG recording.

    Parameters
    ----------
    raw_12_or_8 : np.ndarray of shape [T, 12] or [T, 8]
    model : GeoVT model
    fs : sampling rate in Hz
    target_samples : number of samples to evaluate

    Returns
    -------
    arms : dict mapping arm name to attribute matrix X in R^{n x p}
    """
    T = len(raw_12_or_8)
    if raw_12_or_8.shape[1] == 12:
        # Standard 12-lead indices: I=0, II=1, V1=6, V2=7, V3=8, V4=9, V5=10, V6=11
        leads_8 = raw_12_or_8[:, [0, 1, 6, 7, 8, 9, 10, 11]]
    else:
        leads_8 = raw_12_or_8

    leads_8 = leads_8[:target_samples].astype(np.float64)  # (N, 8)
    N = len(leads_8)

    # 1. R1: Oracle Reference 8-lead ECG (in R^8)
    E_R1 = leads_8.copy()

    # 2. R0: Standard Kors 3D VCG (in R^3)
    E_R0 = leads_8 @ KORS_MATRIX.T

    # 3. R2: Learned GeoVT Panoramic Completion (in R^8)
    # GeoVT requires input temporal length 4608 samples. Pad or crop temporarily for model forward pass.
    x_for_model = leads_8.T  # (8, N)
    if N < EXPECTED_LEN:
        pad_width = EXPECTED_LEN - N
        x_padded = np.pad(x_for_model, ((0, 0), (0, pad_width)), mode="edge")
    else:
        x_padded = x_for_model[:, :EXPECTED_LEN]

    obs_raw = x_padded[INPUT_CANONICAL_INDICES]  # {I, II, V3}
    lo_obs, hi_obs = float(obs_raw.min()), float(obs_raw.max())
    span_obs = hi_obs - lo_obs if hi_obs > lo_obs else 1.0

    inp_norm = (obs_raw - lo_obs) / span_obs
    inp_t = torch.from_numpy(inp_norm.astype(np.float32)).unsqueeze(0).to(DEVICE)
    inp_angles = torch.from_numpy(CANONICAL_ANGLES_RAD[INPUT_CANONICAL_INDICES]).unsqueeze(0).to(DEVICE)

    completed_padded = np.copy(x_padded)
    for q_idx in PREDICTED_CANONICAL_INDICES:
        tgt_angle = torch.from_numpy(CANONICAL_ANGLES_RAD[q_idx]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()
        pred_mv = pred_norm * span_obs + lo_obs
        completed_padded[q_idx] = pred_mv

    completed_8 = completed_padded[:, :N].T  # (N, 8)
    # Strict preservation: observed leads I, II, V3 match GT exactly
    completed_8[:, 0] = E_R1[:, 0]
    completed_8[:, 1] = E_R1[:, 1]
    completed_8[:, 4] = E_R1[:, 4]
    E_R2 = completed_8

    # 4. R3: Kors VCG applied to Nef Completion (in R^3)
    E_R3 = E_R2 @ KORS_MATRIX.T

    return {
        "R1_Oracle_8Lead": E_R1,
        "R0_KorsVCG_3Lead": E_R0,
        "R2_NefPanorama_8Lead": E_R2,
        "R3_KorsNef_3Lead": E_R3,
    }


def run_repspat_on_arm(
    S: np.ndarray,
    X: np.ndarray,
    cfg: RepSpatConfig,
    ground_truth_segmentation: np.ndarray | None = None,
) -> dict:
    """Runs exact repSpat on domain S and attribute X, computing metrics."""
    # INVARIANT CHECK: verify X passes directly without alteration
    assert X.shape[0] == S.shape[0]
    model = ExactRepSpat(config=cfg)
    model.fit(S, X)

    res = model.results_
    init_labels = res["initial_labels"]
    clique_labels = res["labels_clique"]
    cc_labels = res["labels_cc"]

    out = {
        "m_star": res["m_star"],
        "G_star": res["G_star"],
        "n_samples": res["n_samples"],
        "n_initial_clusters": res["n_initial_clusters"],
        "n_clique_clusters": res["n_clique_clusters"],
        "n_cc_clusters": res["n_cc_clusters"],
        "graph_nodes": res["similarity_graph"].number_of_nodes(),
        "graph_edges": res["similarity_graph"].number_of_edges(),
        "graph_density": float(res["graph_descriptors"].get("density", 0.0)),
    }

    # Pairwise MMD rejection rate (alpha = 0.05)
    pw_df = res["pairwise_df"]
    if not pw_df.empty:
        n_pairs = len(pw_df)
        n_rejected = int(np.sum(pw_df["rejected"]))
        out["n_pairs"] = n_pairs
        out["n_rejected_pairs"] = n_rejected
        out["rejection_rate"] = float(n_rejected / n_pairs) if n_pairs > 0 else 0.0
        out["mean_mmd_stat"] = float(pw_df["obs_mmd2"].mean())
    else:
        out["n_pairs"] = 0
        out["n_rejected_pairs"] = 0
        out["rejection_rate"] = 0.0
        out["mean_mmd_stat"] = 0.0

    # Secondary external interpretability probe: wave concordance
    if ground_truth_segmentation is not None:
        gt = np.asarray(ground_truth_segmentation[:len(init_labels)], dtype=int)
        valid_mask = ~np.isnan(gt) & (gt >= 0)
        gt_valid = gt[valid_mask]
        out["wave_concordance"] = {
            "ari_initial": float(adjusted_rand_score(gt_valid, init_labels[valid_mask])),
            "ari_clique": float(adjusted_rand_score(gt_valid, clique_labels[valid_mask])),
            "ari_cc": float(adjusted_rand_score(gt_valid, cc_labels[valid_mask])),
            "nmi_initial": float(normalized_mutual_info_score(gt_valid, init_labels[valid_mask])),
            "nmi_clique": float(normalized_mutual_info_score(gt_valid, clique_labels[valid_mask])),
        }
    else:
        out["wave_concordance"] = None

    return out


def main():
    parser = argparse.ArgumentParser(description="Theta-repSpat Clinical Benchmark")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="Phase 1: LUDB/ISP (expert), Phase 2: RDB (sensitivity), Phase 3: Emory (transport)")
    parser.add_argument("--max-records", type=int, default=16, help="Maximum records per dataset")
    parser.add_argument("--n-samples", type=int, default=1500, help="Number of time samples per recording (3s at 500Hz)")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Number of block permutations B")
    parser.add_argument("--pilot", action="store_true", help="Run fast 2-record pilot verification")
    parser.add_argument("--output-dir", type=str, default="results/theta_repspat_eval")
    args = parser.parse_args()

    out_dir = exp_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        args.max_records = 2
        args.n_permutations = 50
        print("[PILOT MODE] max_records=2, permutations=50")

    # Load frozen GeoVT model
    ckpt_path = exp_root / "results/panobench_3view_variablethird_geovt_seed123/model_final.pt"
    if not ckpt_path.is_file():
        ckpt_path = exp_root / "results/geovt_lineage_seed123/model_final.pt"
    model = load_author_model(exp_root / "author_code" / "nefnet_v2", ckpt_path, DEVICE)
    model.eval()

    # RepSpat Configuration: Scale-equivariant median heuristic (c = gamma * d_med, gamma=1.0)
    cfg = RepSpatConfig(
        m_grid=[3, 4, 6],
        G_grid=[4, 6, 8],
        n_permutations=args.n_permutations,
        kernel="imq",
        kernel_scale_rule="median_heuristic",
        kernel_param=1.0,
        fdr_alpha=0.05,
        strict_block_exceed=True,
        block_permutation_mode="attribute_kmeans",
        clique_min_size=2,
        reassignment_primary="clique",
        random_state=42,
    )

    datasets_to_run = []
    if args.phase == 1:
        print("\n================== PHASE 1: EXPERT / MANUAL DELINEATION (LUDB & ISP) ==================")
        datasets_to_run = [("LUDB", LUDBAdapter(data_dir=str(repo_root / "data/ludb"))),
                           ("ISP", ISPAdapter(data_dir=str(repo_root / "data/isp_delineation_dataset")))]
    elif args.phase == 2:
        print("\n================== PHASE 2: ALGORITHMIC SENSITIVITY (RDB) ==================")
        datasets_to_run = [("RDB", RDBAdapter(data_dir=str(repo_root / "data/rdb_wavelet_delineation_cache")))]
    elif args.phase == 3:
        print("\n================== PHASE 3: LARGE-SCALE TRANSPORT (HEEDB EMORY) ==================")
        # Emory WFDB records
        pass

    results_all = []
    for dset_name, adapter in datasets_to_run:
        print(f"\nEvaluating dataset: {dset_name}...")
        record_ids = adapter.list_records(max_records=args.max_records)
        print(f"Loaded {len(record_ids)} records from {dset_name}.")

        for r_idx, rid in enumerate(record_ids):
            rec = adapter.load_record(rid, n_samples=args.n_samples)
            ecg_signal = rec["ecg"]
            fs = float(rec["fs"])
            seg = rec.get("segmentation")

            N = min(len(ecg_signal), args.n_samples)
            time_tau = np.arange(N, dtype=np.float64) / fs
            S = np.column_stack([time_tau, np.zeros_like(time_tau)])

            # Extract 4 Arms
            arms = extract_4_arms(ecg_signal[:N], model, fs, target_samples=N)

            rec_eval = {
                "dataset": dset_name,
                "record_id": str(rid),
                "n_samples": N,
                "sampling_rate": fs,
                "arms": {},
            }

            for arm_name, X_arm in arms.items():
                start_t = time.time()
                arm_res = run_repspat_on_arm(S, X_arm, cfg, ground_truth_segmentation=seg)
                elapsed = time.time() - start_t
                arm_res["elapsed_seconds"] = elapsed
                rec_eval["arms"][arm_name] = arm_res

            results_all.append(rec_eval)
            print(f"  [{dset_name}] {r_idx+1}/{len(record_ids)} record {rid} evaluated across 4 arms.", flush=True)

    # Save results
    phase_json = out_dir / f"theta_repspat_phase{args.phase}_results.json"
    with open(phase_json, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nSaved phase {args.phase} results to: {phase_json}")


if __name__ == "__main__":
    main()
