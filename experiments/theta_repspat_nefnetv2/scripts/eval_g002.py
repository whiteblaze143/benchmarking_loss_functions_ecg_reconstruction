#!/usr/bin/env python3
"""G002: Representational State-Space Geometry Benchmark for GeoVT (N002).

Evaluates whether the frozen GeoVT checkpoint preserves the state-space metric geometry
consumed by Theta-repSpat, rather than assessing only pointwise waveform reconstruction.

Evaluates the 4-Arm Comparative Hierarchy:
  R0 = Kors VCG in R^3  (K_Kors @ x_GT)
  R1 = Standard 8-lead ECG in R^8  (x_GT reference)
  R2 = Nef completed 8-lead panorama in R^8  ([I, II, V1_hat, V2_hat, V3, V4_hat, V5_hat, V6_hat])
  R3 = Kors VCG applied to Nef completion in R^3  (K_Kors @ R2)

Metric Hierarchy (Patient-by-Patient / Record-by-Record):
  1. Primary Metric: Spearman rho_D = Spearman(vec(D^Arm), vec(D^GT))
  2. Local Manifold Preservation: k-NN Jaccard overlap J_k(t) for k in {10, 20}
  3. CAHC Partition Agreement: Adjusted Rand Index ARI(Y_CAHC^Arm, Y_CAHC^GT)

Evaluated across two datasets:
  Part A: Clinical 8-Lead State Space on HEEDB WFDB records (real patient ECGs)
  Part B: Dense 44-Channel Panorama on Held-Out PanoBench test records

Output: results/g002_eval/g002_clinical_records.json
        results/g002_eval/g002_panobench_records.json
        results/g002_eval/g002_ablation_triangle.json
        results/g002_eval/g002_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import scipy.io
import scipy.stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
import torch

import sys
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))

from theta_repspat.panorama import CANONICAL_ANGLES_RAD
from theta_repspat.panobench import PANOBENCH_ANGLES_RAD, _upsample2x
from theta_repspat.vendor import load_author_model

# ── Kors Regression Matrix ──────────────────────────────────────────────────
# Shape [3, 8], mapping [I, II, V1, V2, V3, V4, V5, V6] -> [Vx, Vy, Vz]
KORS_MATRIX = np.array([
    [ 0.38,   -0.07,    -0.13,    0.05,   -0.01,    0.14,    0.06,    0.54 ],  # Vx
    [-0.07,    0.93,     0.06,   -0.02,   -0.05,    0.06,   -0.17,    0.13 ],  # Vy
    [-0.11,   -0.23,    -0.43,   -0.06,   -0.14,   -0.20,   -0.11,    0.31 ],  # Vz
], dtype=np.float64)

CANONICAL_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
INPUT_CANONICAL_INDICES = [0, 1, 4]       # I, II, V3
PREDICTED_CANONICAL_INDICES = [2, 3, 5, 6, 7]  # V1, V2, V4, V5, V6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_LEN = 4608
EVAL_SEED = 42


def compute_geometry_metrics(E_arm: np.ndarray, E_gt: np.ndarray,
                             n_subsample: int = 500, n_clusters: int = 7) -> dict:
    """Compute Spearman rho_D, k-NN Jaccard overlaps, and CAHC ARI for one arm vs GT."""
    T = E_gt.shape[0]
    if T > n_subsample:
        # Uniform temporal subsampling to keep distance matrix computation fast and stable
        indices = np.linspace(0, T - 1, n_subsample, dtype=int)
        X_arm = E_arm[indices]
        X_gt = E_gt[indices]
    else:
        X_arm = E_arm
        X_gt = E_gt
        n_subsample = T

    # 1. Pairwise distance matrices
    D_arm = np.linalg.norm(X_arm[:, None, :] - X_arm[None, :, :], axis=-1)
    D_gt = np.linalg.norm(X_gt[:, None, :] - X_gt[None, :, :], axis=-1)

    # Upper triangular vector
    iu = np.triu_indices(n_subsample, k=1)
    vec_arm = D_arm[iu]
    vec_gt = D_gt[iu]

    # Primary Metric: Spearman rho
    rho_d, _ = scipy.stats.spearmanr(vec_arm, vec_gt)
    rho_d = float(rho_d) if np.isfinite(rho_d) else 0.0

    # 2. Local k-NN Jaccard overlap (k=10, k=20)
    def knn_jaccard(k: int) -> float:
        k = min(k, n_subsample - 2)
        jaccards = []
        for t in range(n_subsample):
            d_arm_t = D_arm[t].copy()
            d_gt_t = D_gt[t].copy()
            d_arm_t[t] = np.inf
            d_gt_t[t] = np.inf
            nn_arm = set(np.argpartition(d_arm_t, k)[:k])
            nn_gt = set(np.argpartition(d_gt_t, k)[:k])
            intersection = len(nn_arm & nn_gt)
            union = len(nn_arm | nn_gt)
            jaccards.append(intersection / union if union > 0 else 0.0)
        return float(np.mean(jaccards))

    j10 = knn_jaccard(10)
    j20 = knn_jaccard(20)

    # 3. CAHC (Constrained Agglomerative Hierarchical Clustering) ARI
    try:
        # Temporal connectivity graph (each point connected to immediate temporal neighbors)
        conn = kneighbors_graph(np.arange(n_subsample)[:, None], n_neighbors=2, include_self=False)
        model_arm = AgglomerativeClustering(n_clusters=n_clusters, connectivity=conn)
        model_gt = AgglomerativeClustering(n_clusters=n_clusters, connectivity=conn)
        labels_arm = model_arm.fit_predict(X_arm)
        labels_gt = model_gt.fit_predict(X_gt)
        ari = float(adjusted_rand_score(labels_arm, labels_gt))
    except Exception:
        ari = float("nan")

    return {
        "spearman_rho_d": rho_d,
        "knn_jaccard_k10": j10,
        "knn_jaccard_k20": j20,
        "cahc_ari": ari,
    }


def load_clinical_record(hea_path: Path) -> np.ndarray:
    """Load GE MUSE WFDB record into [8, 4608] canonical array (in physical mV)."""
    dat_path = hea_path.with_suffix(".dat")
    lines = hea_path.read_text().strip().split("\n")
    info = lines[0].split()
    n_leads = int(info[1])
    n_samples = int(info[3])
    raw_lead_names = [l.split()[-1] for l in lines[1: 1 + n_leads]]

    gain = 1000.0
    try:
        gain = float(lines[1].split()[2].split("(")[0].split("/")[0])
    except Exception:
        pass

    raw = np.fromfile(dat_path, dtype=np.int16).reshape((n_samples, n_leads)).T
    data_mv = raw.astype(np.float32) / gain

    # Map to canonical order: [I, II, V1, V2, V3, V4, V5, V6]
    name_to_idx = {name: i for i, name in enumerate(raw_lead_names)}
    canonical = np.zeros((len(CANONICAL_LEADS), n_samples), dtype=np.float32)
    for c_idx, lead_name in enumerate(CANONICAL_LEADS):
        canonical[c_idx] = data_mv[name_to_idx[lead_name]]

    # Center crop to 4608 samples
    if n_samples > EXPECTED_LEN:
        start = (n_samples - EXPECTED_LEN) // 2
        canonical = canonical[:, start: start + EXPECTED_LEN]
    elif n_samples < EXPECTED_LEN:
        canonical = np.pad(canonical, ((0, 0), (0, EXPECTED_LEN - n_samples)), mode="edge")

    return canonical


def evaluate_clinical_record_g002(hea_path: Path, model) -> dict:
    """Evaluate R0, R1, R2, R3 on one clinical record."""
    x_gt_8 = load_clinical_record(hea_path)  # (8, 4608)
    E_gt = x_gt_8.T  # (4608, 8)

    # Observed leads: I, II, V3
    obs_raw = x_gt_8[INPUT_CANONICAL_INDICES]  # (3, 4608)
    lo_obs, hi_obs = float(obs_raw.min()), float(obs_raw.max())
    span_obs = hi_obs - lo_obs if hi_obs > lo_obs else 1.0

    # Normalize observed inputs
    inp_norm = (obs_raw - lo_obs) / span_obs
    inp_t = torch.from_numpy(inp_norm).unsqueeze(0).to(DEVICE)
    inp_angles = torch.from_numpy(
        CANONICAL_ANGLES_RAD[INPUT_CANONICAL_INDICES]).unsqueeze(0).to(DEVICE)

    # Complete panorama to get R2
    completed_8 = np.copy(x_gt_8)
    for q_idx in PREDICTED_CANONICAL_INDICES:
        tgt_angle = torch.from_numpy(CANONICAL_ANGLES_RAD[q_idx]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()
        pred_mv = pred_norm * span_obs + lo_obs
        completed_8[q_idx] = pred_mv

    # Construct the 4 Arms:
    # R1: Standard 8-lead ECG in R^8 (Ground Truth reference)
    E_R1 = E_gt  # (4608, 8)

    # R0: Kors VCG in R^3 from GT
    E_R0 = E_gt @ KORS_MATRIX.T  # (4608, 3)

    # R2: Nef Completed 8-lead panorama in R^8
    E_R2 = completed_8.T  # (4608, 8)

    # R3: Kors VCG applied to Nef Completion in R^3
    E_R3 = E_R2 @ KORS_MATRIX.T  # (4608, 3)

    # Metrics vs GT (E_gt)
    m_R0 = compute_geometry_metrics(E_R0, E_gt)
    m_R2 = compute_geometry_metrics(E_R2, E_gt)
    m_R3 = compute_geometry_metrics(E_R3, E_gt)

    return {
        "record_id": hea_path.stem,
        "R0_KorsVCG_GT": m_R0,
        "R2_NefPanorama": m_R2,
        "R3_KorsVCG_Nef": m_R3,
        "deltas": {
            "R2_minus_R0_rho": m_R2["spearman_rho_d"] - m_R0["spearman_rho_d"],
            "R2_minus_R3_rho": m_R2["spearman_rho_d"] - m_R3["spearman_rho_d"],
            "R3_minus_R0_rho": m_R3["spearman_rho_d"] - m_R0["spearman_rho_d"],
            "R2_minus_R0_j10": m_R2["knn_jaccard_k10"] - m_R0["knn_jaccard_k10"],
            "R2_minus_R3_j10": m_R2["knn_jaccard_k10"] - m_R3["knn_jaccard_k10"],
            "R2_minus_R0_ari": m_R2["cahc_ari"] - m_R0["cahc_ari"],
        },
    }


def evaluate_panobench_record_g002(mat_path: Path, model, rng: np.random.Generator) -> dict:
    """Evaluate dense 44-channel panorama geometry on held-out PanoBench test record."""
    raw = scipy.io.loadmat(mat_path)["Panobench"][:44].astype(np.float32)
    values_raw = _upsample2x(raw)  # (44, 5000)
    crop_start = int(rng.integers(0, 5000 - EXPECTED_LEN + 1))
    values_raw = values_raw[:, crop_start: crop_start + EXPECTED_LEN]

    # Sample random third input
    third_input = int(rng.integers(2, 44))
    input_indices = np.array([0, 1, third_input])
    obs_raw = values_raw[input_indices]
    lo_obs, hi_obs = float(obs_raw.min()), float(obs_raw.max())
    span_obs = hi_obs - lo_obs if hi_obs > lo_obs else 1.0

    inp_norm = (obs_raw - lo_obs) / span_obs
    inp_t = torch.from_numpy(inp_norm).unsqueeze(0).to(DEVICE)
    inp_angles = torch.from_numpy(PANOBENCH_ANGLES_RAD[input_indices]).unsqueeze(0).to(DEVICE)

    # Complete all 44 views
    completed_44 = np.copy(values_raw)
    queries_to_pred = [q for q in range(44) if q not in input_indices]
    for q in queries_to_pred:
        tgt_angle = torch.from_numpy(PANOBENCH_ANGLES_RAD[q]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()
        completed_44[q] = pred_norm * span_obs + lo_obs

    E_gt_44 = values_raw.T      # (4608, 44)
    E_nef_44 = completed_44.T   # (4608, 44)

    metrics_44 = compute_geometry_metrics(E_nef_44, E_gt_44)
    return {
        "record": mat_path.stem,
        "third_view_idx": third_input,
        "dense_44_metrics": metrics_44,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("results/panobench_3view_variablethird_geovt_seed123/model_final.pt"))
    p.add_argument("--clinical-data-root", type=Path,
                   default=Path("/data/mithunmanivannan/heedb_emory/WFDB/2010"))
    p.add_argument("--panobench-test-root", type=Path,
                   default=Path("/data/mithunmanivannan/panobench/test"))
    p.add_argument("--output", type=Path,
                   default=Path("results/g002_eval"))
    p.add_argument("--max-clinical-records", type=int, default=500)
    p.add_argument("--max-pano-records", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output if args.output.is_absolute() else (root_dir / args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else (root_dir / args.checkpoint)
    model = load_author_model(
        root_dir / "author_code" / "nefnet_v2",
        checkpoint_path,
        DEVICE,
    )
    torch.backends.cudnn.enabled = False

    # ── PART A: Clinical 8-Lead State-Space Geometry (HEEDB WFDB) ──────────────
    print(f"\n[Part A] Evaluating Clinical 8-Lead State Space Geometry ({args.max_clinical_records} records)...", flush=True)
    hea_files = []
    with os.scandir(args.clinical_data_root) as it:
        for entry in it:
            if entry.name.endswith(".hea"):
                hea_files.append(Path(entry.path))
                if len(hea_files) >= args.max_clinical_records:
                    break
    hea_files.sort(key=lambda p: p.name)

    clinical_results = []
    for i, hea_p in enumerate(hea_files):
        try:
            res = evaluate_clinical_record_g002(hea_p, model)
            clinical_results.append(res)
        except Exception as e:
            continue
        if (i + 1) % 50 == 0 or i == len(hea_files) - 1:
            mean_rho_r2 = np.mean([r["R2_NefPanorama"]["spearman_rho_d"] for r in clinical_results])
            mean_rho_r0 = np.mean([r["R0_KorsVCG_GT"]["spearman_rho_d"] for r in clinical_results])
            print(f"  {i+1}/{len(hea_files)} records | rho_D(R2)={mean_rho_r2:.4f} | rho_D(R0)={mean_rho_r0:.4f}", flush=True)

    (output_dir / "g002_clinical_records.json").write_text(
        json.dumps(clinical_results, indent=2))

    # ── PART B: Dense 44-Channel Panorama Geometry (Held-Out PanoBench) ───────
    print(f"\n[Part B] Evaluating Dense 44-Channel Panorama Geometry ({args.max_pano_records} records)...", flush=True)
    pano_files = sorted(args.panobench_test_root.glob("*.mat"), key=lambda p: int(p.stem))[: args.max_pano_records]
    rng = np.random.default_rng(EVAL_SEED)

    pano_results = []
    for i, mat_p in enumerate(pano_files):
        try:
            res_p = evaluate_panobench_record_g002(mat_p, model, rng)
            pano_results.append(res_p)
        except Exception as e:
            continue
        if (i + 1) % 25 == 0 or i == len(pano_files) - 1:
            mean_rho_44 = np.mean([r["dense_44_metrics"]["spearman_rho_d"] for r in pano_results])
            mean_j10_44 = np.mean([r["dense_44_metrics"]["knn_jaccard_k10"] for r in pano_results])
            print(f"  {i+1}/{len(pano_files)} pano records | rho_D(44)={mean_rho_44:.4f} | J_10(44)={mean_j10_44:.4f}", flush=True)

    (output_dir / "g002_panobench_records.json").write_text(
        json.dumps(pano_results, indent=2))

    # ── Summary & Ablation Triangle ───────────────────────────────────────────
    def dist_summary(vals):
        arr = np.array([v for v in vals if np.isfinite(v)])
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }

    arms = ["R0_KorsVCG_GT", "R2_NefPanorama", "R3_KorsVCG_Nef"]
    metrics = ["spearman_rho_d", "knn_jaccard_k10", "knn_jaccard_k20", "cahc_ari"]

    clinical_summary = {}
    for arm in arms:
        clinical_summary[arm] = {
            m: dist_summary([r[arm][m] for r in clinical_results]) for m in metrics
        }

    pano_summary = {
        m: dist_summary([r["dense_44_metrics"][m] for r in pano_results]) for m in metrics
    }

    ablation_triangle = {
        "R2_vs_R0 (Nef Panorama vs Kors VCG)": {
            "delta_rho_D": dist_summary([r["deltas"]["R2_minus_R0_rho"] for r in clinical_results]),
            "delta_j10": dist_summary([r["deltas"]["R2_minus_R0_j10"] for r in clinical_results]),
            "delta_ari": dist_summary([r["deltas"]["R2_minus_R0_ari"] for r in clinical_results]),
            "interpretation": "Tests whether 8D panoramic completion preserves geometry better than linear 3D Kors projection.",
        },
        "R2_vs_R3 (Nef Panorama vs Kors-on-Nef)": {
            "delta_rho_D": dist_summary([r["deltas"]["R2_minus_R3_rho"] for r in clinical_results]),
            "delta_j10": dist_summary([r["deltas"]["R2_minus_R3_j10"] for r in clinical_results]),
            "interpretation": "Tests whether the panoramic representation contains non-linear structure destroyed by Kors projection.",
        },
        "R3_vs_R0 (Kors-on-Nef vs True Kors)": {
            "delta_rho_D": dist_summary([r["deltas"]["R3_minus_R0_rho"] for r in clinical_results]),
            "interpretation": "Isolates canonicalization while fixing dimensionality at 3.",
        },
    }

    # Falsifiable decision
    r2_rho = clinical_summary["R2_NefPanorama"]["spearman_rho_d"]["mean"]
    r0_rho = clinical_summary["R0_KorsVCG_GT"]["spearman_rho_d"]["mean"]
    r2_j10 = clinical_summary["R2_NefPanorama"]["knn_jaccard_k10"]["mean"]
    r2_ari = clinical_summary["R2_NefPanorama"]["cahc_ari"]["mean"]
    pano_rho = pano_summary["spearman_rho_d"]["mean"]

    g002_passes = (r2_rho > 0.85 and r2_j10 > 0.35 and pano_rho > 0.90)

    verdict = {
        "G002_status": "PASS" if g002_passes else "FAIL_REQUIRES_N003",
        "primary_metric_r2_spearman_rho": r2_rho,
        "primary_metric_r0_kors_spearman_rho": r0_rho,
        "dense_pano_44_spearman_rho": pano_rho,
        "local_knn_jaccard_k10": r2_j10,
        "cahc_partition_ari": r2_ari,
        "decision_fork": (
            "PROCEED_TO_UNCHANGED_REPSPAT: Nef completion preserves state-space metric geometry sufficiently to feed into repSpat recurrence analysis without retraining."
            if g002_passes else
            "FORK_TO_N003: Nef completion distorts state-space metric geometry. Proceed to N003 two-stage Any-Pairs pretraining (Stage I: standard ECG multi-dataset, Stage II: PanoBench dense angular supervision)."
        ),
    }

    final_summary = {
        "checkpoint": str(args.checkpoint),
        "n_clinical_records": len(clinical_results),
        "n_pano_records": len(pano_results),
        "clinical_arms_summary": clinical_summary,
        "panobench_dense_summary": pano_summary,
        "ablation_triangle": ablation_triangle,
        "verdict": verdict,
    }

    (output_dir / "g002_ablation_triangle.json").write_text(
        json.dumps(ablation_triangle, indent=2))
    (output_dir / "g002_summary.json").write_text(
        json.dumps(final_summary, indent=2))

    print("\n=================== G002 BENCHMARK SUMMARY ===================")
    print(f"  Clinical Records Evaluated:   {len(clinical_results)}")
    print(f"  PanoBench Records Evaluated:  {len(pano_results)}")
    print("-" * 62)
    print(f"  Part A: Clinical 8-Lead State-Space Geometry vs GT:")
    print(f"    R1 (Standard 8-lead GT):   rho_D = 1.0000 | J_10 = 1.0000 | ARI = 1.0000 (Reference)")
    print(f"    R0 (Kors VCG in R^3):      rho_D = {r0_rho:.4f} | J_10 = {clinical_summary['R0_KorsVCG_GT']['knn_jaccard_k10']['mean']:.4f} | ARI = {clinical_summary['R0_KorsVCG_GT']['cahc_ari']['mean']:.4f}")
    print(f"    R2 (Nef Completion in R^8):rho_D = {r2_rho:.4f} | J_10 = {r2_j10:.4f} | ARI = {r2_ari:.4f}")
    print(f"    R3 (Kors on Nef in R^3):   rho_D = {clinical_summary['R3_KorsVCG_Nef']['spearman_rho_d']['mean']:.4f} | J_10 = {clinical_summary['R3_KorsVCG_Nef']['knn_jaccard_k10']['mean']:.4f}")
    print("-" * 62)
    print(f"  Part B: Dense 44-Channel Panorama Geometry (PanoBench):")
    print(f"    rho_D(44 channels):        {pano_rho:.4f} ± {pano_summary['spearman_rho_d']['std']:.4f}")
    print(f"    J_10(44 channels):         {pano_summary['knn_jaccard_k10']['mean']:.4f} ± {pano_summary['knn_jaccard_k10']['std']:.4f}")
    print(f"    CAHC ARI(44 channels):     {pano_summary['cahc_ari']['mean']:.4f} ± {pano_summary['cahc_ari']['std']:.4f}")
    print("-" * 62)
    print(f"  Ablation Triangle:")
    print(f"    Delta rho_D (R2 - R0):     {ablation_triangle['R2_vs_R0 (Nef Panorama vs Kors VCG)']['delta_rho_D']['mean']:+.4f} (Nef 8D vs Kors 3D)")
    print(f"    Delta rho_D (R2 - R3):     {ablation_triangle['R2_vs_R3 (Nef Panorama vs Kors-on-Nef)']['delta_rho_D']['mean']:+.4f} (Structure Kors destroys)")
    print(f"    Delta rho_D (R3 - R0):     {ablation_triangle['R3_vs_R0 (Kors-on-Nef vs True Kors)']['delta_rho_D']['mean']:+.4f} (Kors from Nef vs True)")
    print("-" * 62)
    print(f"  VERDICT: {verdict['G002_status']}")
    print(f"  DECISION: {verdict['decision_fork']}")
    print(f"  Output directory: {output_dir}/\n")


if __name__ == "__main__":
    main()
