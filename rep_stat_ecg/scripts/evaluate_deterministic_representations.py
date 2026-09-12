"""Milestone M4A: Full-ECG Deterministic Representation Qualification (R0 to R4).

Implements PRD Sections 8, 9, 10, 12, 13, 14, 49:
- Extract deterministic ECG representations:
    R0: Physical VCG field statistics (v(t))
    R1: Spatial-domain occupancy representation (64-domain histogram)
    R2: Hilbert functional statistics (h_H(t) = z_{G(v(t))}^H)
    R3: Coupled physical + functional representation [R0, R2]
    R4: Full coupled representation [R0, R2, R_residual]
- Clinical diagnostic probing on PTB-XL (trained on Folds 1-7, tested on Fold 8).
- Primary question: Does R3 > R0? (Deterministic Kill Gate).
- Secondary question: Does R4 > R3?
- Patient is the inferential unit: patient-paired bootstrap CIs for Delta AUROC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import wfdb

from rep_stat_ecg.src.vcg.lift import VCGLift
from experiments.nullvcg_observation_constrained.geometry import (
    LEAD_DIRECTIONS,
    independent_from_standard,
)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class DeterministicECGFeatureExtractor:
    """Extracts R0, R1, R2, R3, R4 deterministic feature vectors from 12-lead ECG."""

    def __init__(
        self,
        centroids: np.ndarray,
        median: np.ndarray,
        mad: np.ndarray,
        Z_H: np.ndarray,
        lead_directions: np.ndarray,
    ):
        self.centroids = centroids  # (64, 3)
        self.median = median        # (3,)
        self.mad = mad              # (3,)
        self.Z_H = Z_H              # (64, 9)
        self.A = lead_directions    # (8, 3)
        
        # Standardize centroids
        self.c_std = (self.centroids - self.median) / self.mad  # (64, 3)
        self.c_norm_sq = np.sum(self.c_std ** 2, axis=1, keepdims=True).T  # (1, 64)
        
        # SVD pseudoinverse of A for v(t) = A^+ E_ind(t)
        self.A_pinv = np.linalg.pinv(self.A)  # (3, 8)

    def extract_from_ecg(self, E_12L: np.ndarray, fs: float = 500.0) -> dict[str, np.ndarray]:
        """Extract R0, R1, R2, R3, R4 from E_12L of shape (12, T)."""
        # Independent leads: I, II, V1-V6 (indices 0, 1, 6, 7, 8, 9, 10, 11)
        ind_idx = [0, 1, 6, 7, 8, 9, 10, 11]
        E_ind = E_12L[ind_idx, :]  # (8, T)
        T = E_ind.shape[1]

        # 1. Deterministic Physical Trajectory: v(t) = A^+ E_ind(t)
        v = np.dot(self.A_pinv, E_ind)  # (3, T)

        # Dipole reconstruction and residual
        E_dipole = np.dot(self.A, v)    # (8, T)
        r = E_ind - E_dipole            # (8, T)

        # 2. Functional Hilbert Trajectory: h_H(t) = z_{G(v(t))}^H
        v_std = (v.T - self.median) / self.mad  # (T, 3)
        v_norm_sq = np.sum(v_std ** 2, axis=1, keepdims=True)  # (T, 1)
        dists_sq = v_norm_sq - 2.0 * np.dot(v_std, self.c_std.T) + self.c_norm_sq  # (T, 64)
        domain_assignments = np.argmin(dists_sq, axis=1)  # (T,)
        h_H = self.Z_H[domain_assignments, :]  # (T, 9)

        # ----------------------------------------------------------------------
        # Block 1: Physical Field Statistics (from v(t)) -> R0
        # ----------------------------------------------------------------------
        mean_v = np.mean(v, axis=1)  # (3,)
        cov_v = np.cov(v)             # (3, 3)
        cov_v_triu = cov_v[np.triu_indices(3)]  # (6,)
        range_v = np.ptp(v, axis=1)   # (3,)

        norm_v = np.linalg.norm(v, axis=0)  # (T,)
        norm_stats = np.array([
            np.mean(norm_v),
            np.max(norm_v),
            np.std(norm_v),
            np.percentile(norm_v, 95),
        ])  # (4,)

        # Path length
        dv = np.diff(v, axis=1)
        path_length_v = np.sum(np.linalg.norm(dv, axis=0)) / float(T)  # (1,)

        # Velocity & Acceleration
        dt = 1.0 / fs
        vel_v = np.linalg.norm(dv / dt, axis=0)
        vel_stats = np.array([np.mean(vel_v), np.percentile(vel_v, 95)])  # (2,)
        
        d2v = np.diff(dv, axis=1) / (dt ** 2)
        acc_stats = np.array([np.mean(np.linalg.norm(d2v, axis=0))])      # (1,)

        # Field inertia / planarity eigenvalues
        eigvals_v, _ = np.linalg.eigh(cov_v)
        inertia_eigs = np.sort(eigvals_v)[::-1]  # (3,)

        b1_features = np.concatenate([
            mean_v, cov_v_triu, range_v, norm_stats, [path_length_v],
            vel_stats, acc_stats, inertia_eigs
        ])  # 23D

        # ----------------------------------------------------------------------
        # Block 2: Spatial Domain Occupancy Histogram -> R1
        # ----------------------------------------------------------------------
        domain_counts = np.bincount(domain_assignments, minlength=64)
        domain_hist = domain_counts / float(T)  # (64,)
        occ_entropy = -np.sum(domain_hist * np.log(domain_hist + 1e-12))  # (1,)
        n_unique_domains = np.sum(domain_counts > 0)                      # (1,)
        
        # Dwell time
        diff_domains = np.diff(domain_assignments) != 0
        n_transitions = np.sum(diff_domains)
        mean_dwell = float(T) / float(max(n_transitions, 1))

        r1_features = np.concatenate([domain_hist, [occ_entropy, n_unique_domains, mean_dwell]])  # 67D

        # ----------------------------------------------------------------------
        # Block 3: Hilbert Functional Statistics (from h_H(t)) -> R2
        # ----------------------------------------------------------------------
        mean_h = np.mean(h_H, axis=0)  # (9,)
        cov_h = np.cov(h_H.T)          # (9, 9)
        cov_h_triu = cov_h[np.triu_indices(9)]  # (45,)
        range_h = np.ptp(h_H, axis=0)  # (9,)

        dh = np.diff(h_H, axis=0)
        path_length_h = np.sum(np.linalg.norm(dh, axis=1)) / float(T)  # (1,)

        # Transition distance in Hilbert space
        if n_transitions > 0:
            h_transitions = np.linalg.norm(dh[diff_domains, :], axis=1)
            mean_trans_dist = float(np.mean(h_transitions))
        else:
            mean_trans_dist = 0.0

        b2_features = np.concatenate([
            mean_h, cov_h_triu, range_h, [path_length_h, occ_entropy, mean_dwell, mean_trans_dist]
        ])  # 68D

        # ----------------------------------------------------------------------
        # Block 4: Physical-Functional Coupling
        # ----------------------------------------------------------------------
        # Cross-covariance between top 3 v and top 3 h
        v_centered = v - np.mean(v, axis=1, keepdims=True)
        h_centered = h_H[:, :3].T - np.mean(h_H[:, :3].T, axis=1, keepdims=True)
        cross_cov = np.dot(v_centered, h_centered.T) / float(T)  # (3, 3)
        cross_cov_flat = cross_cov.flatten()                      # (9,)
        norm_vh_corr = float(np.corrcoef(norm_v, np.linalg.norm(h_H, axis=1))[0, 1])
        if np.isnan(norm_vh_corr):
            norm_vh_corr = 0.0

        b3_features = np.concatenate([cross_cov_flat, [norm_vh_corr]])  # 10D

        # ----------------------------------------------------------------------
        # Block 5: Residual ECG Statistics (from r(t))
        # ----------------------------------------------------------------------
        lead_rms = np.sqrt(np.mean(r ** 2, axis=1))  # (8,)
        
        # Limb vs precordial residual energy ratio
        limb_rms_sq = np.sum(lead_rms[:2] ** 2)
        precord_rms_sq = np.sum(lead_rms[2:] ** 2)
        regional_ratio = precord_rms_sq / (limb_rms_sq + 1e-12)  # (1,)

        # Leadwise residual fraction
        lead_signal_rms = np.sqrt(np.mean(E_ind ** 2, axis=1)) + 1e-12
        residual_frac = lead_rms / lead_signal_rms  # (8,)

        # High-frequency residual power via diff
        dr = np.diff(r, axis=1)
        hf_power = np.mean(dr ** 2, axis=1)[[0, 1, 3, 6]]  # Leads I, II, V2, V5 (4,)

        b4_features = np.concatenate([lead_rms, [regional_ratio], residual_frac, hf_power])  # 21D

        # Assemble Baselines
        r0 = b1_features                                    # 23D
        r1 = r1_features                                    # 67D
        r2 = b2_features                                    # 68D
        r3 = np.concatenate([r0, r2, b3_features])          # 23 + 68 + 10 = 101D
        r4 = np.concatenate([r3, b4_features])              # 101 + 21 = 122D

        return {
            "R0": r0,
            "R1": r1,
            "R2": r2,
            "R3": r3,
            "R4": r4,
        }


def bootstrap_patient_delta_auroc(
    y_true: np.ndarray,
    y_score_A: np.ndarray,
    y_score_B: np.ndarray,
    patient_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute patient-paired bootstrap difference Delta AUROC = AUROC(B) - AUROC(A)."""
    unique_pids = np.unique(patient_ids)
    n_pids = len(unique_pids)
    
    # Map patient_id to indices
    pid_to_indices = {}
    for idx, pid in enumerate(patient_ids):
        pid_to_indices.setdefault(pid, []).append(idx)

    rng = np.random.RandomState(seed)
    deltas = []

    for _ in range(n_boot):
        boot_pids = rng.choice(unique_pids, size=n_pids, replace=True)
        boot_idx = []
        for p in boot_pids:
            boot_idx.extend(pid_to_indices[p])
        boot_idx = np.array(boot_idx)

        y_b = y_true[boot_idx]
        if len(np.unique(y_b)) < 2:
            continue

        auc_A = roc_auc_score(y_b, y_score_A[boot_idx])
        auc_B = roc_auc_score(y_b, y_score_B[boot_idx])
        deltas.append(auc_B - auc_A)

    if not deltas:
        return 0.0, 0.0, 0.0

    mean_delta = float(np.mean(deltas))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return mean_delta, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(description="Milestone M4A: Deterministic Representation Qualification")
    parser.add_argument("--data-dir", default="data/ptb_xl")
    parser.add_argument("--qvcg-dir", default="refine-logs/qvcg")
    parser.add_argument("--out-dir", default="refine-logs/qvcg/deterministic_representation")
    parser.add_argument("--max-train-records", type=int, default=3000)
    parser.add_argument("--max-val-records", type=int, default=2185)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    qvcg_dir = Path(args.qvcg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MILESTONE M4A: FULL-ECG DETERMINISTIC REPRESENTATION QUALIFICATION")
    print("=" * 80)

    # 1. Load frozen atlas components
    std_json_path = qvcg_dir / "VCG_COORDINATE_STANDARDIZER.json"
    coords_path = qvcg_dir / "hilbert_atlas" / "HILBERT_ATLAS_COORDS.parquet"

    df_coords = pd.read_parquet(coords_path)
    centroids = df_coords[["centroid_vx", "centroid_vy", "centroid_vz"]].to_numpy(float)

    with open(std_json_path) as f:
        std_p = json.load(f)
    median = np.array(std_p["median"])
    mad = np.array(std_p["mad"])

    Z_H = df_coords[[f"z_{i}" for i in range(1, 10)]].to_numpy()  # (64, 9)

    lead_dirs = LEAD_DIRECTIONS.cpu().numpy()  # (8, 3)

    extractor = DeterministicECGFeatureExtractor(centroids, median, mad, Z_H, lead_dirs)
    print(f"Initialized Deterministic Feature Extractor (64 domains, 9-D Hilbert coords, 8-lead VCG geometry).")

    # 2. Load dataset and select cohorts
    db_path = data_dir / "ptbxl_database.csv"
    df = pd.read_csv(db_path)
    df["ecg_id"] = df["ecg_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)

    # Train cohort: Folds 1-7 (patient-balanced)
    df_train = df[df["strat_fold"].isin(range(1, 8))].copy().reset_index(drop=True)
    rng = np.random.RandomState(args.seed)
    train_pids = rng.choice(df_train["patient_id"].unique(), size=min(args.max_train_records, df_train["patient_id"].nunique()), replace=False)
    df_train_sub = df_train[df_train["patient_id"].isin(train_pids)].drop_duplicates("patient_id").reset_index(drop=True)

    # Validation cohort: Fold 8 (all records up to limit)
    df_val = df[df["strat_fold"] == 8].copy().reset_index(drop=True)
    if len(df_val) > args.max_val_records:
        df_val = df_val.iloc[:args.max_val_records].copy()

    print(f"Extraction cohorts: Train (Folds 1-7): {len(df_train_sub)} records; Validation (Fold 8): {len(df_val)} records.")

    # 3. Feature Extraction Loop
    def extract_features_for_df(df_records: pd.DataFrame, desc: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
        features = {"R0": [], "R1": [], "R2": [], "R3": [], "R4": []}
        valid_indices = []

        for idx, row in tqdm(df_records.iterrows(), total=len(df_records), desc=desc):
            rec_path = data_dir / row["filename_hr"]
            try:
                sig, fields = wfdb.rdsamp(str(rec_path))
            except Exception:
                continue

            fs = float(fields["fs"])
            E_12L = sig.T  # (12, T)

            ecg_feats = extractor.extract_from_ecg(E_12L, fs=fs)
            for k in features:
                features[k].append(ecg_feats[k])
            valid_indices.append(idx)

        res_arrays = {k: np.array(features[k], dtype=np.float32) for k in features}
        return res_arrays, df_records.iloc[valid_indices].reset_index(drop=True)

    train_feats, train_df_valid = extract_features_for_df(df_train_sub, "Extract Train (F1-7)")
    val_feats, val_df_valid = extract_features_for_df(df_val, "Extract Val (Fold 8)")

    for k in train_feats:
        print(f"Feature block {k}: train shape {train_feats[k].shape}, val shape {val_feats[k].shape}")

    # 4. Clinical Concepts Probing
    # Parse scp_codes column to multi-label indicators
    import ast
    def get_labels(df_in: pd.DataFrame, concept: str) -> np.ndarray:
        labels = np.zeros(len(df_in), dtype=int)
        for i, row in enumerate(df_in["scp_codes"]):
            try:
                codes = ast.literal_eval(row)
                if concept in codes:
                    labels[i] = 1
            except Exception:
                pass
        return labels

    concepts_to_probe = [
        "NORM",   # Normal ECG
        "LBBB",   # Left bundle branch block
        "RBBB",   # Right bundle branch block
        "1AVB",   # First degree AV block
        "STACH",  # Sinus tachycardia
        "SBRAD",  # Sinus bradycardia
        "AFIB",   # Atrial fibrillation
        "IMI",    # Inferior myocardial infarction
        "AMI",    # Anterior myocardial infarction
        "ALMI",   # Anterolateral MI
        "STTC",   # ST-T changes
        "ISCA",   # Ischemia anterior
    ]

    print("\n--- CLINICAL DIAGNOSTIC PROBES & DETERMINISTIC KILL GATE ---")
    probe_results = []
    baselines = ["R0", "R1", "R2", "R3", "R4"]

    val_patient_ids = val_df_valid["patient_id"].to_numpy()

    for concept in concepts_to_probe:
        y_train = get_labels(train_df_valid, concept)
        y_val = get_labels(val_df_valid, concept)
        n_pos_train = int(np.sum(y_train))
        n_pos_val = int(np.sum(y_val))

        if n_pos_train < 15 or n_pos_val < 10:
            print(f"Skipping {concept}: insufficient prevalence ({n_pos_train} train, {n_pos_val} val)")
            continue

        concept_row = {
            "concept": concept,
            "n_pos_train": n_pos_train,
            "n_pos_val": n_pos_val,
        }

        val_scores = {}
        for b in baselines:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_feats[b])
            X_val = scaler.transform(val_feats[b])

            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
            clf.fit(X_train, y_train)

            y_pred_proba = clf.predict_proba(X_val)[:, 1]
            val_scores[b] = y_pred_proba

            auroc = float(roc_auc_score(y_val, y_pred_proba))
            auprc = float(average_precision_score(y_val, y_pred_proba))
            concept_row[f"auroc_{b}"] = auroc
            concept_row[f"auprc_{b}"] = auprc

        # Compute patient-paired bootstrap deltas
        delta_r3_r0, ci_low_r3_r0, ci_high_r3_r0 = bootstrap_patient_delta_auroc(
            y_val, val_scores["R0"], val_scores["R3"], val_patient_ids, n_boot=args.n_boot, seed=args.seed
        )
        delta_r4_r3, ci_low_r4_r3, ci_high_r4_r3 = bootstrap_patient_delta_auroc(
            y_val, val_scores["R3"], val_scores["R4"], val_patient_ids, n_boot=args.n_boot, seed=args.seed
        )

        concept_row["delta_auroc_R3_minus_R0"] = delta_r3_r0
        concept_row["ci95_delta_R3_R0"] = [ci_low_r3_r0, ci_high_r3_r0]
        concept_row["delta_auroc_R4_minus_R3"] = delta_r4_r3
        concept_row["ci95_delta_R4_R3"] = [ci_low_r4_r3, ci_high_r4_r3]

        probe_results.append(concept_row)
        print(
            f"  {concept:<6}: R0={concept_row['auroc_R0']:.4f} | R1={concept_row['auroc_R1']:.4f} | "
            f"R2={concept_row['auroc_R2']:.4f} | R3={concept_row['auroc_R3']:.4f} | R4={concept_row['auroc_R4']:.4f} | "
            f"Δ(R3-R0)={delta_r3_r0:+.4f} [{ci_low_r3_r0:+.4f}, {ci_high_r3_r0:+.4f}]"
        )

    if len(probe_results) == 0:
        print("No concepts met minimum prevalence requirements; please increase training cohort size.")
        return

    results_df = pd.DataFrame(probe_results)
    results_df.to_parquet(out_dir / "M4A_DETERMINISTIC_PROBE_RESULTS.parquet", index=False)
    results_df.to_csv(out_dir / "M4A_DETERMINISTIC_PROBE_RESULTS.csv", index=False)

    # 5. Deterministic Kill Gate Evaluation (PRD Section 14)
    # Primary question: Does R3 > R0?
    mean_delta_r3_r0 = float(results_df["delta_auroc_R3_minus_R0"].mean())
    median_delta_r3_r0 = float(results_df["delta_auroc_R3_minus_R0"].median())
    mean_delta_r4_r3 = float(results_df["delta_auroc_R4_minus_R3"].mean())
    median_delta_r4_r3 = float(results_df["delta_auroc_R4_minus_R3"].median())

    n_concepts_r3_beats_r0 = int(np.sum(results_df["delta_auroc_R3_minus_R0"] > 0))
    n_concepts_total = len(results_df)

    # Kill gate criteria:
    # 1. Median delta AUROC (R3 - R0) > 0.
    # 2. Majority of probed concepts show R3 >= R0.
    # 3. Residual branch (R4 - R3) adds non-dipolar information.
    kill_gate_passed = (median_delta_r3_r0 > 0.002) and (n_concepts_r3_beats_r0 >= n_concepts_total // 2)
    gate_verdict = "PASS" if kill_gate_passed else "FAIL"

    summary = {
        "run_id": "M4A_DETERMINISTIC_REPRESENTATION_QUALIFICATION",
        "n_train_records": len(train_df_valid),
        "n_val_records": len(val_df_valid),
        "n_concepts_evaluated": n_concepts_total,
        "n_concepts_R3_exceeds_R0": n_concepts_r3_beats_r0,
        "mean_delta_auroc_R3_minus_R0": mean_delta_r3_r0,
        "median_delta_auroc_R3_minus_R0": median_delta_r3_r0,
        "mean_delta_auroc_R4_minus_R3": mean_delta_r4_r3,
        "median_delta_auroc_R4_minus_R3": median_delta_r4_r3,
        "deterministic_kill_gate_verdict": gate_verdict,
        "explanation": (
            f"R3 (Physical + Functional) achieves a median Delta AUROC of {median_delta_r3_r0:+.4f} over R0 (Physical VCG alone) "
            f"across {n_concepts_total} clinical concepts ({n_concepts_r3_beats_r0}/{n_concepts_total} concepts improved). "
            f"R4 (including non-dipolar residual) provides an additional {median_delta_r4_r3:+.4f} median Delta AUROC. "
            "QVCG-H functional trajectory information is qualified for representation learning."
            if kill_gate_passed else
            f"R3 fails to demonstrate consistent value add over R0 (median Delta AUROC={median_delta_r3_r0:+.4f}). "
            "Functional Hilbert branch does not justify added complexity beyond physical VCG."
        ),
    }

    summary_path = out_dir / "M4A_DETERMINISTIC_REPORT.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 6. Publication Figure: Baseline Comparison Across Concepts
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(results_df))
    width = 0.16

    ax.bar(x - 2*width, results_df["auroc_R0"], width, label="R0 (Physical VCG)", color="#7f7f7f", alpha=0.9)
    ax.bar(x - width, results_df["auroc_R1"], width, label="R1 (Domain Hist)", color="#aec7e8", alpha=0.9)
    ax.bar(x, results_df["auroc_R2"], width, label="R2 (Hilbert Functional)", color="#ffbb78", alpha=0.9)
    ax.bar(x + width, results_df["auroc_R3"], width, label="R3 (Phys + Func)", color="#1f77b4", alpha=0.95)
    ax.bar(x + 2*width, results_df["auroc_R4"], width, label="R4 (Full Coupled + Res)", color="#2ca02c", alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["concept"], rotation=30, ha="right")
    ax.set_ylabel("Linear Probe AUROC (Fold 8)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Milestone M4A: Deterministic Representation Baselines (Fold 8 Probing)\nMedian Δ(R3 - R0) = {median_delta_r3_r0:+.4f} -> Gate: {gate_verdict}")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_m4a_baseline_comparison.png")
    plt.close(fig)

    print("\n" + "=" * 80)
    print(f"MILESTONE M4A DETERMINISTIC KILL GATE: {gate_verdict}")
    print(f"Median Delta AUROC (R3 - R0): {median_delta_r3_r0:+.4f}")
    print(f"Median Delta AUROC (R4 - R3): {median_delta_r4_r3:+.4f}")
    print(f"Explanation: {summary['explanation']}")
    print(f"Saved report to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
