"""Milestone M4: Extract Transparent Feature Vectors (R0, R1, R2, R3) across Folds 1-9.

Adheres strictly to PRD Sections 28, 29, 30-37, 62:
- Uses frozen tokenizer mapping: v -> {R_0, ..., R_{K-1}, OOD}.
- Folds 1-7 (train), Fold 8 (val/development), Fold 9 (confirmation).
- Fold 10 remains strictly sealed!
- Extracts:
    R0: Basic VCG summary baseline.
    R1: Unmerged spatial domain features.
    R2: Quotient motif features.
    R3: Quotient + Where factorized features ([Z_motif, Z_location]).
- Serializes Parquet feature tables and QVCG_FEATURE_SCHEMA.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import torch
from tqdm import tqdm

from rep_stat_ecg.src.vcg.lift import VCGLift
from rep_stat_ecg.src.vcg.dynamics import extract_vcg_microstate_features
from rep_stat_ecg.src.motifs.microstates import detect_r_peaks
from rep_stat_ecg.src.motifs.spatial_domains import CoordinateStandardizer, VoxelGrid3D
from rep_stat_ecg.src.motifs.quotient import QuotientClosure
from rep_stat_ecg.src.motifs.tokenizer import QVCGTokenizer
from rep_stat_ecg.src.motifs.representation import (
    extract_r0_vcg_summary,
    extract_r1_spatial_domain_features,
    extract_r2_quotient_motif_features,
    extract_r3_quotient_plus_where,
)


def main():
    parser = argparse.ArgumentParser(description="Extract R0, R1, R2, R3 representations.")
    parser.add_argument("--data-dir", type=str, default="data/ptb_xl")
    parser.add_argument("--qvcg-dir", type=str, default="refine-logs/qvcg")
    parser.add_argument("--max-records-train", type=int, default=5000, help="Cap training records if needed.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    qvcg_dir = Path(args.qvcg_dir)
    data_dir = Path(args.data_dir)

    # 1. Load frozen standardizer, domain registry, and motif registry
    with open(qvcg_dir / "VCG_COORDINATE_STANDARDIZER.json") as f:
        std_dict = json.load(f)
    standardizer = CoordinateStandardizer()
    standardizer.median = np.array(std_dict["median"], dtype=np.float32)
    standardizer.mad = np.array(std_dict["mad"], dtype=np.float32)
    standardizer.is_fitted = True

    df_voxel_motif = pd.read_parquet(qvcg_dir / "VCG_VOXEL_TO_MOTIF.parquet")
    voxel_to_dom = dict(zip(df_voxel_motif["voxel_id"], df_voxel_motif["domain_id"]))
    dom_to_motif = dict(zip(df_voxel_motif["domain_id"], df_voxel_motif["motif_id"]))

    M = int(df_voxel_motif["domain_id"].max() + 1)
    K = int(df_voxel_motif["motif_id"].max() + 1)

    grid = VoxelGrid3D(grid_dim=24, coord_min=-3.0, coord_max=3.0)
    grid.eligible_voxels = set(df_voxel_motif["voxel_id"].unique())
    grid.voxel_to_domain = voxel_to_dom

    qc = QuotientClosure(n_domains=M, epsilon_mmd=0.01)
    qc.domain_to_motif = dom_to_motif
    qc.motif_to_domains = {k: [] for k in range(K)}
    for d, k in dom_to_motif.items():
        qc.motif_to_domains[k].append(d)

    tokenizer = QVCGTokenizer(standardizer=standardizer, grid=grid, quotient=qc)
    lift_mod = VCGLift(lam=1e-6)

    # 2. Load metadata for Folds 1-9 (Fold 10 is sealed!)
    df_meta = pd.read_csv(data_dir / "ptbxl_database.csv")
    df_eval = df_meta[df_meta["strat_fold"].isin(range(1, 10))].copy().reset_index(drop=True)

    # Subsample training records if specified
    df_train = df_eval[df_eval["strat_fold"].isin(range(1, 8))]
    df_val = df_eval[df_eval["strat_fold"] == 8]
    df_confirm = df_eval[df_eval["strat_fold"] == 9]

    if len(df_train) > args.max_records_train:
        df_train = df_train.sample(n=args.max_records_train, random_state=args.seed)

    df_target = pd.concat([df_train, df_val, df_confirm], ignore_index=True)
    print(f"Extracting representations for {len(df_target):,} ECG records (Train: {len(df_train):,}, Fold 8: {len(df_val):,}, Fold 9: {len(df_confirm):,})...")

    meta_rows = []
    r0_list, r1_list, r2_list, r3_list = [], [], [], []
    z_motif_list, z_where_list = [], []

    for idx, row in tqdm(df_target.iterrows(), total=len(df_target)):
        rec_path = data_dir / row["filename_hr"]
        try:
            signal, fields = wfdb.rdsamp(str(rec_path))
        except Exception:
            continue

        fs = float(fields["fs"])
        duration_s = signal.shape[0] / fs
        E = signal.T  # [12, T]

        with torch.no_grad():
            E_tensor = torch.tensor(E, dtype=torch.float32).unsqueeze(0)
            v_tensor = lift_mod.lift(E_tensor).squeeze(0)
            v_physical = v_tensor.cpu().numpy()  # [3, T]

        # Derivatives & intrinsic descriptor at 500 Hz
        v_dot, v_ddot, xi = extract_vcg_microstate_features(v_physical, fs=fs)

        # R-peaks & RR intervals
        lead_ii = E[1]
        r_peaks = detect_r_peaks(lead_ii, fs=fs)
        rr_intervals_s = (np.diff(r_peaks) / fs) if len(r_peaks) > 1 else np.array([0.8])

        # 1. R0 features
        feat_r0 = extract_r0_vcg_summary(v_physical, v_dot, xi, rr_intervals_s=rr_intervals_s)

        # 2. Tokenize domains and motifs
        domains, _ = tokenizer.tokenize_domains(v_physical)
        motifs, ood_rate = tokenizer.tokenize(v_physical)

        # 3. R1 features
        feat_r1 = extract_r1_spatial_domain_features(domains, num_domains=M, total_duration_s=duration_s)

        # 4. R2 features
        feat_r2 = extract_r2_quotient_motif_features(motifs, num_motifs=K, total_duration_s=duration_s)

        # 5. R3 features
        feat_r3, z_motif, z_where = extract_r3_quotient_plus_where(
            v_physical, motifs, num_motifs=K, total_duration_s=duration_s
        )

        meta_rows.append({
            "ecg_id": str(row["ecg_id"]),
            "patient_id": str(row["patient_id"]),
            "strat_fold": int(row["strat_fold"]),
            "scp_codes": str(row["scp_codes"]),
            "ood_rate": float(ood_rate),
        })

        r0_list.append(feat_r0)
        r1_list.append(feat_r1)
        r2_list.append(feat_r2)
        r3_list.append(feat_r3)
        z_motif_list.append(z_motif)
        z_where_list.append(z_where)

    df_meta_out = pd.DataFrame(meta_rows)

    # Stack feature matrices
    X_r0 = np.stack(r0_list, axis=0)
    X_r1 = np.stack(r1_list, axis=0)
    X_r2 = np.stack(r2_list, axis=0)
    X_r3 = np.stack(r3_list, axis=0)
    X_z_motif = np.stack(z_motif_list, axis=0)
    X_z_where = np.stack(z_where_list, axis=0)

    print(f"Feature Dimensions: R0={X_r0.shape[1]}, R1={X_r1.shape[1]}, R2={X_r2.shape[1]}, R3={X_r3.shape[1]}")

    # Serialize to Parquet
    def _save_feature_df(X, prefix, filename):
        cols = [f"{prefix}_{i}" for i in range(X.shape[1])]
        df_feat = pd.DataFrame(X, columns=cols)
        df_combined = pd.concat([df_meta_out, df_feat], axis=1)
        path = qvcg_dir / filename
        df_combined.to_parquet(path, index=False)
        print(f"Saved {filename} to {path}")

    _save_feature_df(X_r0, "r0", "QVCG_FEATURES_R0.parquet")
    _save_feature_df(X_r1, "r1", "QVCG_FEATURES_R1.parquet")
    _save_feature_df(X_r2, "r2", "QVCG_FEATURES_R2.parquet")
    _save_feature_df(X_r3, "r3", "QVCG_FEATURES_R3.parquet")
    _save_feature_df(X_z_motif, "z_motif", "QVCG_FEATURES_Z_MOTIF.parquet")
    _save_feature_df(X_z_where, "z_where", "QVCG_FEATURES_Z_WHERE.parquet")

    schema = {
        "num_records": len(df_meta_out),
        "dim_r0": int(X_r0.shape[1]),
        "dim_r1": int(X_r1.shape[1]),
        "dim_r2": int(X_r2.shape[1]),
        "dim_r3": int(X_r3.shape[1]),
        "num_domains_M": M,
        "num_motifs_K": K,
        "mean_ood_rate": float(df_meta_out["ood_rate"].mean()),
    }
    with open(qvcg_dir / "QVCG_FEATURE_SCHEMA.json", "w") as f:
        json.dump(schema, f, indent=2)
    print("Feature serialization complete!")


if __name__ == "__main__":
    main()
