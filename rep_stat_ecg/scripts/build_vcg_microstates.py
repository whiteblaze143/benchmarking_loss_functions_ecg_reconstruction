"""Milestone M1: Extract Patient-Balanced 3D VCG Microstates on Folds 1-7.

Adheres strictly to PRD Sections 7, 8, 11, 12, 13, 14:
- Folds 1-7 only.
- Exactly one ECG per patient via deterministic hash of patient_id.
- Derivatives and intrinsic descriptors computed at 500 Hz physical time FIRST.
- Beat normalization: resampled to P=64 normalized phase locations per beat.
- Capped at max_beats_per_patient (e.g. 6 beats) to prevent dominance.
- Computes training-only coordinate standardizer (mu and MAD) and serializes it.
- Saves VCG_MICROSTATE_TABLE.parquet and VCG_MICROSTATE_AUDIT.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from rep_stat_ecg.src.vcg.lift import VCGLift
from rep_stat_ecg.src.motifs.microstates import (
    detect_r_peaks,
    build_beat_microstates,
    select_patient_balanced_ecgs,
)
from rep_stat_ecg.src.motifs.spatial_domains import CoordinateStandardizer


def main():
    parser = argparse.ArgumentParser(description="Extract patient-balanced VCG microstates.")
    parser.add_argument("--data-dir", type=str, default="data/ptb_xl", help="Path to PTB-XL root.")
    parser.add_argument("--out-dir", type=str, default="refine-logs/qvcg", help="Output directory.")
    parser.add_argument("--max-patients", type=int, default=2500, help="Number of training patients.")
    parser.add_argument("--max-beats", type=int, default=6, help="Max beats per patient.")
    parser.add_argument("--phase-points", type=int, default=64, help="Phase points per beat.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing {db_path}")

    df_full = pd.read_csv(db_path)
    df_train = df_full[df_full["strat_fold"].isin(range(1, 8))].copy().reset_index(drop=True)
    df_train["ecg_id"] = df_train["ecg_id"].astype(str)
    df_train["patient_id"] = df_train["patient_id"].astype(str)

    print(f"Selecting patient-balanced training cohort from Folds 1-7 (target: {args.max_patients})...")
    df_balanced = select_patient_balanced_ecgs(df_train, max_patients=args.max_patients, seed=args.seed)
    print(f"Selected {len(df_balanced)} unique training patients (1 ECG each).")

    lift_mod = VCGLift(lam=1e-6)

    all_microstates = []
    skipped_records = 0
    total_beats_extracted = 0

    print("Extracting physical VCG trajectories and phase-normalized microstates...")
    for idx, row in tqdm(df_balanced.iterrows(), total=len(df_balanced)):
        rec_path = data_dir / row["filename_hr"]
        patient_id = row["patient_id"]
        ecg_id = row["ecg_id"]

        try:
            signal, fields = wfdb.rdsamp(str(rec_path))  # [5000, 12] in physical mV
        except Exception as e:
            skipped_records += 1
            continue

        fs = float(fields["fs"])
        # Transpose to [12, 5000]
        E = signal.T  # [12, T]
        import torch
        with torch.no_grad():
            E_tensor = torch.tensor(E, dtype=torch.float32).unsqueeze(0)  # [1, 12, T]
            v_tensor = lift_mod.lift(E_tensor).squeeze(0)                 # [3, T]
            v_physical = v_tensor.cpu().numpy()

        lead_ii = E[1]  # Lead II signal for R-peak detection
        r_peaks = detect_r_peaks(lead_ii, fs=fs)

        ecg_microstates = build_beat_microstates(
            v_physical,
            r_peaks,
            patient_id=patient_id,
            ecg_id=ecg_id,
            fs=fs,
            phase_points=args.phase_points,
            max_beats=args.max_beats,
        )

        if ecg_microstates:
            all_microstates.extend(ecg_microstates)
            n_beats = len(ecg_microstates) // args.phase_points
            total_beats_extracted += n_beats
        else:
            skipped_records += 1

    df_micro = pd.DataFrame(all_microstates)
    print(f"Extracted {len(df_micro):,} microstates from {total_beats_extracted:,} beats across {df_micro['patient_id'].nunique():,} patients.")

    # Serialize microstate table
    micro_parquet = out_dir / "VCG_MICROSTATE_TABLE.parquet"
    df_micro.to_parquet(micro_parquet, index=False)
    print(f"Saved microstates to {micro_parquet}")

    # Fit and serialize training coordinate standardizer
    v_all = df_micro[["v_x", "v_y", "v_z"]].values
    std = CoordinateStandardizer()
    std.fit(v_all)

    std_dict = {
        "median": std.median.tolist(),
        "mad": std.mad.tolist(),
        "scale": (1.4826 * std.mad).tolist(),
        "n_samples": len(df_micro),
        "n_patients": int(df_micro["patient_id"].nunique()),
    }
    std_json = out_dir / "VCG_COORDINATE_STANDARDIZER.json"
    with open(std_json, "w") as f:
        json.dump(std_dict, f, indent=2)
    print(f"Saved coordinate standardizer to {std_json}")

    # Generate Audit Markdown
    audit_md = out_dir / "VCG_MICROSTATE_AUDIT.md"
    with open(audit_md, "w") as f:
        f.write("# VCG Microstate Census & Audit Report\n\n")
        f.write(f"- **Total Patients Selected**: {len(df_balanced):,}\n")
        f.write(f"- **Total Patients Represented**: {df_micro['patient_id'].nunique():,}\n")
        f.write(f"- **Total Beats Processed**: {total_beats_extracted:,}\n")
        f.write(f"- **Total Microstates Extracted**: {len(df_micro):,}\n")
        f.write(f"- **Phase Resolution**: {args.phase_points} points/beat\n")
        f.write(f"- **Skipped Records (insufficient peaks)**: {skipped_records}\n")
        f.write(f"- **Median VCG [x, y, z]**: `{std_dict['median']}`\n")
        f.write(f"- **MAD VCG [x, y, z]**: `{std_dict['mad']}`\n")
        f.write(f"- **Speed s range**: [{df_micro['s'].min():.3f}, {df_micro['s'].max():.3f}]\n")
        f.write(f"- **Radial velocity rho range**: [{df_micro['rho'].min():.3f}, {df_micro['rho'].max():.3f}]\n")
        f.write(f"- **Curvature kappa range**: [{df_micro['kappa'].min():.3f}, {df_micro['kappa'].max():.3f}]\n")
    print(f"Saved audit report to {audit_md}")


if __name__ == "__main__":
    main()
