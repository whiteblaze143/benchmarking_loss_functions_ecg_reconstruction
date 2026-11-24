#!/usr/bin/env python3
"""Script 01: Canonical Physical-Time Dataset Reconstruction.
Builds canonical rhythm timelines and AF run inventories across LTAFDB, SHDB-AF, and IRIDIA-AF.
Generates:
  - manifests/canonical_rhythm_intervals.parquet
  - manifests/canonical_af_runs.parquet
  - manifests/cohort_inventory.csv
"""

import os
import sys
import glob
import time
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.dataset_adapters import (
    load_ltafdb_record,
    load_shdb_record,
    load_iridia_record
)
from src.rhythm_intervals import audit_run_censoring, AFRun

def main():
    root_dir = Path(__file__).resolve().parents[1]
    manifests_dir = root_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE A01: BUILDING CANONICAL PHYSICAL-TIME RHYTHM TIMELINES")
    print("=" * 80)

    all_intervals = []
    all_runs = []
    inventory_rows = []

    # 1. LTAFDB
    ltaf_dir = "/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference"
    ltaf_records = sorted([
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(ltaf_dir, "*.hea"))
    ])
    print(f"Loading {len(ltaf_records)} LTAFDB records...")
    for rec_id in ltaf_records:
        intervals, runs, meta = load_ltafdb_record(rec_id, ltaf_dir)
        for itv in intervals:
            all_intervals.append(itv.to_dict())
        for r in runs:
            all_runs.append(r.to_dict())
        inventory_rows.append(meta)

    # 2. SHDB-AF
    shdb_dir = "/data/mithunmanivannan/datasets/shdb_af"
    shdb_meta_file = "/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv"
    shdb_meta_df = pd.read_csv(shdb_meta_file) if os.path.exists(shdb_meta_file) else None
    shdb_records = sorted([
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(shdb_dir, "*.hea"))
    ])
    print(f"Loading {len(shdb_records)} SHDB-AF records...")
    for rec_id in shdb_records:
        intervals, runs, meta = load_shdb_record(rec_id, shdb_dir, shdb_meta_df)
        for itv in intervals:
            all_intervals.append(itv.to_dict())
        for r in runs:
            all_runs.append(r.to_dict())
        inventory_rows.append(meta)

    # 3. IRIDIA-AF
    iridia_base = "/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1"
    iridia_meta_df = pd.read_csv(os.path.join(iridia_base, "iridia-af-metadata-v1.0.1.csv"))
    iridia_intervals_df = pd.read_csv(os.path.join(iridia_base, "timeline_intervals.csv"))
    calib_file = os.path.join(iridia_base, "calibration_exclusion_intervals.csv")
    calib_df = pd.read_csv(calib_file) if os.path.exists(calib_file) else None

    print(f"Loading {len(iridia_meta_df)} IRIDIA-AF records...")
    for _, r_info in iridia_meta_df.iterrows():
        rec_id = str(r_info["record_id"])
        intervals, runs, meta = load_iridia_record(rec_id, r_info, iridia_intervals_df, calib_df)
        for itv in intervals:
            all_intervals.append(itv.to_dict())
        for r in runs:
            all_runs.append(r.to_dict())
        inventory_rows.append(meta)

    # Convert to DataFrames
    df_intervals = pd.DataFrame(all_intervals)
    df_runs = pd.DataFrame(all_runs)
    df_inv = pd.DataFrame(inventory_rows)

    # Save to manifests
    intervals_parquet = manifests_dir / "canonical_rhythm_intervals.parquet"
    runs_parquet = manifests_dir / "canonical_af_runs.parquet"
    inv_csv = manifests_dir / "cohort_inventory.csv"

    df_intervals.to_parquet(intervals_parquet, index=False)
    df_runs.to_parquet(runs_parquet, index=False)
    df_inv.to_csv(inv_csv, index=False)

    print(f"\nSaved intervals to {intervals_parquet} (shape: {df_intervals.shape})")
    print(f"Saved AF runs to {runs_parquet} (shape: {df_runs.shape})")
    print(f"Saved cohort inventory to {inv_csv} (shape: {df_inv.shape})")

    # Audit censoring per dataset
    print("\n" + "=" * 80)
    print("CENSORING AUDIT AND ARITHMETIC VERIFICATION")
    print("=" * 80)

    for ds in ["LTAFDB", "SHDB-AF", "IRIDIA-AF"]:
        ds_runs_raw = [
            AFRun(**r) for r in all_runs if r["dataset"] == ds
        ]
        audit = audit_run_censoring(ds_runs_raw)
        print(f"\nCohort: {ds}")
        for k, v in audit.items():
            print(f"  {k:25s}: {v}")

    # Cohort inventory summary
    print("\n" + "=" * 80)
    print("COHORT INVENTORY SUMMARY")
    print("=" * 80)
    summary_list = []
    for ds in ["LTAFDB", "SHDB-AF", "IRIDIA-AF"]:
        sub = df_inv[df_inv["dataset"] == ds]
        sub_runs = df_runs[df_runs["dataset"] == ds]
        summary_list.append({
            "cohort": ds,
            "recordings": len(sub),
            "unique_subjects": sub["patient_id"].nunique(),
            "total_monitoring_h": sub["duration_h"].sum(),
            "total_af_h": sub_runs["duration_s"].sum() / 3600.0,
            "total_af_runs": len(sub_runs),
            "observed_onsets": sub_runs["has_observed_onset"].sum(),
            "observed_terminations": sub_runs["has_observed_termination"].sum(),
            "complete_runs": sub_runs["is_complete"].sum(),
            "left_censored": sub_runs["is_left_censored"].sum(),
            "right_censored": sub_runs["is_right_censored"].sum()
        })
    df_sum = pd.DataFrame(summary_list)
    print(df_sum.to_string(index=False))

if __name__ == "__main__":
    main()
