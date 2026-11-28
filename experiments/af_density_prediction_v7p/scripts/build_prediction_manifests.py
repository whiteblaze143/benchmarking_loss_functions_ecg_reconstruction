"""Master builder for V7-P prediction manifests and feasibility audits."""
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import landmarks
import history_features
import validation

CONTINUITY_DIR = PROJECT_DIR.parent / "af_density_continuity_v5" / "manifests"
NOISE_HOLTER_DIR = Path("/data/mithunmanivannan/datasets/noise-holter-database-a-multi-device-multi-arrhythmia-real-world-clinical-ecg-signal-quality-benchmark-1.0.0")


def main():
    print("=" * 80)
    print("V7-P Predictive Framework: Building Authoritative Manifests (P01-P04)")
    print("=" * 80)

    # 1. Load source continuous rhythm manifests
    print("\n[Step 1] Loading verified canonical rhythm timelines...")
    intervals_df = pd.read_parquet(CONTINUITY_DIR / "canonical_rhythm_intervals.parquet")
    runs_df = pd.read_parquet(CONTINUITY_DIR / "canonical_af_runs.parquet")
    print(f"Loaded {len(intervals_df):,} rhythm intervals and {len(runs_df):,} AF runs.")

    # 2. Build P01: Forecasting Sustained AF
    print("\n[Step 2] Building P01 Landmark Manifest (Forecasting Sustained AF Onset)...")
    t0 = time.time()
    p01_raw = landmarks.extract_p01_landmarks(intervals_df, runs_df)
    print(f"Extracted {len(p01_raw):,} raw grid points ({time.time() - t0:.1f}s). Extracting F0-F3 features...")
    p01_df = history_features.extract_features_for_landmarks_dataframe(p01_raw, intervals_df, runs_df)
    p01_df = validation.assign_patient_disjoint_folds(p01_df, patient_col="patient_id", target_col="target_sustained_30m", n_splits=5)
    p01_path = PROJECT_DIR / "P01_landmark_manifest.parquet"
    p01_df.to_parquet(p01_path, index=False)
    print(f"Saved {p01_path} ({len(p01_df):,} rows, {p01_df['patient_id'].nunique()} patients, {p01_df['target_sustained_30m'].sum():,} positive).")

    # 3. Build P02: Early Episode Persistence
    print("\n[Step 3] Building P02 Episode Persistence Manifest (60s Landmark)...")
    t0 = time.time()
    p02_raw = landmarks.extract_p02_landmarks(runs_df, landmark_s=60.0)
    print(f"Extracted {len(p02_raw):,} eligible episodes surviving >=60s. Extracting F0-F3 features...")
    p02_df = history_features.extract_features_for_landmarks_dataframe(p02_raw, intervals_df, runs_df)
    p02_df = validation.assign_patient_disjoint_folds(p02_df, patient_col="patient_id", target_col="target_persists_30m", n_splits=5)
    p02_path = PROJECT_DIR / "P02_episode_persistence_manifest.parquet"
    p02_df.to_parquet(p02_path, index=False)
    print(f"Saved {p02_path} ({len(p02_df):,} rows, {p02_df['patient_id'].nunique()} patients, {p02_df['target_persists_30m'].sum():,} positive).")

    # 4. Build P03: Dynamic Survival Prediction
    print("\n[Step 4] Building P03 Survival Manifest (1m, 5m, 15m Dynamic Landmarks)...")
    t0 = time.time()
    p03_raw = landmarks.extract_p03_landmarks(runs_df, landmarks_s=[60.0, 300.0, 900.0])
    print(f"Extracted {len(p03_raw):,} dynamic landmark episodes. Extracting F0-F3 features...")
    p03_df = history_features.extract_features_for_landmarks_dataframe(p03_raw, intervals_df, runs_df)
    p03_df = validation.assign_patient_disjoint_folds(p03_df, patient_col="patient_id", target_col="target_terminate_in_15m", n_splits=5)
    p03_path = PROJECT_DIR / "P03_survival_manifest.parquet"
    p03_df.to_parquet(p03_path, index=False)
    print(f"Saved {p03_path} ({len(p03_df):,} rows, {p03_df['patient_id'].nunique()} patients, {p03_df['target_terminate_in_15m'].sum():,} 15m terminations).")

    # 5. Build P04: Authentic Noise Holter Classification
    print("\n[Step 5] Building P04 Noise Holter Manifest (PhysioNet v1.0.0)...")
    p04_raw = landmarks.parse_noise_holter_headers(NOISE_HOLTER_DIR)
    p04_df = validation.assign_patient_disjoint_folds(p04_raw, patient_col="patient_id", target_col="af_present", n_splits=5)
    p04_path = PROJECT_DIR / "P04_noise_classification_manifest.parquet"
    p04_df.to_parquet(p04_path, index=False)
    print(f"Saved {p04_path} ({len(p04_df):,} excerpts, {p04_df['patient_id'].nunique()} patients, {p04_df['af_present'].sum():,} AF positive).")

    # 6. Generate V7P_split_manifest.csv
    print("\n[Step 6] Generating Unified Cross-Validation Split Manifest...")
    split_df = validation.generate_split_manifest(p01_df, p02_df, p03_df, p04_df)
    split_path = PROJECT_DIR / "V7P_split_manifest.csv"
    split_df.to_csv(split_path, index=False)
    print(f"Saved {split_path} ({len(split_df)} fold entries across 4 tasks).")

    # 7. Generate V7P_task_feasibility.csv
    print("\n[Step 7] Generating Task Feasibility and Support Summary...")
    feasibility_records = [
        {
            "task_id": "P01",
            "task_name": "Forecast Sustained AF (>=30m within 30m)",
            "primary_cohorts": "SHDB-AF, LTAFDB, IRIDIA-AF",
            "total_samples": len(p01_df),
            "positive_events": int(p01_df["target_sustained_30m"].sum()),
            "negative_samples": int((~p01_df["target_sustained_30m"]).sum()),
            "positive_prevalence": float(p01_df["target_sustained_30m"].mean()),
            "unique_patients": p01_df["patient_id"].nunique(),
            "unique_records": p01_df["record_id"].nunique(),
            "censored_count": 0,
            "feasibility_status": "HIGHLY_FEASIBLE",
        },
        {
            "task_id": "P02",
            "task_name": "Predict Episode Persistence to >=30m at 60s Landmark",
            "primary_cohorts": "IRIDIA-AF, SHDB-AF, LTAFDB",
            "total_samples": len(p02_df),
            "positive_events": int(p02_df["target_persists_30m"].sum()),
            "negative_samples": int((~p02_df["target_persists_30m"]).sum()),
            "positive_prevalence": float(p02_df["target_persists_30m"].mean()),
            "unique_patients": p02_df["patient_id"].nunique(),
            "unique_records": p02_df["record_id"].nunique(),
            "censored_count": int(p02_df["is_censored"].sum()),
            "feasibility_status": "HIGHLY_FEASIBLE",
        },
        {
            "task_id": "P03",
            "task_name": "Dynamic Survival and 15m Termination at [1m, 5m, 15m]",
            "primary_cohorts": "IRIDIA-AF, SHDB-AF, LTAFDB",
            "total_samples": len(p03_df),
            "positive_events": int(p03_df["target_terminate_in_15m"].sum()),
            "negative_samples": int((~p03_df["target_terminate_in_15m"]).sum()),
            "positive_prevalence": float(p03_df["target_terminate_in_15m"].mean()),
            "unique_patients": p03_df["patient_id"].nunique(),
            "unique_records": p03_df["record_id"].nunique(),
            "censored_count": int(p03_df["is_censored"].sum()),
            "feasibility_status": "HIGHLY_FEASIBLE",
        },
        {
            "task_id": "P04",
            "task_name": "Authentic Ambulatory Noise Holter Rhythm Classification",
            "primary_cohorts": "Noise-Holter (PhysioNet v1.0.0)",
            "total_samples": len(p04_df),
            "positive_events": int(p04_df["af_present"].sum()),
            "negative_samples": int((~p04_df["af_present"]).sum()),
            "positive_prevalence": float(p04_df["af_present"].mean()),
            "unique_patients": p04_df["patient_id"].nunique(),
            "unique_records": p04_df["strip_id"].nunique(),
            "censored_count": 0,
            "feasibility_status": "HIGHLY_FEASIBLE",
        },
    ]
    feasibility_df = pd.DataFrame(feasibility_records)
    feasibility_path = PROJECT_DIR / "V7P_task_feasibility.csv"
    feasibility_df.to_csv(feasibility_path, index=False)
    print(f"Saved {feasibility_path}.")

    # 8. Generate V7P_feature_registry.csv
    print("\n[Step 8] Generating Feature Registry...")
    feature_meta = [
        ("time_since_rec_start_s", "F0", "seconds", "Elapsed time on monitoring recording clock"),
        ("elapsed_af_duration_s", "F0", "seconds", "Elapsed duration of active AF episode at landmark"),
        ("last_af_duration_s", "F1", "seconds", "Duration of immediately preceding complete AF episode"),
        ("time_since_last_af_s", "F1", "seconds", "Elapsed gap since last AF episode termination"),
        ("af_burden_15m", "F1", "ratio", "Cumulative AF burden during the preceding 15 minutes"),
        ("af_burden_30m", "F1", "ratio", "Cumulative AF burden during the preceding 30 minutes"),
        ("af_burden_60m", "F1", "ratio", "Cumulative AF burden during the preceding 60 minutes"),
        ("af_burden_1h", "F2", "ratio", "Cumulative AF burden over the preceding 1 hour"),
        ("episode_count_1h", "F2", "count", "Number of distinct AF episodes in preceding 1 hour"),
        ("longest_af_run_1h", "F2", "seconds", "Longest continuous AF run duration in preceding 1 hour"),
        ("brief_run_count_1h", "F2", "count", "Number of brief AF runs (<30s) in preceding 1 hour"),
        ("af_burden_3h", "F2", "ratio", "Cumulative AF burden over the preceding 3 hours"),
        ("episode_count_3h", "F2", "count", "Number of distinct AF episodes in preceding 3 hours"),
        ("longest_af_run_3h", "F2", "seconds", "Longest continuous AF run duration in preceding 3 hours"),
        ("brief_run_count_3h", "F2", "count", "Number of brief AF runs (<30s) in preceding 3 hours"),
        ("af_burden_6h", "F2", "ratio", "Cumulative AF burden over the preceding 6 hours"),
        ("episode_count_6h", "F2", "count", "Number of distinct AF episodes in preceding 6 hours"),
        ("longest_af_run_6h", "F2", "seconds", "Longest continuous AF run duration in preceding 6 hours"),
        ("brief_run_count_6h", "F2", "count", "Number of brief AF runs (<30s) in preceding 6 hours"),
        ("charitos_density_1h", "F3", "density", "Continuous Charitos lower envelope density over 1h"),
        ("af_aggregation_1h", "F3", "ratio", "Charitos AF aggregation s_max / total_af over 1h"),
        ("charitos_density_3h", "F3", "density", "Continuous Charitos lower envelope density over 3h"),
        ("af_aggregation_3h", "F3", "ratio", "Charitos AF aggregation s_max / total_af over 3h"),
        ("charitos_density_6h", "F3", "density", "Continuous Charitos lower envelope density over 6h"),
        ("af_aggregation_6h", "F3", "ratio", "Charitos AF aggregation s_max / total_af over 6h"),
        ("is_density_defined_3h", "F3", "indicator", "Binary indicator that density is mathematically defined (0 < burden < 1)"),
    ]
    feat_reg_df = pd.DataFrame(feature_meta, columns=["feature_name", "feature_group", "units", "description"])
    feat_reg_path = PROJECT_DIR / "V7P_feature_registry.csv"
    feat_reg_df.to_csv(feat_reg_path, index=False)
    print(f"Saved {feat_reg_path}.")

    print("\n" + "=" * 80)
    print("V7-P Manifest Building Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
