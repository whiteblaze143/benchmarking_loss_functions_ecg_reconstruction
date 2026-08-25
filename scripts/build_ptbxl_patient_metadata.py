#!/usr/bin/env python3
"""Build normalized patient metadata asset for PTB-XL under strict split isolation."""

import json, sys, os
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    db_path = ROOT / "data/ptb_xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
    manifest_path = ROOT / "refine-logs/ptbxl_tensor_content_manifest.json"
    output_dir = ROOT / "refine-logs/spatial_arch_1lead_v1/assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_out = output_dir / "ptbxl_patient_metadata.parquet"
    stats_out = output_dir / "metadata_stats.json"

    print("="*70)
    print("  BUILDING PTB-XL PATIENT METADATA ASSET")
    print("="*70)

    df = pd.read_csv(db_path)
    print(f"Loaded {len(df):,} total records from {db_path.name}")

    # Determine split from strat_fold (folds 1-8: train, 9: val, 10: test)
    # or map via ecg_id in ptbxl_tensor_content_manifest.json
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        split_map = {}
        for split, data in manifest.get("splits", {}).items():
            for entry in data.get("entries", []):
                ecg_id = int(Path(entry["relative_path"]).stem)
                split_map[ecg_id] = split
        df["split"] = df["ecg_id"].map(split_map).fillna("train")
    else:
        df["split"] = np.where(df["strat_fold"] <= 8, "train", np.where(df["strat_fold"] == 9, "val", "test"))

    train_mask = df["split"] == "train"
    train_df = df[train_mask].copy()
    print(f"Training split: {len(train_df):,} records ({train_df['patient_id'].nunique():,} unique patients)")

    # 1. Clean outliers in training set for robust statistics
    age_tr = train_df["age"].copy()
    age_tr[(age_tr < 0) | (age_tr > 110)] = np.nan
    age_mean = float(age_tr.mean())
    age_std = float(age_tr.std())

    sex_tr = train_df["sex"].dropna()
    sex_mode = float(sex_tr.mode()[0]) if len(sex_tr) else 0.0

    height_tr = train_df["height"].copy()
    height_tr[(height_tr < 50) | (height_tr > 250)] = np.nan
    height_median = float(height_tr.median())
    height_std = float(height_tr.std())

    weight_tr = train_df["weight"].copy()
    weight_tr[(weight_tr < 20) | (weight_tr > 300)] = np.nan
    weight_median = float(weight_tr.median())
    weight_std = float(weight_tr.std())

    stats = {
        "train_records": int(len(train_df)),
        "train_patients": int(train_df["patient_id"].nunique()),
        "age": {"mean": age_mean, "std": age_std, "missing_val": age_mean},
        "sex": {"mode": sex_mode, "missing_val": sex_mode},
        "height": {"median": height_median, "std": height_std, "missing_val": height_median},
        "weight": {"median": weight_median, "std": weight_std, "missing_val": weight_median},
    }

    # 2. Build full metadata features for all records
    df_out = pd.DataFrame()
    df_out["ecg_id"] = df["ecg_id"].astype(int)
    df_out["patient_id"] = df["patient_id"].astype(str)
    df_out["split"] = df["split"].astype(str)

    # Missingness indicators
    df_out["age_missing"] = df["age"].isna() | (df["age"] < 0) | (df["age"] > 110)
    df_out["sex_missing"] = df["sex"].isna()
    df_out["height_missing"] = df["height"].isna() | (df["height"] < 50) | (df["height"] > 250)
    df_out["weight_missing"] = df["weight"].isna() | (df["weight"] < 20) | (df["weight"] > 300)

    # Imputed & standardized continuous values
    age_raw = df["age"].copy()
    age_raw[df_out["age_missing"]] = age_mean
    df_out["age_z"] = ((age_raw - age_mean) / age_std).astype(np.float32)

    sex_raw = df["sex"].copy()
    sex_raw[df_out["sex_missing"]] = sex_mode
    df_out["sex"] = sex_raw.astype(np.float32)

    height_raw = df["height"].copy()
    height_raw[df_out["height_missing"]] = height_median
    df_out["height_z"] = ((height_raw - height_median) / height_std).astype(np.float32)

    weight_raw = df["weight"].copy()
    weight_raw[df_out["weight_missing"]] = weight_median
    df_out["weight_z"] = ((weight_raw - weight_median) / weight_std).astype(np.float32)

    # Derived BMI feature
    bmi_raw = weight_raw / ((height_raw / 100.0) ** 2)
    bmi_mean = float(bmi_raw[train_mask].mean())
    bmi_std = float(bmi_raw[train_mask].std())
    df_out["bmi_z"] = ((bmi_raw - bmi_mean) / bmi_std).astype(np.float32)
    stats["bmi"] = {"mean": bmi_mean, "std": bmi_std}

    # Cast boolean missing indicators to float32 for direct tensor ingestion
    for col in ["age_missing", "sex_missing", "height_missing", "weight_missing"]:
        df_out[col] = df_out[col].astype(np.float32)

    # 3. Save artifacts
    df_out.to_parquet(parquet_out, index=False)
    stats_out.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    print(f"\nSaved metadata parquet to: {parquet_out}")
    print(f"Saved metadata stats to:   {stats_out}")
    print(f"Columns: {df_out.columns.tolist()}")
    print("\nSample row:")
    print(df_out.head(1).to_dict(orient="records")[0])
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
