#!/usr/bin/env python3
"""
MIMIC-III-Ext-PPG v1.1.0 Safe Event Key & Provenance Generator
Fixes:
  1. Safe Event Key: (subject_id, record_id, event_id)
     Prevents false merging of event_id across multiple waveform records for the same patient.
  2. Single Segment Convention: segment_id == 0
     Uses the clean, reproducible 30-second waveform segment immediately preceding event_time.
  3. Provenance Funnel: Grounded directly in PhysioNet v1.1.0 metadata.csv
     Total Segments: 6,399,754 | Unique Subjects: 6,189 | SR: 3,950,724 | AF: 597,769.
  4. Same-ICU Stay Transitions:
     Enforces icustay_id == target_icustay_id for rigorous clinical monitoring evaluation.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import polars as pl

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("mimic_safe_provenance")

METADATA_CSV = Path("/data/mithunmanivannan/datasets/mimic3_ext_ppg/metadata.csv")
OUTPUT_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/mimic3_ext_ppg_safe_audit")
ARTIFACT_JSON = Path("/home/mithunmanivannan/.gemini/antigravity-ide/brain/ba979872-97f8-4d89-9b0a-621cb4b9c051/mimic_provenance_audit.json")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("Generating MIMIC-III-Ext-PPG Safe Event Table and Verified Provenance")
    logger.info(f"Source Metadata: {METADATA_CSV}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("=" * 70)

    # 1. Full Dataset Baseline Scan
    logger.info("Scanning full metadata.csv for dataset-level ground truth...")
    q_all = pl.scan_csv(METADATA_CSV)
    total_segments = q_all.select(pl.len()).collect().item()
    unique_subjects_all = q_all.select(pl.col("subject_id").n_unique()).collect().item()

    # Rhythms
    sr_segments_all = q_all.filter(pl.col("event_rhythm") == "SR").select(pl.len()).collect().item()
    af_segments_all = q_all.filter(pl.col("event_rhythm") == "AF").select(pl.len()).collect().item()

    logger.info(f"Official PhysioNet Baseline: {total_segments:,} segments across {unique_subjects_all:,} unique subjects.")
    logger.info(f"Target Rhythm Segments: SR={sr_segments_all:,}, AF={af_segments_all:,} (Total SR+AF = {sr_segments_all + af_segments_all:,}).")

    # 2. Extract segment_id == 0 (Pre-Event 30-s Window)
    logger.info("\nExtracting segment_id == 0 (single 30-s window ending at event_time)...")
    cols = [
        "subject_id", "record_id", "event_id", "segment_id", "event_time", "event_rhythm",
        "hadm_id", "icustay_id", "strat_fold", "vector_10s_ecg_sqi", "vector_10s_pleth_sqi"
    ]
    df0 = (
        pl.scan_csv(METADATA_CSV)
        .filter(pl.col("segment_id") == 0)
        .select(cols)
        .collect()
    )
    total_events_all_rhythms = len(df0)
    subjects_all_rhythms = df0["subject_id"].n_unique()
    logger.info(f"Total segment_id==0 events (all rhythms): {total_events_all_rhythms:,} across {subjects_all_rhythms:,} subjects.")

    # 3. Filter to SR and AF
    df0_target = df0.filter(pl.col("event_rhythm").is_in(["SR", "AF"]))
    total_target_events = len(df0_target)
    subjects_target_events = df0_target["subject_id"].n_unique()
    target_counts = dict(df0_target["event_rhythm"].value_counts().iter_rows())
    logger.info(f"SR/AF segment_id==0 events: {total_target_events:,} across {subjects_target_events:,} subjects (SR: {target_counts.get('SR',0):,}, AF: {target_counts.get('AF',0):,}).")

    # 4. Lead-II ECG Presence and Quality Audit
    ecg_present_mask = (
        ~df0_target["vector_10s_ecg_sqi"].is_null() &
        ~df0_target["vector_10s_ecg_sqi"].str.contains("nan") &
        ~df0_target["vector_10s_ecg_sqi"].str.contains("-3")
    )
    df0_target = df0_target.with_columns(ecg_present_mask.alias("ecg_present"))
    events_ecg_present = df0_target.filter(pl.col("ecg_present"))
    n_ecg_present = len(events_ecg_present)
    sub_ecg_present = events_ecg_present["subject_id"].n_unique()
    logger.info(f"Lead-II ECG Present in Metadata: {n_ecg_present:,} ({n_ecg_present/total_target_events*100:.2f}%) across {sub_ecg_present:,} subjects.")

    # Quality threshold: SQI string contains '1'
    ecg_good_mask = df0_target["vector_10s_ecg_sqi"].str.contains("1") & df0_target["ecg_present"]
    df0_target = df0_target.with_columns(ecg_good_mask.alias("ecg_high_quality"))
    events_good_ecg = df0_target.filter(pl.col("ecg_high_quality"))
    n_good_ecg = len(events_good_ecg)
    sub_good_ecg = events_good_ecg["subject_id"].n_unique()
    good_counts = dict(events_good_ecg["event_rhythm"].value_counts().iter_rows())
    logger.info(f"High-Quality Lead-II ECG (SQI passed): {n_good_ecg:,} ({n_good_ecg/total_target_events*100:.2f}%) across {sub_good_ecg:,} subjects.")
    logger.info(f"High-Quality Event Counts: SR={good_counts.get('SR',0):,}, AF={good_counts.get('AF',0):,}.")

    # 5. Build Safe Event Table
    # Construct safe event key: subject_id + "_" + record_id + "_" + event_id
    events_good_ecg = events_good_ecg.with_columns([
        (pl.col("subject_id").cast(str) + "_" + pl.col("record_id") + "_" + pl.col("event_id").cast(str)).alias("safe_event_key"),
        pl.col("event_time").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    ]).sort(["subject_id", "event_time", "record_id", "event_id"])

    # Verify uniqueness of safe_event_key
    assert events_good_ecg["safe_event_key"].n_unique() == len(events_good_ecg), "Safe event key must be globally unique!"
    logger.info(f"Verified Safe Event Key uniqueness: {len(events_good_ecg):,} unique events.")

    # Save safe_event_table.parquet
    safe_event_parquet = OUTPUT_DIR / "safe_event_table.parquet"
    events_good_ecg.write_parquet(safe_event_parquet)
    logger.info(f"Saved {safe_event_parquet} ({os.path.getsize(safe_event_parquet)/(1024*1024):.2f} MB).")

    # 6. Construct Consecutive Longitudinal Event Pairs
    logger.info("\nConstructing chronological consecutive pairs within subject...")
    paired = events_good_ecg.with_columns([
        pl.col("event_rhythm").shift(-1).over("subject_id").alias("target_rhythm"),
        pl.col("event_time").shift(-1).over("subject_id").alias("target_time"),
        pl.col("record_id").shift(-1).over("subject_id").alias("target_record_id"),
        pl.col("event_id").shift(-1).over("subject_id").alias("target_event_id"),
        pl.col("safe_event_key").shift(-1).over("subject_id").alias("target_safe_event_key"),
        pl.col("icustay_id").shift(-1).over("subject_id").alias("target_icustay_id"),
        pl.col("strat_fold").shift(-1).over("subject_id").alias("target_strat_fold"),
    ]).filter(~pl.col("target_rhythm").is_null())

    paired = paired.with_columns([
        (pl.col("event_rhythm") + "->" + pl.col("target_rhythm")).alias("transition_cell"),
        ((pl.col("target_time") - pl.col("event_time")).dt.total_seconds() / 3600.0).alias("dt_hours"),
        (pl.col("icustay_id") == pl.col("target_icustay_id")).alias("same_icustay")
    ])

    total_pairs = len(paired)
    all_cells = dict(paired["transition_cell"].value_counts().iter_rows())
    logger.info(f"Total Consecutive Pairs: {total_pairs:,} -> {all_cells}")

    # Primary analysis: Same ICU Stay
    same_icu_pairs = paired.filter(pl.col("same_icustay"))
    n_same_icu = len(same_icu_pairs)
    same_cells = dict(same_icu_pairs["transition_cell"].value_counts().iter_rows())
    logger.info(f"Same-ICU Stay Pairs (Primary Cohort): {n_same_icu:,} ({n_same_icu/total_pairs*100:.2f}%) -> {same_cells}")

    # Save safe_transition_pairs.parquet
    safe_trans_parquet = OUTPUT_DIR / "safe_transition_pairs.parquet"
    same_icu_pairs.write_parquet(safe_trans_parquet)
    logger.info(f"Saved {safe_trans_parquet} ({os.path.getsize(safe_trans_parquet)/(1024*1024):.2f} MB).")

    # 7. Transition Breakdown by Fold for AF->SR and SR->AF
    af_to_sr_folds = dict(same_icu_pairs.filter(pl.col("transition_cell") == "AF->SR")["target_strat_fold"].value_counts().iter_rows())
    sr_to_af_folds = dict(same_icu_pairs.filter(pl.col("transition_cell") == "SR->AF")["target_strat_fold"].value_counts().iter_rows())
    logger.info(f"AF->SR Transitions by Target Fold: {dict(sorted(af_to_sr_folds.items()))}")
    logger.info(f"SR->AF Transitions by Target Fold: {dict(sorted(sr_to_af_folds.items()))}")

    # 8. Create Canonical Provenance Audit JSON
    audit_data = {
        "metadata": {
            "dataset": "MIMIC-III-Ext-PPG",
            "version": "1.1.0",
            "source_file": str(METADATA_CSV),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "safe_event_key_definition": "(subject_id, record_id, event_id)",
            "segment_convention": "segment_id == 0 (single 30-s strip immediately preceding event_time)",
            "lead_ii_ecg_requirement": "vector_10s_ecg_sqi contains '1' and not null/nan/-3",
            "primary_analysis_constraint": "same_icustay == True",
        },
        "dataset_baseline": {
            "total_segments_official": total_segments,
            "unique_subjects_official": unique_subjects_all,
            "sr_segments_official": sr_segments_all,
            "af_segments_official": af_segments_all,
        },
        "filtering_funnel": [
            {
                "stage": "1. Official PhysioNet Release Segments",
                "segments": total_segments,
                "subjects": unique_subjects_all,
                "sr_count": sr_segments_all,
                "af_count": af_segments_all,
                "survival_pct": 100.0,
            },
            {
                "stage": "2. Pre-Event Segment (segment_id == 0, All Rhythms)",
                "segments": total_events_all_rhythms,
                "subjects": subjects_all_rhythms,
                "survival_pct": round(total_events_all_rhythms / total_segments * 100.0, 2),
            },
            {
                "stage": "3. Target Arrhythmias (segment_id == 0, SR & AF)",
                "segments": total_target_events,
                "subjects": subjects_target_events,
                "sr_count": target_counts.get("SR", 0),
                "af_count": target_counts.get("AF", 0),
                "survival_pct": round(total_target_events / total_segments * 100.0, 2),
            },
            {
                "stage": "4. Simultaneous Lead-II ECG Present in Metadata",
                "segments": n_ecg_present,
                "subjects": sub_ecg_present,
                "survival_pct": round(n_ecg_present / total_target_events * 100.0, 2),
            },
            {
                "stage": "5. Lead-II ECG High Signal Quality (SQI Passed)",
                "segments": n_good_ecg,
                "subjects": sub_good_ecg,
                "sr_count": good_counts.get("SR", 0),
                "af_count": good_counts.get("AF", 0),
                "survival_pct": round(n_good_ecg / total_target_events * 100.0, 2),
            },
            {
                "stage": "6. Consecutive Longitudinal Event Pairs (Same ICU Stay)",
                "segments": n_same_icu,
                "subjects": same_icu_pairs["subject_id"].n_unique(),
                "sr_to_sr": same_cells.get("SR->SR", 0),
                "af_to_af": same_cells.get("AF->AF", 0),
                "sr_to_af": same_cells.get("SR->AF", 0),
                "af_to_sr": same_cells.get("AF->SR", 0),
                "survival_pct": round(n_same_icu / n_good_ecg * 100.0, 2),
            },
            {
                "stage": "7. Raw Waveform Files Locally Present on Disk",
                "segments": 0,
                "subjects": 0,
                "note": "Only metadata.csv (4.6 GB) is stored on VM. PhysioNet WFDB files require credentialed DUA authentication (HTTP 403 on anonymous request).",
                "survival_pct": 0.0,
            }
        ],
        "safe_transition_cohort": {
            "total_same_icu_pairs": n_same_icu,
            "unique_subjects": same_icu_pairs["subject_id"].n_unique(),
            "af_to_sr": {
                "total": same_cells.get("AF->SR", 0),
                "train_folds_0_6": sum(af_to_sr_folds.get(f, 0) for f in range(7)),
                "val_fold_7": af_to_sr_folds.get(7, 0),
                "test_fold_8": af_to_sr_folds.get(8, 0),
                "test_fold_9": af_to_sr_folds.get(9, 0),
            },
            "sr_to_af": {
                "total": same_cells.get("SR->AF", 0),
                "train_folds_0_6": sum(sr_to_af_folds.get(f, 0) for f in range(7)),
                "val_fold_7": sr_to_af_folds.get(7, 0),
                "test_fold_8": sr_to_af_folds.get(8, 0),
                "test_fold_9": sr_to_af_folds.get(9, 0),
            },
            "sr_to_sr": same_cells.get("SR->SR", 0),
            "af_to_af": same_cells.get("AF->AF", 0),
        }
    }

    with open(OUTPUT_DIR / "mimic_provenance_audit.json", "w") as f:
        json.dump(audit_data, f, indent=2)
    with open(ARTIFACT_JSON, "w") as f:
        json.dump(audit_data, f, indent=2)

    logger.info(f"Saved canonical provenance audit to {OUTPUT_DIR / 'mimic_provenance_audit.json'}")
    logger.info(f"Saved canonical provenance audit to {ARTIFACT_JSON}")
    logger.info(f"Completed in {time.time() - t_start:.2f}s.")

if __name__ == "__main__":
    main()
