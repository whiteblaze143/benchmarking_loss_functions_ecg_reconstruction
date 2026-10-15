"""Verify EchoNext dataset integrity, splits, shapes, and metadata alignment."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("echonext_integrity")

ECHONEXT_DIR = Path("/data/mithunmanivannan/echonext")
EXPECTED_COUNTS = {
    "train": 72475,
    "val": 4626,
    "test": 5442,
    "no_split": 17457,
    "total": 100000,
}


def verify_metadata_and_splits() -> Dict[str, Any]:
    csv_path = ECHONEXT_DIR / "echonext_metadata_100k.csv"
    logger.info("Loading metadata from %s...", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Metadata loaded. Rows: %d, Columns: %d", len(df), len(df.columns))

    assert len(df) == EXPECTED_COUNTS["total"], f"Expected {EXPECTED_COUNTS['total']} rows, got {len(df)}"

    # Check split column
    split_counts = df["split"].value_counts().to_dict()
    logger.info("Split counts in metadata: %s", split_counts)
    for split_name, expected_count in EXPECTED_COUNTS.items():
        if split_name != "total":
            assert split_counts.get(split_name, 0) == expected_count, (
                f"Split {split_name}: expected {expected_count}, got {split_counts.get(split_name, 0)}"
            )

    # Check patient disjointness
    train_patients = set(df[df["split"] == "train"]["patient_key"].unique())
    val_patients = set(df[df["split"] == "val"]["patient_key"].unique())
    test_patients = set(df[df["split"] == "test"]["patient_key"].unique())

    logger.info("Unique patients: Train=%d, Val=%d, Test=%d", len(train_patients), len(val_patients), len(test_patients))

    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    logger.info("Patient overlap check:")
    logger.info("  Train & Val overlap: %d", len(train_val_overlap))
    logger.info("  Train & Test overlap: %d", len(train_test_overlap))
    logger.info("  Val & Test overlap: %d", len(val_test_overlap))

    assert len(train_val_overlap) == 0, f"Train and Val share {len(train_val_overlap)} patients!"
    assert len(train_test_overlap) == 0, f"Train and Test share {len(train_test_overlap)} patients!"
    assert len(val_test_overlap) == 0, f"Val and Test share {len(val_test_overlap)} patients!"

    # Check that val and test have 1 ECG per patient
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    assert len(val_df) == len(val_patients), f"Val has multiple ECGs per patient: {len(val_df)} vs {len(val_patients)}"
    assert len(test_df) == len(test_patients), f"Test has multiple ECGs per patient: {len(test_df)} vs {len(test_patients)}"

    # Check SHD primary endpoint
    assert "shd_moderate_or_greater_flag" in df.columns, "shd_moderate_or_greater_flag missing from metadata!"
    shd_col = df["shd_moderate_or_greater_flag"]
    assert shd_col.isna().sum() == 0, f"Found {shd_col.isna().sum()} NaN values in SHD target!"

    prevalence = {
        split: float(df[df["split"] == split]["shd_moderate_or_greater_flag"].mean())
        for split in ["train", "val", "test", "no_split"]
    }
    logger.info("SHD Prevalence by split: %s", prevalence)

    # Check 7 tabular features
    tabular_cols = [
        "sex", "ventricular_rate", "atrial_rate", "pr_interval",
        "qrs_duration", "qt_corrected", "age_at_ecg"
    ]
    for col in tabular_cols:
        assert col in df.columns, f"Required tabular column {col} missing!"

    return {
        "metadata_rows": len(df),
        "split_counts": split_counts,
        "patient_counts": {
            "train": len(train_patients),
            "val": len(val_patients),
            "test": len(test_patients),
        },
        "patient_disjoint": True,
        "shd_prevalence": prevalence,
        "tabular_columns": tabular_cols,
    }


def verify_arrays() -> Dict[str, Any]:
    array_info = {}
    for split in ["train", "val", "test", "no_split"]:
        w_path = ECHONEXT_DIR / f"EchoNext_{split}_waveforms.npy"
        t_path = ECHONEXT_DIR / f"EchoNext_{split}_tabular_features.npy"

        assert w_path.exists(), f"Missing waveform file: {w_path}"
        assert t_path.exists(), f"Missing tabular file: {t_path}"

        w = np.load(w_path, mmap_mode="r")
        t = np.load(t_path, mmap_mode="r")

        expected_n = EXPECTED_COUNTS[split]
        assert w.shape == (expected_n, 1, 2500, 12), f"Unexpected waveform shape {w.shape} for {split}"
        assert t.shape == (expected_n, 7), f"Unexpected tabular shape {t.shape} for {split}"

        # Check sample finite
        sample_w = w[:10]
        sample_t = t[:10]
        assert np.isfinite(sample_w).all(), f"Found non-finite values in sample waveforms for {split}"
        assert np.isfinite(sample_t).all(), f"Found non-finite values in sample tabular for {split}"

        array_info[split] = {
            "waveform_shape": list(w.shape),
            "waveform_dtype": str(w.dtype),
            "tabular_shape": list(t.shape),
            "tabular_dtype": str(t.dtype),
            "file_size_bytes_waveform": w_path.stat().st_size,
            "file_size_bytes_tabular": t_path.stat().st_size,
        }
        logger.info(
            "%s verified: Waveforms %s (%s, %.2f MB), Tabular %s (%s, %.2f KB)",
            split.upper(),
            w.shape,
            w.dtype,
            w_path.stat().st_size / (1024 * 1024),
            t.shape,
            t.dtype,
            t_path.stat().st_size / 1024,
        )

    return array_info


def verify_checksums() -> Dict[str, Any]:
    sums_file = ECHONEXT_DIR / "SHA256SUMS.txt"
    assert sums_file.exists(), f"Missing {sums_file}"

    manifest = {}
    with open(sums_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                manifest[parts[1]] = parts[0]

    logger.info("Manifest contains %d entries", len(manifest))

    # Check non-large files directly (metadata, tabular files, txt)
    small_files = [
        "EchoNext_no_split_tabular_features.npy",
        "EchoNext_test_tabular_features.npy",
        "EchoNext_train_tabular_features.npy",
        "EchoNext_val_tabular_features.npy",
        "LICENSE.txt",
        "README.md",
        "echonext_metadata_100k.csv",
    ]

    verified = {}
    for fn in small_files:
        fp = ECHONEXT_DIR / fn
        if not fp.exists():
            continue
        h = hashlib.sha256(fp.read_bytes()).hexdigest()
        expected = manifest.get(fn)
        matches = (h == expected)
        logger.info("Checksum %s: expected=%s, actual=%s, match=%s", fn, expected[:10] if expected else None, h[:10], matches)
        assert matches, f"Checksum mismatch on {fn}!"
        verified[fn] = {"expected": expected, "actual": h, "verified": matches}

    return {"verified_files": verified, "all_manifest_entries": len(manifest)}


def main():
    t0 = time.time()
    logger.info("=== Stage 1: EchoNext Data Verification Starting ===")

    meta_info = verify_metadata_and_splits()
    arr_info = verify_arrays()
    checksum_info = verify_checksums()

    elapsed = time.time() - t0
    logger.info("=== Stage 1: All Core Data Verification PASSED in %.2fs ===", elapsed)

    out_data = {
        "status": "PASSED",
        "elapsed_seconds": elapsed,
        "metadata": meta_info,
        "arrays": arr_info,
        "checksums": checksum_info,
    }

    out_json = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/scripts/paper07/echonext/tests/echonext_integrity_results.json")
    out_json.write_text(json.dumps(out_data, indent=2))
    logger.info("Saved verification results to %s", out_json)


if __name__ == "__main__":
    main()
