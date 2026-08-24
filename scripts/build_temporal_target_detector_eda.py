#!/usr/bin/env python3
"""Build hash-bound EDA for the immutable PTB-XL target-event cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "results/factorial_v4/temporal_mmd_generation_bound"
DEFAULT_OUTPUT_DIR = ROOT / "results/factorial_v4/temporal_target_detector_eda"
DEFAULT_METADATA = ROOT / "data/ptb_xl/ptbxl_database.csv"
REQUIRED_COLUMNS = {
    "record_id", "lead", "clinical_feature", "beat_index", "sample_index", "value"
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    result.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    result.add_argument("--if-stale", action="store_true")
    return result


def validate_cache(cache_dir: Path) -> tuple[pd.DataFrame, dict, Path, Path]:
    parquet = cache_dir / "_target_ptb_xl_features.parquet"
    metadata_path = cache_dir / "_target_ptb_xl_features.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != 2:
        raise RuntimeError("target-event cache is not schema version 2")
    if metadata.get("records") != 2_198:
        raise RuntimeError("target-event cache does not cover 2,198 records")
    if metadata.get("parquet_sha256") != sha256_file(parquet):
        raise RuntimeError("target-event cache Parquet digest mismatch")
    frame = pd.read_parquet(parquet)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"target-event cache columns missing: {sorted(missing)}")
    if len(frame) != metadata.get("rows"):
        raise RuntimeError("target-event cache row count mismatch")
    if not np.isfinite(frame["value"].to_numpy(dtype=float)).all():
        raise RuntimeError("target-event cache contains non-finite values")
    if (frame["sample_index"] < 0).any() or (frame["sample_index"] >= 5_000).any():
        raise RuntimeError("target-event sample index is outside the 10-second signal")
    duplicated = frame.duplicated(
        ["record_id", "lead", "clinical_feature", "sample_index"]
    )
    if duplicated.any():
        raise RuntimeError("target-event cache contains duplicate feature events")
    return frame, metadata, parquet, metadata_path


def artifact_is_current(args: argparse.Namespace) -> bool:
    summary_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "target_detector_feature_eda.csv"
    subgroup_csv_path = args.output_dir / "target_detector_subgroup_coverage.csv"
    try:
        metadata_path = args.cache_dir / "_target_ptb_xl_features.json"
        metadata = json.loads(metadata_path.read_text())
        summary = json.loads(summary_path.read_text())
        return (
            metadata.get("schema_version") == 2
            and summary.get("target_cache_parquet_sha256")
            == metadata.get("parquet_sha256")
            and summary.get("builder_code_sha256") == sha256_file(Path(__file__))
            and summary.get("csv_sha256") == sha256_file(csv_path)
            and summary.get("subgroup_csv_sha256") == sha256_file(subgroup_csv_path)
            and summary.get("ptbxl_metadata_sha256") == sha256_file(args.metadata)
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def build(frame: pd.DataFrame, records_total: int) -> pd.DataFrame:
    rows = []
    for (lead, feature), group in frame.groupby(
        ["lead", "clinical_feature"], sort=True
    ):
        per_record = group.groupby("record_id").size()
        intervals = (
            group.sort_values(["record_id", "sample_index"])
            .groupby("record_id")["sample_index"]
            .diff()
            .dropna()
            .to_numpy(dtype=float)
            / 500.0
            * 1000.0
        )
        values = group["value"].to_numpy(dtype=float)
        rows.append(
            {
                "lead": lead,
                "clinical_feature": feature,
                "records_total": records_total,
                "records_detected": int(per_record.size),
                "record_detection_coverage": float(per_record.size / records_total),
                "events": int(len(group)),
                "events_per_detected_record_median": float(per_record.median()),
                "events_per_detected_record_q25": float(per_record.quantile(0.25)),
                "events_per_detected_record_q75": float(per_record.quantile(0.75)),
                "value_median": float(np.median(values)),
                "value_minimum": float(np.min(values)),
                "value_q01": float(np.quantile(values, 0.01)),
                "value_q25": float(np.quantile(values, 0.25)),
                "value_q75": float(np.quantile(values, 0.75)),
                "value_q99": float(np.quantile(values, 0.99)),
                "value_maximum": float(np.max(values)),
                "inter_event_interval_ms_median": (
                    float(np.median(intervals)) if len(intervals) else np.nan
                ),
                "inter_event_interval_ms_q05": (
                    float(np.quantile(intervals, 0.05)) if len(intervals) else np.nan
                ),
                "inter_event_interval_ms_q95": (
                    float(np.quantile(intervals, 0.95)) if len(intervals) else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=[np.number]).drop(
        columns=["inter_event_interval_ms_median", "inter_event_interval_ms_q05", "inter_event_interval_ms_q95"]
    )
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("target detector EDA contains non-finite core metrics")
    if not result["record_detection_coverage"].between(0, 1).all():
        raise RuntimeError("target detector coverage is outside [0,1]")
    return result


def load_test_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, usecols=["ecg_id", "age", "sex", "strat_fold"])
    metadata = metadata.loc[metadata["strat_fold"].eq(10)].copy()
    if len(metadata) != 2_198 or metadata["ecg_id"].duplicated().any():
        raise RuntimeError("PTB-XL test metadata is not the unique 2,198-record fold")
    metadata["record_id"] = metadata["ecg_id"].astype(str)
    metadata["sex_group"] = metadata["sex"].map({0: "sex_code_0", 1: "sex_code_1"})
    if metadata["sex_group"].isna().any():
        raise RuntimeError("unexpected PTB-XL sex code")
    metadata["age_group"] = pd.cut(
        metadata["age"],
        bins=[-np.inf, 39, 59, 79, 100, np.inf],
        labels=["age_<40", "age_40_59", "age_60_79", "age_80_100", "age_invalid_>100"],
    ).astype(str)
    return metadata


def build_subgroup_coverage(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lead, feature), events in frame.groupby(
        ["lead", "clinical_feature"], sort=True
    ):
        detected = set(events["record_id"].astype(str).unique())
        cohort = metadata.assign(detected=metadata["record_id"].isin(detected))
        for variable in ("sex_group", "age_group"):
            for group_name, group in cohort.groupby(variable, observed=True):
                rows.append(
                    {
                        "lead": lead,
                        "clinical_feature": feature,
                        "subgroup_variable": variable,
                        "subgroup": str(group_name),
                        "records": int(len(group)),
                        "records_detected": int(group["detected"].sum()),
                        "record_detection_coverage": float(group["detected"].mean()),
                    }
                )
    result = pd.DataFrame(rows)
    if not result["record_detection_coverage"].between(0, 1).all():
        raise RuntimeError("subgroup detector coverage is outside [0,1]")
    if not result.groupby(["lead", "clinical_feature", "subgroup_variable"])[
        "records"
    ].sum().eq(2_198).all():
        raise RuntimeError("subgroup denominators do not close to 2,198")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.if_stale and artifact_is_current(args):
        print(json.dumps({"status": "current", "action": "skipped"}))
        return
    frame, metadata, parquet, metadata_path = validate_cache(args.cache_dir)
    result = build(frame, metadata["records"])
    test_metadata = load_test_metadata(args.metadata)
    subgroup_result = build_subgroup_coverage(frame, test_metadata)
    csv_content = result.to_csv(index=False)
    csv_path = args.output_dir / "target_detector_feature_eda.csv"
    atomic_text(csv_path, csv_content)
    subgroup_csv_content = subgroup_result.to_csv(index=False)
    subgroup_csv_path = args.output_dir / "target_detector_subgroup_coverage.csv"
    atomic_text(subgroup_csv_path, subgroup_csv_content)
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "dataset": "ptb_xl",
        "records": metadata["records"],
        "target_cache_rows": metadata["rows"],
        "target_extractor_sha256": metadata["extractor_sha256"],
        "target_cache_parquet_sha256": metadata["parquet_sha256"],
        "target_cache_metadata_sha256": sha256_file(metadata_path),
        "builder_code_sha256": sha256_file(Path(__file__)),
        "csv_sha256": hashlib.sha256(csv_content.encode()).hexdigest(),
        "subgroup_csv_sha256": hashlib.sha256(
            subgroup_csv_content.encode()
        ).hexdigest(),
        "ptbxl_metadata_sha256": sha256_file(args.metadata),
        "feature_rows": len(result),
        "subgroup_rows": len(subgroup_result),
        "sample_rate_hz": 500,
        "signal_samples": 5_000,
        "files": {
            "eda": csv_path.name,
            "subgroups": subgroup_csv_path.name,
            "target_cache": parquet.name,
        },
    }
    atomic_text(args.output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
