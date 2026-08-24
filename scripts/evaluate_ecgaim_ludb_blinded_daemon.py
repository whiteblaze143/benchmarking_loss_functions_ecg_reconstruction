#!/usr/bin/env python3
"""CPU-only, blinded LUDB delineation daemon for factorial ECG-AIM models.

Primary endpoint
----------------
NeuroKit2 DWT, fully independent R-peak detection, nine reconstructed leads,
macro F1 at 20 ms across P/QRS/T onsets and offsets.  LUDB annotations are
used only after prediction for scoring.

Supporting analyses
-------------------
* all nine P/QRS/T onset, peak, and offset landmarks;
* one-to-one event matching at 10, 20, 40, 80, and 150 ms;
* signed timing bias, MAE, median, SD, p90, and p95 at the 150 ms match gate;
* DWT primary and prominence sensitivity methods;
* fully blind and annotation-R diagnostic modes (no boundary is supplied);
* original-signal detector ceiling and reconstruction-minus-ceiling analyses;
* per-lead and global hybrid-12-lead intervals;
* failures retained as false negatives rather than silently dropped;
* record-level and diagnostic-subgroup summaries in a dedicated SQLite DB.

The global hybrid signal uses the three actually observed LUDB leads unchanged
and the nine ECG-AIM reconstructions, matching the intended clinical object.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

# Must precede torch import.  This experiment is deliberately CPU-only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import neurokit2 as nk
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_store import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_DB as CHECKPOINT_DB,
    connect as connect_checkpoint_db,
    prune_cache,
    store_lock,
)
from scripts.evaluate_comprehensive_registry import load_adapter  # noqa: E402
from scripts.evaluate_ecgaim_ludb_daemon import (  # noqa: E402
    LEADS,
    MISSING,
    OBSERVED,
    TARGET_FS,
    TARGET_SAMPLES,
    completed_ecgaim_models,
    finite_mean,
    finite_pearson,
    load_ludb,
)


DISPLAY_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LANDMARKS = (
    "P_onset", "P_peak", "P_offset",
    "QRS_onset", "QRS_peak", "QRS_offset",
    "T_onset", "T_peak", "T_offset",
)
BOUNDARIES = ("P_onset", "P_offset", "QRS_onset", "QRS_offset", "T_onset", "T_offset")
TOLERANCES_MS = (10, 20, 40, 80, 150)
MAX_MATCH_MS = 150
INTERVALS = {
    "P_duration": ("P_onset", "P_offset"),
    "PR_interval": ("P_onset", "QRS_onset"),
    "PR_segment": ("P_offset", "QRS_onset"),
    "QRS_duration": ("QRS_onset", "QRS_offset"),
    "QT_interval": ("QRS_onset", "T_offset"),
    "JT_interval": ("QRS_offset", "T_offset"),
    "ST_segment": ("QRS_offset", "T_onset"),
    "T_duration": ("T_onset", "T_offset"),
}
STOP_REQUESTED = False


@dataclasses.dataclass
class DelineationResult:
    landmarks: dict[str, np.ndarray]
    beats: list[dict[str, float]]
    status: str
    r_peaks: np.ndarray


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum, "at": utc_now()}), flush=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_identity() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        ROOT / "scripts/evaluate_ecgaim_ludb_daemon.py",
        ROOT / "scripts/evaluate_comprehensive_registry.py",
    ):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    digest.update(nk.__version__.encode())
    return digest.hexdigest()


def empty_result(status: str) -> DelineationResult:
    return DelineationResult(
        landmarks={name: np.array([], dtype=np.int64) for name in LANDMARKS},
        beats=[], status=status, r_peaks=np.array([], dtype=np.int64),
    )


def finite_samples(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    return np.unique(array[np.isfinite(array)].astype(np.int64))


def aligned(values: Iterable[Any], length: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if len(array) < length:
        array = np.pad(array, (0, length - len(array)), constant_values=np.nan)
    return array[:length]


def monotonic_match_indices(
    reference: Iterable[Any], predicted: Iterable[Any], tolerance_samples: int
) -> list[tuple[int, int]]:
    """Maximize one-to-one matches, then minimize total absolute error."""
    left_raw = np.asarray(list(reference), dtype=float).reshape(-1)
    right_raw = np.asarray(list(predicted), dtype=float).reshape(-1)
    left_order = np.flatnonzero(np.isfinite(left_raw))
    right_order = np.flatnonzero(np.isfinite(right_raw))
    left_order = left_order[np.argsort(left_raw[left_order], kind="stable")]
    right_order = right_order[np.argsort(right_raw[right_order], kind="stable")]
    left = left_raw[left_order].astype(np.int64)
    right = right_raw[right_order].astype(np.int64)
    dynamic: list[list[tuple[int, int, tuple[tuple[int, int], ...]]]] = [
        [(0, 0, ()) for _ in range(len(right) + 1)] for _ in range(len(left) + 1)
    ]
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            candidates = [dynamic[i - 1][j], dynamic[i][j - 1]]
            error = abs(int(left[i - 1]) - int(right[j - 1]))
            if error <= tolerance_samples:
                count, negative_error, pairs = dynamic[i - 1][j - 1]
                candidates.append((count + 1, negative_error - error, pairs + ((i - 1, j - 1),)))
            dynamic[i][j] = max(candidates, key=lambda item: (item[0], item[1]))
    return [
        (int(left_order[i]), int(right_order[j]))
        for i, j in dynamic[-1][-1][2]
    ]


def safe_number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def delineate_task(task: tuple[np.ndarray, str, str, np.ndarray]) -> DelineationResult:
    signal_values, method, mode, oracle_r = task
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.ecg_clean(np.asarray(signal_values, dtype=float), sampling_rate=TARGET_FS)
            if mode == "blind":
                _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=TARGET_FS, method="neurokit")
                r_peaks = finite_samples(peak_info.get("ECG_R_Peaks", []))
            elif mode == "oracle_r":
                r_peaks = finite_samples(oracle_r)
            else:
                raise ValueError(f"unsupported mode {mode}")
            if len(r_peaks) < 2:
                return empty_result("insufficient_r_peaks")
            _, waves = nk.ecg_delineate(
                cleaned, r_peaks, sampling_rate=TARGET_FS, method=method, check=False
            )
        source = {
            "P_onset": "ECG_P_Onsets", "P_peak": "ECG_P_Peaks", "P_offset": "ECG_P_Offsets",
            "QRS_onset": "ECG_R_Onsets", "QRS_offset": "ECG_R_Offsets",
            "T_onset": "ECG_T_Onsets", "T_peak": "ECG_T_Peaks", "T_offset": "ECG_T_Offsets",
        }
        aligned_values = {name: aligned(waves.get(key, []), len(r_peaks)) for name, key in source.items()}
        aligned_values["QRS_peak"] = r_peaks.astype(float)
        beats = [
            {name: safe_number(aligned_values[name][index]) for name in LANDMARKS}
            for index in range(len(r_peaks))
        ]
        landmarks = {name: finite_samples(aligned_values[name]) for name in LANDMARKS}
        return DelineationResult(landmarks=landmarks, beats=beats, status="ok", r_peaks=r_peaks)
    except Exception as error:  # failures are retained as evaluation outcomes
        return empty_result(f"detector_error:{type(error).__name__}")


def reference_result(annotations: dict[str, list[dict[str, int | None]]]) -> DelineationResult:
    landmarks: dict[str, np.ndarray] = {}
    wave_prefix = {"P": "P", "QRS": "QRS", "T": "T"}
    for wave, prefix in wave_prefix.items():
        for local in ("onset", "peak", "offset"):
            landmarks[f"{prefix}_{local}"] = finite_samples(
                event[local] for event in annotations[wave] if event[local] is not None
            )
    qrs_events = sorted(
        [event for event in annotations["QRS"] if event["peak"] is not None],
        key=lambda event: int(event["peak"]),
    )
    p_events = sorted(
        [event for event in annotations["P"] if event["peak"] is not None],
        key=lambda event: int(event["peak"]),
    )
    t_events = sorted(
        [event for event in annotations["T"] if event["peak"] is not None],
        key=lambda event: int(event["peak"]),
    )
    beats: list[dict[str, float]] = []
    for index, qrs in enumerate(qrs_events):
        qrs_peak = int(qrs["peak"])
        previous_qrs = int(qrs_events[index - 1]["peak"]) if index else -1
        next_qrs = int(qrs_events[index + 1]["peak"]) if index + 1 < len(qrs_events) else TARGET_SAMPLES
        preceding_p = [event for event in p_events if previous_qrs < int(event["peak"]) < qrs_peak]
        following_t = [event for event in t_events if qrs_peak < int(event["peak"]) < next_qrs]
        p_event = preceding_p[-1] if preceding_p else {}
        t_event = following_t[0] if following_t else {}
        beats.append(
            {
                "P_onset": safe_number(p_event.get("onset")),
                "P_peak": safe_number(p_event.get("peak")),
                "P_offset": safe_number(p_event.get("offset")),
                "QRS_onset": safe_number(qrs.get("onset")),
                "QRS_peak": float(qrs_peak),
                "QRS_offset": safe_number(qrs.get("offset")),
                "T_onset": safe_number(t_event.get("onset")),
                "T_peak": safe_number(t_event.get("peak")),
                "T_offset": safe_number(t_event.get("offset")),
            }
        )
    return DelineationResult(
        landmarks=landmarks, beats=beats, status="reference",
        r_peaks=landmarks["QRS_peak"],
    )


def aggregate_global(per_lead: dict[str, DelineationResult]) -> DelineationResult:
    """Create CSE-style earliest-onset/latest-offset multi-lead beats."""
    canonical = per_lead["ii"]
    if len(canonical.beats) < 2:
        return empty_result("global_insufficient_lead_ii")
    canonical_peaks = np.asarray([beat["QRS_peak"] for beat in canonical.beats], dtype=float)
    aligned_by_lead: dict[str, dict[int, dict[str, float]]] = {}
    tolerance = round(MAX_MATCH_MS * TARGET_FS / 1000)
    for lead, result in per_lead.items():
        peaks = np.asarray([beat["QRS_peak"] for beat in result.beats], dtype=float)
        pairs = monotonic_match_indices(canonical_peaks, peaks, tolerance)
        aligned_by_lead[lead] = {left: result.beats[right] for left, right in pairs}
    beats: list[dict[str, float]] = []
    for beat_index in range(len(canonical.beats)):
        members = [mapping[beat_index] for mapping in aligned_by_lead.values() if beat_index in mapping]
        combined: dict[str, float] = {}
        for landmark in LANDMARKS:
            values = np.asarray([member[landmark] for member in members], dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                combined[landmark] = float("nan")
            elif landmark.endswith("onset"):
                combined[landmark] = float(values.min())
            elif landmark.endswith("offset"):
                combined[landmark] = float(values.max())
            else:
                combined[landmark] = float(np.median(values))
        beats.append(combined)
    landmarks = {
        name: finite_samples(beat[name] for beat in beats if math.isfinite(beat[name]))
        for name in LANDMARKS
    }
    return DelineationResult(
        landmarks=landmarks, beats=beats, status="ok",
        r_peaks=landmarks["QRS_peak"],
    )


def distribution_stats(errors_samples: np.ndarray) -> dict[str, float | int | None]:
    errors = np.asarray(errors_samples, dtype=float) * 1000.0 / TARGET_FS
    errors = errors[np.isfinite(errors)]
    if not len(errors):
        return {
            "matched_150": 0, "timing_bias_ms": None, "timing_sd_ms": None,
            "timing_mae_ms": None, "timing_median_abs_ms": None,
            "timing_p90_abs_ms": None, "timing_p95_abs_ms": None,
        }
    absolute = np.abs(errors)
    return {
        "matched_150": len(errors),
        "timing_bias_ms": float(errors.mean()),
        "timing_sd_ms": float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
        "timing_mae_ms": float(absolute.mean()),
        "timing_median_abs_ms": float(np.median(absolute)),
        "timing_p90_abs_ms": float(np.quantile(absolute, 0.90)),
        "timing_p95_abs_ms": float(np.quantile(absolute, 0.95)),
    }


def f1(tp: int, reference_count: int, predicted_count: int) -> float | None:
    denominator = reference_count + predicted_count
    return (2.0 * tp / denominator) if denominator else None


def delineation_row(
    model_id: str,
    source: str,
    method: str,
    mode: str,
    record_id: str,
    lead: str,
    lead_role: str,
    landmark: str,
    reference: DelineationResult,
    predicted: DelineationResult,
) -> dict[str, Any]:
    real = reference.landmarks[landmark]
    estimate = predicted.landmarks[landmark]
    row: dict[str, Any] = {
        "model_id": model_id, "source": source, "method": method, "mode": mode,
        "record_id": record_id, "lead": lead, "lead_role": lead_role,
        "landmark": landmark, "detector_status": predicted.status,
        "reference_events": len(real), "predicted_events": len(estimate),
    }
    for tolerance_ms in TOLERANCES_MS:
        tolerance = round(tolerance_ms * TARGET_FS / 1000)
        tp = len(monotonic_match_indices(real, estimate, tolerance))
        row[f"tp_{tolerance_ms}"] = tp
        row[f"f1_{tolerance_ms}"] = f1(tp, len(real), len(estimate))
    pairs = monotonic_match_indices(real, estimate, round(MAX_MATCH_MS * TARGET_FS / 1000))
    errors = np.asarray([estimate[right] - real[left] for left, right in pairs], dtype=float)
    row.update(distribution_stats(errors))
    return row


def interval_value(beat: dict[str, float], interval_name: str) -> float:
    start, end = INTERVALS[interval_name]
    if not (math.isfinite(beat[start]) and math.isfinite(beat[end])):
        return float("nan")
    value = beat[end] - beat[start]
    return float(value) if value >= 0 else float("nan")


def interval_rows_for_pair(
    model_id: str,
    source: str,
    method: str,
    mode: str,
    record_id: str,
    lead: str,
    lead_role: str,
    reference: DelineationResult,
    predicted: DelineationResult,
) -> list[dict[str, Any]]:
    ref_peaks = np.asarray([beat["QRS_peak"] for beat in reference.beats], dtype=float)
    pred_peaks = np.asarray([beat["QRS_peak"] for beat in predicted.beats], dtype=float)
    beat_pairs = monotonic_match_indices(
        ref_peaks, pred_peaks, round(MAX_MATCH_MS * TARGET_FS / 1000)
    )
    rows = []
    for interval_name in INTERVALS:
        ref_values = np.asarray([interval_value(beat, interval_name) for beat in reference.beats])
        pred_values = np.asarray([interval_value(beat, interval_name) for beat in predicted.beats])
        errors = np.asarray(
            [
                pred_values[right] - ref_values[left]
                for left, right in beat_pairs
                if np.isfinite(ref_values[left]) and np.isfinite(pred_values[right])
            ],
            dtype=float,
        )
        row: dict[str, Any] = {
            "model_id": model_id, "source": source, "method": method, "mode": mode,
            "record_id": record_id, "lead": lead, "lead_role": lead_role,
            "interval_name": interval_name, "detector_status": predicted.status,
            "reference_intervals": int(np.isfinite(ref_values).sum()),
            "predicted_intervals": int(np.isfinite(pred_values).sum()),
        }
        row.update(distribution_stats(errors))
        row["interval_f1"] = f1(
            int(row["matched_150"]), row["reference_intervals"], row["predicted_intervals"]
        )
        rows.append(row)
    # RR is scored only when two consecutive reference beats map to two
    # consecutive predicted beats.  Matching RR values by duration would lose
    # beat identity and could reward a temporally scrambled prediction.
    ref_rr = np.diff(ref_peaks[np.isfinite(ref_peaks)])
    pred_rr = np.diff(pred_peaks[np.isfinite(pred_peaks)])
    rr_errors = np.asarray(
        [
            (pred_peaks[right_b] - pred_peaks[right_a])
            - (ref_peaks[left_b] - ref_peaks[left_a])
            for (left_a, right_a), (left_b, right_b) in zip(beat_pairs, beat_pairs[1:])
            if left_b == left_a + 1 and right_b == right_a + 1
        ],
        dtype=float,
    )
    rr_row: dict[str, Any] = {
        "model_id": model_id, "source": source, "method": method, "mode": mode,
        "record_id": record_id, "lead": lead, "lead_role": lead_role,
        "interval_name": "RR_interval", "detector_status": predicted.status,
        "reference_intervals": len(ref_rr), "predicted_intervals": len(pred_rr),
    }
    rr_row.update(distribution_stats(rr_errors))
    rr_row["interval_f1"] = f1(
        int(rr_row["matched_150"]), rr_row["reference_intervals"], rr_row["predicted_intervals"]
    )
    rows.append(rr_row)
    return rows


def connect_results(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA page_size=8192")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA wal_autocheckpoint=4096")
    connection.execute("PRAGMA journal_size_limit=33554432")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS protocol_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS evaluations(
            model_id TEXT PRIMARY KEY, factorial_mask TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL, checkpoint_size_bytes INTEGER NOT NULL,
            protocol_sha256 TEXT NOT NULL, dataset_sha256 TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running','complete','error')),
            attempts INTEGER NOT NULL, started_at TEXT, completed_at TEXT,
            duration_seconds REAL, n_records INTEGER, error TEXT, primary_summary_json TEXT
        );
        CREATE TABLE IF NOT EXISTS record_metadata(
            record_id TEXT PRIMARY KEY, age REAL, sex TEXT, rhythm TEXT,
            axis TEXT, comments_json TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS record_subgroups(
            record_id TEXT NOT NULL REFERENCES record_metadata(record_id),
            subgroup TEXT NOT NULL, source TEXT NOT NULL,
            PRIMARY KEY(record_id,subgroup)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS dataset_audit(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS signal_record_metrics(
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            record_id TEXT NOT NULL, missing_pearson REAL, missing_mse REAL,
            missing_mae REAL, missing_derivative_mse REAL,
            PRIMARY KEY(model_id,record_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS signal_lead_metrics(
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            record_id TEXT NOT NULL, lead TEXT NOT NULL, lead_role TEXT NOT NULL,
            pearson REAL, mse REAL, mae REAL, derivative_mse REAL,
            PRIMARY KEY(model_id,record_id,lead)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS delineation_metrics(
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            source TEXT NOT NULL, method TEXT NOT NULL, mode TEXT NOT NULL,
            record_id TEXT NOT NULL, lead TEXT NOT NULL, lead_role TEXT NOT NULL,
            landmark TEXT NOT NULL, detector_status TEXT NOT NULL,
            reference_events INTEGER NOT NULL, predicted_events INTEGER NOT NULL,
            tp_10 INTEGER NOT NULL, f1_10 REAL, tp_20 INTEGER NOT NULL, f1_20 REAL,
            tp_40 INTEGER NOT NULL, f1_40 REAL, tp_80 INTEGER NOT NULL, f1_80 REAL,
            tp_150 INTEGER NOT NULL, f1_150 REAL, matched_150 INTEGER NOT NULL,
            timing_bias_ms REAL, timing_sd_ms REAL, timing_mae_ms REAL,
            timing_median_abs_ms REAL, timing_p90_abs_ms REAL, timing_p95_abs_ms REAL,
            PRIMARY KEY(model_id,source,method,mode,record_id,lead,landmark)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS interval_metrics(
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            source TEXT NOT NULL, method TEXT NOT NULL, mode TEXT NOT NULL,
            record_id TEXT NOT NULL, lead TEXT NOT NULL, lead_role TEXT NOT NULL,
            interval_name TEXT NOT NULL, detector_status TEXT NOT NULL,
            reference_intervals INTEGER NOT NULL, predicted_intervals INTEGER NOT NULL,
            matched_150 INTEGER NOT NULL, interval_f1 REAL,
            timing_bias_ms REAL, timing_sd_ms REAL, timing_mae_ms REAL,
            timing_median_abs_ms REAL, timing_p90_abs_ms REAL, timing_p95_abs_ms REAL,
            PRIMARY KEY(model_id,source,method,mode,record_id,lead,interval_name)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS delineation_summaries(
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            source TEXT NOT NULL, method TEXT NOT NULL, mode TEXT NOT NULL,
            subgroup TEXT NOT NULL, lead_role TEXT NOT NULL, landmark TEXT NOT NULL,
            n_records INTEGER NOT NULL, detector_success_rate REAL,
            reference_events INTEGER NOT NULL, predicted_events INTEGER NOT NULL,
            micro_f1_10 REAL, micro_f1_20 REAL, micro_f1_40 REAL,
            micro_f1_80 REAL, micro_f1_150 REAL, macro_f1_20 REAL,
            timing_bias_ms REAL, record_timing_mae_median_ms REAL,
            record_timing_p95_ms REAL,
            PRIMARY KEY(model_id,source,method,mode,subgroup,lead_role,landmark)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS interval_summaries(
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            source TEXT NOT NULL, method TEXT NOT NULL, mode TEXT NOT NULL,
            subgroup TEXT NOT NULL, lead_role TEXT NOT NULL, interval_name TEXT NOT NULL,
            n_records INTEGER NOT NULL, reference_intervals INTEGER NOT NULL,
            predicted_intervals INTEGER NOT NULL, matched_intervals INTEGER NOT NULL,
            micro_f1 REAL, timing_bias_ms REAL, record_timing_mae_median_ms REAL,
            record_timing_p95_ms REAL,
            PRIMARY KEY(model_id,source,method,mode,subgroup,lead_role,interval_name)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS signal_summaries(
            model_id TEXT PRIMARY KEY REFERENCES evaluations(model_id) ON DELETE CASCADE,
            factorial_mask TEXT NOT NULL, n_records INTEGER NOT NULL,
            pearson_p01 REAL, pearson_p05 REAL, pearson_p10 REAL,
            pearson_median REAL, failure_rate_r_lt_060 REAL,
            mse_median REAL, mse_p95 REAL, mae_p95 REAL
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS subgroup_lookup_idx ON record_subgroups(subgroup,record_id);
        CREATE VIEW IF NOT EXISTS ceiling_degradation_summaries AS
        SELECT
            model.model_id, model.method, model.mode, model.subgroup,
            model.lead_role, model.landmark,
            model.micro_f1_20 AS reconstruction_f1_20,
            ceiling.micro_f1_20 AS original_f1_20,
            model.micro_f1_20 - ceiling.micro_f1_20 AS delta_f1_20,
            model.record_timing_mae_median_ms AS reconstruction_timing_mae_ms,
            ceiling.record_timing_mae_median_ms AS original_timing_mae_ms,
            model.record_timing_mae_median_ms - ceiling.record_timing_mae_median_ms
                AS incremental_timing_mae_ms
        FROM delineation_summaries model
        JOIN delineation_summaries ceiling
          ON ceiling.model_id='__original__'
         AND model.method=ceiling.method AND model.mode=ceiling.mode
         AND model.subgroup=ceiling.subgroup AND model.lead_role=ceiling.lead_role
         AND model.landmark=ceiling.landmark
        WHERE model.model_id!='__original__';
        """
    )
    connection.commit()
    return connection


def store_protocol_and_dataset(
    connection: sqlite3.Connection,
    records: list[dict[str, Any]],
    protocol_sha: str,
    dataset_sha: str,
    methods: tuple[str, ...],
    modes: tuple[str, ...],
) -> None:
    metadata = {
        "schema_version": "2",
        "protocol": "blinded_ludb_delineation_v2",
        "protocol_sha256": protocol_sha,
        "dataset_sha256": dataset_sha,
        "neurokit2_version": nk.__version__,
        "sampling_rate_hz": str(TARGET_FS),
        "methods": json.dumps(methods),
        "modes": json.dumps(modes),
        "tolerances_ms": json.dumps(TOLERANCES_MS),
        "primary_endpoint": "DWT blind missing-lead macro-F1@20ms across six boundaries",
        "annotation_use": "scoring only; oracle_r supplies QRS peaks only and is diagnostic",
        "global_signal": "original observed I/II/V2 plus nine reconstructed leads",
        "interval_f1_semantics": "endpoint-completeness F1 on matched beats; timing accuracy reported separately",
        "materialized_subgroups": "all, sex, age, rhythm, axis, and nine prespecified diagnostic categories; exact diagnoses remain normalized in record_subgroups",
        "storage_contract": "WITHOUT ROWID raw tables; overwrite-only CSV exports; WAL truncated after each completed evaluation",
        "cpu_only": "true",
    }
    with connection:
        connection.executemany(
            "INSERT OR REPLACE INTO protocol_metadata(key,value) VALUES(?,?)", metadata.items()
        )
        for record in records:
            item = record["metadata"]
            connection.execute(
                "INSERT OR REPLACE INTO record_metadata VALUES(?,?,?,?,?,?)",
                (
                    record["record_id"], item["age"], item["sex"], item["rhythm"],
                    item["axis"], json.dumps(item["comments"], ensure_ascii=False),
                ),
            )
            connection.executemany(
                "INSERT OR REPLACE INTO record_subgroups VALUES(?,?,?)",
                [(record["record_id"], subgroup, source) for subgroup, source in item["subgroups"].items()],
            )
        audit = {
            "records": len(records), "annotation_streams": len(records) * 12,
            "missing_leads": len(MISSING), "observed_leads": len(OBSERVED),
        }
        for role, indices in (("all", range(12)), ("missing", MISSING), ("observed", OBSERVED)):
            for landmark in LANDMARKS:
                wave, local = landmark.split("_", 1)
                audit[f"{role}_{landmark}_labels"] = sum(
                    event[local] is not None
                    for record in records for lead_index in indices
                    for event in record["annotations"][LEADS[lead_index]][wave]
                )
        connection.executemany(
            "INSERT OR REPLACE INTO dataset_audit(key,value) VALUES(?,?)",
            [(key, str(value)) for key, value in audit.items()],
        )


def signal_rows(model_id: str, records: list[dict[str, Any]], reconstructions: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    for record, reconstruction in zip(records, reconstructions, strict=True):
        target = record["signal"]
        current = []
        for lead_index, lead in enumerate(LEADS):
            difference = reconstruction[lead_index] - target[lead_index]
            derivative = np.diff(reconstruction[lead_index]) - np.diff(target[lead_index])
            row = {
                "model_id": model_id, "record_id": record["record_id"], "lead": lead,
                "lead_role": "observed" if lead_index in OBSERVED else "missing",
                "pearson": finite_pearson(target[lead_index], reconstruction[lead_index]),
                "mse": float(np.square(difference).mean()),
                "mae": float(np.abs(difference).mean()),
                "derivative_mse": float(np.square(derivative).mean()),
            }
            lead_rows.append(row)
            if lead_index in MISSING:
                current.append(row)
        record_rows.append(
            {
                "model_id": model_id, "record_id": record["record_id"],
                "missing_pearson": finite_mean([row["pearson"] for row in current]),
                "missing_mse": finite_mean([row["mse"] for row in current]),
                "missing_mae": finite_mean([row["mae"] for row in current]),
                "missing_derivative_mse": finite_mean([row["derivative_mse"] for row in current]),
            }
        )
    return record_rows, lead_rows


def quantile(rows: list[dict[str, Any]], key: str, probability: float) -> float | None:
    values = np.asarray([row[key] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, probability)) if len(values) else None


def summarize_delineation(
    model_id: str,
    source: str,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    record_subgroups = {
        record["record_id"]: tuple(
            {
                "all",
                *(
                    subgroup for subgroup in record["metadata"]["subgroups"]
                    if not subgroup.startswith("diagnosis_exact:")
                ),
            }
        )
        for record in records
    }
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for subgroup in record_subgroups[row["record_id"]]:
            key = (subgroup, row["method"], row["mode"], row["lead_role"], row["landmark"])
            grouped.setdefault(key, []).append(row)
    output = []
    for (subgroup, method, mode, lead_role, landmark), selected in sorted(grouped.items()):
        reference_events = sum(row["reference_events"] for row in selected)
        predicted_events = sum(row["predicted_events"] for row in selected)
        item: dict[str, Any] = {
            "model_id": model_id, "source": source, "method": method, "mode": mode,
            "subgroup": subgroup, "lead_role": lead_role, "landmark": landmark,
            "n_records": len({row["record_id"] for row in selected}),
            "detector_success_rate": float(np.mean([row["detector_status"] == "ok" for row in selected])),
            "reference_events": reference_events, "predicted_events": predicted_events,
            "macro_f1_20": finite_mean([row["f1_20"] for row in selected if row["f1_20"] is not None]),
            "timing_bias_ms": None,
            "record_timing_mae_median_ms": quantile(selected, "timing_mae_ms", 0.50),
            "record_timing_p95_ms": quantile(selected, "timing_p95_abs_ms", 0.95),
        }
        weights = np.asarray([row["matched_150"] for row in selected], dtype=float)
        biases = np.asarray([
            row["timing_bias_ms"] if row["timing_bias_ms"] is not None else np.nan for row in selected
        ])
        valid = (weights > 0) & np.isfinite(biases)
        if valid.any():
            item["timing_bias_ms"] = float(np.average(biases[valid], weights=weights[valid]))
        for tolerance in TOLERANCES_MS:
            tp = sum(row[f"tp_{tolerance}"] for row in selected)
            item[f"micro_f1_{tolerance}"] = f1(tp, reference_events, predicted_events)
        output.append(item)
    return output


def summarize_intervals(
    model_id: str,
    source: str,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    record_subgroups = {
        record["record_id"]: tuple(
            {
                "all",
                *(
                    subgroup for subgroup in record["metadata"]["subgroups"]
                    if not subgroup.startswith("diagnosis_exact:")
                ),
            }
        )
        for record in records
    }
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for subgroup in record_subgroups[row["record_id"]]:
            key = (subgroup, row["method"], row["mode"], row["lead_role"], row["interval_name"])
            grouped.setdefault(key, []).append(row)
    output = []
    for (subgroup, method, mode, lead_role, interval_name), selected in sorted(grouped.items()):
        refs = sum(row["reference_intervals"] for row in selected)
        preds = sum(row["predicted_intervals"] for row in selected)
        matched = sum(row["matched_150"] for row in selected)
        weights = np.asarray([row["matched_150"] for row in selected], dtype=float)
        biases = np.asarray([
            row["timing_bias_ms"] if row["timing_bias_ms"] is not None else np.nan for row in selected
        ])
        valid = (weights > 0) & np.isfinite(biases)
        output.append(
            {
                "model_id": model_id, "source": source, "method": method, "mode": mode,
                "subgroup": subgroup, "lead_role": lead_role, "interval_name": interval_name,
                "n_records": len({row["record_id"] for row in selected}),
                "reference_intervals": refs, "predicted_intervals": preds,
                "matched_intervals": matched, "micro_f1": f1(matched, refs, preds),
                "timing_bias_ms": float(np.average(biases[valid], weights=weights[valid])) if valid.any() else None,
                "record_timing_mae_median_ms": quantile(selected, "timing_mae_ms", 0.50),
                "record_timing_p95_ms": quantile(selected, "timing_p95_abs_ms", 0.95),
            }
        )
    return output


def primary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["source"] == "reconstruction" and row["method"] == "dwt"
        and row["mode"] == "blind" and row["subgroup"] == "all"
        and row["lead_role"] == "missing" and row["landmark"] in BOUNDARIES
    ]
    return {
        "endpoint": "DWT blind missing-lead macro-F1@20ms across six boundaries",
        "boundary_macro_f1_20": finite_mean([row["micro_f1_20"] for row in selected if row["micro_f1_20"] is not None]),
        "boundary_min_f1_20": min((row["micro_f1_20"] for row in selected if row["micro_f1_20"] is not None), default=None),
        "boundaries_present": len(selected),
    }


DELINEATION_COLUMNS = (
    "model_id,source,method,mode,record_id,lead,lead_role,landmark,detector_status,"
    "reference_events,predicted_events,tp_10,f1_10,tp_20,f1_20,tp_40,f1_40,"
    "tp_80,f1_80,tp_150,f1_150,matched_150,timing_bias_ms,timing_sd_ms,"
    "timing_mae_ms,timing_median_abs_ms,timing_p90_abs_ms,timing_p95_abs_ms"
)
INTERVAL_COLUMNS = (
    "model_id,source,method,mode,record_id,lead,lead_role,interval_name,detector_status,"
    "reference_intervals,predicted_intervals,matched_150,interval_f1,timing_bias_ms,"
    "timing_sd_ms,timing_mae_ms,timing_median_abs_ms,timing_p90_abs_ms,timing_p95_abs_ms"
)


def insert_named(connection: sqlite3.Connection, table: str, columns: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    names = columns.split(",")
    placeholders = ",".join(f":{name}" for name in names)
    connection.executemany(f"INSERT INTO {table}({columns}) VALUES({placeholders})", rows)


def checkpoint_wal(connection: sqlite3.Connection) -> None:
    """Bound transient disk use after each atomic evaluation commit."""
    busy, log_pages, checkpointed_pages = connection.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    ).fetchone()
    if busy:
        print(
            json.dumps(
                {
                    "event": "wal_checkpoint_busy", "log_pages": log_pages,
                    "checkpointed_pages": checkpointed_pages, "at": utc_now(),
                }
            ),
            flush=True,
        )


def sufficient_disk_space(path: Path, minimum_free_gb: float) -> tuple[bool, int]:
    """Refuse to begin another atomic model evaluation below the disk reserve."""
    free_bytes = shutil.disk_usage(path).free
    minimum_free_bytes = int(minimum_free_gb * (1024 ** 3))
    return free_bytes >= minimum_free_bytes, free_bytes


def store_evaluation_outputs(
    connection: sqlite3.Connection,
    model_id: str,
    mask: str,
    signal_record_rows: list[dict[str, Any]],
    signal_lead_rows: list[dict[str, Any]],
    delineation_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    source = "original" if model_id == "__original__" else "reconstruction"
    delineation_summaries = summarize_delineation(model_id, source, delineation_rows, records)
    interval_summaries = summarize_intervals(model_id, source, interval_rows, records)
    primary = primary_summary(delineation_summaries) if source == "reconstruction" else {}
    with connection:
        if signal_record_rows:
            insert_named(
                connection, "signal_record_metrics",
                "model_id,record_id,missing_pearson,missing_mse,missing_mae,missing_derivative_mse",
                signal_record_rows,
            )
            insert_named(
                connection, "signal_lead_metrics",
                "model_id,record_id,lead,lead_role,pearson,mse,mae,derivative_mse",
                signal_lead_rows,
            )
            pearson = np.asarray([row["missing_pearson"] for row in signal_record_rows], dtype=float)
            signal_summary = {
                "model_id": model_id, "factorial_mask": mask, "n_records": len(signal_record_rows),
                "pearson_p01": quantile(signal_record_rows, "missing_pearson", 0.01),
                "pearson_p05": quantile(signal_record_rows, "missing_pearson", 0.05),
                "pearson_p10": quantile(signal_record_rows, "missing_pearson", 0.10),
                "pearson_median": quantile(signal_record_rows, "missing_pearson", 0.50),
                "failure_rate_r_lt_060": float(np.mean(pearson < 0.60)),
                "mse_median": quantile(signal_record_rows, "missing_mse", 0.50),
                "mse_p95": quantile(signal_record_rows, "missing_mse", 0.95),
                "mae_p95": quantile(signal_record_rows, "missing_mae", 0.95),
            }
            insert_named(
                connection, "signal_summaries",
                "model_id,factorial_mask,n_records,pearson_p01,pearson_p05,pearson_p10,"
                "pearson_median,failure_rate_r_lt_060,mse_median,mse_p95,mae_p95",
                [signal_summary],
            )
        insert_named(connection, "delineation_metrics", DELINEATION_COLUMNS, delineation_rows)
        insert_named(connection, "interval_metrics", INTERVAL_COLUMNS, interval_rows)
        insert_named(
            connection, "delineation_summaries",
            "model_id,source,method,mode,subgroup,lead_role,landmark,n_records,"
            "detector_success_rate,reference_events,predicted_events,micro_f1_10,"
            "micro_f1_20,micro_f1_40,micro_f1_80,micro_f1_150,macro_f1_20,"
            "timing_bias_ms,record_timing_mae_median_ms,record_timing_p95_ms",
            delineation_summaries,
        )
        insert_named(
            connection, "interval_summaries",
            "model_id,source,method,mode,subgroup,lead_role,interval_name,n_records,"
            "reference_intervals,predicted_intervals,matched_intervals,micro_f1,"
            "timing_bias_ms,record_timing_mae_median_ms,record_timing_p95_ms",
            interval_summaries,
        )
    return primary


def run_delineations(
    executor: concurrent.futures.ProcessPoolExecutor,
    tasks: list[tuple[np.ndarray, str, str, np.ndarray]],
) -> list[DelineationResult]:
    return list(executor.map(delineate_task, tasks, chunksize=4))


def build_original_cache(
    connection: sqlite3.Connection,
    records: list[dict[str, Any]],
    methods: tuple[str, ...],
    modes: tuple[str, ...],
    executor: concurrent.futures.ProcessPoolExecutor,
    protocol_sha: str,
    dataset_sha: str,
) -> dict[tuple[str, str, str, str], DelineationResult]:
    model_id = "__original__"
    connection.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
    connection.execute(
        "INSERT INTO evaluations VALUES(?,?,?,?,?,?,'running',1,?,?,NULL,?,NULL,NULL)",
        (model_id, "original", "original_ludb", 0, protocol_sha, dataset_sha, utc_now(), None, len(records)),
    )
    connection.commit()
    started = time.perf_counter()
    tasks = []
    keys = []
    references: dict[tuple[str, str], DelineationResult] = {}
    for record in records:
        for lead_index, lead in enumerate(LEADS):
            reference = reference_result(record["annotations"][lead])
            references[(record["record_id"], lead)] = reference
            for method in methods:
                for mode in modes:
                    supplied_r = reference.r_peaks if mode == "oracle_r" else np.array([], dtype=np.int64)
                    tasks.append((record["signal"][lead_index], method, mode, supplied_r))
                    keys.append((record["record_id"], lead, method, mode))
    results = run_delineations(executor, tasks)
    cache = dict(zip(keys, results, strict=True))
    rows: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for record in records:
        record_id = record["record_id"]
        for method in methods:
            for mode in modes:
                per_reference = {lead: references[(record_id, lead)] for lead in LEADS}
                per_predicted = {lead: cache[(record_id, lead, method, mode)] for lead in LEADS}
                for lead_index, lead in enumerate(LEADS):
                    role = "observed" if lead_index in OBSERVED else "missing"
                    for landmark in LANDMARKS:
                        rows.append(delineation_row(
                            model_id, "original", method, mode, record_id,
                            DISPLAY_LEADS[lead_index], role, landmark,
                            per_reference[lead], per_predicted[lead],
                        ))
                    intervals.extend(interval_rows_for_pair(
                        model_id, "original", method, mode, record_id,
                        DISPLAY_LEADS[lead_index], role, per_reference[lead], per_predicted[lead],
                    ))
                global_reference = aggregate_global(per_reference)
                global_predicted = aggregate_global(per_predicted)
                for landmark in LANDMARKS:
                    rows.append(delineation_row(
                        model_id, "original", method, mode, record_id, "GLOBAL", "global",
                        landmark, global_reference, global_predicted,
                    ))
                intervals.extend(interval_rows_for_pair(
                    model_id, "original", method, mode, record_id, "GLOBAL", "global",
                    global_reference, global_predicted,
                ))
    store_evaluation_outputs(connection, model_id, "original", [], [], rows, intervals, records)
    with connection:
        connection.execute(
            "UPDATE evaluations SET status='complete',completed_at=?,duration_seconds=?,error=NULL WHERE model_id=?",
            (utc_now(), time.perf_counter() - started, model_id),
        )
    checkpoint_wal(connection)
    print(json.dumps({"event": "original_ceiling_complete", "rows": len(rows), "at": utc_now()}), flush=True)
    return cache


def evaluate_model(
    connection: sqlite3.Connection,
    identity: dict[str, Any],
    records: list[dict[str, Any]],
    methods: tuple[str, ...],
    modes: tuple[str, ...],
    executor: concurrent.futures.ProcessPoolExecutor,
    original_cache: dict[tuple[str, str, str, str], DelineationResult],
    protocol_sha: str,
    dataset_sha: str,
    batch_size: int,
) -> None:
    model_id = identity["model_id"]
    old = connection.execute("SELECT attempts FROM evaluations WHERE model_id=?", (model_id,)).fetchone()
    attempts = int(old[0]) + 1 if old else 1
    connection.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
    connection.execute(
        "INSERT INTO evaluations VALUES(?,?,?,?,?,?,'running',?,?,NULL,NULL,?,NULL,NULL)",
        (
            model_id, identity["factorial_mask"], identity["sha256"], identity["size_bytes"],
            protocol_sha, dataset_sha, attempts, utc_now(), len(records),
        ),
    )
    connection.commit()
    started = time.perf_counter()
    adapter = None
    try:
        adapter = load_adapter(
            {
                "id": model_id, "kind": "alitok",
                "checkpoint": f"checkpoints/factorial_ecg_aim_{identity['factorial_mask']}_s42.pt",
                "observed_leads": list(OBSERVED),
            },
            torch.device("cpu"),
        )
        reconstructed_batches = []
        for offset in range(0, len(records), batch_size):
            target = torch.from_numpy(np.stack([record["signal"] for record in records[offset:offset + batch_size]]))
            with torch.inference_mode():
                reconstructed_batches.append(adapter.reconstruct(target).cpu().numpy().astype(np.float32))
        reconstructions = np.concatenate(reconstructed_batches, axis=0)
        signal_record_rows, signal_lead_rows = signal_rows(model_id, records, reconstructions)

        tasks = []
        keys = []
        references: dict[tuple[str, str], DelineationResult] = {}
        for record, reconstruction in zip(records, reconstructions, strict=True):
            for lead_index in MISSING:
                lead = LEADS[lead_index]
                reference = reference_result(record["annotations"][lead])
                references[(record["record_id"], lead)] = reference
                for method in methods:
                    for mode in modes:
                        supplied_r = reference.r_peaks if mode == "oracle_r" else np.array([], dtype=np.int64)
                        tasks.append((reconstruction[lead_index], method, mode, supplied_r))
                        keys.append((record["record_id"], lead, method, mode))
        results = run_delineations(executor, tasks)
        reconstructed_cache = dict(zip(keys, results, strict=True))
        rows: list[dict[str, Any]] = []
        intervals: list[dict[str, Any]] = []
        for record in records:
            record_id = record["record_id"]
            per_reference = {
                lead: reference_result(record["annotations"][lead]) for lead in LEADS
            }
            for method in methods:
                for mode in modes:
                    hybrid = {
                        lead: (
                            original_cache[(record_id, lead, method, mode)]
                            if lead_index in OBSERVED
                            else reconstructed_cache[(record_id, lead, method, mode)]
                        )
                        for lead_index, lead in enumerate(LEADS)
                    }
                    for lead_index in MISSING:
                        lead = LEADS[lead_index]
                        predicted = hybrid[lead]
                        reference = per_reference[lead]
                        for landmark in LANDMARKS:
                            rows.append(delineation_row(
                                model_id, "reconstruction", method, mode, record_id,
                                DISPLAY_LEADS[lead_index], "missing", landmark,
                                reference, predicted,
                            ))
                        intervals.extend(interval_rows_for_pair(
                            model_id, "reconstruction", method, mode, record_id,
                            DISPLAY_LEADS[lead_index], "missing", reference, predicted,
                        ))
                    global_reference = aggregate_global(per_reference)
                    global_predicted = aggregate_global(hybrid)
                    for landmark in LANDMARKS:
                        rows.append(delineation_row(
                            model_id, "reconstruction", method, mode, record_id,
                            "GLOBAL", "global", landmark, global_reference, global_predicted,
                        ))
                    intervals.extend(interval_rows_for_pair(
                        model_id, "reconstruction", method, mode, record_id,
                        "GLOBAL", "global", global_reference, global_predicted,
                    ))
        primary = store_evaluation_outputs(
            connection, model_id, identity["factorial_mask"], signal_record_rows,
            signal_lead_rows, rows, intervals, records,
        )
        with connection:
            connection.execute(
                "UPDATE evaluations SET status='complete',completed_at=?,duration_seconds=?,error=NULL,primary_summary_json=? WHERE model_id=?",
                (utc_now(), time.perf_counter() - started, json.dumps(primary, allow_nan=False), model_id),
            )
        print(json.dumps({"event": "model_complete", "model_id": model_id, **primary}, allow_nan=False), flush=True)
    except Exception as error:
        with connection:
            connection.execute(
                "UPDATE evaluations SET status='error',completed_at=?,duration_seconds=?,error=? WHERE model_id=?",
                (utc_now(), time.perf_counter() - started, f"{type(error).__name__}: {error}", model_id),
            )
        print(json.dumps({"event": "model_error", "model_id": model_id, "error": repr(error)}), flush=True)
        raise
    finally:
        del adapter
        with store_lock(CHECKPOINT_DB):
            catalog = connect_checkpoint_db(CHECKPOINT_DB)
            try:
                prune_cache(catalog, DEFAULT_CACHE_DIR, 0.0)
            finally:
                catalog.close()


def is_complete(
    connection: sqlite3.Connection,
    identity: dict[str, Any],
    protocol_sha: str,
    dataset_sha: str,
) -> bool:
    row = connection.execute("SELECT * FROM evaluations WHERE model_id=?", (identity["model_id"],)).fetchone()
    return bool(
        row and row["status"] == "complete" and row["checkpoint_sha256"] == identity["sha256"]
        and row["protocol_sha256"] == protocol_sha and row["dataset_sha256"] == dataset_sha
    )


def write_csv(connection: sqlite3.Connection, query: str, output: Path) -> None:
    rows = connection.execute(query).fetchall()
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    columns = rows[0].keys()
    with temporary.open("w") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join("" if row[key] is None else str(row[key]) for key in columns) + "\n")
    os.replace(temporary, output)


def write_exports(connection: sqlite3.Connection, output_dir: Path) -> None:
    ranking_query = """
        SELECT e.factorial_mask, e.model_id,
               json_extract(e.primary_summary_json,'$.boundary_macro_f1_20') AS boundary_macro_f1_20,
               json_extract(e.primary_summary_json,'$.boundary_min_f1_20') AS worst_boundary_f1_20,
               s.pearson_p01,s.pearson_p05,s.pearson_median,s.failure_rate_r_lt_060,
               s.mse_p95,s.mae_p95,e.duration_seconds
        FROM evaluations e JOIN signal_summaries s USING(model_id)
        WHERE e.status='complete' AND e.model_id!='__original__'
        ORDER BY boundary_macro_f1_20 DESC, worst_boundary_f1_20 DESC,
                 s.pearson_p01 DESC, s.mse_p95 ASC
    """
    write_csv(connection, ranking_query, output_dir / "blinded_robustness_ranking.csv")
    for table in ("delineation_summaries", "interval_summaries", "signal_summaries"):
        write_csv(connection, f"SELECT * FROM {table} ORDER BY 1,2,3,4,5,6,7", output_dir / f"{table}.csv")
    write_csv(
        connection,
        "SELECT * FROM ceiling_degradation_summaries ORDER BY 1,2,3,4,5,6",
        output_dir / "ceiling_degradation_summaries.csv",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--queue", type=Path, default=ROOT / "refine-logs/queue_3arch/queue_state.json")
    result.add_argument("--checkpoint-db", type=Path, default=CHECKPOINT_DB)
    result.add_argument("--results-db", type=Path, default=ROOT / "results/ecgaim_ludb/ecgaim_ludb_blinded.sqlite")
    result.add_argument("--output-dir", type=Path, default=ROOT / "results/ecgaim_ludb")
    result.add_argument("--ludb-root", type=Path, default=ROOT / "data/ludb")
    result.add_argument("--poll-seconds", type=int, default=300)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--torch-threads", type=int, default=6)
    result.add_argument("--workers", type=int, default=6)
    result.add_argument("--methods", default="dwt,prominence")
    result.add_argument("--modes", default="blind,oracle_r")
    result.add_argument("--max-records", type=int, default=0)
    result.add_argument("--max-models", type=int, default=0)
    result.add_argument(
        "--min-free-gb", type=float, default=5.0,
        help="Pause before starting a model if the results filesystem has less free space",
    )
    result.add_argument("--model-id")
    result.add_argument("--once", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if not methods or set(methods) - {"dwt", "prominence"}:
        raise ValueError("methods must be a nonempty subset of dwt,prominence")
    if not modes or set(modes) - {"blind", "oracle_r"}:
        raise ValueError("modes must be a nonempty subset of blind,oracle_r")
    if min(args.poll_seconds, args.batch_size, args.torch_threads, args.workers) < 1:
        raise ValueError("poll, batch, thread, and worker values must be positive")
    if args.min_free_gb < 0:
        raise ValueError("min-free-gb must be nonnegative")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only contract violated")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    records, dataset_sha = load_ludb(args.ludb_root, args.max_records)
    protocol_sha = protocol_identity()
    connection = connect_results(args.results_db)
    store_protocol_and_dataset(connection, records, protocol_sha, dataset_sha, methods, modes)
    print(json.dumps({
        "event": "daemon_started", "records": len(records), "device": "cpu",
        "workers": args.workers, "methods": methods, "modes": modes,
        "results_db": str(args.results_db), "min_free_gb": args.min_free_gb,
        "at": utc_now(),
    }), flush=True)
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            original_cache = build_original_cache(
                connection, records, methods, modes, executor, protocol_sha, dataset_sha
            )
            while not STOP_REQUESTED:
                eligible = completed_ecgaim_models(args.queue, args.checkpoint_db)
                if args.model_id:
                    eligible = [row for row in eligible if row["model_id"] == args.model_id]
                    if not eligible:
                        raise ValueError(f"completed/cataloged ECG-AIM model not found: {args.model_id}")
                pending = [
                    row for row in eligible
                    if not is_complete(connection, row, protocol_sha, dataset_sha)
                ]
                if args.max_models:
                    pending = pending[:args.max_models]
                print(json.dumps({
                    "event": "pass_started", "eligible": len(eligible),
                    "pending": len(pending), "at": utc_now(),
                }), flush=True)
                for identity in pending:
                    if STOP_REQUESTED:
                        break
                    has_space, free_bytes = sufficient_disk_space(
                        args.results_db.parent, args.min_free_gb
                    )
                    if not has_space:
                        print(json.dumps({
                            "event": "disk_space_pause", "free_bytes": free_bytes,
                            "min_free_gb": args.min_free_gb, "next_model": identity["model_id"],
                            "at": utc_now(),
                        }), flush=True)
                        break
                    evaluate_model(
                        connection, identity, records, methods, modes, executor,
                        original_cache, protocol_sha, dataset_sha, args.batch_size,
                    )
                    write_exports(connection, args.output_dir)
                    checkpoint_wal(connection)
                write_exports(connection, args.output_dir)
                if args.once or args.model_id or STOP_REQUESTED:
                    break
                deadline = time.monotonic() + args.poll_seconds
                while not STOP_REQUESTED and time.monotonic() < deadline:
                    time.sleep(min(1.0, deadline - time.monotonic()))
    finally:
        connection.close()
    print(json.dumps({"event": "daemon_stopped", "at": utc_now()}), flush=True)


if __name__ == "__main__":
    main()
