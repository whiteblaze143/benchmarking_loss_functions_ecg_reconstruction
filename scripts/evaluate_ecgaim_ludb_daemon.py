#!/usr/bin/env python3
"""Focused, resumable LUDB evaluation daemon for factorial ECG-AIM models.

The evaluator watches the three-architecture queue for completed ECG-AIM jobs,
loads immutable checkpoints through the checkpoint catalog, and writes its own
SQLite database.  It intentionally computes inexpensive annotation-aligned
signal and landmark metrics rather than running a detector on every generated
lead.  This makes it suitable for ranking the full ECG-AIM loss grid by lower-
tail robustness while training and checkpoint archival continue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Match the former evaluation daemon: CPU-only even on a CUDA host.  This is
# intentionally set before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import wfdb


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


LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
OBSERVED = (0, 1, 7)
MISSING = tuple(index for index in range(12) if index not in OBSERVED)
TARGET_SAMPLES = 5000
TARGET_FS = 500
MODEL_RE = re.compile(r"^ecg_aim_f_(\d{7})_s42$")
STOP_REQUESTED = False


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


def connect_results(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            model_id TEXT PRIMARY KEY,
            factorial_mask TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL,
            checkpoint_size_bytes INTEGER NOT NULL,
            evaluator_sha256 TEXT NOT NULL,
            dataset_sha256 TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running','complete','error')),
            attempts INTEGER NOT NULL DEFAULT 0,
            started_at TEXT,
            completed_at TEXT,
            duration_seconds REAL,
            n_records INTEGER,
            error TEXT,
            summary_json TEXT
        );

        CREATE TABLE IF NOT EXISTS record_metrics (
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            record_id TEXT NOT NULL,
            missing_pearson REAL,
            missing_mse REAL,
            missing_mae REAL,
            missing_derivative_mse REAL,
            v1_pearson REAL,
            v1_mse REAL,
            PRIMARY KEY(model_id, record_id)
        );

        CREATE TABLE IF NOT EXISTS record_metadata (
            record_id TEXT PRIMARY KEY,
            age REAL,
            sex TEXT,
            rhythm TEXT,
            axis TEXT,
            comments_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS record_subgroups (
            record_id TEXT NOT NULL REFERENCES record_metadata(record_id),
            subgroup TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY(record_id, subgroup)
        );

        CREATE TABLE IF NOT EXISTS dataset_audit (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lead_metrics (
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            record_id TEXT NOT NULL,
            lead TEXT NOT NULL,
            lead_role TEXT NOT NULL,
            pearson REAL,
            mse REAL,
            mae REAL,
            derivative_mse REAL,
            PRIMARY KEY(model_id, record_id, lead)
        );

        CREATE TABLE IF NOT EXISTS landmark_metrics (
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            record_id TEXT NOT NULL,
            lead TEXT NOT NULL,
            wave TEXT NOT NULL,
            landmark TEXT NOT NULL,
            n_labels INTEGER NOT NULL,
            voltage_mae_mean_mv REAL,
            voltage_mae_median_mv REAL,
            voltage_mae_p95_mv REAL,
            voltage_bias_mv REAL,
            timing_abs_mean_ms REAL,
            timing_abs_median_ms REAL,
            timing_abs_p95_ms REAL,
            timing_bias_ms REAL,
            PRIMARY KEY(model_id, record_id, lead, wave, landmark)
        );

        CREATE TABLE IF NOT EXISTS wave_metrics (
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            record_id TEXT NOT NULL,
            lead TEXT NOT NULL,
            wave TEXT NOT NULL,
            n_waves INTEGER NOT NULL,
            window_pearson_mean REAL,
            window_pearson_p05 REAL,
            window_mse_mean REAL,
            window_mse_p95 REAL,
            window_mae_mean REAL,
            area_abs_error_mean_mv REAL,
            PRIMARY KEY(model_id, record_id, lead, wave)
        );

        CREATE TABLE IF NOT EXISTS interval_metrics (
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            record_id TEXT NOT NULL,
            lead TEXT NOT NULL,
            interval_name TEXT NOT NULL,
            n_intervals INTEGER NOT NULL,
            error_bias_ms REAL,
            error_abs_mean_ms REAL,
            error_abs_median_ms REAL,
            error_abs_p95_ms REAL,
            PRIMARY KEY(model_id, record_id, lead, interval_name)
        );

        CREATE TABLE IF NOT EXISTS summaries (
            model_id TEXT PRIMARY KEY REFERENCES evaluations(model_id) ON DELETE CASCADE,
            factorial_mask TEXT NOT NULL,
            n_records INTEGER NOT NULL,
            pearson_p01 REAL,
            pearson_p05 REAL,
            pearson_p10 REAL,
            pearson_p25 REAL,
            pearson_median REAL,
            pearson_mean REAL,
            failure_rate_r_lt_040 REAL,
            failure_rate_r_lt_060 REAL,
            mse_median REAL,
            mse_p90 REAL,
            mse_p95 REAL,
            mae_p95 REAL,
            derivative_mse_p95 REAL,
            v1_pearson_p05 REAL,
            v1_pearson_median REAL,
            v1_failure_rate_r_lt_020 REAL,
            qrs_peak_voltage_p95_mv REAL,
            qrs_peak_timing_p95_ms REAL,
            qrs_duration_error_p95_ms REAL,
            qt_error_p95_ms REAL,
            completed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS landmark_summaries (
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            wave TEXT NOT NULL,
            landmark TEXT NOT NULL,
            records INTEGER NOT NULL,
            labels INTEGER NOT NULL,
            voltage_mae_median_mv REAL,
            voltage_mae_p95_mv REAL,
            voltage_bias_mv REAL,
            timing_abs_median_ms REAL,
            timing_abs_p95_ms REAL,
            timing_bias_ms REAL,
            PRIMARY KEY(model_id, wave, landmark)
        );

        CREATE TABLE IF NOT EXISTS wave_summaries (
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            wave TEXT NOT NULL,
            records INTEGER NOT NULL,
            waves INTEGER NOT NULL,
            window_pearson_p05 REAL,
            window_pearson_median REAL,
            window_mse_p95 REAL,
            area_abs_error_p95_mv REAL,
            PRIMARY KEY(model_id, wave)
        );

        CREATE TABLE IF NOT EXISTS interval_summaries (
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            interval_name TEXT NOT NULL,
            records INTEGER NOT NULL,
            intervals INTEGER NOT NULL,
            error_bias_ms REAL,
            error_abs_median_ms REAL,
            error_abs_p95_ms REAL,
            PRIMARY KEY(model_id, interval_name)
        );

        CREATE TABLE IF NOT EXISTS subgroup_summaries (
            model_id TEXT NOT NULL REFERENCES evaluations(model_id) ON DELETE CASCADE,
            subgroup TEXT NOT NULL,
            n_records INTEGER NOT NULL,
            pearson_p05 REAL,
            pearson_median REAL,
            failure_rate_r_lt_060 REAL,
            mse_p95 REAL,
            qrs_peak_timing_p95_ms REAL,
            qt_error_p95_ms REAL,
            PRIMARY KEY(model_id, subgroup)
        );

        CREATE INDEX IF NOT EXISTS evaluations_status_idx ON evaluations(status);
        CREATE INDEX IF NOT EXISTS record_metrics_model_idx ON record_metrics(model_id);
        CREATE INDEX IF NOT EXISTS lead_metrics_model_idx ON lead_metrics(model_id);
        CREATE INDEX IF NOT EXISTS landmark_metrics_model_idx ON landmark_metrics(model_id);
        CREATE INDEX IF NOT EXISTS wave_metrics_model_idx ON wave_metrics(model_id);
        CREATE INDEX IF NOT EXISTS interval_metrics_model_idx ON interval_metrics(model_id);
        CREATE INDEX IF NOT EXISTS record_subgroups_subgroup_idx ON record_subgroups(subgroup);

        CREATE VIEW IF NOT EXISTS extreme_robustness_ranking AS
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY pearson_p01 DESC, pearson_p05 DESC,
                         failure_rate_r_lt_060 ASC, qrs_peak_timing_p95_ms ASC,
                         qt_error_p95_ms ASC, mse_p95 ASC
            ) AS robustness_rank,
            *
        FROM summaries;
        """
    )
    connection.commit()
    return connection


def tree_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*"), key=lambda value: value.name):
        if not path.is_file() or path.suffix.lower() not in {
            ".dat", ".hea", ".i", ".ii", ".iii", ".avr", ".avl", ".avf",
            ".v1", ".v2", ".v3", ".v4", ".v5", ".v6",
        }:
            continue
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\n")
    return digest.hexdigest()


WAVE_SYMBOLS = {"p": "P", "N": "QRS", "t": "T"}
LANDMARKS = ("onset", "peak", "offset")


def parse_annotations(record_path: Path, lead: str) -> dict[str, list[dict[str, int | None]]]:
    """Assign every raw boundary marker exactly once, including LUDB edge cases."""
    annotation = wfdb.rdann(str(record_path), extension=lead)
    result: dict[str, list[dict[str, int | None]]] = {
        wave: [
            {"onset": None, "peak": int(sample), "offset": None}
            for symbol, sample in zip(annotation.symbol, annotation.sample)
            if WAVE_SYMBOLS.get(symbol) == wave
        ]
        for wave in ("P", "QRS", "T")
    }
    peak_index = [
        (wave, index, int(event["peak"]))
        for wave, events in result.items()
        for index, event in enumerate(events)
    ]
    extras: dict[str, list[dict[str, int | None]]] = {"P": [], "QRS": [], "T": []}
    for landmark, boundary_symbol in (("onset", "("), ("offset", ")")):
        boundaries = [
            int(sample) for symbol, sample in zip(annotation.symbol, annotation.sample)
            if symbol == boundary_symbol
        ]
        assigned: dict[tuple[str, int], list[int]] = {}
        for sample in boundaries:
            def score(item: tuple[str, int, int]) -> tuple[int, int]:
                _wave, _index, peak = item
                wrong_side = sample > peak if landmark == "onset" else sample < peak
                return (abs(sample - peak) + (250 if wrong_side else 0), abs(sample - peak))

            wave, index, _peak = min(peak_index, key=score)
            assigned.setdefault((wave, index), []).append(sample)
        for (wave, index), samples in assigned.items():
            peak = int(result[wave][index]["peak"])
            primary = min(
                samples,
                key=lambda sample: (
                    abs(sample - peak) + (250 if (sample > peak if landmark == "onset" else sample < peak) else 0),
                    abs(sample - peak),
                ),
            )
            result[wave][index][landmark] = primary
            primary_consumed = False
            for sample in samples:
                if sample == primary and not primary_consumed:
                    primary_consumed = True
                    continue
                extra = {"onset": None, "peak": None, "offset": None}
                extra[landmark] = sample
                extras[wave].append(extra)
    for wave in result:
        result[wave].extend(extras[wave])
    return result


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_metadata(comments: list[str]) -> dict[str, Any]:
    age = None
    sex = None
    rhythm = None
    axis = None
    diagnoses: list[str] = []
    for raw in comments:
        value = raw.strip()
        lower = value.lower()
        if lower.startswith("<age>:"):
            try:
                age = float(value.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif lower.startswith("<sex>:"):
            sex = value.split(":", 1)[1].strip().upper()
        elif lower.startswith("rhythm:"):
            rhythm = value.split(":", 1)[1].strip().rstrip(".")
        elif lower.startswith("electric axis of the heart:"):
            axis = value.split(":", 1)[1].strip().rstrip(".")
        elif value and lower != "<diagnoses>:":
            diagnoses.append(value.rstrip("."))

    subgroups: dict[str, str] = {"all": "derived"}
    if sex:
        subgroups[f"sex:{slug(sex)}"] = "header"
    if age is not None:
        age_band = "lt40" if age < 40 else "40_64" if age < 65 else "ge65"
        subgroups[f"age:{age_band}"] = "derived"
    if rhythm:
        subgroups[f"rhythm:{slug(rhythm)}"] = "header"
    if axis:
        subgroups[f"axis:{slug(axis)}"] = "header"
    for diagnosis in diagnoses:
        subgroups[f"diagnosis_exact:{slug(diagnosis)}"] = "header"

    joined = " ".join(filter(None, [rhythm, axis, *diagnoses])).lower()
    category_patterns = {
        "category:atrial_fibrillation_or_flutter": ("atrial fibrillation", "atrial flutter"),
        "category:bundle_branch_block": ("bundle branch block", "hemiblock"),
        "category:av_or_sinoatrial_block": ("av block", "av-block", "sinoatrial block"),
        "category:extrasystole": ("extrasystole", "pac", "pvc"),
        "category:hypertrophy_or_overload": ("hypertrophy", "overload"),
        "category:ischemia_or_scar": ("ischemia", "scar", "stemi", "nstemi"),
        "category:repolarization_abnormality": ("repolarization",),
        "category:cardiac_pacing": ("pacing", "pacemaker"),
        "category:axis_deviation": ("axis deviation",),
    }
    for subgroup, patterns in category_patterns.items():
        if any(pattern in joined for pattern in patterns):
            subgroups[subgroup] = "derived"
    return {
        "age": age,
        "sex": sex,
        "rhythm": rhythm,
        "axis": axis,
        "comments": comments,
        "diagnoses": diagnoses,
        "subgroups": subgroups,
    }


def load_ludb(root: Path, max_records: int = 0) -> tuple[list[dict[str, Any]], str]:
    record_ids = sorted({path.stem for path in root.glob("*.dat")}, key=int)
    if max_records:
        record_ids = record_ids[:max_records]
    records: list[dict[str, Any]] = []
    for record_id in record_ids:
        path = root / record_id
        record = wfdb.rdrecord(str(path), physical=True)
        lookup = {name.lower(): index for index, name in enumerate(record.sig_name)}
        if record.fs != TARGET_FS or set(lookup) != set(LEADS):
            raise ValueError(f"LUDB record {record_id} violates the 500 Hz/12-lead contract")
        signal_mv = np.stack(
            [record.p_signal[:TARGET_SAMPLES, lookup[lead]] for lead in LEADS], axis=0
        ).astype(np.float32)
        if signal_mv.shape != (12, TARGET_SAMPLES) or not np.isfinite(signal_mv).all():
            raise ValueError(f"LUDB record {record_id} has invalid signal shape/content")
        records.append(
            {
                "record_id": record_id,
                "signal": signal_mv,
                "annotations": {lead: parse_annotations(path, lead) for lead in LEADS},
                "metadata": parse_metadata(list(record.comments)),
            }
        )
    if not records:
        raise RuntimeError(f"No LUDB records found under {root}")
    return records, tree_identity(root)


def finite_pearson(target: np.ndarray, reconstruction: np.ndarray) -> float:
    target = target - target.mean()
    reconstruction = reconstruction - reconstruction.mean()
    denominator = math.sqrt(float(np.square(target).sum() * np.square(reconstruction).sum()))
    return float(np.dot(target, reconstruction) / denominator) if denominator > 1e-12 else float("nan")


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def finite_quantile(values: list[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if len(array) else float("nan")


def aggregate_errors(values: list[float]) -> tuple[int, float, float, float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    absolute = np.abs(finite)
    return (
        len(finite), float(np.mean(absolute)), float(np.median(absolute)),
        float(np.quantile(absolute, 0.95)), float(np.mean(finite)),
    )


def local_feature_location(
    signal_values: np.ndarray,
    sample: int,
    wave: str,
    landmark: str,
    polarity: float,
) -> int:
    half_width = 40 if landmark == "peak" else 20
    lower = max(0, sample - half_width)
    upper = min(len(signal_values), sample + half_width + 1)
    segment = signal_values[lower:upper]
    if len(segment) < 3:
        return int(np.clip(sample, 0, len(signal_values) - 1))
    if landmark != "peak":
        return lower + int(np.argmax(np.abs(np.diff(segment)))) + 1
    baseline = float(np.median(segment))
    if wave == "QRS":
        return lower + int(np.argmax(np.abs(segment - baseline)))
    return lower + int(np.argmax(segment) if polarity >= 0 else np.argmin(segment))


def annotation_rows(
    model_id: str,
    record: dict[str, Any],
    reconstruction: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target = record["signal"]
    landmark_rows: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    for lead_index in MISSING:
        lead = LEADS[lead_index]
        target_lead = target[lead_index]
        recon_lead = reconstruction[lead_index]
        positioned: dict[str, list[dict[str, Any]]] = {"P": [], "QRS": [], "T": []}
        for wave, events in record["annotations"][lead].items():
            landmark_values: dict[str, list[float]] = {name: [] for name in LANDMARKS}
            timing_values: dict[str, list[float]] = {name: [] for name in LANDMARKS}
            window_corr: list[float] = []
            window_mse: list[float] = []
            window_mae: list[float] = []
            area_error: list[float] = []
            for event in events:
                predicted: dict[str, int | None] = {name: None for name in LANDMARKS}
                for landmark in LANDMARKS:
                    sample = event[landmark]
                    if sample is None or not 0 <= sample < TARGET_SAMPLES:
                        continue
                    signed_voltage = float(recon_lead[sample] - target_lead[sample])
                    landmark_values[landmark].append(signed_voltage)
                    context = target_lead[max(0, sample - 40) : min(TARGET_SAMPLES, sample + 41)]
                    polarity = float(target_lead[sample] - np.median(context))
                    target_position = local_feature_location(target_lead, sample, wave, landmark, polarity)
                    recon_position = local_feature_location(recon_lead, sample, wave, landmark, polarity)
                    predicted[landmark] = recon_position
                    timing_values[landmark].append(
                        (recon_position - target_position) * 1000.0 / TARGET_FS
                    )
                if event["peak"] is not None:
                    positioned[wave].append({"truth": event, "predicted": predicted})
                onset, offset = event["onset"], event["offset"]
                if onset is None or offset is None or not 0 <= onset < offset < TARGET_SAMPLES:
                    continue
                target_window = target_lead[onset : offset + 1]
                recon_window = recon_lead[onset : offset + 1]
                difference = recon_window - target_window
                window_corr.append(finite_pearson(target_window, recon_window))
                window_mse.append(float(np.square(difference).mean()))
                window_mae.append(float(np.abs(difference).mean()))
                area_error.append(float(abs(np.trapezoid(recon_window) - np.trapezoid(target_window)) / len(target_window)))

            for landmark in LANDMARKS:
                n_labels, voltage_mean, voltage_median, voltage_p95, voltage_bias = aggregate_errors(
                    landmark_values[landmark]
                )
                n_timing, timing_mean, timing_median, timing_p95, timing_bias = aggregate_errors(
                    timing_values[landmark]
                )
                if not n_labels:
                    continue
                landmark_rows.append(
                    {
                        "model_id": model_id, "record_id": record["record_id"], "lead": lead,
                        "wave": wave, "landmark": landmark, "n_labels": n_labels,
                        "voltage_mae_mean_mv": voltage_mean,
                        "voltage_mae_median_mv": voltage_median,
                        "voltage_mae_p95_mv": voltage_p95, "voltage_bias_mv": voltage_bias,
                        "timing_abs_mean_ms": timing_mean if n_timing else float("nan"),
                        "timing_abs_median_ms": timing_median if n_timing else float("nan"),
                        "timing_abs_p95_ms": timing_p95 if n_timing else float("nan"),
                        "timing_bias_ms": timing_bias if n_timing else float("nan"),
                    }
                )
            finite_corr = [value for value in window_corr if np.isfinite(value)]
            if finite_corr:
                wave_rows.append(
                    {
                        "model_id": model_id, "record_id": record["record_id"], "lead": lead,
                        "wave": wave, "n_waves": len(finite_corr),
                        "window_pearson_mean": float(np.mean(finite_corr)),
                        "window_pearson_p05": finite_quantile(finite_corr, 0.05),
                        "window_mse_mean": finite_mean(window_mse),
                        "window_mse_p95": finite_quantile(window_mse, 0.95),
                        "window_mae_mean": finite_mean(window_mae),
                        "area_abs_error_mean_mv": finite_mean(area_error),
                    }
                )

        interval_errors: dict[str, list[float]] = {
            name: [] for name in (
                "P_duration", "PR_interval", "PR_segment", "QRS_duration",
                "QT_interval", "JT_interval", "ST_segment", "T_duration", "RR_interval",
            )
        }

        def add_interval(name: str, start: dict[str, Any], start_key: str, end: dict[str, Any], end_key: str) -> None:
            truth_start = start["truth"][start_key]
            truth_end = end["truth"][end_key]
            pred_start = start["predicted"][start_key]
            pred_end = end["predicted"][end_key]
            if None not in (truth_start, truth_end, pred_start, pred_end):
                truth_duration = int(truth_end) - int(truth_start)
                predicted_duration = int(pred_end) - int(pred_start)
                interval_errors[name].append((predicted_duration - truth_duration) * 1000.0 / TARGET_FS)

        for wave, name in (("P", "P_duration"), ("QRS", "QRS_duration"), ("T", "T_duration")):
            for event in positioned[wave]:
                add_interval(name, event, "onset", event, "offset")
        qrs_events = positioned["QRS"]
        p_events = positioned["P"]
        t_events = positioned["T"]
        for qrs in qrs_events:
            qrs_peak = qrs["truth"]["peak"]
            preceding = [event for event in p_events if event["truth"]["peak"] is not None and event["truth"]["peak"] < qrs_peak]
            if preceding:
                p_event = preceding[-1]
                if qrs_peak - p_event["truth"]["peak"] <= TARGET_FS:
                    add_interval("PR_interval", p_event, "onset", qrs, "onset")
                    add_interval("PR_segment", p_event, "offset", qrs, "onset")
            following = [event for event in t_events if event["truth"]["peak"] is not None and event["truth"]["peak"] > qrs_peak]
            if following:
                t_event = following[0]
                if t_event["truth"]["peak"] - qrs_peak <= TARGET_FS:
                    add_interval("QT_interval", qrs, "onset", t_event, "offset")
                    add_interval("JT_interval", qrs, "offset", t_event, "offset")
                    add_interval("ST_segment", qrs, "offset", t_event, "onset")
        for first, second in zip(qrs_events, qrs_events[1:]):
            add_interval("RR_interval", first, "peak", second, "peak")
        for interval_name, errors in interval_errors.items():
            count, mean_abs, median_abs, p95_abs, bias = aggregate_errors(errors)
            if count:
                interval_rows.append(
                    {
                        "model_id": model_id, "record_id": record["record_id"], "lead": lead,
                        "interval_name": interval_name, "n_intervals": count,
                        "error_bias_ms": bias, "error_abs_mean_ms": mean_abs,
                        "error_abs_median_ms": median_abs, "error_abs_p95_ms": p95_abs,
                    }
                )
    return landmark_rows, wave_rows, interval_rows


def record_rows(
    model_id: str, record: dict[str, Any], reconstruction: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target = record["signal"]
    lead_rows: list[dict[str, Any]] = []
    for lead_index, lead in enumerate(LEADS):
        difference = reconstruction[lead_index] - target[lead_index]
        derivative_difference = np.diff(reconstruction[lead_index]) - np.diff(target[lead_index])
        lead_rows.append(
            {
                "model_id": model_id,
                "record_id": record["record_id"],
                "lead": lead,
                "lead_role": "observed" if lead_index in OBSERVED else "missing",
                "pearson": finite_pearson(target[lead_index], reconstruction[lead_index]),
                "mse": float(np.square(difference).mean()),
                "mae": float(np.abs(difference).mean()),
                "derivative_mse": float(np.square(derivative_difference).mean()),
            }
        )
    missing = [lead_rows[index] for index in MISSING]
    v1 = lead_rows[6]
    result = {
        "model_id": model_id,
        "record_id": record["record_id"],
        "missing_pearson": finite_mean([row["pearson"] for row in missing]),
        "missing_mse": finite_mean([row["mse"] for row in missing]),
        "missing_mae": finite_mean([row["mae"] for row in missing]),
        "missing_derivative_mse": finite_mean([row["derivative_mse"] for row in missing]),
        "v1_pearson": v1["pearson"],
        "v1_mse": v1["mse"],
    }
    landmarks, waves, intervals = annotation_rows(model_id, record, reconstruction)
    return result, lead_rows, landmarks, waves, intervals


def quantile(rows: list[dict[str, Any]], key: str, value: float) -> float:
    array = np.asarray([row[key] for row in rows], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, value)) if len(array) else float("nan")


def summarize(
    model_id: str,
    mask: str,
    rows: list[dict[str, Any]],
    landmarks: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    pearson = np.asarray([row["missing_pearson"] for row in rows], dtype=np.float64)
    v1 = np.asarray([row["v1_pearson"] for row in rows], dtype=np.float64)
    pearson = pearson[np.isfinite(pearson)]
    v1 = v1[np.isfinite(v1)]
    qrs_peak = [row["voltage_mae_p95_mv"] for row in landmarks if row["wave"] == "QRS" and row["landmark"] == "peak"]
    qrs_peak_timing = [row["timing_abs_p95_ms"] for row in landmarks if row["wave"] == "QRS" and row["landmark"] == "peak"]
    qrs_duration = [row["error_abs_p95_ms"] for row in intervals if row["interval_name"] == "QRS_duration"]
    qt = [row["error_abs_p95_ms"] for row in intervals if row["interval_name"] == "QT_interval"]
    return {
        "model_id": model_id,
        "factorial_mask": mask,
        "n_records": len(rows),
        "pearson_p01": quantile(rows, "missing_pearson", 0.01),
        "pearson_p05": quantile(rows, "missing_pearson", 0.05),
        "pearson_p10": quantile(rows, "missing_pearson", 0.10),
        "pearson_p25": quantile(rows, "missing_pearson", 0.25),
        "pearson_median": quantile(rows, "missing_pearson", 0.50),
        "pearson_mean": float(pearson.mean()),
        "failure_rate_r_lt_040": float((pearson < 0.40).mean()),
        "failure_rate_r_lt_060": float((pearson < 0.60).mean()),
        "mse_median": quantile(rows, "missing_mse", 0.50),
        "mse_p90": quantile(rows, "missing_mse", 0.90),
        "mse_p95": quantile(rows, "missing_mse", 0.95),
        "mae_p95": quantile(rows, "missing_mae", 0.95),
        "derivative_mse_p95": quantile(rows, "missing_derivative_mse", 0.95),
        "v1_pearson_p05": quantile(rows, "v1_pearson", 0.05),
        "v1_pearson_median": quantile(rows, "v1_pearson", 0.50),
        "v1_failure_rate_r_lt_020": float((v1 < 0.20).mean()),
        "qrs_peak_voltage_p95_mv": finite_quantile(qrs_peak, 0.95),
        "qrs_peak_timing_p95_ms": finite_quantile(qrs_peak_timing, 0.95),
        "qrs_duration_error_p95_ms": finite_quantile(qrs_duration, 0.95),
        "qt_error_p95_ms": finite_quantile(qt, 0.95),
        "completed_at": utc_now(),
    }


def store_record_metadata(connection: sqlite3.Connection, records: list[dict[str, Any]]) -> None:
    metadata_rows = []
    subgroup_rows = []
    for record in records:
        metadata = record["metadata"]
        metadata_rows.append(
            {
                "record_id": record["record_id"], "age": metadata["age"],
                "sex": metadata["sex"], "rhythm": metadata["rhythm"],
                "axis": metadata["axis"],
                "comments_json": json.dumps(metadata["comments"], ensure_ascii=False),
            }
        )
        subgroup_rows.extend(
            {
                "record_id": record["record_id"], "subgroup": subgroup, "source": source,
            }
            for subgroup, source in metadata["subgroups"].items()
        )
    with connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO record_metadata(
                record_id,age,sex,rhythm,axis,comments_json
            ) VALUES(:record_id,:age,:sex,:rhythm,:axis,:comments_json)
            """,
            metadata_rows,
        )
        connection.executemany(
            """
            INSERT OR REPLACE INTO record_subgroups(record_id,subgroup,source)
            VALUES(:record_id,:subgroup,:source)
            """,
            subgroup_rows,
        )
        audit: dict[str, int] = {"records": len(records), "annotation_streams": len(records) * len(LEADS)}
        for lead_role, indices in (("all", range(12)), ("missing", MISSING), ("observed", OBSERVED)):
            for wave in ("P", "QRS", "T"):
                for landmark in LANDMARKS:
                    audit[f"{lead_role}_{wave}_{landmark}_labels"] = sum(
                        event[landmark] is not None
                        for record in records
                        for lead_index in indices
                        for event in record["annotations"][LEADS[lead_index]][wave]
                    )
        connection.executemany(
            "INSERT OR REPLACE INTO dataset_audit(key,value) VALUES(?,?)",
            [(key, str(value)) for key, value in audit.items()],
        )


def annotation_summaries(
    model_id: str,
    record_rows_values: list[dict[str, Any]],
    landmark_rows: list[dict[str, Any]],
    wave_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    landmark_summary = []
    for wave in ("P", "QRS", "T"):
        for landmark in LANDMARKS:
            selected = [row for row in landmark_rows if row["wave"] == wave and row["landmark"] == landmark]
            if not selected:
                continue
            landmark_summary.append(
                {
                    "model_id": model_id, "wave": wave, "landmark": landmark,
                    "records": len({row["record_id"] for row in selected}),
                    "labels": sum(row["n_labels"] for row in selected),
                    "voltage_mae_median_mv": finite_quantile([row["voltage_mae_median_mv"] for row in selected], 0.50),
                    "voltage_mae_p95_mv": finite_quantile([row["voltage_mae_p95_mv"] for row in selected], 0.95),
                    "voltage_bias_mv": finite_mean([row["voltage_bias_mv"] for row in selected]),
                    "timing_abs_median_ms": finite_quantile([row["timing_abs_median_ms"] for row in selected], 0.50),
                    "timing_abs_p95_ms": finite_quantile([row["timing_abs_p95_ms"] for row in selected], 0.95),
                    "timing_bias_ms": finite_mean([row["timing_bias_ms"] for row in selected]),
                }
            )

    wave_summary = []
    for wave in ("P", "QRS", "T"):
        selected = [row for row in wave_rows if row["wave"] == wave]
        if not selected:
            continue
        wave_summary.append(
            {
                "model_id": model_id, "wave": wave,
                "records": len({row["record_id"] for row in selected}),
                "waves": sum(row["n_waves"] for row in selected),
                "window_pearson_p05": finite_quantile([row["window_pearson_p05"] for row in selected], 0.05),
                "window_pearson_median": finite_quantile([row["window_pearson_mean"] for row in selected], 0.50),
                "window_mse_p95": finite_quantile([row["window_mse_p95"] for row in selected], 0.95),
                "area_abs_error_p95_mv": finite_quantile([row["area_abs_error_mean_mv"] for row in selected], 0.95),
            }
        )

    interval_summary = []
    interval_names = sorted({row["interval_name"] for row in interval_rows})
    for interval_name in interval_names:
        selected = [row for row in interval_rows if row["interval_name"] == interval_name]
        interval_summary.append(
            {
                "model_id": model_id, "interval_name": interval_name,
                "records": len({row["record_id"] for row in selected}),
                "intervals": sum(row["n_intervals"] for row in selected),
                "error_bias_ms": finite_mean([row["error_bias_ms"] for row in selected]),
                "error_abs_median_ms": finite_quantile([row["error_abs_median_ms"] for row in selected], 0.50),
                "error_abs_p95_ms": finite_quantile([row["error_abs_p95_ms"] for row in selected], 0.95),
            }
        )

    record_lookup = {row["record_id"]: row for row in record_rows_values}
    subgroup_to_records: dict[str, set[str]] = {}
    for record in records:
        for subgroup in record["metadata"]["subgroups"]:
            subgroup_to_records.setdefault(subgroup, set()).add(record["record_id"])
    subgroup_summary = []
    for subgroup, record_ids in sorted(subgroup_to_records.items()):
        selected_records = [record_lookup[record_id] for record_id in record_ids if record_id in record_lookup]
        if not selected_records:
            continue
        pearson = [row["missing_pearson"] for row in selected_records]
        qrs_timing = [
            row["timing_abs_p95_ms"] for row in landmark_rows
            if row["record_id"] in record_ids and row["wave"] == "QRS" and row["landmark"] == "peak"
        ]
        qt_error = [
            row["error_abs_p95_ms"] for row in interval_rows
            if row["record_id"] in record_ids and row["interval_name"] == "QT_interval"
        ]
        subgroup_summary.append(
            {
                "model_id": model_id, "subgroup": subgroup, "n_records": len(selected_records),
                "pearson_p05": finite_quantile(pearson, 0.05),
                "pearson_median": finite_quantile(pearson, 0.50),
                "failure_rate_r_lt_060": float(np.mean(np.asarray(pearson) < 0.60)),
                "mse_p95": finite_quantile([row["missing_mse"] for row in selected_records], 0.95),
                "qrs_peak_timing_p95_ms": finite_quantile(qrs_timing, 0.95),
                "qt_error_p95_ms": finite_quantile(qt_error, 0.95),
            }
        )
    return landmark_summary, wave_summary, interval_summary, subgroup_summary


def completed_ecgaim_models(queue_path: Path, checkpoint_db: Path) -> list[dict[str, Any]]:
    queue = json.loads(queue_path.read_text())
    complete_masks = {
        match.group(1)
        for job in queue["jobs"]
        if job.get("status") == "completed"
        and (match := MODEL_RE.fullmatch(job["id"])) is not None
    }
    connection = connect_checkpoint_db(checkpoint_db)
    try:
        rows = connection.execute(
            """
            SELECT model_id, factorial_mask, sha256, size_bytes, status
            FROM checkpoints
            WHERE model_id LIKE 'factorial_ecg_aim_%_s42'
              AND status IN ('local','remote_verified','cached')
            ORDER BY factorial_mask
            """
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows if row["factorial_mask"] in complete_masks]


def is_complete(connection: sqlite3.Connection, identity: dict[str, Any], evaluator_sha: str, dataset_sha: str) -> bool:
    row = connection.execute(
        "SELECT * FROM evaluations WHERE model_id=?", (identity["model_id"],)
    ).fetchone()
    return bool(
        row
        and row["status"] == "complete"
        and row["checkpoint_sha256"] == identity["sha256"]
        and row["evaluator_sha256"] == evaluator_sha
        and row["dataset_sha256"] == dataset_sha
    )


def evaluate_model(
    connection: sqlite3.Connection,
    identity: dict[str, Any],
    records: list[dict[str, Any]],
    evaluator_sha: str,
    dataset_sha: str,
    batch_size: int,
    device: torch.device,
) -> None:
    model_id = identity["model_id"]
    attempts_row = connection.execute(
        "SELECT attempts FROM evaluations WHERE model_id=?", (model_id,)
    ).fetchone()
    attempts = int(attempts_row[0]) + 1 if attempts_row else 1
    connection.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
    connection.execute(
        """
        INSERT INTO evaluations(
            model_id,factorial_mask,checkpoint_sha256,checkpoint_size_bytes,
            evaluator_sha256,dataset_sha256,status,attempts,started_at
        ) VALUES(?,?,?,?,?,?,'running',?,?)
        """,
        (
            model_id, identity["factorial_mask"], identity["sha256"],
            identity["size_bytes"], evaluator_sha, dataset_sha, attempts, utc_now(),
        ),
    )
    connection.commit()
    started = time.perf_counter()
    adapter = None
    try:
        adapter = load_adapter(
            {
                "id": model_id,
                "kind": "alitok",
                "checkpoint": f"checkpoints/factorial_ecg_aim_{identity['factorial_mask']}_s42.pt",
                "observed_leads": list(OBSERVED),
            },
            device,
        )
        all_record_rows: list[dict[str, Any]] = []
        all_lead_rows: list[dict[str, Any]] = []
        all_landmark_rows: list[dict[str, Any]] = []
        all_wave_rows: list[dict[str, Any]] = []
        all_interval_rows: list[dict[str, Any]] = []
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            target = torch.from_numpy(np.stack([record["signal"] for record in batch]))
            with torch.inference_mode():
                reconstruction = adapter.reconstruct(target).cpu().numpy()
            for record, reconstructed in zip(batch, reconstruction):
                row, leads, landmarks, waves, intervals = record_rows(model_id, record, reconstructed)
                all_record_rows.append(row)
                all_lead_rows.extend(leads)
                all_landmark_rows.extend(landmarks)
                all_wave_rows.extend(waves)
                all_interval_rows.extend(intervals)
        summary = summarize(
            model_id, identity["factorial_mask"], all_record_rows,
            all_landmark_rows, all_interval_rows,
        )
        landmark_summary, wave_summary, interval_summary, subgroup_summary = annotation_summaries(
            model_id, all_record_rows, all_landmark_rows, all_wave_rows,
            all_interval_rows, records,
        )
        with connection:
            connection.executemany(
                """
                INSERT INTO record_metrics(
                    model_id,record_id,missing_pearson,missing_mse,missing_mae,
                    missing_derivative_mse,v1_pearson,v1_mse
                ) VALUES(
                    :model_id,:record_id,:missing_pearson,:missing_mse,:missing_mae,
                    :missing_derivative_mse,:v1_pearson,:v1_mse
                )
                """,
                all_record_rows,
            )
            connection.executemany(
                """
                INSERT INTO lead_metrics VALUES(
                    :model_id,:record_id,:lead,:lead_role,:pearson,:mse,:mae,:derivative_mse
                )
                """,
                all_lead_rows,
            )
            connection.executemany(
                """
                INSERT INTO landmark_metrics VALUES(
                    :model_id,:record_id,:lead,:wave,:landmark,:n_labels,
                    :voltage_mae_mean_mv,:voltage_mae_median_mv,:voltage_mae_p95_mv,
                    :voltage_bias_mv,:timing_abs_mean_ms,:timing_abs_median_ms,
                    :timing_abs_p95_ms,:timing_bias_ms
                )
                """,
                all_landmark_rows,
            )
            connection.executemany(
                """
                INSERT INTO wave_metrics VALUES(
                    :model_id,:record_id,:lead,:wave,:n_waves,:window_pearson_mean,
                    :window_pearson_p05,:window_mse_mean,:window_mse_p95,
                    :window_mae_mean,:area_abs_error_mean_mv
                )
                """,
                all_wave_rows,
            )
            connection.executemany(
                """
                INSERT INTO interval_metrics VALUES(
                    :model_id,:record_id,:lead,:interval_name,:n_intervals,
                    :error_bias_ms,:error_abs_mean_ms,:error_abs_median_ms,
                    :error_abs_p95_ms
                )
                """,
                all_interval_rows,
            )
            connection.execute(
                """
                INSERT INTO summaries VALUES(
                    :model_id,:factorial_mask,:n_records,:pearson_p01,:pearson_p05,
                    :pearson_p10,:pearson_p25,:pearson_median,:pearson_mean,
                    :failure_rate_r_lt_040,:failure_rate_r_lt_060,:mse_median,
                    :mse_p90,:mse_p95,:mae_p95,:derivative_mse_p95,
                    :v1_pearson_p05,:v1_pearson_median,:v1_failure_rate_r_lt_020,
                    :qrs_peak_voltage_p95_mv,:qrs_peak_timing_p95_ms,
                    :qrs_duration_error_p95_ms,:qt_error_p95_ms,:completed_at
                )
                """,
                summary,
            )
            connection.executemany(
                """
                INSERT INTO landmark_summaries VALUES(
                    :model_id,:wave,:landmark,:records,:labels,
                    :voltage_mae_median_mv,:voltage_mae_p95_mv,:voltage_bias_mv,
                    :timing_abs_median_ms,:timing_abs_p95_ms,:timing_bias_ms
                )
                """,
                landmark_summary,
            )
            connection.executemany(
                """
                INSERT INTO wave_summaries VALUES(
                    :model_id,:wave,:records,:waves,:window_pearson_p05,
                    :window_pearson_median,:window_mse_p95,:area_abs_error_p95_mv
                )
                """,
                wave_summary,
            )
            connection.executemany(
                """
                INSERT INTO interval_summaries VALUES(
                    :model_id,:interval_name,:records,:intervals,:error_bias_ms,
                    :error_abs_median_ms,:error_abs_p95_ms
                )
                """,
                interval_summary,
            )
            connection.executemany(
                """
                INSERT INTO subgroup_summaries VALUES(
                    :model_id,:subgroup,:n_records,:pearson_p05,:pearson_median,
                    :failure_rate_r_lt_060,:mse_p95,:qrs_peak_timing_p95_ms,
                    :qt_error_p95_ms
                )
                """,
                subgroup_summary,
            )
            connection.execute(
                """
                UPDATE evaluations
                SET status='complete', completed_at=?, duration_seconds=?, n_records=?,
                    error=NULL, summary_json=?
                WHERE model_id=?
                """,
                (
                    summary["completed_at"], time.perf_counter() - started, len(records),
                    json.dumps(summary, sort_keys=True, allow_nan=False), model_id,
                ),
            )
        print(json.dumps({"event": "model_complete", **summary}), flush=True)
    except Exception as error:
        connection.execute(
            "UPDATE evaluations SET status='error', error=? WHERE model_id=?",
            (f"{type(error).__name__}: {error}", model_id),
        )
        connection.commit()
        print(json.dumps({"event": "model_error", "model_id": model_id, "error": repr(error)}), flush=True)
    finally:
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with store_lock(CHECKPOINT_DB):
            catalog = connect_checkpoint_db(CHECKPOINT_DB)
            try:
                prune_cache(catalog, DEFAULT_CACHE_DIR, 0.0)
            finally:
                catalog.close()


def write_ranking_csv(connection: sqlite3.Connection, output: Path) -> None:
    rows = connection.execute("SELECT * FROM extreme_robustness_ranking ORDER BY robustness_rank").fetchall()
    if not rows:
        return
    columns = rows[0].keys()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join("" if row[key] is None else str(row[key]) for key in columns) + "\n")
    os.replace(temporary, output)


def write_table_csv(connection: sqlite3.Connection, table: str, output: Path) -> None:
    rows = connection.execute(f"SELECT * FROM {table} ORDER BY 1,2").fetchall()
    if not rows:
        return
    columns = rows[0].keys()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join("" if row[key] is None else str(row[key]) for key in columns) + "\n")
    os.replace(temporary, output)


def write_exports(connection: sqlite3.Connection, ranking_csv: Path) -> None:
    ranking_csv.parent.mkdir(parents=True, exist_ok=True)
    write_ranking_csv(connection, ranking_csv)
    for table in (
        "landmark_summaries", "wave_summaries", "interval_summaries", "subgroup_summaries",
    ):
        write_table_csv(connection, table, ranking_csv.parent / f"{table}.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "refine-logs/queue_3arch/queue_state.json")
    parser.add_argument("--checkpoint-db", type=Path, default=CHECKPOINT_DB)
    parser.add_argument("--results-db", type=Path, default=ROOT / "results/ecgaim_ludb/ecgaim_ludb.sqlite")
    parser.add_argument("--ludb-root", type=Path, default=ROOT / "data/ludb")
    parser.add_argument("--ranking-csv", type=Path, default=ROOT / "results/ecgaim_ludb/extreme_robustness_ranking.csv")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=6)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-models", type=int, default=0)
    parser.add_argument("--model-id")
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.poll_seconds, args.batch_size, args.torch_threads) < 1:
        raise ValueError("poll-seconds, batch-size, and torch-threads must be positive")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    records, dataset_sha = load_ludb(args.ludb_root, args.max_records)
    evaluator_sha = sha256_file(Path(__file__))
    connection = connect_results(args.results_db)
    store_record_metadata(connection, records)
    device = torch.device("cpu")
    print(
        json.dumps(
            {
                "event": "daemon_started", "records": len(records), "device": str(device),
                "results_db": str(args.results_db), "at": utc_now(),
            }
        ),
        flush=True,
    )
    try:
        while not STOP_REQUESTED:
            eligible = completed_ecgaim_models(args.queue, args.checkpoint_db)
            if args.model_id:
                eligible = [row for row in eligible if row["model_id"] == args.model_id]
                if not eligible:
                    raise ValueError(f"completed/cataloged ECG-AIM model not found: {args.model_id}")
            pending = [
                row for row in eligible
                if not is_complete(connection, row, evaluator_sha, dataset_sha)
            ]
            if args.max_models:
                pending = pending[: args.max_models]
            print(
                json.dumps(
                    {
                        "event": "pass_started", "eligible": len(eligible),
                        "pending": len(pending), "at": utc_now(),
                    }
                ),
                flush=True,
            )
            for identity in pending:
                if STOP_REQUESTED:
                    break
                evaluate_model(
                    connection, identity, records, evaluator_sha, dataset_sha,
                    args.batch_size, device,
                )
                write_exports(connection, args.ranking_csv)
            write_exports(connection, args.ranking_csv)
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
