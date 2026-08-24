#!/usr/bin/env python3
"""CPU-only ECG-AIM fidelity evaluation with LUDB annotations as the clock.

This evaluator never predicts, moves, or re-finds a fiducial.  It compares the
reconstructed and original signals at the same cardiologist-labeled samples
and over the same labeled wave/ST intervals.  Automatic delineation belongs to
a separate secondary track and is deliberately absent from this module.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# This experiment is intentionally CPU-only and must be set before torch import.
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
from scripts.evaluate_ecgaim_ludb_daemon import (  # noqa: E402
    completed_ecgaim_models,
    parse_metadata,
)


LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
DISPLAY_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
OBSERVED = (0, 1, 7)
DERIVED_LIMB = (2, 3, 4, 5)
PRIMARY_PRECORDIAL = (6, 8, 9, 10, 11)
TARGET_FS = 500
TARGET_SAMPLES = 5000
SAMPLE_MS = 1000.0 / TARGET_FS
WAVE_SYMBOLS = {"p": "P", "N": "QRS", "t": "T"}
LANDMARKS = ("onset", "peak", "offset")
STOP_REQUESTED = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum, "at": utc_now()}), flush=True)


def lead_role(index: int) -> str:
    if index in OBSERVED:
        return "observed_control"
    if index in DERIVED_LIMB:
        return "derived_limb_control"
    if index in PRIMARY_PRECORDIAL:
        return "primary_missing_precordial"
    raise ValueError(f"unclassified lead index {index}")


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
    digest.update(np.__version__.encode())
    digest.update(torch.__version__.encode())
    digest.update(wfdb.__version__.encode())
    return digest.hexdigest()


def dataset_identity(root: Path, record_ids: list[str]) -> str:
    digest = hashlib.sha256()
    suffixes = ("dat", "hea", *LEADS)
    for record_id in record_ids:
        for suffix in suffixes:
            path = root / f"{record_id}.{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(sha256_file(path).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def finite_pearson(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(reconstruction, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.square(left).sum() * np.square(right).sum()))
    return float(np.dot(left, right) / denominator) if denominator > 1e-15 else float("nan")


def finite_quantile(values: Iterable[float], probability: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if len(array) else None


def finite_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def finite_rmse(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.sqrt(np.mean(np.square(array)))) if len(array) else None


def parse_symbol_stream(symbols: list[str], samples: list[int]) -> list[dict[str, Any]]:
    """Map only explicit adjacent LUDB onset-peak-offset annotations.

    Orphan boundary symbols remain in the dataset audit and are never guessed
    onto a wave.  Every P/N/T peak is retained even when a boundary is absent.
    """
    counters = defaultdict(int)
    events: list[dict[str, Any]] = []
    for index, symbol in enumerate(symbols):
        wave = WAVE_SYMBOLS.get(symbol)
        if wave is None:
            continue
        event_index = counters[wave]
        counters[wave] += 1
        onset = int(samples[index - 1]) if index > 0 and symbols[index - 1] == "(" else None
        offset = int(samples[index + 1]) if index + 1 < len(symbols) and symbols[index + 1] == ")" else None
        peak = int(samples[index])
        events.append({
            "wave": wave, "event_index": event_index,
            "onset": onset, "peak": peak, "offset": offset,
        })
    return events


def wave_features(values: np.ndarray, onset: int, peak: int, offset: int) -> dict[str, float]:
    """Fixed LUDB interval features with linear endpoint-baseline correction."""
    if not 0 <= onset < peak < offset < len(values):
        raise ValueError("invalid wave interval")
    window = np.asarray(values[onset : offset + 1], dtype=np.float64)
    baseline = np.linspace(window[0], window[-1], len(window), dtype=np.float64)
    residual = window - baseline
    peak_offset = peak - onset
    return {
        "peak_amplitude_mv": float(residual[peak_offset]),
        "signed_area_mv_ms": float(np.trapezoid(residual, dx=SAMPLE_MS)),
        "absolute_area_mv_ms": float(np.trapezoid(np.abs(residual), dx=SAMPLE_MS)),
    }


def linear_slope_mv_per_s(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 2:
        return float("nan")
    x = np.arange(len(y), dtype=np.float64) / TARGET_FS
    x = x - x.mean()
    denominator = float(np.square(x).sum())
    return float(np.dot(x, y - y.mean()) / denominator) if denominator > 0 else float("nan")


def st_features(values: np.ndarray, qrs_offset: int, t_onset: int) -> dict[str, float | None]:
    if not 0 <= qrs_offset < t_onset < len(values):
        raise ValueError("invalid ST interval")
    window = np.asarray(values[qrs_offset : t_onset + 1], dtype=np.float64)
    result: dict[str, float | None] = {
        "j_mv": float(values[qrs_offset]),
        "mean_mv": float(window.mean()),
        "area_mv_ms": float(np.trapezoid(window, dx=SAMPLE_MS)),
        "slope_mv_per_s": linear_slope_mv_per_s(window),
    }
    for delay in (20, 40, 60, 80):
        sample = qrs_offset + round(delay / SAMPLE_MS)
        result[f"j{delay}_mv"] = float(values[sample]) if sample <= t_onset else None
    return result


def load_ludb(root: Path, max_records: int = 0) -> tuple[list[dict[str, Any]], str, dict[str, int]]:
    record_ids = sorted({path.stem for path in root.glob("*.dat")}, key=int)
    if max_records:
        record_ids = record_ids[:max_records]
    records: list[dict[str, Any]] = []
    audit = defaultdict(int)
    next_event_id = next_wave_id = next_st_id = 1
    for record_id in record_ids:
        path = root / record_id
        record = wfdb.rdrecord(str(path), physical=True)
        lookup = {name.lower(): index for index, name in enumerate(record.sig_name)}
        if record.fs != TARGET_FS or set(lookup) != set(LEADS):
            raise ValueError(f"LUDB {record_id} violates 500 Hz / 12-lead contract")
        signal_mv = np.stack(
            [record.p_signal[:TARGET_SAMPLES, lookup[lead]] for lead in LEADS], axis=0
        ).astype(np.float32)
        if signal_mv.shape != (12, TARGET_SAMPLES) or not np.isfinite(signal_mv).all():
            raise ValueError(f"LUDB {record_id} has invalid signal data")
        item: dict[str, Any] = {
            "record_id": record_id, "signal": signal_mv,
            "metadata": parse_metadata(list(record.comments)),
            "events": [], "waves": [], "st_segments": [],
        }
        for lead_index, lead in enumerate(LEADS):
            annotation = wfdb.rdann(str(path), extension=lead)
            symbols = list(annotation.symbol)
            samples = [int(value) for value in annotation.sample]
            audit["raw_onset_symbols"] += symbols.count("(")
            audit["raw_offset_symbols"] += symbols.count(")")
            parsed = parse_symbol_stream(symbols, samples)
            by_wave: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for event in parsed:
                by_wave[event["wave"]].append(event)
                audit["wave_peaks"] += 1
                for landmark in LANDMARKS:
                    sample = event[landmark]
                    if sample is None or not 0 <= sample < TARGET_SAMPLES:
                        continue
                    item["events"].append({
                        "event_id": next_event_id, "record_id": record_id,
                        "lead_index": lead_index, "lead": DISPLAY_LEADS[lead_index],
                        "lead_role": lead_role(lead_index), "wave": event["wave"],
                        "landmark": landmark, "event_index": event["event_index"],
                        "sample": sample, "reference_mv": float(signal_mv[lead_index, sample]),
                    })
                    next_event_id += 1
                    audit["mapped_landmarks"] += 1
                onset, peak, offset = event["onset"], event["peak"], event["offset"]
                if onset is None or offset is None or not 0 <= onset < peak < offset < TARGET_SAMPLES:
                    audit["invalid_or_incomplete_wave_intervals"] += 1
                    continue
                features = wave_features(signal_mv[lead_index], onset, peak, offset)
                item["waves"].append({
                    "wave_id": next_wave_id, "record_id": record_id,
                    "lead_index": lead_index, "lead": DISPLAY_LEADS[lead_index],
                    "lead_role": lead_role(lead_index), "wave": event["wave"],
                    "event_index": event["event_index"], "onset": onset,
                    "peak": peak, "offset": offset, **features,
                })
                next_wave_id += 1
                audit["valid_wave_intervals"] += 1
            qrs_events = by_wave["QRS"]
            t_events = by_wave["T"]
            for qrs_position, qrs in enumerate(qrs_events):
                next_qrs_peak = (
                    qrs_events[qrs_position + 1]["peak"]
                    if qrs_position + 1 < len(qrs_events) else TARGET_SAMPLES
                )
                candidates = [
                    event for event in t_events
                    if qrs["peak"] < event["peak"] < next_qrs_peak
                ]
                if not candidates or qrs["offset"] is None:
                    continue
                t_event = candidates[0]
                if t_event["onset"] is None:
                    continue
                qrs_offset, t_onset = int(qrs["offset"]), int(t_event["onset"])
                if not 0 <= qrs_offset < t_onset < TARGET_SAMPLES:
                    audit["invalid_st_intervals"] += 1
                    continue
                features = st_features(signal_mv[lead_index], qrs_offset, t_onset)
                item["st_segments"].append({
                    "st_id": next_st_id, "record_id": record_id,
                    "lead_index": lead_index, "lead": DISPLAY_LEADS[lead_index],
                    "lead_role": lead_role(lead_index),
                    "qrs_event_index": qrs["event_index"],
                    "t_event_index": t_event["event_index"],
                    "qrs_offset": qrs_offset, "t_onset": t_onset, **features,
                })
                next_st_id += 1
                audit["valid_st_intervals"] += 1
        records.append(item)
    if not records:
        raise RuntimeError("no LUDB records loaded")
    audit["records"] = len(records)
    audit["lead_streams"] = len(records) * len(LEADS)
    audit["orphan_onset_symbols"] = audit["raw_onset_symbols"] - sum(
        1 for record in records for event in record["events"] if event["landmark"] == "onset"
    )
    audit["orphan_offset_symbols"] = audit["raw_offset_symbols"] - sum(
        1 for record in records for event in record["events"] if event["landmark"] == "offset"
    )
    return records, dataset_identity(root, record_ids), dict(audit)


def connect_results(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA page_size=8192")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA wal_autocheckpoint=4096")
    connection.execute("PRAGMA journal_size_limit=33554432")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS protocol_metadata(
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS record_metadata(
            record_id TEXT PRIMARY KEY, age REAL, sex TEXT, rhythm TEXT, axis TEXT,
            comments_json TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS record_subgroups(
            record_id TEXT NOT NULL REFERENCES record_metadata(record_id),
            subgroup TEXT NOT NULL, source TEXT NOT NULL,
            PRIMARY KEY(record_id,subgroup)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS dataset_audit(
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS evaluations(
            evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL UNIQUE, factorial_mask TEXT NOT NULL,
            checkpoint_sha256 TEXT NOT NULL, checkpoint_size_bytes INTEGER NOT NULL,
            protocol_sha256 TEXT NOT NULL, dataset_sha256 TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running','complete','error')),
            attempts INTEGER NOT NULL, started_at TEXT, completed_at TEXT,
            duration_seconds REAL, n_records INTEGER NOT NULL, error TEXT,
            primary_summary_json TEXT
        );
        CREATE TABLE IF NOT EXISTS oracle_events(
            event_id INTEGER PRIMARY KEY, record_id TEXT NOT NULL, lead_index INTEGER NOT NULL,
            lead TEXT NOT NULL, lead_role TEXT NOT NULL, wave TEXT NOT NULL,
            landmark TEXT NOT NULL, event_index INTEGER NOT NULL, sample INTEGER NOT NULL,
            reference_mv REAL NOT NULL,
            UNIQUE(record_id,lead,wave,landmark,event_index)
        );
        CREATE TABLE IF NOT EXISTS oracle_waves(
            wave_id INTEGER PRIMARY KEY, record_id TEXT NOT NULL, lead_index INTEGER NOT NULL,
            lead TEXT NOT NULL, lead_role TEXT NOT NULL, wave TEXT NOT NULL,
            event_index INTEGER NOT NULL, onset INTEGER NOT NULL, peak INTEGER NOT NULL,
            offset INTEGER NOT NULL, reference_peak_amplitude_mv REAL NOT NULL,
            reference_signed_area_mv_ms REAL NOT NULL,
            reference_absolute_area_mv_ms REAL NOT NULL,
            UNIQUE(record_id,lead,wave,event_index)
        );
        CREATE TABLE IF NOT EXISTS oracle_st_segments(
            st_id INTEGER PRIMARY KEY, record_id TEXT NOT NULL, lead_index INTEGER NOT NULL,
            lead TEXT NOT NULL, lead_role TEXT NOT NULL, qrs_event_index INTEGER NOT NULL,
            t_event_index INTEGER NOT NULL, qrs_offset INTEGER NOT NULL, t_onset INTEGER NOT NULL,
            reference_j_mv REAL NOT NULL, reference_j20_mv REAL, reference_j40_mv REAL,
            reference_j60_mv REAL, reference_j80_mv REAL, reference_mean_mv REAL NOT NULL,
            reference_area_mv_ms REAL NOT NULL, reference_slope_mv_per_s REAL NOT NULL,
            UNIQUE(record_id,lead,qrs_event_index)
        );
        CREATE TABLE IF NOT EXISTS event_metrics(
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
            event_id INTEGER NOT NULL REFERENCES oracle_events(event_id),
            reconstruction_mv REAL NOT NULL, error_mv REAL NOT NULL, abs_error_mv REAL NOT NULL,
            PRIMARY KEY(evaluation_id,event_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS wave_metrics(
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
            wave_id INTEGER NOT NULL REFERENCES oracle_waves(wave_id),
            reconstruction_peak_amplitude_mv REAL NOT NULL, peak_amplitude_error_mv REAL NOT NULL,
            reconstruction_signed_area_mv_ms REAL NOT NULL, signed_area_error_mv_ms REAL NOT NULL,
            reconstruction_absolute_area_mv_ms REAL NOT NULL, absolute_area_error_mv_ms REAL NOT NULL,
            window_pearson REAL, window_mse REAL NOT NULL, window_mae REAL NOT NULL,
            window_rmse REAL NOT NULL, window_max_abs_error REAL NOT NULL,
            PRIMARY KEY(evaluation_id,wave_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS st_metrics(
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
            st_id INTEGER NOT NULL REFERENCES oracle_st_segments(st_id),
            j_error_mv REAL NOT NULL, j20_error_mv REAL, j40_error_mv REAL,
            j60_error_mv REAL, j80_error_mv REAL, mean_error_mv REAL NOT NULL,
            area_error_mv_ms REAL NOT NULL, slope_error_mv_per_s REAL NOT NULL,
            window_pearson REAL, window_rmse REAL NOT NULL,
            PRIMARY KEY(evaluation_id,st_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS signal_lead_metrics(
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
            record_id TEXT NOT NULL, lead TEXT NOT NULL, lead_role TEXT NOT NULL,
            pearson REAL, mse REAL NOT NULL, mae REAL NOT NULL, derivative_mse REAL NOT NULL,
            PRIMARY KEY(evaluation_id,record_id,lead)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS record_role_metrics(
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
            record_id TEXT NOT NULL, lead_role TEXT NOT NULL,
            event_abs_error_median_mv REAL, event_abs_error_p95_mv REAL,
            wave_window_rmse_median_mv REAL, qrs_window_rmse_mean_mv REAL,
            t_window_rmse_mean_mv REAL, qrs_area_abs_error_mean_mv_ms REAL,
            t_area_abs_error_mean_mv_ms REAL, st_j_abs_error_mean_mv REAL,
            signal_pearson_mean REAL, signal_mse_mean REAL, signal_derivative_mse_mean REAL,
            PRIMARY KEY(evaluation_id,record_id,lead_role)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS event_summaries(
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
            lead_role TEXT NOT NULL, lead TEXT NOT NULL, wave TEXT NOT NULL, landmark TEXT NOT NULL,
            labels INTEGER NOT NULL, error_bias_mv REAL, error_rmse_mv REAL,
            abs_error_median_mv REAL, abs_error_p95_mv REAL,
            PRIMARY KEY(evaluation_id,lead_role,lead,wave,landmark)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS wave_summaries(
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
            lead_role TEXT NOT NULL, lead TEXT NOT NULL, wave TEXT NOT NULL, waves INTEGER NOT NULL,
            window_pearson_p05 REAL, window_pearson_median REAL,
            window_rmse_median_mv REAL, window_rmse_p95_mv REAL,
            peak_amplitude_abs_error_median_mv REAL, peak_amplitude_abs_error_p95_mv REAL,
            signed_area_abs_error_median_mv_ms REAL, signed_area_abs_error_p95_mv_ms REAL,
            absolute_area_abs_error_median_mv_ms REAL, absolute_area_abs_error_p95_mv_ms REAL,
            PRIMARY KEY(evaluation_id,lead_role,lead,wave)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS st_summaries(
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
            lead_role TEXT NOT NULL, lead TEXT NOT NULL, segments INTEGER NOT NULL,
            j_abs_error_median_mv REAL, j_abs_error_p95_mv REAL,
            j60_abs_error_median_mv REAL, j60_abs_error_p95_mv REAL,
            mean_abs_error_median_mv REAL, mean_abs_error_p95_mv REAL,
            area_abs_error_median_mv_ms REAL, area_abs_error_p95_mv_ms REAL,
            slope_abs_error_median_mv_per_s REAL, slope_abs_error_p95_mv_per_s REAL,
            window_pearson_p05 REAL, window_rmse_p95_mv REAL,
            PRIMARY KEY(evaluation_id,lead_role,lead)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS signal_summaries(
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
            lead_role TEXT NOT NULL, lead TEXT NOT NULL, record_leads INTEGER NOT NULL,
            pearson_p01 REAL, pearson_p05 REAL, pearson_median REAL,
            failure_rate_r_lt_060 REAL, mse_median REAL, mse_p95 REAL,
            mae_p95 REAL, derivative_mse_p95 REAL,
            PRIMARY KEY(evaluation_id,lead_role,lead)
        ) WITHOUT ROWID;
        """
    )
    connection.commit()
    return connection


def store_static_dataset(
    connection: sqlite3.Connection,
    records: list[dict[str, Any]],
    protocol_sha: str,
    dataset_sha: str,
    audit: dict[str, int],
) -> None:
    existing = dict(connection.execute("SELECT key,value FROM protocol_metadata"))
    if existing and (
        existing.get("protocol_sha256") != protocol_sha
        or existing.get("dataset_sha256") != dataset_sha
    ):
        raise RuntimeError("dedicated oracle DB belongs to a different protocol or dataset; use a new DB")
    metadata = {
        "schema_version": "1", "protocol": "ludb_oracle_fixed_clock_v1",
        "protocol_sha256": protocol_sha, "dataset_sha256": dataset_sha,
        "sampling_rate_hz": str(TARGET_FS), "units": "mV",
        "primary_leads": "V1,V3,V4,V5,V6",
        "lead_roles": "observed_control=I,II,V2; derived_limb_control=III,aVR,aVL,aVF; primary_missing_precordial=V1,V3,V4,V5,V6",
        "clock": "LUDB lead-specific cardiologist samples; no feature localization",
        "wave_baseline": "linear chord between labeled onset and offset",
        "selection": "Pareto panel; no weighted composite winner score",
        "timing_metrics": "not applicable: oracle samples are fixed",
        "dice_metrics": "not applicable: no independent mask is emitted",
        "automatic_delineation": "absent; NeuroKit is a separate catastrophe-only track",
        "cpu_only": "true",
    }
    with connection:
        connection.executemany(
            "INSERT OR REPLACE INTO protocol_metadata(key,value) VALUES(?,?)", metadata.items()
        )
        connection.executemany(
            "INSERT OR REPLACE INTO dataset_audit(key,value) VALUES(?,?)",
            [(key, str(value)) for key, value in audit.items()],
        )
        for record in records:
            item = record["metadata"]
            connection.execute(
                "INSERT OR REPLACE INTO record_metadata VALUES(?,?,?,?,?,?)",
                (record["record_id"], item["age"], item["sex"], item["rhythm"], item["axis"],
                 json.dumps(item["comments"], ensure_ascii=False)),
            )
            connection.executemany(
                "INSERT OR REPLACE INTO record_subgroups VALUES(?,?,?)",
                [(record["record_id"], subgroup, source) for subgroup, source in item["subgroups"].items()],
            )
        if connection.execute("SELECT count(*) FROM oracle_events").fetchone()[0] == 0:
            connection.executemany(
                "INSERT INTO oracle_events VALUES(:event_id,:record_id,:lead_index,:lead,:lead_role,:wave,:landmark,:event_index,:sample,:reference_mv)",
                [event for record in records for event in record["events"]],
            )
            connection.executemany(
                "INSERT INTO oracle_waves VALUES(:wave_id,:record_id,:lead_index,:lead,:lead_role,:wave,:event_index,:onset,:peak,:offset,:peak_amplitude_mv,:signed_area_mv_ms,:absolute_area_mv_ms)",
                [wave for record in records for wave in record["waves"]],
            )
            connection.executemany(
                "INSERT INTO oracle_st_segments VALUES(:st_id,:record_id,:lead_index,:lead,:lead_role,:qrs_event_index,:t_event_index,:qrs_offset,:t_onset,:j_mv,:j20_mv,:j40_mv,:j60_mv,:j80_mv,:mean_mv,:area_mv_ms,:slope_mv_per_s)",
                [segment for record in records for segment in record["st_segments"]],
            )


def add_grouped(groups: dict[tuple[Any, ...], list[Any]], key: tuple[Any, ...], value: Any) -> None:
    groups.setdefault(key, []).append(value)
    groups.setdefault((key[0], "ALL", *key[2:]), []).append(value)


def build_summaries(
    evaluation_id: int,
    event_values: list[dict[str, Any]],
    wave_values: list[dict[str, Any]],
    st_values: list[dict[str, Any]],
    signal_values: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_groups: dict[tuple[Any, ...], list[Any]] = {}
    for row in event_values:
        add_grouped(event_groups, (row["lead_role"], row["lead"], row["wave"], row["landmark"]), row)
    event_summaries = []
    for (role, lead, wave, landmark), rows in sorted(event_groups.items()):
        signed = [row["error_mv"] for row in rows]
        absolute = [row["abs_error_mv"] for row in rows]
        event_summaries.append({
            "evaluation_id": evaluation_id, "lead_role": role, "lead": lead,
            "wave": wave, "landmark": landmark, "labels": len(rows),
            "error_bias_mv": finite_mean(signed), "error_rmse_mv": finite_rmse(signed),
            "abs_error_median_mv": finite_quantile(absolute, 0.5),
            "abs_error_p95_mv": finite_quantile(absolute, 0.95),
        })

    wave_groups: dict[tuple[Any, ...], list[Any]] = {}
    for row in wave_values:
        add_grouped(wave_groups, (row["lead_role"], row["lead"], row["wave"]), row)
    wave_summaries = []
    for (role, lead, wave), rows in sorted(wave_groups.items()):
        wave_summaries.append({
            "evaluation_id": evaluation_id, "lead_role": role, "lead": lead,
            "wave": wave, "waves": len(rows),
            "window_pearson_p05": finite_quantile((r["window_pearson"] for r in rows), 0.05),
            "window_pearson_median": finite_quantile((r["window_pearson"] for r in rows), 0.5),
            "window_rmse_median_mv": finite_quantile((r["window_rmse"] for r in rows), 0.5),
            "window_rmse_p95_mv": finite_quantile((r["window_rmse"] for r in rows), 0.95),
            "peak_amplitude_abs_error_median_mv": finite_quantile((abs(r["peak_amplitude_error_mv"]) for r in rows), 0.5),
            "peak_amplitude_abs_error_p95_mv": finite_quantile((abs(r["peak_amplitude_error_mv"]) for r in rows), 0.95),
            "signed_area_abs_error_median_mv_ms": finite_quantile((abs(r["signed_area_error_mv_ms"]) for r in rows), 0.5),
            "signed_area_abs_error_p95_mv_ms": finite_quantile((abs(r["signed_area_error_mv_ms"]) for r in rows), 0.95),
            "absolute_area_abs_error_median_mv_ms": finite_quantile((abs(r["absolute_area_error_mv_ms"]) for r in rows), 0.5),
            "absolute_area_abs_error_p95_mv_ms": finite_quantile((abs(r["absolute_area_error_mv_ms"]) for r in rows), 0.95),
        })

    st_groups: dict[tuple[Any, ...], list[Any]] = {}
    for row in st_values:
        add_grouped(st_groups, (row["lead_role"], row["lead"]), row)
    st_summaries = []
    for (role, lead), rows in sorted(st_groups.items()):
        st_summaries.append({
            "evaluation_id": evaluation_id, "lead_role": role, "lead": lead, "segments": len(rows),
            "j_abs_error_median_mv": finite_quantile((abs(r["j_error_mv"]) for r in rows), 0.5),
            "j_abs_error_p95_mv": finite_quantile((abs(r["j_error_mv"]) for r in rows), 0.95),
            "j60_abs_error_median_mv": finite_quantile((abs(r["j60_error_mv"]) for r in rows if r["j60_error_mv"] is not None), 0.5),
            "j60_abs_error_p95_mv": finite_quantile((abs(r["j60_error_mv"]) for r in rows if r["j60_error_mv"] is not None), 0.95),
            "mean_abs_error_median_mv": finite_quantile((abs(r["mean_error_mv"]) for r in rows), 0.5),
            "mean_abs_error_p95_mv": finite_quantile((abs(r["mean_error_mv"]) for r in rows), 0.95),
            "area_abs_error_median_mv_ms": finite_quantile((abs(r["area_error_mv_ms"]) for r in rows), 0.5),
            "area_abs_error_p95_mv_ms": finite_quantile((abs(r["area_error_mv_ms"]) for r in rows), 0.95),
            "slope_abs_error_median_mv_per_s": finite_quantile((abs(r["slope_error_mv_per_s"]) for r in rows), 0.5),
            "slope_abs_error_p95_mv_per_s": finite_quantile((abs(r["slope_error_mv_per_s"]) for r in rows), 0.95),
            "window_pearson_p05": finite_quantile((r["window_pearson"] for r in rows), 0.05),
            "window_rmse_p95_mv": finite_quantile((r["window_rmse"] for r in rows), 0.95),
        })

    signal_groups: dict[tuple[Any, ...], list[Any]] = {}
    for row in signal_values:
        add_grouped(signal_groups, (row["lead_role"], row["lead"]), row)
    signal_summaries = []
    for (role, lead), rows in sorted(signal_groups.items()):
        correlations = [r["pearson"] for r in rows]
        signal_summaries.append({
            "evaluation_id": evaluation_id, "lead_role": role, "lead": lead,
            "record_leads": len(rows), "pearson_p01": finite_quantile(correlations, 0.01),
            "pearson_p05": finite_quantile(correlations, 0.05),
            "pearson_median": finite_quantile(correlations, 0.5),
            "failure_rate_r_lt_060": float(np.mean(np.asarray(correlations) < 0.60)),
            "mse_median": finite_quantile((r["mse"] for r in rows), 0.5),
            "mse_p95": finite_quantile((r["mse"] for r in rows), 0.95),
            "mae_p95": finite_quantile((r["mae"] for r in rows), 0.95),
            "derivative_mse_p95": finite_quantile((r["derivative_mse"] for r in rows), 0.95),
        })

    record_groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in event_values:
        record_groups[(row["record_id"], row["lead_role"])]["event"].append(row["abs_error_mv"])
    for row in wave_values:
        group = record_groups[(row["record_id"], row["lead_role"])]
        group["wave_rmse"].append(row["window_rmse"])
        if row["wave"] == "QRS":
            group["qrs_rmse"].append(row["window_rmse"])
            group["qrs_area"].append(abs(row["absolute_area_error_mv_ms"]))
        if row["wave"] == "T":
            group["t_rmse"].append(row["window_rmse"])
            group["t_area"].append(abs(row["absolute_area_error_mv_ms"]))
    for row in st_values:
        record_groups[(row["record_id"], row["lead_role"])]["st_j"].append(abs(row["j_error_mv"]))
    for row in signal_values:
        group = record_groups[(row["record_id"], row["lead_role"])]
        group["signal_corr"].append(row["pearson"])
        group["signal_mse"].append(row["mse"])
        group["signal_deriv"].append(row["derivative_mse"])
    record_summaries = []
    for (record_id, role), values in sorted(record_groups.items()):
        record_summaries.append({
            "evaluation_id": evaluation_id, "record_id": record_id, "lead_role": role,
            "event_abs_error_median_mv": finite_quantile(values["event"], 0.5),
            "event_abs_error_p95_mv": finite_quantile(values["event"], 0.95),
            "wave_window_rmse_median_mv": finite_quantile(values["wave_rmse"], 0.5),
            "qrs_window_rmse_mean_mv": finite_mean(values["qrs_rmse"]),
            "t_window_rmse_mean_mv": finite_mean(values["t_rmse"]),
            "qrs_area_abs_error_mean_mv_ms": finite_mean(values["qrs_area"]),
            "t_area_abs_error_mean_mv_ms": finite_mean(values["t_area"]),
            "st_j_abs_error_mean_mv": finite_mean(values["st_j"]),
            "signal_pearson_mean": finite_mean(values["signal_corr"]),
            "signal_mse_mean": finite_mean(values["signal_mse"]),
            "signal_derivative_mse_mean": finite_mean(values["signal_deriv"]),
        })
    return event_summaries, wave_summaries, st_summaries, signal_summaries, record_summaries


def evaluate_model(
    connection: sqlite3.Connection,
    identity: dict[str, Any],
    records: list[dict[str, Any]],
    protocol_sha: str,
    dataset_sha: str,
    batch_size: int,
) -> None:
    model_id = identity["model_id"]
    old = connection.execute("SELECT attempts FROM evaluations WHERE model_id=?", (model_id,)).fetchone()
    attempts = int(old[0]) + 1 if old else 1
    with connection:
        connection.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
        cursor = connection.execute(
            """INSERT INTO evaluations(model_id,factorial_mask,checkpoint_sha256,
            checkpoint_size_bytes,protocol_sha256,dataset_sha256,status,attempts,
            started_at,n_records) VALUES(?,?,?,?,?,?,'running',?,?,?)""",
            (model_id, identity["factorial_mask"], identity["sha256"], identity["size_bytes"],
             protocol_sha, dataset_sha, attempts, utc_now(), len(records)),
        )
        evaluation_id = int(cursor.lastrowid)
    started = time.perf_counter()
    adapter = None
    try:
        adapter = load_adapter({
            "id": model_id, "kind": "alitok",
            "checkpoint": f"checkpoints/factorial_ecg_aim_{identity['factorial_mask']}_s42.pt",
            "observed_leads": list(OBSERVED),
        }, torch.device("cpu"))
        reconstructed_batches = []
        for offset in range(0, len(records), batch_size):
            batch = torch.from_numpy(np.stack([r["signal"] for r in records[offset:offset + batch_size]]))
            with torch.inference_mode():
                result = adapter.reconstruct(batch).cpu().numpy().astype(np.float32)
            if result.shape != (len(batch), 12, TARGET_SAMPLES) or not np.isfinite(result).all():
                raise ValueError(f"invalid reconstruction batch {result.shape}")
            reconstructed_batches.append(result)
        reconstructions = np.concatenate(reconstructed_batches, axis=0)
        event_rows: list[dict[str, Any]] = []
        wave_rows: list[dict[str, Any]] = []
        st_rows: list[dict[str, Any]] = []
        signal_rows: list[dict[str, Any]] = []
        for record, reconstruction in zip(records, reconstructions, strict=True):
            target = record["signal"]
            for lead_index, lead in enumerate(DISPLAY_LEADS):
                reference = target[lead_index].astype(np.float64)
                predicted = reconstruction[lead_index].astype(np.float64)
                difference = predicted - reference
                signal_rows.append({
                    "evaluation_id": evaluation_id, "record_id": record["record_id"],
                    "lead": lead, "lead_role": lead_role(lead_index),
                    "pearson": finite_pearson(reference, predicted),
                    "mse": float(np.mean(np.square(difference))),
                    "mae": float(np.mean(np.abs(difference))),
                    "derivative_mse": float(np.mean(np.square(np.diff(predicted) - np.diff(reference)))),
                })
            for event in record["events"]:
                reconstructed_mv = float(reconstruction[event["lead_index"], event["sample"]])
                error = reconstructed_mv - event["reference_mv"]
                event_rows.append({
                    **event, "evaluation_id": evaluation_id,
                    "reconstruction_mv": reconstructed_mv, "error_mv": error,
                    "abs_error_mv": abs(error),
                })
            for wave in record["waves"]:
                lead_values = reconstruction[wave["lead_index"]]
                reconstructed = wave_features(lead_values, wave["onset"], wave["peak"], wave["offset"])
                reference_window = target[wave["lead_index"], wave["onset"] : wave["offset"] + 1]
                predicted_window = lead_values[wave["onset"] : wave["offset"] + 1]
                difference = predicted_window.astype(np.float64) - reference_window.astype(np.float64)
                wave_rows.append({
                    **wave, "evaluation_id": evaluation_id,
                    "reconstruction_peak_amplitude_mv": reconstructed["peak_amplitude_mv"],
                    "peak_amplitude_error_mv": reconstructed["peak_amplitude_mv"] - wave["peak_amplitude_mv"],
                    "reconstruction_signed_area_mv_ms": reconstructed["signed_area_mv_ms"],
                    "signed_area_error_mv_ms": reconstructed["signed_area_mv_ms"] - wave["signed_area_mv_ms"],
                    "reconstruction_absolute_area_mv_ms": reconstructed["absolute_area_mv_ms"],
                    "absolute_area_error_mv_ms": reconstructed["absolute_area_mv_ms"] - wave["absolute_area_mv_ms"],
                    "window_pearson": finite_pearson(reference_window, predicted_window),
                    "window_mse": float(np.mean(np.square(difference))),
                    "window_mae": float(np.mean(np.abs(difference))),
                    "window_rmse": float(np.sqrt(np.mean(np.square(difference)))),
                    "window_max_abs_error": float(np.max(np.abs(difference))),
                })
            for segment in record["st_segments"]:
                predicted = st_features(
                    reconstruction[segment["lead_index"]], segment["qrs_offset"], segment["t_onset"]
                )
                reference_window = target[segment["lead_index"], segment["qrs_offset"] : segment["t_onset"] + 1]
                predicted_window = reconstruction[segment["lead_index"], segment["qrs_offset"] : segment["t_onset"] + 1]
                difference = predicted_window.astype(np.float64) - reference_window.astype(np.float64)
                row = {**segment, "evaluation_id": evaluation_id}
                for key in ("j_mv", "j20_mv", "j40_mv", "j60_mv", "j80_mv", "mean_mv", "area_mv_ms", "slope_mv_per_s"):
                    left, right = predicted[key], segment[key]
                    row[key.replace("_mv", "_error_mv", 1) if key.startswith("j") else {
                        "mean_mv": "mean_error_mv", "area_mv_ms": "area_error_mv_ms",
                        "slope_mv_per_s": "slope_error_mv_per_s",
                    }.get(key, key)] = None if left is None or right is None else float(left - right)
                row["window_pearson"] = finite_pearson(reference_window, predicted_window)
                row["window_rmse"] = float(np.sqrt(np.mean(np.square(difference))))
                st_rows.append(row)
        event_summaries, wave_summaries, st_summaries, signal_summaries, record_summaries = build_summaries(
            evaluation_id, event_rows, wave_rows, st_rows, signal_rows
        )
        primary_records = [
            row for row in record_summaries
            if row["lead_role"] == "primary_missing_precordial"
        ]
        if len(primary_records) != len(records):
            raise RuntimeError("primary record summary coverage is incomplete")
        primary = {
            "endpoint": "record-aggregated Pareto panel on fixed LUDB samples/intervals; no scalar winner score",
            "primary_leads": "V1,V3,V4,V5,V6",
            "records": len(primary_records),
            "signal_pearson_p05": finite_quantile((r["signal_pearson_mean"] for r in primary_records), 0.05),
            "signal_mse_p95": finite_quantile((r["signal_mse_mean"] for r in primary_records), 0.95),
            "event_abs_error_record_p95_mv": finite_quantile((r["event_abs_error_p95_mv"] for r in primary_records), 0.95),
            "qrs_window_rmse_record_p95_mv": finite_quantile((r["qrs_window_rmse_mean_mv"] for r in primary_records), 0.95),
            "qrs_absolute_area_error_record_p95_mv_ms": finite_quantile((r["qrs_area_abs_error_mean_mv_ms"] for r in primary_records), 0.95),
            "t_window_rmse_record_p95_mv": finite_quantile((r["t_window_rmse_mean_mv"] for r in primary_records), 0.95),
            "t_absolute_area_error_record_p95_mv_ms": finite_quantile((r["t_area_abs_error_mean_mv_ms"] for r in primary_records), 0.95),
            "st_j_abs_error_record_p95_mv": finite_quantile((r["st_j_abs_error_mean_mv"] for r in primary_records), 0.95),
        }
        with connection:
            connection.executemany(
                "INSERT INTO event_metrics VALUES(:evaluation_id,:event_id,:reconstruction_mv,:error_mv,:abs_error_mv)", event_rows
            )
            connection.executemany(
                """INSERT INTO wave_metrics VALUES(:evaluation_id,:wave_id,
                :reconstruction_peak_amplitude_mv,:peak_amplitude_error_mv,
                :reconstruction_signed_area_mv_ms,:signed_area_error_mv_ms,
                :reconstruction_absolute_area_mv_ms,:absolute_area_error_mv_ms,
                :window_pearson,:window_mse,:window_mae,:window_rmse,:window_max_abs_error)""", wave_rows
            )
            connection.executemany(
                """INSERT INTO st_metrics VALUES(:evaluation_id,:st_id,:j_error_mv,
                :j20_error_mv,:j40_error_mv,:j60_error_mv,:j80_error_mv,
                :mean_error_mv,:area_error_mv_ms,:slope_error_mv_per_s,
                :window_pearson,:window_rmse)""", st_rows
            )
            connection.executemany(
                "INSERT INTO signal_lead_metrics VALUES(:evaluation_id,:record_id,:lead,:lead_role,:pearson,:mse,:mae,:derivative_mse)", signal_rows
            )
            connection.executemany(
                """INSERT INTO record_role_metrics VALUES(:evaluation_id,:record_id,:lead_role,
                :event_abs_error_median_mv,:event_abs_error_p95_mv,:wave_window_rmse_median_mv,
                :qrs_window_rmse_mean_mv,:t_window_rmse_mean_mv,
                :qrs_area_abs_error_mean_mv_ms,:t_area_abs_error_mean_mv_ms,
                :st_j_abs_error_mean_mv,:signal_pearson_mean,:signal_mse_mean,
                :signal_derivative_mse_mean)""", record_summaries
            )
            connection.executemany(
                "INSERT INTO event_summaries VALUES(:evaluation_id,:lead_role,:lead,:wave,:landmark,:labels,:error_bias_mv,:error_rmse_mv,:abs_error_median_mv,:abs_error_p95_mv)", event_summaries
            )
            connection.executemany(
                """INSERT INTO wave_summaries VALUES(:evaluation_id,:lead_role,:lead,:wave,:waves,
                :window_pearson_p05,:window_pearson_median,:window_rmse_median_mv,
                :window_rmse_p95_mv,:peak_amplitude_abs_error_median_mv,
                :peak_amplitude_abs_error_p95_mv,:signed_area_abs_error_median_mv_ms,
                :signed_area_abs_error_p95_mv_ms,:absolute_area_abs_error_median_mv_ms,
                :absolute_area_abs_error_p95_mv_ms)""", wave_summaries
            )
            connection.executemany(
                """INSERT INTO st_summaries VALUES(:evaluation_id,:lead_role,:lead,:segments,
                :j_abs_error_median_mv,:j_abs_error_p95_mv,:j60_abs_error_median_mv,
                :j60_abs_error_p95_mv,:mean_abs_error_median_mv,:mean_abs_error_p95_mv,
                :area_abs_error_median_mv_ms,:area_abs_error_p95_mv_ms,
                :slope_abs_error_median_mv_per_s,:slope_abs_error_p95_mv_per_s,
                :window_pearson_p05,:window_rmse_p95_mv)""", st_summaries
            )
            connection.executemany(
                """INSERT INTO signal_summaries VALUES(:evaluation_id,:lead_role,:lead,
                :record_leads,:pearson_p01,:pearson_p05,:pearson_median,
                :failure_rate_r_lt_060,:mse_median,:mse_p95,:mae_p95,
                :derivative_mse_p95)""", signal_summaries
            )
            connection.execute(
                "UPDATE evaluations SET status='complete',completed_at=?,duration_seconds=?,error=NULL,primary_summary_json=? WHERE evaluation_id=?",
                (utc_now(), time.perf_counter() - started, json.dumps(primary, allow_nan=False), evaluation_id),
            )
        print(json.dumps({"event": "model_complete", "model_id": model_id, **primary}), flush=True)
    except Exception as error:
        with connection:
            connection.execute(
                "UPDATE evaluations SET status='error',completed_at=?,duration_seconds=?,error=? WHERE evaluation_id=?",
                (utc_now(), time.perf_counter() - started, f"{type(error).__name__}: {error}", evaluation_id),
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


def is_complete(connection: sqlite3.Connection, identity: dict[str, Any], protocol_sha: str, dataset_sha: str) -> bool:
    row = connection.execute("SELECT * FROM evaluations WHERE model_id=?", (identity["model_id"],)).fetchone()
    return bool(row and row["status"] == "complete" and row["checkpoint_sha256"] == identity["sha256"]
                and row["protocol_sha256"] == protocol_sha and row["dataset_sha256"] == dataset_sha)


def checkpoint_wal(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()


def sufficient_disk_space(path: Path, minimum_free_gb: float) -> tuple[bool, int]:
    free_bytes = shutil.disk_usage(path).free
    return free_bytes >= int(minimum_free_gb * 1024 ** 3), free_bytes


def write_query_csv(connection: sqlite3.Connection, query: str, output: Path) -> None:
    rows = connection.execute(query).fetchall()
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows[0].keys())
        writer.writerows([tuple(row) for row in rows])
    temporary.replace(output)


def write_exports(connection: sqlite3.Connection, output_dir: Path) -> None:
    write_query_csv(connection, "SELECT * FROM evaluations ORDER BY evaluation_id", output_dir / "oracle_evaluations.csv")
    for table in ("event_summaries", "wave_summaries", "st_summaries", "signal_summaries", "record_role_metrics"):
        write_query_csv(connection, f"SELECT * FROM {table} ORDER BY 1,2,3,4", output_dir / f"oracle_{table}.csv")
    write_query_csv(connection, """
        SELECT e.factorial_mask,e.model_id,e.checkpoint_sha256,
          json_extract(e.primary_summary_json,'$.signal_pearson_p05') signal_pearson_p05,
          json_extract(e.primary_summary_json,'$.signal_mse_p95') signal_mse_p95,
          json_extract(e.primary_summary_json,'$.event_abs_error_record_p95_mv') event_abs_error_record_p95_mv,
          json_extract(e.primary_summary_json,'$.qrs_window_rmse_record_p95_mv') qrs_window_rmse_record_p95_mv,
          json_extract(e.primary_summary_json,'$.qrs_absolute_area_error_record_p95_mv_ms') qrs_absolute_area_error_record_p95_mv_ms,
          json_extract(e.primary_summary_json,'$.t_window_rmse_record_p95_mv') t_window_rmse_record_p95_mv,
          json_extract(e.primary_summary_json,'$.t_absolute_area_error_record_p95_mv_ms') t_absolute_area_error_record_p95_mv_ms,
          json_extract(e.primary_summary_json,'$.st_j_abs_error_record_p95_mv') st_j_abs_error_record_p95_mv
        FROM evaluations e WHERE e.status='complete' ORDER BY e.factorial_mask
    """, output_dir / "oracle_primary_pareto.csv")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--queue", type=Path, default=ROOT / "refine-logs/queue_3arch/queue_state.json")
    result.add_argument("--checkpoint-db", type=Path, default=CHECKPOINT_DB)
    result.add_argument("--results-db", type=Path, default=ROOT / "results/ecgaim_ludb_oracle/ecgaim_ludb_oracle.sqlite")
    result.add_argument("--output-dir", type=Path, default=ROOT / "results/ecgaim_ludb_oracle")
    result.add_argument("--ludb-root", type=Path, default=ROOT / "data/ludb")
    result.add_argument("--poll-seconds", type=int, default=300)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--torch-threads", type=int, default=6)
    result.add_argument("--min-free-gb", type=float, default=5.0)
    result.add_argument("--max-records", type=int, default=0)
    result.add_argument("--max-models", type=int, default=0)
    result.add_argument("--model-id")
    result.add_argument("--once", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if min(args.poll_seconds, args.batch_size, args.torch_threads) < 1 or args.min_free_gb < 0:
        raise ValueError("invalid positive runtime option")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only contract violated")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    records, dataset_sha, audit = load_ludb(args.ludb_root, args.max_records)
    protocol_sha = protocol_identity()
    connection = connect_results(args.results_db)
    store_static_dataset(connection, records, protocol_sha, dataset_sha, audit)
    print(json.dumps({
        "event": "daemon_started", "records": len(records), "lead_streams": len(records) * 12,
        "mapped_landmarks": audit["mapped_landmarks"], "valid_wave_intervals": audit["valid_wave_intervals"],
        "valid_st_intervals": audit["valid_st_intervals"], "device": "cpu",
        "min_free_gb": args.min_free_gb, "results_db": str(args.results_db), "at": utc_now(),
    }), flush=True)
    try:
        while not STOP_REQUESTED:
            eligible = completed_ecgaim_models(args.queue, args.checkpoint_db)
            if args.model_id:
                eligible = [row for row in eligible if row["model_id"] == args.model_id]
                if not eligible:
                    raise ValueError(f"eligible ECG-AIM model not found: {args.model_id}")
            pending = [row for row in eligible if not is_complete(connection, row, protocol_sha, dataset_sha)]
            if args.max_models:
                pending = pending[:args.max_models]
            print(json.dumps({"event": "pass_started", "eligible": len(eligible), "pending": len(pending), "at": utc_now()}), flush=True)
            for identity in pending:
                if STOP_REQUESTED:
                    break
                has_space, free_bytes = sufficient_disk_space(args.results_db.parent, args.min_free_gb)
                if not has_space:
                    print(json.dumps({"event": "disk_space_pause", "free_bytes": free_bytes,
                                      "next_model": identity["model_id"], "at": utc_now()}), flush=True)
                    break
                evaluate_model(connection, identity, records, protocol_sha, dataset_sha, args.batch_size)
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
