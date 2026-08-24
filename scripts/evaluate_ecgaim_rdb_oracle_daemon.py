#!/usr/bin/env python3
"""CPU-only ECG-AIM evaluation on fixed RDB cardiologist intervals.

This is a reconstruction-fidelity experiment, not a segmentation experiment:
the released lead-specific P/QRS/T regions are held fixed on reference and
reconstruction.  No peak, boundary, mask, Dice, or F1 prediction is invented.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
from scripts.evaluate_ecgaim_ludb_daemon import completed_ecgaim_models  # noqa: E402
from scripts.rdb_oracle import (  # noqa: E402
    DISPLAY_LEADS,
    OBSERVED,
    TARGET_SAMPLES,
    WAVES,
    finite_mean,
    finite_pearson,
    finite_quantile,
    finite_rmse,
    interval_metric,
    lead_role,
    load_rdb,
    next_t_onset,
    sha256_file,
    st_features,
)


STOP_REQUESTED = False
LUDB_ORACLE_DB = ROOT / "results/ecgaim_ludb_oracle/ecgaim_ludb_oracle.sqlite"
DEFAULT_MIN_LUDB_SIGNAL_PEARSON_P05 = 0.50


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum, "at": utc_now()}), flush=True)


def protocol_identity(selection: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__), ROOT / "scripts/rdb_oracle.py",
        ROOT / "scripts/evaluate_comprehensive_registry.py",
        ROOT / "scripts/evaluate_ecgaim_ludb_daemon.py",
    ):
        digest.update(path.name.encode()); digest.update(sha256_file(path).encode())
    digest.update(np.__version__.encode()); digest.update(torch.__version__.encode())
    digest.update(json.dumps(selection, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def load_ludb_oracle_selection(path: Path, minimum: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze the RDB cohort from completed, identity-bound LUDB oracle rows."""
    if not np.isfinite(minimum):
        raise ValueError("LUDB selection threshold must be finite")
    if not path.is_file():
        raise FileNotFoundError(f"LUDB oracle DB not found: {path}")
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=60)
    connection.row_factory = sqlite3.Row
    try:
        status_counts = dict(connection.execute(
            "SELECT status,COUNT(*) FROM evaluations GROUP BY status"
        ).fetchall())
        rows = connection.execute(
            """
            SELECT model_id,factorial_mask,checkpoint_sha256,protocol_sha256,
                   dataset_sha256,primary_summary_json
            FROM evaluations
            WHERE status='complete'
            ORDER BY factorial_mask,model_id
            """
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise RuntimeError("LUDB oracle DB contains no completed evaluations")
    if set(status_counts) != {"complete"}:
        raise RuntimeError(f"LUDB oracle cohort is not frozen/completely successful: {status_counts}")
    source_protocols = {str(row["protocol_sha256"]) for row in rows}
    source_datasets = {str(row["dataset_sha256"]) for row in rows}
    if len(source_protocols) != 1 or len(source_datasets) != 1:
        raise RuntimeError("LUDB oracle completed rows mix protocol or dataset identities")

    selected: list[dict[str, Any]] = []
    for row in rows:
        summary = json.loads(row["primary_summary_json"] or "null")
        if not isinstance(summary, dict) or "signal_pearson_p05" not in summary:
            raise RuntimeError(f"missing LUDB signal_pearson_p05 for {row['model_id']}")
        value = float(summary["signal_pearson_p05"])
        if not np.isfinite(value):
            raise RuntimeError(f"non-finite LUDB signal_pearson_p05 for {row['model_id']}")
        if value >= minimum:
            selected.append({
                "model_id": str(row["model_id"]),
                "factorial_mask": str(row["factorial_mask"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "signal_pearson_p05": value,
            })
    if not selected:
        raise RuntimeError(f"no LUDB oracle checkpoints meet signal_pearson_p05 >= {minimum}")
    manifest_json = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    metadata = {
        "rule": "completed LUDB oracle signal_pearson_p05 >= threshold",
        "threshold": minimum,
        "source_models": len(rows),
        "selected_models": len(selected),
        "source_protocol_sha256": next(iter(source_protocols)),
        "source_dataset_sha256": next(iter(source_datasets)),
        "manifest_sha256": hashlib.sha256(manifest_json.encode()).hexdigest(),
    }
    return selected, metadata


def match_selected_models(
    archive_models: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Require exact model/mask/checkpoint identity between LUDB and archive."""
    by_id = {row["model_id"]: row for row in archive_models}
    matched: list[dict[str, Any]] = []
    for source in selected:
        archived = by_id.get(source["model_id"])
        if archived is None:
            raise RuntimeError(f"LUDB-selected checkpoint absent from archive/queue: {source['model_id']}")
        if archived["factorial_mask"] != source["factorial_mask"]:
            raise RuntimeError(f"factorial-mask identity mismatch for {source['model_id']}")
        if archived["sha256"] != source["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA-256 identity mismatch for {source['model_id']}")
        matched.append(archived)
    return matched


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
        CREATE TABLE IF NOT EXISTS dataset_audit(
          key TEXT PRIMARY KEY, value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS record_metadata(
          record_id TEXT PRIMARY KEY, chapman_record_id TEXT NOT NULL,
          released_rhythm TEXT NOT NULL, canonical_rhythm TEXT NOT NULL
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
        CREATE TABLE IF NOT EXISTS record_role_signal_metrics(
          evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
          record_id TEXT NOT NULL, canonical_rhythm TEXT NOT NULL, lead_role TEXT NOT NULL,
          leads INTEGER NOT NULL, pearson_mean REAL, pearson_min REAL,
          mse_mean REAL NOT NULL, mae_mean REAL NOT NULL, derivative_mse_mean REAL NOT NULL,
          PRIMARY KEY(evaluation_id,record_id,lead_role)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS record_role_wave_metrics(
          evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
          record_id TEXT NOT NULL, canonical_rhythm TEXT NOT NULL,
          lead_role TEXT NOT NULL, wave TEXT NOT NULL, intervals INTEGER NOT NULL,
          onset_error_bias_mv REAL, onset_error_rmse_mv REAL,
          onset_abs_error_median_mv REAL, onset_abs_error_p95_mv REAL,
          offset_error_bias_mv REAL, offset_error_rmse_mv REAL,
          offset_abs_error_median_mv REAL, offset_abs_error_p95_mv REAL,
          window_pearson_p05 REAL, window_pearson_median REAL,
          window_rmse_median_mv REAL, window_rmse_p95_mv REAL,
          window_mae_p95_mv REAL, window_max_abs_error_p95_mv REAL,
          signed_area_abs_error_median_mv_ms REAL, signed_area_abs_error_p95_mv_ms REAL,
          absolute_area_abs_error_median_mv_ms REAL, absolute_area_abs_error_p95_mv_ms REAL,
          PRIMARY KEY(evaluation_id,record_id,lead_role,wave)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS record_role_st_metrics(
          evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
          record_id TEXT NOT NULL, canonical_rhythm TEXT NOT NULL, lead_role TEXT NOT NULL,
          segments INTEGER NOT NULL, qrs_offset_abs_error_p95_mv REAL,
          j_abs_error_median_mv REAL, j_abs_error_p95_mv REAL,
          j60_abs_error_median_mv REAL, j60_abs_error_p95_mv REAL,
          mean_abs_error_p95_mv REAL, area_abs_error_p95_mv_ms REAL,
          window_pearson_p05 REAL, window_rmse_p95_mv REAL,
          PRIMARY KEY(evaluation_id,record_id,lead_role)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS lead_wave_summaries(
          evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
          canonical_rhythm TEXT NOT NULL, lead_role TEXT NOT NULL, lead TEXT NOT NULL,
          wave TEXT NOT NULL, record_leads INTEGER NOT NULL, intervals INTEGER NOT NULL,
          onset_abs_error_record_p95_mv REAL, offset_abs_error_record_p95_mv REAL,
          window_pearson_record_p05 REAL, window_rmse_record_p95_mv REAL,
          window_max_abs_error_record_p95_mv REAL,
          signed_area_abs_error_record_p95_mv_ms REAL,
          absolute_area_abs_error_record_p95_mv_ms REAL,
          PRIMARY KEY(evaluation_id,canonical_rhythm,lead_role,lead,wave)
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
    selection: dict[str, Any],
) -> None:
    existing = dict(connection.execute("SELECT key,value FROM protocol_metadata"))
    if existing and (existing.get("protocol_sha256") != protocol_sha or existing.get("dataset_sha256") != dataset_sha):
        raise RuntimeError("RDB DB belongs to a different protocol/dataset; use a new DB")
    metadata = {
        "schema_version": "1",
        "protocol": "rdb_oracle_fixed_regions_v1",
        "protocol_sha256": protocol_sha,
        "dataset_sha256": dataset_sha,
        "sampling_rate_hz": "500",
        "input_units": "released numeric units / 1000 = mV (evidence-gated normalization)",
        "primary_leads": "V1,V3,V4,V5,V6",
        "lead_roles": "observed=I,II,V2; derived=III,aVR,aVL,aVF; primary=V1,V3,V4,V5,V6",
        "clock": "RDB lead-specific inclusive cardiologist intervals; half-up resolved to sample grid",
        "invalid_intervals": "excluded, never clipped/swapped/expanded",
        "duplicate_policy": "mapping-flagged SI0211 excluded from primary cohort",
        "peak_metrics": "not applicable: RDB supplies regions but no labeled peaks",
        "timing_metrics": "not applicable in oracle track: boundaries are held fixed",
        "dice_f1_metrics": "not applicable: ECG-AIM emits a signal, not a segmentation mask",
        "automatic_delineation": "absent; a frozen segmenter may be added only as a secondary track",
        "storage": "record-role aggregates plus rhythm/lead summaries; no per-interval model rows",
        "cpu_only": "true",
        "model_selection_rule": str(selection["rule"]),
        "model_selection_threshold": repr(selection["threshold"]),
        "model_selection_source_models": str(selection["source_models"]),
        "model_selection_selected_models": str(selection["selected_models"]),
        "model_selection_source_protocol_sha256": str(selection["source_protocol_sha256"]),
        "model_selection_source_dataset_sha256": str(selection["source_dataset_sha256"]),
        "model_selection_manifest_sha256": str(selection["manifest_sha256"]),
    }
    with connection:
        connection.executemany("INSERT OR REPLACE INTO protocol_metadata VALUES(?,?)", metadata.items())
        connection.executemany(
            "INSERT OR REPLACE INTO dataset_audit VALUES(?,?)",
            [(key, str(value)) for key, value in sorted(audit.items())],
        )
        connection.executemany(
            "INSERT OR REPLACE INTO record_metadata VALUES(?,?,?,?)",
            [(r["record_id"], r["chapman_record_id"], r["released_rhythm"], r["canonical_rhythm"]) for r in records],
        )


def summarize_wave(rows: list[dict[str, float]]) -> dict[str, float | int | None]:
    return {
        "intervals": len(rows),
        "onset_error_bias_mv": finite_mean(r["onset_error_mv"] for r in rows),
        "onset_error_rmse_mv": finite_rmse(r["onset_error_mv"] for r in rows),
        "onset_abs_error_median_mv": finite_quantile((abs(r["onset_error_mv"]) for r in rows), .5),
        "onset_abs_error_p95_mv": finite_quantile((abs(r["onset_error_mv"]) for r in rows), .95),
        "offset_error_bias_mv": finite_mean(r["offset_error_mv"] for r in rows),
        "offset_error_rmse_mv": finite_rmse(r["offset_error_mv"] for r in rows),
        "offset_abs_error_median_mv": finite_quantile((abs(r["offset_error_mv"]) for r in rows), .5),
        "offset_abs_error_p95_mv": finite_quantile((abs(r["offset_error_mv"]) for r in rows), .95),
        "window_pearson_p05": finite_quantile((r["window_pearson"] for r in rows), .05),
        "window_pearson_median": finite_quantile((r["window_pearson"] for r in rows), .5),
        "window_rmse_median_mv": finite_quantile((r["window_rmse_mv"] for r in rows), .5),
        "window_rmse_p95_mv": finite_quantile((r["window_rmse_mv"] for r in rows), .95),
        "window_mae_p95_mv": finite_quantile((r["window_mae_mv"] for r in rows), .95),
        "window_max_abs_error_p95_mv": finite_quantile((r["window_max_abs_error_mv"] for r in rows), .95),
        "signed_area_abs_error_median_mv_ms": finite_quantile((abs(r["signed_area_error_mv_ms"]) for r in rows), .5),
        "signed_area_abs_error_p95_mv_ms": finite_quantile((abs(r["signed_area_error_mv_ms"]) for r in rows), .95),
        "absolute_area_abs_error_median_mv_ms": finite_quantile((abs(r["absolute_area_error_mv_ms"]) for r in rows), .5),
        "absolute_area_abs_error_p95_mv_ms": finite_quantile((abs(r["absolute_area_error_mv_ms"]) for r in rows), .95),
    }


def summarize_st(rows: list[dict[str, float | None]]) -> dict[str, float | int | None]:
    return {
        "segments": len(rows),
        "qrs_offset_abs_error_p95_mv": finite_quantile((abs(float(r["qrs_offset_error_mv"])) for r in rows), .95),
        "j_abs_error_median_mv": finite_quantile((abs(float(r["j_error_mv"])) for r in rows), .5),
        "j_abs_error_p95_mv": finite_quantile((abs(float(r["j_error_mv"])) for r in rows), .95),
        "j60_abs_error_median_mv": finite_quantile((abs(float(r["j60_error_mv"])) for r in rows if r["j60_error_mv"] is not None), .5),
        "j60_abs_error_p95_mv": finite_quantile((abs(float(r["j60_error_mv"])) for r in rows if r["j60_error_mv"] is not None), .95),
        "mean_abs_error_p95_mv": finite_quantile((abs(float(r["mean_error_mv"])) for r in rows), .95),
        "area_abs_error_p95_mv_ms": finite_quantile((abs(float(r["area_error_mv_ms"])) for r in rows), .95),
        "window_pearson_p05": finite_quantile((float(r["window_pearson"]) for r in rows), .05),
        "window_rmse_p95_mv": finite_quantile((float(r["window_rmse_mv"]) for r in rows), .95),
    }


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
            checkpoint_size_bytes,protocol_sha256,dataset_sha256,status,attempts,started_at,n_records)
            VALUES(?,?,?,?,?,?,'running',?,?,?)""",
            (model_id, identity["factorial_mask"], identity["sha256"], identity["size_bytes"],
             protocol_sha, dataset_sha, attempts, utc_now(), len(records)),
        )
        evaluation_id = int(cursor.lastrowid)
    started = time.perf_counter(); adapter = None
    try:
        adapter = load_adapter({
            "id": model_id, "kind": "alitok",
            "checkpoint": f"checkpoints/factorial_ecg_aim_{identity['factorial_mask']}_s42.pt",
            "observed_leads": list(OBSERVED),
        }, torch.device("cpu"))
        signal_rows: list[dict[str, Any]] = []
        wave_rows: list[dict[str, Any]] = []
        st_rows: list[dict[str, Any]] = []
        lead_record_summaries: list[dict[str, Any]] = []
        for offset in range(0, len(records), batch_size):
            subset = records[offset : offset + batch_size]
            batch = torch.from_numpy(np.stack([r["signal"] for r in subset]))
            with torch.inference_mode():
                reconstructions = adapter.reconstruct(batch).cpu().numpy().astype(np.float32)
            if reconstructions.shape != (len(subset), 12, TARGET_SAMPLES) or not np.isfinite(reconstructions).all():
                raise ValueError(f"invalid reconstruction batch: {reconstructions.shape}")
            for record, reconstruction in zip(subset, reconstructions, strict=True):
                signal_by_role: dict[str, list[dict[str, float]]] = defaultdict(list)
                wave_by_role: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
                st_by_role: dict[str, list[dict[str, float | None]]] = defaultdict(list)
                for lead_index, (lead, intervals) in enumerate(zip(DISPLAY_LEADS, record["annotations"], strict=True)):
                    role = lead_role(lead_index); reference = record["signal"][lead_index]; predicted = reconstruction[lead_index]
                    difference = predicted.astype(np.float64) - reference.astype(np.float64)
                    signal_by_role[role].append({
                        "pearson": finite_pearson(reference, predicted),
                        "mse": float(np.mean(np.square(difference))),
                        "mae": float(np.mean(np.abs(difference))),
                        "derivative_mse": float(np.mean(np.square(np.diff(difference)))),
                    })
                    local_waves: dict[str, list[dict[str, float]]] = defaultdict(list)
                    for kind, onset, end in intervals:
                        wave = WAVES[int(kind)]
                        metric = interval_metric(reference, predicted, int(onset), int(end))
                        local_waves[wave].append(metric); wave_by_role[(role, wave)].append(metric)
                        if wave == "QRS":
                            t_onset = next_t_onset(intervals, int(end))
                            if t_onset is not None:
                                ref_st = st_features(reference, int(end), t_onset)
                                pred_st = st_features(predicted, int(end), t_onset)
                                ref_window = reference[int(end)+1:t_onset]; pred_window = predicted[int(end)+1:t_onset]
                                st_by_role[role].append({
                                    "qrs_offset_error_mv": float(pred_st["qrs_offset_mv"] - ref_st["qrs_offset_mv"]),
                                    "j_error_mv": float(pred_st["j_mv"] - ref_st["j_mv"]),
                                    "j60_error_mv": None if ref_st["j60_mv"] is None else float(pred_st["j60_mv"] - ref_st["j60_mv"]),
                                    "mean_error_mv": float(pred_st["mean_mv"] - ref_st["mean_mv"]),
                                    "area_error_mv_ms": float(pred_st["area_mv_ms"] - ref_st["area_mv_ms"]),
                                    "window_pearson": finite_pearson(ref_window, pred_window),
                                    "window_rmse_mv": float(np.sqrt(np.mean(np.square(pred_window.astype(float)-ref_window.astype(float))))),
                                })
                    for wave, values in local_waves.items():
                        lead_record_summaries.append({
                            "canonical_rhythm": record["canonical_rhythm"], "lead_role": role,
                            "lead": lead, "wave": wave, **summarize_wave(values),
                        })
                for role, values in signal_by_role.items():
                    signal_rows.append({
                        "evaluation_id": evaluation_id, "record_id": record["record_id"],
                        "canonical_rhythm": record["canonical_rhythm"], "lead_role": role,
                        "leads": len(values), "pearson_mean": finite_mean(v["pearson"] for v in values),
                        "pearson_min": finite_quantile((v["pearson"] for v in values), 0),
                        "mse_mean": finite_mean(v["mse"] for v in values),
                        "mae_mean": finite_mean(v["mae"] for v in values),
                        "derivative_mse_mean": finite_mean(v["derivative_mse"] for v in values),
                    })
                for (role, wave), values in wave_by_role.items():
                    wave_rows.append({
                        "evaluation_id": evaluation_id, "record_id": record["record_id"],
                        "canonical_rhythm": record["canonical_rhythm"], "lead_role": role,
                        "wave": wave, **summarize_wave(values),
                    })
                for role, values in st_by_role.items():
                    st_rows.append({
                        "evaluation_id": evaluation_id, "record_id": record["record_id"],
                        "canonical_rhythm": record["canonical_rhythm"], "lead_role": role,
                        **summarize_st(values),
                    })
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in lead_record_summaries:
            for rhythm in (row["canonical_rhythm"], "ALL"):
                grouped[(rhythm, row["lead_role"], row["lead"], row["wave"])].append(row)
        lead_summaries = []
        for (rhythm, role, lead, wave), values in sorted(grouped.items()):
            lead_summaries.append({
                "evaluation_id": evaluation_id, "canonical_rhythm": rhythm, "lead_role": role,
                "lead": lead, "wave": wave, "record_leads": len(values),
                "intervals": sum(int(v["intervals"]) for v in values),
                "onset_abs_error_record_p95_mv": finite_quantile((v["onset_abs_error_p95_mv"] for v in values), .95),
                "offset_abs_error_record_p95_mv": finite_quantile((v["offset_abs_error_p95_mv"] for v in values), .95),
                "window_pearson_record_p05": finite_quantile((v["window_pearson_p05"] for v in values), .05),
                "window_rmse_record_p95_mv": finite_quantile((v["window_rmse_p95_mv"] for v in values), .95),
                "window_max_abs_error_record_p95_mv": finite_quantile((v["window_max_abs_error_p95_mv"] for v in values), .95),
                "signed_area_abs_error_record_p95_mv_ms": finite_quantile((v["signed_area_abs_error_p95_mv_ms"] for v in values), .95),
                "absolute_area_abs_error_record_p95_mv_ms": finite_quantile((v["absolute_area_abs_error_p95_mv_ms"] for v in values), .95),
            })
        primary_signal = [r for r in signal_rows if r["lead_role"] == "primary_missing_precordial"]
        primary_waves = [r for r in wave_rows if r["lead_role"] == "primary_missing_precordial"]
        primary_qrs = [r for r in wave_rows if r["lead_role"] == "primary_missing_precordial" and r["wave"] == "QRS"]
        primary_t = [r for r in wave_rows if r["lead_role"] == "primary_missing_precordial" and r["wave"] == "T"]
        primary_st = [r for r in st_rows if r["lead_role"] == "primary_missing_precordial"]
        primary = {
            "endpoint": "record-aggregated Pareto panel on fixed RDB lead-specific regions; no scalar winner score",
            "primary_leads": "V1,V3,V4,V5,V6", "records": len(primary_signal),
            "signal_pearson_p05": finite_quantile((r["pearson_mean"] for r in primary_signal), .05),
            "signal_mse_p95": finite_quantile((r["mse_mean"] for r in primary_signal), .95),
            "boundary_voltage_abs_error_record_p95_mv": finite_quantile(
                (max(r["onset_abs_error_p95_mv"], r["offset_abs_error_p95_mv"]) for r in primary_waves), .95
            ),
            "qrs_window_rmse_record_p95_mv": finite_quantile((r["window_rmse_p95_mv"] for r in primary_qrs), .95),
            "qrs_absolute_area_error_record_p95_mv_ms": finite_quantile((r["absolute_area_abs_error_p95_mv_ms"] for r in primary_qrs), .95),
            "t_window_rmse_record_p95_mv": finite_quantile((r["window_rmse_p95_mv"] for r in primary_t), .95),
            "t_absolute_area_error_record_p95_mv_ms": finite_quantile((r["absolute_area_abs_error_p95_mv_ms"] for r in primary_t), .95),
            "st_j_abs_error_record_p95_mv": finite_quantile((r["j_abs_error_p95_mv"] for r in primary_st), .95),
        }
        with connection:
            connection.executemany(
                "INSERT INTO record_role_signal_metrics VALUES(:evaluation_id,:record_id,:canonical_rhythm,:lead_role,:leads,:pearson_mean,:pearson_min,:mse_mean,:mae_mean,:derivative_mse_mean)", signal_rows)
            connection.executemany(
                """INSERT INTO record_role_wave_metrics VALUES(
                :evaluation_id,:record_id,:canonical_rhythm,:lead_role,:wave,:intervals,
                :onset_error_bias_mv,:onset_error_rmse_mv,:onset_abs_error_median_mv,:onset_abs_error_p95_mv,
                :offset_error_bias_mv,:offset_error_rmse_mv,:offset_abs_error_median_mv,:offset_abs_error_p95_mv,
                :window_pearson_p05,:window_pearson_median,:window_rmse_median_mv,:window_rmse_p95_mv,
                :window_mae_p95_mv,:window_max_abs_error_p95_mv,:signed_area_abs_error_median_mv_ms,
                :signed_area_abs_error_p95_mv_ms,:absolute_area_abs_error_median_mv_ms,
                :absolute_area_abs_error_p95_mv_ms)""", wave_rows)
            connection.executemany(
                """INSERT INTO record_role_st_metrics VALUES(
                :evaluation_id,:record_id,:canonical_rhythm,:lead_role,:segments,
                :qrs_offset_abs_error_p95_mv,:j_abs_error_median_mv,:j_abs_error_p95_mv,
                :j60_abs_error_median_mv,:j60_abs_error_p95_mv,:mean_abs_error_p95_mv,
                :area_abs_error_p95_mv_ms,:window_pearson_p05,:window_rmse_p95_mv)""", st_rows)
            connection.executemany(
                """INSERT INTO lead_wave_summaries VALUES(
                :evaluation_id,:canonical_rhythm,:lead_role,:lead,:wave,:record_leads,:intervals,
                :onset_abs_error_record_p95_mv,:offset_abs_error_record_p95_mv,
                :window_pearson_record_p05,:window_rmse_record_p95_mv,
                :window_max_abs_error_record_p95_mv,:signed_area_abs_error_record_p95_mv_ms,
                :absolute_area_abs_error_record_p95_mv_ms)""", lead_summaries)
            connection.execute(
                "UPDATE evaluations SET status='complete',completed_at=?,duration_seconds=?,error=NULL,primary_summary_json=? WHERE evaluation_id=?",
                (utc_now(), time.perf_counter()-started, json.dumps(primary, allow_nan=False), evaluation_id),
            )
        print(json.dumps({"event": "model_complete", "model_id": model_id, **primary}), flush=True)
    except Exception as error:
        with connection:
            connection.execute(
                "UPDATE evaluations SET status='error',completed_at=?,duration_seconds=?,error=? WHERE evaluation_id=?",
                (utc_now(), time.perf_counter()-started, f"{type(error).__name__}: {error}", evaluation_id),
            )
        raise
    finally:
        del adapter
        with store_lock(CHECKPOINT_DB):
            catalog = connect_checkpoint_db(CHECKPOINT_DB)
            try: prune_cache(catalog, DEFAULT_CACHE_DIR, 0.0)
            finally: catalog.close()


def write_query_csv(connection: sqlite3.Connection, table: str, output: Path) -> None:
    rows = connection.execute(f"SELECT * FROM {table} ORDER BY 1,2,3,4").fetchall()
    if not rows: return
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_suffix(output.suffix+".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(rows[0].keys()); writer.writerows(map(tuple, rows))
    temporary.replace(output)


def write_exports(connection: sqlite3.Connection, output_dir: Path) -> None:
    # Record-level tables stay queryable in SQLite for bootstrap analysis.  Do
    # not duplicate millions of rows into CSV on every daemon pass.
    for table in ("evaluations", "lead_wave_summaries"):
        write_query_csv(connection, table, output_dir / f"rdb_{table}.csv")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--queue", type=Path, default=ROOT/"refine-logs/queue_3arch/queue_state.json")
    result.add_argument("--checkpoint-db", type=Path, default=CHECKPOINT_DB)
    result.add_argument("--ludb-oracle-db", type=Path, default=LUDB_ORACLE_DB)
    result.add_argument("--min-ludb-signal-pearson-p05", type=float,
                        default=DEFAULT_MIN_LUDB_SIGNAL_PEARSON_P05)
    result.add_argument("--results-db", type=Path, default=ROOT/"results/ecgaim_rdb_oracle/ecgaim_rdb_oracle.sqlite")
    result.add_argument("--output-dir", type=Path, default=ROOT/"results/ecgaim_rdb_oracle")
    result.add_argument("--rdb-root", type=Path, default=ROOT/"data/rdb")
    result.add_argument("--mapping", type=Path, default=ROOT/"data/rdb/rdb_chapman_mapping.xlsx")
    result.add_argument("--poll-seconds", type=int, default=300)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--torch-threads", type=int, default=6)
    result.add_argument("--min-free-gb", type=float, default=5.0)
    result.add_argument("--max-records", type=int, default=0)
    result.add_argument("--max-models", type=int, default=0)
    result.add_argument("--model-id")
    result.add_argument("--include-flagged-duplicates", action="store_true")
    result.add_argument("--preflight", action="store_true", help="load/audit data and exit without opening a DB or model")
    result.add_argument("--once", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if min(args.poll_seconds,args.batch_size,args.torch_threads)<1 or args.min_free_gb<0: raise ValueError("invalid runtime option")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise RuntimeError("CPU-only contract violated")
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(args.torch_threads); torch.set_num_interop_threads(1)
    records,dataset_sha,audit=load_rdb(args.rdb_root,args.mapping,args.max_records,args.include_flagged_duplicates)
    selected,selection=load_ludb_oracle_selection(
        args.ludb_oracle_db,args.min_ludb_signal_pearson_p05
    )
    eligible=match_selected_models(
        completed_ecgaim_models(args.queue,args.checkpoint_db),selected
    )
    if args.preflight:
        print(json.dumps({"event":"preflight_complete","records":len(records),"dataset_sha256":dataset_sha,"audit":audit,"selection":selection,"selected_masks":[row["factorial_mask"] for row in eligible]},sort_keys=True)); return
    protocol_sha=protocol_identity(selection); connection=connect_results(args.results_db)
    store_static_dataset(connection,records,protocol_sha,dataset_sha,audit,selection)
    print(json.dumps({"event":"daemon_started","records":len(records),"eligible":len(eligible),"selection":selection,"device":"cpu","results_db":str(args.results_db),"at":utc_now()}),flush=True)
    try:
        while not STOP_REQUESTED:
            if args.model_id:
                eligible=[row for row in eligible if row["model_id"]==args.model_id]
                if not eligible: raise ValueError(f"eligible ECG-AIM model not found: {args.model_id}")
            pending=[]
            for row in eligible:
                old=connection.execute("SELECT status,checkpoint_sha256,protocol_sha256,dataset_sha256 FROM evaluations WHERE model_id=?",(row["model_id"],)).fetchone()
                if not old or tuple(old)!=("complete",row["sha256"],protocol_sha,dataset_sha): pending.append(row)
            if args.max_models: pending=pending[:args.max_models]
            print(json.dumps({"event":"pass_started","eligible":len(eligible),"pending":len(pending),"at":utc_now()}),flush=True)
            for identity in pending:
                free=shutil.disk_usage(args.results_db.parent).free
                if free < args.min_free_gb*1024**3:
                    print(json.dumps({"event":"disk_space_pause","free_bytes":free}),flush=True); break
                evaluate_model(connection,identity,records,protocol_sha,dataset_sha,args.batch_size)
                write_exports(connection,args.output_dir); connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            write_exports(connection,args.output_dir)
            if args.once or args.model_id or STOP_REQUESTED: break
            deadline=time.monotonic()+args.poll_seconds
            while not STOP_REQUESTED and time.monotonic()<deadline: time.sleep(min(1,deadline-time.monotonic()))
    finally: connection.close()
    print(json.dumps({"event":"daemon_stopped","at":utc_now()}),flush=True)


if __name__ == "__main__":
    main()
