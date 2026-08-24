#!/usr/bin/env python3
"""Compact external RDB blinded evaluation with the frozen LUDB SemiSeg model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.evaluate_ecgaim_ludb_semiseg_blinded import (
    DEFAULT_CHECKPOINT, DEFAULT_SELECTION, SignalPreprocessor, build_delineator,
    predict_boundaries, quantile, selected_checkpoint, sha256_file, stable_hash,
)
from scripts.checkpoint_store import (
    DEFAULT_CACHE_DIR, DEFAULT_DB as CHECKPOINT_DB, connect as connect_checkpoint_db,
    prune_cache, store_lock,
)
from scripts.evaluate_comprehensive_registry import load_adapter
from scripts.evaluate_ecgaim_ludb_daemon import completed_ecgaim_models
from scripts.evaluate_ecgaim_ludb_blinded_daemon import BOUNDARIES, monotonic_match_indices
from scripts.evaluate_ecgaim_rdb_oracle_daemon import (
    DEFAULT_MIN_LUDB_SIGNAL_PEARSON_P05, LUDB_ORACLE_DB,
    load_ludb_oracle_selection, match_selected_models,
)
from scripts.rdb_oracle import (
    DERIVED_LIMB, LEADS, OBSERVED, PRIMARY_PRECORDIAL, TARGET_FS,
    finite_mean, finite_pearson, load_rdb,
)

ALL_MISSING = tuple(index for index in range(12) if index not in OBSERVED)
LEAD_GROUPS = {
    "all_missing": ALL_MISSING,
    "primary_missing_precordial": PRIMARY_PRECORDIAL,
    "derived_limb_control": DERIVED_LIMB,
}
DEFAULT_DB = ROOT / "results/ecgaim_rdb_semiseg_blinded/compact.sqlite"
STOP_REQUESTED = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)


def reference_boundaries(intervals: np.ndarray) -> dict[str, np.ndarray]:
    result = {boundary: [] for boundary in BOUNDARIES}
    for kind, start, end in np.asarray(intervals, dtype=np.int64):
        wave = ("P", "QRS", "T")[int(kind)]
        result[f"{wave}_onset"].append(int(start))
        result[f"{wave}_offset"].append(int(end))
    return {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}


def score_group(
    records: list[dict[str, Any]], predictions: dict[tuple[int, int], dict[str, np.ndarray]],
    lead_indices: Iterable[int],
) -> list[dict[str, Any]]:
    accumulator = {
        boundary: {"reference": 0, "predicted": 0, "tp20": 0, "tp150": 0, "errors": [], "record_f1": []}
        for boundary in BOUNDARIES
    }
    tolerance20, tolerance150 = round(20 * TARGET_FS / 1000), round(150 * TARGET_FS / 1000)
    for record_index, record in enumerate(records):
        for lead_index in lead_indices:
            reference = reference_boundaries(record["annotations"][lead_index])
            predicted = predictions[(record_index, lead_index)]
            for boundary in BOUNDARIES:
                real, estimate = reference[boundary], predicted[boundary]
                pairs20 = monotonic_match_indices(real, estimate, tolerance20)
                pairs150 = monotonic_match_indices(real, estimate, tolerance150)
                item = accumulator[boundary]
                item["reference"] += len(real); item["predicted"] += len(estimate)
                item["tp20"] += len(pairs20); item["tp150"] += len(pairs150)
                denominator = len(real) + len(estimate)
                if denominator:
                    item["record_f1"].append(2 * len(pairs20) / denominator)
                item["errors"].extend(
                    float(estimate[right] - real[left]) * 1000 / TARGET_FS
                    for left, right in pairs150
                )
    rows = []
    for boundary, item in accumulator.items():
        denominator = item["reference"] + item["predicted"]
        errors = np.asarray(item["errors"], dtype=float)
        rows.append({
            "boundary": boundary, "reference_events": item["reference"],
            "predicted_events": item["predicted"], "tp_20ms": item["tp20"],
            "micro_f1_20ms": 2 * item["tp20"] / denominator if denominator else None,
            "macro_f1_20ms": finite_mean(item["record_f1"]),
            "matched_150ms": item["tp150"],
            "bias_ms": float(errors.mean()) if len(errors) else None,
            "mae_ms": float(np.abs(errors).mean()) if len(errors) else None,
            "p95_abs_ms": quantile(np.abs(errors), .95),
        })
    return rows


def predict_all(
    model: torch.nn.Module, preprocessor: SignalPreprocessor,
    signals: np.ndarray, batch_size: int,
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    keys = [(record, lead) for record in range(len(signals)) for lead in ALL_MISSING]
    waveforms = [signals[record, lead] for record, lead in keys]
    values = predict_boundaries(model, preprocessor, waveforms, batch_size)
    return dict(zip(keys, values, strict=True))


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL"); connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON"); connection.execute("PRAGMA journal_size_limit=16777216")
    connection.executescript("""
      CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS evaluations(
        model_id TEXT PRIMARY KEY,factorial_mask TEXT NOT NULL,checkpoint_sha256 TEXT NOT NULL,
        protocol_sha256 TEXT NOT NULL,dataset_sha256 TEXT NOT NULL,status TEXT NOT NULL,
        attempts INTEGER NOT NULL,started_at TEXT,completed_at TEXT,duration_seconds REAL,error TEXT,
        signal_pearson_p05 REAL,signal_mse_p95 REAL,primary_mean_micro_f1_20ms REAL,db_bytes_after INTEGER
      ) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS boundary_summaries(
        model_id TEXT NOT NULL,lead_group TEXT NOT NULL,boundary TEXT NOT NULL,
        reference_events INTEGER NOT NULL,predicted_events INTEGER NOT NULL,tp_20ms INTEGER NOT NULL,
        micro_f1_20ms REAL,macro_f1_20ms REAL,matched_150ms INTEGER NOT NULL,
        bias_ms REAL,mae_ms REAL,p95_abs_ms REAL,
        PRIMARY KEY(model_id,lead_group,boundary),
        FOREIGN KEY(model_id) REFERENCES evaluations(model_id) ON DELETE CASCADE
      ) WITHOUT ROWID;
    """)
    return connection


def initialize(connection: sqlite3.Connection, protocol: dict[str, Any], dataset_sha: str) -> str:
    protocol_sha = stable_hash(protocol)
    expected = {
        "schema_version": "1", "protocol": "semiseg_mt_blinded_rdb_compact_v1",
        "protocol_sha256": protocol_sha, "dataset_sha256": dataset_sha,
        "protocol_json": json.dumps(protocol, sort_keys=True, separators=(",", ":")),
        "storage": "aggregate-only; no raw predictions, per-record rows, or CSV exports",
    }
    existing = dict(connection.execute("SELECT key,value FROM metadata"))
    if existing and any(existing.get(key) != value for key, value in expected.items()):
        raise RuntimeError("database belongs to a different protocol or dataset")
    with connection:
        connection.executemany("INSERT OR IGNORE INTO metadata VALUES(?,?)", expected.items())
    return protocol_sha


def is_complete(connection: sqlite3.Connection, identity: dict[str, Any], protocol_sha: str, dataset_sha: str) -> bool:
    row = connection.execute(
        "SELECT status,checkpoint_sha256,protocol_sha256,dataset_sha256 FROM evaluations WHERE model_id=?",
        (identity["model_id"],),
    ).fetchone()
    return bool(row and tuple(row) == ("complete", identity["sha256"], protocol_sha, dataset_sha))


def signal_summary(records: list[dict[str, Any]], reconstructions: np.ndarray) -> tuple[float | None, float | None]:
    pearsons, mses = [], []
    for record, reconstructed in zip(records, reconstructions, strict=True):
        per_p, per_m = [], []
        for lead in PRIMARY_PRECORDIAL:
            target = record["signal"][lead]
            per_p.append(finite_pearson(target, reconstructed[lead]))
            per_m.append(float(np.square(target - reconstructed[lead]).mean()))
        pearsons.append(finite_mean(per_p)); mses.append(finite_mean(per_m))
    return quantile(pearsons, .05), quantile(mses, .95)


def store_result(
    connection: sqlite3.Connection, model_id: str, identity: dict[str, Any], protocol_sha: str,
    dataset_sha: str, rows: dict[str, list[dict[str, Any]]], started: float,
    pearson: float | None, mse: float | None, database: Path,
) -> None:
    primary = finite_mean(row["micro_f1_20ms"] for row in rows["primary_missing_precordial"])
    with connection:
        connection.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
        connection.execute(
            "INSERT INTO evaluations VALUES(?,?,?,?,?,'complete',1,?,?,?,?,?,?,?,?)",
            (model_id, identity["factorial_mask"], identity["sha256"], protocol_sha, dataset_sha,
             utc_now(), utc_now(), time.perf_counter()-started, None, pearson, mse, primary,
             database.stat().st_size),
        )
        flat = [dict(row, model_id=model_id, lead_group=group) for group, values in rows.items() for row in values]
        connection.executemany(
            "INSERT INTO boundary_summaries VALUES(:model_id,:lead_group,:boundary,:reference_events,:predicted_events,:tp_20ms,:micro_f1_20ms,:macro_f1_20ms,:matched_150ms,:bias_ms,:mae_ms,:p95_abs_ms)", flat,
        )
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(json.dumps({"event":"model_complete","model_id":model_id,"primary_mean_micro_f1_20ms":primary}), flush=True)


def ensure_original_ceiling(
    connection: sqlite3.Connection, records: list[dict[str, Any]], delineator: torch.nn.Module,
    preprocessor: SignalPreprocessor, protocol_sha: str, dataset_sha: str,
    delineation_batch: int, database: Path,
) -> None:
    row=connection.execute(
        "SELECT status,protocol_sha256,dataset_sha256 FROM evaluations WHERE model_id='__original__'"
    ).fetchone()
    if row and tuple(row)==("complete",protocol_sha,dataset_sha): return
    started=time.perf_counter(); signals=np.stack([record["signal"] for record in records])
    predictions=predict_all(delineator,preprocessor,signals,delineation_batch)
    rows={group:score_group(records,predictions,leads) for group,leads in LEAD_GROUPS.items()}
    identity={"factorial_mask":"original","sha256":"original_rdb"}
    store_result(connection,"__original__",identity,protocol_sha,dataset_sha,rows,started,1.0,0.0,database)


def evaluate_model(
    connection: sqlite3.Connection, identity: dict[str, Any], records: list[dict[str, Any]],
    delineator: torch.nn.Module, preprocessor: SignalPreprocessor, protocol_sha: str,
    dataset_sha: str, reconstruction_batch: int, delineation_batch: int, database: Path,
) -> None:
    started = time.perf_counter(); adapter = None
    model_id = identity["model_id"]
    with connection:
        connection.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
        connection.execute(
            "INSERT INTO evaluations(model_id,factorial_mask,checkpoint_sha256,protocol_sha256,dataset_sha256,status,attempts,started_at) VALUES(?,?,?,?,?,'running',1,?)",
            (model_id, identity["factorial_mask"], identity["sha256"], protocol_sha, dataset_sha, utc_now()),
        )
    try:
        adapter = load_adapter({
            "id":model_id,"kind":"alitok",
            "checkpoint":f"checkpoints/factorial_ecg_aim_{identity['factorial_mask']}_s42.pt",
            "observed_leads":list(OBSERVED),
        }, torch.device("cpu"))
        batches=[]
        for offset in range(0,len(records),reconstruction_batch):
            target=torch.from_numpy(np.stack([r["signal"] for r in records[offset:offset+reconstruction_batch]]))
            with torch.inference_mode(): batches.append(adapter.reconstruct(target).cpu().numpy().astype(np.float32))
        reconstructions=np.concatenate(batches)
        predictions=predict_all(delineator,preprocessor,reconstructions,delineation_batch)
        rows={group:score_group(records,predictions,leads) for group,leads in LEAD_GROUPS.items()}
        pearson,mse=signal_summary(records,reconstructions)
        store_result(connection,model_id,identity,protocol_sha,dataset_sha,rows,started,pearson,mse,database)
    except Exception as error:
        with connection:
            connection.execute("UPDATE evaluations SET status='error',completed_at=?,duration_seconds=?,error=? WHERE model_id=?",
                               (utc_now(),time.perf_counter()-started,f"{type(error).__name__}: {error}",model_id))
        raise
    finally:
        del adapter
        with store_lock(CHECKPOINT_DB):
            catalog=connect_checkpoint_db(CHECKPOINT_DB)
            try: prune_cache(catalog,DEFAULT_CACHE_DIR,0.0)
            finally: catalog.close()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue",type=Path,default=ROOT/"refine-logs/queue_3arch/queue_state.json")
    parser.add_argument("--checkpoint-db",type=Path,default=CHECKPOINT_DB)
    parser.add_argument("--ludb-oracle-db",type=Path,default=LUDB_ORACLE_DB)
    parser.add_argument("--results-db",type=Path,default=DEFAULT_DB)
    parser.add_argument("--rdb-root",type=Path,default=ROOT/"data/rdb")
    parser.add_argument("--mapping",type=Path,default=ROOT/"data/rdb/rdb_chapman_mapping.xlsx")
    parser.add_argument("--delineator-checkpoint",type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument("--selection-summary",type=Path,default=DEFAULT_SELECTION)
    parser.add_argument("--state",choices=("model","model_ema"),default="model_ema")
    parser.add_argument("--torch-threads",type=int,default=6)
    parser.add_argument("--reconstruction-batch-size",type=int,default=8)
    parser.add_argument("--delineation-batch-size",type=int,default=64)
    parser.add_argument("--max-records",type=int,default=0); parser.add_argument("--max-models",type=int,default=0)
    parser.add_argument("--model-id"); parser.add_argument("--min-free-gib",type=float,default=8.0)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise RuntimeError("CPU-only contract violated")
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    torch.set_num_threads(args.torch_threads); torch.set_num_interop_threads(1)
    selected_checkpoint(args.delineator_checkpoint,args.selection_summary,args.state)
    records,dataset_sha,audit=load_rdb(args.rdb_root,args.mapping,args.max_records,False)
    selected,selection=load_ludb_oracle_selection(args.ludb_oracle_db,DEFAULT_MIN_LUDB_SIGNAL_PEARSON_P05)
    eligible=match_selected_models(completed_ecgaim_models(args.queue,args.checkpoint_db),selected)
    if args.model_id: eligible=[row for row in eligible if row["model_id"]==args.model_id]
    protocol={
        "name":"semiseg_mt_blinded_rdb_compact_v1","evaluator_sha256":sha256_file(Path(__file__)),
        "rdb_loader_sha256":sha256_file(ROOT/"scripts/rdb_oracle.py"),
        "registry_sha256":sha256_file(ROOT/"scripts/evaluate_comprehensive_registry.py"),
        "delineator":{"path":str(args.delineator_checkpoint.resolve()),"sha256":sha256_file(args.delineator_checkpoint),"state":args.state},
        "dataset_audit":audit,"selection":selection,
        "input":"waveform only; RDB lead-specific regions scoring-only",
        "endpoint":"primary V1/V3-V6 six-boundary mean micro-F1 at 20ms",
    }
    connection=connect(args.results_db); protocol_sha=initialize(connection,protocol,dataset_sha)
    delineator=build_delineator(args.delineator_checkpoint,args.state); preprocessor=SignalPreprocessor()
    print(json.dumps({"event":"started","records":len(records),"eligible":len(eligible),"protocol_sha256":protocol_sha}),flush=True)
    try:
        stat=os.statvfs(args.results_db.parent); free=stat.f_bavail*stat.f_frsize
        if free<args.min_free_gib*1024**3: raise RuntimeError("disk gate blocks original ceiling")
        ensure_original_ceiling(connection,records,delineator,preprocessor,protocol_sha,dataset_sha,
                                args.delineation_batch_size,args.results_db)
        pending=[row for row in eligible if not is_complete(connection,row,protocol_sha,dataset_sha)]
        if args.max_models: pending=pending[:args.max_models]
        print(json.dumps({"event":"queue","pending":len(pending)}),flush=True)
        for identity in pending:
            if STOP_REQUESTED: break
            stat=os.statvfs(args.results_db.parent); free=stat.f_bavail*stat.f_frsize
            if free<args.min_free_gib*1024**3:
                print(json.dumps({"event":"disk_pause","free_bytes":free}),flush=True); break
            evaluate_model(connection,identity,records,delineator,preprocessor,protocol_sha,dataset_sha,
                           args.reconstruction_batch_size,args.delineation_batch_size,args.results_db)
    finally: connection.close()
    print(json.dumps({"event":"stopped"}),flush=True)


if __name__=="__main__": main()
