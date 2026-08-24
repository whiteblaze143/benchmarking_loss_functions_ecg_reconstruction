#!/usr/bin/env python3
"""Compact CPU-only blinded LUDB evaluation using the trained SemiSeg ECG model.

The delineator receives only a waveform. LUDB annotations are loaded separately
and are used only after inference for scoring. The database stores aggregate
metrics and resumable job state; it deliberately stores no predictions, raw
per-record metrics, or CSV exports.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "external/semiseg/semi-seg-ecg"
for dependency in (ROOT / "external/semiseg/runtime_deps", VENDOR / "src", ROOT):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

import models.backbones as vendor_backbones  # noqa: E402
import models.decode_heads as vendor_heads  # noqa: E402
from models.encoder_decoder import EncoderDecoder  # noqa: E402
import utils.transforms as vendor_transforms  # noqa: E402

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
    completed_ecgaim_models,
    finite_mean,
    finite_pearson,
    load_ludb,
)
from scripts.evaluate_ecgaim_ludb_blinded_daemon import (  # noqa: E402
    BOUNDARIES,
    monotonic_match_indices,
    reference_result,
)

DEFAULT_CHECKPOINT = (
    ROOT / "results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42/best-MeanIoU.pth"
)
DEFAULT_SELECTION = (
    ROOT / "results/semiseg_ludb_training/vit_tiny_mean_teacher_full_s42/finalization_summary.json"
)
DEFAULT_DB = ROOT / "results/ecgaim_ludb_semiseg_blinded/compact.sqlite"
DEFAULT_TEST_INDEX = ROOT / "external/semiseg/official_indices/ludb/LUDB_test.csv"
STOP_REQUESTED = False


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)


def selected_checkpoint(checkpoint: Path, selection: Path, state: str) -> dict[str, str]:
    summary = json.loads(selection.read_text())
    selected = summary["selected"]
    resolved = checkpoint.resolve()
    if Path(selected["checkpoint"]).resolve() != resolved:
        raise RuntimeError("checkpoint does not match validation-selected finalization record")
    digest = sha256_file(resolved)
    if selected["checkpoint_sha256"] != digest or selected["state"] != state:
        raise RuntimeError("checkpoint hash/state does not match validation-selected finalization record")
    if int(summary["training_epochs"]) != 100:
        raise RuntimeError("selected delineator training is incomplete")
    return {"path": str(resolved), "sha256": digest, "state": state}


def official_test_records(
    ludb_root: Path, index_path: Path, max_records: int
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Load only the frozen subjects never used to train or select the delineator."""
    all_records, full_dataset_sha = load_ludb(ludb_root, 0)
    with index_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    test_ids = sorted({str(row["ID"]).strip() for row in rows}, key=lambda value: int(value))
    if len(test_ids) != 40:
        raise RuntimeError(f"official LUDB test split drift: expected 40 subjects, found {len(test_ids)}")
    by_id = {str(record["record_id"]): record for record in all_records}
    missing = sorted(set(test_ids) - set(by_id), key=lambda value: int(value))
    if missing:
        raise RuntimeError(f"official test subjects missing from LUDB: {missing}")
    selected = [by_id[record_id] for record_id in test_ids]
    if max_records:
        selected = selected[:max_records]
    split = {
        "name": "SemiSegECG official LUDB untouched test",
        "index_path": str(index_path.resolve()),
        "index_sha256": sha256_file(index_path),
        "full_dataset_sha256": full_dataset_sha,
        "all_test_subject_ids": test_ids,
        "evaluated_subject_ids": [str(record["record_id"]) for record in selected],
    }
    return selected, stable_hash(split), split


def build_delineator(checkpoint: Path, state: str) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    backbone_name, backbone_kwargs = next(iter(config["backbone"].items()))
    if backbone_name == "vit_seg_tiny":
        backbone_name = "vit_tiny"
    head_name, head_kwargs = next(iter(config["decode_head"].items()))
    model = EncoderDecoder(
        backbone=getattr(vendor_backbones, backbone_name)(**backbone_kwargs),
        decode_head=getattr(vendor_heads, head_name)(**head_kwargs),
        decode_head_loss=nn.CrossEntropyLoss(),
        use_latent_projection=bool(config.get("use_latent_projection", False)),
        projection_in_dim=config.get("projection_in_dim"),
        projection_out_dim=config.get("projection_out_dim"),
    )
    model.load_state_dict(payload[state], strict=True)
    return model.float().eval()


class SignalPreprocessor:
    """Released SemiSeg preprocessing with no label-dependent padding."""

    def __init__(self) -> None:
        self.resample = vendor_transforms.Resample(target_length=2500)
        self.highpass = vendor_transforms.HighpassFilter(fs=250, cutoff=0.67)
        self.lowpass = vendor_transforms.LowpassFilter(fs=250, cutoff=40)
        self.standardize = vendor_transforms.Standardize(axis=(-1, -2))

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        signal_values = np.asarray(waveform, dtype=np.float64)[None, :]
        signal_values = self.resample(signal_values)
        signal_values = self.lowpass(self.highpass(signal_values))
        return self.standardize(signal_values).astype(np.float32)


def class_boundaries(mask_500hz: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for class_id, wave in ((1, "P"), (2, "QRS"), (3, "T")):
        active = mask_500hz == class_id
        changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(changes == 1).astype(np.int64)
        stops = (np.flatnonzero(changes == -1) - 1).astype(np.int64)
        output[f"{wave}_onset"] = starts
        output[f"{wave}_offset"] = stops
    return output


def predict_boundaries(
    model: nn.Module, preprocessor: SignalPreprocessor, waveforms: list[np.ndarray], batch_size: int
) -> list[dict[str, np.ndarray]]:
    prepared = [preprocessor(item) for item in waveforms]
    masks: list[np.ndarray] = []
    for offset in range(0, len(prepared), batch_size):
        batch = torch.from_numpy(np.stack(prepared[offset:offset + batch_size]))
        with torch.inference_mode():
            logits = model(batch)["seg_logits"]
        for prediction in logits.argmax(dim=1).cpu().numpy():
            masks.append(np.repeat(prediction.astype(np.int8), 2)[:5000])
    return [class_boundaries(mask) for mask in masks]


def quantile(values: Iterable[float], probability: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if len(array) else None


def score_boundaries(
    records: list[dict[str, Any]], predictions: list[dict[str, np.ndarray]], lead_indices: Iterable[int]
) -> list[dict[str, Any]]:
    accumulators = {
        boundary: {"reference": 0, "predicted": 0, "tp20": 0, "tp150": 0, "errors": [], "record_f1": []}
        for boundary in BOUNDARIES
    }
    cursor = 0
    tolerance_20 = round(20 * TARGET_FS / 1000)
    tolerance_150 = round(150 * TARGET_FS / 1000)
    for record in records:
        for lead_index in lead_indices:
            reference = reference_result(record["annotations"][LEADS[lead_index]])
            predicted = predictions[cursor]
            cursor += 1
            for boundary in BOUNDARIES:
                real, estimate = reference.landmarks[boundary], predicted[boundary]
                pairs20 = monotonic_match_indices(real, estimate, tolerance_20)
                pairs150 = monotonic_match_indices(real, estimate, tolerance_150)
                item = accumulators[boundary]
                item["reference"] += len(real)
                item["predicted"] += len(estimate)
                item["tp20"] += len(pairs20)
                item["tp150"] += len(pairs150)
                denominator = len(real) + len(estimate)
                if denominator:
                    item["record_f1"].append(2 * len(pairs20) / denominator)
                item["errors"].extend(
                    (float(estimate[right] - real[left]) * 1000 / TARGET_FS)
                    for left, right in pairs150
                )
    rows = []
    for boundary, item in accumulators.items():
        denominator = item["reference"] + item["predicted"]
        errors = np.asarray(item["errors"], dtype=float)
        rows.append({
            "boundary": boundary,
            "reference_events": item["reference"],
            "predicted_events": item["predicted"],
            "tp_20ms": item["tp20"],
            "micro_f1_20ms": 2 * item["tp20"] / denominator if denominator else None,
            "macro_f1_20ms": finite_mean(item["record_f1"]),
            "matched_150ms": item["tp150"],
            "bias_ms": float(errors.mean()) if len(errors) else None,
            "mae_ms": float(np.abs(errors).mean()) if len(errors) else None,
            "p95_abs_ms": quantile(np.abs(errors), 0.95),
        })
    return rows


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_size_limit=16777216")
    connection.executescript("""
      CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS evaluations(
        model_id TEXT PRIMARY KEY,factorial_mask TEXT NOT NULL,checkpoint_sha256 TEXT NOT NULL,
        protocol_sha256 TEXT NOT NULL,dataset_sha256 TEXT NOT NULL,status TEXT NOT NULL,
        attempts INTEGER NOT NULL,started_at TEXT,completed_at TEXT,duration_seconds REAL,
        error TEXT,signal_pearson_p05 REAL,signal_mse_p95 REAL,primary_mean_micro_f1_20ms REAL,
        db_bytes_after INTEGER
      ) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS boundary_summaries(
        model_id TEXT NOT NULL,boundary TEXT NOT NULL,reference_events INTEGER NOT NULL,
        predicted_events INTEGER NOT NULL,tp_20ms INTEGER NOT NULL,micro_f1_20ms REAL,
        macro_f1_20ms REAL,matched_150ms INTEGER NOT NULL,bias_ms REAL,mae_ms REAL,
        p95_abs_ms REAL,PRIMARY KEY(model_id,boundary),
        FOREIGN KEY(model_id) REFERENCES evaluations(model_id) ON DELETE CASCADE
      ) WITHOUT ROWID;
    """)
    return connection


def initialize(
    connection: sqlite3.Connection, protocol: dict[str, Any], dataset_sha: str
) -> str:
    protocol_sha = stable_hash(protocol)
    expected = {
        "schema_version": "1", "protocol": "semiseg_mt_blinded_ludb_compact_v1",
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
        per_record_p, per_record_m = [], []
        for lead_index in MISSING:
            target = record["signal"][lead_index]
            per_record_p.append(finite_pearson(target, reconstructed[lead_index]))
            per_record_m.append(float(np.square(target - reconstructed[lead_index]).mean()))
        pearsons.append(finite_mean(per_record_p)); mses.append(finite_mean(per_record_m))
    return quantile(pearsons, 0.05), quantile(mses, 0.95)


def ensure_original_ceiling(
    connection: sqlite3.Connection, records: list[dict[str, Any]], delineator: nn.Module,
    preprocessor: SignalPreprocessor, protocol_sha: str, dataset_sha: str,
    delineation_batch: int, database_path: Path,
) -> None:
    row = connection.execute(
        "SELECT status,protocol_sha256,dataset_sha256 FROM evaluations WHERE model_id='__original__'"
    ).fetchone()
    if row and tuple(row) == ("complete", protocol_sha, dataset_sha):
        return
    started = time.perf_counter()
    waveforms = [record["signal"][lead] for record in records for lead in MISSING]
    predictions = predict_boundaries(delineator, preprocessor, waveforms, delineation_batch)
    boundary_rows = score_boundaries(records, predictions, MISSING)
    primary = finite_mean([item["micro_f1_20ms"] for item in boundary_rows])
    with connection:
        connection.execute("DELETE FROM evaluations WHERE model_id='__original__'")
        connection.execute(
            "INSERT INTO evaluations(model_id,factorial_mask,checkpoint_sha256,protocol_sha256,dataset_sha256,status,attempts,started_at,completed_at,duration_seconds,signal_pearson_p05,signal_mse_p95,primary_mean_micro_f1_20ms,db_bytes_after) VALUES('__original__','original','original_ludb',?,?,'complete',1,?,?,?,1.0,0.0,?,?)",
            (protocol_sha, dataset_sha, utc_now(), utc_now(), time.perf_counter() - started, primary, database_path.stat().st_size),
        )
        connection.executemany(
            "INSERT INTO boundary_summaries VALUES(:model_id,:boundary,:reference_events,:predicted_events,:tp_20ms,:micro_f1_20ms,:macro_f1_20ms,:matched_150ms,:bias_ms,:mae_ms,:p95_abs_ms)",
            [dict(item, model_id="__original__") for item in boundary_rows],
        )
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(json.dumps({"event":"original_ceiling_complete","mean_micro_f1_20ms":primary}), flush=True)


def evaluate(
    connection: sqlite3.Connection, identity: dict[str, Any], records: list[dict[str, Any]],
    delineator: nn.Module, preprocessor: SignalPreprocessor, protocol_sha: str,
    dataset_sha: str, reconstruction_batch: int, delineation_batch: int, database_path: Path,
) -> None:
    model_id = identity["model_id"]
    old = connection.execute("SELECT attempts FROM evaluations WHERE model_id=?", (model_id,)).fetchone()
    attempts = int(old[0]) + 1 if old else 1
    with connection:
        connection.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
        connection.execute(
            "INSERT INTO evaluations(model_id,factorial_mask,checkpoint_sha256,protocol_sha256,dataset_sha256,status,attempts,started_at) VALUES(?,?,?,?,?,'running',?,?)",
            (model_id, identity["factorial_mask"], identity["sha256"], protocol_sha, dataset_sha, attempts, utc_now()),
        )
    started = time.perf_counter()
    adapter = None
    try:
        adapter = load_adapter({
            "id": model_id, "kind": "alitok",
            "checkpoint": f"checkpoints/factorial_ecg_aim_{identity['factorial_mask']}_s42.pt",
            "observed_leads": list(OBSERVED),
        }, torch.device("cpu"))
        batches = []
        for offset in range(0, len(records), reconstruction_batch):
            target = torch.from_numpy(np.stack([r["signal"] for r in records[offset:offset + reconstruction_batch]]))
            with torch.inference_mode():
                batches.append(adapter.reconstruct(target).cpu().numpy().astype(np.float32))
        reconstructions = np.concatenate(batches)
        waveforms = [recon[lead] for recon in reconstructions for lead in MISSING]
        predictions = predict_boundaries(delineator, preprocessor, waveforms, delineation_batch)
        boundary_rows = score_boundaries(records, predictions, MISSING)
        pearson_p05, mse_p95 = signal_summary(records, reconstructions)
        primary = finite_mean([row["micro_f1_20ms"] for row in boundary_rows])
        with connection:
            connection.executemany(
                "INSERT INTO boundary_summaries VALUES(:model_id,:boundary,:reference_events,:predicted_events,:tp_20ms,:micro_f1_20ms,:macro_f1_20ms,:matched_150ms,:bias_ms,:mae_ms,:p95_abs_ms)",
                [dict(row, model_id=model_id) for row in boundary_rows],
            )
            connection.execute(
                "UPDATE evaluations SET status='complete',completed_at=?,duration_seconds=?,error=NULL,signal_pearson_p05=?,signal_mse_p95=?,primary_mean_micro_f1_20ms=?,db_bytes_after=? WHERE model_id=?",
                (utc_now(), time.perf_counter() - started, pearson_p05, mse_p95, primary, database_path.stat().st_size, model_id),
            )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(json.dumps({"event":"model_complete","model_id":model_id,"macro_f1_20ms":primary}), flush=True)
    except Exception as error:
        with connection:
            connection.execute(
                "UPDATE evaluations SET status='error',completed_at=?,duration_seconds=?,error=? WHERE model_id=?",
                (utc_now(), time.perf_counter() - started, f"{type(error).__name__}: {error}", model_id),
            )
        raise
    finally:
        del adapter
        with store_lock(CHECKPOINT_DB):
            catalog = connect_checkpoint_db(CHECKPOINT_DB)
            try:
                prune_cache(catalog, DEFAULT_CACHE_DIR, 0.0)
            finally:
                catalog.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "refine-logs/queue_3arch/queue_state.json")
    parser.add_argument("--checkpoint-db", type=Path, default=CHECKPOINT_DB)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ludb-root", type=Path, default=ROOT / "data/ludb")
    parser.add_argument("--delineator-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--selection-summary", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--test-index", type=Path, default=DEFAULT_TEST_INDEX)
    parser.add_argument("--state", choices=("model", "model_ema"), default="model_ema")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--reconstruction-batch-size", type=int, default=2)
    parser.add_argument("--delineation-batch-size", type=int, default=32)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-models", type=int, default=0)
    parser.add_argument("--model-id")
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    args = parser.parse_args()
    if min(args.torch_threads, args.reconstruction_batch_size, args.delineation_batch_size) < 1:
        raise ValueError("thread and batch sizes must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only contract violated")
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(args.torch_threads); torch.set_num_interop_threads(1)
    selected = selected_checkpoint(args.delineator_checkpoint, args.selection_summary, args.state)
    records, dataset_sha, split = official_test_records(
        args.ludb_root, args.test_index, args.max_records
    )
    protocol = {
        "name": "semiseg_mt_blinded_ludb_compact_v1",
        "evaluator_sha256": sha256_file(Path(__file__)),
        "loader_sha256": sha256_file(ROOT / "scripts/evaluate_ecgaim_ludb_daemon.py"),
        "registry_sha256": sha256_file(ROOT / "scripts/evaluate_comprehensive_registry.py"),
        "semiseg_checkpoint": selected,
        "test_split": split,
        "preprocessing": "released resample-250Hz/highpass-0.67Hz/lowpass-40Hz/standardize; no label padding",
        "input": "waveform only; LUDB annotations scoring-only",
        "endpoint": "missing-lead six-boundary mean micro-F1 at 20ms",
    }
    connection = connect(args.results_db)
    protocol_sha = initialize(connection, protocol, dataset_sha)
    delineator = build_delineator(args.delineator_checkpoint, args.state)
    preprocessor = SignalPreprocessor()
    print(json.dumps({"event":"started","records":len(records),"protocol_sha256":protocol_sha,"database":str(args.results_db)}), flush=True)
    try:
        free = os.statvfs(args.results_db.parent)
        free_bytes = free.f_bavail * free.f_frsize
        if free_bytes < args.min_free_gib * 1024**3:
            raise RuntimeError(
                f"disk gate blocks original ceiling: {free_bytes} bytes free"
            )
        ensure_original_ceiling(
            connection, records, delineator, preprocessor, protocol_sha, dataset_sha,
            args.delineation_batch_size, args.results_db,
        )
        eligible = completed_ecgaim_models(args.queue, args.checkpoint_db)
        if args.model_id:
            eligible = [row for row in eligible if row["model_id"] == args.model_id]
        pending = [row for row in eligible if not is_complete(connection, row, protocol_sha, dataset_sha)]
        if args.max_models:
            pending = pending[:args.max_models]
        print(json.dumps({"event":"queue","eligible":len(eligible),"pending":len(pending)}), flush=True)
        for identity in pending:
            if STOP_REQUESTED:
                break
            free = os.statvfs(args.results_db.parent)
            free_bytes = free.f_bavail * free.f_frsize
            if free_bytes < args.min_free_gib * 1024**3:
                print(json.dumps({"event":"disk_pause","free_bytes":free_bytes}), flush=True)
                break
            evaluate(
                connection, identity, records, delineator, preprocessor, protocol_sha,
                dataset_sha, args.reconstruction_batch_size, args.delineation_batch_size,
                args.results_db,
            )
    finally:
        connection.close()
    print(json.dumps({"event":"stopped"}), flush=True)


if __name__ == "__main__":
    main()
