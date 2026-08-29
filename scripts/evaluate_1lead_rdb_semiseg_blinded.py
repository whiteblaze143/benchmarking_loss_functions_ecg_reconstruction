#!/usr/bin/env python3
"""External blinded RDB evaluation for One-Lead reconstruction models with the frozen LUDB SemiSeg delineator."""

from __future__ import annotations
import torch
torch.set_num_threads(4)

import argparse
import datetime as dt
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
    predict_boundaries, quantile, sha256_file, stable_hash,
)
from scripts.onelead_checkpoint_store import (
    DEFAULT_CACHE as DEFAULT_CACHE_DIR, DEFAULT_DB as CHECKPOINT_DB, connect as connect_checkpoint_db,
)
from scripts.evaluate_ecgaim_ludb_blinded_daemon import BOUNDARIES, monotonic_match_indices

DEFAULT_DB = ROOT / "results/onelead_rdb_semiseg_relative_v2/compact.sqlite"
DEFAULT_CACHE_ROOT = ROOT / "data/rdb_wavelet_delineation_cache"
TARGET_FS = 500
TARGET_SAMPLES = 5000
STOP_REQUESTED = False

# Global state for original signal performance and matching sets
# Original Matches: (record_index, lead_index, boundary) -> set(matched_reference_indices)
ORIGINAL_MATCHES: dict[tuple[int, int, str], dict] = {}
# Original Predictions: (record_index, lead_index) -> dict[boundary, np.ndarray]
ORIGINAL_PREDICTIONS: dict[tuple[int, int], dict[str, np.ndarray]] = {}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)


def finite_mean(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else None


def finite_pearson(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(reconstruction, dtype=np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.square(left).sum() * np.square(right).sum()))
    return float(np.dot(left, right) / denominator) if denominator > 1e-15 else float("nan")


def extract_reference_boundaries(seg_12: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Extract lead-specific P/QRS/T onsets and offsets from [12, 5000] segmentation array."""
    lead_dict = {}
    for l in range(12):
        s = seg_12[l]
        b_dict = {}
        for kind, name in [(1, "P"), (2, "QRS"), (3, "T")]:
            on = np.where((s[1:] == kind) & (s[:-1] != kind))[0] + 1
            if s[0] == kind:
                on = np.insert(on, 0, 0)
            off = np.where((s[:-1] == kind) & (s[1:] != kind))[0]
            if s[-1] == kind:
                off = np.append(off, len(s) - 1)
            b_dict[f"{name}_onset"] = on.astype(np.int64)
            b_dict[f"{name}_offset"] = off.astype(np.int64)
        lead_dict[l] = b_dict
    return lead_dict


def load_cached_rdb_split(cache_root: Path, split: str = "test") -> tuple[list[dict[str, Any]], str]:
    split_dir = cache_root / split
    files = sorted(split_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no files found in {split_dir}")
        
    manifest_path = cache_root / "manifest.json"
    dataset_sha = sha256_file(manifest_path) if manifest_path.is_file() else sha256_file(files[0])
    
    records = []
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)
        wf = d["waveform"].float().numpy() # [12, 5000]
        seg = d["segmentation"].numpy() # [12, 5000]
        bounds = extract_reference_boundaries(seg)
        records.append({
            "record_id": d.get("record_id", p.stem),
            "patient_id": d.get("patient_id", "unknown"),
            "canonical_rhythm": d.get("canonical_rhythm", "unknown"),
            "signal": wf,
            "boundaries": bounds,
        })
    return records, dataset_sha


def score_relative_boundaries(
    records: list[dict[str, Any]], 
    model_predictions: dict[tuple[int, int], dict[str, np.ndarray]],
    lead_indices: Iterable[int],
    is_original: bool = False
) -> list[dict[str, Any]]:
    """Calculates per-lead relative metrics (Preserved, Recovered, Delta F1, Delta FP) for reconstructed signals."""
    rows = []
    tolerance20 = round(20 * TARGET_FS / 1000)
    tolerance150 = round(150 * TARGET_FS / 1000)
    
    for lead_index in lead_indices:
        lead_name = f"lead_{lead_index}"
        for boundary in BOUNDARIES:
            ref_total = 0
            model_pred_total = 0
            orig_pred_total = 0
            
            orig_tp20 = 0
            model_tp20 = 0
            
            orig_errors = []
            model_errors = []
            
            preserved_total = 0
            recovered_total = 0
            lost_total = 0
            
            for record_index, record in enumerate(records):
                reference = record["boundaries"][lead_index][boundary]
                ref_total += len(reference)
                
                model_pred = model_predictions[(record_index, lead_index)][boundary]
                model_pred_total += len(model_pred)
                
                # Model matching
                model_pairs20 = monotonic_match_indices(reference, model_pred, tolerance20)
                model_pairs150 = monotonic_match_indices(reference, model_pred, tolerance150)
                model_matched_refs = set(ref_idx for ref_idx, pred_idx in model_pairs20)
                model_tp20 += len(model_pairs20)
                model_errors.extend(
                    float(model_pred[right] - reference[left]) * 1000 / TARGET_FS
                    for left, right in model_pairs150
                )
                
                # Original matching
                cache_key = (record_index, lead_index, boundary)
                if is_original:
                    orig_matched_refs = model_matched_refs
                    orig_pred = model_pred
                    orig_pred_total += len(orig_pred)
                    orig_tp20 += len(model_pairs20)
                    orig_errors_rec = [float(orig_pred[right] - reference[left]) * 1000 / TARGET_FS for left, right in model_pairs150]
                    orig_errors.extend(orig_errors_rec)
                    
                    ORIGINAL_MATCHES[cache_key] = {
                        "matched_refs": orig_matched_refs,
                        "pred_total": len(orig_pred),
                        "tp20": len(model_pairs20),
                        "errors": orig_errors_rec,
                    }
                else:
                    cached = ORIGINAL_MATCHES.get(cache_key)
                    if cached:
                        orig_matched_refs = cached["matched_refs"]
                        orig_pred_total += cached["pred_total"]
                        orig_tp20 += cached["tp20"]
                        orig_errors.extend(cached["errors"])
                    else:
                        orig_matched_refs = set()
                
                # Calculate relative recovery sets
                preserved = orig_matched_refs & model_matched_refs
                recovered = model_matched_refs - orig_matched_refs
                lost = orig_matched_refs - model_matched_refs
                
                preserved_total += len(preserved)
                recovered_total += len(recovered)
                lost_total += len(lost)
            
            orig_denom = ref_total + orig_pred_total
            model_denom = ref_total + model_pred_total
            orig_f1 = 2 * orig_tp20 / orig_denom if orig_denom else None
            model_f1 = 2 * model_tp20 / model_denom if model_denom else None
            
            delta_f1 = (model_f1 - orig_f1) if (orig_f1 is not None and model_f1 is not None) else None
            retention_pct = (model_f1 / orig_f1 * 100) if (orig_f1 and model_f1 is not None) else None
            
            orig_err_arr = np.asarray(orig_errors, dtype=float)
            model_err_arr = np.asarray(model_errors, dtype=float)
            orig_mae = float(np.abs(orig_err_arr).mean()) if len(orig_err_arr) else None
            model_mae = float(np.abs(model_err_arr).mean()) if len(model_err_arr) else None
            delta_mae = (model_mae - orig_mae) if (orig_mae is not None and model_mae is not None) else None
            
            orig_fp = orig_pred_total - orig_tp20
            model_fp = model_pred_total - model_tp20
            delta_fp = model_fp - orig_fp
            
            rows.append({
                "lead_name": lead_name,
                "boundary": boundary,
                "reference_events": ref_total,
                "orig_predicted": orig_pred_total,
                "model_predicted": model_pred_total,
                "orig_tp20": orig_tp20,
                "model_tp20": model_tp20,
                "orig_f1_20ms": orig_f1,
                "model_f1_20ms": model_f1,
                "delta_f1_20ms": delta_f1,
                "retention_pct": retention_pct,
                "orig_mae_ms": orig_mae,
                "model_mae_ms": model_mae,
                "delta_mae_ms": delta_mae,
                "orig_fp": orig_fp,
                "model_fp": model_fp,
                "delta_fp": delta_fp,
                "preserved_events": preserved_total,
                "recovered_events": recovered_total,
                "lost_events": lost_total,
            })
            
    return rows


def predict_all(
    model: torch.nn.Module, preprocessor: SignalPreprocessor,
    signals: np.ndarray, lead_indices: Iterable[int], batch_size: int,
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    keys = [(record, lead) for record in range(len(signals)) for lead in lead_indices]
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
        model_id TEXT PRIMARY KEY,observed_lead INTEGER NOT NULL,factorial_mask TEXT NOT NULL,
        checkpoint_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,dataset_sha256 TEXT NOT NULL,
        status TEXT NOT NULL,attempts INTEGER NOT NULL,started_at TEXT,completed_at TEXT,
        duration_seconds REAL,error TEXT,signal_pearson_p05 REAL,signal_mse_p95 REAL,
        primary_mean_delta_f1 REAL,db_bytes_after INTEGER
      ) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS boundary_summaries(
        model_id TEXT NOT NULL,
        lead_name TEXT NOT NULL,
        boundary TEXT NOT NULL,
        reference_events INTEGER NOT NULL,
        orig_predicted INTEGER NOT NULL,
        model_predicted INTEGER NOT NULL,
        orig_tp20 INTEGER NOT NULL,
        model_tp20 INTEGER NOT NULL,
        orig_f1_20ms REAL,
        model_f1_20ms REAL,
        delta_f1_20ms REAL,
        retention_pct REAL,
        orig_mae_ms REAL,
        model_mae_ms REAL,
        delta_mae_ms REAL,
        orig_fp INTEGER NOT NULL,
        model_fp INTEGER NOT NULL,
        delta_fp INTEGER NOT NULL,
        preserved_events INTEGER NOT NULL,
        recovered_events INTEGER NOT NULL,
        lost_events INTEGER NOT NULL,
        PRIMARY KEY(model_id, lead_name, boundary),
        FOREIGN KEY(model_id) REFERENCES evaluations(model_id) ON DELETE CASCADE
      ) WITHOUT ROWID;
    """)
    return connection


def initialize(connection: sqlite3.Connection, protocol: dict[str, Any], dataset_sha: str) -> str:
    protocol_sha = stable_hash(protocol)
    expected = {
        "schema_version": "2", "protocol": "onelead_rdb_semiseg_blinded_v2_relative",
        "protocol_sha256": protocol_sha, "dataset_sha256": dataset_sha,
        "protocol_json": json.dumps(protocol, sort_keys=True, separators=(",", ":")),
        "storage": "aggregate-only; no raw predictions",
    }
    existing = dict(connection.execute("SELECT key,value FROM metadata"))
    if existing and any(existing.get(key) != value for key, value in expected.items()):
        raise RuntimeError("database belongs to a different protocol or dataset")
    with connection:
        connection.executemany("INSERT OR IGNORE INTO metadata VALUES(?,?)", expected.items())
    return protocol_sha


def is_complete(connection: sqlite3.Connection, model_id: str, sha256: str, protocol_sha: str, dataset_sha: str) -> bool:
    row = connection.execute(
        "SELECT status,checkpoint_sha256,protocol_sha256,dataset_sha256 FROM evaluations WHERE model_id=?",
        (model_id,),
    ).fetchone()
    return bool(row and tuple(row) == ("complete", sha256, protocol_sha, dataset_sha))


def signal_summary(records: list[dict[str, Any]], reconstructions: np.ndarray, missing_leads: tuple[int, ...]) -> tuple[float | None, float | None]:
    pearsons, mses = [], []
    for record, reconstructed in zip(records, reconstructions, strict=True):
        per_p, per_m = [], []
        for lead in missing_leads:
            target = record["signal"][lead]
            per_p.append(finite_pearson(target, reconstructed[lead]))
            per_m.append(float(np.square(target - reconstructed[lead]).mean()))
        pearsons.append(finite_mean(per_p)); mses.append(finite_mean(per_m))
    return quantile(pearsons, .05), quantile(mses, .95)


def store_result(
    connection: sqlite3.Connection, model_id: str, observed_lead: int, factorial_mask: str,
    sha256: str, protocol_sha: str, dataset_sha: str, rows: list[dict[str, Any]],
    started: float, pearson: float | None, mse: float | None, database: Path,
) -> None:
    # Compute mean delta_f1 across precordial leads (V1-V6 are indices 6-11)
    missing_precordial = [f"lead_{l}" for l in range(6, 12) if l != observed_lead]
    precordial_deltas = [r["delta_f1_20ms"] for r in rows if r["lead_name"] in missing_precordial and r["delta_f1_20ms"] is not None]
    primary_delta = finite_mean(precordial_deltas)
    
    with connection:
        connection.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
        connection.execute(
            "INSERT INTO evaluations VALUES(?,?,?,?,?,?,'complete',1,?,?,?,?,?,?,?,?)",
            (model_id, observed_lead, factorial_mask, sha256, protocol_sha, dataset_sha,
             utc_now(), utc_now(), time.perf_counter()-started, None, pearson, mse, primary_delta,
             database.stat().st_size if database.exists() else 0),
        )
        
        flat = [dict(row, model_id=model_id) for row in rows]
        connection.executemany(
            "INSERT INTO boundary_summaries (model_id,lead_name,boundary,reference_events,orig_predicted,model_predicted,orig_tp20,model_tp20,orig_f1_20ms,model_f1_20ms,delta_f1_20ms,retention_pct,orig_mae_ms,model_mae_ms,delta_mae_ms,orig_fp,model_fp,delta_fp,preserved_events,recovered_events,lost_events) "
            "VALUES(:model_id,:lead_name,:boundary,:reference_events,:orig_predicted,:model_predicted,:orig_tp20,:model_tp20,:orig_f1_20ms,:model_f1_20ms,:delta_f1_20ms,:retention_pct,:orig_mae_ms,:model_mae_ms,:delta_mae_ms,:orig_fp,:model_fp,:delta_fp,:preserved_events,:recovered_events,:lost_events)",
            flat,
        )
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(json.dumps({"event":"model_complete","model_id":model_id,"observed_lead":observed_lead,"mean_precordial_delta_f1":primary_delta,"r05":pearson}), flush=True)


def ensure_original_ceiling(
    connection: sqlite3.Connection, records: list[dict[str, Any]], delineator: torch.nn.Module,
    preprocessor: SignalPreprocessor, protocol_sha: str, dataset_sha: str,
    delineation_batch: int, database: Path,
) -> None:
    # If original predictions are already cached in memory, return
    if ORIGINAL_MATCHES:
        return
        
    print(json.dumps({"event": "evaluating_original_ceiling"}), flush=True)
    started = time.perf_counter()
    signals = np.stack([r["signal"] for r in records], axis=0) # [N, 12, 5000]
    all_leads = tuple(range(12))
    
    # predict_all computes delineations on true 12-lead signals
    predictions = predict_all(delineator, preprocessor, signals, all_leads, delineation_batch)
    
    # Compute relative scores (with is_original=True, populating ORIGINAL_MATCHES and ORIGINAL_PREDICTIONS)
    rows = score_relative_boundaries(records, predictions, all_leads, is_original=True)
    
    row = connection.execute(
        "SELECT status,protocol_sha256,dataset_sha256 FROM evaluations WHERE model_id='__original__'"
    ).fetchone()
    
    if not (row and tuple(row) == ("complete", protocol_sha, dataset_sha)):
        store_result(
            connection, "__original__", 0, "1111110", "original", protocol_sha, dataset_sha,
            rows, started, 1.0, 0.0, database,
        )


def load_onelead_reconstruction_adapter(checkpoint_path: Path, observed_lead: int):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    config = payload.get("config", {})
    
    from scripts.train_1lead_wavelet_ssl_mtl import build_model, forward_model
    from types import SimpleNamespace
    
    ns = SimpleNamespace(**config) if isinstance(config, dict) else config
    model = build_model(ns).to("cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    
    def adapter(batch_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            res = forward_model(model, batch_tensor, [observed_lead], compute_delineation=False, compute_ssl=False)
            pred = res["y_pred"].cpu().numpy()
            return pred
            
    return adapter


def run_evaluation_cycle(
    connection: sqlite3.Connection, records: list[dict[str, Any]], delineator: torch.nn.Module,
    preprocessor: SignalPreprocessor, protocol_sha: str, dataset_sha: str,
    database: Path, checkpoint_db: Path, cache_dir: Path,
    delineation_batch: int = 128, recon_batch: int = 32,
) -> int:
    ensure_original_ceiling(connection, records, delineator, preprocessor, protocol_sha, dataset_sha, delineation_batch, database)
    
    # Check registered checkpoints in onelead_checkpoint_store
    completed_count = 0
    raw_signals = np.stack([r["signal"] for r in records], axis=0) # [N, 12, 5000]
    

    if checkpoint_db.is_file():
        conn_chk = connect_checkpoint_db(checkpoint_db)
        
        # Determine if it's the new queue format or the old catalog
        tables = [t[0] for t in conn_chk.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "jobs" in tables:
            # It's a queue.sqlite
            models = conn_chk.execute(
                "SELECT id as model_id, '{}' as factorial_mask, 'unknown' as sha256, '' as local_path, '' as observed_leads_json FROM jobs WHERE status='completed' ORDER BY id"
            ).fetchall()
            is_queue = True
        else:
            # It's catalog.sqlite
            models = conn_chk.execute(
                "SELECT model_id, factorial_mask, sha256, local_path, observed_leads_json FROM checkpoints WHERE status='remote_verified' ORDER BY model_id"
            ).fetchall()
            is_queue = False
            
        conn_chk.close()
        
        for row in models:
            if STOP_REQUESTED:
                break
            model_id = row["model_id"]
            mask = row["factorial_mask"]
            sha256 = row["sha256"]
            
            if is_queue:
                obs_lead = int(model_id.split('_l')[-1]) if '_l' in model_id else 0
                ckpt_path = checkpoint_db.parent / "runs" / model_id / "resume.pt"
            else:
                obs_json = row["observed_leads_json"]
                obs_lead = json.loads(obs_json)[0] if obs_json else 0
                local_path = row["local_path"]
                ckpt_path = Path(local_path) if local_path and Path(local_path).is_file() else cache_dir / f"{model_id}.pt"
            
            if is_complete(connection, model_id, sha256, protocol_sha, dataset_sha):
                continue
                
            if not ckpt_path.is_file():
                continue
                    
            print(json.dumps({"event": "evaluating_model", "model_id": model_id, "observed_lead": obs_lead}), flush=True)
            started = time.perf_counter()
            
            try:
                adapter = load_onelead_reconstruction_adapter(ckpt_path, obs_lead)
                reconstructed_list = []
                
                for start_idx in range(0, len(raw_signals), recon_batch):
                    batch = torch.from_numpy(raw_signals[start_idx : start_idx + recon_batch]).float()
                    recon_batch_out = adapter(batch)
                    reconstructed_list.append(recon_batch_out)
                    
                reconstructed = np.concatenate(reconstructed_list, axis=0) # [N, 12, 5000]
                
                missing_leads = tuple(i for i in range(12) if i != obs_lead)
                
                predictions = predict_all(delineator, preprocessor, reconstructed, missing_leads, delineation_batch)
                
                rows = score_relative_boundaries(records, predictions, missing_leads, is_original=False)
                pearson, mse = signal_summary(records, reconstructed, missing_leads)
                
                store_result(connection, model_id, obs_lead, mask, sha256, protocol_sha, dataset_sha, rows, started, pearson, mse, database)
                completed_count += 1
                
            except Exception as exc:
                print(json.dumps({"event": "eval_error", "model_id": model_id, "error": str(exc)}), flush=True)
                with connection:
                    connection.execute("DELETE FROM evaluations WHERE model_id=?", (model_id,))
                    connection.execute(
                        "INSERT INTO evaluations VALUES(?,?,?,?,?,?,'failed',1,?,?,?,NULL,?,?,?,?)",
                        (model_id, obs_lead, mask, sha256, protocol_sha, dataset_sha,
                         utc_now(), utc_now(), time.perf_counter()-started, str(exc), None, None, None, 0),
                    )
                    
    return completed_count


def main():
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    
    p = argparse.ArgumentParser(description="One-Lead RDB Blinded Evaluation with SemiSeg UNet")
    p.add_argument("--database", type=Path, default=DEFAULT_DB)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint-db", type=Path, default=ROOT / "results/onelead_checkpoint_store/catalog.sqlite")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "checkpoints/onelead_cache")
    p.add_argument("--semiseg-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--watch", action="store_true", help="Continuously poll for new completed models")
    p.add_argument("--poll-seconds", type=int, default=30)
    args = p.parse_args()
    
    print(json.dumps({"event": "startup", "database": str(args.database)}), flush=True)
    
    # 1. Load cached split
    records, dataset_sha = load_cached_rdb_split(args.cache_root, args.split)
    print(json.dumps({"event": "rdb_split_loaded", "split": args.split, "records": len(records), "dataset_sha256": dataset_sha}), flush=True)
    
    # 2. Build SemiSeg Delineator
    delineator = build_delineator(args.semiseg_checkpoint, state="model_ema").to("cpu")
    preprocessor = SignalPreprocessor()
    print(json.dumps({"event": "delineator_loaded", "checkpoint": str(args.semiseg_checkpoint)}), flush=True)
    
    # 3. Protocol
    protocol = {
        "benchmark": "rdb_semiseg_blinded_1lead_v2_relative",
        "evaluator": "vit_tiny_mean_teacher_full_s42",
        "evaluator_checkpoint_sha256": sha256_file(args.semiseg_checkpoint),
        "split": args.split, "target_fs": TARGET_FS, "target_samples": TARGET_SAMPLES,
    }
    
    connection = connect(args.database)
    protocol_sha = initialize(connection, protocol, dataset_sha)
    
    while not STOP_REQUESTED:
        run_evaluation_cycle(
            connection, records, delineator, preprocessor, protocol_sha, dataset_sha,
            args.database, args.checkpoint_db, args.cache_dir,
        )
        if not args.watch or STOP_REQUESTED:
            break
        time.sleep(args.poll_seconds)
        
    connection.close()
    print(json.dumps({"event": "shutdown_clean"}), flush=True)


if __name__ == "__main__":
    main()
