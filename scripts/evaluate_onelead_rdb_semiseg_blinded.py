#!/usr/bin/env python3
"""CPU-only, compact, resumable external RDB evaluation for one-lead models.

The reconstructor and frozen LUDB SemiSeg delineator receive waveforms only.
RDB labels from the frozen 360-record test cache are opened only while scoring.
Only aggregate results and explicit job state are stored in SQLite.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import signal
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from scripts.evaluate_ecgaim_ludb_semiseg_blinded import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SELECTION,
    SignalPreprocessor,
    build_delineator,
    selected_checkpoint,
    sha256_file,
    stable_hash,
)
from scripts.evaluate_comprehensive_registry import ReconstructionAdapter
from scripts.evaluate_ecgaim_ludb_blinded_daemon import BOUNDARIES, monotonic_match_indices
from scripts.onelead_checkpoint_store import (
    DEFAULT_CACHE,
    DEFAULT_DB as CHECKPOINT_DB,
    connect as connect_checkpoint_db,
    materialize,
)
from unified_latents.engineering.utils.common import mask_unobserved_leads
from unified_latents.engineering.utils.regimes import make_lead_indices

TARGET_FS = 500
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
WAVES = ((1, "P"), (2, "QRS"), (3, "T"))
DEFAULT_QUEUE = ROOT / "refine-logs/queue_spatial_1lead/queue_state.json"
DEFAULT_TEST_CACHE = ROOT / "data/rdb_wavelet_delineation_cache"
DEFAULT_DB = ROOT / "results/onelead_rdb_semiseg_blinded/compact.sqlite"
STOP_REQUESTED = False

CALIBRATION_STEMS = (
    "spatial_1lead_a0_1010010_s42_l{lead}",
    "spatial_1lead_e1_panorama_film_1010010_s42_l{lead}",
    "spatial_1lead_pg1_permuted_geometry_1010010_s42_l{lead}",
    "spatial_1lead_a0_1110000_s42_l{lead}",
    "spatial_1lead_e1_panorama_film_1110000_s42_l{lead}",
    "spatial_1lead_t111_exact_theta_1110000_s42_l{lead}",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)


def finite_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def quantile(values: Iterable[float], probability: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if len(array) else None


def finite_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def bootstrap_upper(values: Iterable[float], confidence: float = 0.99, draws: int = 2000) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    generator = np.random.default_rng(20260825)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        means[index] = generator.choice(array, len(array), replace=True).mean()
    return float(np.quantile(means, confidence))


def bootstrap_quantile_upper(values: Iterable[float], target_quantile: float = .05, confidence: float = .99, draws: int = 2000) -> float | None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    generator = np.random.default_rng(20260825)
    statistics = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        statistics[index] = np.quantile(generator.choice(array, len(array), replace=True), target_quantile)
    return float(np.quantile(statistics, confidence))


def class_boundaries(mask: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for class_id, wave in WAVES:
        active = np.asarray(mask) == class_id
        changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        result[f"{wave}_onset"] = np.flatnonzero(changes == 1).astype(np.int64)
        result[f"{wave}_offset"] = (np.flatnonzero(changes == -1) - 1).astype(np.int64)
    return result


def load_test_index(cache_root: Path, pilot_per_rhythm: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"]["counts"].get("test") != 360:
        raise RuntimeError("RDB held-out test contract is not 360 records")
    rows = [row for row in manifest["records"] if row["split"] == "test"]
    if len(rows) != 360 or len({row["patient_id"] for row in rows}) != 360:
        raise RuntimeError("RDB test split is not patient-disjoint or complete")
    for row in rows:
        row["path"] = str(cache_root / row["output"])
    by_rhythm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rhythm[row["canonical_rhythm"]].append(row)
    pilot: list[dict[str, Any]] = []
    for rhythm in sorted(by_rhythm):
        ordered = sorted(
            by_rhythm[rhythm],
            key=lambda row: hashlib.sha256(f"20260825|{row['record_id']}".encode()).hexdigest(),
        )
        if len(ordered) < pilot_per_rhythm:
            raise RuntimeError(f"rhythm {rhythm} has fewer than {pilot_per_rhythm} test records")
        pilot.extend(ordered[:pilot_per_rhythm])
    identity = stable_hash({
        "manifest_sha256": sha256_file(manifest_path),
        "test": [(row["record_id"], row["output_sha256"]) for row in rows],
    })
    audit = {
        "test_records": len(rows),
        "test_patients": len({row["patient_id"] for row in rows}),
        "pilot_records": len(pilot),
        "pilot_per_rhythm": pilot_per_rhythm,
        "pilot_rhythm_counts": dict(Counter(row["canonical_rhythm"] for row in pilot)),
        "split_seed": manifest["split"]["seed"],
        "split_method": manifest["split"]["method"],
    }
    return rows, pilot, identity, audit


def frozen_population(queue_path: Path, checkpoint_db: Path) -> list[dict[str, Any]]:
    queue = json.loads(queue_path.read_text())
    ids = [job["id"] for job in queue.get("jobs", []) if job.get("status") == "completed"]
    if len(ids) != 60 or len(set(ids)) != 60:
        raise RuntimeError(f"expected exact frozen 60 completed one-lead jobs, found {len(ids)}")
    connection = connect_checkpoint_db(checkpoint_db)
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"SELECT * FROM checkpoints WHERE model_id IN ({placeholders})", ids
        ).fetchall()
    finally:
        connection.close()
    found = {row["model_id"]: dict(row) for row in rows}
    missing = sorted(set(ids) - set(found))
    if missing:
        raise RuntimeError(f"frozen jobs absent from verified catalog: {missing}")
    result = [found[model_id] for model_id in ids]
    counts = Counter(json.loads(row["observed_leads_json"])[0] for row in result)
    if counts != Counter({0: 30, 1: 30}):
        raise RuntimeError(f"expected 30 Lead-I plus 30 Lead-II models, found {dict(counts)}")
    bad = [row["model_id"] for row in result if row["status"] not in {"remote_verified", "cached"}]
    if bad:
        raise RuntimeError(f"unverified checkpoints in frozen population: {bad}")
    return result


def load_onelead_adapter(path: Path, identity: dict[str, Any]) -> ReconstructionAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    observed = payload.get("provenance", {}).get("preprocessing", {}).get("observed_leads")
    if observed != json.loads(identity["observed_leads_json"]):
        raise RuntimeError("checkpoint observed-lead identity mismatch")
    architecture = payload.get("architecture")
    if architecture != identity["architecture"]:
        raise RuntimeError("checkpoint architecture identity mismatch")
    if architecture == "unet":
        from scripts.train_mcma_3lead import MCMAModel
        model = MCMAModel(in_channels=12, out_channels=12)
        kind = "onelead_unet"
    elif architecture == "msvae":
        from unified_latents.engineering.experimental.Multi_Scale_VAE import WearECGVAE
        model = WearECGVAE(
            latent_channels=int(payload.get("latent_channels", 4)),
            target_len=int(payload.get("target_len", 5000)), beta_kl=1e-4, missing_lead_weight=1.0,
        )
        kind = "msvae"
    else:
        from unified_latents.engineering.experimental.aim_1_lead import build_alitok_vae_1d
        model = build_alitok_vae_1d(
            architecture=str(payload["alitok_architecture"]),
            target_len=int(payload.get("target_len", 5000)),
            patch_size=int(payload.get("alitok_patch_size", 25)),
            encoder_depth=int(payload.get("alitok_encoder_depth", 8)),
            decoder_depth=int(payload.get("alitok_decoder_depth", 4)),
            lead_conditioning_mode=str(payload.get("lead_conditioning_mode", "learned")),
            use_learned_lead_id=bool(payload.get("use_learned_lead_id", False)),
            use_relative_geometry=bool(payload.get("use_relative_geometry", False)),
            use_spatial_film=bool(payload.get("use_spatial_film", False)),
            spatial_gain_init=float(payload.get("spatial_gain_init", 0.1)),
            geometry_control=str(payload.get("geometry_control", "standard")),
        )
        kind = "alitok"
    state = {key.removeprefix("_orig_mod."): value for key, value in payload["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model = model.float().eval()
    del payload, state
    return ReconstructionAdapter(
        spec={"id": identity["model_id"], "kind": kind, "observed_leads": observed},
        model=model, device=torch.device("cpu"),
    )


@torch.inference_mode()
def reconstruct(adapter: ReconstructionAdapter, target: torch.Tensor) -> torch.Tensor:
    observed = adapter.observed
    target = target.float()
    masked = mask_unobserved_leads(target, observed).contiguous()
    if adapter.spec["kind"] == "onelead_unet":
        result = adapter.model(F.pad(masked, (0, 120)))[:, :12, :5000].float()
    else:
        indices = make_lead_indices(observed, target.shape[0], torch.device("cpu"))
        if hasattr(adapter.model, "impute_from_regressor"):
            result = adapter.model.impute_from_regressor(masked, lead_indices=indices)["y_pred"].float()
        else:
            result = adapter.model(masked, y_full=target, lead_indices=indices, mode="stage1")["y_pred"].float()
        result = result[:, :12, :5000]
    result[:, observed, :] = target[:, observed, :]
    if result.shape != target.shape or not torch.isfinite(result).all():
        raise RuntimeError(f"invalid reconstruction tensor {tuple(result.shape)}")
    return result


def predict_masks(model: torch.nn.Module, preprocessor: SignalPreprocessor, waveforms: list[np.ndarray], batch_size: int) -> list[np.ndarray]:
    prepared = [preprocessor(waveform) for waveform in waveforms]
    masks: list[np.ndarray] = []
    for offset in range(0, len(prepared), batch_size):
        batch = torch.from_numpy(np.stack(prepared[offset:offset + batch_size]))
        with torch.inference_mode():
            logits = model(batch)["seg_logits"]
        masks.extend(np.repeat(item.astype(np.int8), 2)[:5000] for item in logits.argmax(1).numpy())
    return masks


def lead_groups(observed: int) -> dict[str, tuple[int, ...]]:
    missing = tuple(index for index in range(12) if index != observed)
    return {
        "all_missing": missing,
        "primary_missing_precordial": tuple(range(6, 12)),
        "missing_limb": tuple(index for index in range(6) if index != observed),
        "derived_limb_control": (2, 3, 4, 5),
    }


def fresh_accumulator(groups: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    return {
        "groups": groups,
        "boundary": {group: {boundary: {"ref": 0, "pred": 0, "tp20": 0, "tp150": 0, "errors": [], "record_f1": []} for boundary in BOUNDARIES} for group in groups},
        "region": {group: {wave: np.zeros(4, dtype=np.int64) for _, wave in WAVES} for group in groups},
        "signal": {group: {"record_pcc": [], "record_mse": [], "sum": 0.0, "sum2": 0.0, "n": 0} for group in groups},
        "primary_record_f1": [],
    }


def score_record(acc: dict[str, Any], target: np.ndarray, recon: np.ndarray, reference_masks: np.ndarray, predicted_masks: dict[int, np.ndarray]) -> None:
    for group, leads in acc["groups"].items():
        record_pcc, record_mse = [], []
        record_boundary_f1: list[float] = []
        for lead in leads:
            difference = recon[lead].astype(np.float64) - target[lead].astype(np.float64)
            record_pcc.append(finite_pearson(target[lead], recon[lead]))
            record_mse.append(float(np.square(difference).mean()))
            signal_acc = acc["signal"][group]
            signal_acc["sum"] += float(difference.sum()); signal_acc["sum2"] += float(np.square(difference).sum()); signal_acc["n"] += len(difference)
            reference_mask = reference_masks[lead]
            predicted_mask = predicted_masks[lead]
            reference = class_boundaries(reference_mask)
            predicted = class_boundaries(predicted_mask)
            for boundary in BOUNDARIES:
                real, estimate = reference[boundary], predicted[boundary]
                pairs20 = monotonic_match_indices(real, estimate, 10)
                pairs150 = monotonic_match_indices(real, estimate, 75)
                item = acc["boundary"][group][boundary]
                item["ref"] += len(real); item["pred"] += len(estimate)
                item["tp20"] += len(pairs20); item["tp150"] += len(pairs150)
                denominator = len(real) + len(estimate)
                if denominator:
                    value = 2 * len(pairs20) / denominator
                    item["record_f1"].append(value)
                    if group == "primary_missing_precordial":
                        record_boundary_f1.append(value)
                item["errors"].extend(float(estimate[j] - real[i]) * 2 for i, j in pairs150)
            valid = reference_mask >= 0
            for class_id, wave in WAVES:
                real_positive = reference_mask == class_id
                pred_positive = predicted_mask == class_id
                tp = np.count_nonzero(valid & real_positive & pred_positive)
                fn = np.count_nonzero(valid & real_positive & ~pred_positive)
                fp = np.count_nonzero(valid & ~real_positive & pred_positive)
                tn = np.count_nonzero(valid & ~real_positive & ~pred_positive)
                acc["region"][group][wave] += (tp, fn, fp, tn)
        acc["signal"][group]["record_pcc"].append(finite_mean(record_pcc))
        acc["signal"][group]["record_mse"].append(finite_mean(record_mse))
        if group == "primary_missing_precordial" and record_boundary_f1:
            acc["primary_record_f1"].append(float(np.mean(record_boundary_f1)))


def summarize(acc: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, float | None]]:
    boundary_rows, region_rows, signal_rows = [], [], []
    for group in acc["groups"]:
        for boundary, item in acc["boundary"][group].items():
            denominator = item["ref"] + item["pred"]
            errors = np.asarray(item["errors"], dtype=float)
            boundary_rows.append({
                "lead_group": group, "boundary": boundary, "reference_events": item["ref"], "predicted_events": item["pred"], "tp_20ms": item["tp20"],
                "sensitivity_20ms": item["tp20"] / item["ref"] if item["ref"] else None,
                "ppv_20ms": item["tp20"] / item["pred"] if item["pred"] else None,
                "micro_f1_20ms": 2 * item["tp20"] / denominator if denominator else None,
                "macro_f1_20ms": finite_mean(item["record_f1"]), "matched_150ms": item["tp150"],
                "bias_ms": float(errors.mean()) if len(errors) else None,
                "mae_ms": float(np.abs(errors).mean()) if len(errors) else None,
                "p95_abs_ms": quantile(np.abs(errors), .95),
            })
        for wave, counts in acc["region"][group].items():
            tp, fn, fp, tn = map(int, counts)
            region_rows.append({
                "lead_group": group, "wave": wave, "tp_samples": tp, "fn_samples": fn, "fp_samples": fp, "tn_samples": tn,
                "sensitivity": tp / (tp + fn) if tp + fn else None, "specificity": tn / (tn + fp) if tn + fp else None,
                "ppv": tp / (tp + fp) if tp + fp else None, "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            })
        item = acc["signal"][group]
        bias = item["sum"] / item["n"] if item["n"] else None
        variance = max(item["sum2"] / item["n"] - bias * bias, 0.0) if item["n"] else None
        signal_rows.append({
            "lead_group": group, "record_pearson_mean": finite_mean(item["record_pcc"]), "record_pearson_p05": quantile(item["record_pcc"], .05),
            "record_mse_mean": finite_mean(item["record_mse"]), "record_mse_p95": quantile(item["record_mse"], .95),
            "bias_mv": bias, "loa_lower_mv": bias - 1.96 * math.sqrt(variance) if variance is not None else None,
            "loa_upper_mv": bias + 1.96 * math.sqrt(variance) if variance is not None else None,
        })
    primary = [row["micro_f1_20ms"] for row in boundary_rows if row["lead_group"] == "primary_missing_precordial"]
    primary_signal = next(row for row in signal_rows if row["lead_group"] == "primary_missing_precordial")
    headline = {
        "primary_mean_micro_f1_20ms": finite_mean(primary),
        "primary_record_f1_u99": bootstrap_upper(acc["primary_record_f1"]),
        "primary_signal_pearson_p05": primary_signal["record_pearson_p05"],
        "primary_signal_pearson_p05_u99": bootstrap_quantile_upper(acc["signal"]["primary_missing_precordial"]["record_pcc"]),
    }
    return boundary_rows, region_rows, signal_rows, headline


def evaluate_records(adapter: ReconstructionAdapter | None, records: list[dict[str, Any]], observed: int, delineator: torch.nn.Module, preprocessor: SignalPreprocessor, reconstruction_batch: int, delineation_batch: int) -> tuple[Any, Any, Any, Any]:
    acc = fresh_accumulator(lead_groups(observed))
    missing = acc["groups"]["all_missing"]
    for offset in range(0, len(records), reconstruction_batch):
        if STOP_REQUESTED:
            raise InterruptedError("stop requested")
        batch_rows = records[offset:offset + reconstruction_batch]
        payloads = [torch.load(row["path"], map_location="cpu", weights_only=True) for row in batch_rows]
        target = torch.stack([payload["waveform"].float() for payload in payloads])
        reconstructed = target.clone() if adapter is None else reconstruct(adapter, target)
        waveforms = [reconstructed[record_index, lead].numpy() for record_index in range(len(payloads)) for lead in missing]
        masks = predict_masks(delineator, preprocessor, waveforms, delineation_batch)
        cursor = 0
        for record_index, payload in enumerate(payloads):
            predictions = {}
            for lead in missing:
                predictions[lead] = masks[cursor]; cursor += 1
            score_record(acc, target[record_index].numpy(), reconstructed[record_index].numpy(), payload["segmentation"].numpy(), predictions)
        del payloads, target, reconstructed, waveforms, masks
        gc.collect()
    return summarize(acc)


def connect_results(path: Path) -> sqlite3.Connection:
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
        model_id TEXT NOT NULL,stage TEXT NOT NULL,input_lead INTEGER NOT NULL,architecture TEXT NOT NULL,
        factorial_mask TEXT NOT NULL,checkpoint_sha256 TEXT NOT NULL,status TEXT NOT NULL,attempts INTEGER NOT NULL,
        started_at TEXT,completed_at TEXT,duration_seconds REAL,error TEXT,records INTEGER NOT NULL,
        primary_mean_micro_f1_20ms REAL,primary_record_f1_u99 REAL,primary_signal_pearson_p05 REAL,
        primary_signal_pearson_p05_u99 REAL,PRIMARY KEY(model_id,stage)
      ) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS boundary_summaries(
        model_id TEXT NOT NULL,stage TEXT NOT NULL,lead_group TEXT NOT NULL,boundary TEXT NOT NULL,
        reference_events INTEGER NOT NULL,predicted_events INTEGER NOT NULL,tp_20ms INTEGER NOT NULL,
        sensitivity_20ms REAL,ppv_20ms REAL,micro_f1_20ms REAL,macro_f1_20ms REAL,matched_150ms INTEGER NOT NULL,
        bias_ms REAL,mae_ms REAL,p95_abs_ms REAL,PRIMARY KEY(model_id,stage,lead_group,boundary)
      ) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS region_summaries(
        model_id TEXT NOT NULL,stage TEXT NOT NULL,lead_group TEXT NOT NULL,wave TEXT NOT NULL,
        tp_samples INTEGER NOT NULL,fn_samples INTEGER NOT NULL,fp_samples INTEGER NOT NULL,tn_samples INTEGER NOT NULL,
        sensitivity REAL,specificity REAL,ppv REAL,dice REAL,iou REAL,PRIMARY KEY(model_id,stage,lead_group,wave)
      ) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS signal_summaries(
        model_id TEXT NOT NULL,stage TEXT NOT NULL,lead_group TEXT NOT NULL,record_pearson_mean REAL,
        record_pearson_p05 REAL,record_mse_mean REAL,record_mse_p95 REAL,bias_mv REAL,loa_lower_mv REAL,loa_upper_mv REAL,
        PRIMARY KEY(model_id,stage,lead_group)
      ) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS thresholds(
        input_lead INTEGER PRIMARY KEY,status TEXT NOT NULL,created_at TEXT NOT NULL,anchor_model_id TEXT,
        boundary_cutoff REAL,signal_cutoff REAL,competitive_anchors INTEGER,calibration_anchors INTEGER,details_json TEXT NOT NULL
      ) WITHOUT ROWID;
      CREATE TABLE IF NOT EXISTS screening_decisions(
        model_id TEXT PRIMARY KEY,input_lead INTEGER NOT NULL,decision TEXT NOT NULL,reason TEXT NOT NULL,
        boundary_u99 REAL,signal_u99 REAL,boundary_cutoff REAL,signal_cutoff REAL,is_sentinel INTEGER NOT NULL,
        created_at TEXT NOT NULL
      ) WITHOUT ROWID;
    """)
    return connection


def initialize(connection: sqlite3.Connection, protocol: dict[str, Any]) -> str:
    digest = stable_hash(protocol)
    expected = {
        "schema_version": "1", "protocol": "onelead_rdb_semiseg_blinded_compact_v1", "protocol_sha256": digest,
        "protocol_json": json.dumps(protocol, sort_keys=True, separators=(",", ":")),
        "storage": "aggregate-only; no raw predictions, reconstructed waveforms, per-record rows, or CSV",
    }
    existing = dict(connection.execute("SELECT key,value FROM metadata"))
    if existing and any(existing.get(key) != value for key, value in expected.items()):
        raise RuntimeError("results database belongs to another protocol identity")
    with connection:
        connection.executemany("INSERT OR IGNORE INTO metadata VALUES(?,?)", expected.items())
    return digest


def complete(connection: sqlite3.Connection, model_id: str, stage: str) -> bool:
    row = connection.execute("SELECT status FROM evaluations WHERE model_id=? AND stage=?", (model_id, stage)).fetchone()
    return bool(row and row[0] == "complete")


def store_evaluation(connection: sqlite3.Connection, identity: dict[str, Any], stage: str, records: int, started: float, summaries: tuple[Any, Any, Any, Any]) -> None:
    boundary, region, signal_rows, headline = summaries
    model_id = identity["model_id"]
    observed = json.loads(identity["observed_leads_json"])[0]
    with connection:
        connection.execute("DELETE FROM evaluations WHERE model_id=? AND stage=?", (model_id, stage))
        connection.execute(
            "INSERT INTO evaluations VALUES(?,?,?,?,?,?,'complete',1,?,?,?,?,?,?,?,?,?)",
            (model_id, stage, observed, identity["architecture"], identity["factorial_mask"], identity["sha256"],
             utc_now(), utc_now(), time.perf_counter() - started, None, records,
             headline["primary_mean_micro_f1_20ms"], headline["primary_record_f1_u99"],
             headline["primary_signal_pearson_p05"], headline["primary_signal_pearson_p05_u99"]),
        )
        connection.executemany(
            "INSERT OR REPLACE INTO boundary_summaries VALUES(:model_id,:stage,:lead_group,:boundary,:reference_events,:predicted_events,:tp_20ms,:sensitivity_20ms,:ppv_20ms,:micro_f1_20ms,:macro_f1_20ms,:matched_150ms,:bias_ms,:mae_ms,:p95_abs_ms)",
            [dict(row, model_id=model_id, stage=stage) for row in boundary],
        )
        connection.executemany(
            "INSERT OR REPLACE INTO region_summaries VALUES(:model_id,:stage,:lead_group,:wave,:tp_samples,:fn_samples,:fp_samples,:tn_samples,:sensitivity,:specificity,:ppv,:dice,:iou)",
            [dict(row, model_id=model_id, stage=stage) for row in region],
        )
        connection.executemany(
            "INSERT OR REPLACE INTO signal_summaries VALUES(:model_id,:stage,:lead_group,:record_pearson_mean,:record_pearson_p05,:record_mse_mean,:record_mse_p95,:bias_mv,:loa_lower_mv,:loa_upper_mv)",
            [dict(row, model_id=model_id, stage=stage) for row in signal_rows],
        )
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(json.dumps({"event": "evaluation_complete", "model_id": model_id, "stage": stage, **headline}), flush=True)


def store_error(connection: sqlite3.Connection, identity: dict[str, Any], stage: str, records: int, started: float, error: Exception) -> None:
    observed = json.loads(identity["observed_leads_json"])[0]
    with connection:
        connection.execute("DELETE FROM evaluations WHERE model_id=? AND stage=?", (identity["model_id"], stage))
        connection.execute(
            "INSERT INTO evaluations(model_id,stage,input_lead,architecture,factorial_mask,checkpoint_sha256,status,attempts,started_at,completed_at,duration_seconds,error,records) VALUES(?,?,?,?,?,?,'error',1,?,?,?,?,?)",
            (identity["model_id"], stage, observed, identity["architecture"], identity["factorial_mask"], identity["sha256"],
             utc_now(), utc_now(), time.perf_counter() - started, f"{type(error).__name__}: {error}", records),
        )


def cached_checkpoint(identity: dict[str, Any], checkpoint_db: Path, cache: Path) -> Path:
    connection = connect_checkpoint_db(checkpoint_db)
    try:
        row = connection.execute("SELECT * FROM checkpoints WHERE model_id=?", (identity["model_id"],)).fetchone()
        if row is None:
            raise RuntimeError("checkpoint disappeared from catalog")
        local = Path(row["local_path"]) if row["local_path"] else None
        if local and local.is_file() and sha256_file(local) == identity["sha256"]:
            return local
        return materialize(connection, identity["model_id"], cache)
    finally:
        connection.close()


def evict_checkpoint(path: Path, identity: dict[str, Any], checkpoint_db: Path, cache: Path) -> None:
    resolved = path.resolve()
    if cache.resolve() not in resolved.parents:
        return
    connection = connect_checkpoint_db(checkpoint_db)
    try:
        row = connection.execute("SELECT status,sha256,remote_verified_at FROM checkpoints WHERE model_id=?", (identity["model_id"],)).fetchone()
        if row and row["status"] == "cached" and row["sha256"] == identity["sha256"] and row["remote_verified_at"]:
            resolved.unlink(missing_ok=True)
            connection.execute("UPDATE checkpoints SET local_path=NULL,status='remote_verified',updated_at=? WHERE model_id=?", (utc_now(), identity["model_id"]))
            connection.commit()
    finally:
        connection.close()


def run_model(connection: sqlite3.Connection, identity: dict[str, Any], stage: str, records: list[dict[str, Any]], delineator: torch.nn.Module, preprocessor: SignalPreprocessor, args: argparse.Namespace) -> None:
    if complete(connection, identity["model_id"], stage):
        return
    started = time.perf_counter(); path = None; adapter = None
    try:
        path = cached_checkpoint(identity, args.checkpoint_db, args.checkpoint_cache)
        if sha256_file(path) != identity["sha256"]:
            raise RuntimeError("materialized checkpoint digest mismatch")
        adapter = load_onelead_adapter(path, identity)
        result = evaluate_records(adapter, records, json.loads(identity["observed_leads_json"])[0], delineator, preprocessor, args.reconstruction_batch_size, args.delineation_batch_size)
        store_evaluation(connection, identity, stage, len(records), started, result)
    except Exception as error:
        store_error(connection, identity, stage, len(records), started, error)
        print(json.dumps({"event": "evaluation_error", "model_id": identity["model_id"], "stage": stage, "error": str(error)}), flush=True)
        if args.fail_fast:
            raise
    finally:
        del adapter
        gc.collect()
        if path is not None:
            evict_checkpoint(path, identity, args.checkpoint_db, args.checkpoint_cache)


def run_ceiling(connection: sqlite3.Connection, records: list[dict[str, Any]], delineator: torch.nn.Module, preprocessor: SignalPreprocessor, args: argparse.Namespace) -> None:
    for observed in (0, 1):
        identity = {"model_id": f"__original_l{observed}__", "observed_leads_json": json.dumps([observed]), "architecture": "original", "factorial_mask": "original", "sha256": "original_rdb_test"}
        if complete(connection, identity["model_id"], "full"):
            continue
        started = time.perf_counter()
        result = evaluate_records(None, records, observed, delineator, preprocessor, args.reconstruction_batch_size, args.delineation_batch_size)
        store_evaluation(connection, identity, "full", len(records), started, result)


def calibration_ids() -> set[str]:
    return {stem.format(lead=lead) for stem in CALIBRATION_STEMS for lead in (0, 1)}


def fit_thresholds(connection: sqlite3.Connection) -> None:
    for lead in (0, 1):
        rows = connection.execute(
            "SELECT p.model_id,p.primary_record_f1_u99,p.primary_signal_pearson_p05_u99,f.primary_mean_micro_f1_20ms,f.primary_signal_pearson_p05 FROM evaluations p JOIN evaluations f ON f.model_id=p.model_id AND f.stage='full' WHERE p.stage='pilot' AND p.input_lead=? AND p.status='complete' AND f.status='complete' AND p.model_id NOT LIKE '__original%';", (lead,),
        ).fetchall()
        anchor_id = f"spatial_1lead_a0_1010010_s42_l{lead}"
        anchor = next((row for row in rows if row["model_id"] == anchor_id), None)
        if len(rows) < 6 or anchor is None:
            details = {"reason": "requires at least six full calibration anchors and the prespecified A0"}
            status, boundary_cutoff, signal_cutoff, competitive = "inactive_insufficient_calibration", None, None, 0
        else:
            promising = [row for row in rows if row["primary_mean_micro_f1_20ms"] >= anchor["primary_mean_micro_f1_20ms"] - .02 or row["primary_signal_pearson_p05"] >= anchor["primary_signal_pearson_p05"] - .05]
            boundary_cutoff = min(row["primary_record_f1_u99"] for row in promising) - .02
            signal_cutoff = min(row["primary_signal_pearson_p05_u99"] for row in promising) - .05
            false_skips: list[str] = []
            for held_out in rows:
                training_promising = [row for row in promising if row["model_id"] != held_out["model_id"]]
                if not training_promising:
                    false_skips.append(held_out["model_id"])
                    continue
                held_boundary_cutoff = min(row["primary_record_f1_u99"] for row in training_promising) - .02
                held_signal_cutoff = min(row["primary_signal_pearson_p05_u99"] for row in training_promising) - .05
                would_skip = held_out["primary_record_f1_u99"] < held_boundary_cutoff and held_out["primary_signal_pearson_p05_u99"] < held_signal_cutoff
                held_is_promising = held_out in promising
                if held_is_promising and would_skip:
                    false_skips.append(held_out["model_id"])
            status = "active_zero_false_skip_loo" if not false_skips else "inactive_leave_one_out_false_skip"
            competitive = len(promising)
            details = {"rule": "prune only when both 99% pilot UCBs are below cutoffs", "boundary_margin": .02, "signal_margin": .05, "anchors": [row["model_id"] for row in rows], "competitive": [row["model_id"] for row in promising], "leave_one_out_false_skips": false_skips}
        with connection:
            connection.execute(
                "INSERT OR REPLACE INTO thresholds VALUES(?,?,?,?,?,?,?,?,?)",
                (lead, status, utc_now(), anchor_id, boundary_cutoff, signal_cutoff, competitive, len(rows), json.dumps(details, sort_keys=True)),
            )


def screen_and_promote(connection: sqlite3.Connection, population: list[dict[str, Any]], full_records: list[dict[str, Any]], delineator: torch.nn.Module, preprocessor: SignalPreprocessor, args: argparse.Namespace) -> None:
    thresholds = {row["input_lead"]: row for row in connection.execute("SELECT * FROM thresholds")}
    if set(thresholds) != {0, 1} or any(row["status"] != "active_zero_false_skip_loo" for row in thresholds.values()):
        raise RuntimeError("screening is forbidden until both input-lead thresholds pass zero-false-skip leave-one-out audit")
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    decisions: dict[str, tuple[str, str, bool]] = {}
    pilot_values: dict[str, sqlite3.Row] = {}
    for identity in sorted(population, key=lambda row: row["model_id"]):
        pilot = connection.execute("SELECT * FROM evaluations WHERE model_id=? AND stage='pilot' AND status='complete'", (identity["model_id"],)).fetchone()
        if pilot is None:
            failed = connection.execute("SELECT status,error FROM evaluations WHERE model_id=? AND stage='pilot'", (identity["model_id"],)).fetchone()
            reason = f"pilot failed: {failed['error']}" if failed else "pilot produced no terminal row"
            decisions[identity["model_id"]] = ("failed_pilot", reason, False)
            continue
        pilot_values[identity["model_id"]] = pilot
        threshold = thresholds[pilot["input_lead"]]
        futile = pilot["primary_record_f1_u99"] < threshold["boundary_cutoff"] and pilot["primary_signal_pearson_p05_u99"] < threshold["signal_cutoff"]
        boundary_zone = futile and (
            pilot["primary_record_f1_u99"] >= threshold["boundary_cutoff"] - .02
            or pilot["primary_signal_pearson_p05_u99"] >= threshold["signal_cutoff"] - .05
        )
        if not futile:
            decisions[identity["model_id"]] = ("promote_full", "passed conservative dual-endpoint gate", False)
        elif boundary_zone:
            decisions[identity["model_id"]] = ("promote_full", "within prespecified cutoff safety zone", False)
        else:
            candidates[pilot["input_lead"]].append(identity)
    for lead, rows in candidates.items():
        for index, identity in enumerate(rows):
            sentinel = index % 5 == 0
            decisions[identity["model_id"]] = (
                "promote_full" if sentinel else "pruned_futility",
                "deterministic every-fifth blinded sentinel" if sentinel else "both 99% pilot UCBs below conservative cutoffs",
                sentinel,
            )
    with connection:
        for identity in population:
            if identity["model_id"] not in pilot_values:
                decision, reason, sentinel = decisions[identity["model_id"]]
                connection.execute(
                    "INSERT OR REPLACE INTO screening_decisions VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (identity["model_id"], json.loads(identity["observed_leads_json"])[0], decision, reason,
                     None, None, thresholds[json.loads(identity["observed_leads_json"])[0]]["boundary_cutoff"],
                     thresholds[json.loads(identity["observed_leads_json"])[0]]["signal_cutoff"], int(sentinel), utc_now()),
                )
                continue
            pilot = pilot_values[identity["model_id"]]
            threshold = thresholds[pilot["input_lead"]]
            decision, reason, sentinel = decisions[identity["model_id"]]
            connection.execute(
                "INSERT OR REPLACE INTO screening_decisions VALUES(?,?,?,?,?,?,?,?,?,?)",
                (identity["model_id"], pilot["input_lead"], decision, reason, pilot["primary_record_f1_u99"],
                 pilot["primary_signal_pearson_p05_u99"], threshold["boundary_cutoff"], threshold["signal_cutoff"], int(sentinel), utc_now()),
            )
    for identity in population:
        if STOP_REQUESTED:
            return
        if decisions[identity["model_id"]][0] == "promote_full":
            run_model(connection, identity, "full", full_records, delineator, preprocessor, args)
    invalidated: list[str] = []
    for identity in population:
        decision, _, sentinel = decisions[identity["model_id"]]
        if not sentinel:
            continue
        lead = json.loads(identity["observed_leads_json"])[0]
        anchor = connection.execute("SELECT primary_mean_micro_f1_20ms,primary_signal_pearson_p05 FROM evaluations WHERE model_id=? AND stage='full' AND status='complete'", (f"spatial_1lead_a0_1010010_s42_l{lead}",)).fetchone()
        full = connection.execute("SELECT primary_mean_micro_f1_20ms,primary_signal_pearson_p05 FROM evaluations WHERE model_id=? AND stage='full' AND status='complete'", (identity["model_id"],)).fetchone()
        if full and anchor and (full[0] >= anchor[0] - .02 or full[1] >= anchor[1] - .05):
            invalidated.append(identity["model_id"])
    if invalidated:
        with connection:
            connection.execute("UPDATE thresholds SET status='invalidated_by_competitive_sentinel',details_json=json_set(details_json,'$.competitive_sentinels',json(?))", (json.dumps(invalidated),))
            connection.execute("UPDATE screening_decisions SET decision='promote_full',reason='gate invalidated by competitive sentinel' WHERE decision='pruned_futility'")
        for identity in population:
            if not complete(connection, identity["model_id"], "full"):
                run_model(connection, identity, "full", full_records, delineator, preprocessor, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "ceiling", "calibration", "pilot-all", "screened-all", "full-all"), default="calibration")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--checkpoint-db", type=Path, default=CHECKPOINT_DB)
    parser.add_argument("--checkpoint-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--test-cache", type=Path, default=DEFAULT_TEST_CACHE)
    parser.add_argument("--results-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--delineator-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--selection-summary", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--state", choices=("model", "model_ema"), default="model_ema")
    parser.add_argument("--pilot-per-rhythm", type=int, default=6)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--reconstruction-batch-size", type=int, default=4)
    parser.add_argument("--delineation-batch-size", type=int, default=32)
    parser.add_argument("--model-id")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only contract violated")
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(args.torch_threads); torch.set_num_interop_threads(1)
    selected_checkpoint(args.delineator_checkpoint, args.selection_summary, args.state)
    full_records, pilot_records, dataset_sha, dataset_audit = load_test_index(args.test_cache, args.pilot_per_rhythm)
    population = frozen_population(args.queue, args.checkpoint_db)
    if args.model_id:
        population = [row for row in population if row["model_id"] == args.model_id]
        if not population:
            raise RuntimeError("requested model is not in the frozen 60")
    protocol = {
        "name": "onelead_rdb_semiseg_blinded_compact_v1",
        "evaluator_sha256": sha256_file(Path(__file__)),
        "dataset_sha256": dataset_sha, "dataset_audit": dataset_audit,
        "population_sha256": stable_hash([(row["model_id"], row["sha256"]) for row in frozen_population(args.queue, args.checkpoint_db)]),
        "delineator": {"sha256": sha256_file(args.delineator_checkpoint), "state": args.state},
        "primary_endpoint": "six-boundary mean micro-F1 at 20ms on missing V1-V6",
        "pilot": "six per rhythm by SHA-256 seed 20260825",
        "label_access": "scoring only", "observed_passthrough": True,
    }
    connection = connect_results(args.results_db)
    initialize(connection, protocol)
    delineator = build_delineator(args.delineator_checkpoint, args.state)
    preprocessor = SignalPreprocessor()
    print(json.dumps({"event": "started", "phase": args.phase, "population": len(population), "test_records": len(full_records), "pilot_records": len(pilot_records)}), flush=True)
    try:
        if args.phase in {"ceiling", "calibration", "full-all"}:
            run_ceiling(connection, full_records, delineator, preprocessor, args)
        if args.phase == "smoke":
            run_model(connection, population[0], "smoke", pilot_records[:2], delineator, preprocessor, args)
        elif args.phase == "calibration":
            selected = [row for row in population if row["model_id"] in calibration_ids()]
            if len(selected) != 12:
                raise RuntimeError(f"expected 12 prespecified calibration models, found {len(selected)}")
            for identity in selected:
                if STOP_REQUESTED: break
                run_model(connection, identity, "pilot", pilot_records, delineator, preprocessor, args)
                run_model(connection, identity, "full", full_records, delineator, preprocessor, args)
            fit_thresholds(connection)
        elif args.phase == "pilot-all":
            for identity in population:
                if STOP_REQUESTED: break
                run_model(connection, identity, "pilot", pilot_records, delineator, preprocessor, args)
        elif args.phase == "screened-all":
            for identity in population:
                if STOP_REQUESTED: break
                run_model(connection, identity, "pilot", pilot_records, delineator, preprocessor, args)
            if not STOP_REQUESTED:
                screen_and_promote(connection, population, full_records, delineator, preprocessor, args)
        elif args.phase == "full-all":
            for identity in population:
                if STOP_REQUESTED: break
                run_model(connection, identity, "full", full_records, delineator, preprocessor, args)
    finally:
        connection.close()
    print(json.dumps({"event": "stopped", "phase": args.phase}), flush=True)


if __name__ == "__main__":
    main()
