#!/usr/bin/env python3
"""Reconstruct compact-v1 calibration headline rows from its immutable JSONL logs.

This recovers only values explicitly emitted by the original evaluator. It does
not synthesize boundary, region, signal-detail, duration, or per-record rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path

from scripts.evaluate_ecgaim_ludb_semiseg_blinded import sha256_file, stable_hash
from scripts.evaluate_onelead_rdb_semiseg_blinded import (
    DEFAULT_CHECKPOINT,
    DEFAULT_QUEUE,
    DEFAULT_SELECTION,
    DEFAULT_TEST_CACHE,
    CHECKPOINT_DB,
    calibration_ids,
    connect_results,
    fit_thresholds,
    frozen_population,
    initialize,
    load_test_index,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate_onelead_rdb_semiseg_blinded.py"
DEFAULT_OUTPUT = ROOT / "results/onelead_rdb_semiseg_screened_v1/compact.sqlite"
DEFAULT_CALIBRATION_LOG = ROOT / "results/onelead_rdb_semiseg_blinded/calibration.log"
DEFAULT_CEILING_LOG = ROOT / "results/onelead_rdb_semiseg_blinded/ceiling.log"
METRICS = (
    "primary_mean_micro_f1_20ms",
    "primary_record_f1_u99",
    "primary_signal_pearson_p05",
    "primary_signal_pearson_p05_u99",
)


def log_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_events(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") != "evaluation_complete":
            continue
        key = (row.get("model_id"), row.get("stage"))
        if key in rows:
            raise RuntimeError(f"duplicate completion event {key} in {path}:{number}")
        if any(not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric]) for metric in METRICS):
            raise RuntimeError(f"non-finite or missing headline metric at {path}:{number}")
        rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-log", type=Path, default=DEFAULT_CALIBRATION_LOG)
    parser.add_argument("--ceiling-log", type=Path, default=DEFAULT_CEILING_LOG)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing recovery target: {output}")

    full_records, pilot_records, dataset_sha, dataset_audit = load_test_index(DEFAULT_TEST_CACHE, 6)
    population = frozen_population(DEFAULT_QUEUE, CHECKPOINT_DB)
    identities = {row["model_id"]: row for row in population}
    protocol = {
        "name": "onelead_rdb_semiseg_blinded_compact_v1",
        "evaluator_sha256": sha256_file(EVALUATOR),
        "dataset_sha256": dataset_sha,
        "dataset_audit": dataset_audit,
        "population_sha256": stable_hash([(row["model_id"], row["sha256"]) for row in population]),
        "delineator": {"sha256": sha256_file(DEFAULT_CHECKPOINT), "state": "model_ema"},
        "primary_endpoint": "six-boundary mean micro-F1 at 20ms on missing V1-V6",
        "pilot": "six per rhythm by SHA-256 seed 20260825",
        "label_access": "scoring only",
        "observed_passthrough": True,
    }

    calibration = completion_events(args.calibration_log)
    expected = {(model_id, stage) for model_id in calibration_ids() for stage in ("pilot", "full")}
    if set(calibration) != expected:
        missing = sorted(expected - set(calibration)); extra = sorted(set(calibration) - expected)
        raise RuntimeError(f"calibration log contract failed; missing={missing}, extra={extra}")
    ceilings = completion_events(args.ceiling_log)
    expected_ceilings = {(f"__original_l{lead}__", "full") for lead in (0, 1)}
    if set(ceilings) != expected_ceilings:
        raise RuntimeError(f"ceiling log contract failed: {sorted(ceilings)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".recovering.{os.getpid()}")
    connection = connect_results(temporary)
    try:
        protocol_sha = initialize(connection, protocol)
        now = utc_now()
        for (model_id, stage), row in sorted({**ceilings, **calibration}.items()):
            if model_id.startswith("__original"):
                lead = int(model_id.split("_l", 1)[1].split("__", 1)[0])
                identity = {"architecture": "original", "factorial_mask": "original", "sha256": "original_rdb_test"}
            else:
                identity = identities[model_id]
                lead = json.loads(identity["observed_leads_json"])[0]
            records = len(pilot_records) if stage == "pilot" else len(full_records)
            connection.execute(
                "INSERT INTO evaluations VALUES(?,?,?,?,?,?,'complete',1,?,?,?,?,?,?,?,?,?)",
                (model_id, stage, lead, identity["architecture"], identity["factorial_mask"], identity["sha256"],
                 now, now, None, None, records, *(row[metric] for metric in METRICS)),
            )
        connection.executemany("INSERT OR REPLACE INTO metadata VALUES(?,?)", (
            ("recovery_mode", "headline_rows_from_exact_evaluator_jsonl; detail tables intentionally absent for recovered rows"),
            ("recovery_calibration_log_sha256", log_digest(args.calibration_log)),
            ("recovery_ceiling_log_sha256", log_digest(args.ceiling_log)),
            ("recovery_evaluator_sha256", sha256_file(EVALUATOR)),
            ("recovered_at", now),
        ))
        fit_thresholds(connection)
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("recovered database failed integrity_check")
        active = connection.execute("SELECT count(*) FROM thresholds WHERE status='active_zero_false_skip_loo'").fetchone()[0]
        if active != 2:
            raise RuntimeError(f"recovered screening thresholds are not active for both leads: {active}")
        stored = dict(connection.execute("SELECT key,value FROM metadata"))
        if stored.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("stored protocol identity mismatch after recovery")
    finally:
        connection.close()
    os.replace(temporary, output)
    print(json.dumps({"event": "recovery_complete", "database": str(output), "calibration_rows": 24,
                      "ceiling_rows": 2, "protocol_sha256": stable_hash(protocol)}, sort_keys=True))


if __name__ == "__main__":
    main()
